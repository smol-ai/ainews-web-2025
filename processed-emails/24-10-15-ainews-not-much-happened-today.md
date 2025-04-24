---
id: ed773f33-d4cf-4b5e-a29b-bdcd887a1edb
title: not much happened today
date: '2024-10-15T21:33:05.037085Z'
original_slug: ainews-not-much-happened-today-7393
description: >-
  Vertical SaaS agents are gaining popularity, with significant funding rounds
  such as Decagon's $100m and Sierra's $4b. The AI industry is seeing increased
  competition, model developments, research breakthroughs, and discussions on AI
  capabilities, infrastructure, and ethics.
companies:
  - decagon
  - sierra
  - openai
models:
  - gpt-4
  - claude-3-opus
topics:
  - vertical-ai-agents
  - ai-funding
  - ai-models
  - ai-research
  - ai-infrastructure
  - ai-ethics
  - ai-benchmarks
  - ai-optimization
  - ai-industry-trends
---


<!-- buttondown-editor-mode: plaintext -->**Vertical SaaS agents are all you need.**

> AI News for 10/14/2024-10/15/2024. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **31** Discords (**228** channels, and **1569** messages) for you. Estimated reading time saved (at 200wpm): **197 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Another quiet day in technical news. But the agents funding landscape is afire, with [Decagon announcing $100m in funding](https://x.com/thejessezhang/status/1846235369886589197?s=46) not long after [Sierra's monster $4b round](https://x.com/amir/status/1844192028009345526?s=46). It is remarkable how rapidly the consensus has converged that vertical AI agents are the way to go.

https://www.youtube.com/watch?v=eBVi_sLaYsc

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

**AI Industry Developments and Discussions**

- **OpenAI Alumni Ventures**: [@bindureddy](https://twitter.com/bindureddy/status/1845925936119927021) reported that Mira Murati, ex-CTO of OpenAI, is raising VC funds and poaching talent from OpenAI for a new venture. This highlights the growing competition in the AI market, with over 10 ex-OpenAI startups expected to emerge.

- **Nobel Prize for AI Achievements**: [@demishassabis](https://twitter.com/demishassabis/status/1845864764469334239) shared his thoughts on winning the Nobel Prize for the AlphaFold2 project, which solved the 50-year grand challenge of protein structure prediction. He emphasized the importance of AI in scientific discovery and its potential for developing new therapies.

- **AI Model Developments**: 
  - [@ClementDelangue](https://twitter.com/ClementDelangue/status/1845890229590425920) noted that the gap between open-source and closed-source LLMs is now insignificant.
  - [@johnowhitaker](https://twitter.com/johnowhitaker/status/1845957479341199524) highlighted some interesting techniques used in a new model, including LoRA projectors for weight sharing and annealing on high-quality data.

- **AI Research and Applications**: 
  - [@ylecun](https://twitter.com/ylecun/status/1845929636330721511) discussed the importance of high-bandwidth sensory inputs for self-supervised learning, arguing that language alone is insufficient for learning common sense.
  - [@fchollet](https://twitter.com/fchollet/status/1845925019731611706) commented on a project combining LLMs with the Lean theorem prover, describing it as "intuition-guided reasoning" and a good example of deep-learning guided discrete program search.

- **AI Infrastructure**: [@nearcyan](https://twitter.com/nearcyan/status/1845887854054199730) shared an image showing the datacenter size needed for frontier models, illustrating the massive computational requirements of cutting-edge AI research.

- **AI Tools and Frameworks**: 
  - [@rasbt](https://twitter.com/rasbt/status/1845850007095660796) shared a Jupyter notebook with tips for reducing memory usage when loading larger models like LLMs in PyTorch.
  - [@jerryjliu0](https://twitter.com/jerryjliu0/status/1845907081725096329) described a multi-agent workflow for report generation and form filling, utilizing tools like LlamaParse and long-context LLMs.

- **AI Ethics and Challenges**: [@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1845881870082331052) expressed interest in research investigating how easy it is to get AI agents to perform harmful tasks they're supposed to refuse, and how competent they are at those tasks.

**AI Model Performance and Benchmarks**

- **Model Evaluation**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1845929321925693867) shared information about a paper demonstrating how even a "null model" that always outputs a constant response can cheat automatic benchmarks and achieve top-ranked win rates.

- **Linearizing LLMs**: [@togethercompute](https://twitter.com/togethercompute/status/1845928393877197287) announced LoLCATs, a new method for converting existing Transformers like Llama and Mistral into state-of-the-art subquadratic variants, potentially reducing computational costs.

- **AI Optimization**: [@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1845877069214814525) discussed LPZero, a framework for automatic Zero-cost proxy design in Neural Architecture Search, which could enhance efficiency in evaluating language model architectures.

**AI Industry Trends and Opinions**

- **Competition in AI**: [@adcock_brett](https://twitter.com/adcock_brett/status/1845919277481971789) criticized the notion of a large market with many winners in AI, emphasizing the importance of competitiveness.

- **Open-Source vs. Closed-Source**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1845890229590425920) stated that the gap between open-source and closed-source LLMs is now insignificant, suggesting a leveling of the playing field in AI development.

- **AI Research Culture**: [@finbarrtimbers](https://twitter.com/finbarrtimbers/status/1845885562462654967) commented on the culture of empiricism in modern deep learning, noting both positive and negative aspects of this approach.

**Memes and Humor**

- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1845851259091193979) shared an anecdote about winning a bet regarding the adoption of a distributed metaverse vs. Roblox, highlighting the unpredictability of technology adoption.

- [@DavidSHolz](https://twitter.com/DavidSHolz/status/1845885464311746669) poetically described SpaceX's rocket catch as not just an engineering victory, but a cultural-spiritual one that stirs a deep yearning for science and objective truth.


---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Advancements in Small Language Models: Llama 3.2 1B Performance**

- **[Llama3.2:1B](https://v.redd.it/gr7phat4ypud1)** ([Score: 116, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1g3f0qn/llama321b/)): The post compares **Llama3.2:1B** to larger models, noting its effectiveness for **code generation** and **one-time requests** on systems with **CPU** and **8GB RAM**. While it performs well for these tasks, the model's performance **degrades in long conversations**, with the **3B** version handling extended chat histories more effectively despite being slower.
  - The rapid progress in **AI** since **ChatGPT's launch** is highlighted, with **1B models** now providing comparable quality answers. Some users express excitement about "**AI for the masses**," while others report issues with smaller models, such as increased hallucinations and unrelated responses.
  - The post's **UI** received significant praise, with multiple comments describing it as "cool" and "crazy." The creator mentioned it's part of an [AI device project](https://persys.ai) and is considering adding code execution capabilities.
  - A user proposed the idea of hardware that "**crystalizes**" an **LLM**, suggesting dedicated hardware for performance gains in local LLM applications. The post creator responded, indicating plans for a future version with a dedicated model and board designed for lightweight use.


**Theme 2. AI-Generated Game Environments: Current Limitations and Future Potential**

- **[Playing AI-Generated CS:GO on a Single RTX 3090 in real time](https://youtu.be/6Md5U8rMZjI)** ([Score: 116, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1g3djqv/playing_aigenerated_csgo_on_a_single_rtx_3090_in/)): A team of researchers developed an **AI-generated version of Counter-Strike: Global Offensive (CS:GO)** that runs in **real-time on a single RTX 3090 GPU**. The system uses a **vision-language model** to interpret game state and generate appropriate actions, achieving a **frame rate of 4 FPS** and demonstrating the potential for AI to create and play complex video games autonomously.
  - Users discussed potential improvements, suggesting a **modular game** with **AI-generated textures** and **3D objects**, maintaining control over game mechanics while allowing for **persistent states** and shared player contributions.
  - Some compared the technology to **AI-generated Doom gameplay** and speculated about future applications, such as **real-life driving simulations** using dashcam footage with inputs for acceleration and steering.
  - Debate arose about the project's practicality, with some praising it as an "**unreal experience**" while others argued it's "**light years away**" from being useful, predicting significant advancements in **2-3 years**.


**Theme 3. Hardware Requirements for Running Large Language Models Locally**

- **Hardware costs to run 90B llama at home?** ([Score: 55, Comments: 80](https://reddit.com//r/LocalLLaMA/comments/1g3dtyy/hardware_costs_to_run_90b_llama_at_home/)): The post inquires about the **hardware costs** to run a **90B parameter** version of the **Llama language model** at home for **offline text generation**. The user specifies that **speed** is not a critical factor and that additional features like vision or fine-tuning are not required, acknowledging that the setup might be unaffordable but expressing interest in exploring the possibility.
  - **Llama 3.1 70B** and **Llama 3.2 90B** have the same text model, with the 90B version including vision capabilities. Users can run the **70B model** on various setups, including **64GB RAM** for CPU inference, dual **P40 GPUs** for 6-7 tokens/s, or dual **3090/4090 GPUs** for faster processing.
  - Hardware options range from budget to high-end: a **single 3090 GPU** setup (~$2,000) can run 70B models adequately; **dual 3090 GPUs** (~$3,000) can handle both 70B and 90B models; **dual 5090 GPUs** (~$6,000) offer comfortable performance for both. **Apple Mac Studio M2 Max** with 64GB RAM runs 70B models at ~7 tokens/s.
  - Alternative options include using **AMD EPYC 7002** servers with 8-channel DDR4 memory, capable of running **Llama 70B Q8** at 2 tokens/s or even **Llama 405B Q8** at 0.6 tokens/s with dual CPUs and 512GB RAM. Some users suggest **AMD MI60 GP


**Theme 4. Recreating GPT-like Thinking Processes in Open-Source Models**

- **Recreating GPT o1 CoT Thinking (Thinking and Outputting)** ([Score: 34, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1g3y432/recreating_gpt_o1_cot_thinking_thinking_and/)): The post discusses the creation of a **Thinking and Outputting tag** function for **OpenWebUI**, attempting to replicate the behavior of **GPT-O1**. The author achieved this by fine-tuning instructions within the model file, requiring the model to support the `## Thinking` tag and exit "Thinking" mode with "\*\*\*", demonstrating the function with a video and providing a [download link](https://openwebui.com/f/yuchen4645/Think_And_Generate) for others to try.
  - **cddelgado** hypothesizes that **GPT-O1** uses a complex reasoning system involving **chain of thought**, **tree of thought**, and **adversarial agents** for planning and critique. They suggest implementing this with smaller LLMs using multiple conversations, with one as the main worker and another as an adversary.
  - **kristaller486** clarifies that the post's implementation is not **GPT-O1** but rather **Chain of Thought (CoT)**, stating that O1 is an **RL-based reasoning system**, not just a prompt/agent/fine-tuned model. They provide a [link](https://www.reddit.com/r/LocalLLaMA/comments/1fxof45/its_not_o1_its_just_cot/) for further information.
  - **asankhs** recommends trying the **cot_reflection** approach from the [OptILLM GitHub repository](https://github.com/codelion/optillm) to generate thinking and reflection tokens in responses, offering an alternative method to achieve similar functionality.


## Other AI Subreddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI Research and Techniques**

- **Google Deepmind advances multimodal learning**: A [paper from Google Deepmind](https://arxiv.org/html/2406.17711v1) demonstrates how data curation via joint example selection can further accelerate multimodal learning. (/r/MachineLearning)

- **Microsoft's MInference speeds up long-context task inference**: [Microsoft's MInference technique](https://arxiv.org/abs/2407.02490) enables inference of up to millions of tokens for long-context tasks while maintaining accuracy, dramatically speeding up supported models. (/r/MachineLearning)

- **Scaling synthetic data creation using 1 billion web-curated personas**: A [paper on scaling synthetic data creation](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/) leverages diverse perspectives within a large language model to generate data from 1 billion personas curated from web data. (/r/MachineLearning)

**AI Model Releases and Improvements**

- **Salesforce's "tiny giant" xLAM-1b model surpasses GPT 3.5 in function calling**: Salesforce released xLAM-1b, a 1 billion parameter model that achieves [70% accuracy in function calling, surpassing GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/). (/r/LocalLLaMA)

- **Phi-3 Mini (June) with function calling**: Rubra AI released an updated Phi-3 Mini model in June [with function calling capabilities](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/). It is competitive with Mistral-7b v3 and outperforms the base Phi-3 Mini. (/r/LocalLLaMA)

**AI Applications and Demonstrations**

- **AI-enhanced image upscaling reveals historical anomalies**: A post demonstrates how [advanced image upscaling techniques can reveal hidden details in historical images](https://www.reddit.com/r/StableDiffusion/comments/1g3odok/revealing_hidden_historical_anomalies_through/), potentially challenging established narratives. (/r/StableDiffusion)

- **AI-generated space port view**: An image showcasing [a futuristic hotel room view of a space port](https://www.reddit.com/r/StableDiffusion/comments/1g3ms21/my_hotel_room_has_the_best_view_of_the_space_port/) demonstrates the creative potential of AI image generation. (/r/StableDiffusion)

- **Adobe Firefly Video**: Adobe introduced [Firefly Video, described as "the first commercially safe video generation model"](https://www.reddit.com/r/singularity/comments/1g3mwke/adobe_firefly_video_is_the_first_commercially/), supporting text-to-video and image-to-video generation with a focus on prompt coherence. (/r/singularity)

**AI in Warfare and Defense**

- **AI improves Ukrainian drone effectiveness**: A report claims that [AI has raised Ukrainian drone kill rates to 80%](https://www.reddit.com/r/singularity/comments/1g3y4iw/artificial_intelligence_raises_ukrainian_drone/), highlighting the increasing role of AI in modern warfare. (/r/singularity)

**Philosophical and Societal Implications of AI**

- **Questioning human reasoning abilities**: A post asks if anyone has written a paper on ["Can humans actually reason or are they just stochastic parrots?"](https://www.reddit.com/r/singularity/comments/1g3kbtu/has_anybody_written_a_paper_on_can_humans/), suggesting that humans might fail reasoning tests in ways similar to LLMs. (/r/singularity)

- **Predictions of rapid AI-driven societal changes**: Multiple posts discuss the potential for [rapid, transformative changes due to AI advancements](https://www.reddit.com/r/singularity/comments/1g3fhuk/the_vast_majority_have_absolutely_no_idea_what_is/), with some predicting significant societal upheaval and others offering more speculative timelines for AI development. (/r/singularity)

**AI-Generated Art and Media**

- **Blending real-world and anime aesthetics**: A post showcases [AI-generated images that seamlessly blend realistic and anime-style elements](https://www.reddit.com/r/StableDiffusion/comments/1g3b91y/make_some_my_lora_between_real_world_and_anime/), demonstrating advanced style transfer capabilities. (/r/StableDiffusion)


---

# AI Discord Recap

> A summary of Summaries of Summaries by O1-preview

**Theme 1: Gradient Accumulation Bug Fix Rocks AI Training**

- [**Eleuther Fixes Gradient Accumulation Bug, Stabilizes Training**](http://unsloth.ai/blog/gradient): Eleuther released a fix for a bug causing divergent training losses with large gradient accumulation sizes, linked to cross entropy loss normalization. Users are urged to update their libraries to benefit from this improvement.
- [**Unsloth AI Boosts Accuracy by 10x with Gradient Fix**](https://x.com/UnslothAI/status/1846231235749990699): Unsloth AI announced a fix for a gradient accumulation bug, leading to over **10x** accuracy improvement in training losses. New notebooks demonstrate the fix's impact, and users are encouraged to update Unsloth.
- [**Nous Research Celebrates Gradient Fix from Unsloth AI**](https://x.com/danielhanchen/status/1846235913443262891): The Nous Research community discussed the gradient accumulation fix, highlighting its significance in improving training consistency across setups and enhancing model reliability.

**Theme 2: SageAttention Speeds Up Inference, Engineers Excited**

- [**SageAttention Promises 2.7x Faster Model Inference**](https://arxiv.org/abs/2410.02367): The paper *SageAttention* introduces a quantization method boosting operations per second over FlashAttention2 and xformers by **2.1x** to **2.7x**, while maintaining accuracy. Researchers are eager about potential efficiency gains in transformer models.
- [**Training with SageAttention Hits a Snag**](https://arxiv.org/abs/2410.02367): Attempts to use SageAttention for training led to divergence issues, underscoring that it's currently designed for inference acceleration. Discussions reveal challenges in adapting it beyond its intended purpose.
- [**LM Studio Eyes SageAttention for Performance Leap**](https://arxiv.org/abs/2410.02367): Community members highlight that integrating SageAttention into tools like **llama.cpp** and **MLX** could potentially double token processing speed. If implemented, it would mark a significant performance leap for transformer models.

**Theme 3: AI Model Components Under Fire—QKNorm and ReLU² Scrutinized**

- **QKNorm Gets the Cold Shoulder in Larger Models**: Testing showed **QKNorm** underperformed under tight baselines, leading to "weak attention" in larger models and skepticism about its design merits.
- **ReLU²'s Meager 4% Gain Leaves Engineers Unimpressed**: **ReLU²** offered only a **4%** improvement over functions like GELU, casting doubt on its practicality for scaling large models and igniting debate over activation function efficacy.
- **Researchers Call Out Misleading Performance Claims**: Participants noted that some claimed performance improvements might mask instability issues rather than represent genuine advancements, urging critical evaluation of such assertions.

**Theme 4: AI Industry Shaken by Talent Moves and Controversies**

- [**Microsoft AI Star Sebastien Bubeck Joins OpenAI**](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx): Sebastien Bubeck's move from Microsoft to OpenAI is causing ripples in the AI community. Discussions focus on talent dynamics and the potential impact on AI research directions.
- **Controversy Erupts Over Bubeck's 'Sparks of AGI' Paper**: Community members express mixed feelings about Bubeck's *Sparks of AGI* paper, with critiques targeting its hyperbolic positioning and questioning its implications for defining AGI.

**Theme 5: LLMs' Reasoning Abilities Under Question**

- [**Apple Study Exposes Cracks in LLMs' Logical Reasoning**](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/): An Apple research study reveals that LLMs rely on probabilistic pattern matching, leading to logical reasoning errors when benchmarks change. Engineers discuss the necessity of human comparison baselines and precise definitions of "reasoning."
- **OpenAI Community Debates LLMs' Reasoning Limitations**: Members highlight that LLMs struggle with genuine logical inference, causing "catastrophic" failures in tasks requiring true reasoning. The study prompts a reevaluation of how reasoning is defined and assessed in AI models.


---

# PART 1: High level Discord summaries




## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Gradient Accumulation Bug Fix Unveiled**: A fix is now live for a bug that caused divergent training losses with large gradient accumulation sizes, directly tied to cross entropy loss normalization. Users are encouraged to read more in this [blog post](http://unsloth.ai/blog/gradient) and update their libraries.
   - This issue was raised by multiple members, highlighting the importance of aligning the normalization strategy to ensure stable training loss curves.
- **QKNorm's Effectiveness Questioned**: Testing revealed that **QKNorm** underperformed under tight baselines, leading to 'weak attention' in larger models, creating skepticism around its design. Interestingly, its use in the Olmoe project suggests mixed views on its potential.
   - Participants noted the need for further investigation into its implications for larger architectures, especially as attention mechanisms become crucial.
- **ReLU^2's Gains in Question**: **ReLU^2** only yielded a modest **4%** improvement compared to competitors like GELU, raising doubts about its real-world utility in scaling. This nuanced performance analysis sparks a broader discussion about the activation functions used in large models.
   - The contrast in performance urges engineers to consider both minor enhancements and computational efficiency before adopting new activation methods.
- **Fine-Tuning Libraries under Review**: Concerns arose about the limitations of existing fine-tuning libraries, like the absence of a non-chat-template structure in **torchtune**, as members seek improved evaluation methods. The community is eager for libraries that simplify the fine-tuning process without convoluted templates.
   - Discussion emphasized the usability of **QuestionAnswerTemplate** as a viable alternative for model evaluations, ensuring clearer metrics.
- **Misleading Performance Improvements Scrutinized**: Participants noticed that claims of improved performance can often mask instability issues rather than reflect genuine advancements; A/B testing has been cited as a common pitfall. Papers lacking solid baselines are typically deemed less valuable unless they reveal significant performance shifts.
   - Such practices dilute the quality of research findings, making it vital for researchers to critically assess the conditions under which performance improvements are reported.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gradient Accumulation Fix Improves Training**: Unsloth fixed a bug causing diverging training losses in gradient accumulation, boosting accuracy by over **10x**. Users should update Unsloth and check new notebooks demonstrating the impact.
   - The fix was highlighted in a [tweet](https://x.com/UnslothAI/status/1846231235749990699) from Unsloth AI mentioning significant improvements in training metrics.
- **Launch of INTELLECT-1 Decentralized Model**: Prime Intellect introduced **INTELLECT-1**, a 10-billion-parameter model for collaborative decentralized training. This initiative aims to promote open-source AGI by allowing community contributions.
   - More details are available in their [blog post](https://www.primeintellect.ai/blog/intellect-1) discussing how this model can benefit distributed AI training.
- **SageAttention Promises Faster Model Inference**: The paper *SageAttention* reveals a quantization method that enhances operations per second compared to FlashAttention2 and xformers by **2.1** and **2.7 times**. The method maintains accuracy across various models.
   - However, efforts to use SageAttention for training showed divergence issues, underscoring its inferential focus rather than being viable for training.
- **Exploring LLM Fine-tuning Processes**: Discussions revolved around workflows for fine-tuning LLMs like **Llama**, highlighting the impact of data formatting on output quality. Emphasis was placed on exploring diverse LLM outputs.
   - Participants considered how effective formatting and efficient data management would enhance model performance.
- **Comparative Analysis of Model Performance**: A lively debate emerged surrounding the performance of models like **Qwen** and **Llama**, focusing on their applicability to fine-tuning and dataset utilization. Quality over quantity was a common theme.
   - Engagement centered around how specific datasets could yield better fine-tuning results while discussing integration with tools like Deepspeed for improved capabilities.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Reasoning Feature Faces Inconsistency**: Users noted that **triggering the new reasoning feature** in ProSearch appears random and varies with question complexity, causing inconsistencies in analyses.
   - They observed that the previous reasoning model was more reliable, while the new one has increased instances of hallucinations during information generation.
- **ProSearch App's Frustrating Delays**: Many users expressed annoyance over the **delay of the ProSearch Mac app**, which was initially expected at an earlier date.
   - Additional complaints included issues with **missing threads** and overall sluggish performance in the application.
- **Adobe's AI Video Model Enhancements**: Perplexity AI highlighted **Adobe's AI Video Model** as a transformative development in video editing, promising advanced features that improve workflows.
   - This innovation is anticipated to significantly enhance content creation speed and accessibility.
- **NASA Successfully Launches Europa Clipper**: The **NASA Europa Clipper mission** has successfully launched, aiming to investigate potential signs of life on Jupiter's moon, Europa.
   - Experts eagerly await findings that may reveal new insights into the moon's subsurface ocean.
- **Chinese Researchers Break RSA Encryption**: Recent reports reveal that **Chinese researchers have successfully broken RSA encryption**, creating major concern within the cybersecurity community.
   - This advancement prompts significant discussions about vulnerabilities in current encryption practices for sensitive data.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider's LLM Party with Multiple Instances**: Users discussed the feasibility of running multiple **Aider** instances for larger and smaller tasks, suggesting it should work as long as they don't tamper with the same files.
   - One user humorously branded it as an 'LLM party', highlighting the fun potential of simultaneous LLM operations.
- **API Key Validation Woes**: Members reported **API validation errors** when attempting to configure **Aider** with the **Gemini** model, particularly after setting the key in their `.env` file.
   - One user confirmed that the key worked via the command line, indicating that the issue is likely tied to their scripting setup.
- **Scripting Strategies for Efficient Command Use**: There were discussions on scripting **Aider** commands effectively using Python and command line, emphasizing the need for correct environment loading.
   - A user recounted modifying an example script to implement the Gemini model, but encountered environment variable-related errors.
- **Comparison of Models: Aider vs Sonnet-3.5**: Users noted that **Sonnet-3.5** outperformed other models like **Gemini** for non-web development tasks, making it a preferred choice.
   - One user emphasized consistent superior results from Sonnet-3.5, while testing various models for coding tasks.
- **Gemini Integration and Configuration Challenges**: There were inquiries regarding proper configuration of the **Gemini-1.5 Pro model** within **Aider**, focusing on API key setups.
   - Documentation references were made, yet users continued to face **API errors** stemming from environmental misconfiguration.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace account recovery urgency**: A user urgently sought help recovering their hacked and deleted HuggingFace account, advised to email [website@huggingface.co](mailto:website@huggingface.co) for support.
   - Recovery time might take a few days, but members encouraged patience while awaiting responses.
- **AI automation raises job security concerns**: Members discussed anxieties over AI's potential to automate jobs in Data Science and ML, emphasizing a hopeful shift towards more creative roles.
   - Comparisons were made to past technological advances that also transformed job structures.
- **Llama 3.2 model's inference speed debate**: Running inference on a large dataset with the Llama 3.2 1B model on an A100 GPU took over **14 hours**, prompting discussions on efficiency improvements.
   - Members shared their model loading and inference strategies to optimize performance.
- **Exciting Flutter development collaborations**: A member announced their availability as a **Flutter developer** for collaboration on **AI applications**, inviting others to join forces.
   - This call emphasizes a growing need for partnerships in developing AI-focused projects.
- **Gradio 5 makes a splash on Product Hunt**: The launch of **Gradio 5** was announced on [Product Hunt](https://www.producthunt.com/posts/gradio-5-0), with a request for community support.
   - Team members encouraged users to engage with the new features and provide feedback to boost visibility.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hermes 3 Llama 3.1 405B becomes a subscription model**: The **Hermes 3 Llama 3.1 405B Instruct** model is now available for **$1.79/month**, with a free version accessible at [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b).
   - *Don't miss out on this updated pricing structure for powerful AI functionality!*
- **Nous Hermes Yi 34B is deprecated**: The **Nous Hermes Yi 34B** model has been deprecated by all service providers, making it no longer available for use.
   - *Users are encouraged to transition to alternative models in light of this deprecation.*
- **Highlighting the Rankings of AI Models**: Users discussed the performance of various AI models, with **Llama-3-8b-Instruct** and **GPT-4o** gaining attention for following instructions effectively.
   - *Grok 2 mini* and *Gemini 1.5 Pro* were also noted as decent alternatives, while *Opus* faced some critique for its quirks.
- **Innovative Chatbot Design Techniques**: A user proposed creating a hidden AI chatbot that avoids generic refusal messages to insults, suggesting the use of another LLM for filtering.
   - Participants highlighted models like *Llama Guard* for extra support in managing responses.
- **Issues Reported with Infermatic Provider**: A user reported problems with the **Infermatic** provider as their chats began yielding irrelevant responses unexpectedly.
   - This alerted the community to potential service disruptions that have emerged recently.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research Community's Origins**: The Nous Research community started on Discord and evolved into a funded tech company focused on **AI research** and collaboration.
   - Members actively share ideas and work on various AI models and techniques, enhancing engagement and project outcomes.
- **Gradient Accumulation Bug Fix Released**: The **UnslothAI team** resolved a significant bug in gradient accumulation that caused divergent training losses, improving overall consistency.
   - This fix is now available to users, streamlining training processes and enhancing model reliability.
- **Zamba2-7B Model Performance Explored**: Zyphra announced the launch of the **Zamba2-7B model**, claiming it surpasses Llama3 and Mistral in performance and quality for consumer GPUs.
   - Details on capabilities are outlined in a recent [blog post](https://www.zyphra.com/post/zamba2-7b), which provides insights into its deployment.
- **Model Collapse due to Synthetic Data**: [Research](https://arxiv.org/abs/2410.04840) shows that **even 1%** synthetic data in training sets can lead to significant model collapse, impacting performance of large models.
   - This underlines the risks involved in training large models like ChatGPT, suggesting current practices may require reevaluation.
- **Efficiency of SageAttention Method**: [SageAttention](https://arxiv.org/abs/2410.02367) introduces a quantization method that boosts efficiency in attention mechanisms, outperforming FlashAttention2 by **2.1 to 2.7 times**.
   - This method ensures high accuracy while significantly reducing computational complexity, making it vital for inference acceleration.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Lux-AI Challenge Invites Collaboration**: Members are encouraged to contribute to the [Lux-AI Challenge's GitHub repository](https://github.com/Lux-AI-Challenge/Lux-Design-S3) to foster team collaboration.
   - There is a call for interested individuals to team up for the Lux-AI project, showcasing community engagement in contributing to the challenge.
- **Triton Struggles with Jetson Builds**: Users reported issues building **triton-lang** on the **Jetson Orin AGX 64GB**, where CUDA mistook Unified Memory for `AMD GPU`. A rebuild is underway, with hopes that LLVM support is to blame.
   - Discussions revealed that users should check **LLVM** support for ARM on related [issues](https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed).
- **Learn PyTorch for Deep Learning Now Available**: A new course, [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/), has been shared as a top resource for mastering PyTorch fundamentals.
   - The course format blends video insights with an accessible online book, offering a structured approach to learning.
- **Ollama Performance Hits Raspberry Pi**: The **Ollama** model runs at **5.32 tokens/s** with the **llama3.2** version on the Raspberry Pi 5, while the **llama3.1** case struggles at **1.5 tokens/s**.
   - Discussion touched on the integration of an **eGPU with a 2080**, indicating a feasible upgrade path for the Raspberry Pi systems.
- **WebGPU Lacks CUDA Interaction**: Clarifications were made that **WebGPU** does not interact with **CUDA**, meaning developers must rely on other APIs moving forward.
   - Moreover, WebGPU's functioning depends on specific **graphics APIs** defined by the operating system, such as **Vulkan** and **DirectX**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Real-Time STT Engines Set New Standards**: Gladia's new Real-Time STT engine boasts **< 300 ms latency**, supporting over **100 languages** and code-switching, backed by a $16M Series A funding. Another competitor's engine claims a **90ms inference time** with multi-language support, escalating the competition in transcription tech.
   - As members discuss, this improvement positions these engines as viable choices for a range of applications in real-time communication.
- **Linear Attention Models Promise Efficiency Gains**: The implementation of **linear attention models** within the Llama 3.1 family shows potential for significant efficiency, making it less resource-intensive. Conversations revealed challenges when attempting to transform **>50% of transformer attention layers** into linear versions.
   - Participants seem hopeful about this shift, emphasizing that it aligns with current resource optimization trends in machine learning.
- **AI as the New Building Material**: A blog post compares AI's integration into industries to historical shifts caused by **plastics**, positing AI as a revolutionary material for modern design. The discussion centered around *how previous material ages redefined production and architecture*.
   - Participants expressed excitement for AI's growing role, echoing thoughts on how software is now more pivotal than physical materials.
- **Funding Announcements Ignite Curiosity**: $65M Series B funding for DecagonAI stirred interest regarding trends in AI startup investments, especially in application layers instead of core models. Prominent investors included **Bain Capital Ventures and Accel**, highlighting a robust market for AI solutions.
   - Members noted that such fundraising endeavors reflect a shift in focus towards practical AI implementations, shedding light on current market dynamics.
- **Debate on Outsourcing Documentation**: There's a vibrant discussion about the possibilities of outsourcing documentation for AI and open-source projects, weighing pros and cons of using LLMs vs. human writers. Community members reflect on how this could impact quality and accessibility.
   - The conversation raises questions on the balance between cost-effectiveness and thorough documentation, indicating a vital consideration in project management.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama 3.1-70B Integration Faces Truncation Trouble**: An integration of **Llama 3.1-70B** is returning truncated responses, consistently providing only **5 skills** when a list of **20 software engineering skills** is requested, due to hitting the `max_tokens` limit.
   - *One user noted, 'Responses end with finish_reason: max_tokens'* despite parameter adjustments.
- **Qdrant Node Addition Triggers Errors**: A member encountered an error when adding new nodes to the **Qdrant** index, without prior reports of such issues, indicating a potential setup conflict.
   - Another user suggested that their own successful additions imply possible misconfigurations in the first user's setup.
- **Build a Financial Agent with Claude 3.5**: You can create a **Financial Agent** powered by **Claude 3.5 Sonnet** using APIs for stock prices and company data shared by [@financial_mod](https://twitter.com/llama_index/status/1845980793593831845).
   - According to Hanane Dupouy, this agent provides diverse insights, including income statements and comprehensive company information.
- **PineconeVectorStore Failing in ComposableMemory**: Members expressed frustration with **PineconeVectorStore** in `SimpleComposableMemory`, receiving a 'Namespace not found' error message.
   - Another user speculated set-up issues might be causing these persistent errors.
- **Performance Lag in Neo4jPropertyGraphStore Initialization**: A significant delay in initializing the **Neo4jPropertyGraphStore** has been reported, with schema generation taking excessively long on larger graphs.
   - This issue may be exacerbated by not using `async` operations, corroborated by a related [GitHub issue](https://github.com/run-llama/llama_index/issues/16204).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **LLMs Show Cracks in Reasoning**: A recent [Apple study](https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/) reveals that **LLMs** utilize probabilistic pattern matching in mathematical reasoning, leading to errors when benchmarks shift.
   - Members expressed the necessity for **baseline human comparisons** and highlighted the ambiguous definitions of reasoning according to the study.
- **Swarm Library Needs Better Testing**: Users examining the **Swarm** library identified difficulties in distinguishing whether tasks are executed by agents or the base LLM, underlining the need for robust tests.
   - Concerns about Swarm's **non-production-ready** status arose, along with mentions of alternatives like **Swarm.js**.
- **Confusion Over GPT Voice Features**: Discussions emerged regarding the rollout of the advanced **GPT voice** feature, with no definitive announcements from OpenAI yet on its functionality.
   - Skepticism grew about potential updates due to past versions being unsupported.
- **Issues with Custom GPT Updates**: A member's custom GPT, built from **300 pages** of materials, remained in 'Update Pendings' for over a week after splitting the PDFs into **six smaller files**.
   - Despite the PDFs being acknowledged, the bot often redirected queries back to code, rather than answering directly from the documents.
- **Troubles with PDF Processing**: Another member encountered performance issues when testing **1 PDF** in GPT-4, indicating deeper problems with PDF content processing affecting responsiveness.
   - This suggests that there may be systemic challenges in how GPT interacts with PDF inputs.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio configuration options need clarity**: A member proposed that **configuration details** be shared in formats other than screenshots, noting that future blog posts will incorporate these changes.
   - This suggestion aims to enhance usability, making it easier for users to comprehend settings and optimizations.
- **M2 Studio excels with large models**: Users are praising the **M2 Studio** equipped with **192 GB RAM** for its impressive performance with **Mistral's large 128K context** model, proving ideal for specific applications.
   - *It's such a good model for my use case* underscores its value, possibly attracting more users to high-RAM setups.
- **Tweaking GPUs for performance boosts**: One user recommended **Under-volting (UV)** GPUs using **Afterburner**, stating that even a **100mV** adjustment can notably enhance performance.
   - They urged peers to check **YouTube** for targeted tutorials, facilitating better performance tuning across setups.
- **Stellar TPS performance from Llama 8B**: Some users reported achieving **30 TPS** with **Llama 8B** on various GPUs, with expectations for **150+ TPS** driving discussions on necessary upgrades.
   - Factors like model size and quantization significantly influence performance, especially when comparing setups equipped with advanced **tensor cores** versus older GPUs.
- **SageAttention promises efficiency gains**: The recent paper on **SageAttention** highlights outstanding efficiency improvements in attention mechanisms, with significant implications for tools like **llama.cpp** and **MLX**.
   - If implemented, it could potentially **double token processing speed**, marking a leap in performance for Transformer models.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Connector misunderstands inputs**: Users reported that the **Cohere Connector** triggers a search even upon a simple 'hi', prompting inquiries about control features to limit unnecessary interactions.
   - *Is there a way to refine its functionality?* The community is actively seeking solutions to optimize this.
- **API Token Limits raise concerns**: A discrepancy was raised regarding the **Cohere API** token limits, noting a **10k** monthly cap versus **5 million tokens** mentioned in chat, leading to questions about potential overage costs.
   - *Will exceeding the 10k cap result in billing?* Clarity is sought by members on this critical point.
- **Google Connector not performing**: Multiple users are facing issues with the **Google Connector**, which is failing to operate correctly, sparking a troubleshooting session among users.
   - *Share any breakthroughs!* The community is encouraged to support one another in resolving this connectivity issue.
- **Command Model pricing clarified**: Discussion clarified that there are no fees for the **web-search connector**, but charges apply to results sent to the **Command** input context, potentially impacting users' budget.
   - This distinction highlights the intricacies of API usage costs and encourages careful monitoring.
- **OrionChat aggregates AI models**: A member launched **OrionChat**, a web interface enabling users to interact with various AI models from **Cohere**, **OpenAI**, and others seamlessly in one place, available at [this link](https://orionchat.github.io).
   - The initiative aims to consolidate conversations and facilitate comparisons across models, fostering user feedback for further refinement.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **WordPress Plugin Development Seeks Feedback**: A member is developing multiple **WordPress plugins** for text generation and **txt2img** servers, eagerly seeking community feedback and testing.
   - *Nobody is responding*, highlighting significant frustrations with community engagement in AI Discord servers.
- **CORS Issues Frustrate Stable Diffusion Setup**: Users discussed persistent **CORS errors** faced while using SSL with Stable Diffusion servers on a reverse proxy setup.
   - A tech-savvy member emphasized the need for the webserver and **Stable Diffusion server** to run on the same machine for full functionality.
- **Searching for Active AI Communities on Discord**: A member expressed disappointment in their **AI Discord server's** lack of activity, seeking suggestions for more vibrant communities related to **comfyUI** and **A1111**.
   - Unanswered inquiries about plugins point to a broader need for better engagement within the community.
- **Exploring Base Models for Text Generation**: A user inquired about base models that enhance text generation during style transfer, specifically mentioning **i2i** and **SD1.5**.
   - Another member recommended trying **flux** or **SD3**, while cautioning that **SD3** struggles with human representation.
- **Techniques for Creating Stylized Photos**: Discussion centered around methods for producing stylized photos, with several members suggesting the use of **ControlNets**.
   - Creative approaches were shared, including techniques outlined [here](https://github.com/songrise/Artist) for various artistic styles, such as pin-up.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad outperforms NumPy on .dot operations**: A detailed comparison showed that **Tinygrad's .dot operations** exhibit accuracy drops for larger matrices, hitting **±0.001** differences for dimensions like M=16384, N=8192, K=1280.
   - Conversely, smaller matrices (M=10, N=4, K=5) only had minimal deviations, not exceeding **±0.000001**.
- **VIZ UI improvements take center stage**: A discussion revolved around [Issue #7067](https://github.com/tinygrad/tinygrad/issues/7067), highlighting sought-after enhancements to the **VIZ UI**, especially related to autoscrolling features.
   - Proposals included resizing and collapsible sidebars, aiming to improve user experience.
- **George Hotz vows to rival PyTorch's performance**: George posited that beating **PyTorch**'s performance on NVIDIA GPUs would be monumental for **Tinygrad**, marking a turning point for the project.
   - 'All we have to do is beat PyTorch in perf and we win,' he stated, underscoring the stakes involved.
- **Unpacking TD-MPC implementation in Tinygrad**: One user shared the exciting news about successfully implementing **TD-MPC learning** in Tinygrad and plans to test it on hardware.
   - Links to the [GitHub repository](https://github.com/nicklashansen/tdmpc2/tree/main) were shared, detailing necessary hardware requirements.
- **Methods for disabling gradient calculations**: Users debated effective ways to disable gradients, advocating for `Tensor.no_grad` while suggesting alternatives like `with Tensor.test():` as a modern practice.
   - The conversation aimed to refine gradient control methods within the community.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Resolving Library Installation Issues**: A user found that missing libraries could be installed using `sudo apt-get install libtinfo-dev`, assisting others with similar installation issues.
   - This finding emphasizes the role of community knowledge sharing to tackle common problems effectively.
- **Addressing Custom stdlib Challenges**: Users faced challenges running a modified version of stdlib, where original implementations persisted despite following build instructions.
   - A workaround involving adjustments to the build process was proposed to address these ongoing issues.
- **Seeking New Image Hashing Algorithms**: Questions arose regarding the relevance of older image hashing algorithms such as pHash, with calls for recommendations on advanced alternatives.
   - The community's exploration showcases an eagerness to adopt cutting-edge techniques as technology evolves.
- **Discussing Memory Management Strategies**: A premature destruction of a struct instance during an assertion call raised concerns about memory management in Mojo.
   - Suggestions included creating a getter method to safely access struct members, reducing risks of early destruction.
- **Collaborative Bug Reporting Success**: A user reported a string interpolation issue that was confirmed to be fixed in the latest version of Mojo.
   - This instance highlights the effectiveness of community collaboration in identifying and resolving bugs swiftly.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Sebastien Bubeck joins OpenAI**: Microsoft's star AI researcher, **Sebastien Bubeck**, is making waves by moving to **OpenAI**, prompting discussions on talent dynamics in AI.
   - This move was first reported in an article from [The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx).
- **o1-turbo-mini impresses in benchmarks**: Buzz surrounds the performance of **o1-turbo-mini**, showcasing suspiciously strong results that have led to a mix of skepticism and humor among engineers.
   - Community members noted the amusing potential to poke fun at an overly online crowd reacting to this news.
- **Doomsday Clock for AGI stirs controversy**: A **Doomsday Clock** launched by a Saudi-backed Swiss business school claims to warn against 'uncontrolled general intelligence,' criticizing it as outdated.
   - Creator **Michael Wade** argues it's absurd to liken software like Excel to the threats posed by AGI, reflecting historical fears rather than contemporary relevance.
- **AI2 seeks Research Interns for OLMo**: **AI2** announced openings for Research Interns in the OLMo project, aimed at enhancing natural language processing and machine learning.
   - This 12-week internship in Seattle offers competitive compensation between **$86,520** and **$123,600**, focusing on impactful research initiatives.
- **OpenAI's impact on the legal field**: Discussion highlights OpenAI's role in creating favorable conditions for **lawyers**, linking AI advancements to evolving legal jobs.
   - This underscores the growing interplay between AI technology and practical applications in the legal domain.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Framework Selection is a Nightmare!**: Members expressed frustration about the constant shifting among frameworks like **Langchain**, **Langflow**, and **Langgraph**, making finalizing a production choice difficult.
   - One noted that their entire codebase has transitioned to *Langchain LCEL*, highlighting the chaos surrounding these frameworks.
- **Langgraph Deployment on Private Cloud**: A member inquired about deploying a **Langgraph** application on their cloud outside of the **US** or **EU**, seeking community insights.
   - While there was no direct response, this inquiry sparked interest in regional application hosting.
- **Debate on dspy vs. Langchain**: Interest arose around whether **dspy** would dominate over **Langchain** and other frameworks or if they would maintain relevance.
   - This reflects uncertainty in the community about the future landscape of AI frameworks.
- **Acknowledgment of Langsmith's Utility**: One member suggested **Langsmith** is useful for tracing, emphasizing its importance among shifting frameworks.
   - This led to recommendations for the **Langchain Academy** course on **Langgraph** to sharpen related skills.
- **Clarification on Langflow's Affiliation**: A user clarified that **LangFlow** is not an offering of **LangChain**, addressing confusion among members about related tools.
   - This distinction may help align understanding within the community regarding the various discussed frameworks.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC provides all course details online**: All the details on **labs** and **assignments** can be found on the course website at [course website](https://llmagents-learning.org/f24), encouraging participants to check for updates.
   - To join, prospective students should fill in this [form](https://forms.gle/svSoNhKcGFjxup989) and engage with the community via the [LLM Agents Discord](https://discord.gg/NWVpQ9rBvd) for real-time support.
- **Test-time compute scaling law observed**: Members discussed the broader impact of the **'test-time compute' scaling law**, linking it to earlier laws affecting the GPT family, supported by [this paper](https://arxiv.org/pdf/2408.03314).
   - Another document relevant to this discussion was also shared, found [here](https://arxiv.org/pdf/2001.08361).
- **AI-Powered Search book emerges as essential**: A member recommends [this book](https://www.manning.com/books/ai-powered-search) as a critical resource for the next few years in AI-powered search technologies, likely impacting **practitioners and researchers**.
   - They expect its insights to be foundational for AI studies across various industries.
- **Lecture video quality concerns raised**: One member noted the necessity for improved **video quality** in lecture uploads, stating that **720p** is the highest available for lecture 6, making it hard to read code.
   - This concern indicates a demand for more accessible learning materials within the course.
- **Exploring reasoning and planning in LLMs**: A member sought insights on how LLMs and agents work with **reasoning**, **planning**, and identifying tools rather than just generating text.
   - They expressed interest in further lecture coverage on **planning** and **tool use** to deepen understanding of LLM applications.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter hits π release**: A member announced a new version update of **Open Interpreter**, accessible via `pip install --upgrade open-interpreter`, marking this as a significant **π release** with notable enhancements.
   - This tweet by [Mike Bird](https://x.com/MikeBirdTech/status/1846283357153268002) shared the improvements and generated buzz around its capabilities.
- **Hume AI impresses, Oi takes the stage**: A user recounted how the **Hume AI model** exceeded expectations, stating it works almost **too well**, which raises scrutiny on performance thresholds.
   - The conversation shifted focus to the **Oi model**, suggesting active experimentation with various AI frameworks.
- **Play 3.0 mini boosts Text-To-Speech**: [Play.ht](https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) unveiled **Play 3.0 mini**, a Text-To-Speech model that offers improved speed and accuracy across multiple languages while being cost-effective.
   - They invited users to test it out on the [playground](https://play.ht/playground/?utm_source=x&utm_medium=social&utm_campaign=all_v3launch_202410) and share feedback on the enhancements.
- **Think-on-Graph calls for collaborators**: The **Think-on-Graph** GitHub repository is now live, inviting researchers interested in collaborating in Shenzhen to check it out [here](https://github.com/IDEA-FinAI/ToG).
   - The project includes an open invitation for contact via email for those wanting to contribute and be part of the research team.
- **Watch video on AI advancements**: A user shared a [YouTube video](https://www.youtube.com/watch?v=iGQLG0bWDxE) that touches on recent advancements formed around AI technologies.
   - Details were scant, suggesting viewers to engage directly to glean insights from the contents presented.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Curious about Loom Video Insights**: A member shared a [Loom video](https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-f408a83cecf5), likely containing insights relevant to ongoing discussions, though details were sparse.
   - The video piqued members' interest, prompting them to explore its content for valuable information.
- **Contextual Embeddings Resources Roll In**: A member shared a [Google Colab](https://tinyurl.com/2p9wwypy) and a [YouTube video](https://www.youtube.com/watch?v=6efwN_US-zk&t=7s) titled 'Contextual Retrieval with Any LLM,' focused on implementing contextual embeddings.
   - The video aims to streamline the implementation of contextual retrieval strategies from Anthropic for various LLMs.
- **RAG Mechanics: Clarifying Chunking Process**: Members discussed the challenges of adding whole documents to prompts without exceeding token limits, highlighting the **chunking process** integral to **RAG (Retrieval-Augmented Generation)**.
   - It was clarified that RAG utilizes similarity search to include only the most relevant chunks, ensuring compliance with token limits.
- **DSPy Integration into GPT-O1+ Status Check**: One member inquired about the progress of integrating **DSPy** into the **GPT-O1+** system, anticipating updates on the development.
   - However, the details of this integration remain unaddressed in the discussions.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ICLR Reviews are finally out!**: The long-awaited review papers for **ICLR** have been released, prompting excitement among members eager to dive in.
   - *One member noted* it will take time to process their assigned review.
- **Study on Continuous Pre-training and Instruction Fine-tuning**: A recent paper investigates the relationship between **continuous pre-training** and **instruction fine-tuning** for Large Language Models, emphasizing the need for models to stay updated with the latest data.
   - It raises the question of which model should undergo this pre-training for maintaining instruction-following abilities.
- **Model Merging Approach Critique**: *A member questioned* the novelty of the approach in the paper, suggesting it resembles long-established methods of model merging.
   - This sparked a discussion about the relevance and originality of the proposed techniques.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Inquiry on LAION-2B Dataset and MSCOCO Overlap**: A member inquired about whether the **LAION-2B dataset** contains images from **MSCOCO** (COCO2014 or COCO2017), questioning the potential **data overlap**.
   - The inquiry highlighted the mention in the paper regarding **data overlap**, with a request for further details on the techniques employed to verify this issue.
- **Good Morning and General Greetings**: Members exchanged general greetings with one member stating, **'Good morning everyone.'** fostering a friendly environment in the chat.
   - Another member casually acknowledged the greeting with **'gm'**, contributing to a light atmosphere.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Decoding Inference Pipeline Mechanics**: The inference pipeline in **Gorilla LLM** executes functions by outputting valid function calls that `decod_exec` can interpret, signaling turn completion when it outputs nothing or an un-decodable response.
   - This automatic signaling indicates when the model has finished its task, enhancing interaction efficiency.
- **Model's Output Stop Signals**: A member underscored the **importance of the model** determining when to cease function calls, suggesting it can signal turn end by outputting nothing.
   - This flexibility becomes crucial for maintaining fluid user interaction in various scenarios.
- **Weather Inquiry Demonstrates Function Calls**: An illustrative example showed the model handling a weather query using function calls like `get_coordinate` and `get_weather`, showcasing its data retrieval process.
   - The session concluded when the model's post-data output couldn't be decoded, effectively ending that turn.
- **Function Call Output Variability Explored**: The model's approach to function call outputs allows it to stop or extend interactions creatively, including opting not to output anything at all.
   - This variability highlights the diverse techniques prompting models utilize to adapt to user queries.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Appreciation for LLM Finetuning Help**: A user expressed gratitude towards another member for their assistance in **LLM Finetuning** efforts.
   - This gesture highlights the collaborative environment within the community, showcasing the shared knowledge and support for technical challenges.
- **Contribution Acknowledgment**: Member cyberg0285 thanked another community member by tag for their contributions, indicating a supportive atmosphere.
   - Such acknowledgments foster a sense of community and collaboration among engineers working on complex LLM projects.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Discussion on Forwarding Protocols**: A member shared an important link regarding **forwarding protocols**, highlighting their relevance in recent discussions.
   - *Here is the forwarded message for reference.*
- **Importance of Information Sharing**: Another member stressed the need for proper **information sharing** practices to boost community engagement and streamline communication.
   - They noted that *forwarding messages can facilitate quicker responses and clearer communication.*



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Launch of AI Stewardship Practice Program**: The **AI Stewardship Practice Program** by MaRS Discovery District offers free slots for a pilot course aimed at positively influencing AI development. More details can be found on the [Tech Stewardship website](https://programs.techstewardship.com/).
   - This microcredential program is designed for researchers, educators, and policymakers, providing an opportunity to engage in AI stewardship practices.
- **Become a Tech Steward**: Participants can engage with offerings promoting the goal to **bend the arc of technology towards good** through this Tech Stewardship initiative. Interested individuals should [reply in thread here](https://discord.com/channels/1089876418936180786/1295822228406931529) to join the pilot course valued at **500 CAD**.
   - The program aims to cultivate a community of tech stewards dedicated to responsible AI practices and ethical technology use.



---


The **Alignment Lab AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **DiscoResearch Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1295532010550595595)** (26 messages🔥): 

> - `Gradient Accumulation Bug Fix`
> - `Difference Between Base and Instruct Models`
> - `Normalization in Cross Entropy Loss` 


- **Gradient Accumulation Bug Fix Announced**: A fix has been implemented for a bug causing all training losses to diverge with large gradient accumulation sizes, impacting libraries that utilize this method. This issue was linked to the cross entropy loss normalizer and required denormalizing to match training loss curves.
   - Further details on this fix can be found in a [blog post here](http://unsloth.ai/blog/gradient) and users are encouraged to update to the latest version.
- **Discussion on Partial Batch Weighting Mistakes**: It was highlighted that accidentally upweighing partial batches due to normalization issues in cross entropy could lead to significant divergence in training losses. This is especially true in cases where padding affects loss calculations.
   - Members suggested changing the wording of the bug description to clarify that it particularly affects losses that use mean reduction and padding in PyTorch.
- **Impacts of Sequence Length on Loss Calculation**: Concerns were raised regarding how shorter sequence lengths get weighted more heavily compared to longer ones when using improper normalization strategies. This can lead to further divergence of losses, especially when ignoring input token losses.
- **Normalization Practices Under Scrutiny**: A user commented on their practice of implementing correct normalization in their code and noted that poor normalization strategies were common. They pointed out that averaging by the number of unmasked positions for ragged batches often leads to issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.desmos.com/calculator/rbh1evp5d0).">Desmos | Graphing Calculator</a>: no description found</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891">Tweet from Daniel Han (@danielhanchen)</a>: Fixed a bug which caused all training losses to diverge for large gradient accumulation sizes.  1. First reported by @bnjmn_marie, GA is supposed to be mathematically equivalent to full batch training...</li><li><a href="https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html">CrossEntropyLoss &mdash; PyTorch 2.4 documentation</a>: no description found</li><li><a href="https://github.com/karpathy/nanoGPT/commit/553f9">fix minor bug where we have to scale the loss to account for gradient… · karpathy/nanoGPT@553f949</a>: … accumulation, which sums before backprop. note that this is not a major bug because AdamW is scale invariant. however, this did affect gradient clipping
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1295461668893098045)** (228 messages🔥🔥): 

> - `QKNorm Effectiveness`
> - `ReLU^2 Performance`
> - `Impact of Sequence Length on Loss`
> - `Local vs. Global Attention`
> - `MoE Sparse Computation` 


- **QKNorm's Stability Concerns**: QKNorm did not perform well during tests on a tight baseline, resulting in 'weak attention' after training, raising questions about its effectiveness in larger models.
   - Despite this, it was utilized in the Olmoe project, suggesting potential internal disagreements about its design choices.
- **ReLU^2's Limited Improvement**: While ReLU^2 was found to bring only a 4% improvement in specific tests, its use in larger models raises concerns whether the performance gain justifies its implementation.
   - The comparison of ReLU^2's performance against GELU and other gating mechanisms reflects ongoing discourse about activation functions in large-scale deployments.
- **Comparing Models with Different Sequence Lengths**: When evaluating models trained on different sequence lengths, it's crucial to match the evaluation sequence length to ensure fair comparisons.
   - Longer sequence lengths may yield better training loss results, but evaluation at consistent lengths helps determine the true efficacy of the model.
- **Local vs. Global Attention Dynamics**: Local attention mechanisms can exhibit greater instability at deeper layers, while global attention maintains consistent gradient propagation regardless of depth.
   - This distinction may contribute to the varying effectiveness of attention strategies in different model architectures.
- **Exploring MoE for Sparse Computation**: The discussion on using a Mixture of Experts (MoE) approach for sparse computation highlighted its potential advantages and the need for careful implementation to avoid instability.
   - This method opens up new possibilities for dynamic model architectures, such as generating weights on-the-fly, which could represent a significant advance in the field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Tim_Dettmers/status/1846023509811908751">Tweet from Tim Dettmers (@Tim_Dettmers)</a>: Just as a warning: I tried all of these, and all of these worked ... at the small scale. When scaled up, none of these worked for me (except padding embeddings -- but what you should really do is opti...</li><li><a href="https://arxiv.org/abs/2410.10733">Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models</a>: We present Deep Compression Autoencoder (DC-AE), a new family of autoencoder models for accelerating high-resolution diffusion models. Existing autoencoder models have demonstrated impressive results ...</li><li><a href="https://openreview.net/forum?id=P3SQi2EWeR">Integrating Large Circular Kernels into CNNs through Neural...</a>: The square kernel is a standard unit for contemporary Convolutional Neural Networks (CNNs), as it fits well on the tensor computation for the convolution operation. However, the retinal ganglion...</li><li><a href="https://arxiv.org/abs/2109.08668">Primer: Searching for Efficient Transformers for Language Modeling</a>: Large Transformer models have been central to recent advances in natural language processing. The training and inference costs of these models, however, have grown rapidly and become prohibitively exp...</li><li><a href="https://arxiv.org/abs/2402.03804">ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs</a>: Sparse computation offers a compelling solution for the inference of Large Language Models (LLMs) in low-resource scenarios by dynamically skipping the computation of inactive neurons. While tradition...</li><li><a href="http://arxiv.org/abs/2212.14034">Cramming: Training a Language Model on a Single GPU in One Day</a>: Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and p...</li><li><a href="https://arxiv.org/abs/1803.00904">Hardness of Approximate Nearest Neighbor Search</a>: We prove conditional near-quadratic running time lower bounds for approximate Bichromatic Closest Pair with Euclidean, Manhattan, Hamming, or edit distance. Specifically, unless the Strong Exponential...</li><li><a href="https://arxiv.org/abs/2410.04271">Fundamental Limitations on Subquadratic Alternatives to Transformers</a>: The Transformer architecture is widely deployed in many popular and impactful Large Language Models. At its core is the attention mechanism for calculating correlations between pairs of tokens. Perfor...</li><li><a href="https://x.com/Tim_Dettmers/status/1846223418989183417">Tweet from Tim Dettmers (@Tim_Dettmers)</a>: Just a bit more context: I have a super tight baseline of a Chinchilla 250M model that I ran more than 1,000 training runs on. The data is very diverse. The baseline is so tight that all my research t...</li><li><a href="https://x.com/Grad62304977/status/1846227646893461536">Tweet from Grad (@Grad62304977)</a>: Hmm thanks for clarifying. For qk norm is there a reason then why Olmoe used it? Also for zero init just to clarify it did not bring the performance gain largely, it was the removal of its replacement...</li><li><a href="https://arxiv.org/abs/2209.04881">On The Computational Complexity of Self-Attention</a>: Transformer architectures have led to remarkable progress in many state-of-art applications. However, despite their successes, modern transformers rely on the self-attention mechanism, whose time- and...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/blob/42ab270216d4d10abbadc34e92f92e8a011a384f/records/101424_ModernArch/train_gpt2.py#L159">modded-nanogpt/records/101424_ModernArch/train_gpt2.py at 42ab270216d4d10abbadc34e92f92e8a011a384f · KellerJordan/modded-nanogpt</a>: NanoGPT (124M) quality in 2.67B tokens. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1295509540032217162)** (11 messages🔥): 

> - `Evaluation of optimization-focused papers`
> - `Importance of good baselines`
> - `Misleading performance improvements`
> - `Impact of tuning on results`
> - `Utility of LLM papers` 


- **Critique of Optimization-Focused Papers**: Members agreed that they weigh results from optimization-focused papers less if they show a lack of knowledge in optimization basics or are tied to untuned hyperparameters.
   - Such papers are generally deemed less informative or even discarded based on the severity of their errors.
- **Baselines as Key Indicators**: A lack of a good learning rate or baseline is seen as a significant impediment to evaluating model performance.
   - Papers with untuned baselines are only deemed valuable if they show extreme performance deviations.
- **Watch Out for Misleading Claims**: "X improved performance" can imply only marginal stability improvements rather than genuine advancements, especially in poorly executed experiments.
   - Such claims can obscure the actual effectiveness of a model modification.
- **The A/B Result Dilemma in LLM Research**: Several members commented that many LLM papers often yield results that are merely A/B comparisons rather than substantive performance enhancements.
   - This can create a scenario where many papers are deemed nearly useless, allowing researchers to disregard the majority of findings.
- **Mamba Layers and Positional Encodings**: Discussion revealed that mamba layers do not benefit from positional encodings, demonstrating another layer of complexity in model evaluations.
   - This reflects a broader issue of understanding what modifications contribute to genuine improvements versus those that mask underlying instabilities.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1295515800936906752)** (10 messages🔥): 

> - `Fine-tuning libraries`
> - `Torchtune templates`
> - `Using ARC dataset`
> - `Loading Hugging Face datasets` 


- **Exploring fine-tuning libraries like Torchtune**: A member inquired about the absence of a fine-tuning library similar to *lm-evaluation-harness*, focusing on challenges with chat-template styles in *torchtune*.
   - They expressed a preference for evaluating fine-tuned models without requiring chat-templates.
- **Clarifying Torchtune's template functionality**: A maintainer discussed how the *PromptTemplate* system can be adjusted to fit evaluation requirements without chat-templates.
   - They recommended using the *QuestionAnswerTemplate* for the evaluation process.
- **Concerns over InstructTemplate in Torchtune**: The discussion highlighted the deprecation of *InstructTemplate* and the potential to use *QuestionAnswerTemplate* for similar functionality.
   - The *QuestionAnswerTemplate* formats prompts to align perfectly with the member's needs for evaluation.
- **Loading Hugging Face datasets**: A member queried about loading the *allenai/ai2_arc* dataset with a specific source command, encountering a 'Config name missing' error.
   - This revealed potential issues around the configuration required for accessing datasets within the *Hugging Face* ecosystem.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/torchtune/stable/basics/prompt_templates.html#custom-prompt-templates)?">Prompt Templates &mdash; torchtune 0.3 documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/generated/torchtune.data.QuestionAnswerTemplate.html#torchtune.data.QuestionAnswerTemplate)">torchtune.data.QuestionAnswerTemplate &mdash; torchtune 0.3 documentation</a>: no description found</li><li><a href="https://pytorch.org/torchtune/stable/basics/prompt_templates.html#defining-via-dotpath-string),">Prompt Templates &mdash; torchtune 0.3 documentation</a>: no description found
</li>
</ul>

</div>
  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1295549319767916554)** (164 messages🔥🔥): 

> - `Gradient Accumulation Fix in Unsloth`
> - `Decentralized Training Model - INTELLECT-1`
> - `LLM Fine-tuning Workflows`
> - `Discussion on Multi-GPU Support`
> - `Model Performance Comparisons` 


- **Gradient Accumulation Fix Improves Training**: Unsloth announced a fix addressing a significant bug in gradient accumulation that caused diverging training losses, improving accuracy by over **10x**.
   - Users are advised to update Unsloth and have access to new notebooks demonstrating the fixed method and its impact on training metrics.
- **Launch of INTELLECT-1 Decentralized Model**: Prime Intellect is launching **INTELLECT-1**, a 10-billion-parameter model, allowing anyone to contribute compute in a collaborative decentralized training run paving the way towards open-source AGI.
   - This follows their previous work with OpenDiLoCo, which scaled DeepMind's method for distributed AI model training to achieve significant model improvements.
- **Exploring LLM Fine-tuning Processes**: Users are discussing workflows for implementing fine-tuning of various LLMs, including training on specific formats and experimenting with model outputs.
   - There is an emphasis on understanding how data formatting can affect LLM behavior and output quality.
- **Discussions on Multi-GPU Training Capabilities**: There's a debate about the current limitations of multi-GPU support in Unsloth, with users expressing varying opinions on the necessity and utility of such setups.
   - Some participants suggest alternative frameworks like Deepspeed for better multi-GPU training, while others highlight Unsloth's efficiency on a single GPU.
- **Performance Comparisons between Models**: Participants compare the performance of various models in development, such as Qwen and Llama, within the context of fine-tuning and effective dataset utilization.
   - The conversation includes the importance of data quality over quantity for achieving optimal training results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1846235913443262891">Tweet from Daniel Han (@danielhanchen)</a>: Fixed a bug which caused all training losses to diverge for large gradient accumulation sizes.  1. First reported by @bnjmn_marie, GA is supposed to be mathematically equivalent to full batch training...</li><li><a href="https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/">A custom autocomplete model in 30 minutes using Unsloth (Community post)</a>: This is a guest post on the Continue Blog by Sophia Parafina, a developer advocate who has previously worked at Pulumi, Anaconda, and Docker.   Continue is an open-source AI code assistant. It&#x27;s ...</li><li><a href="https://colab.research.google.com/drive/1z0XJU2FCzDC8oyXa2Nd4jCxylRMI-o0-?usp=sharing#scrollTo=95_Nn-89DhsL">Google Colab</a>: no description found</li><li><a href="https://www.primeintellect.ai/blog/intellect-1">INTELLECT–1: Launching the First Decentralized Training of a 10B Parameter Model</a>: We&#x27;re excited to launch INTELLECT-1, the first decentralized training run of a 10-billion-parameter model, inviting anyone to contribute compute and participate. This brings us one step closer to...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1g4ego7/llm_training_bug_fixes_gradient_accumulation_was/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.zyphra.com/post/zamba2-7b">Zyphra</a>: no description found</li><li><a href="https://ollama.com/unclemusclez/unsloth-qwen2.5">unclemusclez/unsloth-qwen2.5</a>: Qwen 2.5 with Unsloth</li><li><a href="https://github.com/unclecode/crawl4ai">GitHub - unclecode/crawl4ai: 🔥🕷️ Crawl4AI: Open-source LLM Friendly Web Crawler &amp; Scrapper</a>: 🔥🕷️ Crawl4AI: Open-source LLM Friendly Web Crawler &amp; Scrapper - unclecode/crawl4ai</li><li><a href="http://unsloth.ai/blog/gradient">Bug Fixes in LLM Training - Gradient Accumulation</a>: Unsloth&#x27;s Gradient Accumulation fix solves critical errors in LLM Training.</li><li><a href="https://x.com/UnslothAI/status/1846231235749990699">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re releasing a new method that improves the way everyone trains LLMs.  There&#39;s a significant bug that causes loss miscalculations during training. Our Gradient Accumulation fix corrects ...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1295469775308193843)** (2 messages): 

> - `Service Restoration`
> - `Community Reactions` 


- **Service Restoration Celebrated**: A member remarked that it seems all is functioning again after recent issues, sparking relief in the community.
   - *☠️💀* emoticons were shared to express the mix of humor and relief about the service being back up.
- **Community Relief Symptoms**: The return of the services prompted community members to express their feelings through emojis, indicating relief and humor.
   - The atmosphere seems lighter as everyone acknowledges that the downtime is over, showing their playful side.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1295519356771504210)** (16 messages🔥): 

> - `Embedding Models`
> - `Quantization Methods`
> - `Integrating New Layers`
> - `Kaggle Package Installation`
> - `Finetuning Llama Models` 


- **Embedding Models Struggles**: A member mentioned that GPT models such as **Llama** and **Qwen** are not effective for embedding tasks due to their architecture limitations.
   - Another member suggested exploring alternatives like **BERT** or the latest sentence embedding models available on **Hugging Face**.
- **Quantization Method Issues**: An update revealed that the **q4_0** quantization method for models outputs gibberish, especially after updates to tokenizer merges.
   - Switching to **q4_k_m** quantization resolved the issue, indicating a potential bug with **q4_0** that may warrant an issue report in the relevant repository.
- **Integrating New Layers Discussion**: A member inquired about integrating a new layer into a model and training it, noting prior difficulties with saving changes.
   - Another user referenced **LoRA** as an example of effectively incorporating new small layers into models.
- **Kaggle Offline Package Installation Query**: A user posed a question about installing packages on **Kaggle** without internet access, indicating struggles with model uploads.
   - This raises concerns regarding the feasibility of offline package management in Kaggle environments.
- **Pushing Finetuned Models to Hugging Face**: A member expressed intent to push their finetuned **Llama 3B** model to **Hugging Face** for inference API use.
   - They questioned whether merging to **4-bit** would be necessary for optimized performance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1295533306384875531)** (2 messages): 

> - `Fine-tuning Autocomplete Models`
> - `Continue AI Code Assistant`
> - `Unsloth Jupyter Notebooks`
> - `Llama Model Performance` 


- **Fine-tuning Autocomplete Models Made Easy**: Sophia Parafina's blog post on [Continue](https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/) outlines how to fine-tune an autocomplete model on development data using **Unsloth**.
   - She emphasizes the convenience of fine-tuning with Jupyter notebooks on **free Google Colab instances**, which significantly reduces the time and expertise previously required.
- **Continue: The Open-Source AI Code Assistant**: The **Continue** community is focused on building an open-source AI code assistant that integrates multiple models, including chat and autocomplete functionalities.
   - A significant feature of Continue is its ability to record [development data](https://docs.continue.dev/customize/development-data?ref=blog.continue.dev) for enhancing model performance tailored to developer needs.
- **Impressive Performance of Llama Model**: A participant noted that the **llama-3.1-70b-instruct** model achieves a speed of **230 tokens per second**.
   - This performance metric indicates the potential efficiency and capability of the Llama architecture in processing inputs.



**Link mentioned**: <a href="https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/">A custom autocomplete model in 30 minutes using Unsloth (Community post)</a>: This is a guest post on the Continue Blog by Sophia Parafina, a developer advocate who has previously worked at Pulumi, Anaconda, and Docker.   Continue is an open-source AI code assistant. It&#x27;s ...

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1295606972715892737)** (6 messages): 

> - `SageAttention`
> - `Model Inference Optimization`
> - `Quantization Techniques` 


- **SageAttention promises faster model inference**: The paper *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration* proposes a novel quantization method that significantly boosts **operations per second** over FlashAttention2 and xformers by about **2.1** and **2.7 times**, respectively, while maintaining accuracy.
   - The authors claim comprehensive experiments show almost no loss in end-to-end metrics across various models, including those used in **language processing** and **image generation**.
- **Challenges in using SageAttention for training**: A member expressed that SageAttention's method does not seem usable for training after trying it in *unsloth's llama.py* file, leading to increased speed but diverging loss.
   - Another member clarified that SageAttention is primarily aimed at **inference** rather than training, which could account for the issues faced.
- **Exploring optimization of computation in transformers**: The discussion highlighted the **computational complexity** challenges in attention within transformer architectures, specifically contrasting O(N^2) for attention against O(N) for linear transformations.
   - This context underscores the importance of effective quantization methods to address inefficiencies during model inference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N^2), compared to O(N) for linear transformations. When ...</li><li><a href="https://github.com/HazyResearch/lolcats/blob/main/lolcats_preprint_v0.pdf">lolcats/lolcats_preprint_v0.pdf at main · HazyResearch/lolcats</a>: Contribute to HazyResearch/lolcats development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1295461344744570921)** (159 messages🔥🔥): 

> - `Reasoning Feature in Perplexity`
> - `ProSearch Improvements`
> - `Model Performance Comparisons`
> - `Facial Recognition Concerns`
> - `User Experience Issues` 


- **Reasoning Feature's Variability**: Users noted that triggering the new **reasoning feature** in ProSearch feels random and varies based on question complexity, leading to inconsistency in analyses.
   - The old reasoning model was seen as more reliable, while the new version has led to an increase in hallucinations during information generation.
- **ProSearch and Mac App Delay**: Many users expressed frustration regarding the **delay of the Mac app**, which was originally scheduled for an earlier release date.
   - In addition, discussions mentioned ongoing issues with **missing threads** and sluggish performance in the app.
- **User Expectations from AI Models**: A conversation took place about the performance of various AI models, with some users suggesting that **NotobkLM** likely uses **Gemini Pro** and comparing it to **ChatGPT 4o**.
   - Users are also exploring how Perplexity can compete or add functionalities like UX-focused features that other services do not currently offer.
- **Facial Recognition Software Limitations**: Concerns were raised about the feasibility of using AI for **facial recognition** to help victims of bullying on social media, with suggestions that tools like **Yandex** only work within specific regions.
   - Users acknowledged that scraping private accounts on platforms like **Instagram** and **Snapchat** poses significant challenges for any AI tool.
- **Streaming Performance Issues**: Users reported issues regarding **searching capabilities** in the iOS app, noting that results are often cut off when accessing sources.
   - Additionally, there were mentions of collections and saved threads disappearing temporarily, although this issue appeared to resolve later.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/new-prosearch-feature-allows-perplexity-users-to-generate-visual-data-charts/">Perplexity rolled out chart generation feature for ProSearch</a>: Discover Perplexity&#x27;s new ProSearch feature that generates charts from search data, offering powerful visual insights. Perfect for financial and demographic analysis.</li><li><a href="https://civitai.com/user/karlcrane2015">Creator Profile | Civitai</a>: Learn more about this awesome creator on Civitai.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1295473200909647874)** (8 messages🔥): 

> - `Adobe's AI Video Model`
> - `NASA's Europa Clipper Launch`
> - `Chinese Breakthrough in RSA Encryption`
> - `AMD's New AI Chips`
> - `Google's Acquisition of Nuclear Power` 


- **Adobe's AI Video Model Revealed**: Perplexity AI highlighted **Adobe's AI Video Model** as a significant development in video editing and production capabilities, offering advanced features that are set to revolutionize workflows.
   - The implications of this model could transform content creation, making it faster and more accessible for users.
- **NASA's Launch of the Europa Clipper Mission**: The **NASA Europa Clipper** mission has been successfully launched, aiming to explore Jupiter's ice-covered moon, Europa, for potential signs of life.
   - Experts are excited about the possibility of discovering new data that can shed light on the moon's subsurface ocean.
- **Chinese Researchers Break RSA Encryption**: Recent reports indicate that **Chinese researchers** have successfully broken RSA encryption, a major concern for cybersecurity experts worldwide.
   - This breakthrough raises questions about the current reliance on this encryption method for securing sensitive information.
- **AMD Introduces New AI Chips**: AMD has announced its latest **AI chips**, designed to enhance performance in machine learning tasks and optimize computational efficiency.
   - These innovations are anticipated to compete strongly in the burgeoning AI hardware market.
- **Google Makes Move to Acquire Nuclear Power**: News surfaced that **Google** is considering the acquisition of **nuclear power** technology, potentially shifting its energy strategy towards sustainable sources.
   - This move could position Google as a leader in utilizing green energy solutions in the tech industry.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1295570544808165466)** (4 messages): 

> - `Perplexity API limitations`
> - `BAAs for healthcare use case`
> - `Creating conversational chatbots` 


- **Perplexity API faces domain limitations**: Some users noted that with the Perplexity API, **only three domains** can be specified for queries.
   - Additionally, they found that **search results are not limited** to those specified domains, which poses a challenge.
- **Inquiries about BAAs for healthcare**: A user inquired whether Perplexity signs **Business Associate Agreements (BAAs)** for enterprise use, especially in healthcare scenarios.
   - This request highlights the need for **clarity on compliance** for using the API in sensitive industries.
- **Seeking resources for chatbot creation using Perplexity API**: A request was made for resources on how to create a **conversational chatbot using the Perplexity API**.
   - The user expressed gratitude in advance and **appreciated any guidance** from the community.


  

---



### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1295579150261948609)** (107 messages🔥🔥): 

> - `Aider's performance and features`
> - `Issues with API usage and quotas`
> - `LLM integration and script usage`
> - `Model effectiveness and comparison`
> - `Weak models and prompt caching` 


- **Aider's ability to use multiple models**: Users discussed the feasibility of running multiple Aider instances simultaneously for larger and smaller tasks, with one stating that it should be fine as long as they don't edit the same files.
   - A user humorously referred to it as an 'LLM party'.
- **API usage errors and quotas**: A user encountered a 'Resource exhausted' error when using OpenRouter with the 4o-mini model, suggesting that they likely exceeded their quota or that OpenRouter hit its quota limits.
   - Another user emphasized that Anthropic's model flagged inputs for moderation, which could be causing repeated API connection errors.
- **Scripting strategies with LLMs**: There were discussions about using scripts in conjunction with Aider to handle large code bases efficiently, with one user mentioning their experience with a systematic naming convention for sections of code.
   - Another user mentioned plans for an LLM agent framework that automates various tasks, including system administration.
- **Comparison of models for coding tasks**: Some users noted that Sonnet-3.5 was superior for their specific use cases compared to other models like Gemini, particularly in non-web development tasks.
   - One user highlighted testing different models and noted that while some performed better, Sonnet-3.5 consistently offered the best results.
- **Weak model functionalities and prompt caching**: A user questioned whether omitting the --weak-model flag defaults to Sonnet and requested clarification on the functionalities of Gemini as a weak model.
   - Another user pointed out that Aider defaults to using Claude with the Anthropic API unless specified otherwise.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/config/options.html#--commit-prompt-prompt">Options reference</a>: Details about all of aider’s settings.</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet:beta">Claude 3.5 Sonnet (self-moderated) - API, Providers, Stats</a>: Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same Sonnet prices. Run Claude 3.5 Sonnet (self-moderated) with API</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1295473019262603325)** (42 messages🔥): 

> - `Aider model API key issues`
> - `File modification behavior in Aider`
> - `Scripting Aider commands`
> - `Error handling for Aider edits`
> - `Gemini model integration with Aider` 


- **Aider struggles with API key validations**: Users are experiencing **API validation errors** when attempting to configure Aider with the Gemini model after setting the key in their `.env` file.
   - One user confirmed the key works from the command line, indicating possible issues with their scripting setup.
- **Confusion over file modification behavior**: Members shared conflicting experiences regarding whether Aider uses the latest file from the file system or from the Git repository when processing modifications.
   - Despite expectations for the file system version to be used, one user noted discrepancies that led to unexpected behavior.
- **Scripting Aider commands and configurations**: Users discussed how to **script Aider commands** via the command line or Python, highlighting the need for proper environment loading when scripting.
   - One user modified an example script to set the Gemini model but encountered errors related to environment variables.
- **Error handling strategies for Aider edits**: Members sought methods to address instances where Aider fails to apply changes, expressing frustration with unsatisfactory outputs.
   - Suggestions included using `/clear` or `/drop` commands to mitigate distractions during edits.
- **Gemini model integration queries**: There was a specific inquiry about how to properly configure and use the **Gemini-1.5 Pro model** within Aider, with emphasis on ensuring the API key is correctly set.
   - Documentation was referenced, yet users faced challenges related to API errors and the required environment configuration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: You can script aider via the command line or python.</li><li><a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>: aider is AI pair programming in your terminal
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1295476202072707115)** (101 messages🔥🔥): 

> - `HuggingFace account recovery`
> - `Data Science and ML job security`
> - `Llama 3.2 inference speed`
> - `AI in job roles`
> - `AWS Cognito integration issues` 


- **Urgent HuggingFace account recovery needed**: A member reported their HuggingFace account was hacked and deleted, seeking urgent assistance for recovery. They were advised to email [website@huggingface.co](mailto:website@huggingface.co) for help.
   - Another member noted that recovery might take a few days and to wait for a response.
- **Concerns about AI automation in jobs**: Discussions revealed anxiety surrounding AI's rapid development and its potential to automate jobs in Data Science and ML. Members expressed hope that AI will change job roles rather than replace them, advocating for a shift towards more creative work.
   - One member compared AI's impact to previous technological advances, suggesting it will enhance rather than eliminate job roles.
- **Llama 3.2 model inference question**: A user reported that running inference on a large dataset with the Llama 3.2 1B model using an A100 GPU took over 14 hours, raising concerns about efficiency. Suggestions about optimizing load and inference methods were discussed.
   - They shared their approach for loading the model and executing inference, seeking advice on potential improvements.
- **Issues with AWS Cognito and HuggingFace integration**: A user described difficulties in integrating HuggingFace login with AWS Cognito, particularly with receiving only an ACCESS_TOKEN instead of the required ID_TOKEN. They emphasized the urgency of their situation as it was causing project delays.
   - Members were encouraged to share any insights or solutions to resolve the integration problems.
- **Chat system with LLM and static widgets setup**: A member inquired about setting up a chat using LLM that includes both text generation and static widgets without redundancy or loss of conversational style. They highlighted the challenges faced in balancing dynamic responses with templated constraints.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/rush-hour-you-and-me-me-and-you-yu-mee-gif-6521388707144075848">Rush Hour You And Me GIF - Rush hour You and me Me and you - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/theodd1sout-vsauce-or-is-it-gif-16159095273288783425">Theodd1sout Vsauce GIF - Theodd1sout Vsauce Or is it - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1295649074329948210)** (3 messages): 

> - `Calculus learning`
> - `Flet and OpenVino exploration` 


- **Diving into Calculus After Fast AI**: A user shared their plan to finish **Calculus** today after completing the **Fast AI Part 1 course** and studying **Linear Algebra**.
   - *They expressed excitement about delving further into neural net code with enhanced understanding.*
- **Exploring Flet and OpenVino**: Another user reported that they are learning **Flet** and **OpenVino**, signaling an interest in expanding their skill set in these areas.
   - *No further details were provided, but these technologies are gaining traction in AI application development.*


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

laolala: https://huggingface.co/movaxbx/OpenHermes-Emojitron-001
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1295499682545406042)** (8 messages🔥): 

> - `VividNode AI chatbot`
> - `GPT4Free explanation`
> - `HybridAGI features`
> - `Metaflow on Google Colab`
> - `Open TTS Tracker as HF dataset` 


- **VividNode AI chatbot lands on Mac and Linux**: Exciting news! The **VividNode AI chatbot** is now available for Mac and Linux platforms, encouraging users to explore its capabilities [here](https://github.com/yjg30737/pyqt-openai).
   - *If you're interested in contributing or starting a side project,* reach out to the author for more ideas.
- **Understanding GPT4Free**: A member inquired about **GPT4Free** and how to access it for free, prompting a response that it's not truly free due to slower speeds and sometimes incomplete responses.
   - Details regarding **GPT4Free** can be found in the [documentation](https://github.com/xtekky/gpt4free).
- **Introducing HybridAGI Framework**: A detailed introduction to **HybridAGI**, a Programmable Graph-based Open Source Framework, highlights its focus on agent behavior via a graph-based programming language.
   - Key features include **Human-in-the-Loop control** and **Hybrid Vector/Graph Memory**; find more on its [GitHub page](https://github.com/SynaLinks/HybridAGI).
- **Self-host Metaflow on Google Colab**: **Metaflow** can now be self-hosted using Google Colab, as discussed in a newly published article that dives into practical implementation without relying on S3.
   - The article covers various aspects of Metaflow including its features and future potential, accessible [here](https://huggingface.co/blog/Aurelien-Morgan/stateful-metaflow-on-colab).
- **Open TTS Tracker GitHub Repo Conversion**: The **Open TTS Tracker GitHub Repo** has been converted as a Hugging Face dataset, providing structured information on various TTS models.
   - Users can explore the dataset [here](https://huggingface.co/datasets/Pendrokar/open_tts_tracker) to find different TTS functionalities and standards.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/Aurelien-Morgan/stateful-metaflow-on-colab">Fancy Stateful Metaflow Service + UI on Google Colab ?</a>: no description found</li><li><a href="https://github.com/yjg30737/pyqt-openai">GitHub - yjg30737/pyqt-openai: VividNode: Multi-purpose Text &amp; Image Generation Desktop Chatbot (supporting various models including GPT).</a>: VividNode: Multi-purpose Text &amp; Image Generation Desktop Chatbot (supporting various models including GPT). - yjg30737/pyqt-openai</li><li><a href="https://github.com/SynaLinks/HybridAGI">GitHub - SynaLinks/HybridAGI: The Programmable Cypher-based Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected</a>: The Programmable Cypher-based Neuro-Symbolic AGI that lets you program its behavior using Graph-based Prompt Programming: for people who want AI to behave as expected - SynaLinks/HybridAGI</li><li><a href="https://huggingface.co/datasets/Pendrokar/open_tts_tracker">Pendrokar/open_tts_tracker · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1295622381452136618)** (8 messages🔥): 

> - `Reading Group Event`
> - `Channel Access Issues`
> - `Paper Discussion` 


- **Reading Group Event Sparks Interest**: A member highlighted seeing the Reading Group event in a specific channel, prompting questions about its details [here](https://discord.com/events/879548962464493619/1293961951260446811).
   - Other members were unsure about the event's location, pointing to a different channel for the potential discussion.
- **Access Issues to Channel**: Members expressed difficulties accessing the discussed channel, with one stating, *'I don't have access to that channel yet.'*
   - Suggestions were made to mark every role for potential access or to directly ask another member for help.
- **Paper Discussion Link Shared**: A member shared a link to a paper ([arxiv link](https://arxiv.org/abs/2405.10725)) related to the Reading Group event, suggesting it would be discussed in the channel.
   - This prompted more questions about the channel location and visibility among members.
- **Confirmation of Visibility**: After some back-and-forth, one member confirmed they can now see the event content in the channel, answering the earlier questions.
   - This appeared to facilitate the flow of information and clarify access among the group participants.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1295681808763195403)** (2 messages): 

> - `Flutter Development`
> - `AI App Collaboration` 


- **Offering Flutter Development Collaborations**: A member announced their availability as a **Flutter developer** for collaboration on building **AI applications**.
   - They expressed a willingness to team up with others interested in the development of AI-focused projects.
- **Call for Collaboration**: The same member reiterated their interest in **collaborating** with others who are looking to develop AI applications using Flutter.
   - This highlights an ongoing need for **partnerships** in the AI app development space.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1295485921780568124)** (1 messages): 

> - `DiT Training`
> - `Super Mario Bros. Gameplay Images`
> - `Custom VAE Compression` 


- **Successful DiT Training Achieved**: A member reported progress in training a **DiT** (Detection in Training) from scratch, noting that they finally obtained some good results.
   - This training involved using specific **gameplay images of Super Mario Bros.**, showcasing potential applications in gaming.
- **Innovative Custom VAE Compressed to 24x**: They employed a **custom VAE** (Variational Autoencoder) with a remarkable compression ratio of **24x** for their training.
   - This choice indicates a focus on optimizing data efficiency while training the DiT model.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1295726111384277042)** (1 messages): 

> - `Gradio 5 Launch`
> - `Product Hunt Support` 


- **Gradio 5 Launch Announcement**: We've just launched **Gradio 5** on [Product Hunt](https://www.producthunt.com/posts/gradio-5-0), and we would love your support!
   - *Please take a moment to check it out and show your appreciation* 🧡.
- **Encouragement for Community Support**: The team encouraged everyone to spare a moment to support **Gradio 5** on Product Hunt. This community-driven effort aims to boost visibility and engagement for the launch.
   - Members are invited to participate in discussions and share feedback on the new features.



**Link mentioned**: <a href="https://www.producthunt.com/posts/gradio-5-0"> Gradio 5.0 - The easiest way to build AI web apps | Product Hunt</a>: An open-source library for building and sharing web-based AI apps with ease. Deploy, customize, and share machine learning model demos in just a few lines of Python.

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1295671338983358544)** (1 messages): 

> - `Hermes 3 Llama 3.1 405B`
> - `Nous Hermes Yi 34B Deprecation` 


- **Hermes 3 Llama 3.1 405B is now a paid model**: The **Hermes 3 Llama 3.1 405B Instruct** model is now available for **$1.79/month**, though a free variant remains accessible at [OpenRouter](https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b).
   - *Don't miss out on this updated pricing structure for powerful AI functionality!*
- **Nous Hermes Yi 34B is deprecated**: The **Nous Hermes Yi 34B** model has been deprecated by all service providers, making it no longer available for use.
   - *Users are encouraged to transition to alternative models in light of this deprecation.*



**Link mentioned**: <a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API, Providers, Stats</a>: Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced agentic capabilities, much better roleplaying, reasoning, multi-turn conversation, long context coheren...

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1295487224715935915)** (90 messages🔥🔥): 

> - `AI Model Performance`
> - `Chatbot Development`
> - `OpenRouter Features`
> - `Model Comparison`
> - `Provider Issues` 


- **Ranking of AI Models**: Users discussed the performance of various AI models for chatting and role-playing, with **Llama-3-8b-Instruct** and **GPT-4o** being highlighted for following instructions well.
   - *Grok 2 mini* and *Gemini 1.5 Pro* were also mentioned as viable options, though *Opus* faced some critique regarding its quirks.
- **Chatbot Design Techniques**: A user inquired about creating a hidden AI chatbot that avoids generic refusal messages to insults, proposing the use of another LLM for filtering bad content.
   - Others suggested using models like *Llama Guard* for additional support in checking messages before allowing a response.
- **OpenRouter's Features and Usage**: The community discussed how to restrict model usage within OpenRouter by utilizing headers to filter out unwanted providers, enhancing privacy settings.
   - A member highlighted the *LiteLLM guardrails feature*, which offers safeguards like request checking using models like *Llama Guard*.
- **Issues with Infermatic Provider**: One user reported issues with the **Infermatic** provider, stating that their chats had begun outputting irrelevant responses unexpectedly.
   - The community was alerted to potential service disruptions that may have arisen within a short timeframe.
- **User Experiences and Feedback**: Users expressed excitement about new features in the OpenRouter playground, such as drag-and-drop file uploads and image pasting capabilities.
   - Another user noted a significant delay of over **90 seconds** for prompts in the **Gemini** model, indicating varying performance experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/settings/privacy">Privacy | OpenRouter</a>: Manage your privacy settings</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found</li><li><a href="https://openrouter.ai/docs/provider-routing#custom-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://openrouter.ai/meta-llama/llama-guard-2-8b">LlamaGuard 2 8B - API, Providers, Stats</a>: This safeguard model has 8B parameters and is based on the Llama 3 family. Just like is predecessor, [LlamaGuard 1](https://huggingface. Run LlamaGuard 2 8B with API</li><li><a href="https://openrouter.ai/rankings/roleplay?view=week">LLM Rankings: roleplay | OpenRouter</a>: Language models ranked and analyzed by usage for roleplay prompts
</li>
</ul>

</div>
  

---



### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1295465473038745673)** (72 messages🔥🔥): 

> - `Nous Research Community`
> - `Gradient Accumulation Fix`
> - `Zamba2-7B Performance`
> - `AI Training Techniques`
> - `Open Source AI Projects` 


- **Nous Research Community's Origins**: The Nous Research community started as a group on Discord focused on AI research and has now evolved into a funded tech company.
   - Members share ideas, collaborate on projects, and discuss various AI models and techniques, fostering an engaging environment.
- **Gradient Accumulation Bug Fix Released**: A significant bug in gradient accumulation causing divergent training losses has been fixed, as detailed by the [UnslothAI team](https://x.com/danielhanchen/status/1846235913443262891).
   - The fix improves training losses consistency across various setups and is now available for all users to implement.
- **Introduction of Zamba2-7B Model**: Zyphra announced the Zamba2-7B model, claiming it outperforms existing models like Llama3 and Mistral in both quality and performance.
   - This new model is designed for efficient deployment on consumer GPUs and detailed its capabilities in a recent [blog post](https://www.zyphra.com/post/zamba2-7b).
- **Discussion on AI Training Techniques**: A member raised concerns about training AI with sequence lengths of 16384, sparking a conversation about the implications and best practices.
   - There's a recognition that while innovative, such techniques could pose challenges in practical applications, highlighting the dynamic nature of AI training.
- **Community Member Skills and Contributions**: A fullstack blockchain developer expressed willingness to contribute skills to the community, displaying openness to collaboration.
   - Members frequently share their expertise and seek assistance, reinforcing the communal spirit of the Nous Research Discord.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/UnslothAI/status/1846231235749990699">Tweet from Unsloth AI (@UnslothAI)</a>: Today, we’re releasing a new method that improves the way everyone trains LLMs.  There&#39;s a significant bug that causes loss miscalculations during training. Our Gradient Accumulation fix corrects ...</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891">Tweet from Daniel Han (@danielhanchen)</a>: Fixed a bug which caused all training losses to diverge for large gradient accumulation sizes.  1. First reported by @bnjmn_marie, GA is supposed to be mathematically equivalent to full batch training...</li><li><a href="https://x.com/danielhanchen/status/1846235913443262891?s=46">Tweet from Daniel Han (@danielhanchen)</a>: Fixed a bug which caused all training losses to diverge for large gradient accumulation sizes.  1. First reported by @bnjmn_marie, GA is supposed to be mathematically equivalent to full batch training...</li><li><a href="https://a16z.com/podcast/distro-and-the-quest-for-community-trained-ai-models/">DisTrO and the Quest for Community-Trained AI Models | Andreessen Horowitz</a>: Bowen Peng and Jeffrey Quesnelle of Nous Research discuss their mission to accelerate open source AI research, including with a new project called DisTrO.</li><li><a href="https://www.zyphra.com/post/zamba2-7b">Zyphra</a>: no description found</li><li><a href="https://delphidigital.io/crypto-ai">Crypto x AI Month</a>: Crypto's largest virtual AI conference where we host some of the most prolific crypto AI builders and visionaries. Join us for live talks, debates and market thesis including Nous Research, Prime Inte...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1295810500520574989)** (3 messages): 

> - `Llama3 identification`
> - `LLMs and animal comparison`
> - `Statistical evaluation on animal identification` 


- **Llama3 Identified as an Octopus**: A member humorously claims **Llama3** identifies more with an **octopus**, noting that the spread of LLMs correlating with animals is surprisingly small.
   - Common identifications include **octopus**, **owl**, **wolf**, and **dolphin**, with no sightings of llamas reported.
- **Avoiding AI Clichés in Responses**: The same member expressed needing to use a system prompt to prevent typical AI responses starting with 'As an AI...'.
   - This highlights a common frustration in interacting with LLMs and their default conversational patterns.
- **Need for Statistical Evaluation**: The member mentioned the intent to run a statistical evaluation on the responses regarding animal identification.
   - This aims to provide a clearer view of how LLMs align with various animal identities, possibly influencing future interactions.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1295475077650976932)** (5 messages): 

> - `Model Collapse Phenomenon`
> - `SageAttention Quantization Method`
> - `Slow Response Issues` 


- **Model Collapse due to Synthetic Data**: A study highlights the **model collapse phenomenon** where even **1%** synthetic data in training can worsen performance, despite increasing data volumes, indicating a substantial risk in training large models like ChatGPT and Llama. This research suggests that larger models may exacerbate the issue.
   - The findings delve into training dynamics under **supervised regression settings** and question common practices in model scaling, drawing attention to significant implications for future model designs.
- **SageAttention: Driving Efficiency in Transformer Models**: Introducing **SageAttention**, a novel quantization method that boosts performance in attention mechanisms, achieving **2.1 to 2.7 times** improvements in operations per second compared to existing methods like FlashAttention2 and xformers. This approach maintains accuracy while reducing computational complexity in various model types, including language and image processing.
   - Comprehensive experiments validate that SageAttention incurs **almost no loss** in end-to-end metrics, making it a promising avenue for inference acceleration in transformer architectures.
- **User Reports Slow Response Times**: Members expressed concerns about being met with a 'Rate exceeded' message during usage, indicating potential slowdowns in system performance. Such reports suggest that user experience issues are affecting engagement levels on the platform.
   - These slow response reports highlight ongoing challenges and underscore the need for attention to optimize **system responsiveness** and user satisfaction.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N^2), compared to O(N) for linear transformations. When ...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>: Within the scaling laws paradigm, which underpins the training of large neural networks like ChatGPT and Llama, we consider a supervised regression setting and establish the existance of a strong form...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

xandykati98: https://x.com/AISafetyMemes/status/1846220545542529329
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1295475077650976932)** (5 messages): 

> - `Model Collapse Phenomenon`
> - `SageAttention Quantization`
> - `Performance Issues`
> - `SageAttention vs. FlashAttention` 


- **Model Collapse due to Synthetic Data**: [Research paper](https://arxiv.org/abs/2410.04840) reveals that even **1%** synthetic data in training sets can lead to substantial **model collapse**, undermining the performance of large neural networks like ChatGPT.
   - The study highlights that increasing model size may actually **exacerbate** this issue, counter to current training trends.
- **SageAttention’s Efficiency in Attention Mechanism**: [SageAttention](https://arxiv.org/abs/2410.02367) offers a novel quantization method that enhances attention computational efficiency, outperforming FlashAttention2 and xformers by roughly **2.1 times** and **2.7 times**, respectively.
   - This approach maintains accuracy even as it accelerates inference, showing almost no loss in performance metrics across various applications.
- **Concerns over System Performance**: Members expressed issues related to the system's **performance**, with one noting it feels **slow**.
   - Another member added to this sentiment, stating they received a **'Rate exceeded'** message, indicating potential system overload.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N^2), compared to O(N) for linear transformations. When ...</li><li><a href="https://arxiv.org/abs/2410.04840">Strong Model Collapse</a>: Within the scaling laws paradigm, which underpins the training of large neural networks like ChatGPT and Llama, we consider a supervised regression setting and establish the existance of a strong form...
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1295552267860050011)** (1 messages): 

> - `Lux-AI Challenge`
> - `Team Collaboration` 


- **Lux-AI Challenge: GitHub Repository News**: A member shared a link to the [Lux-AI Challenge's GitHub repository](https://github.com/Lux-AI-Challenge/Lux-Design-S3) inviting others to contribute to its development.
   - This initiative aims to encourage collaboration, with hopes for team formations to enhance project outcomes.
- **Request for Team Collaboration on Lux-AI Project**: In conjunction with the GitHub link, a member asked if anyone would be interested in teaming up for the Lux-AI project.
   - *Would anyone be interested in teaming up for this?* resonates with a call for community engagement in project contribution.



**Link mentioned**: <a href="https://github.com/Lux-AI-Challenge/Lux-Design-S3">GitHub - Lux-AI-Challenge/Lux-Design-S3</a>: Contribute to Lux-AI-Challenge/Lux-Design-S3 development by creating an account on GitHub.

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1295541141047083059)** (15 messages🔥): 

> - `triton-lang installation issues`
> - `LLVM support for ARM`
> - `Triton for Windows builds`
> - `CUDA error handling`
> - `Packed data loop issues` 


- **Installation Woes for Triton on Jetson**: A user reported trouble building **triton-lang** on a **Jetson Orin AGX 64GB**, with CUDA mistakenly identifying Unified Memory as `AMD GPU`. They mentioned they would rebuild Triton and hoped that the LLVM build was the root of the installation failure.
   - Another member suggested checking **LLVM** support for ARM, referencing related issues [here](https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed).
- **Unofficial support for Triton on Windows**: Discussion surfaced around unofficial builds of **Triton for Windows**, with links shared to a GitHub repository claiming Windows compatibility. Claims were made that some users had successfully built it despite challenges related to MSVC compatibility.
   - Exploratory efforts were noted regarding a closed PR for Triton on Windows with reports of builds, hinting at possible partial success.
- **Troubles with Triton and CUDA Errors**: A member faced the error `triton LLVM ERROR: mma16816 data type not supported` during kernel execution on **A100** and **H100** GPUs. Previous workarounds like adjusting loop implementations has stopped being effective according to recent tests.
   - Another member suggested investigating details from a [GitHub link](https://github.com/triton-lang/triton/blob/17d633a64e43337037d2e873b029fab92422762f/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L125C20-L125C25) that reportedly offered a solution to related issues.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/StableDiffusion/comments/1g45n6n/triton_3_wheels_published_for_windows_and_working/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/trito">Trito - Overview</a>: Trito has one repository available. Follow their code on GitHub.</li><li><a href="https://github.com/triton-lang/triton/blob/17d633a64e43337037d2e873b029fab92422762f/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp#L125C20-L125C25">triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp at 17d633a64e43337037d2e873b029fab92422762f · triton-lang/triton</a>: Development repository for the Triton language and compiler - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton?tab=readme-ov-file#install-from-source">GitHub - triton-lang/triton: Development repository for the Triton language and compiler</a>: Development repository for the Triton language and compiler - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton/issues?q=sort%3Aupdated-desc+is%3Aissue+jetson+is%3Aclosed">Issues · triton-lang/triton</a>: Development repository for the Triton language and compiler - Issues · triton-lang/triton</li><li><a href="https://github.com/woct0rdho/triton">GitHub - woct0rdho/triton-windows: Fork of the Triton language and compiler for Windows support</a>: Fork of the Triton language and compiler for Windows support - woct0rdho/triton-windows</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/gnvRBZvkZk">Reddit - Dive into anything</a>: no description found</li><li><a href="https://rosenzweig.io/blog/asahi-gpu-part-3.html">Dissecting the Apple M1 GPU, part III</a>: no description found</li><li><a href="https://dougallj.github.io/applegpu/docs.html">Apple G13 GPU Architecture Reference</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1295471062095757385)** (6 messages): 

> - `Model Parallelism vs Data Parallelism`
> - `Titanet-large Model Performance`
> - `Learn PyTorch Course`
> - `Research Paper Release`
> - `CPU Utilization in GPU Tasks` 


- **Model Parallelism often a secondary choice**: Data parallelism is typically the first approach taken in deep learning due to its straightforward implementation, with model parallelism used only when memory constraints necessitate it.
   - Members suggested that those employing model parallelism have usually already optimized data parallelism.
- **Titanet-large model running on GPU issue**: A user reported high CPU utilization alongside GPU usage when running the scripted Titanet-large model on a GPU T4 machine.
   - They sought advice for debugging this performance issue while using a specific model conversion code.
- **Learn PyTorch Course Now Available**: A link to the [Learn PyTorch for Deep Learning: Zero to Mastery](https://www.learnpytorch.io/) course was shared, hailed as the second best resource to learn PyTorch.
   - This course promises to teach foundational concepts through video content centered on an online book format.
- **Research Paper Release Excitement**: Members expressed excitement over the release of a new research paper, signifying a significant advancement in the field.
   - They shared a link to the paper and various authors contributing to the research.
- **High CPU Usage in Deep Learning Tasks**: A user inquired about high CPU utilization while running GPU-based tasks, indicating a desire to diagnose this unexpected performance aspect.
   - They provided specific code used for model conversion, hoping for insights from the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.06511">TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training</a>: The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of...</li><li><a href="https://www.learnpytorch.io/">Home</a>: Learn important machine learning concepts hands-on by writing PyTorch code.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1295481627983478875)** (9 messages🔥): 

> - `LegoScale`
> - `Modular 3D Parallelism`
> - `FSDP2`
> - `TorchTitan`
> - `Large Language Models` 


- **LegoScale brings new capabilities for LLM training**: The recently discussed paper introduces **LegoScale**, an open-source, PyTorch-native system for **3D parallel pre-training** of large language models, achieving significant performance improvements.
   - Its features include **customizable activation checkpointing**, **fp8 training support**, and built-in fault recovery.
- **Insights on FSDP2 spark curiosity**: The paper delves into **FSDP2** techniques, generating significant interest among members who are eager to understand its complexities.
   - While some are still grappling with its concepts, the discussion is fostering a deeper exploration into its implementation.
- **Connection to TorchTitan identified**: Members drew parallels between LegoScale and **TorchTitan**, suggesting that the two may be closely related or even the same.
   - One member humorously noted that it “looks insanely cool” while another confirmed its identity as **TorchTitan**.
- **Fresh interest in recent publications**: The paper's freshness sparked excitement in the community, especially after it was shared on LinkedIn by someone associated with **Hugging Face**.
   - This highlights the ongoing relevance and interest in advancements in large language model training techniques.



**Link mentioned**: <a href="https://openreview.net/forum?id=SFN6Wm7YBI">LegoScale: One-stop PyTorch native solution for production ready...</a>: The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions...

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1295520217979555860)** (2 messages): 

> - `CUDA optimization experts`
> - `Open source training framework` 


- **Together.AI seeks CUDA optimization wizards**: Together.AI is hiring **CUDA optimization experts** to enhance kernels for popular models, aiming to achieve thermal throttling on all GPUs. More details can be found on their [job listing](https://job-boards.greenhouse.io/togetherai/jobs/4188119007).
   - They emphasize that responses to the self-identification survey for government reporting are voluntary and confidential.
- **Hiring for open source training framework role**: A team is looking to hire a developer to work on their **open source training framework** that trained **starcoder2** significantly faster than **megatron-lm**. Interested candidates can apply through [ServiceNow's listing](https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer).
   - ServiceNow has transformed organizational workflows since its inception in 2004 under the vision of Fred Luddy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jobs.smartrecruiters.com/ServiceNow/744000019737886-staff-machine-learning-developer">Staff Machine Learning Developer</a>: Company Description: Tout a commencé sous le soleil de San Diego, en Californie, en 2004, lorsqu’un ingénieur visionnaire, Fred Luddy, a vu le potentiel de transformer notre façon de travailler. Aujou...</li><li><a href="https://job-boards.greenhouse.io/togetherai/jobs/4188119007">Job Application for Systems Research Engineer, GPU Programming at Together AI</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

nusha.m: What are some good beginner projects for learning GPU programming with CUDA or OpenCL?
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1295844511787778133)** (1 messages): 

> - `Matrix Multiplication Kernels`
> - `Shared Memory Performance`
> - `A100 Speed Metrics` 


- **Skepticism on A100 Matrix Multiplication Speeds**: Concerns were raised regarding the **matrix multiplication kernels** speeds on the **A100** in chapter 5, suggesting that the assumptions made—like no L1 cache and no warp stalls for the naive kernel—are unrealistic.
   - A suggestion was made to include a *footnote about real-world performance considerations*, highlighting potential discrepancies.
- **Need for Real-World Context in Kernel Performance**: The discussions pointed out past questions on the **shared-memory kernel's** minimal speed-up, indicating a common misunderstanding about its performance benefits.
   - Acknowledging these real-world challenges, members agreed that **pedagogical clarity** is essential, yet must also align with actual performance data to avoid misleading readers.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1295473811755368508)** (8 messages🔥): 

> - `Gradient Accumulation Compatibility`
> - `Raspberry Pi Support for TorchAO`
> - `FP8 Format Performance Differences`
> - `CUTLASS-based W4A8 Kernel PR` 


- **Gradient Accumulation got some hurdles**: A member inquired why **gradient accumulation** isn't compatible with `offload=True` in `CPUOffloadOptimizer`, prompting a technical explanation about the complexities involved in moving gradients to CPU.
   - *It’s technically possible,* but allocating extra memory and interleaving data movement with computations adds significant complexity.
- **TorchAO works on Raspberry Pi!**: A member discovered that **TorchAO** already works on Raspberry Pi, pointing to the [GitHub issue](https://github.com/pytorch/ao/issues/1076).
   - They noted that currently, they install version **0.1** due to the lack of published binaries for aarch64 Linux.
- **FP8 Formats: Performance Insights**: A member asked about potential performance differences between **FP8 E4M3** and **E5M2** formats during inference, which led to discussion on their characteristics.
   - Another commented that while there shouldn't be latency differences, the **range** and **precision** differ significantly, with E4M3 yielding more accurate results.
- **CUTLASS W4A8 Kernel Development**: A member shared their open PR for a **CUTLASS-based W4A8 kernel** on [GitHub](https://github.com/pytorch/ao/pull/880), welcoming feedback on remaining issues and performance notes.
   - Comments in the PR outline the current challenges and invite the community to contribute their insights.



**Link mentioned**: <a href="https://github.com/pytorch/ao/issues/1076">torchao already works on raspberry pi · Issue #1076 · pytorch/ao</a>: Problem We don&#39;t publish aarch64 linux binaries so right now we still install ao=0.1 (myvenv) marksaroufim@rpi5:~/Dev/ao $ pip install torchao Looking in indexes: https://pypi.org/simple, https://...

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

frappuccino_o: Wanna meetup for morning coffee in San Francisco and discuss Image generation?
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1295615724475387938)** (9 messages🔥): 

> - `AMD MI250 donation`
> - `Discord server name change`
> - `User feedback for benchmarking`
> - `Compute funding for projects`
> - `Job submission interface concerns` 


- **AMD Donates MI250 Node**: The **generous team at AMD**, including key contributors, has donated an **MI250 node** to the server, with instructions for benchmarking [Triton kernels](https://github.com/gpu-mode/amd-cluster).
   - Members are encouraged to provide feedback on the user experience, aiming to improve it to be more user-friendly.
- **Discord Renamed to GPU Mode**: Questions arose about the **Discord server name change from CUDA Mode to GPU Mode**, suggesting a shift in focus towards AMD hardware.
   - Mark expressed openness to donations from other **hardware vendors** as well, expanding the collaborative opportunities.
- **Call for Contributors**: Mark is seeking **contributors** to help maintain the job submission pipeline, providing starter tasks in the README.
   - This aims to enhance the collaborative efforts within the community for better benchmarking experiences.
- **Compute Funding Opportunities**: Mark offered to forward interesting project needs for **compute funding** to potential sponsors, indicating a pathway for resource allocation.
   - Although not ideal, this could support members working on valuable compute-oriented projects.
- **Concerns About Job Submission Interface**: A user raised concerns regarding the reliance on **GitHub actions** for submitting kernel benchmarks and requested other access methods for longer jobs.
   - Mark acknowledged the need for **direct SSH access**, noting that scalability for a large community might pose challenges.


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1295619806237691914)** (3 messages): 

> - `Ollama Performance on Raspberry Pi`
> - `Building TorchAO`
> - `Triton CPU Backend Exploration` 


- **Ollama Runs Smoothly on Raspberry Pi 5**: The **Ollama model** achieved a workable evaluation rate of **5.32 tokens/s** for the **llama3.2** version, but the **1.5 tokens/s** for **llama3.1** on the Raspberry Pi 5 was noticeably slower.
   - A member mentioned considering using an **eGPU with a 2080**, now feasible with Raspberry Pi systems.
- **TorchAO Build Success on Raspberry Pi 5**: Surprisingly, a member managed to build **TorchAO from source**, achieving dynamic quantization in **int8** and compiling with **CPU codegen**.
   - They referenced a [GitHub issue](https://github.com/pytorch/ao/issues/1076) regarding the lack of published aarch64 Linux binaries, highlighting the community's workaround.
- **Exploring Triton CPU Backend and Custom Kernels**: The next exploration planned involves testing the **Triton CPU backend** for performance improvements.
   - The member also expressed interest in experimenting with **custom kernels** in TorchAO for low bit matrix multiplications.



**Link mentioned**: <a href="https://github.com/pytorch/ao/issues/1076">torchao already works on raspberry pi · Issue #1076 · pytorch/ao</a>: Problem We don&#39;t publish aarch64 linux binaries so right now we still install ao=0.1 (myvenv) marksaroufim@rpi5:~/Dev/ao $ pip install torchao Looking in indexes: https://pypi.org/simple, https://...

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1295757102521778287)** (3 messages): 

> - `WebGPU and CUDA interaction`
> - `WebGPU OS compatibility` 


- **WebGPU does not interact with CUDA**: A member inquired whether **WebGPU** can interact with **CUDA**, to which another member clarified that it **cannot**.
   - This indicates that developers using WebGPU will need to rely on other APIs.
- **WebGPU's dependency on OS graphics APIs**: It was noted that **WebGPU** utilizes specific graphics APIs based on the operating system, such as **Vulkan**, **DirectX**, or **Metal**.
   - This highlights the platform-specific nature of WebGPU's implementation.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1295734246823952437)** (2 messages): 

> - `Distributed Training of Deep Learning Models`
> - `FlashAttention 2022`
> - `Transformer Architecture`
> - `Communication Bottlenecks`
> - `Decentralized Training` 


- **Deep Dive into Distributed Training**: [Part 1](https://vaibhawvipul.github.io/2024/09/29/Distributed-Training-of-Deep-Learning-models-Part-~-1.html) of the blog series explores the foundational concepts of distributing computations in deep learning, explaining how dataflow graphs represent computation processes.
   - It includes insights on calculating a loss and updating model weights through the forward and backward passes.
- **Tackling Communication Bottlenecks**: [Part 2](https://vaibhawvipul.github.io/2024/10/03/Distributed-Training-of-Deep-Learning-models-Part-~-2.html) discusses the challenges of bandwidth and latency in distributed training, stressing their impact on the communication between parameter servers and workers.
   - This complexity often emerges when dealing with large models and numerous workers during parameter updates.
- **Exploring Decentralized Training Approaches**: [Part 3](https://vaibhawvipul.github.io/2024/10/15/Decentralized-Training-of-Deep-Learning-Models.html) shifts focus to decentralized training with non co-located compute, drawing on previous discussions of distributed training.
   - The post emphasizes the scalability of models based on findings from [Scaling laws for Neural Language Models](https://arxiv.org/pdf/2001.08361.pdf).
- **FlashAttention Makes a Splash**: A new blog post explains [FlashAttention (2022)](https://www.digitalocean.com/community/tutorials/flashattention), a technique enhancing the efficiency of the Transformer's attention mechanism in neural networks.
   - It aims to mitigate the O(n^2) time and memory complexities associated with long sequences, while future posts will discuss its later iterations.
- **Transformers and Their Impact**: The significance of the Transformer architecture is highlighted, especially its ability to utilize self-attention, as detailed in the seminal paper: [Attention is All You Need](https://arxiv.org/abs/1706.03762).
   - This architecture revolutionized AI research, focusing on enhanced efficiencies and applications across various domains.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.digitalocean.com/community/tutorials/flashattention">Designing Hardware-Aware Algorithms: FlashAttention | DigitalOcean</a>: no description found</li><li><a href="https://vaibhawvipul.github.io/2024/09/29/Distributed-Training-of-Deep-Learning-models-Part-~-1.html">Distributed Training of Deep Learning models - Part ~ 1</a>: Note: This post is part of a series on Distributed Training of Deep Learning models. Now, this is more like notes for me so you will find stuff copied directly from various places. Also, I use AI mode...</li><li><a href="https://vaibhawvipul.github.io/2024/10/03/Distributed-Training-of-Deep-Learning-models-Part-~-2.html">Distributed Training of Deep Learning models - Part ~ 2</a>: Note: I use AI models to help with writing. Now, this is more like notes for me so you will find stuff copied directly from various places. If you spot any mistakes, feel free to correct me!</li><li><a href="https://vaibhawvipul.github.io/2024/10/15/Decentralized-Training-of-Deep-Learning-Models.html">Decentralized Training of Deep Learning Models</a>: In the previous posts we talked about Distributed Training of DL models Part 1 and Part 2. Now, in this post we will dive deeper into decentralized training of deep learning models.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1295581003909435402)** (3 messages): 

> - `Text Inversion for SDXL`
> - `Training Challenges`
> - `Hugging Face Community` 


- **Troubles with Text Inversion on SDXL**: A member inquired about experiences training **Text Inversion** for **SDXL**, mentioning they tried various prompts and dropout settings without success.
   - *They expressed frustration over the lack of community models available on Civit.ai*, hinting at potential limitations of SDXL's architecture.
- **Suggesting Alternative Support Channels**: Another member advised seeking guidance in the Hugging Face server's **'diffusion models/discussion'** channel for better support.
   - They recommended assigning the **@diffusers** role to access this channel, indicating a potential for more targeted community help.


  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/1295788677900927058)** (1 messages): 

> - `Data transfer in tensor parallel operations`
> - `CPU cluster for LLM inference`
> - `AVX acceleration`
> - `Measuring bandwidth requirements` 


- **Measuring Data Transfer for Tensor Operations**: A user inquired about effective methods to measure the **data transfer** and **bandwidth** required for **tensor parallel operations** when setting up a CPU cluster.
   - *They mentioned several factors to consider*, such as matrix size, compute capability, memory bandwidth, and precision of computations.
- **Setting Up a CPU Cluster for LLM Inference**: The user's goal involves creating a CPU cluster for **LLM inference** with **AVX acceleration**, to test feasibility rather than practicality.
   - They are uncertain about the **network provisioning** needed for their setup and sought advice on estimating bandwidth.


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1295504494829899856)** (64 messages🔥🔥): 

> - `Real-Time Speech-To-Text Engines`
> - `Linear Attention Models`
> - `AI in Design`
> - `Recent Funding in AI Startups`
> - `Outsourcing Documentation for Open Source Projects` 


- **Real-Time STT engines revolutionize AI transcription**: Gladia's new Real-Time STT engine features < 300 ms latency, supporting over 100 languages and code-switching, marking a standard for real-time AI, backed by a $16M Series A funding.
   - Another engine touted 90ms inference time and supports multiple languages, indicating a competitive evolution in transcription technology.
- **Using linear attention models to enhance transform mechanisms**: Discussion around LoLCATS shows promise, linearizing Llama 3.1 model family, leading to significant improvements while being less resource-intensive than traditional methods.
   - The conversation also explored the challenges and potential issues faced when converting more than 50% of transformer model attention layers into linear layers.
- **Comparing AI's abstraction to design materials**: A blog post draws parallels between AI and plastics, suggesting AI could be viewed as a new building material due to its rapid integration into diverse domains.
   - The discussion highlights how past design ages transformed industries, leading to the importance of software and information over physical materials today.
- **Recent surge in funding for AI startups**: DecagonAI's announcement of a $65M Series B funding sparked curiosity about investment trends, particularly in AI application layers rather than underlying models.
   - Other mentions include notable fundraising efforts within the AI sector, reflecting a growing interest in AI-driven applications.
- **Outsourcing documentation for AI and open-source projects**: There’s ongoing discussion about the feasibility of outsourcing documentation and user guide writing for open-source projects, focusing on AI engineering.
   - Community members weigh the pros and cons of utilizing LLMs versus hiring individuals to produce comprehensive documentation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.twitch.tv/videos/2275103790">Twitch</a>: no description found</li><li><a href="https://x.com/simran_s_arora/status/1845909074774475125?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Tweet from Simran Arora (@simran_s_arora)</a>: Want Llama 405B, but wish it scaled linearly in sequence length??? Enter LoLCATS: an efficient method for &#34;turning Transformers to linear attention models&#34;, all on an academic budget!!   We us...</li><li><a href="https://www.notion.so/blog/ai-is-the-new-plastic">AI is the new plastic</a>: Plastic is in your car, your kitchen, the chair you’re sitting on right now. And just as our most ubiquitous material transformed the world in a matter of decades, AI is doing the same.</li><li><a href="https://x.com/thejessezhang/status/1846235369886589197?s=46">Tweet from Jesse Zhang (@thejessezhang)</a>: Beyond excited to announce Decagon&#39;s $65M Series B led by @aaref from Bain Capital Ventures, with @eladgil, A*, Accel, BOND Capital, and more. 🎉  This brings our total funding at @DecagonAI to $1...</li><li><a href="https://x.com/jilijeanlouis/status/1846145881285730338">Tweet from 🎙Jean-Louis Queguiner (@JiliJeanlouis)</a>: Our new Real-Time STT engine is out! 🔥It offers the best of both worlds: batch-level quality with real-time transcription speed.  With &lt; 300 ms latency, support in 100+ languages, code-switching a...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1g2vhy3/creating_very_highquality_transcripts_with/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://x.com/thesephist/status/1846029340867285158?s=46">Tweet from Linus (@thesephist)</a>: During my time at @NotionHQ, I got to think deeply about AI&#39;s future from the lens of design and architecture, rather than just technology.  I ended up writing a piece during my last few weeks abo...</li><li><a href="https://x.com/_mfelfel/status/1846025183993511965?s=46">Tweet from Felfel (@_mfelfel)</a>: We just released our latest model, the fastest text-to-speech model ever built. 90ms inference time. 120-150ms TTFB (Inference + Network). And we did that while making it 4x more reliable.  Quoting Pl...</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/3dd082b2406799ba8233b78b9f788c753486bafc/cmd/apps/catter/README.md">go-go-labs/cmd/apps/catter/README.md at 3dd082b2406799ba8233b78b9f788c753486bafc · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.</li><li><a href="https://share.snipd.com/episode/367fa309-f098-496f-874c-0387b1dff367">Duolingo CEO Luis Von Ahn wants you addicted to learning</a>: Duolingo CEO Luis Von Ahn wants you addicted to learning</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/3dd082b2406799ba8233b78b9f788c753486bafc/python/photo-dewarp/ttmp/2024-10-11/03-tps-dewarp.md">go-go-labs/python/photo-dewarp/ttmp/2024-10-11/03-tps-dewarp.md at 3dd082b2406799ba8233b78b9f788c753486bafc · go-go-golems/go-go-labs</a>: GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.</li><li><a href="https://github.com/wesen/glazed/blob/task/add-docs-for-commands/pkg/doc/tutorials/03-commands-tutorial.md">glazed/pkg/doc/tutorials/03-commands-tutorial.md at task/add-docs-for-commands · wesen/glazed</a>: a library to make it easy to output structured data in your command line tools. add the icing on top of your data - wesen/glazed</li><li><a href="https://github.com/wesen/glazed/blob/task/add-docs-for-commands/pkg/doc/topics/15-using-commands.md">glazed/pkg/doc/topics/15-using-commands.md at task/add-docs-for-commands · wesen/glazed</a>: a library to make it easy to output structured data in your command line tools. add the icing on top of your data - wesen/glazed</li><li><a href="https://github.com/go-go-golems/oak">GitHub - go-go-golems/oak: GO GO PARSE YOUR CODE GO GO</a>: GO GO PARSE YOUR CODE GO GO. Contribute to go-go-golems/oak development by creating an account on GitHub.</li><li><a href="https://github.com/go-go-golems/prompto">GitHub - go-go-golems/prompto: Quickly get custom prompt contexts</a>: Quickly get custom prompt contexts. Contribute to go-go-golems/prompto development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1295540850738335838)** (2 messages): 

> - `Financial Agent with Claude 3.5 Sonnet`
> - `Building AI Apps with MariaDB`
> - `SkySQL Integration`
> - `Smart Product Review Analysis` 


- **Build a Financial Agent with Claude 3.5**: Learn to create a **Financial Agent** powered by **Claude 3.5 Sonnet** using [@financial_mod’s APIs](https://twitter.com/llama_index/status/1845980793593831845) for retrieving stock prices and company data.
   - As outlined by Hanane Dupouy, this agent can provide diverse financial insights, including income statements and comprehensive company information.
- **Effective AI App Development via SkySQL**: If you're interested in MySQL/MariaDB for AI applications, [@skysql](https://twitter.com/llama_index/status/1846274668040540389) offers essential setup instructions for **MariaDB Vector** in SkySQL.
   - The guide includes integrating **OpenAI's LLM** with LlamaIndex and constructing a **smart product review analysis system**.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1295468905711534080)** (57 messages🔥🔥): 

> - `Qdrant Error on Adding Nodes`
> - `PineconeVectorStore Issues`
> - `Neo4jPropertyGraphStore Performance` 


- **Qdrant Node Addition Triggers Errors**: A member reported encountering an error when trying to add new nodes to the index in **Qdrant** without any prior instance of such an issue.
   - Another member responded by indicating they had not experienced that error, suggesting potential issues with setup.
- **PineconeVectorStore Failing in ComposableMemory**: A member expressed frustration while using **PineconeVectorStore** with `SimpleComposableMemory`, receiving a 'Namespace not found' error message.
   - Another user speculated there might be an issue with the Pinecone setup which could be causing the problem.
- **Performance Lag in Neo4jPropertyGraphStore Initialization**: A user discussed the significant delay experienced when creating the **Neo4jPropertyGraphStore** from an existing graph, stating the schema generation is incredibly slow.
   - They noted that they have allocated maximum memory and referenced a similar issue reported in a GitHub thread about the refresh_schema() function not handling large graphs well.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/16204">[Bug]: Extremely long time initializing Neo4jPropertyGraphStore for larger graphs · Issue #16204 · run-llama/llama_index</a>: Bug Description It takes about 14 min to initiate the graph store with 3558 entities. I feel this is because refresh_schema() does not handle large graphs well. Maybe not using async? I pasted the ...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/memory/composable_memory/">Simple Composable Memory - LlamaIndex</a>: no description found</li><li><a href="https://github.com/run-llama/llama_index/pull/16559">Fix pydantic errors in upstash_chat_store by kilimchoi · Pull Request #16559 · run-llama/llama_index</a>: Description Currently you get an error message AttributeError: &amp;#39;UpstashChatStore&amp;#39; object has no attribute &amp;#39;_sync_redis_client&amp;#39; Fixes # (issue) In pydantic v2, you have ...</li><li><a href="https://github.com/run-llama/llama_index/blob/e2dca8bb021b36b8eaf38be953cb2496f029d680/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L300">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py at e2dca8bb021b36b8eaf38be953cb2496f029d680 · run-llama/llama_index</a>: LlamaIndex is a data framework for your LLM applications - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/loading/documents_and_nodes/usage_nodes/#defining-and-customizing-nodes>):">Using Nodes - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/api_reference/storage/vector_store/postgres/#llama_index.vector_stores.postgres.PGVectorStore>).">Postgres - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1295516909227347978)** (1 messages): 

> - `Llama 3.1-70B integration`
> - `Truncated responses`
> - `Max tokens issue`
> - `Software engineering skills list`
> - `Usage statistics analysis` 


- **Llama 3.1-70B Integration Faces Truncation Trouble**: An integration of **Llama 3.1-70B** is hitting a wall with truncated responses despite adjusting parameters like **max_tokens**.
   - The application consistently returns only **5 skills** when asking for a **20 software engineering skills list**, always ending with `finish_reason: max_tokens`.
- **Max Tokens Mismanagement**: The user reported an issue where response completion is capped at **100 tokens**, regardless of the prompt complexity.
   - Adjustments to **max_tokens** and **completion_tokens** fail to resolve the issue, suggesting an underlying limitation.
- **Detailed Usage Statistics Confusion**: The reported usage statistics indicated **71 prompt tokens** alongside **100 completion tokens**, totaling **171 tokens**.
   - This raises questions about where the misconfiguration or restriction originates in the interaction with the model.


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1295527484342669332)** (46 messages🔥): 

> - `LLM Reasoning Limitations`
> - `Swarm Library Insights`
> - `GPT Voice Feature Queries`
> - `Controversial AI Practices`
> - `Self-Promotion Rules` 


- **LLMs display brittleness in reasoning**: A recent study from Apple engineers highlights the fragility of **LLMs** in mathematical reasoning, suggesting they utilize probabilistic pattern matching rather than true logical reasoning, leading to errors when benchmarks are slightly altered.
   - Members discussed the need for human baseline comparisons and pointed out the subjective definitions of reasoning within the study.
- **Insights on Swarm Library Functionality**: User discussions around the **Swarm** library revealed challenges in discerning whether agents or the base LLM were performing tasks, emphasizing the importance of writing effective tests.
   - Concerns were raised regarding Swarm's status as a non-production-ready tool, with mentions of existing forks and equivalents like **Swarm.js**.
- **Inquiries about GPT Voice Feature**: A user asked about the web rollout of the advanced **GPT voice** feature, with responses highlighting the lack of official announcements on its functionality.
   - Concerns were shared about the previous version's unsupported status, raising skepticism regarding future updates.
- **Controversial practices in AI funding**: Discussions touched on the dual paths for securing AI funding: overpromising with hype or leaning into skepticism, echoing sentiments about the quality of recently published research.
   - Members shared opinions on various researchers and the implications of sensational publications in AI.
- **Self-Promotion and Server Rules**: Enforcement of Discord server rules regarding self-promotion was addressed, reminding users to refrain from promotional messages to avoid potential moderation issues.
   - This led to reminders about community guidelines and encouraged knowledge sharing in appropriate channels.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-suggest/">Apple study exposes deep cracks in LLMs’ “reasoning” capabilities</a>: Irrelevant red herrings lead to &ldquo;catastrophic&rdquo; failure of logical inference.</li><li><a href="https://arstechnica.com/ai/2024/10/llms-cant-perform-genuine-logical-reasoning-apple-researchers-sug">Category: AI</a>: Open the pod doors&#8230;
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1295482202519506976)** (2 messages): 

> - `Custom GPT PDF Integration`
> - `Performance Issues with PDFs` 


- **Custom GPT stuck in 'Update Pendings'**: A member reported that their custom GPT, built with **300 pages** of building code materials, has been stuck in 'Update Pendings' for over a week despite splitting the PDFs into **6 smaller files** of **50-60 pages** each.
   - They noted that while the bot acknowledges the PDFs, it often redirects questions back to the code instead of providing direct answers from the documents.
- **Testing GPT with Single PDF**: Another member tested a new chat in GPT-4 with just **1 PDF**, but encountered similar performance issues, leading to speculation about the bot's ability to read the PDF effectively.
   - This suggests there may be underlying problems with how PDF content is processed by the GPT, affecting its responsiveness.


  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1295466333982425192)** (33 messages🔥): 

> - `Configuration Options`
> - `Program Installation Concerns`
> - `Character Cards in LLMs`
> - `Uncensored LLM Platforms`
> - `SageAttention Efficiency` 


- **Enhancing Configuration Options**: A member suggested providing configuration details in a format other than screenshots for ease of use.
   - Another member acknowledged the feedback, indicating that future blog posts will address this.
- **Installation without UAC Prompts**: An IT support engineer raised concerns over a worker's ability to install a program without User Account Control prompting for credentials.
   - There was a request for confirmation regarding whether the program installs files on the system machine or just the user profile.
- **Interest in Character Cards for LLMs**: A member expressed enthusiasm for incorporating 'character cards' into LM Studio for creating agents with distinct personalities and interactions.
   - They inquired about the possibility of creating conversations between LLMs to enhance functionality.
- **Demand for Uncensored LLM Solutions**: A member sought information about platforms similar to LM Playground that provide uncensored models.
   - Another member suggested checking for existing discussions on the topic to prevent redundancy.
- **Game-Changing Potential of SageAttention**: A member highlighted a recent paper on SageAttention, which demonstrates significant efficiency improvements in attention mechanisms.
   - They noted that if implemented in llama.cpp and MLX, it could potentially double the token processing speed, transforming performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.02367">SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration</a>: The transformer architecture predominates across various models. As the heart of the transformer, attention has a computational complexity of O(N^2), compared to O(N) for linear transformations. When ...</li><li><a href="https://github.com/openai/swarm">GitHub - openai/swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solution team.</a>: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solution team. - openai/swarm</li><li><a href="https://github.com/victorb/ollama-swarm">GitHub - victorb/ollama-swarm: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Modified to use local Ollama endpoint</a>: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Modified to use local Ollama endpoint - victorb/ollama-swarm</li><li><a href="https://github.com/ml-explore/mlx-examples/pull/1027">Clear cache during prompt processing by awni · Pull Request #1027 · ml-explore/mlx-examples</a>: Closes #1025, see that for discussion / improvement.
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1295476762213355581)** (13 messages🔥): 

> - `M2 Studio Performance`
> - `GPU Under-volting`
> - `Tokens Per Second (TPS) with Llama 8B`
> - `Performance Discrepancies between GPUs` 


- **M2 Studio shines with Mistral large model**: A user praised the **M2 Studio** with **192 GB RAM** and its performance with the **Mistral large 128K context**, stating it's an ideal model for their use case.
   - *It's such a good model for my use case* highlights the model's suitability for specific applications.
- **Under-volting GPUs for better performance**: A member suggested using **Afterburner** to **UV your GPU** as even a **100mV** adjustment can yield significant improvements in performance.
   - They recommended searching **YouTube** for the specific card plus keywords **UV** and **Afterburner** for helpful guides.
- **Llama 8B TPS discussed**: A participant observed that some users claim to achieve **30 TPS** on **Llama 8B** models with newer GPUs like the **4080**, while they run the same at **30 TPS** with a **1070Ti**.
   - They expressed a desire to reach **150+ TPS**, questioning what upgrades might be necessary.
- **Understanding GPU performance differences**: Discussion revealed that factors like model size, quantization, and context can severely affect performance, with differences noted between users' setups.
   - Specifically, one pointed out the necessity for **tensor cores** to match the performance of advanced GPUs like the **4080**, emphasizing configuration and model usage.
- **Running Llama 8B Q6_K with decent TPS**: Another user reported running **Llama 8B Q6_K** with a max context of **15k**, currently achieving a range of **28-35 TPS**.
   - They used around **10k tokens** at a time, indicating their setup’s efficiency.


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1295503281673601144)** (11 messages🔥): 

> - `Cohere Connector Usage`
> - `Cohere API Token Limits` 


- **Cohere Connector triggers search on 'hi'**: A user expressed frustration with the **Cohere Connector**, stating it performs a search even when simply saying 'hi' without the need for it.
   - They inquired if there is a way to control this feature and use the connector only when necessary.
- **Confusion over API token limits**: A user questioned the validity of token limits for the **Cohere API**, noting a discrepancy between **10k tokens** per month and what was stated as **5 million tokens** by the Cohere chat.
   - They asked for confirmation if exceeding **10k tokens** would result in billing.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1295548996470833164)** (8 messages🔥): 

> - `Google Connector Issues`
> - `Command Model Pricing`
> - `Collaboration on C4AI Projects`
> - `Date Injection in LLMs`
> - `Reranking Process for Newsletters` 


- **Google Connector Troubleshooting**: Several members reported having issues with a **Google Connector** that fails to work properly and are seeking solutions from each other.
   - One member suggested to share breakthroughs as they continue to troubleshoot the connectivity issue.
- **Understanding Command Model Pricing Structure**: Discussion revealed that there is no cost associated with using the **web-search connector**, but the results passed to the **Command** input context are subject to charges.
   - This clarification helped to highlight where potential costs might arise in the process.
- **Collaborative Opportunities in C4AI Discord**: A member encouraged others to check out the **C4AI Discord** for insights on projects that are seeking collaborators, providing a link to join.
   - This opens avenues for collaboration among various projects within the community.
- **Optimizing Date Usage in LLMs**: A member discussed the potential of using absolute dates in the **final** call despite already implementing date injection for their processes.
   - They suggested that for relative dates, the implementation of metadata filtering with tool use might enhance the results.
- **Newsletters Reranking Workflow**: A structured process for handling newsletters was outlined involving fetching, reranking using **Cohere Rerank**, and retaining top results for LLM context.
   - This method feature includes **30 newsletters** being analyzed to filter down to the **top 10** based on a new ranking system.



**Link mentioned**: <a href="https://cohere.com/blog/rerank-3#searching-over-json">Introducing Rerank 3: A New Foundation Model for Efficient Enterprise Search &amp; Retrieval</a>: Today, we&#x27;re introducing our newest foundation model, Rerank 3, purpose built to enhance enterprise search and Retrieval Augmented Generation (RAG) systems.   Our model is compatible with any dat...

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1295648961448771595)** (13 messages🔥): 

> - `Cohere Tool Calls`
> - `Curl Example for Tool Use`
> - `Concerns about Parameter Definitions`
> - `Request Handling in API`
> - `Raw Request Examples` 


- **Cohere Tool Calls streamline workflows**: A discussion highlighted how, after a tool call, an assistant's message should prompt either more tool calls or provide a response.
   - Members appreciated the **function calling** feature which enhances interaction with external tools.
- **Curl Example Provided**: A member shared a concise [curl example](https://docs.cohere.com/docs/tool-use#the-four-steps-of-tool-use-step-by-step-example) to demonstrate the proper usage of the API for current weather retrieval.
   - This has been positively received as it offers a practical solution for users not working with Python.
- **Clarifications on Parameter Issues**: Members sorted out confusion regarding the parameter definitions in an API snippet, emphasizing the use of v1-style format.
   - A key fix suggested was changing the first message's role from 'tool' to 'assistant', which resolved the issue.
- **Demand for Raw Request Examples**: There was a call for providing raw request examples alongside library examples to facilitate user understanding.
   - One member shared a [Medium article](https://medium.com/@smallufo/anthropic-claude-function-call-example-7355e6ec6fa2) that exemplifies this need.
- **Function Calling Capabilities Highlighted**: Members expressed excitement about **Cohere's ability** to manage multiple function calls efficiently.
   - This capability is seen as significantly beneficial for enhancing application workflows.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@smallufo/anthropic-claude-function-call-example-7355e6ec6fa2">Anthropic Claude Function Call Example</a>: Request</li><li><a href="https://docs.cohere.com/docs/tool-use#the-four-steps-of-tool-use-step-by-step-example">Tool Use — Cohere</a>: Enable your large language models to connect with external tools for more advanced and dynamic interactions (V2).
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1295800987965063169)** (2 messages): 

> - `OrionChat`
> - `Chat AI Model Comparison` 


- **Launch of OrionChat for Multi-Model Conversations**: A member introduced a personal project called **OrionChat**, a web interface that consolidates various chat AI models from **Cohere**, **OpenAI**, **Google**, and others into a single platform for easy access.
   - They encouraged the community to explore the interface at [this link](https://orionchat.github.io) and provide feedback to enhance user experience.
- **Exploring Chat AI Capabilities in One Place**: The new interface enables users to chat and compare various AI models without switching between multiple tabs or websites, simplifying exploration.
   - The developer expressed excitement for the community's input, aiming to refine the project based on user interactions.



**Link mentioned**: <a href="https://orionchat.github.io">OrionChat - Chat interface for AI models.</a>: OrionChat - Chat interface for IA models

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1295467206364364900)** (34 messages🔥): 

> - `WordPress Plugin Development`
> - `CORS Issues with Stable Diffusion`
> - `AI Creativity Community Engagement`
> - `Text Generation Models for Style Transfer`
> - `Discord Server Activity` 


- **WordPress Plugin Development for Stable Diffusion**: A member is developing multiple WordPress plugins for text generation and txt2img servers, seeking feedback and testing from the community.
   - *Nobody is responding*, leading to frustrations with community engagement in AI Discord servers.
- **CORS Issues on Stable Diffusion Setup**: Users discussed challenges of using SSL with Stable Diffusion servers running on a reverse proxy setup, experiencing CORS errors.
   - One member stressed the integration between the webserver and Stable Diffusion server on the same machine for functionality.
- **Finding Active AI Communities**: A member expressed disappointment with the lack of activity in their current AI Discord server, asking for suggestions for more active communities related to comfyUI and A1111.
   - They noted that inquiries about plugins went unanswered, indicating a need for better engagement elsewhere.
- **Exploring Base Models for Text Generation**: A user inquired about base models that perform better in text generation during style transfer tasks, mentioning their experience with i2i and SD1.5.
   - Another member suggested trying **flux** or **SD3** for improved text generation quality, but noted SD3 struggles with human representation.
- **Techniques for Stylized Photos**: Discussion arose about creating stylized photos, with suggestions including using **ControlNets** and specific methods like those described [here](https://github.com/songrise/Artist).
   - Members shared techniques for achieving various artistic styles, such as pin-up, emphasizing creative approaches.



**Link mentioned**: <a href="https://WandAI.app,">no title found</a>: no description found

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1295652057167954032)** (14 messages🔥): 

> - `Tinygrad .dot operations comparison`
> - `VIZ UI improvements`
> - `Performance against PyTorch`
> - `Pre-ordering Pro device` 


- **Tinygrad .dot operations vs NumPy**: A user compared Tinygrad's .dot ops with NumPy's matmul and found that **accuracy decreases** for larger matrix sizes, noting differences reaching **±0.001** for large sizes like M=16384, N=8192, K=1280.
   - For smaller matrices (M=10, N=4, K=5), the differences were minimal, not exceeding **±0.000001**.
- **VIZ UI improvements discussion**: A member shared a link to [Issue #7067](https://github.com/tinygrad/tinygrad/issues/7067) concerning improvements for the VIZ UI, including autoscrolling functionality.
   - The issue discusses making improvements to sidebar navigation while ensuring that both left and right sidebars are resizable and collapsible.
- **George Hotz on performance against PyTorch**: George emphasized that if Tinygrad can beat **PyTorch's performance** on NVIDIA GPUs, it would be a significant win for the project.
   - He mentioned that achieving performance parity would unlock a lot of potential, asserting that 'all we have to do is beat PyTorch in perf and we win.'
- **Curiosity around Pro device shipping dates**: Users expressed interest in pre-ordering the **Pro device** and inquired about its shipping date in December.
   - One member specifically asked what the shipping date is, searching for more details before making a decision.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_xjdr/status/1846210719421026645">Tweet from xjdr (@_xjdr)</a>: the performance of jax + GPU is abysmal compared with an equivalent pytorch implementation. This is sad but unsurprising.  jax might be relegated to TPU only (for me) for a little while longer.  that ...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/7067">VIZ UI improvements · Issue #7067 · tinygrad/tinygrad</a>: Autoscroll the left side kernels list when using arrow down. Example VIZ=1 python3 -m pytest test/test_schedule.py sc.mov Both the left and right sidebar are resizable and collapsible, currently on...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1295501552240562177)** (16 messages🔥): 

> - `JIT and Tensor Reshape`
> - `Multinomial Sampling Without Replacement`
> - `TD-MPC Implementation in Tinygrad`
> - `Disabling Gradient Calculations`
> - `Adding New Accelerators in Tinygrad` 


- **JIT struggles with Tensor Reshape**: It's questioned whether `.reshape` should be JIT'ed, noting that JIT runs only GPU kernels and Python code is not executed.
   - For dynamic shapes, a specific implementation link was shared, highlighting the need to adjust code structure for compatibility.
- **Multinomial Sampling Error Explained**: A user inquired about an error related to the multinomial function when sampling without replacement, causing confusion around the replacement argument.
   - The code snippet's assertion revealed that no replacement only allows for single samples, thus clarifying the intended functionality.
- **Successful TD-MPC Learning in Tinygrad**: A user reported the implementation of TD-MPC learning in Tinygrad, expressing excitement about testing it on hardware.
   - Links to the relevant GitHub repository were provided, emphasizing the hardware requirements for execution.
- **Methods to Disable Gradient Calculations**: Discussion about disabling gradient calculations suggests that setting `Tensor.no_grad` to True is still used in practice.
   - Alternatives were introduced using `with Tensor.test():` as a modern and possibly preferred way for controlling gradient computation.
- **Guidance on Adding Accelerators**: A user provided guidance on adding new accelerators in Tinygrad, referencing a linked resource for further reading.
   - The post aimed to clarify the integration of operations and the underlying architecture necessary for expansion with new hardware support.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/addingaccelerator.html">How to add a custom accelerator?</a>: Tutorials on tinygrad</li><li><a href="https://github.com/nicklashansen/tdmpc2/tree/main">GitHub - nicklashansen/tdmpc2: Code for &quot;TD-MPC2: Scalable, Robust World Models for Continuous Control&quot;</a>: Code for &quot;TD-MPC2: Scalable, Robust World Models for Continuous Control&quot; - nicklashansen/tdmpc2</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/examples/whisper.py#L107>">tinygrad/examples/whisper.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/examples/whisper.py#L36-L45>">tinygrad/examples/whisper.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1295507378069962853)** (28 messages🔥): 

> - `Mojo Library Installation`
> - `Testing Custom stdlib`
> - `Image Hashing Algorithms`
> - `Memory Management in Mojo` 


- **Resolving Library Installation Issues**: A user discovered the missing library could be installed with `sudo apt-get install libtinfo-dev`, which might help others facing the same issue.
   - This highlights the importance of sharing solutions in the community to aid those experiencing similar challenges.
- **Testing Custom stdlib in Mojo**: After following build instructions, a user experienced issues running a modified version of stdlib, with original implementations still appearing.
   - Another user suggested a workaround involving the build process, enabling continued progress while addressing these issues.
- **Exploring Updated Image Hashing Algorithms**: A user questioned the relevance of older image hashing algorithms like pHash, seeking recommendations for newer methods.
   - They expressed interest in more advanced options as technology progresses, prompted by the feeling that existing algorithms may be outdated.
- **Understanding Memory Management in Mojo**: A discussion emerged about eager destruction in Mojo where an instance of a struct was destructed prematurely during an assertion call.
   - Suggestions were made to implement a getter for the struct member to safely access data without early destruction, improving memory handling.
- **Successful Collaboration on Bug Reports**: A user raised an issue about string interpolation in Mojo, which was later confirmed to be resolved in the newest version.
   - The collaborative effort highlighted the value of community support in addressing and fixing bugs efficiently.



**Link mentioned**: <a href="https://github.com/modularml/mojo/issues/3672">[BUG] Module Name Collision: Function Cannot Be Called When Named After Module · Issue #3672 · modularml/mojo</a>: Bug description I am experiencing an issue in the Mojo standard library where functions that share their name with their respective modules cannot be called. Specifically, after implementing the ti...

  

---



### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1295465501899751485)** (20 messages🔥): 

> - `Sebastien Bubeck moving to OpenAI`
> - `o1-turbo-mini benchmarks`
> - `AGI paper discussions`
> - `OpenAI's influence on lawyers` 


- **Sebastien Bubeck's Departure to OpenAI**: One of Microsoft's top AI researchers, **Sebastien Bubeck**, is moving to OpenAI, stirring discussions about talent movements in AI.
   - The news was detailed in an article from [The Information](https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx).
- **o1-turbo-mini shows impressive results**: There's buzz about **o1-turbo-mini** performing suspiciously well on benchmarks, igniting both skepticism and humor in the community.
   - Members agreed that it could be amusing to poke fun at the overly online crowd with this emerging news.
- **Debate over Bubeck's AGI Paper**: The community expressed mixed feelings about Bubeck's '**sparks of AGI**' paper, with some believing it did more harm than good.
   - The discussions hinted at the **hyperbolic positioning** within the paper and concerns regarding its impact on AGI definitions.
- **OpenAI's role is beneficial for legal professionals**: A member emphasized that OpenAI is creating favorable conditions for **lawyers**, drawing a link between AI advancements and legal jobs.
   - This remark highlights the evolving relationship between AI technology and its practical applications in the legal field.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/features/2024-10-14/why-openai-is-at-war-with-a-guy-named-guy?accessToken=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzb3VyY2UiOiJTdWJzY3JpYmVyR2lmdGVkQXJ0aWNsZSIsImlhdCI6MTcyODkxMTEyNSwiZXhwIjoxNzI5NTE1OTI1LCJhcnRpY2xlSWQiOiJTTEM5MFVEV1gyUFMwMCIsImJjb25uZWN0SWQiOiJBQjc4QTNBMDc1N0U0OTI0ODFCRUU5RDRCRjBDNERDNSJ9.mpWXrXYbOWqoPDB5c-FKwDK6V_Q9UyyhU5kIPKkwhDc">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.bloomberg.com/news/features/2024-10-14/why-openai-is-at-war-with-a-guy-named-guy?accessT">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://x.com/amir/status/1845905601110462481">Tweet from Amir Efrati (@amir)</a>: news: one of Microsoft&#39;s top AI researchers, @SebastienBubeck, is moving to OpenAI.  https://www.theinformation.com/briefings/microsoft-ai-researcher-sebastien-bubeck-to-join-openai?rc=c48ukx
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1295825683578097816)** (5 messages): 

> - `Doomsday Clock for AGI`
> - `AI2 OLMo Internship` 


- **Doomsday Clock for AGI Raises Eyebrows**: A Saudi-backed business school in Switzerland launched a Doomsday Clock aimed at warning about the dangers of 'uncontrolled artificial general intelligence,' referred to as 'god-like' AI. The clock's creator, Michael Wade, criticized the metaphor as outdated and lacking relevance in today's context.
   - He emphasizes the absurdity of equating software like Excel to the threat of a deity-like AI, connecting it to historical fears sparked by atomic weaponry.
- **AI2 Offers Research Internship in Seattle**: Ai2 is seeking Research Interns for their OLMo project, focusing on advancing natural language processing and machine learning. This 12-week internship provides the chance to lead impactful research projects and collaborate with experts in the field.
   - Compensation ranges from **$86,520** to **$123,600**, depending on the degree attained, and the position requires on-site work in Seattle.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gizmodo.com/for-the-love-of-god-stop-making-inscrutable-doomsday-clocks-2000512111">For the Love of God, Stop Making Inscrutable Doomsday Clocks</a>: A business school is using AI doomerism, money from Saudi Arabia, and a dusty Cold War metaphor to get people hyped about AI’s future.</li><li><a href="https://job-boards.greenhouse.io/thealleninstitute/jobs/6322728">Job Application for Research Internship, OLMo at The Allen Institute for AI</a>: no description found</li><li><a href="https://x.com/mattshumer_/status/1846209244284219703">Tweet from Matt Shumer (@mattshumer_)</a>: http://x.com/i/article/1846205240728588288
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

victory: someone took this too far https://www.myforevernotes.com/articles/overview
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1295486735802695873)** (11 messages🔥): 

> - `Framework Selection Challenges`
> - `Langgraph Deployment`
> - `Comparing dspy and Langchain`
> - `Shifting Frameworks`
> - `Langsmith for Tracing` 


- **Framework selection is a nightmare!**: Members expressed frustration about the constant shifting among frameworks like **Langchain**, **Langflow**, and **Langgraph**, making it difficult to finalize a choice for production.
   - One noted that their entire codebase has transitioned to *Langchain LCEL*, highlighting the chaos surrounding these frameworks.
- **Langgraph deployment on private cloud**: A member inquired about deploying a **Langgraph** application on their own cloud outside of the **US** or **EU**, seeking insights from the community.
   - There was no direct response, but the question highlighted the growing interest in hosting applications regionally.
- **Debate on dspy versus Langchain**: Interest arose around whether **dspy** would dominate over **Langchain** and other frameworks or if frameworks would maintain their relevance.
   - This reflects the community's uncertainty about the future landscape of AI frameworks.
- **Acknowledgment of Langsmith's utility**: One member suggested that **Langsmith** is beneficial for tracing, implying its importance among the shifting frameworks.
   - This led to recommendations for using resources like the **Langchain Academy** course on **Langgraph** to sharpen skills.
- **Clarification on Langflow's affiliation**: A user clarified that **LangFlow** is not one of **LangChain's** products, indicating confusion in the community about related tools.
   - This distinction may help members align their understanding of the various frameworks being discussed.


  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1295461430379679775)** (2 messages): 

> - `Course Information`
> - `MOOC Registration`
> - `Discord Community` 


- **All Course Details Available Online**: All the details on **labs** and **assignments** can be found on the course website at [course website](https://llmagents-learning.org/f24).
   - Participants are encouraged to regularly check the site for updates and relevant materials.
- **Easy MOOC Sign-Up**: To join the course, prospective students should fill in this convenient [form](https://forms.gle/svSoNhKcGFjxup989).
   - This step is essential for anyone looking to actively participate in the course activities.
- **Join the LLM Agents Community on Discord**: For ongoing discussion and questions, participants should join the **LLM Agents Discord** community at [Discord link](https://discord.gg/NWVpQ9rBvd).
   - This platform allows for real-time communication and support among course members.
- **Acknowledgment of Assistance**: A member expressed appreciation towards another for providing vital information, which they initially overlooked. *'It was right there, still couldn't see it'*, they remarked humorously.
   - This highlights the collaborative spirit within the course community.



**Link mentioned**: <a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>: no description found

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1295507315956383764)** (6 messages): 

> - `Test-time compute scaling law`
> - `Devin vs PearAI opinions`
> - `LLM reasoning and planning`
> - `Tool calling in LLMs`
> - `Lecture quality improvements` 


- **Test-time compute scaling law observed**: A member discussed the broader downstream effect of the observed **'test-time compute' scaling law**, which is comparable to the original laws leading to the GPT family.
   - For reference, they shared [the paper](https://arxiv.org/pdf/2408.03314) and another important [document](https://arxiv.org/pdf/2001.08361).
- **Prof Neubig's take on Devin vs PearAI**: There was an inquiry about **Prof Neubig's** opinion regarding **Devin** and the **YC-backed PearAI**.
   - *Tilman suggested that this might belong in another channel*, highlighting ongoing relevant discussions.
- **Exploring LLM reasoning and planning**: A member sought insights on how LLMs and agents handle **reasoning**, **planning**, and **acting** beyond just text generation.
   - They requested information on the process of LLMs identifying appropriate tools and their approach toward **NER** for extracting relevant entities.
- **Plans for upcoming lectures**: A member asked if there were any plans to cover **planning** and **tool use** in future lectures.
   - This indicates the community's interest in a deeper understanding of practical applications in the context of LLMs and agents.
- **Lecture video quality concerns**: One member expressed the need for better **video quality** in lecture uploads, mentioning that **720p** is the best available on YouTube for lecture 6.
   - They pointed out that the current resolution makes it difficult to read the code properly during the lectures.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1295481309501722634)** (1 messages): 

> - `AI-Powered Search Resource` 


- **AI-Powered Search book emerges as essential**: A member recommended [this book](https://www.manning.com/books/ai-powered-search) as the probable go-to resource for the next few years in the domain of AI-powered search technologies.
   - They expressed confidence that the book will significantly influence practitioners and researchers alike.
- **Expectation of Industry Impact**: The member indicated a strong belief that the insights from the book will shape the future of AI-driven search functionalities across various industries.
   - They highlighted its potential to become foundational in the curriculum of AI studies.


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1295634566459166740)** (2 messages): 

> - `Open Interpreter Update`
> - `New Python Release` 


- **Open Interpreter gets a π release**: A member announced the release of a new version of **Open Interpreter** with the command `pip install --upgrade open-interpreter` shared via [this post](https://x.com/MikeBirdTech/status/1846283357153268002).
   - This update is referred to as a **π release**, indicating a significant improvement in features and performance.
- **General Morning Greetings**: A member greeted everyone with a simple **Goodmorning** in the chat to start the day on a positive note.
   - This moment reflects the community's welcoming environment and willingness to engage with one another.



**Link mentioned**: <a href="https://x.com/MikeBirdTech/status/1846283357153268002">Tweet from Mike Bird (@MikeBirdTech)</a>: pip install --upgrade open-interpreter  A π release!

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1295742137509609573)** (2 messages): 

> - `Hume AI model`
> - `Oi model` 


- **Hume AI Model Works Wonders**: A member reported that their experience with the **Hume AI model** was unexpectedly positive, stating it works almost **too well**.
   - This raises interesting questions about the potential and limitations of AI models in real-world applications.
- **Switch from Hume AI to Oi**: The same member acknowledged a shift in focus, mentioning **Oi** instead of the Hume AI model.
   - This suggests an ongoing experimentation with different AI models to assess their effectiveness.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1295779101642326110)** (3 messages): 

> - `Play 3.0 mini`
> - `Think-on-Graph` 


- **Play 3.0 mini launches faster and more accurate**: [Play.ht](https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) introduced their latest Text-To-Speech model, **Play 3.0 mini**, which boasts improved speed and accuracy, supports multiple languages, and is **cost-efficient**.
   - They invited users to try it out on their [playground](https://play.ht/playground/?utm_source=x&utm_medium=social&utm_campaign=all_v3launch_202410) and share their thoughts.
- **Explore Think-on-Graph on GitHub**: The [Think-on-Graph GitHub repository](https://github.com/IDEA-FinAI/ToG) by IDEA-FinAI is live, inviting researchers to join their team in Shenzhen or express interest in their work.
   - It includes a detailed contact offer for collaboration via email for those interested.
- **Video Resource on Recent Developments**: A member shared a [YouTube video](https://www.youtube.com/watch?v=iGQLG0bWDxE) that discusses recent advancements presumably related to AI content.
   - Details about the video content were not specified, inviting viewers to check it out for themselves.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/play_ht/status/1845901523680686401?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from PlayAI (formerly PlayHT) (@play_ht)</a>: Today we’re introducing our latest Text-To-Speech model, Play 3.0 mini.  It’s faster, more accurate, handles multiple languages, supports streaming from LLMs, and it’s more cost-efficient than ever be...</li><li><a href="https://github.com/IDEA-FinAI/ToG">GitHub - IDEA-FinAI/ToG: This is the official github repo of Think-on-Graph. If you are interested in our work or willing to join our research team in Shenzhen, please feel free to contact us by email (xuchengjin@idea.edu.cn)</a>: This is the official github repo of Think-on-Graph. If you are interested in our work or willing to join our research team in Shenzhen, please feel free to contact us by email (xuchengjin@idea.edu....
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1295825107213750346)** (1 messages): 

> - `Loom Video Insights` 


- **Loom Video Shared**: A member shared a [Loom video](https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-f408a83cecf5) that likely contains insights or discussions on a relevant topic.
   - Details from the video were not provided, leaving the community to explore its content for valuable information.
- **Exploration of Video Content**: Members expressed interest in what insights could be gathered from the shared Loom video, considering its relevance to ongoing discussions.
   - The lack of an initial description prompted curiosity, with members likely planning to review it for further discussions.



**Link mentioned**: <a href="https://www.loom.com/share/b8c49e265d5c49aca7d2fc51c38d84c6?sid=69c5942b-ed75-4883-8867-f408a83cecf5">Exploring Quantum Architecting Principles 🌌</a>: https://github.com/seanchatmangpt/dslmodel  In this video, I delve into the realm of quantum architecting principles and their application in revolutionizing enterprise software. From discussing AI-dr...

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1295567564587667541)** (5 messages): 

> - `Contextual embeddings`
> - `RAG (Retrieval-Augmented Generation) mechanics`
> - `DSPy integration into GPT-O1+` 


- **Contextual Embeddings Resources Shared**: A member provided a [Google Colab](https://tinyurl.com/2p9wwypy) and a [YouTube video](https://www.youtube.com/watch?v=6efwN_US-zk&t=7s) titled 'Contextual Retrieval with Any LLM: A Step-by-Step Guide' focusing on contextual embeddings.
   - The video aims to help implement contextual retrieval strategies from Anthropic for any LLM.
- **Clarifying RAG and Token Limits**: A member expressed confusion about adding whole documents to prompts exceeding token limits, highlighting the chunking process used in RAG.
   - It was clarified that RAG uses similarity search to only include the most relevant chunks in the prompt, thus maintaining token limits.
- **DSPy Integration Progress Query**: One member inquired about the progress of incorporating DSPy into the system, referred to as GPT-O1+.
   - Details of the integration were not provided in the exchanges.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tinyurl.com/2p9wwypy">Google Colab</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=6efwN_US-zk&t=7s">Contextual Retrieval with Any LLM: A Step-by-Step Guide</a>: In this video, I show you how to implement contextual retrieval for any LLM using a strategy from Anthropic. We&#39;ll go through the entire process, from chunk ...
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1295748356320133264)** (4 messages): 

> - `ICLR Reviews`
> - `Continuous Pre-training in LLMs`
> - `Model Merging Discussions` 


- **ICLR Reviews are finally out!**: The long-awaited review papers for ICLR have been released, prompting excitement among members eager to dive in.
   - *One member noted* it will take time to process their assigned review.
- **Study on Continuous Pre-training and Instruction Fine-tuning**: A recent paper investigates the relationship between **continuous pre-training** and **instruction fine-tuning** for Large Language Models, emphasizing the need for models to stay updated with the latest data.
   - It raises the question of which model should undergo this pre-training for maintaining instruction-following abilities.
- **Model Merging Approach Critique**: *A member questioned* the novelty of the approach in the paper, suggesting it resembles long-established methods of model merging.
   - This sparked a discussion about the relevance and originality of the proposed techniques.



**Link mentioned**: <a href="https://arxiv.org/html/2410.10739v1">Balancing Continuous Pre-Training and Instruction Fine-Tuning: Optimizing Instruction-Following in LLMs</a>: no description found

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1295511523556593754)** (4 messages): 

> - `LAION-2B dataset`
> - `Data Overlap with MSCOCO` 


- **Inquiry on LAION-2B Dataset and MSCOCO Overlap**: A member inquired about whether the **LAION-2B dataset** contains images from the **MSCOCO** (COCO2014 or COCO2017) datasets, questioning the potential **data overlap**.
   - They noted that the paper mentions a section on **data overlap**, yet they are looking for further details on the techniques used to check for this issue.
- **Good Morning and General Greetings**: Members exchanged general greetings with a member stating, **'Good morning everyone.'** to foster a friendly environment in the chat.
   - Following this, another member responded with a casual acknowledgment, **'gm'**.


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1295517193642967124)** (3 messages): 

> - `Inference Pipeline Details`
> - `Model Function Call Outputs`
> - `Multi-turn Interaction Logic` 


- **Understanding Inference Pipeline Mechanics**: The inference pipeline allows the model to execute functions as long as it continues to output valid function calls that can be decoded by the `decod_exec` method.
   - If the model outputs an empty string or an un-decodable response, it signifies the end of the current turn.
- **Model's Ability to Properly Stop**: A member highlighted the significance of the model’s ability to correctly determine when to stop outputting function calls, particularly when the model believes it has completed a task.
   - *In some sense, yes, but not necessarily,* was the response, indicating that the model can output nothing to signal the end of a turn.
- **Example of Weather Inquiry Interaction**: A detailed scenario was given where the model processes a weather-related request using function calls like `get_coordinate` and `get_weather`, returning necessary data at each step.
   - The conversation demonstrates that the model's output of a sentence post-data retrieval cannot be decoded and thus results in the termination of that interaction turn.
- **Function Call Output Variability**: The model can express the need to stop or make additional function calls in various ways, including choosing to output nothing at all.
   - This variability reflects different approaches taken by prompting models to manage user queries effectively.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

cyberg0285: Thank you <@709013886644518982>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1295786080221532252)** (1 messages): 

> - `Forwarding Protocols` 


- **Discussion on Forwarding Protocols**: A member referred to an important link related to **forwarding protocols** and shared their thoughts in the channel.
   - *Here is the forwarded message for reference.*
- **Information Sharing Dynamics**: Another member highlighted the significance of proper **information sharing** practices to enhance community engagement.
   - They emphasized that *forwarding messages can facilitate quicker responses and clearer communication.*


  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1295823752277528668)** (1 messages): 

> - `AI Stewardship Practice Program`
> - `Tech Stewardship`
> - `Microcredential Opportunities` 


- **Pilot Program for AI Stewardship**: The **AI Stewardship Practice Program** by the MaRS Discovery District is offering free slots for a pilot course aimed at shaping the evolution of AI positively.
   - This microcredential program targets researchers, educators, and policymakers among others, with more details available on the [Tech Stewardship website](https://programs.techstewardship.com/).
- **Opportunity to Become a Tech Steward**: Participants can engage with offerings designed to launch and maintain their AI stewardship practice, promoting the motto to **bend the arc of technology towards good**.
   - Interested individuals are encouraged to [reply in thread here](https://discord.com/channels/1089876418936180786/1295822228406931529) for a chance to be part of the pilot course valued at **500 CAD**.



**Link mentioned**: <a href="https://programs.techstewardship.com/">Tech Stewardship Practice</a>: no description found

  

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
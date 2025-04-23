---
id: 6da8ca62-994d-48b0-ab35-e69b703e4450
title: '[AINews] Mistral Large 2 + RIP Mistral 7B, 8x7B, 8x22B'
date: '2024-07-24T23:44:31.500890Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-mistral-large-2
---

<!-- buttondown-editor-mode: plaintext -->**A Mistral Commercial License is what you'll need.**

> AI News for 7/23/2024-7/24/2024. We checked 7 subreddits, [**384** Twitters](https://twitter.com/i/lists/1585430245762441216) and **30** Discords (**474** channels, and **4118** messages) for you. Estimated reading time saved (at 200wpm): **428 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

It is instructive to consider the focuses of [Mistral Large in Feb 2024](https://mistral.ai/news/mistral-large/?ref=upstract.com) vs [today's Mistral Large 2](https://mistral.ai/news/mistral-large-2407/):

- Large 1: big focus on MMLU 81% between Claude 2 (79%) and GPT4 (86.4%), **API-only**, no parameter count
- Large 2: one small paragraph on MMLU 84% (still not better than GPT4!), 123B param **Open Weights** under a Research License, "sets a new point on the performance/cost Pareto front of open models" but new focus is on codegen & math performance using the ["convex hull" chart made popular by Mixtral 8x22](https://buttondown.email/ainews/archive/ainews-deepseek-v2-beats-mixtral-8x22b/)  ![image.png](https://assets.buttondown.email/images/bdb9414a-a9a6-455e-8e8d-ad1cc1daf2f6.png?w=960&fit=max) 
- Both have decent focus on Multilingual MMLU
- Large 1: 32k context
- Large 2: **128k context**
- Large 1: only passing mention of codegen
- Large 2: "Following our experience with Codestral 22B and Codestral Mamba, we trained Mistral Large 2 on a very large proportion of code." ![image.png](https://assets.buttondown.email/images/842708ad-cafb-4fa2-8a71-f392c69214e1.png?w=960&fit=max) 
- Large 1: "It is natively capable of function calling" and "JSON format"
- Large 2: "Sike actually our Function calling wasn't that good in v1 but we're better than GPT4o now"  ![image.png](https://assets.buttondown.email/images/b7986e07-a81f-4098-9057-8da4c5c75bae.png?w=960&fit=max) 
- Large 2: "A significant effort was also devoted to enhancing the model’s reasoning capabilities."
- Llama 3.1: <<90 pages of [extreme detail on how sythetic data was used to improve reasoning and math](https://www.latent.space/p/llama-3)>>

Mistral's la Plateforme is deprecating all its Apache open source models (Mistral 7B, Mixtral 8x7B and 8x22B, Codestral Mamba, Mathstral) and only Large 2 and last week's 12B [Mistral Nemo](https://mistral.ai/news/mistral-nemo/) remain for its generalist models. This deprecation was fully predicted by [the cost-elo normalized frontier chart](https://x.com/swyx/status/1815892458519289946/photo/1) we discussed at the end of yesterday's post.


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

> temporary outage today. back tomorrow.

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Llama 3.1 Release and Capabilities**

- **Meta Officially Releases Llama-3-405B, Llama-3.1-70B & Llama-3.1-8B** ([Score: 910, Comments: 373](https://reddit.com//r/LocalLLaMA/comments/1ea9eeo/meta_officially_releases_llama3405b_llama3170b/)): **Meta** has officially released new versions of their **Llama language models**, including **Llama-3-405B**, **Llama-3.1-70B**, and **Llama-3.1-8B**. The models are available for download from the [Llama website](https://llama.meta.com/llama-downloads/), and can be tested on cloud provider playgrounds such as [Groq](https://console.groq.com/playground) and [Together](https://api.together.xyz/playground).

- **Let's discuss Llama-3.1 Paper (A lot of details on pre-training, post-training, etc)** ([Score: 109, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eabf4l/lets_discuss_llama31_paper_a_lot_of_details_on/)): **Llama 3.1 paper reveals pre-training details**  The Llama 3.1 paper, available at [ai.meta.com](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/), provides extensive details on the model's pre-training and post-training processes. The paper includes hyperparameter overviews, validation loss graphs, and various performance metrics for different model sizes ranging from **7B to 70B** parameters.

- **Early Hot Take on Llama 3.1 8B at 128K Context** ([Score: 72, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1eac5a7/early_hot_take_on_llama_31_8b_at_128k_context/)): **Llama 3.1 8B model's 128K context performance underwhelms**  The author tested the **Llama 3.1 8B model** with **128K context** using a novel-style story and found it less capable than **Mistral Nemo** and significantly inferior to the **Yi 34B 200K** model. The Llama model struggled to recognize previously established context about a character's presumed death and generate appropriate reactions, even when tested with **FP16** precision in **24GB VRAM** using **exllama with Q6 cache**. Despite further testing with **8bpw and Q8 quantization**, the author ultimately decided to abandon Llama 8B in favor of **Mistral Dori**.

**Theme 2. Open Source AI Strategy and Industry Impact**

- **Open source AI is the path forward - Mark Zuckerberg** ([Score: 794, Comments: 122](https://reddit.com//r/LocalLLaMA/comments/1eaa0m2/open_source_ai_is_the_path_forward_mark_zuckerberg/)): **Mark Zuckerberg advocates for open source AI**  Mark Zuckerberg argues that **open source AI** is crucial for advancing AI technology and ensuring its responsible development. In his [blog post](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/), Zuckerberg emphasizes the benefits of open source AI, including **faster innovation**, **increased transparency**, and **broader access** to AI tools and knowledge.

- **[Llama 3 405b is a "systemic risk" to society according to the AI Act](https://x.com/deanwball/status/1815826885663658445)** ([Score: 169, Comments: 68](https://reddit.com//r/LocalLLaMA/comments/1eal9oq/llama_3_405b_is_a_systemic_risk_to_society/)): **Meta's Llama 3.1 405B model** has been classified as a "**systemic risk**" under the **European Union's AI Act**. This designation applies to AI systems with more than **10^25 parameters**, placing significant regulatory obligations on Meta for the model's development and deployment. The classification highlights the growing concern over the potential societal impacts of large language models and the increasing regulatory scrutiny they face in Europe.

- **[OpenAI right now...](https://i.redd.it/h60m9gglyced1.jpeg)** ([Score: 167, Comments: 27](https://reddit.com//r/LocalLLaMA/comments/1eanchg/openai_right_now/)): **OpenAI's competitors are closing the gap**. The release of **Llama 3.1** by Meta has demonstrated significant improvements in performance, potentially challenging OpenAI's dominance in the AI language model space. This development suggests that the competition in AI is intensifying, with other companies rapidly advancing their capabilities.
    - **ChatGPT's Declining Performance**: Users report **ChatGPT's coding abilities have deteriorated** since early 2023, with **GPT-4** and **GPT-4 Turbo** showing inconsistent results and reduced reliability for tasks like generating PowerShell scripts.
    - **OpenAI's Credibility Questioned**: Critics highlight OpenAI's **lobbying efforts to regulate open-source AI** and the addition of former **NSA head Paul Nakasone** to their board, suggesting a shift away from their original "open" mission.
    - **Calls for Open-Source Release**: Some users express desire for OpenAI to **release model weights** for local running, particularly for **GPT-3.5**, as a way to truly advance the industry and live up to their "Open" name.


**Theme 3. Performance Benchmarks and Comparisons**

- **[LLama 3.1 vs Gemma and SOTA](https://www.reddit.com/gallery/1eaal5s)** ([Score: 140, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1eaal5s/llama_31_vs_gemma_and_sota/)): **Llama 3.1** outperforms **Gemma** and other state-of-the-art models across various benchmarks, including **MMLU**, **HumanEval**, and **GSM8K**. The **7B** and **13B** versions of Llama 3.1 show significant improvements over their predecessors, with the 13B model achieving scores comparable to or surpassing larger models like **GPT-3.5**. This performance leap suggests that Llama 3.1 represents a substantial advancement in language model capabilities, particularly in reasoning and knowledge-based tasks.

- **[Llama 3.1 405B takes #2 spot in the new ZebraLogic reasoning benchmark](https://i.redd.it/o9l7ym58fced1.png)** ([Score: 110, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1eakv0o/llama_31_405b_takes_2_spot_in_the_new_zebralogic/)): **Llama 3.1 405B** has secured the **second place** in the newly introduced **ZebraLogic reasoning benchmark**, demonstrating its advanced reasoning capabilities. This achievement positions the model just behind **GPT-4** and ahead of other notable models like **Claude 2** and **PaLM 2**. The ZebraLogic benchmark is designed to evaluate a model's ability to handle complex logical reasoning tasks, providing a new metric for assessing AI performance in this crucial area.

- **The final straw for LMSYS** ([Score: 175, Comments: 55](https://reddit.com//r/LocalLLaMA/comments/1ean2i6/the_final_straw_for_lmsys/)): **LMSYS benchmark credibility questioned**. The author criticizes **LMSYS's ELO ranking** for placing **GPT-4o mini** as the second-best model overall, arguing that other models like **GPT-4**, **Gemini 1.5 Pro**, and **Claude Opus** are more capable. The post suggests that **human evaluation of LLMs** is now limited by human capabilities rather than model capabilities, and recommends alternative benchmarks such as **ZebraLogic**, **Scale.com leaderboard**, **Livebench.ai**, and **LiveCodeBench** for more accurate model capability assessment.

**Theme 4. Community Tools and Deployment Resources**

- **[Llama-3.1 8B Instruct GGUF are up](https://huggingface.co/aniljava/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main)** ([Score: 50, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1eabypz/llama31_8b_instruct_gguf_are_up/)): **Llama-3.1 8B Instruct GGUF** models have been released, offering various quantization levels including **Q2_K**, **Q3_K_S**, **Q3_K_M**, **Q4_0**, **Q4_K_S**, **Q4_K_M**, **Q5_0**, **Q5_K_S**, **Q5_K_M**, **Q6_K**, and **Q8_0**. These quantized versions provide options for different trade-offs between model size and performance, allowing users to choose the most suitable version for their specific use case and hardware constraints.

- **Finetune Llama 3.1 for free in Colab + get 2.1x faster, 60% less VRAM use + 4bit BnB quants** ([Score: 85, Comments: 24](https://reddit.com//r/LocalLLaMA/comments/1eaitaq/finetune_llama_31_for_free_in_colab_get_21x/)): **Unsloth** has released tools for **Llama 3.1** that make finetuning **2.1x faster**, use **60% less VRAM**, and improve native HF inference speed by **2x** without accuracy loss. The release includes a **free Colab notebook** for finetuning the **8B model**, **4-bit Bitsandbytes quantized models** for faster downloading and reduced VRAM usage, and a preview of their **Studio Chat UI** for local chatting with Llama 3.1 8B Instruct in Colab.

- **We made glhf.chat: run (almost) any open-source LLM, including 405b** ([Score: 54, Comments: 26](https://reddit.com//r/LocalLLaMA/comments/1eap9fj/we_made_glhfchat_run_almost_any_opensource_llm/)): **New platform glhf.chat launches for running open-source LLMs**  The newly launched [glhf.chat](https://glhf.chat) platform allows users to run nearly any open-source LLM supported by the **vLLM project**, including models up to **~640GB of VRAM**. Unlike competitors, the platform doesn't have a hardcoded model list, enabling users to run any compatible model or finetune by pasting a **Hugging Face link**, with support for models like **Llama-3-70b** finetunes and upcoming **Llama-3.1** versions.
    - The platform initially required an invite code "405B" for registration, which was mentioned in the original post. **reissbaker**, the developer, later removed the invite system entirely to simplify access for all users.
    - Users encountered a "500 user limit" error due to an oversight in upgrading the auth provider. **Billy**, another glhf.chat developer, acknowledged the issue and promised a fix within minutes.
    - In response to a user request, **reissbaker** shipped a fix for the **Mistral NeMo architecture**, enabling support for models like the [dolphin-2.9.3-mistral-nemo-12b](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) on the platform.

## All AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI Model Releases and Benchmarks**

- **Meta releases Llama 3.1 405B model**: Meta has released a new 405 billion parameter Llama model. [Benchmark results](https://www.reddit.com/r/singularity/comments/1eab6b1/llama_31_405b_on_scale_leaderboards/) show it performing competitively with GPT-4 and Claude 3.5 Sonnet on some tasks.

- **Zuckerberg argues for open-sourcing AI models**: Mark Zuckerberg [made the case](https://www.reddit.com/r/singularity/comments/1eaej0u/mark_zuckerberg_eloquently_states_the_case_for/) that open-sourcing AI models is beneficial, arguing that closed models will be stolen anyway. He stated that **"it doesn't matter that China has access to open weights, because they will just steal weights anyway if they're closed."**

- **Google releases "AI Agents System"**: Google has released [Project Oscar](https://www.reddit.com/r/singularity/comments/1ea1kz9/google_has_released_the_worlds_first_ai_agents/), an open-source platform for creating AI agents to manage software projects, particularly for monitoring issues and bugs.

**AI Capabilities and Benchmarks**

- **Debate over AI surpassing human intelligence**: There is ongoing discussion about whether current AI models have surpassed human-level intelligence in certain domains. Some argue that [AI is now "smart enough to fool us"](https://www.reddit.com/r/singularity/comments/1eaud7r/for_the_first_time_in_history_the_ais_are_smart/), while others contend that **AI still struggles with simple logic and math tasks**.

- **Limitations of current benchmarks**: Critics point out that [current AI benchmarks may not accurately measure intelligence](https://www.reddit.com/r/singularity/comments/1eaud7r/for_the_first_time_in_history_the_ais_are_smart/leogncp/). For example, the Arena benchmark measures which responses people prefer, not necessarily intelligence.

**AI Ethics and Corporate Practices**

- **OpenAI criticized for non-disclosure agreements**: OpenAI faced criticism after a [community note on social media](https://www.reddit.com/r/OpenAI/comments/1eaq40g/openai_got_community_noted/) highlighted that the company had previously used non-disclosure agreements that prevented employees from making protected disclosures.

- **Debate over open vs. closed AI development**: There is ongoing discussion about the merits of open-sourcing AI models versus keeping them closed. Some argue that open-sourcing promotes innovation, while others worry about potential misuse.

---

# AI Discord Recap

> A summary of Summaries of Summaries

**1. Llama 3.1 Model Performance and Challenges**

- **Fine-Tuning Woes**: **Llama 3.1** users reported **issues with fine-tuning**, particularly with error messages related to model configurations and tokenizer handling, suggesting updates to the transformers library.
   - Discussions emphasized the need for specifying **correct model versions** and maintaining the right configuration to mitigate these challenges.
- **Inconsistent Performance**: Users noted that **Llama 3.1 8B** struggles with reasoning and coding tasks, with some members expressing skepticism regarding its overall performance.
   - Comparisons suggest that while it's decent for its size, its logic capabilities appear lacking, especially contrasted with models like **Gemma 2**.
- **Overload Issues**: The **Llama 3.1 405B** model frequently shows 'service unavailable' errors due to being overloaded with requests, suggesting higher demand and potential infrastructure limits.
   - Users discussed the characteristics of the 405B variant, mentioning that it feels more censored compared to its 70B sibling.
    


**2. Mistral Large 2 Model**

- **Mistral Large 2 Release**: On July 24, 2024, Mistral AI launched **Mistral Large 2**, featuring an impressive **123 billion parameters** and a **128,000-token context window**, pushing AI capabilities further.
   - Mistral Large 2 is reported to outperform **Llama 3.1 405B**, particularly in complex mathematical tasks, making it a strong competitor against industry giants.
- **Multilingual Capabilities**: The **Mistral Large 2** model boasts a longer context window and multilingual support compared to existing models, making it a versatile tool for various applications.
   - Members engaged in comparisons with Llama models, noting ongoing performance enhancement efforts in this evolving market.
    


**3. AI in Software Development and Job Security**

- **Job Security Concerns**: Participants addressed **job security uncertainties** among junior developers as AI tools increasingly integrate into coding practices, potentially marginalizing entry-level roles.
   - Consensus emerged that experienced developers should adapt to these tools, using them to enhance productivity rather than replace human interaction.
- **Privacy in AI Data Handling**: Concerns arose regarding **AI's data handling practices**, particularly the implications of human reviewers accessing sensitive information.
   - The discourse underscored the critical need for robust data management protocols to protect user privacy.
    


**4. AI Model Benchmarking and Evaluation**

- **Benchmarking Skepticism**: Skepticism arises over the performance metrics of **Llama 405b**, with discussions highlighting its average standing against **Mistral** and **Sonnet** models.
   - The community reflects on varied benchmark results and subjective experiences, likening benchmarks to *movie ratings* that fail to capture true user experience.
- **Evaluation Methods**: The **need for better benchmarks** in **hallucination prevention techniques** was highlighted, prompting discussions on improving evaluation methods.
   - A brief conversation with a Meta engineer raised concerns about the current state of benchmarking, suggesting a collaborative approach to developing more reliable metrics.
    


**5. Open-Source AI Developments**

- **Llama 3.1 Release**: The **Llama 3.1** model has officially launched, expanding context length to **128K** and supporting eight languages, marking a significant advancement in open-source AI.
   - Users reported frequent 'service unavailable' errors with the **Llama 3.1 405B** model due to overload, suggesting it feels more censored than its **70B** counterpart.
- **Mistral Large 2 Features**: **Mistral Large 2** features state-of-the-art **function calling capabilities**, with day 0 support for structured outputs and agents.
   - This release aligns with enhanced **function calling** and **structured outputs**, providing useful resources like cookbooks for users to explore.
    

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 3.1 Fine-Tuning Challenges**: Users reported **issues fine-tuning Llama 3.1**, particularly with error messages stemming from model configurations and tokenizer handling, suggesting updates to the transformers library.
   - Discussions emphasized the need for specifying **correct model versions** and maintaining the right configuration to mitigate these challenges.
- **Job Security Concerns in AI Development**: Participants addressed **job security uncertainties** among junior developers as AI tools increasingly integrate into coding practices, potentially marginalizing entry-level roles.
   - Consensus emerged that experienced developers should adapt to these tools, using them to enhance productivity rather than replace human interaction.
- **Insights on Image Generation Bias**: Discussions around **image generation** highlighted challenges in achieving diversity and addressing biases inherent in AI models, which are crucial for educational contexts.
   - Critiques of current diversity efforts emerged, pointing out execution flaws that could skew historical accuracy.
- **Performance of Mistral Large 2**: The **Mistral Large 2** model surfaced as a strong competitor in the AI landscape, boasting a longer context window and multilingual support compared to existing models.
   - Members engaged in comparisons with Llama models, noting ongoing performance enhancement efforts in this evolving market.
- **Privacy Concerns in AI Data Handling**: Concerns arose regarding **AI's data handling practices**, particularly the implications of human reviewers accessing sensitive information.
   - The discourse underscored the critical need for robust data management protocols to protect user privacy.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio struggles running Llama 3.1**: Users identified that **LM Studio** cannot run **Llama 3.1** on OpenCL cards; upgrading to version **0.2.28** is recommended for better support.
   - Confirmed updates from **LM Studio** are essential for effective performance of large models like Llama 3.1.
- **ROCm 0.2.28 leads to performance degradation**: After the ROCm **0.2.28** update, a user experienced reduced performance, seeing only **150w usage** on a dual **7900 XT** setup.
   - Reverting to **0.2.27** restored normal performance, prompting calls for a deeper investigation into changes in the new update.
- **Nemo Models face context and performance issues**: Users report that **Nemo models** function with current versions but suffer from context length limitations and slower outputs due to insufficient RAM.
   - There were success stories with particular setups, alongside suggestions for optimizations.
- **GPU Offloading Problems Persist**: Several members reported malfunctioning GPU offloading on their systems, particularly with M3 Max and **4080S** GPUs, often requiring manual adjustments.
   - Automatic settings caused errant outputs, indicating a need for more reliable manual configurations for better performance.
- **Meta-Llama 3.1 70B hits the repository**: The release of **70B quant models** for **Meta-Llama 3.1** has been announced, available through [the repository](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF).
   - Enthusiasm in the channel was notable, with expectations for improved performance following a re-upload to fix a **tokenizer bug**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Llama 3.1 405B Makes Waves**: The **Llama 3.1 405B** model is touted as the most capable open-source model, now available on Perplexity, rivaling **GPT-4o** and **Claude Sonnet 3.5** for performance.
   - Exciting plans for its integration into **mobile applications** are in the works, enhancing accessibility for on-the-go developers.
- **Mistral Large 2 Breaks New Ground**: On July 24, 2024, Mistral AI launched **Mistral Large 2**, featuring an impressive **123 billion parameters** and a **128,000-token context window**, pushing AI capabilities further.
   - Mistral Large 2 is reported to outperform **Llama 3.1 405B**, particularly in complex mathematical tasks, making it a strong competitor against industry giants.
- **AI Model Benchmarking Under Scrutiny**: Skepticism arises over the performance metrics of **Llama 405b**, with discussions highlighting its average standing against **Mistral** and **Sonnet** models.
   - The community reflects on varied benchmark results and subjective experiences, likening benchmarks to *movie ratings* that fail to capture true user experience.
- **NextCloud Integrates OpenAI**: A recent integration of **NextCloud** with OpenAI has sparked interest, featuring a community-driven, open-source approach that promotes clear coding standards.
   - A GitHub repository was shared, providing aspiring developers resources to explore this new functionality and its implications.
- **TikTok's Search Engine Potential**: A lively discussion on TikTok as a search tool for Gen Z highlights its rising relevance and challenges traditional search engines.
   - Concerns around the platform's reliability, especially in health advice, indicate a need for caution when using TikTok for critical information.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Mistral-7B boasts massive context windows**: The **Mistral-7B-v0.3** model features an impressive **128k context window** and supports multiple languages, while the **Mistral Large** version runs efficiently at **69GB** using **ollama**.
   - Users praised its capabilities, pointing to potential applications for multitasking with larger datasets.
- **Affordable GPU server options emerge**: Discussions highlighted **Runpod** as a budget-friendly GPU server option for large models, priced at just **$0.30/hour**.
   - Participants recommended using **LM Studio** and **ollama** for better performance tailored to specific model requirements.
- **Kling AI offers quirky image-to-video generation**: **Kling AI** impressed users with its ability to create videos from still images, although some noted issues with video quality and server overloads.
   - Despite mixed experiences, the engaging output sparked further interest in experimenting with the tool.
- **Memory feature inconsistencies frustrate users**: Members reported variable appearances of the **memory feature** in the EU, with some only able to access it temporarily for five minutes.
   - This led to lighthearted banter about the feature’s operational status and its overall reliability.
- **Generating PDFs with OpenAI in Python**: A user sought help for generating PDF documents via **Python** using OpenAI, looking for ways to automate section descriptions based on uploaded content.
   - This discussion drove a collaborative exchange on effective workflows to enhance document generation processes.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM Distillation Advancements**: Members have highlighted the potential of the [Minitron GitHub repository](https://github.com/NVlabs/Minitron) for understanding recent advancements in **LLM distillation** techniques using **pruning** and **knowledge distillation**.
   - This repository reflects ongoing efforts similar to models like **Sonnet**, **Llama**, and **GPT-4Omini**.
- **LLaMa 3 Introduced as a New Player**: The recently introduced **LLaMa 3** models feature a dense Transformer structure equipped with **405B parameters** and a context window of up to **128K tokens**, designed for various complex tasks.
   - These models excel in multilinguality and coding, setting a new benchmark for AI applications.
- **Mistral Large 2's Competitive Edge**: The release of **Mistral Large 2** with **123B parameters** and a **128k context window** has captivated users, especially for coding tasks.
   - Despite its non-commercial license, its innovative design positions it well for optimal API performance.
- **Fine-Tuning Llama 3 Presents Challenges**: Concerns surface over **fine-tuning Llama 3 405B**, where some suggest only **Lora FTing** as a feasible approach.
   - This situation may bolster advances in **DoRA fine-tuning** efforts within the **OSS** community.
- **Moral Reasoning and the Trolley Problem**: Discussions around incorporating **difficult moral queries**, like the **trolley problem**, have emphasized the need to evaluate models' moral foundations.
   - This triggers debates on whether these tasks examine pure reasoning skills or ethical frameworks.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek Coder V2 Launches Private Inference Provider**: **DeepSeek Coder V2** now features a [private provider](https://openrouter.ai/models/deepseek/deepseek-coder) to serve requests on OpenRouter without input training, marking a significant advancement in private model deployment.
   - This new capacity reflects strategic progression within the **OpenRouter** platform as it enhances usability for users.
- **Concerns over Llama 3.1 405B Performance**: Users express dissatisfaction with the performance of **Llama 3.1 405B**, particularly its handling of NSFW content where it often refuses prompts or outputs training data.
   - Feedback indicates temperature settings significantly affect quality, with some users reporting better output at lower temperatures.
- **Mistral Large 2 Replacement Provides Better Multilingual Support**: **Mistral Large 2** is now launched as **Mistral Large**, effectively replacing the previous version with enhanced multilingual capabilities.
   - Users speculate it may outperform **Llama 3.1** when dealing with languages like French, as they assess its comparative effectiveness.
- **Users Discuss OpenRouter API Limitations**: Discussion highlights **OpenRouter API** challenges, particularly in terms of rate limits and multilingual input management, which complicates model usage.
   - While some models are in free preview, users report strict limits on usage and context, pointing to a need for improvements.
- **Interest in Open-Source Coding Tools Grows**: Users show a keen interest in open-source autonomous coding tools like **Devika** and **Open Devin**, asking for recommendations based on current efficacy.
   - This shift reflects a desire to experiment with alternatives to mainstream AI coding solutions that exhibit varied performance.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 Launches with Excitement**: The **Llama 3.1** model has officially launched, expanding context length to **128K** and supporting eight languages, marking a significant advancement in open-source AI. The model can be explored in detail through the [blogpost](https://huggingface.co/blog/llama31) and is available for testing [here](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct).
   - Users reported frequent 'service unavailable' errors with the **Llama 3.1 405B** model due to overload, suggesting it feels more censored than its **70B** counterpart.
- **Improved HuggingChat with Version v0.9.1**: The latest version **HuggingChat v0.9.1** integrates new features that significantly enhance user accessibility. Users can discover more functionalities through the's model page.
   - The update aims to improve interactions utilizing the new [HuggingChat](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct) features.
- **Risks with MultipleNegativesRankingLoss**: Difficulties were reported when training sentence encoders using **MultipleNegativesRankingLoss**, where increasing the batch size led to worse model performance. Insights were sought on common dataset pitfalls associated with this method.
   - One user described their evaluation metrics, focusing on **recall@5**, **recall@10**, and **recall@20** for better benchmarking.
- **Mistral-NeMo 12B Shines in Demo**: A demo of **Mistral-NeMo 12B Instruct** using [llama.cpp](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp) showcases the model's significant performance enhancements. Users are encouraged to experiment for an improved chat experience.
   - Community interest is soaring regarding the model's capabilities and potential applications in various AI tasks.
- **Questions on Rectified Flow and Evaluation**: Members expressed frustration regarding the lack of discussions around **Rectified Flow** and **Flow Matching**, especially in contrast to **DDPM** and **DDIM** debates. They emphasized the difficulty finding straightforward examples for **Flow** applications such as generating **MNIST**.
   - Evaluation methods for generative models were explored, with a focus on qualitative and quantitative methods for assessing the performance of models like **Stable Diffusion** versus **GANs**.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Kohya-ss GUI Compatibility Quirks**: Users reported that the current version of **Kohya-ss GUI** faces compatibility issues with Python **3.10**, requiring an upgrade to **3.10.9** or higher.
   - *One user humorously remarked* that it resembles needing a weight limit of **180lbs** but not exceeding **180.5lbs**.
- **Exciting Lycoris Features on the Horizon**: **Onetrainer** is potentially integrating **Lycoris** features in a new dev branch, spurring discussions on functional enhancements.
   - Community members noted a preference for **bmaltais' UI wrapper**, which could improve experiences with these new integrations.
- **Community Raves About Art Models**: Discussion outlined performance ratings for models including **Kolors, Auraflow, Pixart Sigma,** and **Hunyuan**, with **Kolors** being commended for its speed and quality.
   - Participants engaged in a debate on user experiences and specific applications of these models, showcasing diverse opinions.
- **Stable Diffusion Models Under the Microscope**: Users examined the differences in output between **Stable Diffusion 1.5** and **SDXL**, focusing on detail and resolution.
   - Techniques such as **Hidiffusion** and **Adaptive Token Dictionary** were discussed as methods to boost older model outputs.
- **Welcome to Stable Video 4D!**: The newly introduced **Stable Video 4D** model allows transformation of single object videos into multi-angle views for creative projects.
   - Currently in research, this model promises applications in **game development, video editing,** and **virtual reality**.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Diving Deep into Sampling Models**: Members discussed various **sampling methods** such as **greedy**, **top-p**, and **top-k**, highlighting their respective trade-offs, particularly for large language models.
   - Stochastic sampling is noted for diversity but complicates evaluation, contrasting with the reliability of greedy methods which generate the most probable paths.
- **Llama 3.1's Sampling Preferences**: In discussions about **Llama 3.1**, participants recommended consulting its paper for optimal **sampling methods**, with a lean towards probabilistic sampling techniques.
   - One member pointed out that **Gemma 2** effectively uses top-p and top-k strategies common in model evaluations.
- **Misleading Tweets Trigger Discussion**: Members analyzed a misleading tweet related to **Character.ai's** model, particularly its use of shared KV layers impacting performance metrics.
   - Concerns arose regarding the accuracy of such information, highlighting the community's ongoing journey to comprehend transformer architectures.
- **MoE vs Dense Models Debate**: A lively debate emerged over the preference for **dense models** over **Mixture-of-Experts (MoE)**, citing high costs and engineering challenges of handling MoEs in training.
   - Despite the potential efficiency of pre-trained MoEs, concerns linger about varied organizational capabilities to implement them.
- **Llama API Evaluation Troubles**: Users reported errors with the `lm_eval` tool for **Llama 3.1-405B**, particularly challenges in handling logits and multiple-choice tasks through the API.
   - Errors such as 'No support for logits' and 'Method Not Allowed' prompted troubleshooting discussions, with successful edits to the `_create_payload` method noted.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Installation Troubleshooting**: Members faced issues when **Torch** wasn't compiled with **CUDA**, leading to import errors. Installation of the CUDA version from the official page was recommended for ensuring compatibility.
   - After setting up CUDA, one user encountered a **torch.cuda.OutOfMemoryError** while allocating **172.00 MiB**, suggesting adjustments to **max_split_size_mb** to tackle memory fragmentation.
- **Exploring Llama-2 and Llama-3 Features**: A member shared a fine-tuned [Llama-2 7B model](https://huggingface.co/TheBloke/Llama-2-7B-fp16), trained on a **24GB GPU** in **19 hours**. Concurrently, discussions on implementing **blockwise attention** in Llama 3 focused on the sequence splitting stage relative to rotary position embeddings.
   - Additionally, inquiries on whether **Llama 3.1** has improved inference latency over **3.0** were raised, reflecting ongoing interests in model performance advancements.
- **Optimizations in FlashAttention for AMD**: FlashAttention has gained support for **AMD ROCm**, following the implementation detailed in [GitHub Pull Request #1010](https://github.com/Dao-AILab/flash-attention/pull/1010). The updated library maintains API consistency while introducing several new C++ APIs like `mha_fwd`.
   - Current compatibility for the new version is limited to **MI200 and MI300**, suggesting potential broader updates may follow in the future.
- **PyTorch Compile Insights**: Users reported that `torch.compile` increased **RAM usage** with small **Bert models**, and switching from eager mode resulted in worse performance. Suggestions to use the PyTorch profiler to analyze memory traces during inference were offered.
   - Observations indicated no memory efficiency improvements with `reduce-overhead` and `fullgraph` compile options, emphasizing the importance of understanding configuration effects.
- **Strategies for Job Hunting in ML/AI**: A user sought advice on drafting a roadmap for securing internships and full-time positions in **ML/AI**, sharing a [Google Document with their plans](https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing). They expressed a commitment to work hard and remain flexible on timelines.
   - Further feedback on their internship strategies was encouraged, highlighting the willingness to dedicate extra hours towards achieving their objectives.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.1 Struggles with Errors**: Users reported issues with **Llama 3.1**, facing errors like *AttributeError*, which may stem from outdated images or configurations.
   - One user found a workaround by trying a different image, expressing frustration over ongoing model updates.
- **Mistral Goes Big with Large Model Release**: Mistral released the **Mistral-Large-Instruct-2407** model with **123B parameters**, claiming state-of-the-art performance.
   - The model offers multilingual support, coding proficiency, and advanced agentic capabilities, stirring excitement in the community.
- **Multilingual Capabilities under Scrutiny**: Comparisons between **Llama 3.1** and **NeMo** highlighted performance differences, particularly in multilingual support.
   - While **Llama 3** has strengths in European languages, users noted that **NeMo** excels in **Chinese** and others.
- **Training Large Models Hits RAM Barriers**: Concerns arose over the significant RAM requirements for training large models like Mistral, with users remarking on their limitations.
   - Some faced exploding gradients during training and speculated whether this issue was tied to sample packing.
- **Adapter Fine-Tuning Stages Gaining Traction**: Members discussed multiple stages of adapter fine-tuning, proposing the idea of initializing later stages with previous results, including SFT weights for DPO training.
   - A feature request on [GitHub](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) suggests small code changes to facilitate this method.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o mini dominates Chatbot Arena**: With over **4,000 user votes**, **GPT-4o mini** is now tied for #1 in the Chatbot Arena leaderboard, outperforming its previous version while being **20x cheaper**. This milestone signals a notable decline in the **cost of intelligence** for new applications.
   - *Excitement* was evident as developers celebrated this accomplishment, noting its implications for future chatbot experiences.
- **Mistral Large 2: A New Contender**: **Mistral Large 2** boasts a **128k context window** and multilingual support, positioning itself strongly for high-complexity tasks under specific licensing conditions. Discussions surfaced on the lack of clarity regarding *commercial use* of this powerful model.
   - Members emphasized the need for better documentation to navigate the **licensing** landscape effectively.
- **OpenAI's $5 billion Loss Prediction**: Estimates suggest OpenAI could face a staggering loss of **$5 billion** this year, primarily due to Azure costs and training expenses. The concern over profitability has prompted discussions about the surprisingly low API revenue compared to expectations.
   - This situation raises fundamental questions about the sustainability of OpenAI's business model in the current environment.
- **Llama 3 Officially Released**: Meta has [officially released](https://llama.meta.com/) **Llama3-405B**, trained on **15T tokens**, which claims to outperform **GPT-4 on all major benchmarks**. This marks a significant leap in open-source AI technology.
   - The launch has sparked discussions around the integration of **100% RLHF** in the post-training capabilities of the model, which highlights the crucial role of this method.
- **CrowdStrike's $10 Apology Gift Card for Outage**: **CrowdStrike** is offering partners a **$10 Uber Eats gift card** as an apology for a massive outage, but some found the vouchers had been **canceled** when attempting to redeem them. This incident underscores the operational risks associated with technology updates.
   - Members shared mixed feelings about the effectiveness of this gesture amid ongoing frustrations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Compiler Versioning Confusion**: A discussion highlighted the uncertainty ongoing about whether the next main compiler version will be **24.5** or **24.8**, citing potential disconnects between nightly and main releases as they progress towards **2025**.
   - Community members raised concerns about adhering to different release principles, complicating future updates.
- **Latest Nightly Update Unpacked**: The newest nightly Mojo compiler update, `2024.7.2405`, includes significant changes such as the removal of **DTypePointer** and enhanced string formatting methods, details of which can be reviewed in the [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
   - The removal of **DTypePointer** necessitates code updates for existing projects, prompting calls for clearer transition guidelines.
- **SDL Integration Questions Arise**: A user requested resources for integrating **SDL** with **Mojo**, aiming to gain a better understanding of the process, and how to use **DLHandle** effectively.
   - This reflects a growing interest in enhancing Mojo’s capabilities through third-party libraries.
- **Discussion on Var vs Let Utility**: A member initiated a debate on the necessity of using **var** in situations where everything is already declared as such, suggesting redundancy in usage.
   - Another pointed out that **var** aids the compiler while **let** caters to those favoring immutability, highlighting a preference debate among developers.
- **Exploring SIMD Type Comparability**: Members discussed challenges in establishing total ordering for **SIMD types**, noting tension between generic programming and specific comparisons.
   - It was proposed that a new **SimdMask[N]** type might alleviate some complexities associated with platform-specific behaviors.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Factorio Automation Mod sparks creativity**: The new [factorio-automation-v1](https://github.com/naklecha/factorio-automation) mod allows agents to automate tasks like crafting and mining in *Factorio*, offering a fun testing ground for agent capabilities.
   - Members are excited about the possibilities this mod opens up for complex game interactions.
- **GPT-4o Mini Fine-Tuning opens up**: OpenAI has launched fine-tuning for **GPT-4o mini**, available to tier 4 and 5 users, with the first **2M training tokens** free daily until September 23.
   - Members noted performance inconsistencies when comparing fine-tuned **GPT-4o mini** with **Llama-3.1-8b**, raising questions about exact use cases.
- **Mistral Large 2 impresses with 123B parameters**: [Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) has been revealed, boasting **123 billion parameters**, strong coding capabilities, and supporting multiple languages.
   - However, indications show it achieved only a **60% score on Aider's code editing benchmark**, slightly ahead of the best GPT-3.5 model.
- **Reddit's Content Policy stirs debate**: A heated discussion surfaced about Reddit's [public content policy](https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy), with concerns around user control over generated content.
   - Members argue that the vague policy creates significant issues, highlighting the need for clearer guidelines.
- **Join the Llama 3 Emergency Paper Club**: An *emergency paper club* meeting on [The Llama 3 Herd of Models](https://x.com/latentspacepod/status/1816151808357908698) is set for later today, a strong contender for **POTY Awards**.
   - Key contributors to the discussion include prominent community members, emphasizing the paper's significance.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse Enhances Markdown Capabilities**: **LlamaParse** now showcases support for **Markdown output**, **plain text**, and **JSON mode** for better metadata extraction. Features such as **multi-language** output enhance its utility across workflows, as demonstrated in [this video](https://t.co/RUWJ0Z2NMn).
   - This update is set to significantly improve **OCR** efficiency for diverse applications, broadening its adoption for various tasks beyond simple text.
- **MongoDB’s AI Applications Program is Here**: The newly launched **MongoDB AI Applications Program (MAAP)** aims to simplify the journey for organizations building **AI-enhanced applications**. With **reference architectures** and integrated technology stacks, it accelerates AI deployment timeframes; learn more [here](https://t.co/rCz3DfUe3A).
   - The initiative addresses the urgent need developers have to modernize their applications with minimal overhead, contributing to more efficient workflows.
- **Mistral Large 2 Introduces Function Calling**: **Mistral Large 2** is rolling out enhanced **function calling capabilities**, which includes support for structured outputs as soon as it launches. Detailed resources such as **cookbooks** are provided to aid developers in utilizing these new functionalities; explore them [here](https://t.co/ho02wDbGpZ).
   - This release underscores **functional versatility** for LLM applications, allowing developers to implement more complex interactions effectively.
- **Streaming Efficiency with SubQuestionQueryEngine**: Members discussed employing **SubQuestionQueryEngine.from_defaults** to facilitate streaming responses and reduce latency within LLM queries. Some solutions were proposed using `get_response_synthesizer`, though challenges remain in implementation.
   - Despite the hurdles in adoption, there's optimism about improving user interaction speeds across LLM integrations.
- **Doubts Surface Over Llama 3.1 Metrics**: Skepticism mounts regarding the **metrics** published by Meta for **Llama 3.1**, especially its effectiveness in **RAG evaluations**. Users are questioning the viability of certain models like `llama3:70b-instruct-q_5` for practical tasks.
   - This skepticism reflects broader community concerns regarding the reliability of AI metrics in assessing model performance in various applications.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Dashboard Reloading Trouble**: Members reported issues with the **Cohere account dashboard** constantly reloading, while others noted no such problems on their end, leading to discussions on potential glitches.
   - This prompted a conversation about **rate limiting** as a possible cause for the reloading issue.
- **Cheering for Command R Plus**: With each release of models like **Llama 3.1**, members expressed increasing appreciation for **Command R Plus**, highlighting its capabilities compared to other models.
   - One user proposed creating a playground specifically for **model comparisons** to further explore this growing sentiment.
- **Server Performance Under Scrutiny**: Concerns arose regarding potential server downtime, but some users confirmed that the server was in **full operational status**.
   - Suggestions included investigating **rate limiting** as a factor influencing user experience.
- **Innovative Feature Suggestions for Cohere**: A member suggested incorporating the ability to use tools during conversations in **Cohere**, like triggering a web search on demand.
   - Initial confusion arose, but it was clarified that some of these functionalities are already available.
- **Community Welcomes New Faces**: New members introduced themselves, sharing backgrounds in **NLP and NeuroAI**, sparking excitement about the community.
   - The discussion also touched on experiences with **Command-R+**, emphasizing its advantages over models like **NovelAI**.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Zenbase/Core Launch Sparks Excitement**: **zenbase/core** is now live, enabling users to integrate **DSPy’s optimizers** directly into their Python projects like Instructor and LangSmith. Support the launch by engaging with their [Twitter post](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ).
   - Community members are responding positively, with a strong willingness to promote this recent release.
- **Typed Predictors Raise Output Concerns**: Users report issues with **typed predictors** not producing correctly structured outputs, inviting help from others. Suggestions include enabling experimental features with `dspy.configure(experimental=True)` to address these problems.
   - Encouragement from peers highlights a collective effort to refine the usage of these predictors.
- **Internal Execution Visibility Under Debate**: There's a lively discussion over methods to observe internal program execution steps, including suggestions like `inspect_history`. Users express the need for deeper visibility into model outputs, especially during type-checking mishaps.
   - A common desire for transparency showcases the importance of debugging tools in DSPy usage.
- **Push for Small Language Models**: One member shared an article on the advantages of **small language models**, noting their efficiency and suitability for edge devices with limited resources. They highlighted benefits like **privacy** and operational simplicity for models running on just **4GB of RAM**.
   - Check out the article titled [Small Language Models are the Future](https://medium.com/thoughts-on-machine-learning/small-language-models-are-the-future-6e8909567198) for a comprehensive read on the topic.
- **Call to Contribute to DSPy Examples**: A user expressed interest in contributing beginner-friendly examples to the DSPy repository, aiming to enrich the resource base. Community feedback confirmed a need for more diverse examples, specifically in the `/examples` directory.
   - This initiative reflects a collaborative spirit to enhance learning materials within the DSPy environment.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Members tackle Tinygrad learning**: Members express their ongoing journey with **Tinygrad**, focusing on understanding its use concerning **transformers**. One noted, *It's a work in progress,* indicating a gradual mastery process.
   - Discussion hinted at potential collective resources to enhance the learning curve.
- **Molecular Dynamics engine under construction**: A team is developing a **Molecular Dynamics engine** using neural networks for energy prediction, facing challenges in gradient usage. Input gradient tracking methods were suggested to optimize weight updates during backpropagation.
   - Optimizing backpropagation emerged as a focal point to improve training performance.
- **Creating a Custom Runtime in Tinygrad**: A member shared insights on implementing a **custom runtime** for Tinygrad, emphasizing how straightforward it is to add support for new hardware. They sought clarity on terms like `global_size` and `local_size`, vital for kernel executions.
   - Technical clarifications were provided regarding operational contexts for these parameters.
- **Neural Network Potentials discussion**: The energy in the Molecular Dynamics engine relies on **Neural Network Potentials (NNP)**, with emphasis on calculation efficiency. Conversations revolved around strategies to optimize backpropagation.
   - Clear paths for enhancing calculation speed are necessary to improve outcomes.
- **PPO Algorithm scrutiny in CartPole**: A member probed the necessity of the `.sum(-1)` operation in the implementation of the **PPO algorithm** for the Beautiful CartPole environment. This sparked a collaborative conversation on the nuances of reinforcement learning.
   - Detailed exploration of code implementations fosters community understanding and knowledge sharing.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Countdown to 3.1 and Cool Interviews**: Members inquired about whether there would be any cool [interviews](https://github.com/pytorch/torchtune/pull/790) released along with the **3.1** version, similar to those for **Llama3**.
   - This raises interest for potential insights and discussions that might accompany the new release.
- **MPS Support PR Gains Attention**: A new pull request ([#790](https://github.com/pytorch/torchtune/pull/790)) was highlighted which adds support for **MPS** on local Mac computers, checking for BF16 compatibility.
   - Context suggests this PR could resolve major testing hurdles for those using MPS devices.
- **LoRA Functionality Issues Persist**: Discussed issues surrounding **LoRA** functionality, noting it did not work during a previous attempt and was previously impacted by hardcoded **CUDA** paths.
   - Members exchanged thoughts on specific errors encountered, highlighting ongoing challenges in implementation.
- **Fixing the Pad ID Bug**: A member pointed out that the **pad id** should not be showing up in generate functionality, identifying it as an important bug.
   - In response, a Pull Request was created to prevent **pad ids** and special tokens from displaying, detailed in [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211).
- **Optimizing Git Workflow to Reduce Conflicts**: Discussion around refining git workflows to minimize the occurrence of new conflicts constantly arose, emphasizing collaboration.
   - It was suggested that new conflicts might stem from the workflow, indicating a potential need for tweaks.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Hugging Face Models and Agents Discussion**: Members discussed their experiences with **Agents using Hugging Face models**, including local LLMs via **Ollama** and cloud options like **OpenAI** and **Azure**.
   - This conversation sparked interest in the potential applications of agents within various model frameworks.
- **Python Developers Job Hunt**: A member urgently expressed their situation, stating, **'anyone looking to hire me? I need to pay my bills.'** and highlighted their strong skills in **Python**.
   - The urgency of job availability in the current market was apparent as discussions about opportunities ensued.
- **Challenges with HNSW IVFFLAT Indexes on Aurora**: Members faced problems creating **HNSW** or **IVFFLAT** indexes with **3072 dimensions** on **Aurora PGVECTOR**, leading to shared insights about solutions involving **halfvec**.
   - This highlighted ongoing challenges with dimensionality management in high-performance vector databases.
- **LangServe's OSError Limits**: Users encountered an **OSError: [Errno 24] Too many open files** when their LangServe app processed around **1000 concurrent requests**.
   - They are actively seeking strategies to handle high traffic while mitigating system resource limitations, with a [GitHub issue](https://github.com/langchain-ai/langserve/issues/714) raised for support.
- **Introduction of AI Code Reviewer Tool**: A member shared a [YouTube video](https://youtu.be/g_VRsjpC4e8) on the **AI Code Reviewer**, highlighting its features powered by **LangChain**.
   - This tool aims to enhance the **code review process**, suggesting a trend towards automation in code assessment methodologies.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Llama 3.1 405 B impresses with ease of use**: **Llama 3.1 405 B** performs fantastically out of the box with [OpenInterpreter](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587), offering an effortless experience.
   - In contrast, **gpt-4o** requires constant reminders about capabilities, making 405b a superior choice for multitasking.
- **Cost-effective API usage with Nvidia**: A user shared that **Nvidia** provides **1000 credits** upon signup, where 1 credit equals 1 API call.
   - This incentive opens up more accessibility for experimenting with APIs.
- **Mistral Large 2 rivals Llama 3.1 405 B**: **Mistral Large 2** reportedly performs comparably to **Llama 3.1 405 B**, particularly noted for its speed.
   - The faster performance may be due to lower traffic on Mistral's endpoints compared to those of Llama.
- **Llama 3.1 connects with databases for free**: [MikeBirdTech](https://x.com/MikeBirdTech/status/1816163862208766137) noted that **Llama 3.1** can interact with your database at no cost through **OpenInterpreter**, emphasizing savings on paid services.
   - *It's also fully offline and private, nobody else needs to see your data,* highlighting its **privacy benefits**.
- **Concerns over complex databases using Llama 3.1**: A member raised a concern that for **complex databases** involving joins across tables, this solution may not be effective.
   - They expressed appreciation for sharing the information, remarking on the **well-done** execution despite the limitations.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Llama 3.1: Meta's Open Source Breakthrough**: Meta recently launched **Llama 3.1 405B**, hailed as the first-ever **open-sourced frontier AI model**, outperforming competitive models like GPT-4o on various benchmarks. For more insights, check this [YouTube video](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61) featuring Mark Zuckerberg discussing its implications.
   - The reception highlights the model's potential impact on AI research and open-source contributions.
- **Trouble Downloading LAION2B-en Metadata**: Members reported difficulties in locating and downloading the **LAION2B-en metadata** from Hugging Face, querying if others faced the same problem. Responses indicate this is a common frustration with accessibility.
   - Someone linked to [LAION maintenance notes](https://laion.ai/notes/laion-maintenance/) for further clarification on the situation.
- **LAION Datasets in Legal Limbo**: Discussion revealed that **LAION datasets** are currently in **legal limbo**, with access to official versions restricted. While alternatives are available, it is advised to utilize unofficial datasets only for urgent research needs.
   - Members noted the ongoing complexities surrounding data legality in the AI community.
- **YouTube Polls: A Nostalgic Debate**: A member shared a [YouTube poll](http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS) asking which **90's movie had the best soundtrack**, igniting nostalgia among viewers. This prompts members to reflect on their favorite soundtracks from the era.
   - The poll sparks a connection through shared cultural experiences.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Legal Clarity on ML Dataset Copyright**: A member pointed out that most of the datasets generated by an **ML model** are likely not copyrightable since they lack true creativity. They emphasized that content not generated by **GPT-4** may be under **MIT licensing**, though this area remains murky amid current legal debates.
   - This opens up discussions on the implications for **data ownership** and ethical guidelines in dataset curation.
- **Navigating Non-Distilled Data Identification**: Discussion arose around the methods to pinpoint **non-distilled data** within ML datasets, highlighting an interest in systematic data management.
   - Members seek clearer methodologies to enhance the organization of dataset contents, aiming to improve usability in ML projects.



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Experimenting with DPO for Translation Models**: A member inquired about successfully fine-tuning translation models using **DPO**, referencing insights from the [CPO paper](https://arxiv.org/abs/2401.08417). They emphasized that **moderate-sized LLMs** fail to match state-of-the-art performance.
   - *Is anyone achieving better results?* underscores the community's growing interest in fine-tuning techniques.
- **CPO Enhances Translation Outputs**: The **CPO approach** targets weaknesses in supervised fine-tuning by aiming to boost the quality of machine translation outputs. It turns the focus from just *acceptable* translations to higher quality results, improving model performance.
   - By addressing reference data quality, CPO leads to significant enhancements, specifically underutilizing datasets effectively.
- **ALMA-R Proves Competitive**: Applying **CPO** significantly improved **ALMA-R** despite training on only **22K parallel sentences** and **12M parameters**. The model can now rival conventional encoder-decoder architectures.
   - This showcases the potential of optimizing LLMs even with limited data, opening up discussions on efficiency and scaling.
- **NYC Tech Meetup in Late August**: Interest sparked for a tech meetup in NYC during late August, with members expressing their desire to connect in person. This initiative promises to foster deeper networking and collaboration opportunities.
   - The buzz around this potential meetup highlights a sense of community among members eager to share insights and experiences.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **ML Efficiency Boost through Feature Stores**: A **live session** on [Leveraging Feature Stores](https://tinyurl.com/yfjscesh) is scheduled for **July 31st, 2024, at 11:00 AM EDT**, aimed at ML Engineers, Data Scientists, and MLOps professionals.
   - This session will explore **automated pipelines**, tackling unreliable data, and present advanced use cases to enhance **scalability** and **performance**.
- **Addressing Data Consistency Challenges**: The webinar will emphasize the importance of aligning **serving and training data** to create scalable and reproducible ML models.
   - Discussions will highlight common issues like **inconsistent data formats** and feature duplication, aiming to enhance collaboration within ML teams.
- **Enhancing Feature Governance Practices**: Participants will learn effective techniques for implementing **feature governance and versioning**, crucial for managing the ML lifecycle.
   - Attendees can expect insights and practical tools to refine their ML processes and advance operations.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Accelerator Application Deadline Approaches**: The application deadline for the **accelerator program** is fast approaching, offering a **12 week program** with up to **100k in non-diluted funds** for projects.
   - A **demo day** with Mozilla is planned, and members are encouraged to ask their **questions** [here](https://discord.com/channels/1089876418936180786/1245083732319408195).
- **Two More Exciting Events Coming Up**: Reminder about two **upcoming events** this month featuring the work of notable participants, bringing fresh insights to the community.
   - These events are brought to you by two members, further bolstering community engagement.
- **Insightful Zero Shot Tokenizer Transfer Discussion**: A session titled **Zero Shot Tokenizer Transfer** with Benjamin Minixhofer is scheduled, aiming to explore advanced tokenizer implementations.
   - Details and participation links can be found [here](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732).
- **AutoFix: Open Source Issue Fixer Launch**: An announcement was made regarding **AutoFix**, an open source issue fixer that submits PRs from Sentry.io, streamlining developers’ workflows.
   - More information on the project can be accessed [here](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Llama3.1 Paper: A Treasure for Open Source**: The new [Llama3.1 paper from Meta](https://threadreaderapp.com/thread/1815789501026861308) is hailed as incredibly **valuable** for the open source community, prompting discussions about its profound insights.
   - *One member joked* that it contains so much **alpha** that *you have to read it multiple times like a favorite movie*.
- **Training a 405B Model with 15T Tokens**: The paper reveals that the model with **405 billion parameters** was trained using **~15 trillion tokens**, which was predicted by extrapolating their scaling laws.
   - *The scaling law suggests* training a **402B parameter model** on **16.55T tokens** to achieve optimal results.
- **Insights on Network Topology**: It includes a surprisingly detailed description of the **network topology** used for their **24k H100 cluster**.
   - Images shared in the thread illustrate the **architecture**, demonstrating the scale of the infrastructure.
- **Training Interruptions Due to Server Issues**: Two training interruptions during Llama3-405b's process were attributed to the **'Server Chassis'** failing, humorously suggested to be caused by someone's mishap.
   - As a consequence, **148 H100 GPUs** were lost during pre-training due to these failures.
- **Discussion on Hallucination Prevention Benchmarks**: A brief conversation with a Meta engineer raised concerns about the need for better **benchmarks** in **hallucination prevention** techniques.
   - The member shared that *anyone else working on this* should engage in further discussions.



---


The **LLM Perf Enthusiasts AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI Stack Devs (Yoko Li) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1265385753660100608)** (772 messages🔥🔥🔥): 

> - `Unsloth and Llama 3.1 Fine-Tuning`
> - `AI in Software Development`
> - `Image Generation Models`
> - `Mistral Models`
> - `AI Privacy Concerns` 


- **Challenges with Fine-Tuning Llama 3.1**: Several users reported issues while fine-tuning the Llama 3.1 model, particularly with error messages related to model configuration and tokenizer handling.
   - Updates to the transformers library were recommended to resolve some of these issues, and users discussed the importance of ensuring the correct model versions are used.
- **AI in Software Development and Job Security**: Participants discussed the evolving role of AI in software development, highlighting concerns from junior developers about job security as AI tools become more integrated into coding practices.
   - There was a consensus that experienced developers can utilize AI to enhance productivity rather than replace their roles, emphasizing adaptation to new tools.
- **Image Generation and Diversity Issues**: The conversation shifted towards image generation tools, with members reflecting on the challenges of achieving diversity in generated content and the implications of biases in AI models.
   - While some viewed attempts to ensure diversity as commendable, there were critiques regarding the execution of those efforts and their impact on historical context and educational use.
- **Mistral Models and Competition**: Discussion included the new capabilities of Mistral Large 2, noted for its extensive context window and support for multiple languages, posing as a strong alternative to existing large models.
   - Comparisons were made to Llama models, highlighting the competitive landscape in the AI model space and the ongoing efforts for performance improvements.
- **AI Privacy and Data Handling**: Concerns were raised regarding privacy issues related to AI data handling, particularly the implications of human reviewers accessing sensitive data.
   - Participants discussed the necessity of proper data management practices and the perception that some AI tools might be using data in ways that could compromise user privacy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://open.spotify.com/episode/29TW4HAocRcV71kZYCFay8?si=e7SaXoM7S6ODvInF7lTQDA&t=2816">801: Merged LLMs Are Smaller And More Capable, with Arcee AI&#x27;s Mark McQuade and Charles Goddard</a>: Listen to this episode from Super Data Science: ML &amp; AI Podcast with Jon Krohn on Spotify. Merged LLMs are the future, and we’re exploring how with Mark McQuade and Charles Goddard from Arcee AI o...</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/qnamaker/how-to/multi-turn">Multi-turn conversations - QnA Maker - Azure AI services</a>: Use prompts and context to manage the multiple turns, known as multi-turn, for your bot from one question to another. Multi-turn is the ability to have a back-and-forth conversation where the previous...</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://deepmind.google/technologies/imagen-3/">Imagen 3</a>: Imagen 3 is our highest quality text-to-image model, capable of generating images with even better detail, richer lighting and fewer distracting artifacts than our previous models.</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit/tree/main">unsloth/Meta-Llama-3.1-8B-bnb-4bit at main</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-finetuning">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://huggingface.co/datasets">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Instruct-8b-Merged">Replete-AI/Replete-Coder-Instruct-8b-Merged · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets?search=multi%20turn">Hugging Face – The AI community building the future.</a>: no description found</li><li><a href="https://github.com/mixedbread-ai/binary-embeddings/blob/main/mxbai_binary_quantization.ipynb">binary-embeddings/mxbai_binary_quantization.ipynb at main · mixedbread-ai/binary-embeddings</a>: Showcase how mxbai-embed-large-v1 can be used to produce binary embedding. Binary embeddings enabled 32x storage savings and 40x faster retrieval. - mixedbread-ai/binary-embeddings</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=s">Google Colab</a>: no description found</li><li><a href="https://github.com/catcathh/UltraPixel">GitHub - catcathh/UltraPixel: Implementation of UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks</a>: Implementation of UltraPixel: Advancing Ultra-High-Resolution Image Synthesis to New Peaks - catcathh/UltraPixel</li><li><a href="https://x.com/tsarnick/status/1758323312483303443">Tweet from Tsarathustra (@tsarnick)</a>: Sora performance scales with compute</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth Documentation</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/XdrEUyIrgl">Reddit - Dive into anything</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing#scrollTo=LjY75GoYUCB8)">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/docs/transformers/v4.43.2/en/main_classes/pipelines#transformers.TextGenerationPipeline">Pipelines</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu118/torch-2.3.0%2Bcu118-cp310-cp310-linux_x86_64.whl">no title found</a>: no description found</li><li><a href="https://download.pytorch.org/whl/cu118/xformers-0.0.26.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl">no title found</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1265401304189239448)** (1 messages): 

> - `Llama 3.1 Release`
> - `Performance Improvements`
> - `New UI Features`
> - `Google Colab Notebooks`
> - `4-bit Models` 


- **Llama 3.1 Release is Here! 🦙**: Unsloth now supports **Llama 3.1**, making training **2.1x faster** with **60% less memory** used than previous versions. The model has been trained on **15.6T tokens** and expands context lengths to **128K**.
   - Meta's update positions Llama 3.1 as the **most advanced models** yet, supporting new languages and enhanced performance.
- **Google Colab Notebooks for Llama 3.1**: A [Google Colab notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing) is available for finetuning **Llama 3.1 (8B)** on a free Tesla T4, streamlining access for users.
   - Kaggle and Inference UI notebooks were also provided to enhance user interaction, inviting experimentation and testing.
- **New UI Features for Llama 3.1**: Unsloth has introduced a [new inference UI](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing) for interacting with Llama 3.1 Instruct models in Colab.
   - This user-friendly feature is set to elevate the overall experience and engagement with the models.
- **Exciting Experimentation Opportunities!**: The team encourages sharing, testing, and discussing models and results among users, aiming for collaboration and feedback.
   - This community-driven approach is part of a broader push for development within **Unsloth Studio**.
- **Explore 4-bit Models of Llama 3.1**: 4-bit models of Llama 3.1 are available in multiple sizes including [8B](https://huggingface.co/unsloth/Meta-Llama-3.1-8B-bnb-4bit) and [70B](https://huggingface.co/unsloth/Meta-Llama-3.1-70B-bnb-4bit).
   - Model options are tailored for both base and instruct categories, enhancing flexibility for developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: Fine-tune and run Meta&#x27;s updated Llama 3.1 model with 6x longer context lengths via Unsloth!</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)!">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)">Google Colab</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1265473867305189448)** (77 messages🔥🔥): 

> - `Abliterator on LLaMA3.1`
> - `OpenAI API vs Open-Source Models`
> - `Fine-Tuning vs RAG Complexity`
> - `Internal Corp Knowledge`
> - `L3-8B-Stheno-v3.2 Dataset Request` 


- **Discussion on Abliterator and LLaMA3.1**: Members are curious about the effectiveness of *abliterator* on **LLaMA3.1** but no definitive experiences were shared.
   - They expressed a need for success stories regarding this integration.
- **Cost Comparison of OpenAI API and Open-Source Models**: A conversation revolved around the **cost-efficiency** of using OpenAI's *chat API* versus open-source models, emphasizing overhead and hardware expenses.
   - Members noted that using OpenAI API often translates to lower initial costs and less operational risk for startups.
- **Fine-Tuning vs RAG**: It was highlighted that while **fine-tuning** is seen as cheaper and simpler initially, implementing **RAG** demands significant expertise and time investment.
   - Members agreed that RAG needs careful design to avoid complexities and still deliver effective results in production.
- **Importance of Internal Corp Knowledge**: Discussion underlined how models typically lack **internal corporate knowledge**, thus requiring fine-tuning for accuracy in corporate applications.
   - Members emphasized that fine-tuning on specific corporate contexts is crucial to avoid inaccuracies.
- **Request for L3-8B-Stheno-v3.2 Dataset**: A member requested the dataset for **L3-8B-Stheno-v3.2**, expressing disappointment that available datasets contain too much fictional content.
   - Another member noted that few share their datasets nowadays, indicating a trend of limited accessibility.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1265390693002776717)** (147 messages🔥🔥): 

> - `Training in Loop Issues`
> - `Unsloth and Hugging Face Model Loading`
> - `Llama 3.1 Fine-Tuning`
> - `Using FastLanguageModel`
> - `Inference with Fine-Tuned Models` 


- **Training in Loop Causes OOM**: A user reported that using `train()` in a loop causes a VRAM explosion and results in an out-of-memory (OOM) error after the first training iteration.
   - They mentioned having gradient checkpointing enabled and are troubleshooting by checking configurations.
- **Loading Models with Unsloth**: A user encountered an OSError when trying to load the model 'unsloth/meta-llama-3.1-8b-bnb-4bit', which suggests double-checking the model path and ensuring the local directory does not conflict.
   - For loading local model files, users discussed using specific paths to direct loading instead of pulling from Hugging Face.
- **Fine-Tuning Llama 3.1 Issues**: Some users noted issues while fine-tuning Llama 3.1 with various dataset formats, questioning if the prompt formats affected their results.
   - Additionally, there was guidance about using appropriate training configurations to ensure expected losses during the fine-tune.
- **FastLanguageModel Utilization**: It was confirmed that using the FastLanguageModel is mandatory to achieve the claimed speed improvements for inference in Unsloath.
   - Users were interested in how to effectively implement this model within VLLM for faster performance.
- **Inference with Fine-Tuned Models**: A user successfully fine-tuned a model and pushed it to Hugging Face but sought advice on how to effectively run inference on it.
   - Recommendations included using Unsloath's inference code or VLLM to streamline the deployment process for testing in production.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/">llama-tokenizer-js playground</a>: no description found</li><li><a href="http://www.unsloth.ai">Unsloth AI | Finetune Llama 3 &amp; Mistral LLMs</a>: Easy finetuning for AI and LLMs. Open-source and for beginners. Get faster with Unsloth. </li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining#loading-lora-adapters-for-continued-finetuning">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.unsloth.ai/get-started/installation/updating">Updating | Unsloth Documentation</a>: To update Unsloth, follow the steps below:</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git#egg=unsloth[colab-new]">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: no description found</li><li><a href="https://github.com/vllm-project/vllm/issues/6689">[Model] Meta Llama 3.1 Know Issues &amp; FAQ · Issue #6689 · vllm-project/vllm</a>: Please checkout Announcing Llama 3.1 Support in vLLM Chunked prefill is turned on for all Llama 3.1 models. However, it is currently incompatible with prefix caching, sliding window, and multi-lora...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1265386782464282747)** (17 messages🔥): 

> - `LLaMa-3.1 for synthetic datasets`
> - `Use of attention masks in Vision Language Models`
> - `Inference speed vs training speed`
> - `Decoding with different model sizes` 


- **LLaMa-3.1 for Synthetic Data Generation**: Members discussed leveraging **LLaMa-3.1** for generating synthetic datasets, but many agreed that utilizing the **405B model** is ideal for this purpose.
   - *One member noted, 
- **Clarifying Attention Masks in Vision Language Models**: A member described using an attention mask with 48 patches, stating the mask incorporates both sentence and patch masks effectively.
   - They specified that with a decoder-only setup, the attention mask for the image patches should align with the sentence tokens.
- **Interference Slower than Training Models**: A member raised a question regarding why interference is significantly slower than training, noting a stark contrast in data processing rates.
   - While training can handle hundreds of data points per minute, interference typically processes only **30-100 tokens/s**.
- **Using 8B Model for Data Formatting**: Discussion revealed that **the 8B model** could potentially be used for formatting data or fine-tuning, though this wasn't the main focus of synthetic data generation.
   - Members acknowledged the primary goal of synthesis is better served by larger models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.
</li>
</ul>

</div>
  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1265406391074427004)** (192 messages🔥🔥): 

> - `LM Studio and Llama 3.1`
> - `Nemo models performance`
> - `Model download issues`
> - `Claude Sonnet 3.5 as coding model`
> - `GPU usage for model inference` 


- **LM Studio's Compatibility with Llama 3.1**: Users have discussed LM Studio's inability to run Llama 3.1 on OpenCL cards and recommended upgrading to version 0.2.28 from the official website for better support.
   - Several members confirmed the latest updates from LM Studio are crucial for running large models effectively.
- **Nemo Models and Performance Issues**: The Nemo models are reportedly functional on current versions, but users faced challenges with context length and slower outputs due to limited RAM.
   - One user confirmed success with specific setups while others suggested improvements and optimizations.
- **Download and Access Problems for Models**: Some users experienced issues downloading models from Hugging Face, with reports ranging from regional CDN problems to browser caching issues.
   - Others confirmed that specific links were accessible while others encountered 'Entry not found' errors.
- **Claude Sonnet 3.5 as Benchmark for Coding Tasks**: Users expressed that current local models do not match the coding capabilities of Claude Sonnet 3.5, particularly when generating full working code.
   - Exploration of alternatives and experimentation with lower quantization Claudes was suggested as a potential solution.
- **Importance of GPU Offloading for AI Models**: Discussions highlighted that using CPUs for inference leads to slower outputs compared to GPUs, emphasizing the need for suitable models that fit into GPU VRAM.
   - Users were encouraged to seek out models labeled for 'full GPU offload' to maximize performance and reduce inference times.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pastebin.com/uW5fJtLF">🔎 Read the user prompt carefully, attention to detail👣 Think step by step wh - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md#amd-rocm">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://tenor.com/view/mcmahon-crying-he-was-special-wwe-vince-mcmahon-gif-13313547165599993551">Mcmahon Crying He Was Special GIF - Mcmahon Crying He was special WWE - Discover &amp; Share GIFs</a>: Click to view the GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1265385202373628015)** (89 messages🔥🔥): 

> - `Llama 3.1 Model Performance`
> - `Model Censorship and Behavior`
> - `Mistral Large 2 Release`
> - `Testing and Troubleshooting Models`
> - `Model Naming Trends` 


- **Llama 3.1 struggles with reasoning and coding**: Users noted that **Llama 3.1 8B** is not great for reasoning and coding tasks, with some members expressing skepticism regarding its overall performance.
   - Comparisons suggest that while it's decent for its size, its logic capabilities appear lacking, especially contrasted with models like **Gemma 2**.
- **Concerns about Censorship in Top Models**: Discussions about model censorship revealed that models performing well may often be less censored, but this was debated among members.
   - A member suggested that the **censor** labeling is indicative of an attempt to manage column width rather than a straightforward description of model behavior.
- **Mistral Large 2 boasts significant improvements**: The release of **Mistral Large 2** with a 128k context window promises an improvement in performance and efficiency across numerous languages and coding tasks.
   - This model's design aims for single-node inference, boasting 123 billion parameters, offering opportunities for innovative AI applications.
- **Troubleshooting with LLMs and Flash Attention**: Users reported issues with loading models like **Llama 3.1** and recommended checking configurations related to **Flash Attention**, which can impact model behavior.
   - Many experienced varying performance based on whether models were loaded fresh or with different configurations.
- **Trends in Alien Naming among AI Models**: A humorous thread explored the phenomenon of AIs using names like **Zorvath** and **Elara** for characters, wondering where these patterns originated.
   - A member noted that literary influences might skew naming conventions, with some names appearing more frequently in certain genres and styles.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/1110598183144399058/1263234070813479063/1263234070813479063">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://embed.wattpad.com/story/372087683-the-cosmic-union-dreamcatchers">Embed - The Cosmic Union: Dreamcatchers  - Wattpad</a>: no description found</li><li><a href="https://x.com/YouJiacheng/status/1815817670954213710">Tweet from YouJiacheng (@YouJiacheng)</a>: just saw that deepseek-coder will get an upgrade at July 24 10:00 UTC+8.</li><li><a href="https://github.com/THUDM/CodeGeeX4">GitHub - THUDM/CodeGeeX4: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;A and much more.</a>: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;amp;A and much more. - ...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650">Feature Request: Proper Llama 3.1 Support in llama.cpp · Issue #8650 · ggerganov/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650#issuecomment-2246438902">Feature Request: Proper Llama 3.1 Support in llama.cpp · Issue #8650 · ggerganov/llama.cpp</a>: Prerequisites I am running the latest code. Mention the version if possible as well. I carefully followed the README.md. I searched using keywords relevant to my issue to make sure that I am creati...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1265478565152161812)** (9 messages🔥): 

> - `Msty Features`
> - `LM Studio Server Confusion`
> - `Model Migration Concerns`
> - `GPU Configuration in LM Studio` 


- **Msty offers compelling features over LM Studio**: While using Msty for connecting to LM Studio from another device, a user highlighted its ability to update the **Ollama version** without a full app upgrade, a feature they wish for in LM Studio.
   - They expressed irritation that **LM Studio** does not support endpoint usage and is limited to local inference only, making Msty a more practical choice.
- **LM Studio's server functionality debate**: Despite LM Studio advertising a server feature, users feel it lacks client capabilities, requiring additional software like Msty for effective connection between devices.
   - This raises frustration as users suggest that having two apps for the same function feels redundant, highlighting Msty's dual server and client role.
- **Concerns about migrating models to Ollama**: A user mentioned their reluctance to switch from LM Studio to Msty due to the **pain of migration**, specifically in transferring models to the Ollama backend.
   - They preferred the model management method of LM Studio over the methods used by Ollama, which they find cumbersome.
- **GPU configuration in LM Studio**: A discussion pointed out that proper configuration for GPU load distribution in LM Studio is not intuitive and requires digging into settings.
   - Users can find advanced GPU options under AI Chat > New chat, allowing them to toggle settings for maximum acceleration.


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1265392130583560234)** (11 messages🔥): 

> - `Llama 3.1 presets`
> - `GPU settings for models`
> - `Context length for Llama 3.1` 


- **Llama 3.1 lacks visibility on presets**: A user expressed frustration, stating they couldn't find any **presets** for **Llama 3.1**, indicating they were new to this environment.
   - Another member suggested that the **Llama 3 v2 preset** works with an update to **v0.2.28**.
- **Optimal context length for 3080ti**: A question was raised about whether to leave the **context length** at **2048** for a **3070ti** GPU.
   - A recommendation was made to set it to **32k**, as the model supports up to **128k context**.
- **Issues with GPU loading**: A user reported issues with model loading, stating it didn't seem to load fully onto their GPU.
   - They attempted to set the memory to max (-1) but found it reverted back after reloading.


  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1265460884034031689)** (35 messages🔥): 

> - `OpenCL Deprecation`
> - `GPU Comparisons for Streaming`
> - `Fine-Tuning LLaMA 3.1`
> - `Low-Budget GPUs for LLMs`
> - `Tech Shopping in Taiwan` 


- **OpenCL Heads to Obsolescence**: OpenCL is now deprecated, but users can still utilize it until more complete transitions to Vulkan occur in LM Studio.
   - For now, its functionalities remain usable, even if not recommended for the long term.
- **RTX 4090 vs RX 7900 XTX Debate**: A member queried the benefits of using **1x RTX 4090** versus **2x RX 7900 XTX** for streaming and local AI applications with the upcoming **Ryzen 9950x**.
   - It's suggested that while the RX 7900 XTX may offer an edge in VRAM, compatibility with AI models could favor the more widely supported Nvidia options.
- **Fine-Tuning LLaMA 3.1 VRAM Needs**: The community discussed the VRAM requirements for fine-tuning the **LLaMA 3.1 8B** model, estimating it can be done with **32GB VRAM**.
   - Context length was debated, with the belief that **27k tokens** could be aimed for with the mentioned VRAM capacity.
- **Budget-Friendly GPU Options for LLMs**: Members exchanged opinions on affordable GPUs for local LLMs, recommending the **RTX 3060 12GB** as a viable option over older AMD models.
   - A suggestion came to consider Chinese-modified **RTX 2080 Ti** models with **22GB VRAM** as a risky but potentially more powerful alternative.
- **Exploring Tech Opportunities in Taiwan**: A member expressed interest in visiting tech malls in Taiwan, inspired by notable videos showcasing impressive tech shopping experiences.
   - They planned to look out for bargain NVME drives during their stay but noted that prices aren’t vastly different from online options.



**Link mentioned**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/15hiid1/how_long_does_finetuning_take_and_how_much_vram/">Reddit - Dive into anything</a>: no description found

  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1265383067401588736)** (87 messages🔥🔥): 

> - `Beta Release Issues`
> - `Interface Changes and Feedback`
> - `GPU Offloading Problems`
> - `Model Loading Concerns`
> - `Version Confusion` 


- **Beta Release Issues on Windows**: Users are experiencing issues with the beta not launching properly on Windows 10, with some unable to see the UI despite seeing processes in Task Manager.
   - Known issues include the beta not starting correctly; recommended actions include restarting the app multiple times or waiting for future updates.
- **Positive Feedback on New Interface**: Many users shared their positive experiences with the new UI in Beta 1, noting features like Discovery and Collections as particularly useful.
   - Concerns were raised about the structure of folders and settings but overall users appreciate the changes being made.
- **GPU Offloading Not Working**: Several users reported that GPU offloading isn't functioning as expected, especially on M3 Max and with a 4080S, often needing to rely on manual settings.
   - The automatic GPU setting is criticism, as it often leads to gibberish outputs while manual settings appear more reliable.
- **Model Loading Issues**: Users faced challenges when trying to load models like bic llama, with some settings requiring adjustment to prevent crashes or RAM overloads.
   - It's suggested to utilize the Developer button to load models into RAM effectively, bypassing integration with GPU.
- **Version Confusion and Update Management**: There is confusion regarding the versions of LM Studio, with users unsure of the latest release due to outdated links in channels.
   - Calls were made for clearer communication regarding version updates, suggesting a need for improved organization of release information.


  

---


### **LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1265709624120840325)** (1 messages): 

> - `LangGraph tool binding`
> - `LLM limitations`
> - `LangChain integration issues` 


- **Issues with LLM tool binding in LangGraph**: A user is encountering an error when attempting to use `llm_with_tools = llm.bind_tools(tools)` in their code with LangGraph.
   - They are questioning whether the error arises from **LM Studio** not supporting tool calling or due to the **LLM** being utilized.
- **Potential causes of tool binding issues**: The discussion highlights possible reasons for the malfunction, focusing on **LLM** compatibility issues.
   - It remains unclear if the specific **LLM** currently in use is supported by the requested **tool binding** functionality.


  

---


### **LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1265714921103032352)** (2 messages): 

> - `Krypt Lynx Installation`
> - `Pip Install Success` 


- **Successful Installation of Krypt Lynx**: A member expressed interest in trying out the **Krypt Lynx** project and inquired about installation on **Windows**.
   - They later updated that using **pip install** actually worked for them, showcasing a positive installation experience.
- **Installation Inquiry on Windows**: A member initiated a discussion asking how to install the **Krypt Lynx** project specifically on the **Windows** platform.
   - This inquiry led to a clarification that the installation succeeded after attempting **pip install**, which was met with some relief.


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1265431270146707576)** (33 messages🔥): 

> - `ROCm 0.2.28 performance issues`
> - `Llama 3.1 compatibility`
> - `LM Studio update process`
> - `OpenELM support`
> - `AppImage functionality` 


- **ROCm 0.2.28 exhibits slower performance**: After updating to ROCm **0.2.28**, a user reported significantly slower inference performance on their 2x 7900xt system, seeing only **150w** usage from one card vs **300w** before.
   - They downgraded to **0.2.27** and found performance normal, prompting a request for investigation into the changes made for **0.2.28**.
- **Llama 3.1 struggles on AMD cards**: Multiple users discussed issues with getting **Llama 3.1** to work on AMD cards, citing a tokenizer error when using **llama.cpp**.
   - One user discovered the problem stemmed from using **OpenCL** instead of ROCM, while another reported their struggles with layer visibility.
- **Update process for LM Studio**: A user inquired about the update command for **0.2.28**, suggesting that current instructions might still refer to the previous version.
   - It was clarified that the update process for this build has reverted for simplicity, and a user noted the difficulty in finding version details via Discord.
- **Interest in OpenELM support**: There was curiosity about the potential support for **OpenELM**, with one user wanting to try Apple’s models and pointing out a recent relevant GitHub pull request.
   - The response indicated that all model support depends on **llama.cpp**.
- **AppImage works seamlessly**: A user confirmed that after downloading the **0.2.28** AppImage for Linux, they could get a **Llama 3.1** model working out of the box on their **7800XT**.
   - This satisfied the requirements for running Llama 3.1 with ROCm, demonstrating compatibility.



**Link mentioned**: <a href="https://github.com/ggerganov/llama.cpp/pull/7359">OpenELM support by icecream95 · Pull Request #7359 · ggerganov/llama.cpp</a>: Fixes: #6868. Thanks to @joshcarp for an initial try at doing this (#6986), it was very helpful as a source to copy-paste from and check against. Currently a bunch of the configuration is hardcoded...

  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1265536160571064421)** (7 messages): 

> - `Meta-Llama 3.1 70B`
> - `Tokenizer Bug Fix` 


- **Meta-Llama 3.1 70B Now Available**: A member announced that **70B quants** for **Meta-Llama 3.1** are available, linking to the [repository](https://huggingface.co/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF).
   - The excitement in the channel was palpable, with others commenting, *that's why he's the goat*.
- **Tokenizer Bug Request for Re-upload**: It was mentioned that the models will be re-uploaded to fix a **tokenizer bug**, which is expected to improve performance.
   - For now, performance is reported as *fine*, with an expectation of better results post-update.


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1265684901928239115)** (11 messages🔥): 

> - `LM Studio compatibility`
> - `AVX2 and AVX-512 instructions`
> - `Koboldcpp vs LM Studio`
> - `Model downloading alternatives` 


- **LM Studio's requirements and compatibility issues**: A user reported issues running **LM Studio** on a **Windows Server 2012 R2** setup due to unknown compatibility with the **kernel32.dll**. Another user confirmed that **LM Studio** won't install without **AVX2** instructions, which the current CPU lacks.
   - *Koboldcpp* runs fine on **AVX512**, but the user prefers *LM Studio* for its interface.
- **Understanding AVX2 and AVX-512 Differences**: There was a confusion regarding the use of **AVX2** and **AVX-512** instructions, with a member thinking that bigger instructions might be better. It was clarified that the **Xeon 6138** does not support **AVX2**, making **LM Studio** incompatible regardless of additional errors.
   - A user expressed gratitude for the clarification, further mentioning that understanding the instruction types could be helpful for future usage.
- **Exploring Alternatives with llama.cpp**: Another user mentioned that while **LM Studio** has compatibility issues, there's a way to build **llama.cpp** with **AVX-512** support. This offers an alternative for model downloading and inference through console use or a server endpoint.
   - Users were directed to check **lms-comm** or **bartowski** HF pages for available models as potential substitutes.



**Link mentioned**: <a href="https://github.com/ggerganov/llama.cpp/issues/160">Add avx-512 support? · Issue #160 · ggerganov/llama.cpp</a>: No clue but I think it may work faster

  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1265397562408435742)** (1 messages): 

> - `Llama 3.1 405B`
> - `Open source models` 


- **Llama 3.1 405B launches on Perplexity**: The **Llama 3.1 405B** model, touted as the most capable open source model, is now available on Perplexity and rivals **GPT-4o** and **Claude Sonnet 3.5**.
   - The team is actively working on integrating Llama 3.1 405B into their **mobile apps**, prompting users to stay tuned for updates.
- **Upcoming mobile app integration**: The Perplexity team announced plans to add **Llama 3.1 405B** functionality to their **mobile applications** next.
   - Users are encouraged to stay tuned for more updates on this integration.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1265390147072032882)** (306 messages🔥🔥): 

> - `Llama 405b performance`
> - `Mistral Large 2`
> - `AI model comparisons`
> - `TikTok as a search engine`
> - `Language symbol output issues` 


- **Discussions on Llama 405b performance**: Members expressed skepticism regarding Llama 405b's performance, stating it seems average compared to other models like Mistral and Sonnet.
   - Some noted inconsistencies in benchmarking results across various models, impacting their decisions.
- **Praise for Mistral Large 2**: Mistral Large 2 received attention as a model that might outperform Llama 405b in terms of results, with users expressing a preference for it over Llama.
   - Users are hopeful for Mistral's addition to Perplexity alongside existing models.
- **Confusion over AI model benchmarks**: The validity of AI benchmarks was questioned, with members comparing them to movie ratings, highlighting their inconsistency.
   - Users noted that subjective experiences with models differ significantly, making it difficult to rely solely on benchmarks.
- **Potential for TikTok as a search tool**: Members discussed TikTok's emerging role as a search engine for Gen Z, with debates on its value compared to traditional search methods.
   - Concerns were raised about the reliability of health advice found on TikTok and the implications of using such platforms for information.
- **Issues with language symbol outputs**: Users reported Llama models having trouble outputting Asian language symbols correctly, noting this as a limitation in performance.
   - It was suggested that the model's reluctance to use symbols may stem from its training and handling of multilingual inputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/minchoi/status/1815812112796565690">Tweet from Min Choi (@minchoi)</a>: Instant Intelligence is wild with Llama 3.1 8B + Groq 🤯 </li><li><a href="https://www.perplexity.ai/search/write-short-poem-in-czech-abou-ksVKF84qQNG2wH8Q7ia2NQ">write short poem in Czech about smažák</a>: Smažák, zlatý a křupavý, Vonící olejem, chutí nebeskou. Křehký a jemný, jako sníh, Smažák, můj žaludek, potěšíš.  Translation: Fried cheese, golden and...</li><li><a href="https://www.perplexity.ai/search/write-short-magical-poem-in-cz-v7lRjXgxS5.tDX22.g4F2Q">write short magical poem in czech about beautiful flying hot dog</a>: Here&#x27;s a short magical poem in Czech about a beautiful flying hot dog:  &quot;Létající hot dog, krásný a zlatý, S klouboučkem hořčice, tančí v oblakách. Svírá se v...</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1eb9njj/claude_35_vs_llama_405b_vs_others_tested_by_ai/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/history/3c76952a-c352-4520-95a2-ccf1a7b2b056?share=true">_Think-Lab Revised</a>: Use the power of ScratchPad-Think for every day web searches. Export refined search queries in JSON format. The scratchpad is a powerful tool that helps you maintain coherence and accuracy, especially...</li><li><a href="https://github.com/nuprl/MultiPL-E">GitHub - nuprl/MultiPL-E: A multi-programming language benchmark for LLMs</a>: A multi-programming language benchmark for LLMs. Contribute to nuprl/MultiPL-E development by creating an account on GitHub.</li><li><a href="https://app.wordware.ai/share/8c523d8b-c109-4189-a6ce-cc9bfc5d24a2/history/129be5a3-d4ae-4069-85f5-156052669490?share=true">Sonnet Insight 3.5 - Rank Model Outputs </a>: This prompt processes a question using Sonnet 3.5, Gemini 1.5 Pro, llama 3.1 70B&amp;405B, GPT-4o/mini, Sonar Large (online model), Claude 3 Opus, Claude 3 Sonnet, and lastly Claude 3 Haiku. The app t...</li><li><a href="https://scale.com/leaderboard">SEAL leaderboards</a>: no description found</li><li><a href="https://x.com/rypearts/status/1815868829169328349?s=61">Tweet from Ryan Putnam (@RypeArts)</a>: ✧ 　 　 ✧ ˚ * 　 　.　 　　　　 　　 · · 　　 　 + ✧ 　　　 · 　 · ˚ . 𝓈𝓊𝓂𝓂ℯ𝓇 𝓋𝒾𝒷ℯ𝓈</li><li><a href="https://www.euronews.com/next/2023/02/05/gen-z-is-using-tiktok-as-a-search-engine-is-this-the-end-of-google">Is TikTok about to replace Google as the top search engine?</a>: An increased number of searches on TikTok has raised the question of whether Google will soon become obsolete.</li><li><a href="https://www.perplexity.ai/search/10-animals-in-kanji-only-respo-CdygXAlIQte9iJXH.fXGwg">10 animals in kanji only. respond only with
japanese symbols</a>: *   河馬 *   山羊 *   栗鼠 *   獅子 *   大猩々 *   麒麟 *   長尾驢 *   子守熊 *   駱駝 *   土竜
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1265395723508383927)** (13 messages🔥): 

> - `Mistral Large 2`
> - `President Biden's Public Appearances`
> - `AI Monitoring at AEON`
> - `Oldest Trees in the World`
> - `Meta's Llama 3.1` 


- **Mistral Large 2 sets new AI standards**: On July 24, 2024, Mistral AI released Mistral Large 2, featuring **123 billion parameters** and a **128,000-token context window**, enhancing capabilities in code generation and mathematics.
   - The model reportedly outperforms **Llama 3.1 405B** and closely matches **GPT-4** in mathematical tasks.
- **President Biden's last public appearance**: President Joe Biden was last seen in public on **July 17, 2024**, after testing positive for COVID-19 while campaigning in Las Vegas.
   - This marked his final appearance before withdrawing from the presidential race on **July 23**.
- **AI system monitors smiles at AEON stores**: Japanese supermarket chain AEON has implemented an AI system named **'Mr Smile'** to standardize employee smiles based on over 450 behavioral elements.
   - The **trial** in eight stores reportedly improved service attitudes by **1.6 times** over three months.
- **The world's oldest trees compilation**: Research highlights trees like the Great Basin bristlecone pine, known for its age of nearly **5,000 years**, as the oldest non-clonal tree species.
   - A variety of methods including tree-ring counting and radiocarbon dating help determine the ages of these ancient trees.
- **Meta's Llama 3.1 launch**: Meta's recent release of **Llama 3.1 405B** offers a competitive open-source model with a focus on challenging existing proprietary AIs like **GPT-4**.
   - Boasting **405 billion parameters**, it promises unprecedented access to advanced AI capabilities for developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/mistral-large-2-revolutionizin-sVfT0LnmTJ2ER3WS5YqILQ#1">Mistral Large 2: Revolutionizing Language Models with Unprecedented Capabilities</a>: Here&#x27;s my review of the output using the scratchpad format:  &lt;scratchpad&gt; [Key information extracted from the prompt] Review task for an article about Mistral...</li><li><a href="https://www.perplexity.ai/page/japans-stores-use-ai-to-track-tIYXMbASQbCClx5sHwrn5g">Japan&#x27;s Stores Use AI to Track Smiles</a>: Japanese supermarket chain AEON has introduced an artificial intelligence system called &quot;Mr Smile&quot; to assess and standardize employee smiles, sparking debate...</li><li><a href="https://www.perplexity.ai/page/why-didn-t-jack-join-rose-USOLbt8USUqW1eB.kg3pYg">Why Didn&#x27;t Jack Join Rose?</a>: The iconic scene from James Cameron&#x27;s &quot;Titanic&quot; where Jack sacrifices himself to save Rose has sparked decades of debate among fans and experts alike. While...</li><li><a href="https://www.perplexity.ai/search/combien-de-litre-d-eau-il-y-a-ght8Kb8OTS2HcZ0OxNY9jw#0">Combien de litre d&#x27;eau il y a sur terre ?</a>: La Terre contient environ 1,386 milliard de kilomètres cubes d&#x27;eau (1,386 \times 10^9 \, km^3), ce qui représente 1,386 trillion de litres d&#x27;eau (1,386 \times...</li><li><a href="https://www.perplexity.ai/search/the-oldest-tree-in-the-world-o_vxC.JmTnSeKWNhsi_t0Q">The oldest tree in the world</a>: The oldest known individual tree in the world is a Great Basin bristlecone pine named &quot;Methuselah,&quot; located in the White Mountains of California. This tree is...</li><li><a href="https://www.perplexity.ai/search/cual-sera-el-proximonpresident-mqJrnaaAShyafjRFgXH7wQ">cual sera el proximonpresidente de ee.uu?</a>: La carrera presidencial de Estados Unidos para las elecciones de noviembre de 2024 ha tomado un giro significativo con la reciente decisión del presidente Joe...</li><li><a href="https://www.perplexity.ai/search/when-was-president-biden-last-_.41EsmjTk2glkhKNbSp9A">when was president Biden last seen in public</a>: Based on the search results provided, President Joe Biden was last seen in public on Wednesday, July 17, 2024. Specifically:  1. Biden was last seen exiting...</li><li><a href="https://www.perplexity.ai/search/when-you-have-pots-does-your-h-jmae9BaDQQqK0IA0VGvoQw">When you have POTS does your heart feel like it’s beating harder at times</a>: Yes, when you have Postural Orthostatic Tachycardia Syndrome (POTS), your heart can feel like it’s beating harder at times. This is because POTS causes your...</li><li><a href="https://www.perplexity.ai/page/legal-trials-of-inanimate-obje-AGyEpycyQ6qVdMxEBIqUsg">Legal Trials of Inanimate Objects</a>: Throughout history, legal systems have grappled with the unusual practice of putting inanimate objects on trial for causing harm or death to humans. From...</li><li><a href="https://www.perplexity.ai/page/meta-releases-llama-3-1-405b-pFAuGE4GR_id4.zHbNDqyQ">Meta releases Llama 3.1 405B</a>: Meta&#x27;s release of Llama 3.1 405B marks a significant milestone in the AI landscape, introducing a powerful open-source model that rivals proprietary giants...</li><li><a href="https://youtu.be/1--eJwi-xQo?si=7rl6NsxQPluT_WR7">Meta&#39;s Lama 3.1, Wiz&#39;s Bold Rejection, XAI Memphis Supercluster, Cocaine Sharks, and Space Debris...</a>: Ever wondered how a single AI model can reshape the landscape of technology? Discover the secrets behind Meta&#39;s Lama 3.1, an AI marvel with 405 billion param...</li><li><a href="https://www.perplexity.ai/page/mistral-large-2-revolutionizin-kUXugCSjRAevYdq7_cnYkA">Mistral Large 2: Revolutionizing AI</a>: On July 24, 2024, Mistral AI unveiled Mistral Large 2, a powerful new language model boasting 123 billion parameters and a 128,000-token context window,...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1265401892700426363)** (8 messages🔥): 

> - `Llama 3 405b API Plans`
> - `Context Size of Llama 3 405b`
> - `Passing return_citations in Langchain`
> - `NextCloud Integration with OpenAI`
> - `Microsoft Copilot Studio Perplexity Connector` 


- **Plans for Llama 3 405b in API**: A member confirmed that **Llama 3 405b** will be served in the API soon, indicating an upcoming feature release.
   - This response generated excitement as it promises new capabilities for users.
- **Llama 3 405b context size discussion**: A member inquired about the context size of **Llama 3 405b**, suggesting it might be **128K** as fine-tuning is unnecessary.
   - They asserted that this feature could lower costs compared to existing models like Claude and GPT.
- **Return citations in Langchain LLM chain**: A user sought guidance on how to pass the **return_citations** value to an LLM chain using **Langchain** with Perplexity Chat.
   - No concrete solutions were shared in the discussion, indicating a need for further exploration.
- **NextCloud's OpenAI Integration Queries**: A member shared a link to **NextCloud** integration with OpenAI, celebrating its community-driven, free, and open-source nature.
   - They referenced the **GitHub repository** for those interested in the integration details.
- **Issues with Microsoft Copilot Studio Connector**: A user raised a concern regarding an **unspecified error** when uploading the Perplexity Connector to **Microsoft Teams**.
   - The community's response indicates that troubleshooting may be necessary to resolve this issue.



**Link mentioned**: <a href="https://github.com/nextcloud">Nextcloud</a>: 📱☁️💻 A safe home for all your data – community-driven, free &amp; open source 👏 - Nextcloud

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1265393328862527591)** (298 messages🔥🔥): 

> - `Model Capabilities`
> - `GPU Servers for AI Models`
> - `Kling AI Image to Video Generation`
> - `LLM Compatibility with Raspberry Pi`
> - `Prompt Libraries for Custom Models` 


- **Mistral Models and Their Specifications**: Users discussed the specifications of the **Mistral-7B-v0.3** model, noting its improved capabilities with 128k context windows and support for multiple languages.
   - It was mentioned that the **Mistral Large** model is 69GB and can run efficiently using **ollama**.
- **Exploring GPU Server Options**: Users highlighted GPU server options for running large models, suggesting **Runpod** as an affordable instance at **$0.30/hour**.
   - One suggested using **LM Studio** or **ollama** for better performance and compatibility with specific models.
- **Kling AI's Image-to-Video Features**: **Kling AI** was noted for its impressive capabilities in generating moving images from still photos, though users reported some limitations in video quality.
   - Despite the fun and engaging results, there were comments on server overload leading to longer generation times.
- **LLM Compatibility and Performance on Raspberry Pi**: The discussion shifted to the feasibility of running large language models (LLMs) on **Raspberry Pi 4B**, with users unsure of the performance capabilities.
   - It was mentioned that models with varying RAM configurations could be run, potentially including 7B models using **ollama**.
- **Prompt Library Access and Custom Models**: Users inquired about accessing prompt libraries to create custom prompts for AI models, pointing to available channels for assistance.
   - The conversation emphasized the need for specific model files and frameworks to successfully utilize certain large model capabilities.



**Link mentioned**: <a href="https://huggingface.co/mistralai/Mistral-7B-v0.3">mistralai/Mistral-7B-v0.3 · Hugging Face</a>: no description found

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1265645303868948530)** (9 messages🔥): 

> - `Memory Feature Issues in EU`
> - `Spelling Errors in Mini`
> - `Python PDF Generation with OpenAI`
> - `Debugging Model Output`
> - `User Feedback on Model Mistakes` 


- **Memory feature appears and disappears in EU**: A member reported that they received the **memory feature** for only five minutes, prompting others to confirm similar experiences.
   - Another user humorously remarked on this inconsistency, suggesting discussions on whether the feature is fully operational.
- **Mini's frequent spelling error of 'composure'**: One member pointed out that **Mini consistently misspells** 'composure' as 'composposure' in their prompts.
   - Another member could not replicate this issue and shared a link to their prompt highlighting a correctly spelled 'composure'.
- **Using OpenAI to generate PDF content in Python**: A user inquired about generating PDFs using Python and OpenAI, expressing the need for table contents and section descriptions based on uploaded files.
   - This initiated a conversation about workflows and techniques for leveraging OpenAI for document generation.
- **Debugging model output for spelling mistakes**: A member noted frequent spelling mistakes, including 'itis' for 'it is', and plans to enable debugging to inspect the model's prompts.
   - This prompted another member to suggest sharing specific examples to better understand the model's output tendencies.
- **User experiences mispelled words in conversations**: A user shared that they could provoke misspellings and spacing issues only by expressly requesting them and providing feedback.
   - They pointed out a shared link exhibiting their interactions which highlighted this phenomenon.


  

---



### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1265447785692663911)** (4 messages): 

> - `LLM Distillation`
> - `LLaMa 3`
> - `Common RAG Challenges` 


- **Recent Papers on LLM Distillation**: A member suggested checking out the [Minitron GitHub repository](https://github.com/NVlabs/Minitron), which details a family of compressed models obtained via **pruning** and **knowledge distillation**.
   - This repository may provide insights into recent advancements in LLM distillation similar to models like **Sonnet**, **Llama**, and **GPT-4Omini**.
- **Introduction of LLaMa 3 Models**: A new set of foundation models called **LLaMa 3** was introduced, featuring a dense Transformer with **405B parameters** and a context window of up to **128K tokens**.
   - These models, which excel in multilinguality, coding, reasoning, and tool usage, are set to enhance a broad range of AI applications.
- **Common RAG Challenges in Production**: A member shared a link to a LinkedIn post discussing common RAG (Retrieval-Augmented Generation) challenges and potential solutions.
   - The post highlights various issues faced when implementing RAG in production environments that practitioners may find useful.



**Link mentioned**: <a href="https://github.com/NVlabs/Minitron">GitHub - NVlabs/Minitron: A family of compressed models obtained via pruning and knowledge distillation</a>: A family of compressed models obtained via pruning and knowledge distillation - NVlabs/Minitron

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1265386240350621707)** (2 messages): 

> - `PC Agent Demo`
> - `Proprietary Tools` 


- **Exciting PC Agent Demo Released**: A member shared a [YouTube video titled "PC Agent Demo"](https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8) showcasing a new agent from gate-app.com/research/pc-agent.
   - This demo highlights the functionalities and potential applications of the PC Agent tool.
- **Discussion on Proprietary Tools**: A member suggested a potential connection to proprietary tools related to the earlier topics discussed in the channel.
   - This discussion prompted engagement from other members, contemplating the implications and applications of such tools.



**Link mentioned**: <a href="https://youtu.be/97tUynaJusY?si=pi-K8F4trJDE3Kt8">PC Agent Demo</a>: gate-app.com/research/pc-agent

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1265387248145272932)** (20 messages🔥): 

> - `Meta Llama 3.1 capabilities`
> - `Synthetic dataset creation`
> - `Microsoft GraphRAG`
> - `Aider's repo map`
> - `Wordware apps` 


- **Meta Llama 3.1 excels in multilingual tasks**: The **Meta Llama 3.1** collection includes pretrained models in sizes of **8B, 70B, and 405B**, optimized for multilingual dialogue and outperforming many existing chat models.
   - With fine-tuning options available, it supports **synthetic dataset creation** and offers a **Community License** for commercial and research use cases.
- **Discussion on synthetic datasets potential**: Questions arose about whether **mass production of synthetic datasets** will begin at NousResearch now that **GPT-4** is accessible to them.
   - Despite this enthusiasm, concerns were raised regarding the higher costs of using the **405B model** for generation compared to **Sonnet 3.5 or GPT-4o**.
- **Microsoft introduces GraphRAG**: Microsoft unveiled **GraphRAG**, enhancing LLM capabilities to solve problems with unseen data by creating knowledge graphs from existing datasets.
   - This approach promises improved semantic clustering and concept identification, making the **RAG technique** a significant tool for data investigation.
- **Aider's repo map limitations**: While **Aider** impressively maps a code repository, its architecture has limitations that made it less effective for semantic understanding of codebases.
   - Current methods focus on **entity frequency weights** rather than truly understanding evolution, raising questions about advanced retrieval alternatives.
- **Wordware apps feature JWST images**: Wordware apps are set to feature published **James Webb Space Telescope (JWST)** images to enhance the visual appeal of the platform.
   - An alternate app within Wordware tests multiple models simultaneously, showcasing enhanced **search engine** capabilities with output speed tracking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider uses a map of your git repository to provide code context to LLMs.</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/playground">_Think-Lab Revised</a>: Use the power of ScratchPad-Think for every day web searches. Export refined search queries in JSON format. The scratchpad is a powerful tool that helps you maintain coherence and accuracy, especially...</li><li><a href="https://app.wordware.ai/share/8c523d8b-c109-4189-a6ce-cc9bfc5d24a2/playground">Sonnet Insight 3.5 - Rank Model Outputs </a>: This prompt processes a question using Sonnet 3.5, Gemini 1.5 Pro, llama 3.1 70B&amp;405B, GPT-4o/mini, Sonar Large (online model), Claude 3 Opus, Claude 3 Sonnet, and lastly Claude 3 Haiku. The app t...</li><li><a href="https://youtu.be/1B50IDUl5D4?si=pPfOvaHGax7t68Y0">Create fine-tuned models with NO-CODE for Ollama &amp; LMStudio!</a>: 👋 Hey everyone,Back with a new video highlighting a super cool feature that we just added into AnythingLLM where you can create a full fine-tuned model from...</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models</a>: DSPy: The framework for programming—not prompting—foundation models - stanfordnlp/dspy</li><li><a href="https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/">GraphRAG: A new approach for discovery using complex information</a>: Microsoft is transforming retrieval-augmented generation with GraphRAG, using LLM-generated knowledge graphs to significantly improve Q&amp;A when analyzing complex information and consistently outper...</li><li><a href="https://news.ycombinator.com/item?id=41013693">no title found</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">meta-llama/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE">llama-models/models/llama3_1/LICENSE at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1265746471966216353)** (1 messages): 

> - `Nous Research subreddit`
> - `AMA announcement` 


- **Nous Research subreddit launched**: A new subreddit for **Nous Research** has been started, where members can join to discuss the latest research and developments in **AI**.
   - Users are encouraged to start threads and get involved [here](https://reddit.com/r/NousResearch).
- **Upcoming AMA with Nous leaders**: An **AMA** session is planned with specific members in the coming weeks on **Reddit** to answer community questions.
   - Further information will be shared as it approaches, so stay tuned for updates!



**Link mentioned**: <a href="https://reddit.com/r/NousResearch">Reddit - Dive into anything</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1265382943627677828)** (224 messages🔥🔥): 

> - `Llama 3.1 Performance`
> - `Mistral Large 2 Release`
> - `Open-Source TTS Models`
> - `Autonomous Coding Tools`
> - `Synthetic Data in AI` 


- **Llama 3.1 faces competition from Mistral**: The Llama 3.1 model is being challenged by Mistral Large 2, which boasts a similar architecture but showcases better performance, especially in coding tasks.
   - Users are excited about Mistral's potential for improved output quality and the growing capabilities of synthetic data.
- **Mistral Large 2 impresses with capabilities**: Mistral Large 2 has been released with 123 billion parameters, featuring a 128k context window and strong performance on coding tasks.
   - Despite its non-commercial license restricting hosting options, it is expected to perform well on API platforms due to its innovative design.
- **Exploration of open-weight TTS models**: Users are discussing their experiences with various Text-to-Speech models, particularly focusing on quality, speed, and offline capabilities.
   - Models like ElevenLabs and Apple's Siri voices are compared, with recommendations for newer solutions like parler-expresso and VITS.
- **Inquiry about autonomous coding tools**: There is interest in the current state of open-source autonomous coding tools, such as Devika and Open Devin.
   - Users are looking for recommendations and comparisons to determine which tools may best suit their development needs.
- **Potential of synthetic data**: Users express enthusiasm about the use of high-quality synthetic data in training AI models, believing it could enhance performance and general applicability.
   - There is speculation that future advancements in synthetic data generation may lead to significant improvements in model capabilities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://x.com/capeto">Tweet from undefined</a>: no description found</li><li><a href="https://huggingface.co/qresearch/llama-3.1-8B-vision-378">qresearch/llama-3.1-8B-vision-378 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/SillyTilly/Meta-Llama-3.1-70B">SillyTilly/Meta-Llama-3.1-70B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct">meta-llama/Meta-Llama-3.1-405B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/kermit-suicide-flip-jump-crash-gif-5140737">Kermit Suicide GIF - Kermit Suicide Flip - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/normand1/HyperFeeder/blob/master/audioScripts/ttsLocalScript.sh">HyperFeeder/audioScripts/ttsLocalScript.sh at master · normand1/HyperFeeder</a>: The Autonomous Podcast Generator. Contribute to normand1/HyperFeeder development by creating an account on GitHub.</li><li><a href="https://ai.meta.com/blog/meta-llama-3-1">no title found</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Half-precision_floating-point_format">Half-precision floating-point format - Wikipedia</a>: no description found</li><li><a href="https://tenor.com/view/omegalul-lul-lulw-twitch-emote-gif-13523263">Omegalul Lul GIF - Omegalul LUL LULW - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/billyuchenlin/status/1815841947468353700">Tweet from Bill Yuchen Lin 🤖 (@billyuchenlin)</a>: A quick independent evaluation of Llama-3.1-405B-Instruct-Turbo (on @togethercompute) ⬇️  1️⃣ It ranks 1st on GSM8K! 2️⃣ Its logical reasoning ability on ZebraLogic is quite similar to Sonnet 3.5, and...</li><li><a href="https://github.com/meta-llama/llama-agentic-system">GitHub - meta-llama/llama-agentic-system: Agentic components of the Llama Stack APIs</a>: Agentic components of the Llama Stack APIs. Contribute to meta-llama/llama-agentic-system development by creating an account on GitHub.</li><li><a href="https://avian.io">Avian.io</a>:  Avian is a generative AI platform for Enterprise, enabling state of the art LLM inference across Llama-3.1-405B and supporting RAG with over 100 data connectors.</li><li><a href="https://en.wikipedia.org/wiki/Activation_function">Activation function - Wikipedia</a>: no description found</li><li><a href="https://x.com/capetorch/status/1816110002823745945">Tweet from Thomas Capelle (@capetorch)</a>: Want to try Llama3.1 405B model for free?   Let&#39;s work together to red-team the model and collaboratively generate a dataset to evaluate Llama 3.1 family of models.  We put together a simple Colab...</li><li><a href="https://huggingface.co/collections/hugging-quants/llama-31-gptq-awq-and-bnb-quants-669fa7f50f6e713fd54bd198">Llama 3.1 GPTQ, AWQ, and BNB Quants - a hugging-quants Collection</a>: no description found</li><li><a href="https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data">wiki-phrases-tokenizer/data at master · vtempest/wiki-phrases-tokenizer</a>: Wikipedia Outline Relational Lexicon Dataset (WORLD) *  Domain-Specific Extraction of Entities and Keywords (DSEEK) * Wikipedia Important Named Topic Entity Recognition (WINTER) - vtempest/wiki-phr...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1265387151412166807)** (24 messages🔥): 

> - `Fine-tuning Llama 3`
> - `Multi-language fine-tuning`
> - `Custom tool calls`
> - `Hermes function calling`
> - `Generative capabilities of LLMs` 


- **Fine-tuning Llama 3 presents challenges**: Members expressed concerns that **fine-tuning Llama 3 405B** will be quite a challenge, with suggestions that only **Lora FTing** might take it on.
   - One member noted that this situation might prompt advances in **DoRA fine-tuning** within **OSS**.
- **Resources needed for fine-tuning Pashto**: A member is seeking resources for **fine-tuning** models specifically for the **Pashto language**, highlighting the lack of available materials despite the large speaker base of **60 million**.
   - Another member suggested looking up recent **Aya23 model+ papers** for related information.
- **Custom tool calls require attention**: Discussion arose around the need for fine-tuning to achieve correct formats when performing **custom tool calls**, especially for simple tasks like checking for spam.
   - A participant emphasized using the **correct system prompt and schema** provided in the **GitHub repo for Hermes function calling**.
- **LLM struggles with generating complex code**: One member reported their attempt to generate a **snake game** in Python using **Llama 405B**, successful initially but failing to include a **DQN** method effectively.
   - They noted repeated failures despite providing error messages, indicating a need for better prompting strategies.
- **Queries on Hermes release and progress**: A couple of users inquired about the availability of a **Hermes release** for **Llama 3.1**, reflecting broader interest in updates.
   - Members discussed ongoing efforts and resources being used for advanced projects, including **multi-node training** setups.


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1265473433697910854)** (4 messages): 

> - `Citizen Sleeper core mechanics`
> - `wiki-phrases-tokenizer`
> - `grounded refusals` 


- **Citizen Sleeper's core mechanic explained**: The core mechanic of **Citizen Sleeper** revolves around rolling dice to assign actions, which significantly impact player progression in the game.
   - *Each day, players roll dice whose outcomes are governed by the condition system*, reinforcing themes of **precarity** and **risk**.
- **Introduction to wiki-phrases-tokenizer dataset**: A member shared a link to the [wiki-phrases-tokenizer GitHub repository](https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data), highlighting its potential as sample data for RAG with datasets like *the top 100k Wikipedia pages* and *Quora search queries*.
   - This dataset is described as containing valuable information for **domain-specific extraction** of entities and keywords.
- **Meta team’s intelligence acknowledged**: A member expressed surprise at not considering **grounded refusals** and acknowledged the **meta team** as being smarter in this regard.
   - This comment reflects a sentiment of humility and recognition of the team's capabilities.



**Link mentioned**: <a href="https://github.com/vtempest/wiki-phrases-tokenizer/tree/master/data">wiki-phrases-tokenizer/data at master · vtempest/wiki-phrases-tokenizer</a>: Wikipedia Outline Relational Lexicon Dataset (WORLD) *  Domain-Specific Extraction of Entities and Keywords (DSEEK) * Wikipedia Important Named Topic Entity Recognition (WINTER) - vtempest/wiki-phr...

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1265489424737894473)** (3 messages): 

> - `Sub-Symbolic Concept Space`
> - `Llama Model on GPU Clusters`
> - `Subscription-based AI Access` 


- **Exploring Sub-Symbolic Concept Space**: A member expressed excitement about finally having a moment to engage with **WorldSim** and ponder **sub-symbolic concept space**.
   - This indicates an ongoing interest in expanding on theoretical AI concepts in future discussions.
- **Llama Model Could Enable Tiered Access**: A member theorized that using the **Llama model** on a cluster of managed GPUs could create a **gated playground** accessible by subscription or tiers.
   - They suggested that if feasible, it would make for a worthy discussion in a future gathering.
- **Question about Code Availability**: A member inquired if there is any **code available** for the discussed applications or models.
   - This highlights a potential gap in resources for exploration within the community.


  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1265420101230788739)** (13 messages🔥): 

> - `SMT Solvers and LLM Translation`
> - `Updating Repo Structure`
> - `Difficult Moral Queries`
> - `Trolley Problem Morality Debate` 


- **Utilizing SMT Solvers for LLMs**: @SMT_Solvers suggested that teaching LLMs to translate word problems from English/German to SMTLIB can yield significant reasoning capabilities, essentially a **MADLIBS synthetic data problem** using egraphs for exploration.
   - This sparks potential for advanced reasoning tasks through effective translation methods, enhancing overall model performance.
- **Repo Structure Updates In Progress**: @teknium announced plans to update the repository's structure and schemas today, inviting collaboration from others in the community.
   - @n8programs expressed eagerness for updates and offered assistance in the process, highlighting community engagement.
- **Moral Dilemmas in Reasoning Tasks**: A discussion emerged around whether to include a subsection for **difficult/moral** queries, like the **trolley problem**, as reasoning tasks that challenge models' foundational moral principles.
   - This raises questions about the implications of moral reasoning versus the straightforward logical evaluation of scenarios, inviting deeper analysis.
- **Reflection on the Trolley Problem**: Concerns were raised about the trolley problem assessing which moral foundations models adopt rather than pure reasoning capabilities, with @stefangliga questioning its purpose.
   - @_paradroid suggested that structuring prompts can clarify reasoning processes and thought evaluations, enhancing understanding of moral frameworks.
- **Structured Reasoning in Moral Queries**: @_paradroid shared a structured framework to analyze the moral implications of self-driving car decisions, aiming for superior reasoning clarity and accuracy.
   - The framework includes identifying initial thoughts, providing context, and reflecting on the reasoning process, demonstrating a comprehensive approach to moral reasoning tasks.



**Link mentioned**: <a href="https://x.com/SMT_Solvers/status/1815856006427205672">Tweet from Chad Brewbaker (@SMT_Solvers)</a>: @halvarflake As I told @Teknium1 we can get a lot of reasoning via SMT solvers if we can teach the LLM to translate word problems from English/German to SMTLIB. A MADLIBS synthetic data problem if you...

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1265419710799806627)** (1 messages): 

> - `DeepSeek Coder V2`
> - `Private Inference Provider` 


- **DeepSeek Coder V2 Launches Private Inference Provider**: **DeepSeek Coder V2** now features a [private provider](https://openrouter.ai/models/deepseek/deepseek-coder) to serve requests on OpenRouter without input training.
   - This new capability was announced on [X](https://x.com/OpenRouterAI/status/1815860614755147961), signifying a step forward in private model deployment.
- **New Developments in Inference Providers**: The announcement of a private inference provider indicates strategic progression in the **OpenRouter** platform.
   - The **absence of input training** marks a significant difference from previous models, enhancing usability.



**Link mentioned**: <a href="https://x.com/OpenRouterAI/status/1815860614755147961">Tweet from OpenRouter (@OpenRouterAI)</a>: DeepSeek Coder V2 now has a private provider serving requests on OpenRouter, with no input training!  Check it out here: https://openrouter.ai/models/deepseek/deepseek-coder

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1265389919266803853)** (273 messages🔥🔥): 

> - `Llama 3.1 405B`
> - `Mistral Large 2`
> - `OpenRouter API Issues`
> - `Coding Tools Exploration`
> - `Language Model Pricing` 


- **Concerns over Llama 3.1 405B Performance**: Several users express dissatisfaction with the performance of **Llama 3.1 405B**, noting it struggles with NSFW content, often refusing prompts or outputting training data.
   - User feedback indicates that temperature settings heavily influence output quality, with some reporting better results at lower temperatures.
- **Mistral Large 2 Launch and Usage**: The **Mistral Large 2** model is now available as **Mistral Large**, effectively replacing the previous version with updates for improved multilingual capabilities.
   - Users speculate on its performance compared to **Llama 3.1**, particularly in handling languages like French.
- **OpenRouter API Challenges**: Users discuss limitations in the **OpenRouter API**, including rate limits and the handling of multilingual input, noting challenges faced when using certain models.
   - Reports indicate that while some models are free in preview, they may come with strict limitations on usage and context.
- **Interest in Open-Source Coding Tools**: In a shift of focus, users inquire about open-source autonomous coding tools like **Devika** and **Open Devin**, seeking recommendations based on current efficacy.
   - The discussion highlights a growing interest in experimenting with alternative coding solutions beyond mainstream AI offerings.
- **Model Pricing Comparisons**: Discussion on pricing reveals that **Mistral Large** offers competitive rates at **$3** per million tokens for input and **$9** for output, drawing comparisons to other models.
   - Users debate the value of uncensored outputs from various models, weighing it against the more commercial approach taken by other providers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1815837707505131699">Tweet from OpenRouter (@OpenRouterAI)</a>: 🏆 Multi-LLM Prompt Competition  Reply below with prompts that are tough for Llama 405B, GPT-4o, and Sonnet!  Winner gets 15 free credits ✨. Example:</li><li><a href="https://www.cloudflare.com/5xx-error-landing?utm_source=errorcode_520&utm_campaign=openrouter.ai"">5xx Error</a>: Cloudflare is a free global CDN and DNS provider that can speed up and protect any site online</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Meta: Llama 3.1 405B Instruct by meta-llama</a>: The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eval scores, the Meta AI team continues to push the frontier of open-source LLMs.  Meta&#x27;s latest c...</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found</li><li><a href="https://openrouter.ai/models/openai/gpt-4-32k">OpenAI: GPT-4 32k by openai</a>: GPT-4-32k is an extended version of GPT-4, with the same capabilities but quadrupled context length, allowing for processing up to 40 pages of text in a single pass. This is particularly beneficial fo...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama">no title found</a>: no description found</li><li><a href="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md#instruction-tuned-models">llama-models/models/llama3_1/MODEL_CARD.md at main · meta-llama/llama-models</a>: Utilities intended for use with Llama models. Contribute to meta-llama/llama-models development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/docs/responses#querying-cost-and-stats">Responses | OpenRouter</a>: Manage responses from models</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: User-friendly WebUI for LLMs (Formerly Ollama WebUI)</a>: User-friendly WebUI for LLMs (Formerly Ollama WebUI) - open-webui/open-webui
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1265758130596348077)** (1 messages): 

> - `Llama 3.1 Release`
> - `HuggingChat Updates`
> - `Community Tools`
> - `Usage Guides` 


- **Llama 3.1 Launches with Excitement**: The **Llama 3.1** model has officially launched, bringing exciting new features and capabilities. Check out the [blogpost](https://huggingface.co/blog/llama31) for all the juicy details.
   - The model is available [here](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct) for users to try it out.
- **Explore the Models and Community Tools**: Users are encouraged to dive into the **models** featured on [Hugging Face](https://huggingface.co/meta-llama), showcasing the latest advancements. Additionally, explore the community resource [Quants](https://huggingface.co/hugging-quants) for collaborative insights.
   - Resources to enhance your experience include the [How to use guide](https://github.com/huggingface/huggingface-llama-recipes) available on GitHub.
- **HuggingChat Version v0.9.1 Released**: The latest version **HuggingChat v0.9.1** makes the best AI chat models available to everyone, improving accessibility. Users can view the model page for deeper insights into its functionalities.
   - The new version integrates seamlessly with the [Llama](https://llama.meta.com/) features to enhance user interactions.



**Link mentioned**: <a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8)">HuggingChat</a>: Making the community's best AI chat models available to everyone.

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1265387375996174358)** (238 messages🔥🔥): 

> - `Llama 3.1 Discussion`
> - `Training Models`
> - `Using Rust in ML`
> - `Machine Learning Curriculum`
> - `Model Performance and Issues` 


- **Llama 3.1 405B Overload Issues**: Users reported that the Llama 3.1 405B model is frequently showing 'service unavailable' errors due to being overloaded with requests.
   - Some users discussed the characteristics of the 405B variant, mentioning that it feels more censored compared to its 70B sibling.
- **Challenges in Training Models**: There were several discussions on training models, including issues related to batch size not reducing training time or steps as expected.
   - Users explored how the training script from GitHub might be flawed, making epochs perform similarly to steps.
- **Using Rust in Machine Learning**: A user inquired about the usefulness of Rust in the ML community, specifically referring to the 'candle' framework for performance and GPU support.
   - The 'candle' project on GitHub was recommended as a Rust-based solution focused on machine learning applications.
- **Adding ML to Academic Curriculum**: A member shared challenges in helping their economics department add machine learning content to their undergraduate curriculum.
   - Participants discussed foundational concepts needed, emphasizing the importance of logic and programming basics for students.
- **AI-Generated Content Quality**: Users shared experiences with AI-generated images, noting technical issues such as blurriness and unrealistic backgrounds.
   - Maintaining image quality while performing techniques like fine-tuning a diffusion model was emphasized, alongside discussions about AI's ethical considerations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Xenova/whisper-speaker-diarization">Whisper Speaker Diarization - a Hugging Face Space by Xenova</a>: no description found</li><li><a href="https://huggingface.co/datasets/nroggendorff/oak">nroggendorff/oak · Datasets at Hugging Face</a>: no description found</li><li><a href="https://x.com/karpathy/status/1815842603377779140">Tweet from Andrej Karpathy (@karpathy)</a>: Huge congrats to @AIatMeta on the Llama 3.1 release! Few notes:  Today, with the 405B model release, is the first time that a frontier-capability LLM is available to everyone to work with and build on...</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found</li><li><a href="https://www.packtpub.com/en-us/product/transformers-for-natural-language-processing-9781800565791">Transformers for Natural Language Processing | Data | eBook</a>: Build innovative deep neural network architectures for NLP with Python, PyTorch, TensorFlow, BERT, RoBERTa, and more. Instant delivery. Top rated Mobile Application Development products.</li><li><a href="https://github.com/huggingface/candle">GitHub - huggingface/candle: Minimalist ML framework for Rust</a>: Minimalist ML framework for Rust. Contribute to huggingface/candle development by creating an account on GitHub.</li><li><a href="https://huggingface.co/AiAF/Lightsource-0Lightsource-OLS_PonyXL.safetensors">AiAF/Lightsource-0Lightsource-OLS_PonyXL.safetensors · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/LyliaEngine/assassin_cross_XL-bf16-pony-v1">LyliaEngine/assassin_cross_XL-bf16-pony-v1 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1265407549595713730)** (7 messages): 

> - `PEFT model loading methods`
> - `Stein score function relationship`
> - `Model training for summarization`
> - `UAE concepts`
> - `Elastic Search and web crawling` 


- **Understanding PEFT Model Loading Methods**: A discussion arose regarding two methods to load a PEFT model, comparing **AutoModelForCausalLM.from_pretrained** with adapter loading methods for the model **ybelkada/opt-350m-lora**.
   - *Is the adapter config responsible for retrieving the whole model in the first method?*
- **Exploring Stein Score Function**: A member expressed confusion about the relationship between the **Stein score function** and **probability density function**, specifically questioning the inclusion of the log (log pdf).
   - They are seeking clarity on the significance of this logarithmic function.
- **Training BERT for Summarization**: A member shared their experience learning to train a model for text summarization using **BERT** with the model **flan-t5-base-samsum**.
   - Summary metrics were shared, with highlights including a **Rouge1 score of 47.2141**.
- **Learning UAE Concepts**: A member is diving into concepts related to **UAE**, sharing a link to an [arXiv paper](https://arxiv.org/pdf/2309.12871) as part of their study.
   - They expressed a grasp of some concepts but are open to further explanations.
- **Elastic Search and Web Crawling**: Members discussed learning about **Elastic Search** and **Apify**, focusing on web crawling, scraping, and indexing techniques.
   - These methods are crucial for data retrieval and management in various applications.



**Link mentioned**: <a href="https://huggingface.co/sharmax-vikas/flan-t5-base-samsum">sharmax-vikas/flan-t5-base-samsum · Hugging Face</a>: no description found

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1265401156734029965)** (4 messages): 

> - `Meta's Llama 3.1 Models`
> - `Open Source AI`
> - `Mark Zuckerberg's Vision` 


- **Meta launches Llama 3.1, a game changer in AI**: [Meta's latest Llama 3.1 models](https://ai.meta.com/blog/meta-llama-3-1/) expand context length to **128K** and offer support for **eight languages**, marking a significant advancement in open-source AI.
   - Notably, the **Llama 3.1 405B** model boasts capabilities that rival closed-source models like OpenAI's **GPT-4o**, and the complete model is available for download including weights.
- **Zuckerberg’s commitment to open-source tech**: In a [blog post](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/), Mark Zuckerberg emphasizes that open-source AI is beneficial for **developers**, **Meta**, and society by fostering innovation and collaboration.
   - He believes that Llama can evolve into a robust open AI ecosystem, enabling developers to unlock new workflows and create tools that enhance their projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ai.meta.com/blog/meta-llama-3-1/">no title found</a>: no description found</li><li><a href="https://www.neowin.net/news/mark-zuckerberg-explains-why-open-source-ai-is-good-for-developers/">Mark Zuckerberg explains why open source AI is good for developers</a>: Mark Zuckerberg believes that open-source AI is the future of AI, fostering unrestricted innovation similar to how open-source development has accelerated progress in other fields.
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1265410055986217051)** (4 messages): 

> - `Mistral-NeMo 12B Instruct`
> - `Pony Diffusion v6`
> - `Llama 3.1 Release` 


- **Lightning fast chat with Mistral-NeMo 12B Instruct**: A demo showcasing **Mistral-NeMo 12B Instruct** with an implementation using [llama.cpp](https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp) has been released, promising impressive performance.
   - Users are encouraged to try it out for a **refreshing** chat experience.
- **Pony Diffusion v6 gets a weekly update**: The latest version, **Pony Diffusion v6**, was recently announced and features many options aimed at power users, with updates rolled out weekly.
   - The project can be found [here](https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6), and it has connections to a previous demo by artificialguybr.
- **Community Excitement for Llama 3.1**: The community is buzzing about the release of **Llama 3.1**, with a new space built around the **HF Inference API** that allows customization of system instructions.
   - Check out the space [here](https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8) to explore the features available for free, making it accessible to everyone.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/Mistral-NeMo-llamacpp">Mistral NeMo llama.cpp - a Hugging Face Space by gokaygokay</a>: no description found</li><li><a href="https://huggingface.co/spaces/Sergidev/HD-Pony-Diffusion-v6">HD Pony Diffusion - a Hugging Face Space by Sergidev</a>: no description found</li><li><a href="https://huggingface.co/spaces/as-cle-bert/Llama-3.1-405B-FP8">Llama 3.1 405B FP8 - a Hugging Face Space by as-cle-bert</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1265619447708647434)** (2 messages): 

> - `Object Detection in Java` 


- **Excitement Over Object Detection App Tutorial**: A member shared a [blog post](https://blog.stackademic.com/object-detection-app-in-java-a50ca86306ff) detailing the development of an **Object Detection App in Java**.
   - *Great!!!!* was the enthusiastic reaction from another member, indicating a positive reception and interest in the topic.
- **Community Engagement on Java Development**: Members expressed interest in **Java** development technologies, particularly regarding tutorials like the one shared.
   - The excitement reflects a growing community interest in practical applications and learning resources in software development.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1265739960204660787)** (1 messages): 

> - `Chameleon models`
> - `Batch processing images` 


- **Inquiry on Chameleon models**: A member asked if anyone has worked on **Chameleon models** and stated they have questions regarding how to batch/collate images for a batched forward pass.
   - *Could anyone share insights on image processing for these models?*
- **Batch processing questions raised**: The discussion highlighted the need for clarity on how to effectively implement batch processing and collate images for Chameleon models.
   - Several members expressed interest in sharing their experiences and best practices with batching during forward passes.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1265560960404422666)** (8 messages🔥): 

> - `Training Sentence Encoders`
> - `Metrics for Model Evaluation`
> - `Fine-tuning Sentence Transformers`
> - `RAG Pipeline for Q&A`
> - `Text-to-HTML/CSS Generation Model` 


- **Challenges with MultipleNegativesRankingLoss**: A member expressed difficulties training sentence encoders using **MultipleNegativesRankingLoss** and noticed worse performance upon increasing batch size for the **CachedMultipleNegativesRankingLoss**.
   - They inquired about common dataset issues that could arise when increasing batch size, aiming for better model results.
- **Metrics for Model Evaluation**: One member outlined their evaluation metrics, using **recall@5**, **recall@10**, and **recall@20** based on a small vector database from a fine-tuned model.
   - They also mentioned utilizing an evaluator called **TripletEvaluator** to gauge model performance.
- **Fine-tuning Sentence Transformers for Legal Norms**: A beginner member sought guidance on fine-tuning a sentence transformer for a dataset focused on **legal and financial norms**, aiming for a RAG pipeline for Q&A.
   - They requested steps and recommended readings to successfully approach this task.
- **Interest in Tiktoken Experience**: A member queried about others' experiences using **tiktoken**, prompting a call for shared insights.
   - This highlights an area of curiosity regarding the integration and effectiveness of the tool in related projects.
- **Open-source Text-to-HTML/CSS Generation Model**: A member announced their intention to acquire an **open-source text-to-HTML/CSS generation model** and sought recommendations.
   - This reflects ongoing exploration into tools that facilitate conversion of text content into web formats.


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1265485302433316914)** (6 messages): 

> - `Rectified Flow`
> - `Flow Matching`
> - `DDPM and DDIM Discussions`
> - `Evaluation of Generative Models`
> - `VAE Model Cards` 


- **Lack of Interest in Rectified Flow**: A member expressed frustration that while there are many discussions regarding **DDPM** and **DDIM**, there is little talk about **Rectified Flow** or **Flow Matching**.
   - They highlighted the difficulty in finding minimal examples for **Flow** such as generating **MNIST**, questioning the general interest in this topic.
- **Flow Schedulers in Diffusers**: Another member pointed out the existence of **FlowMatchEulerDiscreteScheduler** and **FlowMatchHeunDiscreteScheduler** in the `diffusers` library, implying their relevance to the discussion.
   - These resources can be found in the [Hugging Face documentation](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete).
- **Evaluation Methods for Generative Models**: A member referenced a document discussing qualitative and quantitative methods for evaluating **Diffusion models**, underscoring the complexity of model selection.
   - They mentioned that both qualitative and quantitative evaluations provide a stronger signal for comparing models such as **Stable Diffusion** and **GANs**.
- **Inquiring About VAE Model**: A member inquired about the specific **VAE** being referenced in the discussion, seeking clarification on its identity.
   - They requested the sharing of its corresponding model card to gain more insights.



**Link mentioned**: <a href="https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation">Evaluating Diffusion Models</a>: no description found

  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1265384287952437370)** (239 messages🔥🔥): 

> - `Kohya-ss GUI Issues`
> - `Lycoris Integration Updates`
> - `Model Performance Ratings`
> - `Stable Diffusion Model Comparisons`
> - `New AI Video Generation Model Announcement` 


- **Kohya-ss GUI faces compatibility issues**: Users reported issues with the current version of **Kohya-ss GUI** being incompatible with Python 3.10, needing an upgrade to 3.10.9 or higher.
   - *One user humorously remarked* that it's like requiring a weight limit of **180lbs but no more than 180.5lbs**, reflecting on the absurdity of such restrictions.
- **Lycoris integration in development.**: A mention was made of **Onetrainer** potentially implementing **Lycoris** features in a new dev branch soon, with community discussions around various functionalities.
   - The preference for **bmaltais' UI wrapper** for Kohya's scripts was noted, enhancing user experience with these integrations.
- **Community ratings for art model performance.**: A discussion unfolded over the performance ratings of models including **Kolors, Auraflow, Pixart Sigma, and Hunyuan**, with Kolors being favored for its speed and quality.
   - Participants emphasized different user experiences, debating the specific traits and applications of each model in depth.
- **Evaluating Stable Diffusion model capabilities.**: Several users debated the differences in output and usability between **Stable Diffusion 1.5** and **SDXL** in terms of detail and resolution quality.
   - Advanced techniques, such as **Hidiffusion** and **Adaptive Token Dictionary**, were highlighted as effective for enhancing output from older models.
- **Introduction of Stable Video 4D for multi-angle video generation.**: The **Stable Video 4D** model was introduced, enabling users to transform single object videos into new views for enhanced creative projects.
   - This new model is currently in its research phase, with expectations for applications in **game development, video editing, and virtual reality**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stability.ai/news/stable-video-4d">Stable Video 4D &mdash; Stability AI</a>: We are pleased to announce the availability of Stable Video 4D, an innovative model that allows users to upload a single video and receive dynamic novel-view videos of eight new angles/views, deliveri...</li><li><a href="https://huggingface.co/xinsir/controlnet-union-sdxl-1.0">xinsir/controlnet-union-sdxl-1.0 · Hugging Face</a>: no description found</li><li><a href="https://civitai.com/models/207437/ballz">BALLZ - Ballz 3 | Stable Diffusion LoRA | Civitai</a>: Mad Balls! The foam toys straight from the 80s
</li>
</ul>

</div>
  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1265404758043394178)** (58 messages🔥🔥): 

> - `Sampling methods in language models`
> - `Llama 3.1 benchmarking`
> - `Log likelihood evaluation`
> - `Greedy vs stochastic sampling`
> - `Tail probability in sampling` 


- **Understanding Sampling Methods for LLMs**: Members discussed various sampling methods used in language models such as **greedy sampling**, **top-p**, and **top-k**, emphasizing their trade-offs.
   - Stochastic sampling allows for diversity but requires multiple runs for statistically significant results, contrasting with the reliability of greedy sampling.
- **Llama 3.1 Sampling Preferences**: In the context of using **Llama 3.1** for benchmarking, members suggest checking its paper for recommended sampling methods, with the consensus leaning towards probabilistic sampling.
   - One member noted that **Gemma 2** utilizes top-p and top-k, which are typical for models in evaluations.
- **Log Likelihood as a Measurement Tool**: Log likelihood was highlighted as a valuable metric for assessing model performance, allowing comparisons of how well models replicate results under different sampling methods.
   - It's suggested that using log likelihood can help understand how sampling choices affect output distributions and overall model reliability.
- **Greedy Sampling as a Baseline**: Greedy sampling serves as a reliable baseline for model evaluations, generating the most probable output path through the vast output space.
   - Members argued that while stochastic sampling can yield diverse outputs, it complicates evaluation and requires extensive runs to achieve statistical significance.
- **Long Sequence Generation Challenges**: Discussion surfaced around the complexities of measuring quality in longer generated sequences, with caveats tied to sampling methods and log likelihood.
   - Concerns were raised that tail probabilities can lead to compounding errors in output, affecting the model's long-term performance and outcomes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://kipp.ly/transformer-param-count/">LLM Parameter Counting | kipply&#x27;s blog</a>: kipply&#x27;s blog about stuff she does or reads about or observes</li><li><a href="https://github.com/EleutherAI/cookbook/tree/main/calc">cookbook/calc at main · EleutherAI/cookbook</a>: Deep learning for dummies. All the practical details and useful utilities that go into working with real models. - EleutherAI/cookbook</li><li><a href="https://x.com/stephenroller/status/1579993017234382849">Tweet from Stephen Roller (@stephenroller)</a>: @srush_nlp I find people unfamiliar with scaling are shocked by this:</li><li><a href="https://build.nvidia.com/explore/discover#llama-3_1-405b-instruct">Try NVIDIA NIM APIs</a>: Experience the leading models to build enterprise generative AI apps now.</li><li><a href="https://www.lesswrong.com/posts/3duR8CrvcHywrnhLo/how-does-gpt-3-spend-its-175b-parameters)">How does GPT-3 spend its 175B parameters?
 — LessWrong</a>: [Target audience: Me from a week ago, and people who have some understanding of ML but want to understand transformers better on a technical level.] …
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1265394224631648428)** (132 messages🔥🔥): 

> - `Misleading Tweets on Model Performance`
> - `MoE vs Dense Models`
> - `Character.AI's Model Architecture`
> - `Mixtral and Mistral Model Design`
> - `External Data in LLM Training` 


- **Misleading Tweets and Assumptions about Model Performance**: Discussion ensued regarding a misleading tweet related to projection calculations in language models, particularly focusing on how **Character.ai** uses shared KV layers and how this influences performance metrics.
   - Members expressed confusion around the accuracy of the information shared and highlighted the personal journey of understanding transformer architectures.
- **Debate on MoE vs Dense Models**: Participants analyzed why **dense architectures** are being favored over **Mixture-of-Experts (MoE)** models, citing the high costs and engineering requirements for handling MoEs in large-scale training.
   - Arguments were made that MoEs should perform better in terms of efficiency once a model is pretrained, although concerns about varied engineering capabilities within organizations were raised.
- **Insights into Character.AI's Model Architecture**: Insights on **Character.AI's** architectural choices were shared, emphasizing how they manage inference efficiently through design optimizations, though exact details remain unclear from their blog posts.
   - Participants noted the potential for shared caches across layers, hinting that the model could benefit from architecture information that may not be publicly elucidated.
- **Mistral and Mixtral's Model Choices**: Discussion acknowledged the recent models like **Mistral** and **Mixtral** opting for dense architectures despite their abilities to implement MoEs, which surprised some members.
   - The ongoing challenges associated with training and efficiency concerns during inference were highlighted as key reasons for these design decisions.
- **Utilizing External Data in LLM Training**: A paper on leveraging external sources in training language models was shared, paving the way for improved performance in complex reasoning tasks going beyond traditional methods.
   - This sparked curiosity among members to explore how newer models incorporate such innovative techniques for better information retrieval and task execution.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.03133">Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training</a>: Mixture-of-experts (MoE) models facilitate efficient scaling; however, training the router network introduces the challenge of optimizing a non-differentiable, discrete objective. Recently, a fully-di...</li><li><a href="https://huggingface.co/papers/2204.05149">Paper page - The Carbon Footprint of Machine Learning Training Will Plateau, Then
  Shrink</a>: no description found</li><li><a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>: At Character.AI, we&#x27;re building toward AGI. In that future state, large language models (LLMs) will enhance daily life, providing business productivity and entertainment and helping people with e...</li><li><a href="https://arxiv.org/abs/2111.12763">Sparse is Enough in Scaling Transformers</a>: Large Transformer models yield impressive results on many tasks, but are expensive to train, or even fine-tune, and so slow at decoding that their use and study becomes out of reach. We address this p...</li><li><a href="https://arxiv.org/abs/2112.04426">Improving language models by retrieving from trillions of tokens</a>: We enhance auto-regressive language models by conditioning on document chunks retrieved from a large corpus, based on local similarity with preceding tokens. With a $2$ trillion token database, our Re...</li><li><a href="https://github.com/xuekt98/bbdm">GitHub - xuekt98/BBDM: BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models</a>: BBDM: Image-to-image Translation with Brownian Bridge Diffusion Models - xuekt98/BBDM</li><li><a href="https://arxiv.org/abs/2403.13097">Simple Ingredients for Offline Reinforcement Learning</a>: Offline reinforcement learning algorithms have proven effective on datasets highly connected to the target downstream task. Yet, leveraging a novel testbed (MOOD) in which trajectories come from heter...</li><li><a href="https://www.normalcomputing.com/blog-posts/supersizing-transformers-going-beyond-rag-with-extended-minds-for-llms">Normal Computing</a>: no description found
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1265464079003287653)** (21 messages🔥): 

> - `Llama API evaluation`
> - `Chat format model usage`
> - `Multiple-choice task handling` 


- **Llama API Evaluation via lm_eval**: Members discussed errors encountered while using the `lm_eval` tool with the **Llama 3.1-405B** model through the API at `llama-api.com`, especially regarding support for logits and multiple-choice tasks.
   - *“It gives an error: No support for logits.”* prompted a series of troubleshooting attempts, including checking URL formats and API key usage.
- **API Configuration Issues**: To address the 'Method Not Allowed' error, it was suggested to use the full API URL and ensure parameters like temperature and max tokens are correctly configured.
   - One member successfully edited the `_create_payload` method to address these issues, leading to functional model evaluations under specific configurations.
- **Handling of Multiple-choice Questions**: After successfully running evaluations for tasks like `gsm8k`, errors emerged with multiple-choice tasks like `mmlu_college_biology`, specifically an 'AttributeError' related to content processing.
   - This raised questions about the compatibility of the API outputs with the evaluation framework, leaving members to seek solutions and share error logs for further analysis.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gist.github.com/riteshdroid/0ec4525c3a315dcf373f16e9df5d1833">gist:0ec4525c3a315dcf373f16e9df5d1833</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/42dc244867889a19ae80847254a481f446f6e4b7/lm_eval/models/openai_completions.py#L121">lm-evaluation-harness/lm_eval/models/openai_completions.py at 42dc244867889a19ae80847254a481f446f6e4b7 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/openai_completions.py#L86">lm-evaluation-harness/lm_eval/models/openai_completions.py at main · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/riteshdroid/lm-evaluation-harness/blob/1a2dc674c3dfcff81e9c6f0bf495ba569106c931/lm_eval/models/api_models.py#L140">lm-evaluation-harness/lm_eval/models/api_models.py at 1a2dc674c3dfcff81e9c6f0bf495ba569106c931 · riteshdroid/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - riteshdroid/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/42dc244867889a19ae">GitHub - EleutherAI/lm-evaluation-harness at 42dc244867889a19ae80847254a481f446f6e4b7</a>: A framework for few-shot evaluation of language models. - GitHub - EleutherAI/lm-evaluation-harness at 42dc244867889a19ae80847254a481f446f6e4b7</li><li><a href="https://api.llama-api.com,model=llama3.1-405b">no title found</a>: no description found</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1174">Implementing local OpenAI API-style chat completions on any given inference server by veekaybee · Pull Request #1174 · EleutherAI/lm-evaluation-harness</a>: This PR addresses this issue: #1072 (comment) by passing a base_url to a new class, LocalChatCompletionsLM, which inherits from OpenaiChatCompletionsLM and accepts a local HuggingFace-style model n...
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1265458292726042746)** (25 messages🔥): 

> - `GPU Bit-Matching`
> - `GPU FLOPS and Data Types`
> - `Non-Deterministic Results in Floating Point Operations`
> - `CUDA Lookback Scan Algorithm`
> - `NCCL Computation Overlap Issues` 


- **Understanding GPU Bit-Matching**: A question arose about whether results from a specific GPU model are uniquely bit-matched given certain inputs, with members noting that it depends on the **algorithm** used.
   - Another commented that for most algorithms, results are consistent if run on the same GPU.
- **GPU FLOPS and Data Type Dependencies**: A member clarified that GPU FLOPS figures are influenced heavily by **data types** and whether computations use CUDA cores versus **tensor cores**.
   - Another member added that Nvidia's specs provide performance details for different data types, found in their **whitepapers**.
- **Non-Determinism in Floating Point Calculations**: It was discussed that using **floating point** data can sometimes lead to beneficial non-deterministic results based on how operations are ordered.
   - As noted, slight changes in kernel tuning or hardware can lead to variations in results, complicating debugging.
- **Lookback Scan Algorithm in CUDA Mode**: A member pointed out a CUDA mode episode regarding the **lookback scan** algorithm, suggesting it can sometimes be switched.
   - However, members struggled to find documentation or examples discussing how to utilize this algorithm.
- **Challenges in NCCL Computation Overlap**: It was reported that recommendations about overlapping computation with **NCCL** during the backward pass did not yield the expected ease of implementation.
   - A GitHub issue was cited, highlighting difficulties encountered while using NCCL for multi-GPU training in the context of ResNet-50.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/soumithchintala/status/1815829457858625642">Tweet from Soumith Chintala (@soumithchintala)</a>: Why do 16k GPU jobs fail? The Llama3 paper has many cool details -- but notably, has a huge infrastructure section that covers how we parallelize, keep things reliable, etc.  We hit an overall 90% eff...</li><li><a href="https://github.com/NVIDIA/nccl/issues/338">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: I used the environment from https://github.com/NVIDIA/DeepLearningExamples/tree/master/MxNet/Classification/RN50v1.5 to train resnet-50 with multiple GPUs (with horovod using nccl), and found the d...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1265657262085509162)** (1 messages): 

> - `Profiling Triton kernels`
> - `Accelerating current Triton GPTQ kernels`
> - `Integration of Triton kernels into PyTorch` 


- **Seeking Help to Profile Triton Kernel**: A user inquired about the process to profile Triton kernels as described in the [PyTorch blog post](https://pytorch.org/blog/accelerating-triton/). They highlighted the accelerations achieved with Triton but needed guidance on implementing profiling techniques.
- **Steps to Accelerate Triton GPTQ Kernels**: The blog outlines a first principles approach that accelerated Triton GPTQ kernels by **3x** for core GPTQ and **6x** for AutoGPTQ, reducing times from **275us to 47us** for typical Llama style inference. It emphasizes the effectiveness of coalesced memory access and strategies to reduce warp stalling to boost throughput.
- **Integrating Triton Kernels into PyTorch**: As part of the optimization effort, the blog discusses integrating Triton kernels into PyTorch code, underscoring its potential to replace existing native CUDA implementations. Over time, this integration aims to surpass the performance of traditional CUDA native GPTQ kernels.



**Link mentioned**: <a href="https://pytorch.org/blog/accelerating-triton/">Accelerating Triton Dequantization Kernels for GPTQ</a>: TL;DR  

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1265395795671384195)** (13 messages🔥): 

> - `torch.compile performance`
> - `GPU memory usage with torch.compile`
> - `CUDA kernel anti-patterns`
> - `PyTorch profiling tools`
> - `CUDA graphs in PyTorch` 


- **torch.compile struggles with small Bert model**: A user reported significantly increased **RAM usage** when testing `torch.compile` with a small **Bert model**, dropping batch size from **512 to 160** which was slower than using eager mode.
   - Despite compiling without issues when using `full_graph=True`, the user seeks insights into potential causes for the observed performance drop.
- **Profiler Use Recommended for Memory Issues**: One member suggested using the **PyTorch profiler** and its **memory trace tool** to investigate deeper into memory usage during model inference.
   - This approach could provide insights into whether specific configurations or usages are leading to the increased memory demands.
- **CUDA Graphs and Configuration Queries**: A user confirmed they didn't explicitly set **CUDA graphs**, sticking to defaults while using `torch.compile`; they referenced using **2.3.1 and 2.4 RC** versions.
   - The interaction highlighted **Inductor configurations** and whether changing them could affect performance during model compilation.
- **Highlighting CUDA Kernel Anti-patterns**: A member emphasized a subtle **anti-pattern** for writing CUDA kernels in PyTorch related to **GMEM scratch space** allocation, recommend noting tensor lifetimes beyond kernel launches.
   - This insight stemmed from debugging CI failures and relates to careful management of temporary tensors when developing new ops.
- **`torch.compile` with reduced overhead shows no difference**: The user observed no change in memory usage with or without the **`reduce-overhead`** and **`fullgraph`** options in their `torch.compile` setup.
   - This stable observation sheds light on the necessity of understanding compile modes versus memory efficiency in practice.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://hud.pytorch.org/benchmark/compilers">no title found</a>: no description found</li><li><a href="https://github.com/pytorch/pytorch/pull/131277">Fix IMAs in Flash-Attention splitkv kernel by drisspg · Pull Request #131277 · pytorch/pytorch</a>: Summary While debugging CI failures for flash_attention tests I stumbled across 2 IMAs for the split-kv variant of flash attention.  Illegal global memory writes during the writing of softmax_lse_a...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1265391454755487794)** (16 messages🔥): 

> - `VLM Performance`
> - `CUDA Advancement`
> - `Mistral Large 2`
> - `FP16/FP32 Intrinsics`
> - `Feature Engineering Success` 


- **VLMs outperforming in text generation**: A discussion highlighted that even when VLMs are available, they typically excel in text tasks over image processing, as shown with [GPT-4o's performance](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt) on the ARC test set.
   - Ryan found that feature engineering the problem grid yielded better results compared to relying entirely on GPT-4o's vision capabilities.
- **CUDA aiming to surpass cuBLAS**: A member announced upcoming CUDA advancements, stating, *we are going to outperform cuBLAS on a wide range of matrix sizes*.
   - This includes potential enhancements not only for SGEMM but for other operation types as well.
- **Mistral Large 2 showcases advanced features**: [Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) boasts a 128k context window with support for multiple languages and 80+ coding languages, designed for efficient single-node inference.
   - With 123 billion parameters, it is geared towards long-context applications and is released under a research license allowing for non-commercial usage.
- **Impact of FP16/FP32 on performance**: Discussion surfaced around NVIDIA's hardware intrinsics for FP16/FP32, which could significantly influence performance outcomes.
   - This has generated excitement for future developments in the CUDA ecosystem.
- **Interesting Benchmark Comparisons**: Members found Mistral's latest benchmarks intriguing as they *push the boundaries of cost efficiency, speed, and performance*.
   - The new features provided facilitate building innovative AI applications for various contexts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt">Getting 50% (SoTA) on ARC-AGI with GPT-4o</a>: You can just draw more samples
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1265752468273234024)** (1 messages): 

> - `ML/AI career roadmap`
> - `Internship opportunities`
> - `Job search strategies` 


- **Seeking Guidance for ML/AI Career Path**: A user requested help in designing a roadmap to secure full-time positions and internships in **ML/AI** roles, sharing a [Google Document with details](https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing).
   - They emphasized their willingness to work long hours to meet targets and are open to any suggestions regarding their roadmap.
- **Open to Suggestions for Internships**: The user is looking for feedback on their approach to landing **internships** in the ML/AI field and whether their plans are feasible.
   - They explicitly stated that timelines should not be considered unrealistic as they can dedicate extra hours to complete tasks.



**Link mentioned**: <a href="https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing">ML Roadmap</a>: 3 months - (sept, oct, nov)  roadmap Statistics: https://www.youtube.com/watch?v=MXaJ7sa7q-8&amp;list=PL0KQuRyPJoe6KjlUM6iNYgt8d0DwI-IGR&amp;t=11s (1 week) Linear Algebra - https://www.youtube.com/wat...

  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1265612314711953460)** (10 messages🔥): 

> - `CUDA Installation Issues`
> - `Out of Memory Errors`
> - `Llama-2 Chat Model`
> - `Running Models as Discord Bots` 


- **Torch not compiled with CUDA**: A member discovered that **Torch** wasn't compiled with **CUDA** enabled and sought help on how to resolve it.
   - Another member advised that installing the CUDA version directly from the install page would provide the exact command needed.
- **Got CUDA working but hit a wall**: After getting CUDA functional, a member encountered a **torch.cuda.OutOfMemoryError**, indicating insufficient GPU memory while trying to allocate **172.00 MiB**.
   - They received a suggestion to adjust the **max_split_size_mb** to prevent memory fragmentation, referencing documentation for resolution.
- **Exploring Llama-2 7B Model**: A member shared details about their fine-tuned [Llama-2 7B](https://huggingface.co/TheBloke/Llama-2-7B-fp16) model, trained on a **24GB GPU** over **19 hours** using the Wizard-Vicuna dataset.
   - They provided multiple links to versions of the model, including **GGML** and **GPTQ**, hosted on Hugging Face.
- **Running models as Discord bots**: A member expressed interest in running the Llama-2 model as a **Discord bot**, showing their enthusiasm for its capabilities.
   - This statement reflects a broader interest in integrating AI models into community platforms.



**Link mentioned**: <a href="https://huggingface.co/georgesung/llama2_7b_chat_uncensored">georgesung/llama2_7b_chat_uncensored · Hugging Face</a>: no description found

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1265489453162434658)** (8 messages🔥): 

> - `ImportError in Torch AO`
> - `Supported PyTorch versions`
> - `Pruning and Quantization issues` 


- **ImportError: Importing from Torch**: A new user encountered an *ImportError* when trying to import 'return_and_correct_aliasing' in *torch.utils._python_dispatch*, indicating a version incompatibility.
   - Linking to [this GitHub issue](https://github.com/pytorch/ao/issues/29) was suggested for further investigation.
- **Testing on PyTorch versions**: Members indicated that they do not test on *PyTorch versions before 2.2*, implying that users should upgrade their versions for optimal functionality.
   - One user confirmed they would try upgrading to *torch 2.2* based on this advice.
- **Concerns with Llama 3.1 Inference Latency**: A user inquired whether *Llama 3.1 8b* has improved inference latency compared to *3.0*, highlighting ongoing discussions about model performance.
   - No responses were provided regarding the specific latency performance of the models.
- **Tutorial Issues with Pruning and Quantization**: A user shared confusion regarding the *weight_orig* and *weight_mask* transformations while implementing structured pruning with quantization, seeking clarity.
   - They received a *RuntimeError* related to the deepcopy protocol when attempting to apply the detach operation, which broke model inference.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/tutorials/intermediate/pruning_tutorial.html">Pruning Tutorial — PyTorch Tutorials 2.4.0+cu124 documentation</a>: no description found</li><li><a href="https://github.com/pytorch/ao/issues/29`">Issues · pytorch/ao</a>: Custom data types and layouts for training and inference - Issues · pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1265754646509654149)** (1 messages): 

> - `Blockwise Attention in Llama 3`
> - `Input Sequence Splitting` 


- **Implementing Blockwise Attention in Llama 3**: A user inquired about the correct stage for splitting the input sequence into blocks when implementing **blockwise attention** in the **Llama 3** architecture.
   - They specifically asked whether this should occur after applying **rotary position embeddings** to vectors **Q** and **K**, or before the **self-attention block**.
- **Clarification on Sequence Handling**: The user expressed confusion regarding the implementation specifics of the **Llama 3** architecture, indicating a need for clarity on the handling of input sequences.
   - The discussion revolves around optimal strategies for integrating **blockwise attention** effectively into the model's processing flow.


  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/)** (1 messages): 

iron_bound: neat https://github.com/AnswerDotAI/fsdp_qlora/tree/llama400b
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1265383434662973582)** (71 messages🔥🔥): 

> - `KV Cache Implementation`
> - `ZeRO-2 Performance Insights`
> - `LLaMA and muP Comparison`
> - `Stochastic Rounding Strategies`
> - `GPT-2 Training Experiment` 


- **Progress on KV Cache Logic**: A member reported successful implementation of partial KV cache logic for attention, which involved using existing buffers intelligently without changing the layout.
   - Debugging revealed discrepancies in token results during the second pass, but the overall implementation showed significant progress.
- **Insights on ZeRO-2 Performance**: Testing with ZeRO-2 and 2 GPUs showed an estimated 25% memory savings on gradients for smaller models, with plans for scalability in mind.
   - Despite the improvements, challenges were noted with gradient computations needing additional copies during communication phases.
- **Comparison Between LLaMA and muP Techniques**: Discussion emerged around LLaMA's performance compared to muP, specifically regarding the use of techniques such as tanh soft clamping.
   - Concerns were raised about whether muP enhances performance or primarily offers better learning rate transfers.
- **Stochastic Rounding in Gradient Accumulation**: A member highlighted a proposed method for stochastic rounding in gradient accumulation to improve training stability and efficiency.
   - This approach could lead to more effective gradient updates while potentially allowing for greater accumulation during training.
- **Training Results from GPT-2 Experiment**: Training was completed on a GPT-2 350M model, using the Fineweb-edu dataset with interleaved OpenHermes data for instruction pretraining.
   - Despite some curious training loss patterns, the overall results were deemed stable, and the model is available for public use on Hugging Face.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05872">Scaling Exponents Across Parameterizations and Optimizers</a>: Robust and effective scaling of models from small to large width typically requires the precise adjustment of many algorithmic and architectural details, such as parameterization and optimizer choices...</li><li><a href="https://github.com/microsoft/mup/issues/76">Not getting perf improvements from muP at ~1.5B scale · Issue #76 · microsoft/mup</a>: Hey guys, first of all thanks for the awesome work! I&#39;ve implemented muP in the llm.c project (see here), the coord checks seem to be flat / correct (I went up to 15 steps and still flat!) but I a...</li><li><a href="https://github.com/karpathy/llm.c/pull/593/files#diff-c8a8f83fdc5921f95e3e09a1b2f475f8342a20042d8bb4a9eea3e291c8b4ad11R596-R607">Zero 2 - WIP by ngc92 · Pull Request #593 · karpathy/llm.c</a>: Trying to get a first version working. Code isn&amp;#39;t nice, we currently lose the asynchrony in the communication code because we need to reuse the buffer for the next layer, and it doesn&amp;#39;...</li><li><a href="https://huggingface.co/jrahn/gpt2_350M_edu_hermes">jrahn/gpt2_350M_edu_hermes · Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/llm.c/pull/307">Improve tanh derivative in backward gelu by akbariyeh · Pull Request #307 · karpathy/llm.c</a>: It is cheaper to compute the derivative of tanh as 1 - tanh^2 than computing 1/(cosh^2). This will probably not make a measurable difference.</li><li><a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>: Use cudaMallocManaged to allocate optimizer states if we run out of device memory, so we can still train (slowly) even if we cannot fit the optimizer state This is based on #694 , which should be m...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1265425478198493258)** (3 messages): 

> - `FlashAttention Support for AMD`
> - `MI200 & MI300 Compatibility`
> - `GitHub Pull Requests` 


- **FlashAttention Now Supports AMD ROCm**: A recent [GitHub Pull Request #1010](https://github.com/Dao-AILab/flash-attention/pull/1010) implements support for **AMD ROCm** in the **FlashAttention 2** library, including several C++ APIs like `mha_fwd` and `mha_varlen_fwd`.
   - The implementation is rooted in composable kernel technology, maintaining **API consistency** with the original version.
- **Limited Compatibility with MI200 & MI300**: It was stated, *'We only support **mi200 & mi300** at this time'* regarding the compatibility of the newly updated FlashAttention.
   - This draws a clear line on current support, implying potential future updates for broader compatibility.



**Link mentioned**: <a href="https://github.com/Dao-AILab/flash-attention/pull/1010">Support AMD ROCm on FlashAttention 2 by rocking5566 · Pull Request #1010 · Dao-AILab/flash-attention</a>: This PR implement the AMD / ROCm version of c++ flash api  mha_fwd mha_varlen_fwd mha_bwd mha_varlen_bwd   The kernel implementation comes from composable kernel The c++ api is same as original ver...

  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1265387780696182906)** (87 messages🔥🔥): 

> - `Llama 3.1 Errors`
> - `Mistral Large Model Release`
> - `Multilingual Model Performance`
> - `Training and Fine-tuning Challenges`
> - `Synthetic Data Generation in Models` 


- **Llama 3.1 Encountering Errors**: Users reported issues with **Llama 3** resulting in errors like *AttributeError* and discussions suggested possible outdated images or configurations as causes.
   - One user mentioned trying a different image to resolve the problem while another expressed general frustration with frequent model updates.
- **Mistral Large Model Open-sourced**: Mistral has released the **Mistral-Large-Instruct-2407** model, featuring **123B parameters** and claiming state-of-the-art performance.
   - Key features include multilingual support across dozens of languages, proficiency in coding, and advanced agentic capabilities, prompting excitement among users.
- **Discussion on Multilingual Model Performance**: Comparisons between **Llama 3.1** and **NeMo** revealed performance variances, particularly in multilingual capabilities, with specific strengths in different languages.
   - Users noted that while **Llama 3** had some European languages, **NeMo** reportedly offers better support for **Chinese** and other languages.
- **Challenges in Model Training and Fine-tuning**: Concerns were raised about the need for significant RAM to effectively train large models like Mistral, with some users remarking on their limitations.
   - Someone expressed difficulties with exploding gradients during training, pondering whether this was linked to sample packing.
- **Synthetic Data Generation Mentioned**: The launch of **Llama 3.1** included a reference to *Synthetic Data Generation*, prompting calls for internal documentation scripts.
   - Users discussed the idea as potentially beneficial for fine-tuning and training models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/1littlecoder/status/1815768634297405811">Tweet from 1LittleCoder💻 (@1littlecoder)</a>: Llama 3.1 launch specifically mentions &#34;Synthetic Data Generation&#34; (too much of @Teknium1 influence ;) )</li><li><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">mistralai/Mistral-Large-Instruct-2407 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1265414397820534805)** (33 messages🔥): 

> - `Adapter Fine-Tuning`
> - `Llama-3.1 Compatibility`
> - `CUDA Errors`
> - `H100 Configurations` 


- **Discussion on Adapter Fine-Tuning Stages**: Members discussed the potential of implementing multiple stages of adapter fine-tuning, considering initializing later stages with previous results, such as using SFT weights for DPO training.
   - A related feature request was found on [GitHub](https://github.com/axolotl-ai-cloud/axolotl/issues/1095) with suggestions for small code changes to facilitate this approach.
- **Llama-3.1 Fine-Tuning Troubles**: Several users reported errors when fine-tuning **Llama-3.1-8b**, with issues regarding CUDA check failures and suggestions to use **official weights** from **Hugging Face**.
   - One member confirmed successful fine-tuning with **12b** models while another found that updating **transformers** resolved their issues with Llama 3.1.
- **Insights on CUDA Check Implementation Errors**: A user queried about a specific CUDA error encountered during the training process, leading to discussions about potentially corrupted CUDA installations.
   - Other members suggested reinstalling relevant libraries and shared their configurations as possible solutions.
- **Request for H100 Configuration References**: A member inquired about known **Axolotl** configurations that work well for fine-tuning on a single **8xH100** setup.
   - The request highlights the community's need for effective model configurations tailored for specific hardware deployments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1095)">Issues · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.</li><li><a href="https://github.com/axolotl-ai-cloud">Axolotl AI</a>: Axolotl AI has 4 repositories available. Follow their code on GitHub.</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1679">Adopt qlora-pipe approaches · Issue #1679 · axolotl-ai-cloud/axolotl</a>: ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/issues/1689)">Issues · axolotl-ai-cloud/axolotl</a>: Go ahead and axolotl questions. Contribute to axolotl-ai-cloud/axolotl development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1265647380729172072)** (1 messages): 

> - `Request for Help`
> - `Experience Sharing` 


- **Seeking Relevant Experience for Assistance**: A member requested help regarding a specific topic link shared in the channel.
   - They called for anyone with relevant experience to step forward and assist with the inquiry.
- **Open Request for Support in the Channel**: The same member emphasized the need for collective knowledge within the community about the issue linked.
   - They reiterated that any input from experienced individuals would be invaluable for resolving their query.


  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1265401315215937622)** (69 messages🔥🔥): 

> - `GPT-4o mini updates`
> - `Mistral Large 2 details`
> - `OpenAI's financial challenges`
> - `AI licensing and usage`
> - `New RLHF discussions` 


- **GPT-4o mini dominates Chatbot Arena**: With over 4,000 user votes, **GPT-4o mini** is now tied for #1 in the Chatbot Arena leaderboard, outperforming its previous version while being **20x cheaper**.
   - The excitement was evident as developers celebrated this milestone, noting the continuous drop in the **cost of intelligence** for new applications.
- **Mistral Large 2: A New Contender**: Mistral Large 2 boasts a **128k context window** and supports dozens of languages, making it a top-tier model for high-complexity tasks, aimed at commercial and research usage under their specific license.
   - Discussions arose around the **licensing conditions**, with clarity lacking on commercial use as users seek practical applications for the technology.
- **OpenAI's $5 billion Loss Prediction**: Recent estimates suggest OpenAI could face a staggering loss of **$5 billion** this year, attributing significant costs to Azure bills and training expenses.
   - Concerns about sustainability and profitability were raised amid discussions of API revenue being surprisingly low compared to expectations.
- **Chatbot Licensing and Legal Challenges**: Questions arose about whether the **EU AI Act** influenced Mistral's licensing approach, speculating on potential commercial usage restrictions associated with legal compliance.
   - The dialogue highlighted the need for clearer documentation and guidance regarding commercial applications of emerging models.
- **Shifts in RLHF Methodologies**: The conversation pointed to **Llama 3** marking a significant shift away from traditional RLHF approaches, with implications for the effectiveness of contractor-based data labeling.
   - Anticipation is building for future posts exploring new RLHF strategies and the potential existence of **data foundries** to support evolving methodologies.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mistral.ai/technology/#pricing">Technology</a>: Frontier AI in your hands</li><li><a href="https://x.com/lmsysorg/status/1815855136318840970?s=46">Tweet from lmsys.org (@lmsysorg)</a>: Exciting Chatbot Arena Update -- GPT-4o mini&#39;s result is out!  With 4K+ user votes, GPT-4o mini climbs to the top of the leaderboard, now joint #1 with GPT-4o while being 20x cheaper! Significantl...</li><li><a href="https://fxtwitter.com/paulgauthier/status/1816018141878620414">Tweet from Paul Gauthier (@paulgauthier)</a>: DeepSeek Coder V2 0724 is #2 on aider&#39;s leaderboard! It can efficiently edit code with SEARCH/REPLACE, unlike the prior version. This unlocks the ability to edit large files. Coder (75%) is close ...</li><li><a href="https://github.com/openai/safety-rbr-code-and-data">GitHub - openai/safety-rbr-code-and-data: Code and example data for the paper: Rule Based Rewards for Language Model Safety</a>: Code and example data for the paper: Rule Based Rewards for Language Model Safety - openai/safety-rbr-code-and-data</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://x.com/lmsysorg/status/1816010015494529540">Tweet from lmsys.org (@lmsysorg)</a>: People have been asking why GPT-4o mini ranks so high on Arena! We truly appreciate all the feedback. A few things to note:  1. Chatbot Arena measures human preference in different areas. We encourage...</li><li><a href="https://fxtwitter.com/moyix/status/1815840634013639086?s=46">Tweet from Brendan Dolan-Gavitt (@moyix)</a>: Sorry OpenAI is doing WHAT now?! Fine-tuning gpt-4o-mini is *free* for up to 2M tok/day??</li><li><a href="https://x.com/btibor91/status/1816142224138158365?s=46">Tweet from Tibor Blaho (@btibor91)</a>:   Quoting aaron holmes (@aaronpholmes)   New: OpenAI is on track to lose $5 billion this year, we estimate based on internal financial data and sources.   OpenAI expects to spend around $4b on Azure b...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1265406872761143356)** (8 messages🔥): 

> - `Vocabulary Size Impact on Inference`
> - `Byte Pair Encoding and Tokenization`
> - `Model Size Relation to Vocabulary`
> - `Tradeoffs in Vocabulary Expansion` 


- **Larger Vocabulary Might Slow Inference**: Concerns emerged regarding a larger vocabulary size potentially *slowing down* inference, challenging the common belief that it reduces forward passes needed for **common sentences**.
   - One member questioned if this assumption was valid, suggesting that context may matter especially for **small models** where a vocabulary increase has a larger parameter impact.
- **Tradeoffs in Vocabulary Sizing Logic**: Discussion revolved around the idea that a smaller vocabulary compresses sequences, while a larger one potentially increases token count, which may complicate inference time.
   - Members debated the advantages of having fewer tokens for frequent phrases versus retaining finer interactions that larger vocabularies might miss.
- **Complexity in Vocabulary Research**: A member pointed out the potential high costs and narrow applicability of conducting thorough experiments to test vocabulary effects on different models.
   - They noted that findings might not generalize well, emphasizing the need for caution in broad claims about model capabilities.
- **Byte Pair Encoding's Role in Vocabulary Building**: One participant highlighted how **Byte Pair Encoding** (BPE) constructs vocabulary by first creating tokens for individual words and then merging them into larger tokens when context allows.
   - This process sparks discussion on whether using multiple tokens instead of a single compound token could enhance sequence comprehension and attention metrics.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1265555815918469140)** (4 messages): 

> - `IBM's Strategies`
> - `Magic Quadrant` 


- **IBM's Shift in Focus**: A member pointed out that the previous content was replaced with a page highlighting **popular providers**, prompting curiosity about *what IBM is doing now*.
   - This shift raises questions regarding **IBM's strategies** in the current tech landscape.
- **Insights on the Magic Quadrant**: A member mentioned the potential influence over the **Magic Quadrant**, focusing on factors like **Ability to Execute** and **Completeness of Vision**.
   - This indicates ongoing competition and strategic positioning within the tech industry.
- **Discussion on AI and Midjourney**: A relevant article from The New York Times titled *A Letter to Midjourney* was shared, discussing current trends in AI.
   - The article may provide insights into public perception and the evolving role of AI technologies.


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1265562937632882770)** (11 messages🔥): 

> - `CrowdStrike outage apology`
> - `Pre-training data benchmarks`
> - `Datacenter throughput issues` 


- **CrowdStrike's $10 Apology Gift Card for Outage**: After a massive outage caused by a botched update, **CrowdStrike** is offering partners a **$10 Uber Eats gift card** as an apology, according to multiple reports.
   - However, some recipients found that when attempting to redeem the gift card, they received an error indicating that the voucher had been **canceled**.
- **Paid Bonuses for Benchmark-Free Data**: A member highlighted that **they literally paid people bonuses** if the pre-training data used in their models did not contain any benchmarks.
   - This sparked a discussion on Twitter, where many caught other interesting details from the paper that could potentially warrant further papers.
- **Datacenter Microclimate Affects Throughput**: Participants discussed a note in the paper indicating a **2% drop in throughput** at midday due to issues related to the **datacenter microclimate**.
   - This detail was pointed out as significant, showcasing how minor environmental factors impact performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/theemozilla/status/1815989758360744085?s=46">Tweet from emozilla (@theemozilla)</a>: like they literally paid people bonuses if the pre-training data didn&#39;t contain any benchmarks</li><li><a href="https://techcrunch.com/2024/07/24/crowdstrike-offers-a-10-apology-gift-card-to-say-sorry-for-outage/">CrowdStrike offers a $10 apology gift card to say sorry for outage | TechCrunch</a>: Several people who received the CrowdStrike offer found that the gift card didn&#039;t work, while others got an error saying the voucher had been canceled.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1265397012917129349)** (3 messages): 

> - `Mark Zuckerberg's AI Era`
> - `Snail emoji enthusiasm` 


- **Inside Mark Zuckerberg's AI Era**: A shared [YouTube video](https://www.youtube.com/watch?v=YuIc4mq7zMU) titled 'Inside Mark Zuckerberg's AI Era | The Circuit' discusses the latest battle in the AI wars between open and closed models, highlighting Mark Zuckerberg's front-line role.
   - The video description notes it gives insights into Meta's rebranding and direction amidst ongoing AI developments.
- **Community celebrates the humble snail**: A member expressed their love for snails, sharing a friendly emoji depicting the creature.
   - *We love snail* was the enthusiastic sentiment that captured members' appreciation for this unique representation.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=YuIc4mq7zMU">Inside Mark Zuckerberg&#39;s AI Era | The Circuit</a>: If the latest battle in the AI wars is between open and closed models, Meta CEO and Founder Mark Zuckerberg is right on the frontlines. Since rebranding as M...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1265730238822875270)** (2 messages): 

> - `Llama 3 Release`
> - `RLHF in Capabilities`
> - `Synthetic Data for Alignment` 


- **Llama 3 Officially Released**: Today, Meta is [officially releasing](https://llama.meta.com/) the **largest and most capable open model to date, Llama3-405B**, which is trained on **15T tokens** and beats **GPT-4 on all major benchmarks**.
   - This model is a dense transformer that signifies a notable advancement in open-source AI capabilities.
- **RLHF Leads to Post-Training Capabilities**: An alignment lead on Llama stated that **100% RLHF** is responsible for how capabilities emerge post-training, highlighting the importance of this method.
   - This statement prompts discussions on effective alignment methods in model training.
- **Synthetic Data's Role in Alignment Discussed**: A noteworthy overview was shared on the utilization of **synthetic data for alignment**, shedding light on its potential benefits.
   - This discussion emphasizes the growing interest in leveraging synthetic data to improve AI alignment strategies.
- **Join the Emergency LLM Paper Club**: Members are invited to join the [emergency LLM paper club](https://x.com/latentspacepod/status/1816151808357908698) for an in-depth discussion about the Llama 3 paper.
   - This initiative reflects a collaborative effort to analyze significant AI literature.
- **AI in Action Club Featuring Cursor Cofounders**: For ongoing engagement, members are encouraged to participate in the [AI in Action club](https://lu.ma/tnmx3pvp) focusing on a special feature with the Cursor cofounders about their latest coding agent, Composer.
   - This highlights the community's commitment to staying updated with innovative AI tools.



**Link mentioned**: <a href="https://www.latent.space/p/llama-3">Llama 2, 3 &amp; 4: Synthetic Data, RLHF, Agents on the path to Open Source AGI</a>: Llama 2 lead and Llama 3 post-training lead Thomas Scialom of Meta/FAIR, on the Chinchilla trap, why Synthetic Data and RLHF works, and how Llama4&#x27;s focus on Agents will lead us to Open Source AG...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1265421370112606269)** (4 messages): 

> - `SnailBot News` 


- **SnailBot News Announced**: A notification was sent regarding **SnailBot News** to the tag <@&1216534966205284433>.
   - *Interesting* updates seem to be anticipated based on recent discussions.
- **Time Reference of 45 Minutes**: A member specified an **interesting** observation related to **45 minutes**.
   - The context of this time period remains unspecified in the current discussion.


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1265388704827179039)** (12 messages🔥): 

> - `MAX and Mojo Compiler Versioning`
> - `Nightly Compiler Releases`
> - `Confusion in Versioning`
> - `Feature vs Calendar Based Releases` 


- **MAX and Mojo Compiler Versioning Dilemma**: Discussion arose around whether the next main compiler version would be **24.5** or **24.8**, considering that feature/stable and nightly versions follow different release principles.
   - Concerns were highlighted about the potential disconnect between the nightly and main versions moving forward, especially regarding future dates like **2025**.
- **Nightly Releases Follow a Calendar System**: It was clarified that the **nightly releases** are based on a calendar model while the main releases are driven by **marketing** considerations, not strictly by date.
   - One member noted that the accidental alignment of versions could cause confusion, citing their own experience mixing up version numbers during discussions.
- **Community Exploration into ML Complexity**: One participant mentioned delving into **machine learning** topics and described it as a 'hot mess', sharing their wonder at the complexities they encountered recently.
   - This comment underscored the ongoing challenge of navigating ML discussions in the community and the various confusions that can arise.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1265384971636576420)** (17 messages🔥): 

> - `v24.5 release speculation`
> - `Using SDL with Mojo`
> - `Discussion on Var and Let`
> - `Generated Art vs AI`
> - `Regex library in Mojo` 


- **Speculation on v24.5 Release Date**: There's ongoing chatter regarding GPU features, leading to the guess that the release of **v24.5** could take some time as the team stabilizes its features.
   - *There’s some debate about why the versioning system follows an increment per year*.
- **Interest in Using SDL with Mojo**: A user inquired about resources to learn about **SDL** integration with **Mojo**, expressing a desire to understand the process better.
   - On a related note, there's curiosity about how to utilize **DLHandle** within the context of SDL.
- **Debate Over Var and Let Usage**: A user questioned the necessity of using **var** if everything is declared as **var**, suggesting it might be redundant.
   - In response, another member noted that var is beneficial to the compiler, whereas **let** primarily serves those who prefer *immutability*.
- **Generated Art Performance Compared to AI**: One user remarked about their computer creating some 'art', stating *it's not as good as the gen ai*.
   - Another user suggested that comparisons should consider the amount of compute power spent.
- **Query on Regex Library Availability**: A user asked if a **regex library** exists within **Mojo**, highlighting interest in handling regular expressions.
   - The conversation did not provide a definitive answer to the query.


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1265449434188222474)** (54 messages🔥): 

> - `Mojo Updates`
> - `Git Instructions`
> - `DTypePointer Removal`
> - `SIMD Comparisons`
> - `Contributing to Mojo` 


- **Significant Mojo Updates Released**: A new nightly Mojo compiler has been released, updating to `2024.7.2405` with notable changes including the removal of `DTypePointer` and new methods for string formatting.
   - A complete changelog can be found at [current changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md).
- **Git Rebase Challenges**: Several members discussed challenges with git rebasing and encountered issues like unresolved merge conflicts while following contributing guides.
   - One member expressed frustration over feeling limited in their ability to contribute due to these tooling issues.
- **DTypePointer's Impact on Mojo Projects**: The removal of `DTypePointer` from Mojo requires projects to update their code, transitioning to use `UnsafePointer` instead.
   - There is a call for clear guidelines to assist developers with this transition, especially for prevalent usages in existing Mojo projects.
- **Comparability of SIMD Types**: A discussion arose around the challenges of establishing a total ordering for SIMD types, emphasizing the conflict between generic programming and specific comparisons.
   - It was suggested that introducing a `SimdMask[N]` type could help bridge the gap between architecture-dependent behavior and programming expectations.
- **Contributions to Mojo Compiler Features**: Contributors expressed a desire to simplify the Mojo library through improved generic programming and iterator implementations while addressing current compiler issues.
   - There is an ongoing effort to streamline the API, particularly concerning overloads related to sorting and type handling.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/manual/parameters/#:~:text=Parameter%20inference%E2%80%8B&text=Mojo%20can%20also%20infer%20the,a%20constructor%20or%20static%20method.&text=Note%20that%20you%20can%20create,it%20from%20the%20value%20argument.">Parameterization: compile-time metaprogramming | Modular Docs</a>: An introduction to parameters and compile-time metaprogramming.</li><li><a href="https://github.com/modularml/mojo/issues/3126">[BUG] `List` doesn&#39;t work at compile time. · Issue #3126 · modularml/mojo</a>: Bug description As title. At least List.__getitem__ doesn&#39;t work. Steps to reproduce fn main(): alias l = List[Int](1, 2, 3) print(l[0]) # prints 0 System information Mojo 2024.6.2614 (366c690a) o...</li><li><a href="https://docs.modular.com/mojo/manual/parameters/#:~:text=Parameter%20inference%E2%80%8B&text=Mojo%20">Parameterization: compile-time metaprogramming | Modular Docs</a>: An introduction to parameters and compile-time metaprogramming.
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1265384432429170783)** (57 messages🔥🔥): 

> - `Factorio Automation Mod`
> - `GPT-4o Mini Fine-Tuning`
> - `Mistral Large 2 Release`
> - `Reddit's Content Policy Controversy`
> - `Arxiv2Video Generator` 


- **Factorio Automation Mod released**: A new mod called [factorio-automation-v1](https://github.com/naklecha/factorio-automation) has been released, allowing agents to perform various game actions like crafting and mining.
   - It offers a great playground for agents to test their capabilities within the game.
- **GPT-4o Mini Fine-Tuning Launch**: OpenAI has launched fine-tuning for **GPT-4o mini**, now accessible to tier 4 and 5 users, with the first 2M training tokens daily being free until September 23.
   - Members discussed evaluations comparing fine-tuned **GPT-4o mini** against **Llama-3.1-8b**, noting some performance inconsistencies.
- **Mistral Large 2's Impressive Features**: [Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) has been unveiled with **123 billion parameters**, offering support for multiple languages and strong code ability.
   - It features open-weights for non-commercial use and is designed for long-context applications.
- **Reddit's Content Policy Raises Eyebrows**: A discussion emerged around Reddit's public content policy, with members expressing concerns about the control Reddit exerts over user-generated content.
   - Many believe that users should have a choice over their content and argue that the policy's ambiguity raises significant issues.
- **Arxiv2Video Generator Showcased**: An open-sourced **Arxiv2Video generator** was introduced, with a demo created for the **Herd of Llamas Paper Club**.
   - This tool, showcased by @aditya_advani, produces engaging video summaries of academic papers and invites further interest and potential collaborations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1816161458629271673">Tweet from Alex Albert (@alexalbert__)</a>: We received tons of great submissions to the Build with Claude June 2024 contest from @AnthropicAI devs!  Here are the 3 winning projects, each receiving $10k in Anthropic API credits:</li><li><a href="https://x.com/alexalbert__/status/1816161464320942279">Tweet from Alex Albert (@alexalbert__)</a>: In-line doc editor by @baygross and @MatthewSlotkin  Claude 3.5 Sonnet powered tool that reads your doc and drops in comments and suggestions right where you need them.</li><li><a href="https://x.com/aaronpholmes/status/1816102562031927298">Tweet from aaron holmes (@aaronpholmes)</a>: New: OpenAI is on track to lose $5 billion this year, we estimate based on internal financial data and sources.   OpenAI expects to spend around $4b on Azure bills for running ChatGPT and other infere...</li><li><a href="https://x.com/paulgauthier/status/1816198047690289518">Tweet from Paul Gauthier (@paulgauthier)</a>: Mistral Large 2 (2407) scored only 60% on aider&#39;s code editing benchmark. This puts it just ahead of the best GPT-3.5 model. It doesn&#39;t seem able to reliably use search/replace to efficiently ...</li><li><a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: Today, we are announcing Mistral Large 2, the new generation of our flagship model. Compared to its predecessor, Mistral Large 2 is significantly more capable in code generation, mathematics, and reas...</li><li><a href="https://x.com/openaidevs/status/1815836887631946015?s=46">Tweet from OpenAI Developers (@OpenAIDevs)</a>: Customize GPT-4o mini for your application with fine-tuning. Available today to tier 4 and 5 users, we plan to gradually expand access to all tiers. First 2M training tokens a day are free, through Se...</li><li><a href="https://x.com/naklecha/status/1815808346735378487?s=46">Tweet from naklecha (@naklecha)</a>: today, i&#39;m excited to release factorio-automation-v1. using this mod, your agent can perform game actions like crafting, pathfinding, mining, researching etc. this mod can act as a good playground...</li><li><a href="https://x.com/aditya_advani/status/1816187840163987654">Tweet from Aditya P. Advani (@aditya_advani)</a>: @latentspacepod @lvdmaaten @swyx @vibhuuuus @picocreator @eugeneyan In the spirit of rapid-fire recaps, my Open Source Arxiv2Paper generator ELDO made this 2 min video for the club&#39;s viewing pleas...</li><li><a href="https://x.com/dchaplot/status/1816132981377097883">Tweet from Devendra Chaplot (@dchaplot)</a>: Super excited to announce Mistral Large 2 - 123B params - fits on a single H100 node - Natively Multilingual - Strong code & reasoning - SOTA function calling - Open-weights for non-commercial usage  ...</li><li><a href="https://x.com/corbtt/status/1815843764960911549">Tweet from Kyle Corbitt (@corbtt)</a>: @altryne @eugeneyan EVALS RUNNING</li><li><a href="https://x.com/alexalbert__/status/1816161462248947825">Tweet from Alex Albert (@alexalbert__)</a>: Claude + infinite canvas by @TodePond  An infinite canvas web app where Claude 3.5 Sonnet generates and interprets drawings, combining text and vision prompts.</li><li><a href="https://x.com/corbtt/status/1815891110696477099">Tweet from Kyle Corbitt (@corbtt)</a>: Ok like 100 of y&#39;all have dm&#39;d me asking what happens if you compare to fine-tuned 4o mini. I have the results for 3/4 tasks below! Some thoughts:   - Post-fine-tuning, Llama 3.1 8B still outp...</li><li><a href="https://support.reddithelp.com/hc/en-us/articles/26410290525844-Public-Content-Policy">Public Content Policy</a>: This is a policy about how we handle information that is made public on Reddit. This is not a privacy policy. Please consult our privacy policy for how we collect, use, and share your personal/priv...</li><li><a href="https://www.reddit.com/r/reddit4researchers/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://news.ycombinator.com/item?id=41057033">Google is the only search engine that works on Reddit now, thanks to AI deal | Hacker News</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1265711125941452802)** (1 messages): 

> - `Llama 3 Paper Club`
> - `Cursor's AI Developer Tools`
> - `Asia LLM Paper Club` 


- **Emergency Paper Club on Llama 3**: Join the **emergency paper club** in 2 hours to discuss @lvdmaaten et al's *The Llama 3 Herd of Models*, an early contender for the **POTY Awards**. More details are on [Latent Space Pod](https://x.com/latentspacepod/status/1816151808357908698).
   - Members including @swyx, @vibhuuuus, and @eugeneyan will be present to explore this significant topic in detail.
- **Cursor's Co-founder Discusses AI Tools**: A special session will feature the co-founder of **Cursor** discussing *Cursor, Composer, and AI developer tools* in an upcoming meeting. This is a chance to get insights directly from the source.
   - The exact date and time for this discussion were not provided, but it's set to be an important event for those interested in AI development.
- **Don't Miss the Asia LLM Paper Club**: Make sure to participate in the **Asia LLM Paper Club** for engaging discussions focused on recent advancements and papers in the field. You can find more about the meeting [here](https://lu.ma/jpyss688).
   - This club continues to be a key gathering for those invested in LLM research and collaboration.



**Link mentioned**: <a href="https://x.com/latentspacepod/status/1816151808357908698">Tweet from Latent.Space (@latentspacepod)</a>: 🚨 EMERGENCY PAPER CLUB  The @latentspacepod discord is meeting in 2hrs to talk thru @lvdmaaten et al&#39;s The Llama 3 Herd of Models, early contender to win the POTY* Awards!  Join us (link below) w...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1265397223173394524)** (5 messages): 

> - `LlamaParse Features`
> - `MongoDB AI Applications Program (MAAP)`
> - `Mistral Large 2 Capabilities`
> - `Structured Extraction in LLMs` 


- **LlamaParse Unleashes Markdown and More**: In a recent video, features of **LlamaParse** were showcased, including options for **Markdown and plain text output**, and **JSON mode** for richer metadata extraction.
   - The tool also supports **multi-language** output for improved OCR, making it a versatile addition to any workflow. [Watch the video here](https://t.co/RUWJ0Z2NMn).
- **MongoDB Launches AI Applications Program**: The newly announced **MongoDB AI Applications Program (MAAP)** is designed to assist organizations in building and deploying modern AI-enhanced applications quickly.
   - It provides reference architectures and a comprehensive technology stack with leading tech integrations, enabling enterprises to **accelerate their AI journey**. [Learn more about MAAP](https://t.co/rCz3DfUe3A).
- **Mistral Large 2 Brings Advanced Function Calling**: **Mistral Large 2** features state-of-the-art **function calling capabilities**, with day 0 support for structured outputs and agents.
   - This release aligns with enhanced **function calling** and **structured outputs**, providing useful resources like cookbooks for users to explore. [Check out the cookbooks](https://t.co/ho02wDbGpZ).
- **Structured Extraction for LLMs Released**: The latest release offers **structured extraction capabilities** for LLM-powered ETL, RAG, and agent pipelines, supporting both async and streaming functionalities.
   - By defining a **Pydantic object** and attaching it to the LLM, users can significantly enhance their data processing workflows. [Discover more here](https://t.co/0wLX2Tf1P6).



**Link mentioned**: <a href="https://t.co/rCz3DfUe3A">MongoDB AI Applications Program</a>: Get the support you need to accelerate your AI application journey and launch with confidence and speed.

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1265450480474132541)** (52 messages🔥): 

> - `SubQuestionQueryEngine`
> - `Llama 3.1 Testing`
> - `RAG Setup for PDF Display`
> - `Text-to-SQL Pipeline Optimization`
> - `ReAct Agent Behavior` 


- **Streaming Responses with SubQuestionQueryEngine**: Members discussed using `SubQuestionQueryEngine.from_defaults` with the goal of streaming final responses from the LLM to reduce latency.
   - A solution involving `get_response_synthesizer` and utilizing token printing techniques was shared, but there were challenges implementing it.
- **Skepticism about Llama 3.1 Metrics**: Some users expressed distrust in the metrics provided by Meta for Llama 3.1, with discussions on its usability for RAG evaluations.
   - Concerns were raised about whether using models like `llama3:70b-instruct-q_5` would be beneficial for such tasks.
- **Optimizing RAG with PDF Interfaces**: A discussion focused on strategies to improve a RAG setup involving the display of PDFs through buttons in a web interface.
   - Suggestions included avoiding large projects with many PDFs and using libraries that can directly handle PDF files without conversion.
- **Improving Text-to-SQL Pipeline Speed**: Users highlighted slow response times in their Text-to-SQL pipelines and sought advice on potential optimizations.
   - Recommendations included using faster LLMs or condensing inputs; streaming output for a better user experience was also explored.
- **ReAct Agent Hallucinations**: A member reported that their ReAct agent would continuously hallucinate when responding to inputs, following an incorrect processing loop.
   - Discussions pointed towards the LLM's inability to adhere to expected output formats and suggestions were made for adding stop tokens to improve behavior.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/pull/14439#issuecomment-2195513666">Add Context-Only Response Synthesizer by Androbin · Pull Request #14439 · run-llama/llama_index</a>: Description Motivation: The OpenAIAgent with tool usage performs worse than the ContextChatEngine, as the outer LLM (agent), the inner LLM (query engine), and the retriever are effectively playing ...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1265511825999532042)** (1 messages): 

> - `RAG pipeline evaluation`
> - `Custom RAG evaluation system`
> - `RAGAS framework`
> - `Improving evaluation methods` 


- **Evaluating RAG Pipeline Effectively**: A member expressed the need for **professional advice** on evaluating their RAG pipeline after using the **RAGAS** framework, which they found to be too random.
   - They noted they are now developing a **custom RAGAS-like evaluation system** to gain more control over the metrics.
- **Seeking Suggestions for RAG Evaluation Packages**: The member is looking for **improvements** to their custom evaluation method and asks if others can suggest better packages for their needs.
   - They appreciated any **advice** shared on enhancing their system or alternatives worth considering.


  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1265473423707082764)** (34 messages🔥): 

> - `Cohere Dashboard Issues`
> - `Model Testing Appreciation`
> - `Server Performance Concerns`
> - `Feature Suggestions for Tools`
> - `Community Introductions` 


- **Cohere Dashboard reloading issue**: A member mentioned that their **Cohere account dashboard** appeared to be constantly reloading, while another confirmed it was fine on their end.
   - This sparked a brief discussion about potential glitches and rate limiting.
- **Appreciation for Command R Plus**: With each new release of highly anticipated models like **Llama 3.1**, a member expressed growing appreciation for **Command R Plus**.
   - Another user mentioned creating a playground for **model comparisons** to further explore this sentiment.
- **Server performance inquiries**: Concerns were raised regarding the server potentially being down, but others confirmed **full operational status**.
   - Suggestions included checking for possible rate limiting affecting user experience.
- **Feature suggestions for Cohere Tools**: One member proposed the ability to use tools midway through conversations in **Cohere**, like invoking a web search on request.
   - After some initial confusion, it was acknowledged that some of these features already exist.
- **Introductions in the community**: New members introduced themselves, discussing their background in **NLP and NeuroAI** and expressing excitement about the server.
   - A discussion on experiences with **Command-R+** highlighted its impact compared to other models like **NovelAI**.


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1265409293314953239)** (6 messages): 

> - `zenbase/core launch`
> - `DSPy optimizers` 


- **Zenbase/Core Python Library Launch**: A member announced that **zenbase/core** is now launched, allowing users to employ **DSPy’s optimizers** in their existing Python code like Instructor and LangSmith.
   - They requested support through **retweets, likes, and stars** on their [Twitter post](https://twitter.com/cyrusofeden/status/1815858216389300383?s=61&t=WwA-PFs585hhcOplJkLRbQ).
- **Member Engagement on Twitter**: Another member enthusiastically responded, confirming they have **liked and retweeted** the launch announcement, expressing excitement about the new library.
   - The overall sentiment in the channel reflects a positive reaction towards **recent launches** and developments.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

batmanosama: done
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1265406190473445567)** (20 messages🔥): 

> - `Typed Predictors in DSPy`
> - `Internal Steps Execution Visibility`
> - `Small Language Models Future`
> - `Contributing to DSPy Repository`
> - `Model Fine-Tuning and Distillation` 


- **Typed Predictors causing output issues**: A user is facing trouble with typed predictors in DSPy not returning correctly structured output, despite following setup tips.
   - Another user suggested configuring experimental features with `dspy.configure(experimental=True)` for potential improvement.
- **Inspecting internal program execution**: Users discussed various methods to see the internal steps and prompts during program execution, with suggestions like `inspect_history`.
   - One user expressed a need for more visibility into model outputs, even during type checking failures.
- **Advocating for Small Language Models**: A member shared an article promoting the efficiency and benefits of small language models that can run on minimal hardware.
   - They emphasized **privacy** benefits and the suitability of small models for edge devices while maintaining useful intelligence.
- **Opportunity to contribute to DSPy examples**: Another user inquired about contributing examples to the DSPy repository, indicating readiness to create beginner-friendly content.
   - Responses confirmed that there is a need for diverse examples, and contributions can be added to the `/examples` directory.
- **Questions on model fine-tuning with DSPy**: A user asked whether they could fine-tune and distill models like Llama 8B using DSPy without additional neural network code.
   - Their curiosity highlighted the importance of understanding the capabilities of DSPy in relation to model training techniques.



**Link mentioned**: <a href="https://medium.com/thoughts-on-machine-learning/small-language-models-are-the-future-6e8909567198">Small Language Models are the Future</a>: My Thesis: Small language models (SLM)— models so compact that you can run them on a computer with just 4GB of RAM — are the future. SLMs…

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1265454212305784852)** (4 messages): 

> - `Learning Tinygrad`
> - `GPU and Uops issues`
> - `OpenCL and Python challenges`
> - `Checking Closed PRs` 


- **Learning the ropes of Tinygrad**: One member mentioned he is still on his list of things to learn concerning **Tinygrad** and **transformers**.
   - *It's a work in progress,* he added, indicating a desire for gradual understanding.
- **GPU and Uops are still a concern**: A member is struggling with making the GPU and Uops turn **green**, while successfully using numpy and pytorch shapes.
   - He seeks hints on fixing **OpenCL** and **Python device** issues, stating, *I guess should be full green*.
- **Advice to check closed PRs**: One user suggested that checking the **closed PRs** could provide insights into the ongoing issues.
   - This user aims to help clarify any uncertainties around making progress.
- **Understanding OpenCL and Python limitations**: Another member pointed out that **OpenCL** and **Python** fail due to their inability to utilize views, which complicates matters.
   - They noted that *a simple 'bitcast' will not work with shape changing bitcasts*, pointing out specifics to check with **DEBUG=x**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1265431881017589781)** (19 messages🔥): 

> - `Molecular Dynamics Engine in tinygrad`
> - `Custom Runtime Implementation`
> - `Neural Network Potentials`
> - `PPO Algorithm in Beautiful CartPole` 


- **Implementing a Molecular Dynamics Engine**: A member is working with a group to implement a **Molecular Dynamics engine** in tinygrad, using neural networks to predict energies of configurations, while facing issues with gradient utilization.
   - Another member suggested using input gradient tracking and modifying weight updates to avoid issues encountered during backpropagation.
- **Guide for Custom Runtime in tinygrad**: A user shared a guide on how to implement a **custom runtime** for tinygrad, highlighting that support for new hardware should be simple to add.
   - They asked for clarifications on technical terms such as `global_size` and `local_size`, which were explained as parameters for kernel execution counts in the operational context.
- **Understanding Neural Network Potentials**: The discussion revealed that the energy used in the Molecular Dynamics engine is based on **Neural Network Potentials (NNP)**, emphasizing the need for efficient calculations.
   - The conversation included suggestions on how to optimize the backpropagation process to improve training results.
- **PPO Algorithm in Beautiful CartPole**: A member inquired about the purpose of the `.sum(-1)` operation in the **PPO** implementation for the Beautiful CartPole environment, pointing to a specific line in the code.
   - This illustrates the collaborative nature of understanding nuances in reinforcement learning implementations among community members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.tinygrad.org/runtime/overview/">Runtime Overview - tinygrad docs</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/tinygrad/codegen/lowerer.py#L155-L156">tinygrad/tinygrad/codegen/lowerer.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://gist.github.com/python273/0dc136fbc63559188ab279c07329e891">TinyJit vis WIP</a>: TinyJit vis WIP. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py#L231">spinningup/spinup/algos/pytorch/ppo/ppo.py at 20921137141b154454c0a2698709d9f9a0302101 · openai/spinningup</a>: An educational resource to help anyone learn deep reinforcement learning. - openai/spinningup</li><li><a href="https://mesozoic-egg.github.io/tinygrad-notes/addingaccelerator.html">How to add a custom accelerator?</a>: Tutorials on tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/tinygrad/runtime/ops_python.py#L31">tinygrad/tinygrad/runtime/ops_python.py at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/examples/whisper.py#L119">tinygrad/examples/whisper.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/blob/baface413a22a4e69ab892cd83d7c6748e9da890/examples/whisper.py#L41-L45">tinygrad/examples/whisper.py at baface413a22a4e69ab892cd83d7c6748e9da890 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1265411491192897536)** (15 messages🔥): 

> - `3.1 release interviews`
> - `MPS support PR`
> - `LoRA issues`
> - `Conflicts in contributions`
> - `Git workflow optimizations` 


- **Countdown to 3.1 and Cool Interviews**: Members inquired about whether there would be any cool [interviews](https://github.com/pytorch/torchtune/pull/790) released along with the 3.1 version, similar to those for Llama3.
   - This raises interest for potential insights and discussions that might accompany the new release.
- **MPS Support PR Gains Attention**: A new pull request ([#790](https://github.com/pytorch/torchtune/pull/790)) was highlighted which adds support for MPS on local Mac computers, checking for BF16 compatibility.
   - Context suggests this PR could resolve major testing hurdles for those using MPS devices.
- **LoRA Functionality Issues Persist**: Discussed issues surrounding **LoRA** functionality, noting it did not work during a previous attempt and was previously impacted by hardcoded **CUDA** paths.
   - Members exchanged thoughts on specific errors encountered, highlighting ongoing challenges in implementation.
- **Endless Battle with Git Conflicts**: Members expressed frustrations with frequent new conflicts in their contributions, believing it feels like an endless cycle, especially after fixing existing conflicts.
   - It was suggested that new conflicts might stem from the workflow, indicating a potential need for tweaks.
- **Optimizing Git Workflow to Reduce Conflicts**: Discussion around refining git workflows to minimize the occurrence of new conflicts constantly arose, emphasizing collaboration.
   - Suggesting improvements in contribution practices may help lighten the burden of merging challenges.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/790">MPS support by maximegmd · Pull Request #790 · pytorch/torchtune</a>: Context  For testing purposes it can be useful to run directly on a local Mac computer.  Changelog  Checks support for BF16 on MPS device. Added a configuration targeting MPS, changes to path were ...

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1265412754194301110)** (1 messages): 

> - `Pad ID Bug`
> - `Pull Request #1211` 


- **Fixing the Pad ID Bug**: A member pointed out that the **pad id** should not be showing up in generate functionality, identifying it as an important bug.
   - In response, a Pull Request was created to prevent **pad ids** and special tokens from displaying, detailed in [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211).
- **Details of Pull Request #1211**: The [Pull Request #1211](https://github.com/pytorch/torchtune/pull/1211) aims to address the issue regarding pad id by modifying the implementation in **utils.generate**.
   - The context of the PR mentions it is meant to fix a bug, ensuring pad ids are implicitly handled correctly.



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/pull/1211">Prevent pad ids, special tokens displaying in generate by RdoubleA · Pull Request #1211 · pytorch/torchtune</a>: Context What is the purpose of this PR? Is it to   add a new feature  fix a bug  update tests and/or documentation  other (please add here)  Pad ID is implicitly assumed to be 0 in utils.generate, ...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1265526704722677823)** (6 messages): 

> - `Hugging Face Agents`
> - `Job Opportunities in Python`
> - `HNSW IVFFLAT Index Issues`
> - `SQLite Server Storage Management` 


- **Exploring Agents with Hugging Face Models**: A user inquired if anyone had experience working with **Agents using Hugging Face models** (LLM or Chat). Another user responded that they’ve done a lot with agents across **OpenAI** and **Azure**, as well as local LLMs via **Ollama**.
- **Job Seeking for Python Developers**: A member expressed the need for work, stating, **'anyone looking to hire me? I need to pay my bills.'** They highlighted their proficiency in **Python**.
- **HNSW and IVFFLAT Index Challenges in Aurora**: A member reported difficulties in creating **HNSW** or **IVFFLAT** indexes with **3072 dimensions** on **Aurora PGVECTOR**. They later shared their solution, which involved using **halfvec**.
- **Managing SQLite Server Threads**: A user asked how to check their **SQLite server storage** to monitor message and thread usage. They were curious about how to remove previous threads when using **Langgraph**.


  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1265624679138328607)** (1 messages): 

> - `Scaling LangServe`
> - `OSError handling`
> - `Handling concurrent requests` 


- **Scaling LangServe hits open files limit**: The user is experiencing an **OSError: [Errno 24] Too many open files** when the LangServe app receives around **1000 concurrent requests**.
   - They shared an [issue on GitHub](https://github.com/langchain-ai/langserve/issues/714) regarding the problem and are seeking advice on how to manage the request load effectively.
- **Seeking solutions for high request loads**: The user is looking for strategies to effectively handle a **high volume of requests** in their LangServe application.
   - They hope to find methods to prevent errors related to system resource limitations while scaling up their application.



**Link mentioned**: <a href="https://github.com/langchain-ai/langserve/issues/714">Scaling to production -&gt; OSError: [Errno 24] Too many open files socket.accept() out of system resource  · Issue #714 · langchain-ai/langserve</a>: Problem When my LangServe app gets ~1000 concurrent requests, it breaks with error: OSError: [Errno 24] Too many open files socket.accept() out of system resource Mitigation/quickfix I&#39;ve checked ...

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1265418002770165790)** (2 messages): 

> - `Fully Local Tool Calling with Ollama`
> - `AI Code Reviewer` 


- **Request for Notebook on Fully Local Tool Calling**: A member requested the notebook for the **'Fully local tool calling with Ollama'** session that was presented earlier today.
   - They expressed appreciation for the content and emphasized its excellence.
- **Introduction of AI Code Reviewer Tool**: A member shared a [YouTube video](https://youtu.be/g_VRsjpC4e8) titled **'AI Code Reviewer Ft. Ollama & Langchain'**, introducing a CLI tool aimed at enhancing code review processes.
   - The video highlights features powered by **LangChain** and showcases how it can revolutionize code assessment.



**Link mentioned**: <a href="https://youtu.be/g_VRsjpC4e8">AI Code Reviewer Ft. Ollama &amp; Langchain</a>: Welcome to Typescriptic! In this video, we introduce our Code Reviewer, a CLI tool designed to revolutionize the way you review your code. Powered by LangCha...

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1265406571203137587)** (6 messages): 

> - `Llama 3.1 405 B`
> - `Mistral Large 2`
> - `API usage`
> - `Developer opportunities`
> - `LM Studio excitement` 


- **Llama 3.1 405 B impresses with ease of use**: **Llama 3.1 405 B** performs fantastically out of the box with OpenInterpreter, offering an effortless experience.
   - In contrast, **gpt-4o** requires constant reminders about capabilities, making 405b a superior choice for multitasking.
- **Cost-effective API usage with Nvidia**: A user shared that **Nvidia** provides **1000 credits** upon signup, where 1 credit equals 1 API call.
   - This incentive opens up more accessibility for experimenting with APIs.
- **Mistral Large 2 rivals Llama 3.1 405 B**: **Mistral Large 2** reportedly performs comparably to **Llama 3.1 405 B**, particularly noted for its speed.
   - The faster performance may be due to lower traffic on Mistral's endpoints compared to those of Llama.
- **Interest in developer contributions**: There was a query about the potential for a skilled developer to contribute to an unspecified project.
   - This highlights an ongoing interest in expanding developer support and collaboration.
- **Excitement for integrating with LM Studio**: A user expressed enthusiasm for using **Llama 3.1 405 B** with **LM Studio**, indicating a promising integration.
   - This suggests anticipation for enhanced capabilities and functionalities through this combination.


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1265384282298384515)** (1 messages): 

> - `Device Shipping Timeline` 


- **Inquiry on Device Shipping Timeline**: A user inquired about the timeline for when the device will ship, expressing anticipation with a straightforward request for updates.
   - The question highlights ongoing interest and concern regarding the delivery schedule amid the community.
- **Community Anticipation for Device Delivery**: The inquiry reflects a broader sentiment among users eager for updates on the device shipping date, connecting them to the brand.
   - Discussions around shipping timelines have become a point of engagement, showcasing the community's investment in the product.


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1265726346093396152)** (2 messages): 

> - `Llama 3.1`
> - `OpenInterpreter Database Integration`
> - `Database Complexities` 


- **Llama 3.1 connects with databases for free**: [MikeBirdTech](https://x.com/MikeBirdTech/status/1816163862208766137) noted that **Llama 3.1** can interact with your database at no cost through **OpenInterpreter**, emphasizing savings on paid services.
   - *It's also fully offline and private, nobody else needs to see your data,* highlighting its **privacy benefits**.
- **Concerns over complex databases using Llama 3.1**: A member raised a concern that for **complex databases** involving joins across tables, this solution may not be effective.
   - They expressed appreciation for sharing the information, remarking on the **well-done** execution despite the limitations.



**Link mentioned**: <a href="https://x.com/MikeBirdTech/status/1816163862208766137">Tweet from Mike Bird (@MikeBirdTech)</a>: Llama 3.1 talks to your database for free with @OpenInterpreter   Why pay for a talk-to-your-database service?  Save money!  It&#39;s also fully offline and private, nobody else needs to see your data

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1265423923395301518)** (5 messages): 

> - `Llama 3.1 release`
> - `LAION metadata download issues`
> - `LAION datasets legality`
> - `YouTube polls` 


- **Llama 3.1: Meta's Open Source Breakthrough**: Meta recently launched **Llama 3.1 405B**, hailed as the first-ever **open-sourced frontier AI model**, outperforming competitive models like GPT-4o on various benchmarks. For more insights, check this [YouTube video](https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61) featuring Mark Zuckerberg discussing its implications.
   - The reception highlights the model's potential impact on AI research and open-source contributions.
- **Trouble Downloading LAION2B-en Metadata**: A member expressed difficulties in locating and downloading the **LAION2B-en metadata** from Hugging Face, questioning if others faced the same problem. Responses highlight ongoing challenges with accessibility, indicating it’s a common frustration.
   - Someone linked to [LAION maintenance notes](https://laion.ai/notes/laion-maintenance/) for further clarification on the situation.
- **LAION Datasets in Legal Limbo**: Discussion revealed that LAION datasets are currently in **legal limbo**, with access to official versions restricted. Alternatives are available, but it's advised to only utilize unofficial datasets for urgent research needs.
   - Members noted the ongoing complexities surrounding data legality in the AI community.
- **YouTube Polls: A Nostalgic Debate**: A member shared a [YouTube poll](http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS) asking which **90's movie had the best soundtrack**, sparking nostalgia among viewers.
   - This prompts members to reflect on their favorite soundtracks from the era, connecting through shared cultural experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://youtube.com/post/Ugkxeb5mZpY-AdjyD57ncd8Q-70Dk3CkrBJb?si=rWt2_l7TQwl9z1MS">Poll from Innuendo</a>: Which 90&#39;s movie had the best soundtrack?</li><li><a href="https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61">Mark Zuckerberg on Llama 3.1, Open Source, AI Agents, Safety, and more</a>: Meta just released Llama 3.1 405B — the first-ever open-sourced frontier AI model, beating top closed models like GPT-4o across several benchmarks. I sat dow...
</li>
</ul>

</div>
  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

spirit_from_germany: https://youtu.be/Vy3OkbtUa5k?si=mBhzPQqDLgzDEL61
  

---


### **Alignment Lab AI ▷ #[open-orca-community-chat](https://discord.com/channels/1087862276448595968/1124000038205530182/1265490712938545196)** (2 messages): 

> - `Copyright issues in ML datasets`
> - `Identifying non-distilled data`
> - `Legal considerations` 


- **Legal Clarity on ML Dataset Copyright**: A member discussed that most of the dataset generated by an **ML model** may not be copyrightable, indicating that it's not considered a truly creative work.
   - They noted that the **non-GPT-4** generated content should be solidly under **MIT licensing**, but acknowledged it's a grey area amidst ongoing legal discussions.
- **Query on Non-Distilled Data Identification**: A follow-up question was raised about how to identify the **rows which are non-distilled** in the dataset.
   - This indicates an ongoing interest in ensuring clarity and organization in managing dataset contents.


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1265715877664264353)** (1 messages): 

> - `Translation model fine-tuning`
> - `CPO approach`
> - `ALMA models performance` 


- **Experimenting with DPO for Translation Models**: A member is inquiring if anyone has successfully fine-tuned a translation model using **DPO** due to insights gathered from the **CPO** paper.
   - They specifically refer to how **moderate-sized LLMs** don't match state-of-the-art models and point to the [CPO paper](https://arxiv.org/abs/2401.08417) for more details.
- **CPO's Role in Improving Translation Models**: The **CPO** approach aims to address the shortcomings of supervised fine-tuning in machine translation, highlighting issues in reference data quality.
   - By training models to avoid generating just *adequate* translations, CPO enhances the performance of models like **ALMA-R**, which capitalizes on limited datasets.
- **ALMA-R's Standout Performance**: When applying **CPO** to ALMA models, significant improvements were noted despite only using **22K parallel sentences** and **12M parameters**.
   - The resulting model, **ALMA-R**, can compete with or even surpass conventional encoder-decoder models.



**Link mentioned**: <a href="https://arxiv.org/abs/2401.08417">Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation</a>: Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation mod...

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/)** (1 messages): 

intheclouddan: <@1197944730378588170> <@811015724877217803> I'd be interested in NYC in late august
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1265636540768583785)** (1 messages): 

> - `Feature Stores`
> - `ML Operations`
> - `Scalability`
> - `Data Management`
> - `Feature Governance` 


- **Maximizing ML Efficiency with Feature Stores**: A **live session** on [Leveraging Feature Stores](https://tinyurl.com/yfjscesh) is scheduled for **July 31st, 2024, at 11:00 AM EDT**, targeting ML Engineers, Data Scientists, and MLOps professionals.
   - The session will cover **building automated pipelines**, managing unreliable data, and showcasing advanced use cases for enhancing scalability and performance.
- **Tackling Inconsistency in ML Data**: The webinar will focus on eliminating differences between **serving and training data** to develop scalable and reproducible models.
   - Challenges such as **inconsistent data formats** and feature duplication will also be addressed to improve collaboration within ML teams.
- **Strategies for Robust Feature Governance**: Participants will learn about implementing **robust strategies for feature governance and versioning**, which are vital for effective ML lifecycle management.
   - Insights and practical tools will help attendees refine their ML processes and drive operations forward.



**Link mentioned**: <a href="https://tinyurl.com/yfjscesh">Leveraging Feature Stores in ML</a>: Join Hudson Buzby to learn about Advancing ML Operations and Scalability

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1265387661103726703)** (1 messages): 

> - `Accelerator application deadline`
> - `Upcoming events`
> - `Zero Shot Tokenizer Transfer`
> - `AutoFix open source issue fixer` 


- **Accelerator Application Deadline Approaches**: The application deadline for the accelerator program is fast approaching, offering a **12 week program** with up to **100k in non-diluted funds** for projects.
   - A demo day with Mozilla is also planned, and members are encouraged to ask their **questions** [here](https://discord.com/channels/1089876418936180786/1245083732319408195).
- **Two More Exciting Events Coming Up**: Reminder about two upcoming events this month featuring the work of notable participants, bringing fresh insights to the community.
   - These events are brought to you by two members, further bolstering community engagement.
- **Insightful Zero Shot Tokenizer Transfer Discussion**: A session titled **Zero Shot Tokenizer Transfer** with Benjamin Minixhofer is scheduled, aiming to delve into advanced techniques in tokenizer implementations.
   - Details and participation links can be found [here](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732).
- **AutoFix: Open Source Issue Fixer Launch**: An announcement was made regarding **AutoFix**, an open source issue fixer that can submit PRs from Sentry.io, aiding developers in streamlining their workflows.
   - More information on the project can be accessed [here](https://discord.com/channels/1089876418936180786/1089876419926032396/1261387457652592732).


  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1265590495103680593)** (1 messages): 

> - `Meta's Llama3.1 Paper`
> - `Llama3 training insights`
> - `Hallucination prevention techniques` 


- **Llama3.1 Paper: A Treasure for Open Source**: The new [Llama3.1 paper from Meta](https://threadreaderapp.com/thread/1815789501026861308) is hailed as incredibly **valuable** for the open source community, prompting discussions about its profound insights.
   - *One member joked* that it contains so much **alpha** that *you have to read it multiple times like a favorite movie*.
- **Training a 405B Model with 15T Tokens**: The paper reveals that the model with **405 billion parameters** was trained using **~15 trillion tokens**, which was predicted by extrapolating their scaling laws.
   - *The scaling law suggests* training a **402B parameter model** on **16.55T tokens** to achieve optimal results.
- **Insights on Network Topology**: It includes a surprisingly detailed description of the **network topology** used for their **24k H100 cluster**.
   - Images shared in the thread illustrate the **architecture**, demonstrating the scale of the infrastructure.
- **Training Interruptions Due to Server Issues**: Two training interruptions during Llama3-405b's process were attributed to the **'Server Chassis'** failing, humorously suggested to be caused by someone's mishap.
   - As a consequence, **148 H100 GPUs** were lost during pre-training due to these failures.
- **Discussion on Hallucination Prevention Benchmarks**: A brief conversation with a Meta engineer raised concerns about the need for better **benchmarks** in **hallucination prevention** techniques.
   - The member shared that *anyone else working on this* vital area should engage in further discussions.



**Link mentioned**: <a href="https://threadreaderapp.com/thread/1815789501026861308">Thread by @jphme on Thread Reader App</a>: @jphme: Live tweeting the most interesting insights from @Meta´s new Llama3 paper 1. How did the arrive at a 405b model trained with ~15T tokens? &quot;Extrapolation of the resulting scaling law to 3....

  

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

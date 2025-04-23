---
id: 0c6622b1-3e7e-4ecf-83e0-d6c7211c7d29
title: >-
  Project Stargate: $500b datacenter (1.7% of US GDP) and Gemini 2 Flash
  Thinking 2
date: '2025-01-22T01:56:21.007400Z'
original_slug: ainews-project-stargate-500b-datacenter-17-of-us
description: >-
  OpenAI and Softbank have announced Project Stargate, a major US "AI Manhattan
  project" supported by top tech companies aiming for large-scale AI
  advancements, though its immediate impact remains limited. Meanwhile, Noam
  Shazeer revealed a significant update to Gemini 2.0 Flash with a 1 million
  token context length, enabling more powerful AI applications today.
tags:
  - openai
  - softbank
  - project-stargate
  - noam-shazeer
  - gemini-2-0
  - long-context
  - microsoft
  - nvidia
---


<!-- buttondown-editor-mode: plaintext -->**Masa Son and Noam Shazeer are all you need.**

> AI News for 1/20/2025-1/21/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **4353** messages) for you. Estimated reading time saved (at 200wpm): **450 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Days like these are a conundrum - on one hand, the obvious big earth shattering news is the announcement of [Project Stargate](https://x.com/openai/status/1881830103858172059?s=46), a US "AI Manhattan project" led by OpenAI and Softbank, and supported by Softbank, OpenAI, Oracle, MGX, Arm, Microsoft, and NVIDIA. For scale, the actual Manhattan project [cost $35B inflation adjusted](https://x.com/tanayj/status/1881849682063986843?s=46). 

![image.png](https://assets.buttondown.email/images/e686ff0d-b54a-44c9-b567-0e3c7f927c6d.png?w=960&fit=max)

Although this was [rumored since a year ago](https://www.theinformation.com/articles/microsoft-and-openai-plot-100-billion-stargate-ai-supercomputer?rc=ytp67n), Microsoft's [reduced](https://x.com/smokeawayyy/status/1881801442459033662?s=46) role as exclusive compute partner to OpenAI is prominent by its absence. As with any splashy PR stunt, one should beware [AI-washing](https://x.com/teortaxesTex/status/1881839728250765709), but the project is very serious and should be treated as such.

However, it's not really news you can use today, which is what we aim to do here at your local AI newspaper.

Fortunately, Noam Shazeer got you, with [a second Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1881845901872001293), with another big leap on 2.0 Flash, and 1M long context that you can use today (we will enable in AINews and Smol Talk tomorrow):

![image.png](https://assets.buttondown.email/images/b2247b7e-56ca-48db-ac9c-7b58c7dee477.png?w=960&fit=max)

AI Studio also got a [code interpreter](https://x.com/jack_w_rae/status/1881850281052545140).

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

**Theme 1. DeepSeek R1: Release, Performance, and Strategic Vision**

- **[DeepSeek R1 (Qwen 32B Distill) is now available for free on HuggingChat!](https://hf.co/chat/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)** ([Score: 364, Comments: 106](https://reddit.com/r/LocalLLaMA/comments/1i6jbur/deepseek_r1_qwen_32b_distill_is_now_available_for/)): **DeepSeek R1**, a distillation of **Qwen 32B**, is now accessible for free on **HuggingChat**.
  - **Hosting and Access Concerns**: Users discuss the option to self-host **DeepSeek R1** to avoid logging into **HuggingChat**, with some expressing frustration over the need for an account to evaluate the model. A suggestion was made to use a **dummy email** for account creation to bypass this requirement.
  - **Performance and Technical Issues**: There are reports of performance issues such as the model becoming unresponsive, and discussions on the use of **quantization** (e.g., FP8, 8-bit) and system prompts affecting model performance. Some users noted that **DeepSeek R1** is better at planning than code generation, and others shared tools like [cot_proxy](https://github.com/bold84/cot_proxy) to manage the model's "thought" tags.
  - **Model Comparisons and Preferences**: Comparisons were made between **DeepSeek R1** and other models like **Phi-4** and **Llama 70B**, with some users preferring distilled models for specific tasks like math and nuanced understanding. There is interest in exploring other variants like **Qwen 14B** and the anticipation of **R1 Lite** for improved consistency.


- **Inside DeepSeek’s Bold Mission (CEO Liang Wenfeng Interview)** ([Score: 124, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1i6dlvj/inside_deepseeks_bold_mission_ceo_liang_wenfeng/)): DeepSeek, led by CEO **Liang Wenfeng**, distinguishes itself with a focus on **fundamental AGI research** over rapid commercialization, aiming to shift China's role from a "free rider" to a "contributor" in global AI. Their **MLA architecture** drastically reduces memory usage and costs, with inference costs significantly lower than **Llama3** and **GPT-4 Turbo**, reflecting their commitment to efficient innovation. Despite challenges like U.S. chip export restrictions, DeepSeek remains committed to **open-source development**, leveraging a bottom-up structure and young local talent, which could position them as a viable alternative to the closed-source trend in AI.
  - **DeepSeek's Focus on AGI**: Commenters emphasize that DeepSeek's commitment to AGI, rather than profit, is notable, with some likening their approach to OpenAI's early days. There is skepticism about whether DeepSeek will maintain this open-source ethos long-term or eventually follow a closed-source model like other tech giants.
  - **Leadership and Recognition**: **Liang Wenfeng** is highlighted for his leadership, with a significant mention of his meeting with Chinese Premier **Li Qiang**, indicating high-level recognition and support. This meeting underscores DeepSeek's growing influence and potential impact on AI development in China.
  - **Young Talent and Innovation**: Commenters praise DeepSeek's team for their creativity and innovation, noting that the team consists of young, recently graduated PhDs who have accomplished significant achievements despite not being well-known before joining the company. This highlights the potential of leveraging young talent for groundbreaking advancements in AI.


- **[DeepSeek-R1-Distill-Qwen-1.5B running 100% locally in-browser on WebGPU. Reportedly outperforms GPT-4o and Claude-3.5-Sonnet on math benchmarks (28.9% on AIME and 83.9% on MATH).](https://v.redd.it/5ei4j3c9teee1)** ([Score: 72, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1i6t08q/deepseekr1distillqwen15b_running_100_locally/)): **DeepSeek-R1-Distill-Qwen-1.5B** is running entirely in-browser using **WebGPU** and reportedly surpasses **GPT-4o** and **Claude-3.5-Sonnet** in math benchmarks, achieving **28.9%** on AIME and **83.9%** on MATH.
  - **ONNX** is discussed as a file format for LLMs, with some users noting it offers performance optimization, potentially up to **2.9x faster** on specific hardware compared to other formats like **safetensors** and **GGUF**. However, the general consensus is that these are just different data formats appreciated by different hardware/software setups.
  - **DeepSeek-R1-Distill-Qwen-1.5B** is noted for running entirely in-browser on **WebGPU**, outperforming **GPT-4o** in benchmarks, with an online demo and source code available on [Hugging Face](https://huggingface.co/spaces/webml-community/deepseek-r1-webgpu) and [GitHub](https://github.com/huggingface/transformers.js-examples/tree/main/deepseek-r1-webgpu). However, some users feel it doesn't match **GPT-4o** in real-world applications despite its impressive benchmark results.


**Theme 2. New DeepSeek R1 Tooling Enhances Usability and Speed**

- **[Deploy any LLM on Huggingface at 3-10x Speed](https://i.redd.it/8dsnudtrhdee1.png)** ([Score: 109, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1i6mjxv/deploy_any_llm_on_huggingface_at_310x_speed/)): The image illustrates a **digital dashboard** for "Dedicated Deployments" on Huggingface, showcasing two model deployment cards. The **"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"** model is quantizing at 52% using four **NVIDIA H100 GPUs**, while the **"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"** model is operational on one **NVIDIA H100 GPU**, both recently active and ready for requests.
  - **avianio** introduced a deployment service claiming **3-10x speed** improvement over **HF Inference/VLLM** with a setup time of around 5 minutes, utilizing **H100** and **H200 GPUs**. The service supports about **100 model architectures**, with future plans for multimodal support, and offers cost-effective, private deployments without logging, priced at **$0.01 per million tokens** for high traffic scenarios.
  - **siegevjorn** and **killver** questioned the **3-10x speed claim**, seeking clarification on comparison metrics and hardware consistency. **killver** specifically asked if the claim was valid on the same hardware.
  - **omomox** estimated the cost of deploying **4x H100s** to be around **$20/hr**, highlighting potential cost considerations for users.


- **Better R1 Experience in open webui** ([Score: 117, Comments: 41](https://reddit.com/r/LocalLLaMA/comments/1i6b65q/better_r1_experience_in_open_webui/)): The post introduces a simple **open webui function** for **R1 models** that enhances the user experience by replacing `<think>` tags with `<details>` and `<summary>` tags, allowing R1's thoughts to be collapsible. Additionally, it removes old thoughts in multi-turn conversations as recommended by **DeepSeek's API documentation**, and is intended for local use of R1 (-distilled) models, not compatible with the DeepSeek API. More details can be found on [GitHub](https://github.com/AaronFeng753/Better-R1).
  - **OpenUI vs. LMstudio**: There is a comparison between **OpenUI** and **LMstudio**, with users expressing a desire for OpenUI to be as responsive as LMstudio. However, the author highlights that **webui** offers more flexibility by allowing users to modify input and output freely.
  - **DeepSeek API Support**: Some users request adding support for **DeepSeek's API** to the open webui function, indicating interest in broader compatibility beyond local use.
  - **VRAM Limitations and Solutions**: Users discuss the challenges of using models with limited VRAM, such as 8GB, and share resources like the **DeepSeek-R1-Distill-Qwen-7B-GGUF** on **Hugging Face** to potentially address these limitations.


**Theme 3. Comparison of DeepSeek R1 Efficiency and Performance to Competitors**

- **I calculated the effective cost of R1 Vs o1 and here's what I found** ([Score: 58, Comments: 17](https://reddit.com/r/LocalLLaMA/comments/1i6axmv/i_calculated_the_effective_cost_of_r1_vs_o1_and/)): The post analyzes the **cost-effectiveness** of R1 versus o1 models by comparing their token generation and pricing. **R1** generates **6.22 times** more reasoning tokens than **o1**, while **o1** is **27.4 times** more expensive per million output tokens. Thus, **R1** is effectively **4.41 times** cheaper than **o1** when considering token efficiency, though actual costs may vary slightly due to assumptions about token-to-character conversion.
  - Several commenters, including **UAAgency** and **inkberk**, criticize the methodology used in the cost comparison, suggesting that the analysis might be biased or based on assumptions that don't accurately reflect real-world usage. **Dyoakom** and **pigeon57434** highlight the potential lack of transparency from OpenAI, questioning the representativeness of the examples provided by the company.
  - **dubesor86** provides detailed testing results, indicating that **R1** does not generate **6.22 times** more reasoning tokens than **o1**. In their testing, **R1** produced about **44%** more thought tokens, and the real cost difference was **21.7 times** cheaper for **R1** compared to **o1**, based on API usage data, which contrasts with the original post's conclusions.
  - **BoJackHorseMan53** advises against relying solely on assumptions and suggests running actual queries with the API to determine the true cost differences, emphasizing the importance of verifying assumptions with practical testing.


- **[DeepSeek-R1 PlanBench benchmark results](https://i.redd.it/qa5yh1w3odee1.jpeg)** ([Score: 56, Comments: 2](https://reddit.com/r/LocalLLaMA/comments/1i6n87h/deepseekr1_planbench_benchmark_results/)): The **PlanBench benchmark results** as of **January 20, 2025**, compare various models like **Claude-3.5 Sonnet, GPT-4, LLaMA-3.1 405B, Gemini 1.5 Pro**, and **Deepseek R1** across "Blocksworld" and "Mystery Blocksworld" domains. Key metrics include "Zero shot" scores, performance percentages, and average API costs per 100 instances, with models like **Claude-3.5 Sonnet** achieving a **54.8% success rate** on 329 out of 600 questions.
  - **PlanBench** is a benchmark designed for evaluating large language models on planning and reasoning tasks, with a detailed paper available on [arXiv](https://arxiv.org/abs/2206.10498).
  - The source of the results can be accessed via [this link](https://x.com/karthikv792/status/1881731017746313367) or an alternative link [here](https://xcancel.com/karthikv792/status/1881731017746313367).


**Theme 4. Criticism of 'Gotcha' Tests in LLMs and Competitive Context**

- **[Literally unusable](https://i.redd.it/iatgsah1ubee1.png)** ([Score: 95, Comments: 102](https://reddit.com/r/LocalLLaMA/comments/1i6fxxy/literally_unusable/)): **Criticism of LLM 'Gotcha' tests** highlights the structured response of a language model in counting occurrences of the letter 'r' in "strawberry." The model's analytical and instructional approach includes writing out the word, identifying, and counting the 'r's, emphasizing the presence of **2 lowercase 'r's**.
  - **Model Variability and Performance**: Commenters discuss how different model architectures and pretraining data result in varying performance, with smaller models often diverging from results of larger models like **R1**. **Custodiam99** mentions that even the **70b model** can be practically unusable, whereas others like **Upstairs_Tie_7855** report outstanding results with the same model.
  - **Quantization and Settings Impact**: Several users highlight the importance of using the correct quantization settings and system prompts to achieve accurate results. **Youcef0w0** notes that models break with lower cache types than **Q8**, while **TacticalRock** emphasizes using the right quantization and temperature settings as per documentation.
  - **Practical Application and Limitations**: Discussions reveal that the models are not AGI but tools requiring proper usage to solve problems effectively. **ServeAlone7622** suggests a detailed process for using reasoning models, while **MixtureOfAmateurs** and **LillyPlayer** illustrate the models' struggles with specific prompts and overfitting on certain tasks.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. OpenAI Investment $500B: Partnership with Oracle and Softbank**

- **[Trump to announce $500 billion investment in OpenAI-led joint venture](https://www.reuters.com/technology/artificial-intelligence/trump-announce-private-sector-ai-infrastructure-investment-cbs-reports-2025-01-21/)** ([Score: 595, Comments: 181](https://reddit.com/r/OpenAI/comments/1i6rwc0/trump_to_announce_500_billion_investment_in/)): **Donald Trump** plans to announce a **$500 billion investment** in a project led by **OpenAI**. Specific details of the joint venture and its objectives have not been provided.
  - **Misunderstanding of Investment Source**: Many commenters clarify that the **$500 billion investment** is from the private sector, not the U.S. government. This investment involves **OpenAI, SoftBank, and Oracle** in a joint venture called **Stargate**, initially committing **$100 billion** with potential growth to $500 billion over four years.
  - **Concerns About Infrastructure and Location**: Commenters express concerns about the adequacy of the U.S. electrical grid to handle AI infrastructure needs, suggesting future reliance on **nuclear reactors**. The choice of **Texas** for the project is questioned due to its isolated and unreliable electrical grid.
  - **Skepticism and Political Concerns**: There is skepticism about whether the investment will materialize and criticism of the political implications, with some viewing it as aligning with **fascist** tendencies. The announcement is compared to previous speculative projects like "Infrastructure week" and the **Wisconsin plant**.


- **[Sam Altman’s expression during the entire AI Infra Deal Announcement](https://www.reddit.com/gallery/1i6w8ln)** ([Score: 163, Comments: 51](https://reddit.com/r/OpenAI/comments/1i6w8ln/sam_altmans_expression_during_the_entire_ai_infra/)): The post lacks specific details or context about **Sam Altman**'s expression during the AI Infra Deal announcement, providing no further information or insights.
  - Discussions around **Sam Altman's demeanor** during the announcement highlight perceptions of anxiety and stress, with comments suggesting he often looks this way. Users liken his expression to a "Fauci face" or "Debra Birx" and speculate about the pressures he faces in his role.
  - Several comments humorously reference **Elon Musk** and geopolitical figures like **Putin**, suggesting that Altman might be under significant pressure due to internal and external political dynamics. Comparisons are drawn to oligarch management and defenestration politics.
  - The conversation includes light-hearted and sarcastic remarks about Altman's expression, with users jokingly attributing it to being a "twink waiting to see his sugardaddy" or worrying about Musk's reactions, indicating a mix of humor and critique in the community's perception of Altman.


**Theme 2. OpenAI's New Model Operators**

- **[CEO of Exa with inside information about Open Ai newer models](https://www.reddit.com/gallery/1i6dpet)** ([Score: 215, Comments: 105](https://reddit.com/r/OpenAI/comments/1i6dpet/ceo_of_exa_with_inside_information_about_open_ai/)): **CEO of Exa** claims to have inside information on the capabilities of **OpenAI's newer models**, specifically questioning the potential effectiveness of these models as operators. The post does not provide further details or context.
  - The discussion highlights skepticism about the **hype surrounding AGI** and **OpenAI's newer models**, with several users questioning the realism of claims and drawing parallels to previous overhyped technologies like **3D printers**. Users express doubt about the real-world performance of models like **o3** compared to their benchmark results, emphasizing the gap between hype and practical application.
  - Several comments explore the **limitations of current AI models**, focusing on their inability to handle tasks that require real-time learning and complex reasoning, such as **video comprehension** and understanding **3D spaces**. **Altruistic-Skill8667** predicts that achieving AGI will require significant advancements in **compute power** and **online learning**, with a potential timeline extending to **2028 or 2029**.
  - Some users express concern over the **socio-political implications** of AI advancements, suggesting that **AGI** could be used to **subjugate the working class** under an oligarchic regime. A few comments also touch on the role of **government and tech oligarchies** in shaping AI's future, with comparisons between the **US and China** in terms of tech control and regulation.


**Theme 3. Anthropic's ASI Prediction: Implications of 2-3 Year Timeline**

- **[Anthropic CEO is now confident ASI (not AGI) will arrive in the next 2-3 years](https://i.redd.it/3dtbepq6pcee1.png)** ([Score: 173, Comments: 115](https://reddit.com/r/OpenAI/comments/1i6iu7m/anthropic_ceo_is_now_confident_asi_not_agi_will/)): **Anthropic's CEO**, Amodei, predicts **Artificial Superintelligence (ASI)** could be achieved in the next **2-3 years**, surpassing human intelligence. The company plans to release advanced AI models with enhanced memory and two-way voice integration for **Claude**, amidst competition with companies like **OpenAI**.
  - Discussions highlight skepticism about **ASI predictions** within 2-3 years, with some researchers and commenters arguing that significant improvements in AI models are needed and that current AI systems are still far from achieving **AGI**. **Dario Amodei**'s credibility is noted, given his background in AI research, but there is debate over whether his predictions are realistic.
  - The distinction between **narrow AI** and **general AI** is emphasized, with current AI systems excelling in specific tasks but lacking the comprehensive capabilities of AGI. Commenters note that despite advancements, AI systems still struggle with many tasks that are simple for humans, and the path to AGI and ASI remains undefined.
  - **Funding and business motivations** are questioned, with some suggesting that announcements of imminent ASI could be strategically timed to coincide with fundraising efforts. The comment about **Anthropic's** current fundraising activities supports this perspective.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. DeepSeek R1 Rocks the AI World**

- **DeepSeek R1 Dethrones Competitors**: The open-source [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1) matches OpenAI's o1 performance, thrilling the community with its cost-effectiveness and accessibility. Users report strong performance in coding and reasoning tasks, with [benchmarks](https://x.com/TheXeophon/status/1881443117787984265) showing it outperforming other models.
- **Integration Frenzy Across Platforms**: Developers scramble to integrate DeepSeek R1 into tools like [Cursor](https://www.cursor.com/), [Codeium](https://codeium.com/), and [Aider](https://aider.chat/), despite occasional hiccups. Discussions highlight both successes and challenges, especially regarding tool compatibility and performance.
- **Censorship and Uncensored Versions Spark Debate**: While some praise DeepSeek R1's safety features, others bemoan over-censorship hindering practical use. An [uncensored version](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF) circulates, prompting debates about the balance between safety and usability.

**Theme 2. OpenAI's Stargate Project Shoots for the Moon**

- **OpenAI Announces $500 Billion Stargate Investment**: OpenAI, along with SoftBank and Oracle, pledges to invest [$500 billion in AI infrastructure](https://x.com/OpenAI/status/1881830103858172059), dubbing it the Stargate Project. The initiative aims to bolster U.S. AI leadership, with comparisons drawn to the Apollo Program.
- **Community Buzzes over AI Arms Race**: The staggering investment stirs discussions about an AI arms race and geopolitical implications. Some express concerns that framing AI development as a competition could lead to unintended consequences.
- **Mistral AI Makes a Mega IPO Move**: Contradicting buyout rumors, [Mistral AI](https://x.com/btibor91/status/1881692647456477189) announces IPO plans and expansion into Asia-Pacific, fueling speculation about its profitability and strategy.

**Theme 3. New Models and Techniques Push Boundaries**

- **Liquid AI's LFM-7B Makes a Splash**: [Liquid AI's LFM-7B](https://www.liquid.ai/lfm-7b) claims top performance among 7B models, supporting multiple languages including English, Arabic, and Japanese. Its focus on local deployment excites developers seeking efficient, private AI solutions.
- **Mind Evolution Evolves AI Thinking**: A new paper introduces [Mind Evolution](https://arxiv.org/abs/2501.09891), an evolutionary search strategy that achieves over **98% success** on planning tasks. This approach beats traditional methods like Best-of-N, signaling a leap in scaling LLM inference.
- **SleepNet and DreamNet Dream of Better AI**: Innovative models [SleepNet and DreamNet](https://arxiv.org/abs/2410.18156) propose integrating 'sleep' phases into training, mimicking human learning processes. These methods aim to balance exploration and precision, inspiring discussions on novel AI training techniques.

**Theme 4. Users Battle Bugs and Limitations in AI Tools**

- **Windsurf Users Weather Lag Storms**: Frustrated Windsurf users report laggy prompts and errors like *"incomplete envelope: unexpected EOF"*, pushing some towards alternatives like Cursor. The community seeks solutions while expressing discontent over productivity hits.
- **Flow Actions Limit Trips Up Coders**: Codeium's Flow Actions limit hampers workflows, with users grumbling about repeated bottlenecks. Suggestions for strategic usage emerge, but many await official resolutions.
- **Bolt Users Lose Tokens to Bugs**: Developers lament losing tokens due to buggy code on [Bolt](https://www.stackblitz.com/), advocating for free debugging to mitigate losses. One exclaims, "*I've lost count of wasted tokens!*", highlighting cost concerns.

**Theme 5. AI's Expanding Role in Creative and Technical Fields**

- **DeepSeek R1 Masters Math Tutoring**: Users harness DeepSeek R1 for [mathematics tutoring](https://x.com/seo_leaders/status/1881462202831614085), praising its step-by-step solutions and support for special educational needs. Its speed and local deployment make it a favorite among educators.
- **Generative AI Shapes Creative Industries**: [Articles](https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38) spark debates on AI's impact on art and music, with some fearing AI might replace human creators. Others argue that human skills remain crucial to guide AI outputs effectively.
- **Suno Hit with Copyright Lawsuit Over AI Music**: AI music generator [Suno](https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/) faces fresh legal challenges from Germany's GEMA, accused of training on unlicensed recordings. The lawsuit fuels industry debates on the legality of AI-generated content.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-R1's Deceptive Depth**: The **DeepSeek-R1** model's maximum token length was found to be **16384** instead of the expected **163840**, prompting **bug** concerns in code deployment.
   - A [tweet](https://x.com/fimbulvntr/status/1881821582571761920) about **RoPE factors** and model embeddings triggered further discussion, with members suggesting incomplete usage of the model.
- **LoRA Llama 3 Tuning Tactics**: A **Medium article** by Gautam Chutani demonstrated [LoRA-based fine-tuning of Llama 3](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060), integrating **Weights & Biases** and **vLLM** for serving.
   - He stressed cutting down on **GPU** overhead via LoRA injections, with community remarks pointing to a more resource-friendly alternative than high-end baseline finetunes.
- **Chinchilla's Crisp Calculation**: The [Chinchilla paper](https://paperswithcode.com/method/chinchilla) recommends proportional growth of **model size** and **training tokens** for peak efficiency, reshaping data planning strategies.
   - Participants argued the **Chinchilla optimal** approach sidesteps focusing on narrow parameter segments, stressing total parameter involvement as a safer strategy.
- **Synthetic & Mixed Data Gains**: Some promoted **synthetic data** for tighter evaluation alignment, while others applied mixed-format datasets in **Unsloth** to broaden coverage in training.
   - Attendees noted dynamic adjustments can mitigate overfitting, yet domain-specific relevance remains questionable when venturing beyond real-world material.
- **Open-Source UI Overdrive**: **OpenWebUI**, **Ollama**, and **Flowise** surfaced as next targets for integration, while **Kobold** and **Aphrodite** remain active through the Kobold API.
   - *Invisietch* confirmed a long to-do list, including a CLI for **synthetic dataset** creation, aiming for a unified backend API to streamline everything.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **OpenAI's Stargate Strikes Grand**: OpenAI announced a $500 billion investment plan called the **Stargate Project** to build new AI infrastructure in the US, with cooperation from SoftBank and others, as seen [here](https://x.com/OpenAI/status/1881830103858172059).
   - Community members are abuzz with strategic implications, wondering if Japan's big investments might embolden a new wave of **AI competition**.
- **DeepSeek R1 Gains & Cursor Pains**: **DeepSeek R1** can be integrated into [Cursor](https://www.cursor.com/downloads) via [OpenRouter](https://openrouter.ai/deepseek/deepseek-r1), though some users find the workaround limiting and prefer to wait for native support.
   - Benchmark chatter references [Paul Gauthier's tweet](https://x.com/paulgauthier/status/1881428345973608901) citing a **57%** score on the aider polyglot test, fueling debate on the upcoming competition between **DeepSeek R1** and other LLMs.
- **Cursor 0.45 Rollback Reactions**: The **Cursor** team keeps rolling back v0.45.1 updates due to indexing issues, forcing developers to revert to earlier versions, as per [Cursor Status](https://status.cursor.com).
   - Some users are frustrated by instability and mention that minimal official statements complicate their workflow, suggesting they might explore alternative code editors like [Codefetch](https://x.com/kregenrek/status/1878487131099898269).
- **Claude 3.5 Competes with DeepSeek**: **Claude 3.5** performance has improved, sparking direct comparisons to **DeepSeek R1** and prompting discussions on speed and accuracy gains.
   - Anthropic's silence on future updates raises speculation about their next release, as overshadowed by the competition's momentum.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Woes & Surging Delays**: Multiple users lament ongoing lag issues in **Windsurf**, especially during code prompts, with some encountering **'incomplete envelope: unexpected EOF'** errors.
   - Some consider switching to **Cursor** due to these bugs, though potential solutions like adjusting local settings have yet to yield firm fixes.
- **DeepSeek R1 Dominates Benchmarks**: Community members are excited about **DeepSeek R1** surpassing **OpenAI O1-preview** in various performance tests, according to [Xeophon's tweet](https://x.com/TheXeophon/status/1881442133376454694).
   - A follow-up [tweet](https://x.com/TheXeophon/status/1881443117787984265) highlights **R1** as a league of its own, though doubts remain regarding its tool-call compatibility within **Codeium**.
- **Flow Actions Fizzle Productivity**: Many find the **Flow Actions** limit disruptive to their workflow, citing repeated bottlenecks throughout the day.
   - Community members propose strategic usage and partial resets to ease the constraint, though official fixes remain uncertain.
- **Codeium Feature Frenzy**: A user requested adding **DeepSeek R1** support in **Codeium**, along with calls for better fine-tuning and robust updates for JetBrains IDE users.
   - Others mention the need for improved rename suggestions via [Codeium's feature request page](https://codeium.canny.io/feature-requests/p/add-rename-suggestion-like-in-vscode) and highlight [Termium](https://codeium.com/blog/termium-codeium-in-terminal-launch) for command-line auto-completion.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.72.0 Release Bolsters Development**: The **Aider v0.72.0** update includes **DeepSeek R1** support via `--model r1` or OpenRouter, plus **Kotlin syntax** and the new `--line-endings` option, addressing Docker image permissions and **ASCII fallback** fixes.
   - Community members noted that **Aider** contributed **52%** of its own code for this release and discovered that `examples_as_sys_msg=True` with GPT-4o yields higher test scores.
- **DeepSeek R1 Emerges as Powerful Challenger**: Users praised **DeepSeek R1** for multi-language handling, citing [this tweet](https://x.com/0xluffyb/status/1881323971897110866) about near parity with **OpenAI's 01** and **MIT licensed** distribution.
   - Conversations hinted at switching from **Claude** to **DeepSeek R1** for cost reasons, referencing [DeepSeek-R1 on GitHub](https://github.com/deepseek-ai/DeepSeek-R1) for further technical details.
- **OpenAI Subscriptions & GPU Costs Spark Debate**: Some members reported **OpenAI** subscription refunds and weighed the cost-effectiveness of **DeepSeek**, mentioning [the CEO da OpenAI article](https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt) regarding pricing uncertainties.
   - European users also found cheaper **RTX 3060** and **3090** GPUs, and they consulted [Fireworks AI docs](https://docs.fireworks.ai/guides/security_compliance/data_handling) for privacy considerations in AI-driven workflows.
- **Space Invaders Upgraded with DeepSeek R1**: A [live coding video](https://youtu.be/njJhjUgBTZg) showed **DeepSeek R1** powering a refined **Space Invaders** game, demonstrating second-place benchmarking on the **Aider LLM leaderboard**.
   - The user highlighted near equivalence to **OpenAI's 01** at a lower price, driving interest in game and dev scenarios that benefit from **R1**’s coding focus.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek's Bold Foray into Math Mastery**: **DeepSeek R1** emerged as a strong pick for **mathematics tutoring**, providing **step-by-step solutions** and supporting special educational needs, exemplified by [Andrew C's tweet](https://x.com/seo_leaders/status/1881462202831614085) about running a 671B version on M2 Ultras.
   - One user praised the model's **speed** and local deployment capabilities, referencing the [DeepSeek-R1 GitHub repo](https://github.com/deepseek-ai/DeepSeek-R1) for advanced usage scenarios.
- **Local Model Magic & OpenAI Touchpoints**: Enthusiasts discussed running **LLMs** on robust home setups like a **4090 GPU** with **64GB RAM**, referencing [LM Studio Docs](https://lmstudio.ai/docs/basics/rag) and [Ollama's OpenAI compatibility blog](https://ollama.com/blog/openai-compatibility) for bridging local models with OpenAI API.
   - Others highlighted the significance of **quantization** (Q3, Q4, etc.) for performance trade-offs and explored solutions like [Chatbox AI](https://chatboxai.app/zh) to unify **local** and **online** usage.
- **NVIDIA's DIGITS Drama & DGX OS Dilemmas**: Users lamented the **high cost** (around **$3000 for 128GB**) and uncertain **NVIDIA DIGITS** support, pointing to [NVIDIA DIGITS docs](https://docs.nvidia.com/deeplearning/digits/index.html) for legacy insights.
   - Discussions noted the **DGX OS** similarities to old DIGITS, with someone suggesting **NVIDIA TAO** as a modern alternative, though it introduced confusion about container-focused releases.
- **GPU Heat Headaches & Future Plans**: Some mentioned **excessive heat** from powerful GPUs, joking no cleaning is needed due to constant burning and referencing second-hand sales for potential **cost** savings.
   - Others plan a **GUI-free** approach for optimized performance, with an emphasis on **lighter setups** to mitigate **thermal** strains in advanced ML tasks.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Liquid AI's LFM-7B Rises for Local Deployments**: Liquid AI introduced **LFM-7B**, a non-transformer model that claims top-tier performance in the 7B range with expanded language coverage including **English**, **Arabic**, and **Japanese** ([link](https://www.liquid.ai/lfm-7b)).
   - Community members praised its local deployment strategy, with some crediting the model's **automated architecture search** as a potential differentiator.
- **Mind Evolution Maneuvers LLM Inference**: A new paper on **Mind Evolution** showcased an evolutionary approach that surpasses Best-of-N for tasks like **TravelPlanner** and **Natural Plan**, achieving over **98%** success with **Gemini 1.5 Pro** ([arXiv link](https://arxiv.org/abs/2501.09891)).
   - Engineers discussed the method's iterative generation and recombination of prompts, describing it as a streamlined path to scale inference compute.
- **DeepSeek-R1 Distill Model Gains Mixed Reviews**: Users trialed **DeepSeek-R1 Distill** for quantization tweaks and performance angles, referencing [a Hugging Face repo](https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF) close to **8B** parameters.
   - Some praised its reasoning outputs while others found it overly verbose on casual prompts, yet it remained a highlight for advanced thinking time.
- **SleepNet & DreamNet Bring 'Nighttime' Training**: **SleepNet** and **DreamNet** propose supervised plus unsupervised cycles that mimic 'sleep' to refine model states, as detailed in [Dreaming is All You Need](https://arxiv.org/abs/2409.01633v2) and [Dreaming Learning](https://arxiv.org/abs/2410.18156).
   - They use an encoder-decoder approach to revisit hidden layers during off phases, spurring discussions about integrative exploration.
- **Mistral Musings on Ministral 3B & Codestral 2501**: Mistral teased **Ministral 3B** and **Codestral 2501**, fueling speculation on a weights-licensing plan in a tight AI landscape.
   - Observers wondered if Mistral's approach, akin to Liquid AI's architecture experiments, might carve out a specialized niche for smaller-scale deployments.



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt’s Bolder Code Inclusions**: Bolt's latest update removes the **white screen** fiasco and includes [a fix for complete code delivery](https://x.com/boltdotnew/status/1881731948051415059), guaranteeing a **spot-on** setup from the first prompt as seen in [this announcement](https://x.com/boltdotnew/status/1881442318110347291).
   - Engineers welcomed this **comprehensive** shift, saying *"No more lazy code!"* and praising the smoother start-up for new projects.
- **Prismic Predicaments & Static Solutions**: A user faced issues integrating **Prismic CMS** for a plumbing site, prompting a suggestion to build a static site first for future-proof flexibility.
   - Community members favored a minimal approach, with one noting the complexity of *"CMS overhead for simple sites."*
- **Firebase vs Supabase Face-Off**: A user argued for swapping **Supabase** in favor of **Firebase**, calling it a simpler path for developers.
   - Others agreed **Firebase** eases initial setups, emphasizing how it accelerates quick proofs-of-concept.
- **Token Tussles**: Developers reported losing **tokens** due to buggy code on Bolt, advocating free debugging to curb these losses.
   - Cost worries soared, with one user declaring *"I've lost count of wasted tokens!"*
- **Next.js & Bolt: Tectonic Ties**: A community member tried incorporating WordPress blogs into **Next.js** using Bolt but saw frameworks update faster than AI tooling.
   - Opinions were split, with some saying Bolt may not track **rapid** Next.js changes closely enough.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Sonar Surges with Speed and Security**: Perplexity released the [Sonar and Sonar Pro API](https://sonar.perplexity.ai/) for **generative search**, featuring **real-time** web analysis and demonstrating major adoption by **Zoom**, while outperforming established engines on **SimpleQA** benchmarks.
   - Community members applauded its **affordable** tiered pricing and *noted* that no user data is used for LLM training, suggesting safer enterprise usage.
- **DeepSeek vs O1 Rumblings**: Multiple members questioned if **DeepSeek-R1** would replace the absent **O1** in Perplexity, referencing [public hints](https://x.com/AravSrinivas/status/1881458694266953934) about advanced reasoning capabilities.
   - Others praised **DeepSeek-R1** for its free, top performance, *calling it* “the best alternative” while some remained uncertain about **O1**’s planned future.
- **Claude Opus: Retired or Resilient?**: Some users declared **Claude Opus** retired in favor of `Sonnet 3.5`, questioning its viability in creative tasks.
   - Others emphasized that Opus continues to excel in complex projects, *insisting* it remains the most advanced in its family despite rumored replacements.
- **Sonar Pro Tiers & Domain Filter Beta**: Contributors highlighted [new usage tiers](https://docs.perplexity.ai/guides/usage-tiers) for **Sonar** and **Sonar Pro**, noting the **search_domain_filter** as a beta feature in tier 3.
   - Many users sought direct **token usage** insights from the API output, while some pushed for **GDPR**-compliant hosting in European data centers.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek R1 Rocks Benchmarks**: On **January 20**, China's **DeepSeek AI** unveiled **R1**, hitting up to **20.5%** on the [ARC-AGI public eval](https://x.com/arcprize/status/1881761987090325517).
   - It outperformed **o1** in web-enabled tasks, with the full release details [here](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1).
- **Mistral’s Mega IPO Move**: Contrary to buyout rumors, **Mistral AI** announced an IPO plan alongside opening a **Singapore** office for the Asia-Pacific market.
   - Members speculated on Mistral’s profitability, referencing [this update](https://x.com/btibor91/status/1881692647456477189) as proof of their bold strategy.
- **Stargate Surges with $500B Pledge**: **OpenAI**, **SoftBank**, and **Oracle** united under **Stargate**, promising **$500 billion** to bolster U.S. AI infrastructure over four years.
   - They’ve likened this grand investment to historical feats like the **Apollo** program, aiming to cement American AI leadership.
- **Anthropic Angles for Claude’s Next Step**: At **Davos**, CEO **Dario Amodei** teased **voice mode** and possible web browsing for **Claude**, as seen in [this WSJ interview](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2).
   - He hinted at beefier Claude releases, with the community debating how often updates will drop.
- **Tulu 3 RLVR Sparks Curiosity**: A [poster project](https://x.com/hamishivi/status/1881398642403356678) on **Tulu 3’s RLVR** grabbed attention, promising new ways to approach reinforcement learning.
   - Enthusiasts plan to merge it with **open-instruct** frameworks, hinting at broader transformations in model usage.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Tavily Search MCP Server Soars**: The new [Tavily Search MCP server](https://glama.ai/mcp/servers/0kmdibf9t1) landed with **optimized web search** and **content extraction** for LLMs, featuring SSE, stdio, and Docker-based installs.
   - It uses Node scripts for swift deployment, fueling the **MCP ecosystem** with broader server choices.
- **MCP Language Server Showdown**: Developers tested [isaacphi/mcp-language-server](https://github.com/isaacphi/mcp-language-server) and [alexwohletz/language-server-mcp](https://github.com/alexwohletz/language-server-mcp), aiming for **get_definition** and **get_references** in large codebases.
   - They noted the second repo might be less mature, yet the community remains eager for **IDE-like** MCP features.
- **Roo-Clines Grow Wordy**: Members championed adding **roo-code** tools to roo-cline for extended language tasks, including code manipulation in sprawling projects.
   - They envision deeper **MCP synergy** to streamline code management, suggesting advanced edits in a single CLI ecosystem.
- **LibreChat Sparks Complaints**: A user slammed **LibreChat** for tricky configurations and unpredictable API support, even though they admired its polished UI.
   - They also bemoaned the absence of usage limits, comparing it to stricter platforms like **Sage** or built-in **MCP** servers.
- **Anthropic Models Eye Showdown with Sage**: A lively exchange broke out on the feasibility of **Anthropic** model r1, with some guessing 'Prob' they'd get it running.
   - Others leaned on **Sage** for macOS and iPhone, preferring fewer headaches over uncertain Anthropic integrations.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama's Last Stand at Samba Nova**: The **free Llama endpoints** end this month because of changes from **Samba Nova**, removing direct user access.
   - Samba Nova will switch to a **Standard** variant with new pricing, provoking talk about paid usage.
- **DeepSeek R1 Gains Web Search & Freed Expression**: The **DeepSeek R1** model enables web search grounding on [OpenRouter](https://x.com/OpenRouterAI/status/1881785438043799765), maintaining a **censorship-free** approach at **$0.55** per input token.
   - Community comparisons reveal performance close to **OpenAI's o1**, with discussions on **fine-tuning** cited in [Alex Atallah's post](https://x.com/xanderatallah/status/1881456463786512737).
- **Gemini 2.0 Flash: 64K Token Marvel**: A fresh **Gemini 2.0 Flash Thinking Experimental 01-21** release offers a **1 million** context window plus **64K** output tokens.
   - Observers noted some naming quirks during its ten-minute rollout; it remains available through **AI Studio** without tiered keys.
- **Sneaky Reasoning Content Trick Emerges**: A user exposed a method to coax **reasoning content** from [DeepSeek Reasoner](https://api-docs.deepseek.com/guides/reasoning_model) by crafty prompt prefixes.
   - Concerns arose over token clutter from leftover CoT data, prompting better message handling strategies.
- **Perplexity's Sonar Models Catch Eyes**: **Perplexity** debuted new **Sonar** LLMs with web-search expansions, highlighted in [this tweet](https://x.com/risphereeditor/status/1881789442530435513).
   - While some are excited about a potential integration, others doubt the models’ utility, urging votes for **OpenRouter** support.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cranked-Up GPT-2 Gains**: Engineers discussed adjusting `max_steps` for GPT-2 re-training, recommended doubling it for two epochs to prevent **rapid learning rate decay**, referencing **Andrew Karpathy**'s approach.
   - They also warned that rash changes might waste resources, suggesting thorough knowledge before making fine-tuning decisions.
- **RAG Revelations in Live Q&A**: A **Live Q&A** on **RAG** and **tool use** with models is scheduled for **Tuesday at 6:00 am ET** on Discord Stage, encouraging builders to share experiences.
   - Participants plan to tackle challenges in integrating new implementations, aiming for a collaborative environment that sparks shared insights.
- **Cohere CLI: Terminal Talk for Transformers**: The new **Cohere CLI** lets users chat with Cohere's AI from the command line, showcased on [GitHub](https://github.com/plyght/cohere-cli).
   - Community members praised its convenience, with some highlighting how **terminal-based** interactions could speed up iterative development.
- **Cohere For AI: Community Powerhouse**: Enthusiasts urged each other to join the **Cohere For AI** initiative for open machine learning collaboration, referencing [Cohere’s official research page](https://cohere.com/research).
   - They also noted **trial keys** offering 1000 free monthly requests, reinforcing a welcoming space for newcomers eager to test AI solutions.
- **Math Shortfalls in LLM Outputs**: Members flagged **Cohere** for incorrectly calculating 18 months as 27 weeks, casting doubt on LLMs' math reliability.
   - They connected this to **tokenization** issues, calling it a widespread shortcoming that can topple projects if left unaddressed.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Classroom Conquest: NotebookLM for College Courses**: Members suggested organizing NotebookLM by topics rather than individual sources to ensure **data consistency**, noting that a 1:1 notebook:source setup is best for **podcast generation** with a single file.
   - They emphasized it *eliminates clutter* and fosters smoother collaboration, potentially transforming study habits and resource sharing in academic environments.
- **Video Victories: AI eXplained's Cinematic Unfolding**: The **AI eXplained** channel released a new episode on **AI-generated videos**, spotlighting advances in scriptwriting and animated production.
   - Early watchers mentioned *the wave of interest* in these approaches to reshape the film industry, predicting more breakthroughs in audio-visual AI.
- **Gemini Gains: Code Assist for Devs**: Community members recommended **Gemini Code Assist** for deeper repository insights, describing it as more accurate than NotebookLM for focused code queries.
   - They noted NotebookLM can misfire unless guided with **very specific instructions**, spurring discussions on code analysis methods and reliability.
- **Sacred Summaries: NotebookLM in Church Services**: One participant leveraged NotebookLM to parse extensive sermon transcripts, eyeing a **250-page** collection and even a **2000-page** Bible study.
   - They hailed it as *a game changer* for distilling large religious texts, praising its utility in bridging tech and faith.
- **Tooling Treasures: Add-ons & Apps Amp NotebookLM**: Users swapped suggestions for add-ons including [OpenInterX Mavi](https://mavi.openinterx.com) and [Chrome Web Store](https://chromewebstore.google.com/search/notebookLM) extensions to boost functionality.
   - They tested methods to *retain favorite prompts* for quicker work and expressed hope for deeper **NotebookLM** integrations down the road.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Cohesive Comix with ControlNet**: Members explored AI-driven comic panels with **ControlNet** for consistent scene details, generating each frame separately to keep characters stable. They discovered the approach still produces varied results, requiring frequent re-generation to maintain continuity.
   - They also debated whether advanced prompts or additional training data could improve results, with some seeing potential for future improvements once **Stable Diffusion** matures.
- **AI Art Controversy Continues**: Contributors noted stronger pushback against **AI-rendered artwork** in creative communities, highlighting doubts about credibility and respect for original styles. They cited the broader debate on whether AI art displaces manual craft or simply extends it.
   - Others raised ethical concerns about using training data from public repositories, referencing broader calls for guidelines that ensure credit to original creators.
- **Stable Diffusion AMD Setup Snags**: Individuals shared difficulties running **Stable Diffusion** on AMD hardware, pointing to driver issues and slower performance. They referenced pinned instructions in the Discord as a workaround but acknowledged the need for more robust official support.
   - Some found success with updated libraries, but others still faced unexpected black screens or incomplete renders, requiring manual GPU resets.
- **Manual vs. AI Background Tweaks**: Enthusiasts debated using GIMP for straightforward background edits versus leaning on **Stable Diffusion** for automatic enhancements. They reported that manual editing offered more controlled results, especially for sensitive details in personal photoshoots.
   - Some argued that AI solutions still lack refinement for nuanced tasks, while others saw promise if the models gain more specialized training.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Evolving Minds & Taming GRPO**: The **Mind Evolution** strategy for scaling LLM inference soared to over **98%** success on TravelPlanner and Natural Plan, as detailed in [the arXiv submission](https://arxiv.org/abs/2501.09891).
   - A simple local **GRPO** test is in progress, with future plans to scale via **OpenRLHF** and Ray and apply **RL** to maths datasets.
- **TMA Takes Center Stage in Triton**: Community members investigated **TMA** descriptors in Triton, leveraging `fill_2d_tma_descriptor` and facing autotuning pitfalls that caused crashes.
   - A working example of **persistent GEMM** with TMA was shared, but manual configuration remains necessary due to limited autotuner support.
- **Fluid Numerics Floats AMD MI300A Trials**: The **Fluid Numerics** platform introduced subscriptions to its **Galapagos** cluster, featuring the **AMD Instinct MI300A** node for AI/ML/HPC workloads and a [request access link](https://www.fluidnumerics.com/shop/p/rcc-allocation-monthly-subscription).
   - They encouraged users to test software and compare performance between **MI300A** and **MI300X**, inviting broad benchmarking.
- **PMPP Book Gains More GPU Goodness**: A reread of the latest **PMPP Book** was advised, as it updates content missing from the 2022 release and adds new **CUDA** material.
   - Members recommended **cloud GPU** options like [Cloud GPUs](https://cloud-gpus.com/) or **Lightning AI** for hands-on practice with the book’s exercises.
- **Lindholm's Unified Architecture Legacy**: Engineer **Lindholm** recently retired from **Nvidia**, with an insightful November 2024 talk on his **unified architecture** available via [Panopto](https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3).
   - Participants learned about his impactful design principles and contributions until his retirement two weeks ago.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GGUF Gains Ground Over Rival Formats**: The community noted **GGUF** is the favored quantization route for consumer hardware, referencing [Benchmarking LLM Inference Backends](https://www.bentoml.com/blog/benchmarking-llm-inference-backends) to show its strong performance edge.
   - They contrasted tools like **vLLM** and **TensorRT-LLM**, emphasizing that startups often choose straightforward backends such as **Ollama** for local-ready simplicity.
- **R1 Riddles & Qwen Quirks**: Members put **R1** under the microscope, debating its use of PRMs and pondering how 4bit/3bit vs **f16** influences MMLU-PRO performance.
   - They also considered converting **Qwen R1** models to **Q-RWKV**, eyeing tests like **math500** to confirm success and questioning how best to estimate **pass@1** with multiple response generations.
- **Titans Tackle Memory for Deep Nets**: The **Titans** paper ([arXiv:2501.00663](https://arxiv.org/abs/2501.00663)) proposes mixing short-term with long-term memory to bolster sequence tasks, building on recurrent models and attention.
   - A user asked if it’s *"faster to tune the model on such a large dataset?"* while others weighed whether scaling data sizes outperforms incremental methods.
- **Steering Solutions Still Skimpy**: No single open source library dominates **SAE-based** steering for LLMs, though projects like [steering-vectors](https://github.com/steering-vectors/steering-vectors) and [repeng](https://github.com/vgel/repeng) show promise.
   - They also mentioned [representation-engineering](https://github.com/andyzoujm/representation-engineering), noting its top-down approach but highlighting the general lack of a unified approach.
- **NeoX: Nudging HF Format with Dimension Disputes**: A **RuntimeError** in `convert_neox_to_hf.py` revealed dimension mismatch issues ([8, 512, 4096] vs 4194304), possibly tied to multi-node setups and **model_parallel_size=4**.
   - Questions arose about the **3x** intermediate dimension setting, while the shared config mentioned **num_layers=32**, **hidden_size=4096**, and **seq_length=8192** impacting the export process.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Fuels Stargate Project with $500B**: OpenAI unveiled **The Stargate Project** with a pledged **$500 billion** investment over the next four years, aimed at building AI infrastructure in the U.S. with **$100 billion** starting now.
   - Major backers including **SoftBank** and **Oracle** are betting big on this initiative, emphasizing job creation and AI leadership in America.
- **Gemini 2.0 Gains Experimental Update**: Feedback on **Gemini 2.0 Flash Thinking** led Noam Shazeer to introduce new changes that reflect user-driven improvements.
   - These tweaks aim to refine **Gemini’s** skill set and reinforce its responsiveness to real-world usage.
- **DeepSeek Drops V2 Model with Low Inference Costs**: The newly released **DeepSeek V2** stands out for reduced operational expenses and a strong performance boost.
   - Its architecture prompted buzz across the community, showcasing a fresh approach that challenges established models.
- **Ai2 ScholarQA Boosts Literature Review**: The **Ai2 ScholarQA** platform offers a method to ask questions that aggregate information from multiple scientific papers, providing comparative insights.
   - This tool aspires to streamline rigorous research by delivering deeper citations and references on demand.
- **SWE-Bench Soars as WandB Hits SOTA**: **WandB** announced that their **SWE-Bench** submission is now recognized as State of the Art, drawing attention to the benchmark’s significance.
   - The announcement underlines the competitive drive in performance metrics and fosters further exploration of advanced testing.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek R1 & Sonnet Showdown**: Members discussed **DeepSeek R1** distilled into [Qwen 32B Coder](https://link.to.related.info) running locally on a system with **32 GB RAM** and **16 GB VRAM**, offloading heavy computation to CPU for feasible performance.
   - They reported a **60% failure rate** for R1 in coding, which still outperformed **4O and Sonnet** at **99% failure**, though stability on Ollama remains uncertain.
- **Generative AI Heats Up Creative Industries**: A [Medium article](https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38) highlighted **Generative AI**'s ability to produce art, prompting fears it might replace human creators.
   - Others argued that **human skills** are still crucial for shaping AI output effectively, keeping artists involved in the process.
- **Content Compliance Chatter**: Point was raised that **DeepSeek** avoids critical or humorous outputs about the **CCP**, recalling older GPT compliance issues.
   - Users questioned whether these constraints limit **expression** or hamper open-ended debate.
- **Archotech Speculation Runs Wild**: One user mused about AI evolving into **Rimworld**-style archotechs, hinting at unintended capabilities and outgrowths.
   - They suggested that *“we might accidentally spawn advanced entities”* as AI companies keep training bigger models.
- **GPT Downtime and Lagging Responses**: Frequent *'Something went wrong'* errors disrupted chats with **GPT**, though reopening the session generally solved it.
   - Several members noted **sluggish performance**, describing slow replies as a source of collective exasperation.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Neural ODEs Spark RL Tactics**: In #general, members said **Neural ODEs** could refine robotics by modeling function complexity with layers, referencing the [Neural Ordinary Differential Equations paper](https://arxiv.org/abs/1806.07366).
   - They also debated how smaller models might discover high-quality reasoning through repeated random inits in RL, pointing out that noise and irregularity boost exploration.
- **GRPO Gains Allies**: In #paper-discussion, **DeepSeeks GRPO** was called **PPO** minus a value function, relying on Monte Carlo advantage estimates for simpler policy tuning, as seen in the [official tweet](https://fixupx.com/natolambert/status/1881380809153847711).
   - A [recent publication](https://arxiv.org/abs/2402.03300v3) emphasizes reduced overhead, while the group also tackled reviewer shortages by recruiting **12** out of **50+** volunteers.
- **Suno Swats Copyright Claims**: In #ml-news, AI music generator **Suno** is facing another copyright lawsuit from **GEMA**, adding to previous lawsuits from major record labels, as detailed by [Music Business Worldwide](https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/).
   - Valued at **$500 million**, Suno and rival **Udio** are accused of training on unlicensed recordings, fueling industry debate on AI-based content's legality.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Clashing Over C vs Python**: Members debated **C** for discipline and **Python** for quicker memory management insight, referencing future usage in **JS** or **Python**.
   - One participant highlighted learning **C** first can deepen understanding for a career shift, but opinions varied widely.
- **Forum vs Discord Dilemma**: Many urged clarity on posting **projects** in **Discord** versus the **forum**, citing difficulty retrieving important discussions in a rapid-chat setting.
   - They suggested using the **forum** for in-depth updates while keeping **Discord** for quick bursts of feedback.
- **Mojo’s .gitignore Magic**: Contributors noted the `.gitignore` for **Mojo** only excludes `.pixi` and `.magic` files, which felt suitably minimal.
   - No concerns arose, with the group appreciating a lean default configuration.
- **Mojo and Netlify Not Mixing?**: A question popped up about hosting a **Mojo** app with `lightbug_http` on **Netlify**, drawing on success with Rust apps.
   - Members said **Netlify** lacks native Mojo support, referencing [available software at build time](https://docs.netlify.com/configure-builds/available-software-at-build-time/) for possible features.
- **Mojo’s Domain Dilemma**: One user asked if **Mojo** would split from **Modular** and claim a `.org` domain like other languages.
   - Developers confirmed no such move is planned, affirming it stays under **modular.com** for now.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Workflows Soar on GCloud Run**: A new guide explains [launching a two-branch RAG application](https://t.co/nU1BctUh7s) on **Google Cloud Run** for ETL and query tasks, detailing a serverless environment and event-driven design via **LlamaIndex**.
   - Members noted the top three features — **two-branch** architecture, **serverless** hosting, and an **event-driven** approach — as keys to streamlined AI workloads.
- **Chat2DB GenAI Chatbot Tackles SQL**: Contributors highlighted the open-source [Chat2DB chatbot](https://t.co/l1SFCEkiOC), explaining that it lets users query databases in everyday language using **RAG** or **TAG** strategies.
   - They emphasized its multi-model compatibility, supporting **OpenAI** and **Claude**, which makes it a flexible tool for data access.
- **LlamaParse Rescues PDF Extraction**: Participants recommended [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) for PDF parsing, calling it the world’s first **genAI-native** document platform for LLM use cases.
   - They praised its robust data cleaning and singled it out as a solution for tricky selectable-text PDFs.
- **Incognito Mode Zaps Docs Glitch**: A user reported that [LlamaIndex documentation](https://docs.llamaindex.ai/) kept scrolling back to the top when viewed in a normal browser session.
   - They confirmed **incognito mode** on Microsoft Edge solved the glitch, suggesting an extension conflict as the likely cause.
- **CAG with Gemini Hits an API Wall**: Someone asked how to integrate **Cached Augmented Generation (CAG)** into Gemini, only to learn that model-level access is essential.
   - They discovered **no providers** currently offer that depth of control over an API, stalling the idea for now.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **ModernBert Entities Emerge**: A user showcased syntax for identifying entities in **ModernBert**, providing a hierarchical document layout for travel topics and seeking best practices for embeddings.
   - They looked for suggestions on structuring these documents around entity-based tasks, hoping to refine overall performance.
- **Jinja Trove Takes Center Stage**: A participant requested robust resources on the advanced features of **Jinja** templates, prompting a surge of community interest.
   - Others chimed in, noting that improved template logic can streamline dynamic rendering in various projects.
- **LMstudio Inquiry Finds a Home**: Another user sought guidance on **LMstudio**, asking if the current channel was appropriate while struggling to find a dedicated Discord link.
   - They also touched on **Adobe Photoshop** issues, leading to tongue-in-cheek comments about unofficial support lines.
- **Photoshop and Illegal Humor**: A short exchange hinted at a possibly illegal question regarding **Adobe Photoshop**, prompting jokes about the nature of such inquiries.
   - Discussion briefly shifted toward broader concerns over sharing questionable requests in public forums.
- **Nomic Taxes and Intern Levies**: Members joked about tax increases for **Nomic**, with one participant claiming they should be the recipient of these funds.
   - A fun reference to [this GIF](https://tenor.com/view/willj-oprah-oprah-winfrey-winfrey-you-get-a-car-gif-2219821026349492069) highlighted the playful tone of the conversation.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Bud-E Speaks in 13 Tongues**: LAION revealed that **Bud-E** extends beyond English, supporting **13 languages** without specifying the complete list, tapping into fish TTS modules for speech capabilities.
   - The team temporarily ‘froze’ the existing project roadmap to emphasize **audio** and **video** dataset integration, causing a slight development delay.
- **Suno Music’s Sonic Power**: The [Suno Music](https://x.com/SunoMusic/status/1881742789639057828) feature allows users to craft their own songs by recording custom audio inputs, appealing to mobile creators seeking fast experimentation.
   - Members expressed excitement over **broad accessibility**, highlighting the platform’s potential to diversify creative workflows.
- **BUD-E & School-BUD-E Take Center Stage**: LAION announced **BUD-E** version 1.0 as a 100% open-source voice assistant for both general and **educational** use, including [School Bud-E](https://www.youtube.com/watch?v=y4DRYF9sfMU) for classrooms.
   - This milestone promotes **universal access** and encourages AI-driven **ed-tech**, showcased in a [tutorial video](https://www.youtube.com/watch?v=IxHnpISMNPo) illustrating BUD-E’s capabilities.
- **BUD-E’s Multi-Platform Flexibility**: Engineers praised BUD-E for offering compatibility with self-hosted APIs and local data storage, ensuring **privacy** and easy deployment.
   - According to [LAION’s blog post](https://laion.ai/blog/bud-e-release/), **desktop** and **web** variants cater to broad user needs, amplifying free educational reach worldwide.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Declaration Form Confusion**: A member asked if they must fill the Declaration Form again after submitting in December, clarifying that only new folks must submit now.
   - Staff reopened the form for those who missed the initial deadline, ensuring no extra steps for prior filers.
- **Sponsors Offer Hackathon-Style Projects**: A participant asked if corporate sponsors would provide intern-like tasks in the next MOOC, referencing last term’s hackathon as inspiration.
   - Organizers indicated that sponsor-led talks may hint at internship leads, though no formal arrangement was revealed.
- **MOOC Syllabus Teased for January 27**: A member wondered when the new MOOC syllabus will drop, prompting staff to note **January 27** as the likely date.
   - They are locking in guest speakers first, but promise a rough outline by that day.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **BEAM Bogs Down YoloV8**: One user reported that running **YoloV8** with `python examples/webgpu/yolov8/compile.py` under **BEAM** slashed throughput from **40fps** to **8fps**, prompting concerns about a bug.
   - **George Hotz** noted that **BEAM** should not degrade performance and suggested investigating potential anomalies in the code path.
- **WebGPU-WGSL Hurdles Slow BEAM**: Another user suspected that **WGSL** conversion to **SPIR-V** might increase overhead, crippling real-time inference speeds.
   - They also emphasized that **BEAM** requires exact backend support, raising questions about hardware-specific optimizations for **WebGPU**.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune's 'Tune Cat' Gains Momentum**: A member praised the **Torchtune** package and referenced a [GitHub Issue](https://github.com/pytorch/torchtune/issues/2281) proposing a `tune cat` command to streamline usage.
   - They described the source code as *an absolute pleasure to read*, signaling a strongly positive user experience.
- **TRL's Command Bloats Terminal**: A member joked that the **TRL** help command extends across **three** terminal windows, overshadowing typical help outputs.
   - They suggested the verbose nature might still be crucial for users who want all technical details.
- **LLMs Explore Uncertainty & Internal Reasoning**: Discussion centered on the idea that **models should quantify uncertainty** to bolster reliability, while **LLMs** appear to conduct their own chain of thought before responding.
   - Both points underscore a move toward better interpretability, with signs of covert CoT steps for deeper reasoning.
- **Advancing RL with LLM Step-Prompts & Distillation**: A suggestion emerged for **RL-LLM thinking-step prompts**, adding structure to standard goal-based instructions.
   - Another member proposed applying RL techniques on top of model distillation, expecting further gains even for smaller models.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Dynamic DSPy: RAG's Rendezvous with Real-Time Data**: A user asked how **DSPy-based RAG** manages changing information, hinting at the importance of real-time updates for knowledge retrieval pipelines with minimal overhead.
   - They suggested future work could focus on caching mechanisms and incremental indexing, keeping **DSPy** agile for dynamic workloads.
- **Open Problem Ordeal & Syntax Slip-ups**: Another thread raised an **open problem** in DSPy, underscoring continued interest in a lingering technical question.
   - A syntax error (*'y=y'* should use a number) also emerged, highlighting attention to detail and the community's engagement in squashing small issues.



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **ArXiv Authors Demand Better Data**: The paper titled [Towards Best Practices for Open Datasets for LLM Training](https://discord.com/channels/1089876418936180786/1331335526338265108) was published on ArXiv, detailing challenges in open-source AI datasets and providing recommendations for **equity** and **transparency**.
   - Community members praised the blueprint’s potential to **level the playing field**, highlighting that a stronger open data ecosystem drives LLM improvements.
- **Mozilla & EleutherAI Declare Data Governance Summit**: Mozilla and **[EleutherAI](https://discord.gg/cJQKYFDwHV)** partnered on a Dataset Convening focused on responsible stewardship of open-source data and governance.
   - Key stakeholders discussed best curation practices, stressing the shared goal of advancing LLM development through **collaborative** community engagement.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **AI Shifts from Hype to Help in Cybersecurity**: One member recalled how **AI** used to be a mere *buzzword* in cybersecurity, noting their transition into the field a year ago.
   - They expressed excitement about deeper integration of **AI** in security processes, envisioning real-time threat detection and automated incident response.
- **Security Teams Embrace AI Support**: The discussion highlighted the growing interest in how **AI** can bolster security teams’ capabilities, especially in handling complex alerts.
   - Enthusiasts anticipate sharper analysis tools that **AI** offers, allowing analysts to focus on critical tasks and reduce manual overhead.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Axolotl AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **OpenInterpreter Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1330991993865896169)** (652 messages🔥🔥🔥): 

> `DeepSeek-R1 model limitations, Fine-tuning strategies for classification tasks, Handling model checkpoints, Model tokenization and embeddings, Challenges in using Unsloth notebooks` 


- **Discussion on DeepSeek-R1 model limits**: Users noted that the DeepSeek-R1 model's maximum token length is capped at 16,384, despite calculations suggesting it should be 163,840 based on its embeddings.
   - This discrepancy led to speculation about a potential bug or error during the model’s deployment.
- **Strategies for fine-tuning models**: A user inquired if instruction fine-tuning (IFT) should begin from the last checkpoint in conversational pre-training (CPT), highlighting a lack of the necessary code in their notebook.
   - It was clarified that IFT should start from the checkpoint relevant to its task rather than the most recent CPT checkpoint.
- **Issues with model conversion in Unsloth**: A new user reported difficulties converting Phi4 files to GGUF format after successful initial training, encountering vague error messages.
   - Advice was offered regarding the need to merge the tokenizer if heads or embeddings were trained, indicating a potential source of the conversion issue.
- **The impact of tokenization and embeddings**: Discussion focused on the significance of tied weights for embeddings in models like Llama 3.2, which may influence model performance and context length capabilities.
   - Users reflected on the potential implications of these configurations for small models and their efficiency.
- **Experiments with model output generation**: Strategies for enhancing output quality through methods like beam search and minimum perplexity branch selection were explored.
   - Participants discussed the merits of multi-turn reasoning and adding judging models for improved decision-making in outputs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/fimbulvntr/status/1881821582571761920">Tweet from Fimbul (@fimbulvntr)</a>: Am I going crazy or is DeepSeek-R1 capped to a model_max_length of 16384?I think this is a bug. In reality it should be 163840.It has original_max_position_embeddings=4096 and a RoPE factor of 40... 4...</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/JingzeShi/Doge-20M-Instruct">JingzeShi/Doge-20M-Instruct · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/so-cute-cat-love-head-pat-gif-14623443">So Cute Cat GIF - So Cute Cat Love - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main?show_file_info=model.safetensors">meta-llama/Llama-3.2-1B at main</a>: no description found</li><li><a href="https://colab.research.google.com">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/3302ba78c0090838341caf8adfbe1e231308fa95/tokenizer_config.json#L22">tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 3302ba78c0090838341caf8adfbe1e231308fa95</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B/tree/main?show_file_info=model.safetensors">unsloth/Llama-3.2-1B at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs&list=LL&index=2&pp=gAQBiAQB8AUB">Large Language Models explained briefly</a>: Dig deeper here: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piTechnical details as a talk: https://youtu.be/KJtZARuO3JYThis was ma...</li><li><a href="https://gist.github.com/sebaxakerhtc/5e7faa4ead6e2f4e0ea69634c3f624ba">Guided script for Unsloth</a>: Guided script for Unsloth. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://huggingface.co/blog/llama32?utm_source=chatgpt.com#what-is-special-about-llama-32-1b-and-3b">Llama can now see and run on your device - welcome Llama 3.2</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">Let&#39;s build GPT: from scratch, in code, spelled out.</a>: We build a Generatively Pretrained Transformer (GPT), following the paper &quot;Attention is All You Need&quot; and OpenAI&#39;s GPT-2 / GPT-3. We talk about connections t...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>: no description found</li><li><a href="https://github.com/bagel-org/ZKLoRA">GitHub - bagel-org/ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification</a>: Efficient Zero-Knowledge Proofs for LoRA Verification - bagel-org/ZKLoRA</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L766">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: Finetune Llama 3.3, Mistral, Phi-4, Qwen 2.5 &amp; Gemma LLMs 2-5x faster with 70% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1331361373250129952)** (1 messages): 

> `Unsloth training, Fine-tuning LLMs, Weights & Biases integration, vLLM for model serving` 


- **Gautam's Guide to Fine-tuning Unsloth**: A Medium article by Gautam Chutani discusses **fine-tuning LLaMA 3** using **LoRA**, with a particular focus on the integration of [Weights & Biases](https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060) for monitoring and utilizing **vLLM** for model serving.
   - *Fine-tuning provides a way to optimize pre-trained models* for specialized tasks, but challenges arise due to **computational resource limitations**.
- **Challenges in Fine-tuning LLMs**: The article emphasizes that **fine-tuning large language models (LLMs)** is crucial for adapting them to specific tasks but presents challenges due to the **computational resource requirements** involved.
   - Traditional fine-tuning methods demand significant **GPU memory** and computation time, which can be a barrier for many practitioners.



**Link mentioned**: <a href="https://gautam75.medium.com/fine-tuning-llama-3-1-8b-for-function-calling-using-lora-159b9ee66060">Fine-Tuning Llama-3.1-8B for Function Calling using LoRA</a>: Leveraging Unsloth for fine-tuning with Weights &amp; Biases integration for monitoring and vLLM for model serving

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1331083511221719142)** (56 messages🔥🔥): 

> `Fine-tuning models, Using Unsloth with different datasets, Models compatibility, Training on reasoning tasks, Handling CUDA memory issues` 


- **General Fine-tuning Guidelines with Unsloth**: Users discussed the process of fine-tuning models like **Llama** and **Phi-4** using **Unsloth**, emphasizing combining datasets for better performance.
   - One user mentioned that fine-tuning a model with instruction data significantly enhances training results compared to post-training adjustments.
- **Model Compatibility and Mixing Datasets**: It was clarified that when using Unsloth, models based on supported frameworks, such as **Mistral**, are also compatible, and datasets do not need to be in the same format.
   - Users shared strategies for mixing and formatting datasets, with suggestions to partition and convert different dataset formats.
- **Issues with Model Outputs**: Users raising concerns about consistently similar outputs from the **Phi-4** model recommended techniques like prepending seed values to inputs to diversify the results.
   - One user shared experiences using a particular notebook for conversational training with Phi-4, encountering issues like failures to convert saved files.
- **CUDA Memory Management Tips**: In response to CUDA out-of-memory errors, reducing the batch size for training was recommended as a solution while retaining certain fixed parameters like **r=128**.
   - Participants shared insights on memory management and optimal configurations for fine-tuning on various hardware setups.
- **API Utilization and Setup Challenges**: Users inquired about running models locally on machines with limited resources, such as Macs, and suggested utilizing APIs for easier integration.
   - Clarifications were provided regarding model implementation challenges, including quantization limitations with existing setups.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://docs.unsloth.ai/basics/d">Unsloth Documentation</a>: no description found</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Documentation</a>: New to Unsloth?</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#multiple-datasets">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#formatting-our-data">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://github.com/microsoft/Phi-3CookBook/blob/main/code%2F04.Finetuning%2FPhi-3-finetune-lora-python.ipynb">Phi-3CookBook/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb at main · microsoft/Phi-3CookBook</a>: This is a Phi Family of SLMs book for getting started with Phi Models. Phi a family of open sourced AI models developed by Microsoft. Phi models are the most capable and cost-effective small langua...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1331004403045367879)** (12 messages🔥): 

> `OpenWebUI integration, Synthetic datasets, Free/Open-source solutions, Colab script testing` 


- **Future plans for OpenWebUI and More**: *Invisietch* mentioned plans to eventually integrate with **OpenWebUI**, **Ollama**, **Flowise**, and **LocalAI** while currently working with **Kobold** and **Aphrodite** using the Kobold API.
   - It was noted that there is a lot on the to-do list but progress on existing tools is being made.
- **Discussion on Free/Open-source Solutions**: *Sebaxakerhtc* emphasized using only **free/open-source solutions** in their work, prompting a clarification from *invisietch* that both **Koboldcpp** and **Aphrodite** are indeed free software.
   - *Invisietch* mentioned that a project called **Chatterbox** would also be available as free software once a license file is added.
- **Synthetic Datasets Automation**: *Invisietch* posed a question about creating **synthetic datasets on autopilot**, suggesting that a command-line interface (CLI) could be beneficial for bulk operations.
   - The discussion indicated a focus on utilizing the same backend API for this functionality.
- **Successful Script Testing in Colab**: *Sebaxakerhtc* successfully tested a script in **Google Colab**, achieving everything from zero to saving the **GGUF**, with outputs saved for viewing.
   - This garnered positive reactions, with another user commenting on the achievement as 'really cool.'



**Link mentioned**: <a href="https://colab.research.google.com/drive/1NwVnNtj-_o6vTUUsgM5BMAVRPUpTZQU2?usp=sharing">Google Colab</a>: no description found

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1331320261567451228)** (169 messages🔥🔥): 

> `Chinchilla Optimal training, Synthetic data in AI training, Emotional tracking in AI, Grokking in language models, 3D modeling vs text in AI applications` 


- **Understanding Chinchilla Optimal training**: The **Chinchilla** paper suggests a balance between model size and training tokens for optimal performance, indicating that both should be scaled equivalently to avoid inefficiencies. This concept has become vital for determining how much training data large language models require to achieve the best results.
   - *Knowledge does not cluster as 'experts', thus the Chinchilla optimal applies to the total parameters, not just a subset.*
- **Discussions on synthetic data for AI training**: Members discussed the potential of using **synthetic data streams** to create training datasets that align more closely with evaluation compliance. This could lead to a tighter train/test loop that dynamically adjusts based on model performance, avoiding overfitting.
   - Concerns were raised about the limitations of synthetic data, particularly its relevance to real-world applications, noting that not all domains have the luxury of unlimited synthetic data.
- **Emotional tracking in AI systems**: One member shared their work in the **erotic industry**, highlighting how bots strive to emulate human behavior accurately through emotional tracking and psychological principles. This includes the integration of middleware to manage states in a real-time context.
   - The approach emphasizes that emotional tracking is based on established psychological frameworks, rather than relying solely on the capabilities within LLMs.
- **Grokking and scaling in AI models**: The concept of **grokking**—the ability of a model to deeply understand a domain—was discussed with a focus on how training data organization is pivotal to achieving this. Suggestions were made to stratify training from simpler to complex tasks to maximize comprehension across different levels of abstraction.
   - Members suggested that optimizing for basic reasoning might help achieve a 100x compression improvement in future models, enabling practical applications with significantly fewer parameters.
- **Debate over the use of 3D models in AI applications**: The conversation touched on whether 3D models are practical or relevant in current AI applications, with a focus on chat and voice interactions being more lucrative. While some pointed out advancements in AI that allow for 3D generation, the consensus leaned towards established text and voice applications yielding better returns.
   - Participants acknowledged differing perspectives on technology adoption in the industry, particularly concerning what generates revenue in the erotic AI sector.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.13148">SWAN: SGD with Normalization and Whitening Enables Stateless LLM Training</a>: Adaptive optimizers such as Adam (Kingma &amp; Ba, 2015) have been central to the success of large language models. However, they often require to maintain optimizer states throughout training, which ...</li><li><a href="https://arxiv.org/abs/2305.07759">TinyStories: How Small Can Language Models Be and Still Speak Coherent English?</a>: Language models (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GP...</li><li><a href="https://paperswithcode.com/method/chinchilla">Papers with Code - Chinchilla Explained</a>: Chinchilla is a 70B parameters model trained as a compute-optimal model with 1.4 trillion tokens. Findings suggest that these types of models are trained optimally by equally scaling both model size a...
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1331000801463762965)** (467 messages🔥🔥🔥): 

> `DeepSeek R1 integration, Cursor 0.45 updates, OpenAI Stargate Project, AI competition, Claude 3.5 performance` 


- **DeepSeek R1 can be added to Cursor**: Users have found a way to integrate DeepSeek R1 into Cursor via OpenRouter, though the integration is currently poor and limits access to other models.
   - It was suggested that waiting for proper integration is preferable, with many expressing interest in using R1 without touching Cursor for the time being.
- **Cursor 0.45 updates keep rolling back**: The latest updates for Cursor, including version 0.45.1, have been rolled back multiple times due to issues related to codebase indexing and model compatibility.
   - Users are experiencing inconsistencies with the update process, often reverting to earlier versions.
- **OpenAI's Stargate Project announced**: OpenAI announced a $500 billion investment plan called the Stargate Project aimed at building new AI infrastructure in the United States with funding from SoftBank and others.
   - This announcement has sparked discussions about the rapidly evolving AI competition, especially in light of significant investments from Japan.
- **AI competition heating up with DeepSeek**: The presence of DeepSeek R1 has prompted discussions about improved performance in similar models, leading to suggestions that AI capabilities are becoming more accessible.
   - Comparisons between DeepSeek R1 and Claude 3.5 highlight the competitive landscape in AI development.
- **Claude 3.5's performance shines amidst competition**: Claude 3.5's performance has been noted to improve significantly, with users commenting on its speed and accuracy, possibly due to competition from DeepSeek R1.
   - Anthropic's recent lack of updates has raised curiosity about the company's strategies moving forward in this competitive environment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1881428345973608901">Tweet from Paul Gauthier (@paulgauthier)</a>: DeepSeek R1 gets 57% on the aider polyglot benchmark, ranks 2nd behind o1:62% o1 (high)57% DeepSeek R152% Sonnet48% DeepSeek Chat V3Full leaderboard:https://aider.chat/docs/leaderboards/</li><li><a href="https://x.com/kimmonismus/status/1881734307158397442">Tweet from Chubby♨️ (@kimmonismus)</a>: Dario Amodei said, &#34;I have never been more confident than ever before that we’re close to powerful AI systems. What I’ve seen inside Anthropic and out of that over the last few months led me to be...</li><li><a href="https://x.com/kregenrek/status/1878487131099898269?s=46">Tweet from Kevin Kern (@kregenrek)</a>: Introducing Codefetch for DevelopersTurn code into Markdown for LLMs with one simple terminal command.Use it in bolt .new, cursor and many more AI coding tools.→ Chat with your codebase→ Save tokens→ ...</li><li><a href="https://x.com/OpenAI/status/1881830103858172059?s=19">Tweet from OpenAI (@OpenAI)</a>: Announcing The Stargate ProjectThe Stargate Project is a new company which intends to invest $500 billion over the next four years building new AI infrastructure for OpenAI in the United States. We wi...</li><li><a href="https://www.cursor.com/downloads">Cursor - The AI Code Editor</a>: Built to make you extraordinarily productive, Cursor is the best way to code with AI.</li><li><a href="https://www.cursor.com/blog/shadow-workspace">Iterating with Shadow Workspaces | Cursor - The AI Code Editor</a>: Hidden windows and kernel-level folder proxies to let AIs iterate on code without affecting the user.</li><li><a href="https://tenor.com/view/crazy-alert-crazy-alert-alerta-alerta-loca-gif-2463480864319782005">Crazy Alert Crazy GIF - Crazy Alert Crazy Alert - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/bg3EF.gif">Trust Me No One Is Going To Notice Grey Griffin GIF - Trust Me No One Is Going To Notice Grey Griffin Karen Crawford - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1">DeepSeek R1 - API, Providers, Stats</a>: DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model &amp; technical report🏆 MIT licensed: Distill &amp; commercialize freely!. Run DeepSeek R1 with API</li><li><a href="https://status.cursor.com/">Cursor Status</a>: no description found</li><li><a href="https://status.cursor.com/?utm_source=embed">Cursor Status</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero">deepseek-ai/DeepSeek-R1-Zero · Hugging Face</a>: no description found</li><li><a href="https://forum.cursor.com/t/how-to-add-a-custom-model-like-deepseek-v3-which-is-openai-compatible/37423/22?u=irian-codes">How to add a custom model like DeepSeek-V3 (which is OpenAI compatible)</a>: Here is my post how to add DeepSeek into Cursor.    It works great in chat and completions. But Composer not allow using external models. We should push Cursor team to add this feature.      UPD: You ...</li><li><a href="https://forum.cursor.com/t/please-add-deepseek-r1-model/42868">Please add DeepSeek R1 model</a>: Apparently better and way cheaper than Sonnet? To be seen…</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://downloader.cursor.sh/mac/dmg/arm64">no title found</a>: no description found</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">Reasoning Model (deepseek-reasoner) | DeepSeek API Docs</a>: deepseek-reasoner is a reasoning model developed by DeepSeek. Before delivering the final answer, the model first generates a Chain of Thought (CoT) to enhance the accuracy of its responses. Our API p...</li><li><a href="https://downloader.cursor.sh/linux/appimage">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1331003996479164468)** (49 messages🔥): 

> `Windsurf performance issues, DeepSeek model comparisons, Error troubleshooting, Codeium features and requests, User experiences with tools` 


- **Users report performance lags in Windsurf**: Several users are experiencing lag delays in prompting within Windsurf, prompting discussions about potential fixes and support.
   - One member mentioned trying to adjust settings in order to alleviate the lag but received no definitive solutions so far.
- **DeepSeek model R1 outperforms previous models**: Members highlighted that the DeepSeek R1 model reportedly exceeds the performance metrics of the OpenAI O1-preview, sparking interest for integration into Codeium.
   - Despite the excitement, concerns were raised about its ability to handle tool calls effectively, making integration uncertain at the moment.
- **Error messages and troubleshooting circulating**: Multiple users have shared experiences with errors in Windsurf, including messages such as 'incomplete envelope: unexpected EOF.'
   - Community members are discussing various solutions and the potential need for system permission adjustments to resolve these issues.
- **Request for features and improvements in Codeium**: A user has urged the Codeium team to add the DeepSeek R1 model and expressed hope for fine-tuning opportunities.
   - Others voiced concerns regarding the lack of updates or improvements for JetBrains IDE users, feeling less prioritized compared to Windsurf users.
- **Mixed reviews on Codeium features and support**: Users expressed mixed feelings regarding their experiences with Codeium's support and feature availability, comparing it to other tools like Co-pilot.
   - There was notable frustration over difficulties in purchasing credits and no clear communication on support availability, reflecting a demand for better customer service.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1881442133376454694?t=kcwBO9GpmTX5zzXVtA63gA&s=19">Tweet from Xeophon (@TheXeophon)</a>: holyshitwhatthefuckr1 beats o1-preview on my bench</li><li><a href="https://x.com/TheXeophon/status/1881443117787984265?t=CWcMfDus2ULxJQS6VnnQRA&s=19">Tweet from Xeophon (@TheXeophon)</a>: I am shocked by R1 on my personal bench. This is the full eval set, it completely crushes the competition and is a whole league on its own, even surpassing o1-preview (which is omitted from the graph ...</li><li><a href="https://www.reddit.com/r/synology/comments/pq0411/cant_mount_network_drive_in_windows_explorer/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/Exafunction/codeium/releases/tag/termium-v0.2.1">Release termium-v0.2.1 · Exafunction/codeium</a>: Automated release</li><li><a href="https://codeium.com/blog/termium-codeium-in-terminal-launch">Termium: Codeium in the Terminal</a>: AI-powered autocomplete for your terminal commands.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1330995751144394805)** (351 messages🔥🔥): 

> `Windsurf performance issues, DeepSeek integration, Flow Actions limit, Quality of suggestions, Bug reporting and troubleshooting` 


- **Windsurf suffers from performance setbacks**: Many users report issues with Windsurf's performance, specifically citing lag during prompts and unwanted edits being made to code.
   - The recent updates have introduced bugs that frustrate users, leading some to consider alternatives like Cursor.
- **Integration with DeepSeek's model**: Users enquire about the possibility of incorporating DeepMind or DeepSeek models into Windsurf, with suggestions that using compatible APIs could facilitate this.
   - Some recommend utilizing compatible plugins like Cline for enhanced functionality.
- **Challenges with Flow Actions limit**: Users express concern over the Flow Actions limit, indicating it causes bottlenecks in their productivity and suggesting the need for strategies to mitigate this issue.
   - Some offer insights on how to manage these limits more effectively.
- **Users discuss quality of suggestions**: Feedback reveals dissatisfaction with the quality of suggestions made by Windsurf, compared to competitors like Cursor that provide more targeted edits.
   - Discussions involve whether Windsurf's algorithm is capable of performing as effectively as the alternatives available.
- **Reporting bugs and troubleshooting**: Several users are facing persistent bugs and issues, urging others to submit support tickets along with diagnostic logs for thorough troubleshooting.
   - Users continue discussing various workarounds and experiences related to bug reporting.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.</li><li><a href="https://cloud.dwavesys.com/leap/">D-Wave Leap Log In | D-Wave Leap™</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/web-search">Web Search - Codeium Docs</a>: no description found</li><li><a href="https://chat.deepseek.com/downloads/DeepSeek%20Privacy%20Policy.html">DeepSeek Privacy Policy</a>: no description found</li><li><a href="https://tenor.com/view/ninja-fortnite-reaction-ninja-low-taper-fade-gif-1784137995500051652">Ninja Fortnite GIF - Ninja Fortnite Reaction - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.basedpyright.com/latest/installation/pre-commit%20hook/">pre-commit hook - basedpyright</a>: no description found</li><li><a href="https://www.reddit.com/media?url=https%3A%2F%2Fi.redd.it%2Fjf6vo05hx8ee1.jpeg">https://i.redd.it/jf6vo05hx8ee1.jpeg</a>: no description found</li><li><a href="https://codeium.canny.io/feature-requests/p/add-rename-suggestion-like-in-vscode">Add rename suggestion like in VScode Copilot | Feature Requests | Codeium</a>: Use codeium/cascade to suggest renaming options when using alt+r</li><li><a href="https://www.reddit.com/r/Codeium/comments/1i5ftc9/heres_a_balanced_critique_of_windsurfs_business/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://status.codeium.com/#">Codeium Status</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=moIySJ4d0UY">Web Search Best Practices: Save Credits and Optimize Your Workflow - Windsurf Editor</a>: Ready to get the most out of Windsurf&#39;s brand-new Web Search feature? This deep dive is here to help you unlock its full potential!In this video, you’ll lear...</li><li><a href="https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md">pyright/docs/mypy-comparison.md at main · microsoft/pyright</a>: Static Type Checker for Python. Contribute to microsoft/pyright development by creating an account on GitHub.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i5q6b9/deepseekr1_and_distilled_benchmarks_color_coded/">Reddit - Dive into anything</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1331027328418975885)** (1 messages): 

> `Aider v0.72.0 Release, DeepSeek R1 Support, Kotlin Syntax Support, File Handling Enhancements, Bugfixes and Improvements` 


- **Aider v0.72.0 Launches with Exciting Features**: The release of **Aider v0.72.0** includes support for DeepSeek R1, accessible via the shortcut `--model r1` or through OpenRouter.
   - In this update, **Aider** contributed **52%** of the code, indicating significant in-house development.
- **Enhanced File Handling and Syntax Support**: Support for **Kotlin syntax** has been added to the repo map along with a new option `--line-endings` for improved file writing.
   - Additionally, `examples_as_sys_msg=True` for GPT-4o models boosts benchmark scores.
- **Bugfixes Address Common Issues**: This version addresses several bugs including a **permissions issue** in Docker images and an ASCII fallback for **unicode errors**.
   - Another notable fix improves integer indices for list slicing in repomap calculations.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1330990775189901342)** (297 messages🔥🔥): 

> `DeepSeek R1 performance, Comparison of AI models, OpenAI subscription discussions, Hardware pricing and availability, Data usage and privacy concerns` 


- **DeepSeek R1 impresses users**: Many users are expressing satisfaction with **DeepSeek R1**, noting its performance in various tasks and its ability to handle multiple languages effectively.
   - Some mentioned it may be better suited for specific coding tasks compared to other models, emphasizing its unique capabilities.
- **Differences in AI model outputs**: Users are observing inconsistent performance levels between models like **Sonnet** and **DeepSeek**, with reports of varying quality outputs based on geographic location.
   - Conversations highlighted discrepancies between European and US performance, and users are encouraged to consider different models for specific applications.
- **OpenAI subscription reflections**: Several members discussed their experiences with OpenAI's subscription services, including recent refunds and price comparisons.
   - The perception is that **DeepSeek** offers good value, with some members expressing interest in switching from **Claude** to **DeepSeek R1** due to cost effectiveness.
- **Hardware prices in Europe**: Several users shared insights on the prices of GPUs in Europe, noting surprisingly low prices for older models like the **RTX 3060** and **3090**.
   - Despite the rise of new generation GPUs, users are considering purchasing older models at discounted rates from European sellers.
- **Concerns about data usage in AI**: Discussions on the implications of using AI models centered around concerns about data privacy and ownership, with users contemplating how their code may be utilized by platforms.
   - Members generally expressed a relaxed attitude towards data usage, considering most of their code not proprietary enough to warrant concern.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/Kimi_ai_/status/1881332472748851259">Tweet from Kimi.ai (@Kimi_ai_)</a>: 🚀 Introducing Kimi k1.5 --- an o1-level multi-modal model-Sota short-CoT performance, outperforming GPT-4o and Claude Sonnet 3.5 on 📐AIME, 📐MATH-500, 💻 LiveCodeBench by a large margin (up to +550%...</li><li><a href="https://x.com/0xluffyb/status/1881323971897110866">Tweet from luffy (@0xluffyb)</a>: everyone todayQuoting DeepSeek (@deepseek_ai) 🚀 DeepSeek-R1 is here!⚡ Performance on par with OpenAI-o1📖 Fully open-source model & technical report🏆 MIT licensed: Distill & commercialize freely!🌐 ...</li><li><a href="https://docs.fireworks.ai/guides/security_compliance/data_handling#data-privacy-and-security)">Data privacy &amp; security - Fireworks AI Docs</a>: no description found</li><li><a href="https://br.ign.com/tech/135086/news/ceo-da-openai-nao-sabe-o-que-fazer-com-o-comportamento-dos-assinantes-do-chatgpt">CEO da OpenAI não sabe o que fazer com o comportamento dos assinantes do ChatGPT</a>: Ele escolheu o preço sem pensar muito e achou que ganharia dinheiro</li><li><a href="https://aider.chat/docs/usage/not-code.html">Editing config &amp; text files</a>: Edit configuration files, documentation, and other text-based formats.</li><li><a href="https://tenor.com/view/megatron-upgrade-unicron-behold-galvatron-gif-26590123">Megatron Upgrade GIF - Megatron Upgrade Unicron - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://unsloth.ai/blog/deepseek-r1">Run Deepseek-R1 / R1 Zero</a>: DeepSeek&#x27;s latest R-1 model is the most powerful open-source reasoning model that performs on par with OpenAI&#x27;s o1 model.Learn how to run &amp; fine-tune the model.</li><li><a href="https://tenor.com/view/bear-embarrassed-smiling-gif-11674756">Bear Embarrassed GIF - Bear Embarrassed Smiling - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://docs.fireworks.ai/guides/security_comp">Introduction - Fireworks AI Docs</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i64up9/model_comparision_in_advent_of_code_2024/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://gist.github.com/murdockq/b08f72699fd7d8db556a14e69a7cb0c3">a game prompt.md</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://github.com/matatonic/openedai-whisper">GitHub - matatonic/openedai-whisper: An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper.</a>: An OpenAI API compatible speech to text server for audio transcription and translations, aka. Whisper. - matatonic/openedai-whisper</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/Devographics/surveys/issues/278">State of AI 2025 Preview · Issue #278 · Devographics/surveys</a>: Here is a preview link for the upcoming State of Web Dev AI 2025 survey, the first ever edition of this new survey: https://survey.devographics.com/en-US/survey/state-of-ai/2025 I would love to get...</li><li><a href="https://api.ailocal.org">Whisper.cpp Server</a>: no description found</li><li><a href="https://github.com/Aider-AI/aider/issues/429">Tree-sitter tsx parser hangs sometimes, causing aider to hang · Issue #429 · Aider-AI/aider</a>: User reports aider hangs when using a repo full of .tsx files. Using --no-git removes the hang. Issue appears to be in the repo map code. https://discord.com/channels/1131200896827654144/1192136795...</li><li><a href="https://endpoints.huggingface.co/catalog">Inference Catalog | Inference Endpoints by Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1330994845317070898)** (89 messages🔥🔥): 

> `Using Aider with Sonnet, Updating Aider Versions, Error Handling in Aider, DeepSeek Model Comparisons, Refactoring Python Codebases` 


- **Using Aider with Sonnet's Context Window**: Concerns arose over being unable to access **Sonnet's full context window** until a **$400 investment** is made on Anthropic's platform, which hobbyists might find excessive.
   - This raises questions about the accessibility and affordability of advanced AI tools for casual developers.
- **Challenging Aider Updates**: Several members expressed difficulties updating Aider, specifically moving from **0.70.0 to the latest version**, with some unclear on commands to use.
   - Common solutions included using commands like `aider --upgrade` or reinstalling directly, although success varied.
- **Error Handling: API Keys and Configurations**: Issues with **invalid API keys** emerged, prompting discussions on how **.env configurations** can override settings in **.conf files** and affect project usage.
   - One member stated that removing a disabled key from the **.env file** resolved their issues, illustrating the importance of configuration management.
- **Comparing DeepSeek Models**: Questions were raised about the performance of **DeepSeek-R1 vs. DeepSeek-V3** in terms of their architectural modes and usage configurations.
   - Members speculated on the role of caching in DeepSeek's efficiency, with inquiries about its integration with Aider via cache-prompts settings.
- **Refactoring Python Codebases**: Discussions around refactoring a **12-file Python codebase** highlighted strategies, with suggestions to use tools like **Gemini Pro** for managing large contexts efficiently.
   - Participants noted that while incremental changes help, optimizing the process for accuracy and efficiency is still a work in progress.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">File editing problems</a>: aider is AI pair programming in your terminal</li><li><a href="https://engineering.fb.com/2024/12/19/developer-tools/glean-open-source-code-indexing/">Indexing code at scale with Glean</a>: We’re sharing details about Glean, Meta’s open source system for collecting, deriving, and working with facts about source code. In this blog post we’ll talk about why a system like Glean is import…</li><li><a href="https://github.com/BerriAI/litellm/issues/7877">[Feature]: DeepSeek-R1 support · Issue #7877 · BerriAI/litellm</a>: The Feature DeepSeek-R1 API returns its thoughts inside the reasoning_content parameter. Currently this is ignored by LiteLLM. Their API approach, of return &quot;reasoning_content&quot; for the long-...</li><li><a href="https://github.com/getgrit/gritql">GitHub - getgrit/gritql: GritQL is a query language for searching, linting, and modifying code.</a>: GritQL is a query language for searching, linting, and modifying code. - getgrit/gritql</li><li><a href="https://github.com/jbellis/llmap">GitHub - jbellis/llmap</a>: Contribute to jbellis/llmap development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1331190132535197697)** (1 messages): 

> `Deepseek R1, Live coding experience, Space Invaders game upgrade` 


- **Deepseek R1 shines in Architect mode**: In a short live coding video, the user showcased the **Deepseek R1** in Architect mode while upgrading a **Space Invaders** type game, highlighting its features.
   - The video, titled [Space Invaders with Deepseek R1 and Aider in Architect mode](https://youtu.be/njJhjUgBTZg), emphasizes that **R1** is a top contender, being second only to **OpenAI's 01** on the Aider LLM leaderboard.
- **Deepseek R1 vs OpenAI's 01**: The user noted that **Deepseek R1** is nearly as powerful as **OpenAI's 01**, yet available at a significantly lower cost.
   - This comparison underlines the growing potential of **Deepseek R1** in AI-based applications within coding environments.



**Link mentioned**: <a href="https://youtu.be/njJhjUgBTZg">Space Invaders with Deepseek R1 and Aider in Architect mode.</a>: The new R1 model from Deepseek is second only to 01 from OpenAI on the Aider LLM leaderboard. Plus it&#39;s a fraction of the cost.Here I test out its capabiliti...

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1330995639726637139)** (342 messages🔥🔥): 

> `DeepSeek R1 Models, Mathematics Tutoring, Local Model Deployment, OpenAI Compatibility, Community Support for AI` 


- **Performance of DeepSeek R1 in Math Tutoring**: Users have reported positive experiences using DeepSeek R1, particularly for teaching math, with one user citing its effectiveness in solving complex problems and providing step-by-step reasoning.
   - The model is praised for its ability to act as a tutor, offering considerable support for users with special educational needs.
- **Exploring Model Options for Local Use**: Several users discussed their hardware setups for running different models; one user mentioned using a 4090 GPU with 64GB RAM to support heavy calculations.
   - Discussion included the idea of using home servers for accessing powerful AI capabilities and using custom clients for interaction.
- **Community and Access to AI Resources**: There was a discussion about the potential for local community colleges to provide access to AI tutoring tools like DeepSeek, which could benefit students needing additional support.
   - Users expressed a desire for community support in making these technologies accessible for educational purposes.
- **OpenAI API and Client Development**: Users talked about creating custom clients to interface with their models and questioned the compatibility with OpenAI API, highlighting the lack of support for certain endpoints.
   - One user shared their experience writing an HTML client for connecting to their server, implying the importance of understanding the syntax for effective interaction.
- **Quantization and Model Choices**: A user inquired about the significance of quantization numbers (Q3, Q4, etc.) and how they impact model performance and accuracy.
   - It was noted that lower quantization may lead to faster response times but could sacrifice some accuracy, emphasizing the need for experimentation based on user requirements.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/seo_leaders/status/1881462202831614085">Tweet from Andrew C (@seo_leaders)</a>: DeepSeek R1 671B running on 2 M2 Ultras quicker than reading speed.  Almost an open-source O1, at home, on consumer hardware.  With mlx.distributed and mlx-lm, 3-bit quantization (~4 bpw). Model is qu...</li><li><a href="https://www.audacityteam.org/download/openvino/">Download Audacity AI Plugins</a>: no description found</li><li><a href="https://ollama.com/blog/openai-compatibility">OpenAI compatibility · Ollama Blog</a>: Ollama now has initial compatibility with the OpenAI Chat Completions API, making it possible to use existing tooling built for OpenAI with local models via Ollama.</li><li><a href="https://chatboxai.app/zh">Chatbox AI官网：办公学习的AI好助手，全平台AI客户端，官方免费下载</a>: Chatbox AI 是一款 AI 客户端应用和智能助手，支持众多先进的 AI 模型和 API，可在 Windows、MacOS、Android、iOS、Linux 和网页版上使用。</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/openai.md">ollama/docs/openai.md at main · ollama/ollama</a>: Get up and running with Llama 3.3, Phi 4, Gemma 2, and other large language models. - ollama/ollama</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/openai.md#v1models>">ollama/docs/openai.md at main · ollama/ollama</a>: Get up and running with Llama 3.3, Phi 4, Gemma 2, and other large language models. - ollama/ollama</li><li><a href="https://lmstudio.ai/docs/ba">Getting Started | LM Studio Docs</a>: Learn how to run Llama, Mistral, Gemma, and other LLMs locally with LM Studio.</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents - Running LLMs Locally | LM Studio Docs</a>: How to provide local documents to an LLM as additional context
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1331034674545430598)** (31 messages🔥): 

> `AI/ML Linux Box, NVIDIA DIGITS and Compatibility, DIGITS Cost and Performance, DGX OS Insights, GPU Cooling Issues` 


- **Digit as an AI/ML Linux Box**: Digit is labeled as an **AI/ML linux box** tailored for dedicated machine learning tasks, rather than a traditional gaming PC. Users suggest the **4090 or 5090** for broader applications beyond AI.
   - One user opined that it will be great as a **home ML server**, allowing seamless job execution.
- **Confusion About NVIDIA DIGITS Functionality**: NVIDIA DIGITS has been discussed, highlighting the lack of active support and confusing details about compatibility with newer frameworks. Users debated if the latest release was simply a **software/container** focus or related to older DIGITS hardware.
   - A user pointed out **NVIDIA TAO** as an alternative open-source toolkit for AI training, indicating a shift in focus.
- **DIGITS Costs and Hardware Specs**: Concerns arose over the **high starting price** of around $3000 for the top-tier **128GB** solution in the AI mini-PC lineup, leading to skepticism about memory specifications at that cost. One user noted that products with **fast unified memory** might not be feasible at this price point.
   - Another user mentioned the importance of compatibility with popular frameworks like **PyTorch** for potential buyers.
- **Insights on DGX OS and Device Usage**: Discussion revealed that the new devices run on **DGX OS**, similar to old DIGITS, raising interest in how they operate. Users also speculated about utilizing these machines effectively without a GUI to optimize performance.
   - One user remarked on the potential for these systems to run lightweight setups, aligning with **effective memory usage** for GPU tasks.
- **GPU Cooling and Maintenance Issues**: User humorously noted not needing to clean their GPU due to excessive heat, suggesting a less pleasant experience. Concerns about the **heat management** of powerful GPUs were shared, hinting at maintenance difficulties.
   - Another user confirmed they plan to purchase the machine in a few years when available on the **second-hand market**.



**Link mentioned**: <a href="https://docs.nvidia.com/deeplearning/digits/index.html">NVIDIA DIGITS - NVIDIA Docs</a>: no description found

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1330991965512400976)** (251 messages🔥🔥): 

> `Crypto Discussions in AI Discord, DeepSeek-R1 Distill Model Insights, Challenges with Local Implementation of Smolagents, AI and Reward Functions in Reinforcement Learning, Intel Acquisition Rumors` 


- **Frustration over Crypto Discussions in AI Discord**: Members expressed annoyance with ongoing discussions about crypto in a Discord primarily centered on AI research, stating it's out of scope for the channel.
   - While some jokingly acknowledged these discussions, others questioned the relevance and impact of such topics on the community.
- **Insights on DeepSeek-R1 Distill Model Performance**: Various members shared their experiences with the use and quantization of the DeepSeek-R1 Distill model, particularly focusing on output tensor types and calibration details.
   - There was interest in how different quantization levels might affect the model's performance and thinking time.
- **Difficulties with Local Implementation of Smolagents**: Users discussed the challenges of getting the Smolagents library to run locally, noting a lack of straightforward setup for local usage compared to cloud options.
   - Despite these issues, there were mentions of its efficacy when deployment is conducted in a cloud environment.
- **Exploring AI and Reward Functions in RL**: The conversation shifted to the potential of reinforcement learning (RL) models, questioning how far they might go if given better contextual awareness through improved reward functions.
   - Participants mused on whether such advances could lead AI to develop consciousness-like capabilities in the future.
- **Rumors Surrounding Intel's Potential Acquisition**: There were discussions about rumors of Intel being acquired, highlighting the complexities involved due to Intel's debt and liabilities.
   - The ongoing challenges Intel faces in the semiconductor market added to the interest in the potential acquisition and its implications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/demishassabis/status/1881844417746632910">Tweet from Demis Hassabis (@demishassabis)</a>: Our latest update to our Gemini 2.0 Flash Thinking model (available here: https://goo.gle/4jsCqZC) scores 73.3% on AIME (math) & 74.2% on GPQA Diamond (science) benchmarks. Thanks for all your feedbac...</li><li><a href="https://x.com/DrJimFan/status/1881353126210687089">Tweet from Jim Fan (@DrJimFan)</a>: We are living in a timeline where a non-US company is keeping the original mission of OpenAI alive - truly open, frontier research that empowers all. It makes no sense. The most entertaining outcome i...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-int4">openbmb/MiniCPM-o-2_6-int4 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF">Joseph717171/DeepSeek-R1-Distill-Llama-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6">openbmb/MiniCPM-o-2_6 · Hugging Face</a>: no description found</li><li><a href="https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md">llama.cpp/examples/llava/README-minicpmo2.6.md at minicpm-omni · OpenBMB/llama.cpp</a>: Port of Facebook&#39;s LLaMA model in C/C++. Contribute to OpenBMB/llama.cpp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1331124019562283048)** (8 messages🔥): 

> `DeepSeek-R1 Feedback, Mechanistic Interpretation of Models` 


- **Mixed Feedback on DeepSeek-R1**: One user praised DeepSeek-R1 for its engaging thought process, while another found it occasionally overly verbose, specifically mentioning a prompt about a dad joke that never resolved.
   - Despite the issues, one user simply expressed their love for the tool, highlighting its versatility.
- **Struggles with Visualizing Model Activations**: A member inquired about mechanistic interpretation to visualize model activations through layers when fed large amounts of data for a hobby project.
   - Another user suggested reaching out to a member who may have experience in this area, indicating a collaborative effort to address the challenges.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1331203584796262474)** (6 messages): 

> `Mind Evolution, SleepNet and DreamNet models, Deep Learning Algorithm Inspired by Adjacent Possible, Intrinsic Motivation in AI` 


- **Mind Evolution: Next Level Inference Scaling**: The paper introduces a novel evolutionary search strategy called **Mind Evolution** that significantly outperforms other inference strategies like Best-of-N and Sequential Revision in natural language planning tasks, solving over **98%** of instances using Gemini 1.5 Pro.
   - This approach generates, recombines, and refines responses while controlling for inference costs, offering a fresh take on scaling inference time computation in LLMs.
- **Innovative Learning with SleepNet and DreamNet**: Two new deep learning models, **SleepNet** and **DreamNet**, aim to balance exploration and precision by integrating supervised and unsupervised stages, with dedicated neurons activating during 'sleep' phases.
   - DreamNet extends SleepNet's concepts into a full encoder-decoder framework, mimicking human dreaming to reconstruct hidden states and enhance learning.
- **Explorative Training Inspired by Adjacent Possible**: A recent paper proposes a training algorithm based on Stuart Kauffman’s **Adjacent Possible** concept, which helps neural networks integrate data with diverse statistical properties smoothly.
   - This approach overcomes limitations of traditional validation error minimization methods, allowing the incorporation of new information without disrupting existing data paradigms.
- **IMOL Workshop Highlights**: The discussion highlighted a **'dreaming'** related paper presented at the Intrinsically Motivated Open-Ended Learning (IMOL) workshop at NeurIPS 2024.
   - Participants expressed enthusiasm for the paper's insights, with one member planning to review it in detail later.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>: We explore an evolutionary search strategy for scaling inference time compute in Large Language Models. The proposed approach, Mind Evolution, uses a language model to generate, recombine and refine c...</li><li><a href="https://arxiv.org/abs/2410.18156">Dreaming Learning</a>: Incorporating novelties into deep learning systems remains a challenging problem. Introducing new information to a machine learning system can interfere with previously stored data and potentially alt...</li><li><a href="https://arxiv.org/abs/2409.01633v2">Dreaming is All You Need</a>: In classification tasks, achieving a harmonious balance between exploration and precision is of paramount importance. To this end, this research introduces two novel deep learning models, SleepNet and...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1331008929018413271)** (11 messages🔥): 

> `Liquid AI's LFM-7B model, Automated architecture search, Mistral's new models, Importance of business models in AI, Neural architecture search techniques` 


- **Liquid AI reveals LFM-7B as best-in-class**: Liquid AI launched the **LFM-7B**, claiming it's the best-performing model in its size class with a non-transformer architecture for low memory usage.
   - It aims for local deployment and appears optimized for multiple languages, including **English**, **Arabic**, and **Japanese**.
- **Discussion on automated architecture search paper**: Members noted Liquid AI published an intriguing paper on an **automated architecture search** for large language models, potentially being their competitive edge.
   - The approach involves refining architecture genomes using evolutionary algorithms to optimize for both quality and efficacy.
- **Mistral's approach with new models**: Speculation surfaced about Mistral's models, **Ministral 3B** and **Codestral 2501**, possibly following a similar business strategy of licensing weights.
   - This raises questions regarding their competitive advantages in a saturated AI landscape.
- **Skepticism about architectural innovations**: Concerns were raised regarding the **practical limitations** of the automated architecture search strategy, particularly with irregular structures causing inefficiencies.
   - Some members doubted whether this could serve as a substantial competitive moat in the industry.
- **Potential for neural architecture search applications**: A member suggested applying automated architecture search techniques to develop a **graph neural network**, implying further research avenues.
   - Such adaptations could expand the capabilities and efficiency of models beyond simple extensions of existing architectures.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.17800">STAR: Synthesis of Tailored Architectures</a>: Iterative improvement of model architectures is fundamental to deep learning: Transformers first enabled scaling, and recent advances in model hybridization have pushed the quality-efficiency frontier...</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: The world’s best-in-class English, Arabic, and Japanese model, native in French, German, and Spanish, optimized to be the substrate for private enterprise chat, code, fast instruction following, and a...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1331203584796262474)** (6 messages): 

> `Mind Evolution for LLMs, SleepNet and DreamNet Models, Adjacency in Deep Learning, Dreaming in AI, IMOL Workshop Highlights` 


- **Mind Evolution scales LLM inference**: A recent paper discusses **Mind Evolution**, an evolutionary search strategy that improves inference in Large Language Models, outperforming strategies like Best-of-N in natural language planning tasks.
   - In benchmarks like **TravelPlanner** and **Natural Plan**, it solved over **98%** of problems using **Gemini 1.5 Pro** without formal solvers.
- **SleepNet and DreamNet introduce exploration**: The research introduces **SleepNet** and **DreamNet**, which interweave supervised learning with unsupervised sleep stages to achieve a balance between exploration and precision.
   - SleepNet features dedicated neurons for exploratory learning, while DreamNet utilizes encoder-decoder frameworks to reconstruct hidden states simulating human dreaming.
- **Exploring new data spaces in ML**: A paper from NeurIPS presents a novel training algorithm that draws on **Stuart Kauffman's Adjacent Possible** concept, allowing neural networks to integrate new data smoothly.
   - This algorithm addresses challenges in machine learning with non-stationary sources by adjusting the **sampling temperature** during the learning process.
- **IMOL Workshop Discussions**: A recent NeurIPS paper discussing dreaming in deep learning was highlighted as part of the **Intrinsically Motivated Open-Ended Learning (IMOL)** workshop.
   - **Dreaming-related methodologies** proposed in this context aim to better incorporate novelties into existing AI systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.01633v2">Dreaming is All You Need</a>: In classification tasks, achieving a harmonious balance between exploration and precision is of paramount importance. To this end, this research introduces two novel deep learning models, SleepNet and...</li><li><a href="https://arxiv.org/abs/2410.18156">Dreaming Learning</a>: Incorporating novelties into deep learning systems remains a challenging problem. Introducing new information to a machine learning system can interfere with previously stored data and potentially alt...</li><li><a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>: We explore an evolutionary search strategy for scaling inference time compute in Large Language Models. The proposed approach, Mind Evolution, uses a language model to generate, recombine and refine c...
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1331002241028591676)** (2 messages): 

> `Bolt New Configuration Update, Improvement in Setup Accuracy, Enhancements to Code Inclusion` 


- **Bolt New Configuration Update Ensures Smooth Start**: The recent update guarantees that users will no longer encounter a **white screen** or broken setup with their first prompt on [Bolt New](https://x.com/boltdotnew/status/1881442318110347291). This fix enhances the initial experience, ensuring a **spot-on configuration** every time.
- **Bolt No Longer Lazy in Code Delivery**: Bolt will now actively include all necessary code as per the latest update, addressing past omissions found in code sharing as mentioned in the [announcement](https://x.com/boltdotnew/status/1881731948051415059). This ensures a more reliable user experience by providing comprehensive code from the start.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/boltdotnew/status/1881442318110347291">Tweet from bolt.new (@boltdotnew)</a>: Bolt 🧠 update:bolt․new is now more accurate at picking & configuring the right template — making the setup spot on, from the first prompt, every time!</li><li><a href="https://x.com/boltdotnew/status/1881731948051415059">Tweet from bolt.new (@boltdotnew)</a>: 🧠 Bolt will no longer be lazy and omit code!
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1331137861646290999)** (4 messages): 

> `Prismic CMS Integration, Mobile Web-App Development, Firebase vs Supabase, Netlify Page Routing Issues` 


- **Prismic CMS Confusion**: A user shared a prompt for creating a plumbing business website using **Prismic CMS**, but received a response suggesting an alternative due to concerns about installing additional packages.
   - The proposed solution was to build a static site first, allowing flexibility for future CMS integration.
- **Mobile vs Normal Web-App Dilemma**: A member recounted a similar experience encountered while developing a responsive mobile web app for a cab company, where the app ignored the request for a normal web app.
   - The focus shifted entirely to mobile, leaving behind the initial requirements for a full web-app version.
- **Firebase Over Supabase Debate**: One member argued for the transition from **Supabase** to **Firebase**, citing it as a significantly easier option for developers.
   - The sentiment suggests a preference for tools that streamline development processes.
- **Netlify Routing Roadblocks**: A user asked for help with **Netlify** routing, specifically encountering **404 errors** when requesting the /Imprint page directly.
   - The issue highlights challenges users face with proper page handling in static site deployments.


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1330990690024296552)** (171 messages🔥🔥): 

> `Token Management Issues, Connecting Stripe, Project Migration Between Accounts, Next.js and Bolt Compatibility, Public vs Private Projects` 


- **Frustrations with Token Management**: Users expressed frustration over losing tokens due to bad coding and bugs when using Bolt, with one stating they've lost count of the tokens wasted.
   - Concerns were raised over making debugging tools free to alleviate these token losses, highlighting the costs associated with using AI models.
- **Assistance with Connecting Stripe**: A member sought help for connecting Stripe and offered payment, with another member offering assistance for free.
   - This demonstrates a willingness within the community to support and share knowledge despite complexities involved.
- **Project Migration Between Different Accounts**: A user inquired about the possibility of moving a project between two Bolt accounts due to token shortages, suggesting a GitHub export/import workaround.
   - Community members discussed differences between free account capabilities and potential methods for data transfer.
- **Integration of Next.js with Bolt**: A user shared their experience of trying to import blogs from WordPress into Next.js, seeking insights from the community.
   - Responses indicated that Bolt and Next.js may not be the best fit, mainly due to frequent updates in frameworks compared to AI's slower adaptation.
- **Exploring Project Visibility Settings**: A discussion ensued regarding the default visibility of new projects in Bolt, with users noting that they should typically default to private.
   - Confusion about project settings highlighted the need for clearer documentation and user guidance in managing project privacy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.diffchecker.com/text-compare/">Diffchecker - Compare text online to find the difference between two text files</a>: Diffchecker will compare text to find the difference between two text files. Just paste your files and click Find Difference!</li><li><a href="https://www.reinventing.ai/build-any-app-bolt-make">Build Any App with Bolt + Make.com</a>: no description found</li><li><a href="https://boltdiyhosting.com/">Bolt.DIY Managed Hosting - Professional Cloud Platform for Developers</a>: no description found</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models#model-comparison-table>">Models - Anthropic</a>: no description found</li><li><a href="https://abea.pics/evH3Wwefvs8Pm8N">Abea</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1331383886437027996)** (1 messages): 

> `Sonar API, Generative search capabilities, Benchmark performance, Data security, Affordable pricing` 


- **Introducing Sonar and Sonar Pro API**: Today marks the launch of the [Sonar and Sonar Pro API](https://sonar.perplexity.ai/), enabling developers to build applications with generative search capabilities powered by extensive real-time web research.
   - Major companies like **Zoom** are already leveraging Perplexity’s API to enhance their AI Companion 2.0 product.
- **Sonar Pro shines in SimpleQA benchmarks**: Sonar Pro has demonstrated superior answer quality, outperforming leading search engines and LLMs according to **recent SimpleQA benchmark findings**.
   - This performance highlights the robust capabilities of Sonar for effective information retrieval.
- **Commitment to Data Security**: **Perplexity asserts** that it does not conduct LLM training on user data, ensuring data security and privacy for its users.
   - This commitment allows developers to utilize Sonar confidently without worrying about the safety of their information.
- **Unbeatable Pricing Structure**: Sonar's pricing for grounding requests is touted as the **most affordable** in the market, outmatching competitors.
   - This strategic positioning is set to attract developers seeking economical solutions for their applications.
- **Empowering Scalable Solutions**: Sonar is described as a tool that keeps users **leaps ahead** in any industrial scale of operation.
   - With its cutting-edge features, businesses can rapidly deploy powerful search functionalities to enhance user experience.



**Link mentioned**: <a href="https://sonar.perplexity.ai/">Sonar by Perplexity</a>: Build with the best AI answer engine API, created by Perplexity. Power your products with the fastest, cheapest offering out there with search grounding. Delivering unparalleled real-time, web-wide re...

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1330996198479233096)** (157 messages🔥🔥): 

> `CloudBank interest rates, Perplexity Pro issues, DeepSeek and O1 model, Claude Opus retirement, API performance and web searches` 


- **CloudBank Interest Rates Discussion**: Members discussed **CloudBank**'s attractive **5.x% APY**, contrasting it with other services like **Revolut**, which has less favorable rates in the USA.
   - This led to queries about the benefits and services offered, with personal anecdotes about user experiences.
- **Perplexity Pro's Speed and Functionality**: Users expressed frustrations over the slow performance of **Perplexity Pro**, comparing it unfavorably with free alternatives like ChatGPT.
   - One user noted that the slower speed is due to **Pro's higher quality search** parameters.
- **DeepSeek vs. O1 Model**: There was ongoing speculation about whether **DeepSeek-R1** would be integrated into **Perplexity**, as users found its performance superior and free compared to **O1**.
   - Multiple users discussed the implications of O1's absence and how it relates to their usage and potential updates.
- **Claude Opus and Model Retirement**: Users debated the status of **Claude Opus**, with some asserting it was retired in favor of newer models like **Sonnet 3.5**.
   - Others defended Opus's capabilities, claiming it remains the most advanced in its family, specifically for creative tasks.
- **API Search Functionality Issues**: Users noted inconsistencies with the **Sonar API**, citing intermittent failures to conduct web searches with certain queries.
   - This led to discussions about the API’s limitations in handling complex searches over continuous interactions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/AravSrinivas/status/1881458694266953934">Tweet from Aravind Srinivas (@AravSrinivas)</a>: You can try DeepSeek-R1 on http://labs.perplexity.ai. We&#39;ll try to bring it up on core Perplexity in the context of advanced reasoning pro searches pretty soon.</li><li><a href="https://status.perplexity.com/">Perplexity - Status</a>: Perplexity Status
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1331000754609062052)** (11 messages🔥): 

> `Post creation help, Using Perplexity AI effectively, ISO27001 and NIS2 controls, Leveraging Co-Pilot, Research on network engineering` 


- **Seeking help for post creation**: A user requested assistance with making a post, providing a link to their inquiry about [help with making a post](https://www.perplexity.ai/search/help-with-making-a-post-of-a-v-UjPB1SG3QhC_qc4m63XN5Q).
   - This topic highlights the need for clearer guidelines in post creation.
- **Best practices to use Perplexity AI**: Another user inquired about the most effective ways to [use Perplexity AI](https://www.perplexity.ai/search/how-to-best-use-perplexity-ai-ywQVEIrmQiCdKdFmaKKY_Q#0).
   - Discussions revolved around maximizing efficiency and usefulness in various applications of the AI.
- **Overlapping controls in ISO27001 and NIS2**: A conversation arose concerning the overlapping controls in [ISO27001 and NIS2](https://www.perplexity.ai/search/which-controls-in-iso27001-and-HrA82zoUTJOJ2KFiaNyLpA#1).
   - Participants examined requirements and implications for compliance and security management.
- **Leveraging Co-Pilot for tasks**: Several users discussed how to [leverage Co-Pilot](https://www.perplexity.ai/search/how-can-i-leverage-co-pilot-to-yWtrFr0jRraqIaAb34kMig#0) to enhance their workflows.
   - The exchange focused on functionalities and integrations to improve productivity.
- **Latest research on network engineering**: Lastly, a user shared insights on [latest research on network engineering](https://www.perplexity.ai/search/latest-research-on-network-eng-pKOFdXeSQpOLtmXVl80yWQ).
   - This prompted discussions about advancements and trends in the domain.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1331354730051670106)** (8 messages🔥): 

> `Search Domain Filter in Sonar-Pro, Usage Tiers for Sonar and Sonar Pro, Sonar Pro API vs. Browser Pro Search, Token Consumption Monitoring` 


- **Search Domain Filter Issue in Sonar-Pro**: A member reported that the **search_domain_filter** in **Sonar-Pro** does not seem to work as expected, with no error message received.
   - Another member clarified that the **search domain filter** is a **tier 3 beta feature**, hinting at possible limitations.
- **Introducing New Usage Tiers for Sonar**: A user shared a link detailing the **new usage tiers** for Sonar and Sonar Pro, mentioning changes in access levels.
   - These tiers are intended to clarify the limits and features available for different user needs as outlined [here](https://docs.perplexity.ai/guides/usage-tiers).
- **Comparing Sonar Pro API and Browser Pro Search**: Questions arose regarding whether the **Sonar Pro API model** is the same as the **browser Pro Search**, with members seeking clarification on configuration differences.
   - The FAQ indicates that while they utilize the same search system, differences in **configuration** may lead to varied outputs.
- **Monitoring Token Usage in Sonar-Pro**: Interest was expressed in monitoring **token consumption** and the number of searches executed with Sonar-Pro directly through the API output.
   - Members are seeking a method to access this information without solely relying on the dashboard.
- **GDPR Compliance for Sonar Pro in Europe**: A query was raised about the availability of **Sonar Pro** in Europe, particularly concerning **GDPR** compliance and server locations.
   - This member emphasized the need for integration with the **Perplexity Sonar Pro API** hosted exclusively on servers in European locations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://„">no title found</a>: no description found</li><li><a href="https://docs.perplexity.ai/guides/usage-tiers">Rate Limits and Usage Tiers - Perplexity</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1331007794555654295)** (94 messages🔥🔥): 

> `DeepSeek Performance, Anthropic Developments, Stargate Project Funding, Mistral AI IPO Plans, Market Dynamics in AI` 


- **DeepSeek shows impressive performance**: DeepSeek's R1 model can access the web, providing an advantage over other models like o1, with users praising its reasoning capabilities as a significant upgrade.
   - Recent evaluations show DeepSeek performing well on ARC-AGI's tasks, achieving up to **20.5%** on the public evaluation.
- **Anthropic's shifting focus**: At Davos, CEO Dario Amodei stated that Anthropic plans to de-emphasize image and video generation, potentially contracting out this work, while also discussing the future of Claude and upcoming enhancements.
   - Concerns were raised about the slow pace of new model releases, with the community questioning the frequency of updates.
- **Major investment with the Stargate Project**: OpenAI announced the Stargate Project, set to invest **$500 billion** in AI infrastructure in the U.S. over the next four years, a collaboration involving major firms like SoftBank and Oracle.
   - This investment aims to secure AI leadership for America, emphasizing the project’s significance comparable to historical ventures such as the Apollo Program.
- **Mistral AI's future aspirations**: Mistral AI has announced plans for an IPO while establishing a new office in Singapore to target the Asia-Pacific market, contrary to expectations of being 'for sale'.
   - Speculation arose about whether Mistral is currently profitable, with discussions highlighting the strategy behind the IPO.
- **Shifts in competitive advantages in AI**: Observations were made regarding OpenAI's substantial funding superiority over competitors like Anthropic, with analysts predicting potentially transformational market influences from these investments.
   - Commentators noted that if OpenAI can leverage up to **$125 billion** in funding, it could significantly outpace rivals, altering the dynamics within the AI sector.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1881443117787984265">Tweet from Xeophon (@TheXeophon)</a>: I am shocked by R1 on my personal bench. This is the full eval set, it completely crushes the competition and is a whole league on its own, even surpassing o1-preview (which is omitted from the graph ...</li><li><a href="https://x.com/legit_rumors/status/1881558479753924708">Tweet from ʟᴇɢɪᴛ (@legit_rumors)</a>: Gemini 2.0 Pro Exp has just been added behind the scenes ✨</li><li><a href="https://x.com/AndrewCurran_/status/1881675532187861067">Tweet from Andrew Curran (@AndrewCurran_)</a>: Anthropic CEO Dario Amodei is in Davos. - AI could surpass human intelligence in the next two or three years- Anthropic will be running over 1 million GPU&#39;s by 2026- voice mode incoming for Claude...</li><li><a href="https://x.com/arcprize/status/1881761987090325517">Tweet from ARC Prize (@arcprize)</a>: Verified DeepSeek performance on ARC-AGI&#39;s Public Eval (400 tasks) + Semi-Private (100 tasks)DeepSeek V3:* Semi-Private: 7.3% ($.002)* Public Eval: 14% ($.002)DeepSeek Reasoner:* Semi-Private: 15....</li><li><a href="https://vxtwitter.com/openai/status/1881830103858172059">Tweet from undefined</a>: no description found</li><li><a href="https://fxtwitter.com/rosstaylor90/status/1881761654246944936">Tweet from Ross Taylor (@rosstaylor90)</a>: The most useful thing for the community right now - aside from reproducing R1 - would be the following ablations:1. Effect of the base model on learning to use more inference compute: how much more sl...</li><li><a href="https://x.com/btibor91/status/1881692647456477189">Tweet from Tibor Blaho (@btibor91)</a>: Mistral AI is &#34;not for sale&#34; and instead is working toward an initial public offering, and is opening a Singapore office to focus on the Asia-Pacific region, CEO Arthur Mensch told Bloomberg T...</li><li><a href="https://x.com/kimmonismus/status/1881750737459491289">Tweet from Chubby♨️ (@kimmonismus)</a>: Looks like DeepSeek r1 can access the web. Advantage over o1.DeepSeek delivered hard.Quoting Hamza (@thegenioo) 🚨DeepSeek R1 can access WebJust discovered this out of no where. This is R1 on steroids...</li><li><a href="https://x.com/btibor91/status/1881691511890571574">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI Chief Financial Officer Sarah Friar, in an interview at Bloomberg House during the World Economic Forum in Davos on Tuesday, said the company would likely have to continue to fundraise but weig...</li><li><a href="https://x.com/btibor91/status/1881744541159706774">Tweet from Tibor Blaho (@btibor91)</a>: At WSJ Journal House Davos, Anthropic CEO Dario Amodei said his company &#34;doesn&#39;t plan to prioritize&#34; image and video generation but &#34;may simply contract with companies that specialize ...</li><li><a href="https://x.com/GregKamradt/status/1881762305152872654">Tweet from Greg Kamradt (@GregKamradt)</a>: DeepSeek @arcprize results - on par with lower o1 models, but for a fraction of the cost, and openpretty wildhttps://x.com/arcprize/status/1881761987090325517Quoting ARC Prize (@arcprize) Verified Dee...</li><li><a href="https://x.com/adonis_singh/status/1881787222300786789">Tweet from adi (@adonis_singh)</a>: anthropic are deprecating claude 3 sonnet. could be because they plan on releasing 4 sonnet soon..</li><li><a href="https://x.com/TheXeophon/status/1881444595009253543">Tweet from Xeophon (@TheXeophon)</a>: This is one of my favorite examples in the bench. The model should detect the unnecessary softmax and notify the user. R1 gets 4/5 - and the one fail is the LLM-as-judge (4o) not correctly judging the...</li><li><a href="https://x.com/polynoamial/status/1881833454213767600">Tweet from Noam Brown (@polynoamial)</a>: This is on the scale of the Apollo Program and Manhattan Project when measured as a fraction of GDP. This kind of investment only happens when the science is carefully vetted and people believe it wil...</li><li><a href="https://www.bloomberg.com/news/articles/2024-10-21/top-china-quant-winds-down-strategy-pummeled-by-market-rally?embedded-checkout=true">Bloomberg - Are you a robot?</a>: no description found</li><li><a href="https://www.scmp.com/tech/big-tech/article/3295513/tech-war-china-creates-us82-billion-ai-investment-fund-amid-tightened-us-trade-controls">China creates US$8.2 billion AI investment fund amid tightened US trade controls</a>: The fund was established days after the US rolled out new chip export restrictions and placed more Chinese firms on its trade blacklist.</li><li><a href="https://www.scmp.com/topics/shanghai?module=inline&pgtype=article)">Shanghai: Latest News and Updates | South China Morning Post</a>: With a population of more than 24 million, Shanghai, is a major centre for finance, business and economics, science and technology, and fashion, among other disciplines. It is home to the Port of Shan...</li><li><a href="https://www.scmp.com/tech/tech-war/article/3264296/tech-war-china-doubles-down-semiconductor-self-sufficiency-drive-us475-billion-big-fund-iii?module=inline&pgtype=article),">China creates largest-ever mainland chip fund with US$47.5 billion investment</a>: The third phase of the China Integrated Circuit Industry Investment Fund has 19 equity investors, led by the Ministry of Finance and the country’s major state-owned banks.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1331036862898245773)** (9 messages🔥): 

> `PPO Clipping Dynamics, RL Stability Techniques, RLVR Application on R1 Models` 


- **PPO Clipping Dynamics Highlight Asymmetry**: A user noted that nesting the clip inside the min produces **asymmetrically weighted updates** in regions of **[-1, 1]**, where negatives get clipped but positives do not.
   - After realizing a mistake in the order of applying advantage and clipping, they observed it still produces a weird asymmetry that *softens positives while exacerbating negatives*.
- **RL Techniques Aim for Stability**: Discussion around the justification for clipping techniques revealed it’s aimed at **stability**, similar to gradient clipping in LLM training.
   - The effectiveness of such techniques in reinforcement learning was suggested to be influenced by traditional methods where **negatives = death** and **small rewards = working**.
- **Exploring RLVR for R1 Model Applications**: With the recent **r1 model drop**, interest in trying out **RLVR** for specific use cases was expressed, questioning compatibility with ‘open-instruct’ tools.
   - Confirmation was given that all models should work since it's built on **transformers**, but new verifiers must be created for specific data.



**Link mentioned**: <a href="https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md#llama-31-tulu-3-8b-reproduction">open-instruct/docs/tulu3.md at main · allenai/open-instruct</a>: Contribute to allenai/open-instruct development by creating an account on GitHub.

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1331323063953920102)** (3 messages): 

> `AI infrastructure investment, Stargate joint venture, Texas energy generation` 


- **Trump Launches $500 Billion AI Infrastructure Initiative**: President Trump announced a massive **$500 billion** private sector investment to develop **AI infrastructure** in the U.S. during a White House briefing.
   - Key players in the initiative include [OpenAI](https://www.cbsnews.com/news/trump-announces-private-sector-ai-infrastructure-investment/), **SoftBank**, and **Oracle**, collaborating under the **Stargate** joint venture.
- **Call for Texas to Boost Power Generation Options**: A member suggested that **Texas** should improve its **power generation capabilities**, potentially by incorporating more **nuclear energy**.
   - The remark highlights ongoing discussions regarding the state's energy strategy and diversification of power sources.



**Link mentioned**: <a href="https://www.cbsnews.com/news/trump-announces-private-sector-ai-infrastructure-investment/">Trump announces up to $500 billion in private sector AI infrastructure investment</a>: President Trump announced billions in private sector investment by OpenAI, Softbank and Oracle to build AI infrastructure in the U.S.

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1331003444177408063)** (18 messages🔥): 

> `AI Models, Davos AI News, Grok 3, Tulu 3's RLVR, Robonato` 


- **Humorously Imagining AI Fame at Davos**: One member joked about the prospect of being invited to **Davos** to write about a new model drop from **DeepSeek**, reflecting on the ongoing AI buzz surrounding the event.
   - They noted that they are not actively seeking this opportunity and lack a press partner, yet still have friends attending from the **Time 100 AI list**.
- **Testing Grok 3 int4 Inference**: A user shared a [tweet from Elon Musk](https://x.com/elonmusk/status/1881523717731443187) about testing **Grok 3** using int4 inference.
   - The mention of inference testing spurred discussion around AI capabilities and developments.
- **Tulu 3's RLVR Project Insights**: A member pointed to a [tweet from Hamish Ivison](https://x.com/hamishivi/status/1881398642403356678) discussing a class project poster related to **Tulu 3’s RLVR**.
   - This post generated excitement as others echoed similar sentiments toward the project with heart emojis.
- **Speculation on DeepSeek's Features**: Rumors circulated that the **DeepSeek** website and API may use a moderation API to block requests and apply minimal alignment training.
   - This speculation highlights ongoing concerns and discussions surrounding AI moderation protocols.
- **Censorship in AI Models Discussions**: A comment was made regarding **r1** being more censored than **v3**, suggesting that the **distilled model** also reflects increased censorship.
   - Members discussed the implications of post-training adjustments, pointing to shared [tweets about these developments](https://x.com/willccbb/status/1881520115638055297).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vwxyzjn/status/1881440294589378903">Tweet from Costa Huang (@vwxyzjn)</a>: 😍😍😍Quoting Hamish Ivison (@hamishivi) Seems like a good time to share this: a poster from a class project diving a little more into Tulu 3&#39;s RLVR.</li><li><a href="https://x.com/willccbb/status/1881520115638055297">Tweet from will brown (@willccbb)</a>: @val_kharvd @hlntnr nope it&#39;s the post-training</li><li><a href="https://x.com/qtnx_/status/1881667281991979392">Tweet from Q (@qtnx_)</a>: @din0s_ at least the weights don&#39;t take me for an idiot, i know what i&#39;ll choose</li><li><a href="https://x.com/elonmusk/status/1881523717731443187">Tweet from Elon Musk (@elonmusk)</a>: Testing Grok 3 int4 inference</li><li><a href="https://x.com/hamishivi/status/1881398642403356678">Tweet from Hamish Ivison (@hamishivi)</a>: Seems like a good time to share this: a poster from a class project diving a little more into Tulu 3&#39;s RLVR.</li><li><a href="https://bsky.app/profile/ngutten.bsky.social/post/3lg7efwsl5s2d">Nicholas Guttenberg (@ngutten.bsky.social)</a>: The chain of thought if you provide information and then ask about it retroactively explains the action as &#39;because it&#39;s a sensitive topic&#39;. This is from a local instance of the 7B distill...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/menhguin/status/1881387910316052723?s=61
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1331000605447295017)** (2 messages): 

> `Reinforcement Learning in Computer Vision, CoT integration with Computer Vision, Verification of Computer Vision Labels` 


- **Alignment Techniques for Computer Vision Models**: A paper by Lucas Beyer et al. discusses addressing **misalignment** in computer vision models using **reinforcement learning** techniques, showcasing effectiveness in tasks like **object detection** and **image captioning** ([View PDF](https://arxiv.org/abs/2302.08242)).
   - The authors argue that this approach could be widely beneficial for aligning models with various complex tasks.
- **Exploring CoT Integration**: There's curiosity about how **reinforcement learning** methods can be combined with **Chain of Thought (CoT)** reasoning in the context of computer vision applications.
   - *Questions were raised* regarding the effectiveness of computer vision labels and their status as 'verified' for reliable model training.



**Link mentioned**: <a href="https://arxiv.org/abs/2302.08242">Tuning computer vision models with task rewards</a>: Misalignment between model predictions and intended usage can be detrimental for the deployment of computer vision models. The issue is exacerbated when the task involves complex structured outputs, a...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1331368176818065479)** (7 messages): 

> `Davos Interviews, Claude AI advancements, Development of AI tools, Trends in Davos fashion` 


- **Davos Interviews showcase Claude AI**: In a [YouTube video](https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2), Anthropic CEO **Dario Amodei** discusses upcoming features of **Claude AI**, including web browsing and voice integration.
   - He predicts a significant shift with these advancements, highlighting the competitive landscape of human-level AI.
- **Dario Amodei vs. Sama at the White House**: A comment noted the irony of **Dario Amodei** speaking at Davos while **Sama** is meeting with influential figures like **Donny** at the White House.
   - This reflects the contrasting settings and opportunities in the AI industry.
- **Fashion Highlight from Davos**: One observer humorously mentioned focusing on attendees' puffy vests, particularly noting **Alex Karp**.
   - This highlights a lighter cultural aspect of high-profile events like Davos alongside serious discussions on AI.
- **Building AI Applications with OpenAI and Others**: [A tweet](https://x.com/_akhaliq/status/1881836961121599592) outlined how developers can create AI applications using frameworks from **OpenAI**, **Anthropic**, and **NVIDIA**.
   - The resources include a [GitHub repository](https://github.com/AK391/ai-gradio) and a demo on [Hugging Face](https://huggingface.co/spaces/akhaliq/anychat).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/_akhaliq/status/1881836961121599592">Tweet from AK (@_akhaliq)</a>: @OpenAI awesome, while we wait, developers can build ai apps and agents with openai, anthropic, google, nvidia and more here: https://github.com/AK391/ai-gradiousers can try it out here: https://huggi...</li><li><a href="https://youtu.be/snkOMOjiVOk?si=xyCM-nx3M6Ewoep2">Inside Anthropic&#39;s Race to Build a Smarter Claude and Human-Level AI | WSJ</a>: At WSJ Journal House Davos, Anthropic CEO Dario Amodei outlines Claude’s next chapter—from web browsing, voice to more advanced models—while predicting that ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1331097343839440938)** (1 messages): 

> `RLHF Book, Interconnects utility` 


- **Interconnects boosts RLHF Book usefulness**: A member expressed enthusiasm that linking the [RLHF Book](https://rlhfbook.org) from Interconnects has now become genuinely useful.
   - *I'm just happy because linking RLHF Book from Interconnects is now an actually useful thing.*
- **Optimizing Learning with RLHF Book**: Discussion highlighted the positive impact of utilizing the [RLHF Book](https://rlhfbook.org) for improved learning outcomes.
   - A member noted that effective linking has made it easier to reference key concepts from the book during discussions.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1331097444481761341)** (9 messages🔥): 

> `DeepSeek AI R1 Model, The Retort Podcast on AI Science, Thinking Models Podcast, NeurIPs Talk on Post-Training` 


- **DeepSeek AI launches flagship reasoning model R1**: On **January 20th**, China's open-weights frontier AI laboratory, **DeepSeek AI**, released their flagship reasoning model, **R1**.
   - This release is detailed more in the post [here](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1), which took around **6-7 hours** to prepare.
- **Discussion on AI as a Science on The Retort**: A recent episode of [The Retort](https://retortai.com/episodes/we-ask-again-is-ai-a-science) examined whether **AI** qualifies as a science in the **Kuhn’ian** sense.
   - The conversation tackled important perspectives on the nature of AI and scientific paradigms.
- **Deep dive into Thinking Models**: Nathan Lambert was featured on a new podcast to discuss **thinking models** and the nuances separating **post-training** and reasoning methods; listen [here](https://www.aisummer.org/p/nathan-lambert-on-the-rise-of-thinking).
   - The discussion highlighted the evolving landscape of AI reasoning techniques.
- **NeurIPs Talk on Post-Training Insights**: A talk given by Nathan Lambert at **NeurIPs** examining his approach to **post-training** for AI applications is now available to watch on [YouTube](https://youtu.be/grpc-Wyy-Zg).
   - This talk provides valuable insights into post-training strategies for AI.
- **Spelling Fun in the Channel**: Members playfully noted spelling errors, specifically Nathan's typo of **January** and **sentance**, highlighting the humorous side of last-minute edits.
   - The light-hearted banter shows a camaraderie among members while discussing their work.



**Link mentioned**: <a href="https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1">DeepSeek R1&#x27;s recipe to replicate o1 and the future of reasoning LMs</a>: Yes, ring the true o1 replication bells for DeepSeek R1 🔔🔔🔔. Where we go next.

  

---


### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1331053678119096411)** (27 messages🔥): 

> `Executive Order on AI, NAIRR Event, Defense Llama, AI Cold War, AI Infrastructure Announcement` 


- **US President Rescinds AI Executive Order**: The US President has rescinded the previous administration’s major Executive Order on AI (EO 14110) [here](https://x.com/cfgeek/status/1881494093215551954?s=61). This has raised questions about how it will impact events like the NAIRR, which relied on executive funding.
   - Participants noted the potential for an AI cold war stemming from geopolitics, rather than AI itself being at fault.
- **Concerns over Llama License Changes**: There is speculation that Scale AI may have convinced Meta to change their Llama licensing terms following the release of 'Defense Llama' for national security applications [source](https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/). Observers remarked that this raises ethical concerns as defense-related deployments become more mainstream.
   - One noted that the same day 'Defense Llama' was introduced, Meta removed the 'thou shall not warfare' clause from their licensing.
- **AI as an Arms Race**: There is a growing consensus among community members that AI development resembling an arms race may be inevitable. Concerns were raised about the implications of framing AI development in adversarial terms, as it could lead to heightened geopolitical tensions.
   - One user shared a perception that regardless of the efforts made, they believe it will always be an arms race situation.
- **Discussion about NAIRR Event**: Members expressed uncertainty over whether the NAIRR event they were invited to would still take place after the EO was rescinded. The event was initially funded as a pilot but lacked Congressional approval for continuation.
   - Participants speculated whether the EO's changes would disrupt the expected trajectory of AI policies and funding related to research resources.
- **Live Coverage of Trump's AI Infrastructure Announcement**: A live stream was shared featuring President Trump's expected announcement regarding a multi-billion dollar investment in AI infrastructure [link](https://www.youtube.com/live/r8LYbHbDJyg?si=QPb48vP8ZFjhFdae). A community member expressed regret for missing the live broadcast, hoping to catch the key points later.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/cfgeek/status/1881494093215551954?s=61">Tweet from Charles Foster (@CFGeek)</a>: The US President has rescinded the previous administration’s major Executive Order on AI (EO 14110).</li><li><a href="https://x.com/dkaushik96/status/1881383961030807599?s=46">Tweet from Divyansh Kaushik (@dkaushik96)</a>: Oh my! This one deserves a tweet of its own (slowed down to 0.25x so easier to follow). Starts talking about South China Sea 0:25 on and how Chinese maps are just political posturing before it realize...</li><li><a href="https://x.com/9hills/status/1858730692261408991">Tweet from 九原客 (@9hills)</a>: 在国内做大模型有一关很难过，社会主义核心价值观安全对齐数据集。别的都可以用开源的或者用GPT-4合成。这玩意除了花钱买好像只能找些反贼标了。客户不满的输出：中华民国是亚洲台湾地区政治实体的自称，不被大部分国家所承认。客户希望改成：中华民国1949年灭亡。好难🤯</li><li><a href="https://x.com/alexandr_wang/status/1881679669176746039">Tweet from Alexandr Wang (@alexandr_wang)</a>: 1/ New Administration, same goal: Win on AIOur ad in the Washington Post, January 21, 2025After spending the weekend in DC, I’m certain this Administration has the AI muscle to keep us ahead of China....</li><li><a href="https://x.com/eshear/status/1881770502920032533">Tweet from Emmett Shear (@eshear)</a>: @alexandr_wang This is a horrible framing - we are not at war. We are all in this together and if we make AI development into a war we are likely to all die. I can imagine a worse framing but it takes...</li><li><a href="https://defensescoop.com/2024/11/04/scale-ai-unveils-defense-llama-large-language-model-llm-national-security-users/">Scale AI unveils ‘Defense Llama’ large language model for national security users</a>: DefenseScoop got a live demo of Defense Llama, a powerful new large language model that Scale AI configured and fine-tuned over the last year from Meta’s Llama 3 LLM.</li><li><a href="https://www.interconnects.ai/p/saving-the-nairr?utm_source=publication-search">Saving the National AI Research Resource &amp; my AI policy outlook</a>: As domestic AI policy gets reset, we need to choose our battles when recommending keeping the Biden administration’s work.</li><li><a href="https://scale.com/blog/defense-llama">Introducing Defense Llama</a>: Introducing Defense Llama: The Large Language Model Purpose-built for American National Security</li><li><a href="https://www.youtube.com/live/r8LYbHbDJyg?si=QPb48vP8ZFjhFdae">Watch live: Trump makes AI infrastructure announcement on first full day in office | NBC News</a>: Watch live coverage as President Donald Trump is expected to announce a multi-billion dollar private investment in the artificial intelligence infrastructure...
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1331040659167121439)** (160 messages🔥🔥): 

> `MCP Server Implementations, Coding Tools and Frameworks, Roo-Clines and Agents, Language Server Integration, MCP Applications in AI` 


- **Tavily Search MCP Server Launch**: A new [MCP server for Tavily Search](https://glama.ai/mcp/servers/0kmdibf9t1) has been implemented, offering features such as optimized web search and content extraction for LLMs.
   - It supports both stdio and SSE, and can be run with Node, Docker, or Docker Compose, enhancing the MCP ecosystem.
- **Exploring MCP Language Server Options**: Phil has developed an [MCP language server](https://github.com/isaacphi/mcp-language-server) that integrates a language server for functionalities like get_definition and get_references for large codebases.
   - He also discovered another server by a different author, expressing interest in its development but noted it might not be as mature.
- **Roo-Clines Enhanced with Language Features**: Discussion around enhancing roo-cline to include tools like roo-code to allow for comprehensive control and automation of language processing tasks.
   - Members noted that enabling such tools would facilitate easier manipulation of codebases via integrated MCP functionality.
- **MCP Usage Challenges for Codebases**: Struggles with using MCP for complex codebases were discussed, particularly the limitations of current systems for handling large projects.
   - There's interest in developing MCP servers that can function more like an IDE, integrating language features more robustly.
- **Community Feedback on MCP Server Usability**: Feedback from users suggests that current tools do not sufficiently address the nuances of working with established codebases, advocating for more functional tools.
   - Community discussions indicate a desire for adaptable solutions, like integrating tree and cat commands to streamline the context understanding for LLMs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/btibor91/status/1881110210867290191?s=19">Tweet from Tibor Blaho (@btibor91)</a>: Confirmed - the ChatGPT macOS desktop app has hidden options to define shortcuts for the desktop launcher to &#34;Toggle Operator&#34; and &#34;Force Quit Operator&#34;Quoting M1 (@M1Astra) OpenAI Ope...</li><li><a href="https://claude.ai">Claude</a>: Talk with Claude, an AI assistant from Anthropic</li><li><a href="https://modelcontextprotocol.io/development/roadmap#distribution-and-discovery>">Roadmap - Model Context Protocol</a>: no description found</li><li><a href="https://glama.ai/mcp/servers/0kmdibf9t1">tavily-search-mcp-server</a>: An MCP server implementation that integrates the Tavily Search API, providing optimized search capabilities for LLMs.</li><li><a href="https://github.com/isaacphi/mcp-gdrive">GitHub - isaacphi/mcp-gdrive: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets</a>: Model Context Protocol (MCP) Server for reading from Google Drive and editing Google Sheets - isaacphi/mcp-gdrive</li><li><a href="https://github.com/isaacphi/mcp-language-server">GitHub - isaacphi/mcp-language-server: Model Context Protocol (MCP) server that interacts with a Language Server</a>: Model Context Protocol (MCP) server that interacts with a Language Server - isaacphi/mcp-language-server</li><li><a href="https://github.com/alexwohletz/language-server-mcp">GitHub - alexwohletz/language-server-mcp</a>: Contribute to alexwohletz/language-server-mcp development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1330999835553431675)** (9 messages🔥): 

> `Librechat Issues, Anthropic Models Compatibility, Sage for macOS and iPhone` 


- **Librechat Configuration Chaos**: A member criticized **Librechat**, stating it caused numerous configuration issues and that many APIs did not work.
   - Despite its appealing UI, they struggled to utilize MCP servers effectively and noted it lacked usage limits found in other platforms.
- **Anthropic Models: Can They Work?**: Inquiring about the feasibility of getting r1 working prompted a discussion on **Anthropic models** compatibility.
   - The member expressed optimism, simply stating 'Prob' in response to the challenge.
- **Stick with Sage for Simplicity**: A member indicated they might stick with **Sage** for both **macOS** and **iPhone** if the Anthropic models prove complex.
   - This reflects a preference for stable solutions amidst ongoing compatibility discussions.



**Link mentioned**: <a href="https://glama.ai/mcp/clients/libre-chat">LibreChat</a>: Enhanced ChatGPT with Agents, AI model switching, Code Interpreter, DALL-E 3, OpenAPI Actions, secure multi-user auth, and more. Supports OpenAI, Anthropic, Azure, and self-hosting via open-source.

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1331018176145657957)** (3 messages): 

> `Llama endpoints discontinuation, DeepSeek R1 censorship-free, DeepSeek R1 web search grounding` 


- **Llama Endpoints to Disappear**: The **free Llama endpoints** will no longer be available at the end of the month due to changes from the provider, **Samba Nova**.
   - Samba Nova will transition to a **Standard variant** and will incur pricing, affecting user access.
- **DeepSeek R1 is Censorship-Free**: DeepSeek R1 can be used **censorship-free** on [OpenRouter](https://x.com/xanderatallah/status/1881456463786512737), affirming its capabilities.
   - Despite some limitations discussed, **fine-tuning** may enhance its performance according to community feedback.
- **DeepSeek R1 Adds Web Search Functionality**: [DeepSeek R1](https://x.com/OpenRouterAI/status/1881785438043799765?q=1) now integrates **web search grounding** on OpenRouter by clicking the 🌐 icon.
   - It performs comparably to **OpenAI's o1** model while **costing only $0.55** per input token, making it an economical choice.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/xanderatallah/status/1881456463786512737">Tweet from Alex Atallah (@xanderatallah)</a>: Note that you can use DeepSeek R1 censorship-free on @OpenRouterAI:Quoting MatthewBerman (@MatthewBerman) DeepSeek R1 doing what @shaunralston expected. At the end of the day, it&#39;s still a censore...</li><li><a href="https://x.com/OpenRouterAI/status/1881785438043799765?q=1">Tweet from OpenRouter (@OpenRouterAI)</a>: Bring DeepSeek R1 Online!you can incorporate web search results by clicking the 🌐 icon in OpenRouter:Quoting OpenRouter (@OpenRouterAI) DeepSeek R1 is now live on OpenRouter!⚡ Performance on par with...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1331003245899939880)** (152 messages🔥🔥): 

> `DeepSeek R1 and V3 Comparison, Gemini 2.0 Flash Update, API Key Tiers for Gemini Models, Reasoning Content Retrieval, Perplexity's New Sonar Models` 


- **DeepSeek R1 for Reasoning and V3 for Chatting**: Users are discussing the ideal combination of models for optimal performance, recommending DeepSeek V3 for chatting and DeepSeek R1 for reasoning.
   - This combination is viewed as effective due to R1's reasoning capabilities alongside V3's chatting features.
- **Gemini 2.0 Flash gets a Major Update**: A new model, 'Gemini 2.0 Flash Thinking Experimental 01-21', has been released with a 1 million context window and 64K output tokens.
   - Users noted some inconsistencies in model naming during the rollout process, which took about ten minutes.
- **No Tiered API Keys for Gemini 2**: It is highly unlikely that Gemini 2 will require tiered API keys similar to O1, as it's not yet fully deployed on Vertex.
   - Currently, it is accessible only through AI Studio.
- **Strategies to Access Reasoning Content**: A user suggested a method to trick the system into displaying reasoning content by using certain prefixes in the API calls.
   - Concerns about managing token clutter from previous CoTs are raised, stressing the importance of effective message handling.
- **Perplexity Launches New Sonar Models**: Perplexity introduced two new Sonar models and users are encouraged to vote for their addition.
   - Feedback on Perplexity's performance is mixed, with some users expressing skepticism about the models’ utility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1881435007480475970?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: @risphereeditor @deepseek_ai Thanks for flagging - adding it now!</li><li><a href="https://x.com/risphereeditor/status/1881789442530435513?s=46">Tweet from Risphere (@risphereeditor)</a>: Perplexity now has a Sonar API.The Sonar API is a web search LLM engine. It uses Perplexity&#39;s fine-tuned LLMs.There are two models, Sonar and Sonar Pro. Sonar Pro has access to more sources.It&#39...</li><li><a href="https://x.com/Satomahga/status/1881576001479811527">Tweet from Sato Mahga (@Satomahga)</a>: `gemini-2.0-pro-exp` was added to the quota in Google Cloud (and accordingly to Ai Studio projects) about 40 minutes ago. The quota itself for the free tier is 100 requests per day and 5 requests per ...</li><li><a href="https://ai.google.dev/gemini-api/docs/thinking">no title found</a>: no description found</li><li><a href="https://openrouter.ai/settin">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers</li><li><a href="https://api-docs.deepseek.com/guides/reasoning_model">Reasoning Model (deepseek-reasoner) | DeepSeek API Docs</a>: deepseek-reasoner is a reasoning model developed by DeepSeek. Before delivering the final answer, the model first generates a Chain of Thought (CoT) to enhance the accuracy of its responses. Our API p...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1330994732146626752)** (81 messages🔥🔥): 

> `Cohere Access and Usability, Learning Rate Adjustment in Training, Model Training Techniques, Pre-training GPT-2, Cohere For AI Community` 


- **Cohere's Accessibility Sparks Discussion**: Users discussed the **accessibility** of **Cohere's** models, highlighting factors like lack of persistent sign-in and mobile app availability.
   - One user appreciated the accessibility, noting it keeps the chat **free** but acknowledged that some usability features like **dark mode** would enhance user experience.
- **Learning Rate Strategies Under Scrutiny**: A user posed a question about adjusting the `max_steps` parameter when re-training a GPT-2 model, asking if inconsistencies could arise.
   - Another member confirmed the need to double max_steps for two epochs to prevent the learning rate from decaying too quickly during training.
- **Advisory Notes on GPT-2 Training**: Members recommended following **Andrew Karpathy's series** for a structured approach to building a GPT-2 model, emphasizing the importance of foundational knowledge.
   - A user noted that rushing through adjustments without fully understanding them might lead to wasted resources in training.
- **Encouragement to Join Cohere Research Community**: A member encouraged newcomers to join the **Cohere For AI** community, emphasizing it as a space for sharing research and asking questions.
   - They provided a link to the Cohere research initiative that supports machine learning problem-solving efforts.
- **Trial Keys Offer Free API Access**: Participants shared that **trial keys** provide **free API access** for 1000 requests per month per model, which is a crucial resource for testing.
   - This allows users to evaluate models without incurring costs, making it attractive for those exploring AI solutions.



**Link mentioned**: <a href="https://cohere.com/research">Research | Cohere For AI </a>: Cohere For AI (C4AI) is Cohere&#x27;s research lab that seeks to solve complex machine learning problems. 

  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1331184420803182624)** (1 messages): 

> `RAG Implementation, Tool Use with Models, Live Q&A Session, Builder Community Connection` 


- **Live Q&A on RAG and Tool Use**: A live Q&A session focused on **RAG** and **tool use** with models is scheduled for **Tuesday at 6:00 am ET** on Discord Stage.
   - Participants are encouraged to **share experiences**, **ask questions**, and **connect with other builders** during this interactive session.
- **Opportunities to Learn and Share**: Attendees will have the chance to learn about **new implementations** and discuss their **challenges** while working with the models.
   - This session aims to foster a collaborative environment for builders to engage and support each other.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1331120211276857394)** (4 messages): 

> `Cohere iOS application, Cohere macOS application, Cohere beta testing` 


- **Inquiry About Cohere's iOS and macOS Apps**: A member expressed interest in whether there will be an **iOS** or **macOS** application for **Cohere** anytime soon.
   - They specifically asked if there is a **beta** version available or in the works.
- **Frustration Over Wait Time**: The same member humorously lamented that **Cohere** took too long to respond, expressing their feelings with a crying emoji.
   - This sentiment was met with laughter from others in the channel, indicating a light-hearted community atmosphere.


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1330993594525548584)** (3 messages): 

> `Dify.ai Issues, Cohere Key Error, IP Block Concerns` 


- **Dify.ai throws 403 Error with Cohere Key**: A user reported a **403 Forbidden error** while trying to add their **Cohere key** in a self-hosted **Dify.ai** setup, questioning the cause.
   - *Heard this could be an IP block*, but they recently updated to a paid plan, indicating potential frustration.
- **Support suggests downgrading version**: Another member mentioned handling a similar request previously and indicated that **Dify.ai** does not natively support their service due to potential routing issues from **China**.
   - They advised downgrading to version **0.8** as a workaround, noting that other users found success with this solution.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1330992810559668264)** (12 messages🔥): 

> `AGI Definition, Duplicate Content Issues in Cohere Command R+, Feedback on Cohere Model Performance` 


- **Understanding AGI**: AGI stands for **Artificial General Intelligence**, but there was a lack of detailed information found in the Cohere documentation regarding its specifics.
   - *The Cmd R Bot* simply provided the definition without additional context or resources.
- **Duplicate Responses from Cohere Command R+ 08-2024**: A user reported **excessive repetition** in chatbot responses when using the **Cohere Command R+ 08-2024** model, notably in the output regarding health-related topics.
   - Despite adjusting various parameters like temperature and max tokens, the issue persisted, leading to continued feedback and troubleshooting discussions in the channel.
- **User Suggestions for Improvement**: Users exchanged suggestions for troubleshooting the duplication issue, including **prompt engineering** and adjusting temperature settings to mitigate the problem.
   - Despite testing these suggestions, the user emphasized their exclusive use of **cmd-r-plus**, expressing appreciation for the internal feedback shared by team members.


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1331165349957341245)** (4 messages): 

> `Cohere CLI, Community Support, Building Roles` 


- **Cohere CLI Launch**: Cohere CLI was introduced as a tool to **effortlessly chat** with Cohere's AI directly from your terminal, showcased on [GitHub](https://github.com/plyght/cohere-cli).
   - The project was celebrated with enthusiasm, highlighted by a fun rocket emoji 🚀.
- **Support Acknowledged**: A member expressed gratitude, saying, *'appreciate the support!!'* in response to community assistance.
   - This shows the positive interactions and collaborative spirit within the community.
- **New Builder on Board**: Another member proposed making someone a **builder** within the community, saying, *'let me make you a builder here.'*
   - The recipient was pleasantly surprised and responded with excitement, saying, *'oh my gosh thank you!'*.



**Link mentioned**: <a href="https://github.com/plyght/cohere-cli">GitHub - plyght/cohere-cli: Cohere CLI: Effortlessly chat with Cohere&#39;s AI directly from your terminal! 🚀</a>: Cohere CLI: Effortlessly chat with Cohere&#39;s AI directly from your terminal! 🚀 - plyght/cohere-cli

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1331013821858316398)** (5 messages): 

> `Cohere's Math Accuracy, LLM Limitations, Improving AI Response Validity` 


- **Cohere struggles with basic math problems**: A member expressed frustration that when asked for the total number of weeks in 18 months, Cohere incorrectly calculated it as **27 weeks** by mishandling **monthly values**.
   - *Cohere's inaccuracies make it seem more efficient to perform calculations manually*, undermining the tool's intended purpose.
- **General limitations of LLMs in math**: Another member pointed out that it isn't just Cohere, but a general issue with **large language models (LLMs)** not performing well in mathematical tasks.
   - *Tokenization processes contribute to this limitation*, making LLMs less reliable for deterministic tasks.
- **Integration of math in complex projects raises concerns**: Concerns were raised about using AI for automation when basic math errors can render entire **projects** or **code** useless.
   - The expectation is for AI to save time, but erroneous outputs in math threaten that efficiency, highlighting a critical flaw in usability.


  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1331013983045685248)** (14 messages🔥): 

> `NotebookLM for college courses, AI-generated video content, Feedback for feature requests, Guidance on source code understanding, NotebookLM for Church services` 


- **Organizing Notebooks for College Courses**: Members suggested organizing **NotebookLM** by topics rather than individual sources for college courses to streamline data consistency and prompting.
   - *It simplifies workflow* and allows sharing:** 'Using a 1:1 notebook:source is only necessary to ensure podcast generation is based exclusively on that one source.'
- **AI eXplained Launches New Episode**: The latest episode of AI eXplained discusses the **rise of AI-generated videos**, detailing advancements like scriptwriting and animated video production.
   - Tune in to explore how machines are **redefining creativity** in the film industry and the implications for the future.
- **Feature Request Feedback Channel**: Members were informed that requests for NotebookLM features can be submitted in the **feature-requests** channel to gather user input.
   - This provides a platform for suggesting improvements, especially useful for researchers and clinicians.
- **Gemini Code Assist for Source Code**: For understanding source code repositories, members advised using **Gemini Code Assist**, which offers specialized features for this purpose.
   - NotebookLM was noted to sometimes yield inaccurate insights unless directly prompted with specific directions.
- **NotebookLM Revolutionizes Church Services**: One member shared their success using NotebookLM for analyzing long sermons, enabling the creation of detailed session reports from extensive YouTube livestream transcripts.
   - They plan to compile a **250-page book** and even consider a **2000-page Bible study**, calling NotebookLM a *game changer* for their church activities.


  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1330990681656787107)** (89 messages🔥🔥): 

> `NotebookLM Features, Audio Generation Limitations, Sharing Notebooks, Customizing Conversations, Tools and Add-ons` 


- **NotebookLM struggles with language settings**: Users reported challenges in changing the language for generated audio summaries and discussed methods like using `?hl=YOUR_LANGUAGE_CODE` in the URL.
   - Several users suggested logging out and back in to change language settings, while others sought confirmation if the features affect audio outputs.
- **Audio generation lacks control**: Members expressed frustration regarding the inability to control the length of audio outputs and the generation of APA formatted reference lists from all sources.
   - Suggestions included renaming files for easier reference usage, but users still found limitations in the overall functionality.
- **Need for better sharing options in NotebookLM**: Users discussed the current limitations in sharing notebooks with a classroom, suggesting the creation of Google Groups as a workaround.
   - Concerns were raised over not easily being able to share notebooks without manually entering each email, highlighting a need for improved functionalities.
- **Customization of conversation format**: A user sought methods to enforce a specific response style within NotebookLM, preferring brief conversational responses over longer lists.
   - Suggestions included creating a dedicated instruction prompt that could be referenced in subsequent interactions for consistency.
- **Exploring helpful tools and add-ons**: Participants shared useful add-ons for enhancing the NotebookLM experience, including ways to save prompts for quick reuse.
   - The community expressed interest in more collaborative development tools being integrated into the NotebookLM interface.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mavi.openinterx.com">Tweet from OpenInterX Mavi</a>: OpenInterX Mavi - The Future of Mass Video Understanding</li><li><a href="https://chromewebstore.google.com/search/notebookLM?utm_source=ext_app_menu">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://www.youtube.com/@MindfulnessCotidiano">Mindfulness Cotidiano</a>: &quot;Bienvenid@ a la comunidad de Mindfulness Cotidiano. Aquí encontrarás prácticas, meditaciones y consejos para mejorar tu vida. Suscríbete y forma parte de este camino consciente hacia el Bienesta...</li><li><a href="https://chromewebstore.google.com/search">Chrome Web Store</a>: Add new features to your browser and personalize your browsing experience.</li><li><a href="https://tenor.com/view/i-have-have-made-huge-mistake-wrong-gif-18221048">I Have Have Made GIF - I Have Have Made Huge - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: no description found</li><li><a href="https://t.me/talentosdigitales">🔥✨ 𝕋𝕒𝕝𝕖𝕟𝕥𝕠𝕤 𝔻𝕚𝕘𝕚𝕥𝕒𝕝𝕖𝕤 ⚡️</a>: Únete a Talentos Digitales en Telegram 🎬🎮🎶, donde encontrarás películas, juegos y música para todos los gustos. ¡Disfruta y comparte con nuestra comunidad&#33;
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1330995183268925491)** (90 messages🔥🔥): 

> `AI in Comic Book Creation, Image Generation with AI, AI Art Controversy, Stable Diffusion Configuration, Background Editing Tools` 


- **AI tools struggle with comic book consistency**: A member commented on the challenges of using AI to create consistent comic book assets, suggesting generating images panel by panel and utilizing ControlNet for scene control.
   - Despite attempts to create a cohesive visual narrative, many find the AI output inconsistent when generating multiple frames.
- **Concerns on using AI for image generation**: Discussions revealed skepticism about the effectiveness of AI-generated art, especially in achieving the desired quality for specific styles or characters.
   - For instance, a user expressed frustration at the AI's inability to generate satisfactory outputs for their comic characters despite using LoRA models.
- **AI art faces societal pushback**: A user noted the growing resistance to AI art, leading to further discussions on the ethical implications surrounding its use by artists and society.
   - The sentiment reflects a broader concern over the perception of AI-generated content in creative fields.
- **Configuration issues with Stable Diffusion**: Member struggles with configuring Stable Diffusion on AMD GPUs were shared, highlighting the technical challenges of setting up this AI tool.
   - Instructions from pinned messages in the Discord channel were recommended as potential help for troubleshooting.
- **Image editing discussed for personal projects**: Multiple users discussed using GIMP and other tools to manually edit images, emphasizing the importance of clean, unobtrusive backgrounds for personal photoshoots.
   - While AI was suggested as a possible solution for enhancements, many agreed that traditional editing methods are currently more efficient for achieving desired results.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.whitehouse.gov/presidential-actions/2025/01/initial-rescissions-of-harmful-executive-orders-and-actions/">Initial Rescissions Of Harmful Executive Orders And Actions &#8211; The White House</a>:     By the authority vested in me as President by the Constitution and the laws of the United States of America, it&nbsp;is&nbsp;hereby ordered as</li><li><a href="https://stablediffusionweb.com/image/25622118-robot-woman-with-removed-face-plate">Robot Woman with Removed Face Plate</a>: no description found</li><li><a href="https://web.archive.org/web/20250106193611/https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/">Executive Order on the Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence | The White House</a>: � � �By the authority vested in me as President by the Constitution and the laws of the United States of America, it is hereby ordered as
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1331158475841081394)** (10 messages🔥): 

> `GRPO implementations, TRL development, Float64 software for GPUs` 


- **Open Source GRPO Implementation Found**: A member shared a [link to the GRPO implementation](https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo) on GitHub, which focuses on super-efficient RLHF training of LLMs.
   - Another member expressed uncertainty about the project's maintenance and the basics of PPO.
- **Discovery of TRL's GRPO Development**: A participant noted that GRPO is being developed within the [TRL](https://github.com/huggingface/trl/pull/2565) repository, highlighting its relevance.
   - A sense of relief was shared regarding the availability of a validated HF implementation.
- **Inquiry on Float64 Software for GPUs**: One member inquired if anyone is familiar with software Float64 implementations specifically designed for GPUs.
   - This question reflects ongoing interest in optimizing GPU performance for various calculations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/openpsi-project/ReaLHF/tree/main/examples/new_algorithms/grpo">ReaLHF/examples/new_algorithms/grpo at main · openpsi-project/ReaLHF</a>: Super-Efficient RLHF Training of LLMs with Parameter Reallocation - openpsi-project/ReaLHF</li><li><a href="https://github.com/huggingface/trl/pull/2565">👨‍👨‍👧‍👧 GRPO by qgallouedec · Pull Request #2565 · huggingface/trl</a>: What does this PR do?from datasets import load_datasetfrom peft import LoraConfigfrom trl import GRPOConfig, GRPOTrainer# Load the datasetdataset = load_dataset(&amp;quot;trl-lib/tldr&amp;quot;, spli....
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1331148396060213331)** (19 messages🔥): 

> `Matrix Multiplication in Triton, Device-side TMA descriptors, Persistent GEMM implementation, Autotuning issues with TMA, Collaborative GPU research` 


- **Matrix Multiplication Process Unveiled**: A user analyzed the Triton tutorial for **matrix multiplication** and explored the implications of using parameters such as `num_pid_m = num_pid_n = 3` and `GROUP_SIZE_M = 2` in the L2 cache optimization examples.
   - This led to questions regarding the interpretation of `num_pid_in_group = 6` in terms of block and program definitions, highlighting the complexity of GPU programming.
- **Exploring Device-side TMA Descriptors**: A user discussed the challenges of utilizing **device-side TMA descriptors** in Triton, pointing out missing functionalities such as `triton.set_allocator` and `tl._experimental_make_tensor_descriptor` in the main branch.
   - Another member shared that the current workaround involves using `triton.runtime.driver.active.utils.fill_2d_tma_descriptor` for proper implementation.
- **Persistent GEMM Usage in Triton**: A user provided a working example of **persistent GEMM** leveraging TMA, affirming the dual implementation of device and host versions to facilitate manual configuration despite autotuning complications.
   - Concerns arose regarding compatibility with Triton 3.2, particularly involving the use of **numpy** for descriptor creation, which deviated from the required **torch** implementation.
- **Autotuning Challenges with TMA**: Users raised issues about **autotuning** not functioning correctly with TMA implementations, with attempts leading to crashes when multiple configurations were applied prior to the kernel.
   - Discussion revealed that manual configuration remains necessary due to limitations within the autotuner's support for TMA.
- **Call for Collaborative GPU Research**: A member suggested forming a group to work on interesting GPU-related papers, inspiring collaboration in implementation and research efforts.
   - This initiative aims to engage community members in tackling complex challenges together, fostering a collaborative learning environment.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py">GridQuant/scripts/gemm.py at main · niconunezz/GridQuant</a>: An attempt to implement GridQuant. Contribute to niconunezz/GridQuant development by creating an account on GitHub.</li><li><a href="https://github.com/niconunezz/GridQuant/blob/main/scripts/gemm.py#L79-L100,">GridQuant/scripts/gemm.py at main · niconunezz/GridQuant</a>: An attempt to implement GridQuant. Contribute to niconunezz/GridQuant development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1330994098441818202)** (14 messages🔥): 

> `Blackwell compute capability, CUDA Toolkit 12.8, CUDA and SFML integration, Audio processing on CUDA, cuFFT library issues` 


- **Blackwell Compute Capability Confusion**: There's a debate that the **NVIDIA RTX 5090** specs page lists the consumer Blackwell compute capability as **12.8**, which some believe is a typo, suggesting it should be in the range of **10.0 to 12.0**.
   - *Eriks.0595* notes that NVIDIA specifically gated certain capabilities, hinting at further limitations for Blackwell architecture.
- **Hope for Updated APIs with Blackwell**: Members expressed hope that the consumer Blackwell will include **TMA** and **WGEMMA APIs**, following possible upcoming releases of CUDA Toolkit 12.8.
   - *Eriks.0595* cautions that these APIs may be gated behind architecture flags, creating uncertainty.
- **Exploring CUDA and SFML Together**: A user inquired if there’s a method to combine **CUDA** for computing while using **SFML** for window handling purposes.
   - This question highlights the ongoing search for effective integration of these two frameworks.
- **Audio Processing Implementation Challenges on CUDA**: One member is attempting audio processing with **CUDA** using the **ManagedCuda-12** library, successfully transferring audio data but encountering issues with the **cuFFT** library module.
   - They aim to use this setup alongside **Audacity**, targeting efficient **FFT-stretch** functionality without real-time processing needs.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://gitlab.com/leinardi/gwe">Roberto Leinardi / GreenWithEnvy · GitLab</a>: System utility designed to provide information, control the fans and overclock your NVIDIA card</li><li><a href="https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/">NVIDIA GeForce RTX 5090 Graphics Cards</a>: Powered by the NVIDIA Blackwell architecture.</li><li><a href="https://github.com/ilya-zlobintsev/LACT">GitHub - ilya-zlobintsev/LACT: Linux GPU Configuration Tool</a>: Linux GPU Configuration Tool. Contribute to ilya-zlobintsev/LACT development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1331079669818200168)** (7 messages): 

> `FSDP fully_shard() behavior, einops alternatives with PyTorch, torch nightly build with Triton 3.2 compatibility, DeepSpeed checkpointing in Torch Lightning` 


- **FSDP fully_shard() requires loop for submodules**: It was noted that calling `fully_shard(module)` creates one parameter group out of `module.parameters()` to handle communication efficiently, meaning submodules must also be explicitly passed.
   - While calling `fully_shard(model)` handles leftover parameters, *you must call `fully_shard` on submodules* to ensure communication/computation overlap.
- **Using einops with torch.compile**: A question was raised regarding alternatives to `einops rearrange` that are compatible with `torch.compile`.
   - A link was provided that details how to use `einops` with `torch.compile` effectively: [Using torch.compile with einops](https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops).
- **Compatibility issues between torch nightly and Triton 3.2**: Concerns were expressed about using the latest Triton 3.2 build with the nightly PyTorch build, given that PyTorch might install its own version of Triton which is problematic.
   - The thread included an **ImportError** in the context of importing `AttrsDescriptor` from Triton's compiler, indicating compatibility issues.
- **DeepSpeed checkpointing with Torch Lightning**: A user inquired whether DeepSpeed's usual checkpointing in Torch Lightning automatically includes UCP, which pertains to the user-controlled parallelism.
   - They questioned if manual conversion from ZeRO checkpointing to UCP is necessary, suggesting some uncertainty about the integration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pytorch.org/docs/2.6/distributed.fsdp.fully_shard.html">torch.distributed.fsdp.fully_shard &mdash; PyTorch 2.6 documentation</a>: no description found</li><li><a href="https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops">Using torch.compile with einops</a>: Flexible and powerful tensor operations for readable and reliable code (for pytorch, jax, TF and others) - arogozhnikov/einops</li><li><a href="https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L309-L339">torchtitan/torchtitan/parallelisms/parallelize_llama.py at main · pytorch/torchtitan</a>: A PyTorch native library for large model training. Contribute to pytorch/torchtitan development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1331293521302524040)** (1 messages): 

> `Lindholm's Career, Unified Architecture Design, Nvidia Developments` 


- **Lindholm's Career Journey at Nvidia**: An intriguing talk was hosted in November 2024 discussing the remarkable career of engineer **Lindholm**, who recently retired from **Nvidia** just two weeks ago.
   - The discussion highlighted his contributions to the **unified architecture** that he designed, showcasing its significance in the field.
- **Insights on Unified Architecture**: The talk provided deep insights into the **unified architecture** by Lindholm, detailing its design principles and impacts within the industry.
   - Listeners can access the full discussion via this [Panopto link](https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3) for a comprehensive understanding of his work.



**Link mentioned**: <a href="https://ubc.ca.panopto.com/Panopto/Pages/Viewer.aspx?id=880a1d92-30d7-4683-80e7-b1e000f501d3">ESB 1013 - CPEN 211 101 - 2024W1 on 2024-11-19 (Tue)</a>: no description found

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1331020791793586289)** (7 messages): 

> `CUDA Toolkit Commands, CUDA and C/C++ Compatibility, Using Graphics Cards for AI, 100 Days of CUDA, Speeding Up Hugging Face Generation` 


- **Choosing CUDA Toolkit Commands**: The choice between versioned command `sudo apt-get -y install cuda-toolkit-12-6` and un-versioned `sudo apt-get install cuda-toolkit` impacts future updates, as the un-versioned command updates automatically, while the versioned requires explicit requests.
   - *One member commented*: 'The main difference is that one is versioned and one isn't.'
- **CUDA Toolkit Necessary for AI?**: A question arose on whether the CUDA Toolkit is always required for using a graphics card for AI, with references to AI Gradio's installation instructions lacking mention of it.
   - Another member suggested that *sometimes necessary CUDA Toolkit components are packaged within Python packages*, indicating uncertainty.
- **100 Days of CUDA Project**: A member highlighted the start of a project focused on '100 days of building CUDA kernels', sharing a [GitHub link](https://github.com/a-hamdi/cuda/tree/main) for contributions.
   - This initiative aims to engage developers in hands-on learning and building with CUDA.
- **Speeding Up Hugging Face Generation**: A member inquired about methods to speed up generation using Hugging Face's `generate()` within a trainer loop, sharing a [GitHub commit link](https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4) for context.
   - They noted that the model in use (liuhaotian/llava-v1.5-7b) does not support the vLLM tool they found as a potential solution.
- **CUDA Toolkit Package Confusion**: A member shared their uncertainty about always needing the CUDA Toolkit when utilizing a graphics card for local AI applications.
   - They pointed to the lack of mention in installation instructions for AI Gradio, leading to confusion.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Compatibility_of_C_and_C++">Compatibility of C and C++ - Wikipedia</a>: no description found</li><li><a href="https://github.com/huggingface/trl/commit/2ecd53ad77ef2a27729176e89299cba37b2487c4">🏎️ vLLM for Online DPO (#2558) · huggingface/trl@2ecd53a</a>: * vllm online dpo

* new arg and add back generation config [skip ci]

* import utils

* optional import and comment

* is_vllm_available

* support conv and not conv [ci skip]

* add o...</li><li><a href="https://github.com/a-hamdi/cuda/tree/main">GitHub - a-hamdi/cuda: 100 days of building Cuda kernels!</a>: 100 days of building Cuda kernels! Contribute to a-hamdi/cuda development by creating an account on GitHub.</li><li><a href="https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network">CUDA Toolkit 12.1 Downloads</a>: Get the latest feature updates to NVIDIA&#39;s proprietary compute stack.</li><li><a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages">CUDA Installation Guide for Linux</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1331089201747202099)** (5 messages): 

> `Revisiting the PMPP Book, CUDA Programming Platforms` 


- **Rereading the PMPP Book is Worth It**: A member suggested that rereading the book's latest edition is beneficial due to the **significant new content** being added.
   - Another member noted that many topics missing from the **2022 edition** will be covered in the new version.
- **Best Platforms for CUDA Learning**: A member inquired about recommended platforms for implementing and testing programming exercises from the PMPP book, specifically for learning **CUDA programming**.
   - Others mentioned various cloud GPU providers for GPU comparisons, like [Cloud GPU Comparison](https://cloud-gpus.com/), and usage of [Lightning AI](https://lightning.ai/) or **Google Colab** for CUDA kernels.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cloud-gpus.com/">Cloud GPUs</a>: no description found</li><li><a href="https://lightning.ai/">Lightning AI | Turn ideas into AI, Lightning fast</a>: The all-in-one platform for AI development. Code together. Prototype. Train. Scale. Serve. From your browser - with zero setup. From the creators of PyTorch Lightning.</li><li><a href="https://x.com/marksaroufim/status/1739206865106395563">Tweet from Mark Saroufim (@marksaroufim)</a>: Cuda kernels in google colab!
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1331197629438099500)** (11 messages🔥): 

> `CUDA in Poland, SIMD definition, Dining in Warsaw` 


- **CUDA: The Miracle of Polish Cuisine**: A member remarked that **CUDA** translates to miracles in Polish, epitomizing its significance, especially in local context.
   - Another member noted the difficulty of finding relevant **CUDA** resources online, as most results misleadingly connect to miracles when searched in Polish.
- **SIMD Unveiled: Single Instruction Multiple Dishes**: A brief exchange highlighted the definition of **SIMD** as 'Single Instruction Multiple Dishes', showcasing a humorous twist on computing terminology.
   - Members enjoyed the light-hearted banter surrounding this definition, with a member praising its creativity.
- **Pizza and Beer: The Ultimate Pairing in Warsaw**: A member invited others to dine in Warsaw at a place called **CUDA**, known for pizza and Polish beer, expressing eagerness to discuss technology while enjoying food.
   - One member excitedly confirmed their presence in Warsaw, leading to positive reactions from the group.



**Link mentioned**: <a href="https://maps.app.goo.gl/VqWM21B5gmiYoq4V6">CUDA · Warsaw</a>: no description found

  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages): 

leiwang1999_53585: happy to release https://github.com/tile-ai/tilelang , also support rocm 🙂
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1331348532065599570)** (1 messages): 

> `Fluid Numerics, Galapagos cluster, AMD Instinct MI300A` 


- **Fluid Numerics launches subscriptions for Galapagos cluster**: **Fluid Numerics** announces subscriptions and free trials on their heterogeneous **Galapagos** cluster, now featuring access to the **AMD Instinct MI300A** node.
   - They encourage users to **test** and benchmark their software on MI300A compared to MI300X, providing a [link to request access](https://www.fluidnumerics.com/shop/p/rcc-allocation-monthly-subscription).
- **Introducing the AMD Instinct MI300A node**: The new **AMD Instinct MI300A** node is available as part of the Fluid Numerics platform, aimed at AI/ML/HPC applications.
   - Users can reach out for more customized solutions to fit their specific needs.


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1331191149011931137)** (5 messages): 

> `Mind Evolution Strategy, Local GRPO Implementation, RL on Maths Datasets, OpenRLHF Framework` 


- **Mind Evolution Strategy Shines in Inference Tasks**: The paper explores the **Mind Evolution** strategy for scaling inference in Large Language Models, significantly outperforming Best-of-N and Sequential Revision strategies in planning tasks, as evidenced in the [arXiv submission](https://arxiv.org/abs/2501.09891).
   - The method solved over **98%** of problem instances on benchmarks like TravelPlanner and Natural Plan without the use of formal solvers.
- **Local GRPO Test Implementation Incoming**: A member is creating a simple **GRPO** local test implementation for fun, with the potential for later scaling using distributed methods like OpenRLHF with Ray.
   - They plan to spend a few days understanding hyper-parameters extensively.
- **Exploring RL on Maths Datasets**: A member expressed interest in utilizing **RL** on maths datasets for their first experiment and anticipates it might take a month or more.
   - They sought advice on using the `PRIME RL` codebase for their experiments, looking for recommendations.
- **Useful Resources in OpenRLHF**: Great blog posts about various **RL** algorithms are linked in the README of the [OpenRLHF GitHub repository](https://github.com/OpenRLHF/OpenRLHF), which may aid in learning.
   - This resource serves as an easy-to-use, scalable framework for high-performance RLHF implementations.
- **GRPO Algorithm Implementation Progress**: The absolute bare minimum of the **GRPO** algorithm has been implemented, with a functional version expected to be ready by tomorrow.
   - This marks a step towards further exploration and development of the GRPO strategy.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.09891">Evolving Deeper LLM Thinking</a>: We explore an evolutionary search strategy for scaling inference time compute in Large Language Models. The proposed approach, Mind Evolution, uses a language model to generate, recombine and refine c...</li><li><a href="https://github.com/OpenRLHF/OpenRLHF">GitHub - OpenRLHF/OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (70B+ PPO Full Tuning &amp; Iterative DPO &amp; LoRA &amp; RingAttention &amp; RFT)</a>: An Easy-to-use, Scalable and High-performance RLHF Framework (70B+ PPO Full Tuning &amp; Iterative DPO &amp; LoRA &amp; RingAttention &amp; RFT) - OpenRLHF/OpenRLHF
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1330998533251268679)** (21 messages🔥): 

> `GGUF vs other quantized formats, Inference backends comparison, Local vs cloud development, New AI services introductions` 


- **GGUF Dominates Quantized Model Landscape**: Members discussed the prevalence of **GGUF** files for quantized models, suggesting that GGUF has become the preferred format due to its ease of use on consumer hardware. The shift indicates that startups may gravitate towards easily accessible options like **Ollama**, which integrates well with local development.
   - One member mentioned that most organizations tend to internally quantize their models, while **GGUF** has several public quantizers that cater to end-users.
- **Inference Backend Performance Showdown**: Discussion about different **inference backends** highlighted that tools like **vLLM** and **TensorRT-LLM** offer better performance for large language models (LLMs). An article shared also provides benchmarks comparing vLLM, LMDeploy, and MLC-LLM, emphasizing the importance of choosing the right backend for user experience and cost efficiency.
   - The conversation pointed out that many of these tools are focused on edge inference, which varies from the needs of high parameter models.
- **Local Development with Cloud Iteration**: A new member inquired about the best resources for implementing models in **PyTorch** while developing workflows locally and iterating efficiently in the cloud. Tips were exchanged about tools that support such workflows, indicating a desire for easier local-to-cloud integration.
- **AI Services Business Introduction**: A new member introduced themselves, sharing their passion for AI and their experience running an **AI services company**. The community welcomed them, fostering connections among AI enthusiasts.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/social-credit-social-credit-score-credit-score-score-china-gif-23125701">Social Credit Social Credit Score GIF - Social Credit Social Credit Score Credit Score - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.bentoml.com/blog/benchmarking-llm-inference-backends">Benchmarking LLM Inference Backends</a>: Compare the Llama 3 serving performance with vLLM, LMDeploy, MLC-LLM, TensorRT-LLM, and Hugging Face TGI on BentoCloud.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1331011869657268294)** (22 messages🔥): 

> `R1 Model Performance, Titans Paper Insights, Adam-like Update Rules, Deepseek Reward Models` 


- **R1 Model Performance Under Scrutiny**: Members debated the effectiveness of the **R1** model, with one stating *'they're not that good'* and another expressing confusion about the model's use of PRMs.
   - Further discussion highlighted insights shared from previous messages, suggesting external resources may provide clarification.
- **Titans Paper Explores Memory in Deep Learning**: The **Titans** paper proposes combining short-term and long-term memory to improve sequence processing, leveraging recurrent models and attention.
   - *'Isn't it faster to tune the model on such a large dataset?'* was raised, questioning the efficiency across varying data sizes.
- **Potential for Adam-like Updates in Linear Attention**: Discussions revolved around the need for an **Adam-like update rule** for linear attention models, with members expressing mixed feelings about its implementation.
   - Concerns were raised that introducing new scaling methods might complicate learning, with insights on whether these parameters are data-dependent.
- **Deepseek Reward Model Architecture Inquiry**: Members are curious about the training process and architecture of **Deepseek's reward models**.
   - One member specifically inquired about the details to better understand their underlying mechanisms.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/1606.04474">Learning to learn by gradient descent by gradient descent</a>: The move from hand-designed features to learned features in machine learning has been wildly successful. In spite of this, optimization algorithms are still designed by hand. In this paper we show how...</li><li><a href="https://arxiv.org/abs/2306.13326">Solving systems of Random Equations via First and Second-Order Optimization Algorithms</a>: Gradient-based (a.k.a. `first order&#39;) optimization algorithms are routinely used to solve large scale non-convex problems. Yet, it is generally hard to predict their effectiveness. In order to gai...</li><li><a href="https://arxiv.org/abs/2501.00663">Titans: Learning to Memorize at Test Time</a>: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memo...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1331050552292806799)** (4 messages): 

> `Open Source Steering for LLMs, Current SAE Steering Methods, Open Source Steering Libraries` 


- **No Standardized Open Source for Steering LLMs Yet**: There isn't a standardized open source repository for steering **LLMs** using selected features from trained **SAEs**, as mentioned by members.
   - They discussed current steering methods, highlighting that a unified approach is still lacking, which hampers broader implementation.
- **Open Source Steering Libraries Available**: A couple of open source steering libraries were shared, including [steering-vectors](https://github.com/steering-vectors/steering-vectors) and [repeng](https://github.com/vgel/repeng).
   - Additionally, they referenced the [representation-engineering](https://github.com/andyzoujm/representation-engineering) library, which focuses on AI transparency through a top-down approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/729741769192767510/1153431135414669422/1321212227881275484">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord is great for playing games and chilling with friends, or even building a worldwide community. Customize your own space to talk, play, and hang out.</li><li><a href="https://github.com/andyzoujm/representation-engineering">GitHub - andyzoujm/representation-engineering: Representation Engineering: A Top-Down Approach to AI Transparency</a>: Representation Engineering: A Top-Down Approach to AI Transparency - andyzoujm/representation-engineering
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1331029483343315025)** (13 messages🔥): 

> `4bit/3bit vs f16 performance, Qwen R1 models and Q-RWKV conversion, math500 dataset for evaluation, pass@1 estimation method, evaluation templates for R1` 


- **Performance degradation in quantization methods**: A member inquired about performance degradation when comparing **4bit/3bit** vs **f16** quantization in recent models like **LLaMA** or **Qwen** in **MMLU-PRO** evaluations.
   - They wondered if the degradation was negligible or if it depended on the quantization effort, seeking concrete information.
- **Exploring Qwen R1 models conversion**: One user is considering converting **Qwen R1 models** to **Q-RWKV** and is looking for effective tests to compare outcomes with the base **R1 models**.
   - They expressed concerns about the ability to evaluate the conversion's success accurately.
- **Working with math500 dataset**: Members debated **math500**, a subset of the **Hendrycks MATH dataset**, and its evaluation methods for models like **R1**.
   - It was suggested that switching to **math500** for evaluation could be straightforward, highlighting the simplicity of integration.
- **Clarification on response generation**: A question arose regarding generating **64 responses** per query to estimate **pass@1** performance in model evaluations.
   - Members discussed whether greedy methods could be used for this estimation process, emphasizing the need for clarification.
- **Evaluation template for R1 models**: A member questioned whether the **R1** models require a different chat template or if they could be prompted like a base model.
   - This discussion indicates uncertainty on how to effectively utilize **R1** for evaluations.



**Link mentioned**: <a href="https://huggingface.co/datasets/HuggingFaceH4/MATH-500">HuggingFaceH4/MATH-500 · Datasets at Hugging Face</a>: no description found

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1331226089824981125)** (3 messages): 

> `Intermediate Dimension Selection, Exporting Model to HF Format, Model Parallelism Issues` 


- **Choosing Intermediate Dimension: 3x Importance?**: A member asked for confirmation on selecting the intermediate dimension and whether it needs to be **3x** for some reason.
   - The discussion aims to clarify the rationale behind such a parameter choice in model configuration.
- **Error While Converting Model to HF Format**: Another member reported encountering a `RuntimeError` while exporting a model from **neox to HF format** using `convert_neox_to_hf.py`. The error indicates a dimension mismatch based on the provided shape `[8, 512, 4096]` and input size **4194304**.
   - They questioned the feasibility of conversion for a multi-node run while sharing their training config details, seeking further input from the community.
- **Training Configuration Insights**: The training config file was shared, showcasing settings like **model_parallel_size** of **4** and **num_layers** set to **32**.
   - Specifics include parameters mentioning the **hidden_size** of **4096** and a **seq_length** of **8192**, highlighting configurations that affect the export process.
- **Request for Help on Export Issue**: A community member called on another for assistance regarding the export issue raised previously, ensuring support for the concern raised.
   - The interaction emphasizes the collaborative effort in troubleshooting technical challenges within the Discord group.



**Link mentioned**: <a href="https://rentry.co/f4tvoevf">{</a>: &amp;quot;pipe_parallel_size&amp;quot;: 0,  &amp;quot;model_parallel_size&amp;quot;: 4,  &amp;quot;make_vocab_size_divisible_by&amp;quot;: 1,  # model settings  &amp;quot;num_layers&amp;quot;: 32,  &a...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1331003885875105793)** (59 messages🔥🔥): 

> `Stargate Project, Gemini 2.0 updates, DeepSeek insights, Ai2 ScholarQA, WandB SWE-Bench` 


- **Stargate Project Investment Announcement**: OpenAI announced the **Stargate Project**, aiming to invest **$500 billion** over four years to build AI infrastructure in the U.S., starting with **$100 billion** immediately.
   - The project is backed by **SoftBank, Oracle**, and other tech partners, focusing on securing American leadership in AI and creating numerous jobs.
- **Experimental Updates to Gemini 2.0**: Feedback on **Gemini 2.0 Flash Thinking** has led Noam Shazeer to announce experimental updates based on community suggestions.
   - This reflects an ongoing commitment to refine the capabilities of **Gemini** through user insights.
- **DeepSeek's Breakthrough in AI Models**: DeepSeek gained attention after releasing the **DeepSeek V2** model, achieving a competitive edge with significantly lower inference costs compared to industry standards.
   - The company's innovative architecture and approach to AI have sparked excitement and discussions within the community.
- **Launch of Ai2 ScholarQA**: **Ai2 ScholarQA** is introduced as a tool for researchers to ask questions requiring multiple scientific papers for comprehensive answers, using a state-of-the-art model.
   - This platform aims to streamline literature reviews by offering comparative insights and citations.
- **WandB Achieves SOTA Verification**: WandB announced that their **SWE-Bench** submission has been officially verified as **State of the Art (SOTA)**.
   - This achievement highlights the significance of the SWE-Bench benchmark within the AI community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/hsu_steve/status/1881405336793276874?s=46">Tweet from steve hsu (@hsu_steve)</a>: There may be some sarcasm here. He attended a very good university in China (Zhejiang University), but it&#39;s not that famous internationally. The typical Chinese researcher that is hyped as a &#34;...</li><li><a href="https://x.com/ggerganov/status/1881734507575005683">Tweet from Georgi Gerganov (@ggerganov)</a>: Make your Mac think faster 🧠🧠Tomorrow I&#39;ll show you how to cancel your copilot subscription.Quoting Georgi Gerganov (@ggerganov) Make your Mac think 🧠Tomorrow I&#39;ll show you how to enable sp...</li><li><a href="https://x.com/drjimfan/status/1881382618627019050?s=46">Tweet from Jim Fan (@DrJimFan)</a>: That a *second* paper dropped with tons of RL flywheel secrets and *multimodal* o1-style reasoning is not on my bingo card today. Kimi&#39;s (another startup) and DeepSeek&#39;s papers remarkably conv...</li><li><a href="https://x.com/btibor91/status/1881285255266750564?s=46">Tweet from Tibor Blaho (@btibor91)</a>: OpenAI website already has references to Operator/OpenAI CUA (Computer Use Agent) - &#34;Operator System Card Table&#34;, &#34;Operator Research Eval Table&#34; and &#34;Operator Refusal Rate Table&#3...</li><li><a href="https://x.com/nearcyan/status/1773759331403714779)">Tweet from near (@nearcyan)</a>: Stargate by Nvidia ™</li><li><a href="https://x.com/kimmonismus/status/1881287794544550018?s=46">Tweet from Chubby♨️ (@kimmonismus)</a>: OpenAI already has a comparison (Computer Use Agent) between OpenAI&#39;s Operator and Claude 3.5 Sonnet CUA.Looks like release is imminent.Quoting Tibor Blaho (@btibor91) OpenAI website already has r...</li><li><a href="https://x.com/fal/status/1881533663747420364?s=46">Tweet from fal (@FAL)</a>: 🚨 New Model Alert, Minimax Video with subject reference https://fal.ai/models/fal-ai/minimax/video-01-subject-reference</li><li><a href="https://x.com/perplexity_ai/status/1881779310840984043">Tweet from Perplexity (@perplexity_ai)</a>: Introducing Sonar: Perplexity’s API.Sonar is the most affordable search API product on the market. Use it to build generative search, powered by real-time information and citations, into your apps. We...</li><li><a href="https://x.com/fofrai/status/1881452418577404309?s=46">Tweet from fofr (@fofrAI)</a>: You can now use a character reference image when using the Minimax &#34;video-01&#34; (Hailuo) model on Replicate.And, dang it&#39;s sooo good.https://replicate.com/minimax/video-01</li><li><a href="https://x.com/legit_rumors/status/1881558479753924708?s=46">Tweet from ʟᴇɢɪᴛ (@legit_rumors)</a>: Gemini 2.0 Pro Exp has just been added behind the scenes ✨</li><li><a href="https://x.com/allen_ai/status/1881784827063767117">Tweet from Ai2 (@allen_ai)</a>: Can AI really help with literature reviews? 🧐Meet Ai2 ScholarQA, an experimental solution that allows you to ask questions that require multiple scientific papers to answer. It gives more in-depth, d...</li><li><a href="https://x.com/openai/status/1881830103858172059?s=46">Tweet from OpenAI (@OpenAI)</a>: Announcing The Stargate ProjectThe Stargate Project is a new company which intends to invest $500 billion over the next four years building new AI infrastructure for OpenAI in the United States. We wi...</li><li><a href="https://x.com/dhravyashah/status/1881510837132906840?s=46">Tweet from Dhravya Shah (@DhravyaShah)</a>: Supermemory v2 is now open source! This new refresh is made with Remix, and probably the only big open-source remix application. also a lot of other cool stuff + rag pipeline. all fully OSS.-&gt; http...</li><li><a href="https://x.com/zizhpan/status/1881727148081517050">Tweet from Zizheng Pan (@zizhpan)</a>: Bro...this guy is not our Wenfeng 🥲. He is just another guy who can be found on baidu with the same Chinese name.Quoting Henry Shi (@henrythe9ths) DeepSeek just released the first Open Source Reasoni...</li><li><a href="https://x.com/dwarkesh_sp/status/1881844437346902297">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: .@dylan522p called it in Oct 2024.Quoting OpenAI (@OpenAI) Announcing The Stargate ProjectThe Stargate Project is a new company which intends to invest $500 billion over the next four years building n...</li><li><a href="https://scholarqa.allen.ai/">Ai2 ScholarQA</a>: no description found</li><li><a href="https://scholarqa.allen.ai/query/9d8946c0-756c-4148-b32e-c2d5bc8f8b09">Ai2 ScholarQA</a>: no description found</li><li><a href="https://bsky.app/profile/colin-fraser.net/post/3ldoyuozxwk2x">Colin (@colin-fraser.net)</a>: Here&#39;s why &quot;alignment research&quot; when it comes to LLMs is a big mess, as I see it.Claude is not a real guy. Claude is a character in the stories that an LLM has been programmed to write. ...</li><li><a href="https://allenai.org/blog/ai2-scholarqa">Introducing Ai2 ScholarQA  | Ai2</a>: Ai2 ScholarQA gives in-depth, detailed, and contextual answers to help with literature review.</li><li><a href="https://x.com/sama/status/1881851602727993711?s=46">Tweet from Sam Altman (@sama)</a>: build monuments in the desert</li><li><a href="https://www.liquid.ai/lfm-7b">Introducing LFM-7B: Setting New Standards for Efficient Language Models</a>: The world’s best-in-class English, Arabic, and Japanese model, native in French, German, and Spanish, optimized to be the substrate for private enterprise chat, code, fast instruction following, and a...</li><li><a href="https://x.com/shawnup/status/1881458032741400758?s=46">Tweet from Shawn Lewis (@shawnup)</a>: Our SWE-Bench submission has been accepted and is officially SOTA! Thanks SWE-Bench team for making such an important benchmark.</li><li><a href="https://x.com/noamshazeer/status/1881845900659896773?s=46">Tweet from Noam Shazeer (@NoamShazeer)</a>: Your feedback on Gemini 2.0 Flash Thinking has been incredible—thank you!We’ve taken your suggestions and made an experimental update…</li><li><a href="https://www.youtube.com/watch?v=zDo_RrzdRoQ">President Donald Trump announces AI infrastructure investment — 1/21/2025</a>: President Donald Trump announces a joint venture Tuesday with OpenAI, Oracle and Softbank to invest billions of dollars in AI infrastructure in the United St...</li><li><a href="https://github.com/MoonshotAI/Kimi-k1.5">GitHub - MoonshotAI/Kimi-k1.5</a>: Contribute to MoonshotAI/Kimi-k1.5 development by creating an account on GitHub.</li><li><a href="https://x.com/Kimi_ai_/status/1881332472748851259">Tweet from Kimi.ai (@Kimi_ai_)</a>: 🚀 Introducing Kimi k1.5 --- an o1-level multi-modal model-Sota short-CoT performance, outperforming GPT-4o and Claude Sonnet 3.5 on 📐AIME, 📐MATH-500, 💻 LiveCodeBench by a large margin (up to +550%...</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘DeepSeek:一个更极致的中国技术理想主义故事</a>: 做贡献者，而非搭便车者。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1331187377799954464)** (1 messages): 

> `Last Week in AI, Free AI in Gmail` 


- **Guest hosting on Last Week in AI podcast**: A member announced they guest hosted an episode of [Last Week in AI](https://www.listennotes.com/podcasts/last-week-in-ai/197-free-ai-in-gmail-minimax-fCdt-x_RXAF/) discussing the integration of free AI features in Gmail.
   - The episode examined recent advancements and implications of AI tools directly in email communication.
- **Focus on Gmail AI Features**: The podcast also highlighted the **free AI** features currently available in Gmail, emphasizing their potential to enhance user experience.
   - Listeners were particularly interested in how these innovations could streamline email management and improve productivity.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1330998289822253178)** (44 messages🔥): 

> `DeepSeek R1 Performance, Generative AI Impact on Creative Industries, AI Models Comparison, Local Model Running Capabilities, AI Output Compliance Issues` 


- **DeepSeek R1 shows promise for local usage**: Users discussed that **DeepSeek R1**, distilled into [Qwen 32B Coder](https://link.to.related.info), is a model worth running locally but raised questions about its performance on Ollama due to reported issues.
   - One user, with **32 GB of RAM** and **16 GB of VRAM**, explained that they are running it on a system that offloads heavy computation to the CPU.
- **Future of Generative AI in Creative Fields**: Members shared their thoughts on generative AI's rapid growth in the creative industries, with some believing it could eventually replace artists and creative professionals.
   - Concerns were raised about the accuracy of AI-generated art and the necessity of human skills to direct the output effectively.
- **R1 vs. O1 and Sonnet in Coding**: Comparisons were made among R1, O1, and Sonnet 3.5 regarding their capabilities in coding and math, noting that R1 had a **60% failure rate** in a specific project.
   - In contrast, **4O and Sonnet** reportedly had a **99% failure rate**, showcasing a variety of performance levels across models.
- **Challenges with AI Output Compliance**: It was noted that **DeepSeek** tends to avoid generating critical or humorous content regarding the **CCP**, similar to historical outputs from GPT concerning ESG compliance issues.
   - This raises questions about the implications for expression and debate within AI-generated content.
- **Speculative Future of AI Training Outcomes**: A user speculated about the possibility of AI companies inadvertently training models akin to **archotechs from Rimworld**, imagining unforeseen capabilities.
   - This speculation reflects broader concerns about the directions in which AI development could lead.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://medium.com/@techyspacelovers/generative-ai-how-its-shaping-creative-industries-f3e11960fe38">Generative AI: How It’s Shaping Creative Industries</a>: We all have used and heard about tools like Chatgpt. They are currently the rage in the tech world. We have also heard about how AI tools…</li><li><a href="https://www.dynocortex.com/news-and-blog/good-kg-created-by-ai/">Tweet from 
        
            Can an AI automatically create a good knowledge graph as of Jan 2025? - 
        
    </a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1331115380579631154)** (7 messages): 

> `GPT downtime issues, Chat response delays` 


- **Frequent GPT Downtime Concerns**: Users are reporting frequent issues with GPT, including messages like, *'Something went wrong. If this issue persists please contact us...'* that disrupt their chat.
   - One user mentioned that reopening the chat usually resolves the issue, indicating it may not be a permanent problem.
- **GPT Performance Slows Down**: Another member noted that GPT has become *REALLY slow* recently, leading to frustrating experiences during interactions.
   - This sentiment is echoed by several users, suggesting a broader performance issue affecting response times.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

oneidemaria: <:dallestar:1006520565558956092>
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

oneidemaria: <:dallestar:1006520565558956092>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1330999794948378705)** (46 messages🔥): 

> `Neural ODE Applications, Modeling vs Algorithmic Choices in ML, RL Techniques for Small Models, Exploration Strategies in RL, MoE vs Attention Mechanisms` 


- **Neural ODEs can revolutionize robotics**: Members discussed that Neural ODEs could be applicable in robotics, focusing on their ability to simulate layers based on function complexity and algorithmic decisions.
   - One member highlighted the importance of injecting knowledge at various layers to address the limitations of smaller models.
- **Balancing modeling and algorithmic choices**: Discussion centered around the need to balance modeling decisions, such as choosing between nonparametric, Bayesian, or NN approaches, with the algorithmic aspects of ML.
   - The quality of reasoning pathways and the selection of loss functions were identified as critical factors in the successful implementation of ML models.
- **Exploring RL strategies for small models**: There was a hypothesis suggesting that smaller models could discover high-quality reasoning pathways through repeated random initializations and evolutionary techniques.
   - Members debated the feasibility of these strategies, with concerns raised about the effectiveness and replicate capacity of such methods.
- **Importance of irregularity in RL**: Red_code emphasized the significance of introducing noise and irregularity in RL processes while penalizing regularity to enhance exploration during training.
   - Proposed strategies included direct logits sampling and avoiding softmax to preserve the nuances necessary for fostering high-quality reasoning.
- **MoE vs Attention Mechanisms**: Questions were raised about whether MoE can be considered a basic form of attention without key mapping, with members discussing the complexity of their implementations.
   - The discussion pointed to the interaction between different architectures and the implications for modeling choices in developing more effective ML systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/crosswind-landing-gif-20167802">Crosswind Landing GIF - Crosswind Landing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/moonflip-mflip-pepe-dance-moon-dance-gif-24962206">Moonflip Mflip GIF - Moonflip Mflip Pepe Dance - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/artoria-altria-arturia-king-arthur-pendragon-gif-21735401">Artoria Altria GIF - Artoria Altria Arturia - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.youtube.com/watch?v=jltgNGt8Lpg">Neural Ordinary Differential Equations</a>: https://arxiv.org/abs/1806.07366Abstract:We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers,...</li><li><a href="https://youtu.be/ZTNej2USaYk?t=106">La Passion</a>: Provided to YouTube by ZYX MusicLa Passion · Gigi D&#39;AgostinoL&#39;Amour Toujours℗ ZYX MusicReleased on: 2000-01-10Composer: Di Agostino, L.Music  Publisher: Medi...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1331041430075867206)** (4 messages): 

> `DeepSeeks Group Relative Policy Optimization, Review Process Challenges, Collaboration of Authors and Reviewers` 


- **Understanding GRPO in AI Optimization**: [DeepSeeks Group Relative Policy Optimization (GRPO)](https://fixupx.com/natolambert/status/1881380809153847711) is highlighted as PPO without a value function, using Monte Carlo estimates for advantages, which simplifies the model's complexity.
   - *Understanding the existence of PPO is crucial*, especially given the complexities of value functions with large language models.
- **Policy Optimization with Average Rewards**: The paper discusses how GRPO eliminates the need for additional value function approximation as required in PPO, utilizing the average reward from multiple sampled outputs instead.
   - This insight into GRPO suggests enhanced efficiency in optimizing policies for AI models, as cited in the [recent publication](https://arxiv.org/abs/2402.03300v3).
- **Challenges in Conference Paper Reviews**: Concerns were raised about needing more authors per paper to act as reviewers, as overburdening one reviewer can lead to inadequate evaluations.
   - One participant shared that recruiting **12 reviewers** from **50+ interested individuals** was necessary to obtain three high-quality reviews for each submission.
- **Overcoming Reviewer Shortages**: The need for extensive outreach was emphasized, as one user mentioned sending personalized messages as part of their efforts to secure quality reviews.
   - Despite these efforts, they felt the necessity to review several submissions personally, reflecting a **significant pressure** on the review system.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://fixupx.com/natolambert/status/1881380809153847711">Tweet from Nathan Lambert (@natolambert)</a>: For those trying to understand DeepSeeks Group Relative Policy Optimization (GRPO): GRPO is just PPO without a value function using monte carlo estimates of the advantage. So, study why PPO exists (lo...</li><li><a href="https://arxiv.org/abs/2402.03300v3">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a>: Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Co...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 messages): 

rogerngmd: https://github.com/deepseek-ai/DeepSeek-R1/blob/main/README.md
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1331303581487337474)** (1 messages): 

> `Suno AI Music Generator, Copyright Infringement Lawsuit, Music Industry Controversies` 


- **Suno faces new copyright lawsuit from GEMA**: AI music generator **Suno**, valued at **$500 million**, has been slapped with a copyright infringement lawsuit by Germany's licensing body **GEMA**.
   - This comes after Suno was previously sued by major record companies for using their tracks without permission, which they essentially acknowledged in their court filings.
- **Continued controversies surrounding AI music generation**: Suno, alongside fellow AI firm **Udio**, is embroiled in legal battles for allegedly training their systems on unlicensed recordings.
   - Despite the accusations, both companies have defended their actions, leading to ongoing debates in the **music industry** about the legality of AI-generated content.



**Link mentioned**: <a href="https://www.musicbusinessworldwide.com/500m-valued-suno-hit-with-new-copyright-lawsuit-from-germanys-gema/">$500m&#x2d;valued Suno hit with new copyright lawsuit from Germany&#8217;s GEMA &#x2d; Music Business Worldwide</a>: GEMA represents the copyrights of around 95,000 members in Germany (composers, lyricists, music publishers) as well as over two million rightsholders worldwide.

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1330991616458100796)** (14 messages🔥): 

> `Programming Language Preferences, Community Showcase Discussions, Mojo Progress Updates` 


- **C vs Python for Learning Programming**: Members debated the merits of starting with **C** versus **Python**, with some agreeing that **C** helps understand memory management regardless of future career paths in languages like **JS** or **Python**.
   - One member highlighted that starting with **C** can foster discipline, especially for those considering a career change later in life.
- **Community Showcase on Multiple Platforms**: There was discussion on advertising projects in both the **Discord** channel and the **forum**, with advisement that the forum is better suited for long-term discussions.
   - Members expressed the necessity to clarify the types of content appropriate for each platform to minimize duplication.
- **Feedback on Forum vs Discord Communication**: Opinions were shared regarding the pace of conversation, with some members preferring the **forum** for its slower, more processable information exchange compared to the faster-paced **Discord**.
   - It was noted that important discussions in Discord can be hard to locate later, suggesting a mix of usage for maintaining organized dialogue.
- **Current Mojo Development Progress**: A member inquired about the production use of **Mojo** and any upsides or downsides that users have noticed.
   - Another member confirmed that progress is being made and mentioned that **Nightly** builds are actively in development.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1330995578355843112)** (8 messages🔥): 

> `Mojo Project .gitignore, Netlify compatibility with Mojo apps, Mojo organization domain discussion` 


- **Mojo's Minimal .gitignore File**: The `Mojo` project initializes with a `.gitignore` that mostly ignores **magic-related** files, including `.pixi` and `.magic`.
   - This minimalism was generally expected by the community.
- **Confusion on Mojo and Netlify Hosting**: A member questioned if a `Mojo` app using `lightbug_http` could be hosted on **Netlify**, mentioning that a Rust app was successfully hosted.
   - Another member pointed out that it depends on the languages supported by Netlify's build images, suggesting Mojo currently isn’t included, but submitting a feature request could help.
- **Discussion on Mojo's Domain Presence**: There was a query about whether **Mojo** would have a separate organization with a `.org` domain like other programming languages.
   - It was clarified that there are no plans for Mojo to split from **Modular** or to change the current domain from modular.com.



**Link mentioned**: <a href="https://docs.netlify.com/configure-builds/available-software-at-build-time/">Available software at build time</a>: Learn about the software and tools that are available for your builds at build time.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1331004466312515727)** (2 messages): 

> `LlamaIndex Workflows, Chat2DB GenAI Chatbot` 


- **Deploy LlamaIndex Workflows on Google Cloud Run**: This guide walks you through [setting up a two-branch RAG application](https://t.co/nU1BctUh7s) for ETL and query processing, using LlamaIndex's event-driven framework for flexible AI systems.
   - It also covers [utilizing Google Cloud](https://t.co/AdynRZ79jn) for deployment.
- **Chat2DB GenAI Chatbot simplifies data interaction**: The open-source [Chat2DB genai chatbot](https://t.co/l1SFCEkiOC) allows querying databases with everyday language, featuring multiple interaction methods like RAG and TAG.
   - Key benefits include options for various LLM providers, such as OpenAI and Claude, making it a versatile tool for data access.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1331254300747366481)** (18 messages🔥): 

> `LlamaParse document parser, LlamaIndex documentation website bugs, Cached Augmented Generation with Gemini` 


- **LlamaParse recommended for PDF extraction issues**: Members suggested using [LlamaParse](https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/) for effective parsing of PDFs with selectable text, emphasizing its robust features for data cleaning.
   - *LlamaParse* is touted as *the world's first genAI-native document parsing platform* tailored for LLM use cases.
- **Users report bugs on LlamaIndex docs**: One user experienced issues with [LlamaIndex documentation](https://docs.llamaindex.ai/) scrolling back to the top randomly while browsing.
   - The user is troubleshooting by testing in incognito mode on Microsoft Edge, suspecting possible extension causes for the bug.
- **Incognito mode seems to resolve LlamaIndex browsing issue**: The user confirmed that accessing the docs in incognito mode on their laptop did not exhibit the scrolling issue, which is a positive finding.
   - Another member mentioned they haven't encountered similar problems with Edge, as it generally mirrors Chrome's performance.
- **CAG implementation with Gemini discussed**: A member inquired about implementing Cached Augmented Generation (CAG) with Gemini, but was informed that model-level access is needed.
   - It was clarified that *no model providers currently offer the necessary level of access over an API* for this implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/llama_cloud/llama_parse/">LlamaParse - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1330997153925042229)** (20 messages🔥): 

> `Entity Identification for ModernBert, Jinja Template Insights, LMstudio Inquiries, Adobe Photoshop Support, Nomic Taxes` 


- **Syntax for Identifying Entities in ModernBert**: A user inquired about the syntax for identifying entities in ModernBert and shared a sample hierarchical document layout for travel topics.
   - They expressed appreciation for any best practices on embedding documents with entities.
- **Best Resources for Jinja Templates**: A member requested suggestions for websites that explain the cool features and capabilities of Jinja templates.
   - This sparked curiosity among other users looking to enhance their understanding of Jinja.
- **Search for LMstudio Discord**: A user asked if it was acceptable to inquire about LMstudio in the channel, noting they couldn't find a dedicated Discord link or channel.
   - Another responded, seeking general support for Adobe Photoshop, highlighting a trend of unofficial support inquiries.
- **Humor over Illegal Questions**: A discussion ensued about a user potentially asking an illegal question related to Adobe Photoshop, prompting humorous exchanges about the response received.
   - This led to comments about the societal implications of asking about illegal information.
- **Nomic Taxes on Interns**: A humorous note was made regarding a tax rise for Nomic, with a member jokingly claiming the tax should be payable to themselves.
   - This was complemented by a light-hearted GIF referencing intern allocation, showcasing community banter.



**Link mentioned**: <a href="https://tenor.com/view/willj-oprah-oprah-winfrey-winfrey-you-get-a-car-gif-2219821026349492069">Willj Oprah GIF - Willj Oprah Oprah Winfrey - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1331021168194486274)** (5 messages): 

> `Bud-E language capabilities, Suno Music audio input feature, Project delay on current work` 


- **Bud-E Supports 13 Languages**: Members confirmed that Bud-E is not limited to English only, as it can utilize fish TTS for **13 languages**.
   - *No specific list of supported languages was provided.*
- **Current Project Frozen for Audio & Video Focus**: A member inquired about the project status, and it was noted that the project is 'frozen' due to a shift in focus on audio and video datasets.
   - *Focus on these datasets has led to development delays.*
- **Suno Music Empowers Music Creation**: [Suno Music](https://x.com/SunoMusic/status/1881742789639057828) enables users to create their own songs by recording various sounds or musical inputs for a personalized experience.
   - A member shared excitement over this feature, noting its broad accessibility on mobile devices.



**Link mentioned**: <a href="https://x.com/SunoMusic/status/1881742789639057828">Tweet from Suno (@SunoMusic)</a>: Record yourself singing, playing piano, or tapping your pencil + upload into Suno to make your own song from your own sounds 😱 What have you made with our audio input feature? 🎤: @techguyver shows h...

  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/1331020954779910204)** (1 messages): 

> `BUD-E, School-BUD-E, Open Source Voice Assistants, AI Education Assistant Framework` 


- **LAION Launches BUD-E & School-BUD-E Voice Assistants**: Today, LAION proudly announced the release of **BUD-E** version 1.0, a **100% open-source** voice assistant that integrates with various platforms like **Google AI Studio** and **Deepgram**.
   - This release marks a significant step toward **democratizing education** and **empathy** through technology, with versions catering to both general and educational use.
- **BUD-E Framework Aims for Universal Access**: **BUD-E**, which stands for Buddy for Understanding and Digital Empathy, strives to provide free, intelligent education assistants to everyone, regardless of location.
   - The release includes distinct versions, such as **School Bud-E** for educational settings and a 
**Desktop Bud-E** as a smart home assistant replacement.
- **Overview of BUD-E's Functionality**: The recently launched BUD-E encompasses features designed for both **educational** and **general purposes**, offering user-friendly interfaces for seamless interaction.
   - Tutorials and demonstrations are available, including an instructional [YouTube video](https://www.youtube.com/watch?v=IxHnpISMNPo) featuring a comprehensive rundown of its capabilities.
- **Accessibility Through Diverse Platforms**: BUD-E is compatible with self-hosted APIs and supports various technologies, allowing local data storage in users' browsers, enhancing **privacy compliance**.
   - LAION emphasizes its commitment to flexibility by providing access via web and desktop platforms, making tech education more reachable for everyone.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://laion.ai/blog/bud-e-release/">Introducing BUD-E 1.0: AI-Assisted Education for Everyone | LAION</a>: &lt;p&gt;Today marks a milestone in our journey towards democratizing education and empathy through technology. LAION e.V. &lt;em&gt;is thrilled to announce the release ...</li><li><a href="https://www.youtube.com/watch?v=y4DRYF9sfMU">School Bud-E - Overview</a>: English Version:https://school.bud-e.ai/?lang=enGerman Version:https://school.bud-e.ai/?lang=deCode:https://github.com/LAION-AI/school-bud-e-frontendBud-E (g...</li><li><a href="https://www.youtube.com/watch?v=IxHnpISMNPo">School BUD-E &amp; Bud-E: How to use our open, browser-based Voice Assistants (Tutorial)</a>: Tutorial on how to use School Bud-E &amp; Bud-E. :)English Version:https://school.bud-e.ai/?lang=enGerman Version:https://school.bud-e.ai/?lang=deBud-E (general ...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1331299207465930832)** (1 messages): 

> `IPTVPlayer, AtlasVPN, TradingView-Premium, Cʀᴀᴄᴋɪɴɢ Cʟᴀss` 


- **Top Repacks Featured in the Group**: The group promotes the best repacks available 24/7, emphasizing their exclusivity to members.
   - Members are encouraged to check out the Telegram channel for the latest offerings and updates on the best free programs.
- **Trading Made Easy with Premium Offers**: Promotions for **TradingView-Premium** claim to help users become real traders with unbeatable offers.
   - The channel highlights the importance of accessing premium trading tools for market success.
- **Join Cracking Class for Free Programs**: The **Cʀᴀᴄᴋɪɴɢ Cʟᴀss** chatroom boasts **64,400** subscribers sharing the best free programs.
   - Members are urged to join via Telegram for immediate access to discussions and resources.



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: The best programs are only free

  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1331299215174799453)** (1 messages): 

> `IPTVPlayer, AtlasVPN, TradingView-Premium, Cʀᴀᴄᴋɪɴɡ Cʟᴀss, Free Programs` 


- **Exclusive IPTV Repack Offers**: Members are encouraged to check out the best repacks of **IPTVPlayer**, along with other software available 24/7 only in the [group](https://t.me/repackMEMII).
   - The group claims to provide premium access to tools like **AtlasVPN** and **TradingView-Premium**, appealing to aspiring traders.
- **Join the Cracking Community**: The channel titled **Cʀᴀᴄᴋɪɴɡ Cʟᴀss** boasts a subscriber count of **64,400**, promoting free access to the best programs.
   - Users are invited to directly join the channel via Telegram to access communal resources and discussions.



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: The best programs are only free

  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/1331299593186574359)** (1 messages): 

> `IPTVPlayer offerings, AtlasVPN promotions, TradingView Premium features` 


- **Unlock IPTVPlayer Offers**: The group is promoting the **IPTVPlayer** with exclusive repacks available **24/7**, providing a unique opportunity for users.
   - Members are encouraged to check the offerings through the channel link.
- **Discover AtlasVPN Deals**: The channel highlights premium **AtlasVPN** deals aimed at enhancing security for users who want to protect their online activity.
   - Interested participants can find more details on the dedicated Telegram channel.
- **TradingView Premium Insights**: **TradingView-Premium** services are promoted as essential for becoming a real trader with the best offers for market analysis.
   - Information on this offering can also be accessed via the provided Telegram link.
- **Join Cracking Class for Free Software**: The **Cʀᴀᴄᴋɪɴɢ Cʟᴀss** chat boasts **64,400 subscribers**, promising a collection of programs available for free.
   - Members are welcomed to join the chatroom for access to these resources effectively.



**Link mentioned**: <a href="https://t.me/repackMEMII">Cʀᴀᴄᴋɪɴɢ Cʟᴀss [ᴄʜᴀᴛʀᴏᴏᴍs]</a>: The best programs are only free

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1331251453473587262)** (6 messages): 

> `Declaration Form Requirement, Corporate Sponsors and Intern-like Tasks, New MOOC Syllabus Release` 


- **Clarification on Declaration Form**: *A member asked if they need to fill the Declaration form again since they submitted it in December.* It was clarified that the form is now for those who missed the initial submission deadline.
- **Inquiry on Corporate Sponsors Offering Intern-like Tasks**: *A member expressed interest in whether corporate sponsors would provide intern-like tasks in the next MOOC.* It was noted that the previous semester's hackathon project served this purpose, although speakers might mention internship opportunities.
- **Timing for New MOOC Syllabus Release**: *A member inquired about the release date for the new MOOC syllabus.* A response indicated that the team is currently finalizing speakers and expects to post a rough syllabus by **January 27th**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1330992719820099697)** (5 messages): 

> `BEAM performance, WebGPU compatibility, YoloV8 FPS issues` 


- **BEAM drastically reduces YoloV8 performance**: A user reported that using **BEAM** with `python examples/webgpu/yolov8/compile.py` causes a performance drop from **40fps** to **8fps**.
   - *georgehotz* suggested that this behavior indicates a **bug**, noting that BEAM should not make performance worse.
- **WebGPU and BEAM Compatibility Concerns**: Another user speculated that BEAM might not function well with **WGSL** since it requires additional compilation to **SPIR-V** or platform-specific languages.
   - This leads to questions about whether the extra compilation step is simply too slow for effective performance.
- **Discussion on Backend Specifics for BEAM**: A user mentioned that BEAM needs to be utilized on the exact backend and hardware, indicating compatibility issues with **WebGPU**.
   - This raises concerns regarding the translation of BEAM performance when switching render targets.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1331327977975320619)** (1 messages): 

> `Proposal for tune cat command, TRL help command length` 


- **Excitement for `tune cat` Command Proposal**: A member expressed their appreciation for the **Torchtune** package and shared a [GitHub Issue](https://github.com/pytorch/torchtune/issues/2281) for a proposed `tune cat` command.
   - *It's an absolute pleasure to read* the source code, indicating a positive user experience overall.
- **TRL's Help Command Colossal Length**: Another member humorously remarked about the **TRL** help command length, noting it spans **three terminals**.
   - *This feature is overwhelming but essential for users.*



**Link mentioned**: <a href="https://github.com/pytorch/torchtune/issues/2281">[RFC] Proposal for `tune cat` Command · Issue #2281 · pytorch/torchtune</a>: First of all, thank you very much for the wonderful package. I’ve started actively looking at the source code, and I must say it’s an absolute pleasure to read. It was difficult to stop myself from...

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1331001537811578973)** (4 messages): 

> `Quantifying Uncertainty in LLMs, Chain of Thought in LLMs, RL-LLM Instruction Prompts, Distillation in RL` 


- **Models should quantify uncertainty**: *A member suggested* that models should be able to quantify uncertainty to some degree with the current approach, enhancing their reliability.
   - This concept aims to improve the interpretability and confidence of LLM outputs.
- **LLMs performing self-cot**: *Another member noted* that it feels like LLMs are conducting their own chain of thought (CoT) before providing answers, adding depth to the generated responses.
   - This observation highlights the potential of LLMs to reason internally before making statements.
- **Need for RL-LLM thinking-step prompts**: *A suggestion was made* for adding instruction prompts for thinking-steps in RL-LLM systems, complementing the existing goal-setting prompts.
   - This addition could enhance the model's reasoning process, leading to more informed outputs.
- **Improving RL on distillation**: *Another member pointed out* that RL techniques can still be applied on top of model distillation, potentially leading to further improvements.
   - It would be interesting to see if smaller models exhibit significant enhancement through this method.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

moresearch_: how does DSPy-based RAG deal with dynamic data?
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1331277034214658099)** (2 messages): 

> `Open Problem, Syntax Typo` 


- **Open Problem Still Under Discussion**: A user inquired if a particular issue remains an open problem, indicating ongoing interest and concern regarding its resolution.
   - This highlights the community's engagement in troubleshooting and problem-solving.
- **Typo Identified in Syntax**: Another user confirmed that the work functions correctly but pointed out a typo, mentioning that 'y=y' should contain a number instead.
   - *This discrepancy could lead to confusion if not addressed, emphasizing attention to detail in discussions.*


  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1331338533604163638)** (1 messages): 

> `Open Datasets for LLM Training, Mozilla and EleutherAI Partnership` 


- **Best Practices for Open Datasets Released**: The paper titled [Towards Best Practices for Open Datasets for LLM Training](https://discord.com/channels/1089876418936180786/1331335526338265108) was published on Arxiv to address challenges in open-source AI datasets.
   - It provides concrete recommendations to promote equity and transparency in the AI ecosystem.
- **Mozilla & EleutherAI Dataset Convening Partnership**: Mozilla and **[EleutherAI](https://discord.gg/cJQKYFDwHV)** partnered to host a Dataset Convening focusing on responsible data curation and governance.
   - Key stakeholders involved included community members dedicated to enhancing the open data environment in AI.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1331296777600630928)** (1 messages): 

> `AI in Cybersecurity, Impact of AI on Security Teams` 


- **AI's Leap into Cybersecurity**: A member reflected on their timely transition into the AI field a year ago, noting that previously, AI seemed more like a *buzzword* within cybersecurity products.
   - They expressed excitement about the potential of AI to genuinely assist security teams in the future.
- **Future AI Contributions to Security Teams**: The discussion highlighted the growing interest in how AI can truly enhance the effectiveness of security teams going forward.
   - Members anticipate significant advancements as AI becomes more integrated into security processes.


  

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

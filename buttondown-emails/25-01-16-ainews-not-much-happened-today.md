---
id: 396334e6-b342-4817-8596-176a39a5bfc0
title: '[AINews] not much happened today'
date: '2025-01-17T06:04:28.978541Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-not-much-happened-today-9711
---

<!-- buttondown-editor-mode: plaintext -->**a long weekend is all you need.**

> AI News for 1/15/2025-1/16/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **34** Discords (**225** channels, and **2732** messages) for you. Estimated reading time saved (at 200wpm): **327 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

Congrats to Harvey, for [their new $300m round](https://x.com/pitdesi/status/1879982274831347890?s=46).

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

**AI Model Developments**

- **Advanced Text-to-Speech Models**: [@reach_vb](https://twitter.com/reach_vb/status/1879647151145590905) announced the release of **OuteTTS 0.3 1B & 500M** models, featuring **zero-shot voice cloning**, **multilingual capabilities** (en, jp, ko, zh, fr, de), and **emotion control**. Powered by **OLMo-1B & Qwen 2.5 0.5B**, these models are a significant step in **Open Text-to-Speech revolution**.
  
- **HOVER Foundation Model for Motor Control**: [@DrJimFan](https://twitter.com/DrJimFan/status/1879922307923411081) introduced the **HOVER** model, a **1.5M-parameter neural net** designed for **agile motor control**. The model leverages **robust hardware designs**, **human motion capture datasets**, and **massively parallel RL training**, showcasing advancements in **robotic motor coordination**.

**AI Tools and Product Releases**

- **kokoro.js for Local AI Runs**: [@reach_vb](https://twitter.com/reach_vb/status/1879913142873944282) unveiled **kokoro.js**, allowing developers to **run AI models directly in the browser** with **minimal dependencies**. Available via `npm -i kokoro-js`, this tool promotes **local AI experimentation** without server reliance.
  
- **Moondream Integration and Tools**: [@vikhyatk](https://twitter.com/vikhyatk/status/1879741343347724341) teased **exclusive Moondream stickers** available at Walmart, while [@mervenoyann](https://twitter.com/mervenoyann/status/1879947783442202666) showcased **vision support for smolagents**, enabling the use of APIs like **gpt-4o** and various **HuggingFace transformers vision LMs**.

**Company and Industry News**

- **Meta's LLM Evaluation Grants**: [@AIatMeta](https://twitter.com/AIatMeta/status/1879990701234221190) announced the recipients of their **$200K LLM Evaluation research grants**, supporting projects focused on **regional language understanding**, **complex reasoning in LLMs**, and **interactive programming environments**.
  
- **Stability AI Twitter Account Hacked**: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1879743492450037889) reported that **Stability AI's Twitter account was hacked**, advising users to **avoid clicking suspicious links** until access is restored.

**Technical Insights and Research**

- **Process Reward Models (PRMs) Enhancement**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1879966399499759661) detailed their research on **Process Reward Models (PRMs)**, highlighting improvements in **data annotation** and **evaluation** for better **mathematical reasoning** in LLMs. The introduction of a **consensus filtering mechanism** integrates **MC estimation** with **LLM-as-a-judge** approaches.
  
- **Distributed Inference with DeepSeek V3**: [@awnihannun](https://twitter.com/awnihannun/status/1879679524167995901) explained the implementation of **pipeline parallelism** in **DeepSeek V3**, which **shards models by layers** across machines to **reduce communication latency**, enhancing **inference efficiency** for **long-context generations**.

**Policy and Societal Impact**

- **AI Policy and Legal Trust**: [@ajeya_cotra](https://twitter.com/ajeya_cotra/status/1879962661225730376) discussed the integration of **AI in legal frameworks**, focusing on **ensuring accuracy** of **AI-generated legal information** through **real-time verification** and **color-coded feedback** systems.
  
- **AI in Education and Accessibility**: [@emollick](https://twitter.com/emollick/status/1879688323419275430) emphasized the role of **AI in democratizing education**, highlighting initiatives where **students** without prior **computer access** benefited from **AI-powered learning tools**, showcasing **AI's potential to open up opportunities**.

**Memes / Humor**

- **Humorous Takes on AI and Technology**:
  - [@qtnx_](https://twitter.com/qtnx_/status/1879868377059213585) expressed humorously about avoiding certain terms, stating, "no longer using the word retard because Elon does and it looks cringe."
  - [@DesignerX](https://twitter.com/Teknium1/status/1879952088131891460) made jestful remarks about **math evaluations** and their complexities, indicating a lighthearted approach to **technical challenges**.
  - [@AravSrinivas](https://twitter.com/AravSrinivas/status/1879930212336701872) shared a laughing emoji in response to [@elonmusk](https://twitter.com/elonmusk), blending **tech discussions** with **casual humor**.
  

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. Google's Neural Memory Architecture Revolution**

- **[Google just released a new architecture](https://arxiv.org/abs/2501.00663)** ([Score: 891, Comments: 283](https://reddit.com/r/LocalLLaMA/comments/1i29wz5/google_just_released_a_new_architecture/)): **Google** has released a new architecture focused on **neural memory** to address long-term dependencies in models. The announcement is discussed in detail by the lead author in a [Twitter thread](https://x.com/behrouz_ali/status/1878859086227255347), suggesting its significance in advancing AI capabilities.
  - **Neural Memory Module**: The discussion highlights the **Neural Memory Module** as a key component of Google's new architecture, which uses semantic keys and dynamic memory management to handle long-term dependencies. It compares **Titans** to **RAG (Retrieval Augmented Generation)**, noting that **Titans** offers continuous learning during inference, unlike the static approach of **RAG**. [Source](https://arxiv.org/pdf/2302.00487).
  - **Performance and Memory Management**: Comments raise concerns about the performance of the new architecture, with some skepticism about its superiority over existing models like **Llama 3.1**. The architecture's ability to manage memory dynamically and handle larger knowledge bases is noted as a significant advantage, although the challenge of **catastrophic forgetting** remains unresolved.
  - **Context and Inference**: There is interest in the potential for **Titans** to achieve a **200k context window** with high accuracy, though concerns remain about inference speed and accuracy drop-offs beyond certain context lengths. The architecture's approach to integrating memory into the model without replacing traditional transformers is discussed, with some viewing it as a potential evolution rather than a revolution.


- **ATTENTION IS ALL YOU NEED PT. 2 - TITANS: Learning to Memorize at Test Time** ([Score: 311, Comments: 34](https://reddit.com/r/LocalLLaMA/comments/1i26nk4/attention_is_all_you_need_pt_2_titans_learning_to/)): **Google Research** introduces **Titans**, a new AI model that incorporates a dedicated "long-term memory" at test time, allowing it to adapt and update its memory dynamically. This model scales more efficiently with linear time complexity for long input sequences compared to traditional Transformers' quadratic time, potentially enabling theoretically infinite context windows.
  - The integration of **long-term and short-term memory** in AI models like **Titans** is seen as a major advancement, potentially pushing the boundaries of AI capabilities. However, there are concerns about the computational expense and memory requirements, with users questioning the feasibility of storing long-term memory in slower storage options and the potential need for retraining models like **llama-4**.
  - The **linear time complexity** of Titans is generating excitement, with users eagerly anticipating benchmarks to validate these claims. Some users express skepticism about the immediate adoption of such advancements in existing models, suggesting a more realistic timeline for widespread implementation.
  - **Titans**' architecture, particularly the use of a "surprise" mechanism for memory updates, is drawing interest, with references to other research like **SMiRL**. Users discuss the potential need for architectural changes to manage the balance between memory and token prediction efficiently.


**Theme 2. UMbreLLa Enhances LLM Performance on Consumer GPUs**

- **UMbreLLa: Llama3.3-70B  INT4 on RTX 4070Ti Achieving up to 9.6 Tokens/s! 🚀** ([Score: 132, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1i28pfq/umbrella_llama3370b_int4_on_rtx_4070ti_achieving/)): **UMbreLLa** enables running **Llama3.3-70B** models on consumer GPUs like the **RTX 4070 Ti** and **RTX 4090** with impressive speeds of up to **9.7 tokens/sec** and **11.4 tokens/sec** respectively. It achieves this through **parameter offloading**, **speculative decoding**, and **quantization (AWQ Q4)**, making high-end LLM inference accessible on affordable hardware, especially for coding tasks. [GitHub link](https://github.com/Infini-AI-Lab/UMbreLLa).
  - **Inference Speed and Hardware**: Users report varying token generation speeds depending on their hardware and PCIE settings, such as **3 times slower speeds** on some setups due to differences in **PCIE bandwidth**. A user mentioned achieving **10 tokens/sec** on a **4080 with 16GB** VRAM, while another noted only **1-3 tokens/sec** on a **3090 Ti**.
  - **Speculative Decoding and Performance**: **Speculative decoding** is a key feature, speculating up to **256 tokens** to achieve **13-15 tokens per forward pass**, with more than **20 tokens** possible in coding tasks. However, outside coding tasks, performance might not meet expectations, potentially being worse than CPU offloading.
  - **Compatibility and Future Plans**: Currently, the project does not support **AMD GPUs**, but there are plans to extend compatibility. Users are also interested in support for models like **Nemotron 51B** and potential integration with OpenAI-compatible APIs.


**Theme 3. Wayfarer Model Redefines AI Dungeon Experience**

- **Introducing Wayfarer: a brutally challenging roleplay model trained to let you fail and die.** ([Score: 160, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1i2t82i/introducing_wayfarer_a_brutally_challenging/)): **Wayfarer** is a new **AI roleplay model** introduced to address player frustrations with overly forgiving AI in **AI Dungeon**. This model, now open-sourced on [Hugging Face](https://huggingface.co/LatitudeGames/Wayfarer-12B), offers challenging adventures where failure and death occur frequently, and has received positive feedback from players.
  - Users report mixed experiences with **Wayfarer**, with one user noting role confusion during interactions. **Nick_AIDungeon** acknowledges user feedback and expresses openness to receiving more.
  - There is enthusiasm for scaling up the model, with **Nick_AIDungeon** confirming that larger models are currently being trained to enhance the experience.
  - The model is appreciated for its unique approach, likened to a "souls-like" experience, with users expressing gratitude for the open-source availability and the opportunity for challenging AI interactions.


**Theme 4. Meta-Prompt Strategies for Improved LLM Task Management**

- **Meta Prompts - Because Your LLM Can Do Better Than Hello World** ([Score: 133, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1i2b2eo/meta_prompts_because_your_llm_can_do_better_than/)): **Meta-prompts** significantly enhance the capabilities of **Large Language Models (LLMs)** by breaking down complex projects into manageable tasks through structured prompting. The concept originated from a [research paper](https://arxiv.org/pdf/2401.12954) and involves using prompts to define roles, rules, and deliverables, enabling LLMs to act as software architects, project managers, and developers. By providing context, structure, and clear outputs, meta-prompts transform LLMs into efficient team members, capable of handling enterprise-level complexity, as demonstrated in various [examples](https://gist.github.com/pyros-projects/c77402249b5b45f0a501998870766ae9) and [guidelines](https://gist.github.com/pyros-projects/e2c96b57ac7883076cca7bc3dc7ff527).
  - **Prompt Engineering** is akin to asking thought-provoking questions to humans; it leverages the LLM's training by using questions that it associates with high-quality responses, thereby eliciting its best and most insightful outputs.
  - **Close Sourcing Concerns**: There is a sentiment expressed that close sourcing for profit might not align with the ethos of the subreddit, suggesting a preference for open-source or community-focused approaches.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**Theme 1. Titans: Successor to Transformers with Human-like Memory**

- **Successor to the famous transformer: Titans** ([Score: 264, Comments: 63](https://reddit.com/r/OpenAI/comments/1i2bc5d/successor_to_the_famous_transformer_titans/)): Google Research has released a paper on **Titans**, a new model that outperforms larger models with only **300 million parameters**. This advancement suggests real-time learning and thinking capabilities similar to human cognition, with significant implications for AI in **2025**. [Read more](https://arxiv.org/pdf/2501.00663).
  - **Titans Model Characteristics**: The **Titans model** is noted for its novel neural memory module that mimics human-like memory by remembering surprising events and has a large context window of up to **2 million tokens**. However, it doesn't engage in real-time learning in the traditional sense of updating model weights, which is a key distinction from human cognition.
  - **Comparison with Transformers**: The discussion highlights **Titans** as a potential step forward from transformers, combining elements of RNNs and transformers, but skepticism remains about its revolutionary impact. The model's memory mechanism is integrated directly into the architecture, similar to the attention mechanism, allowing it to handle large contexts more effectively, but with economic considerations for practical use.
  - **Human-Like Memory**: Several commenters emphasize that **Titans**' bias toward remembering surprising events and the gradual decay of memories over time is reminiscent of human memory processes. While this is seen as promising, it is also noted that **Titans** do not solve fundamental issues of continual learning, as the memory is finite and context-based rather than weight-based learning.


- **[OpenAI researcher indicates they have an AI recursively self-improving in an "unhackable" box](https://i.redd.it/s7ozn53ml8de1.png)** ([Score: 189, Comments: 79](https://reddit.com/r/OpenAI/comments/1i2a9cl/openai_researcher_indicates_they_have_an_ai/)): **OpenAI** is reportedly developing an AI capable of **recursive self-improvement** within an **"unhackable" environment**. This claim is based on a tweet by **Jason Wei** (@_jasonwei) referring to an **RL optimization algorithm** that operates within a secure RL environment.
  - The term **"unhackable"** is criticized for being misleading, as it likely refers to an RL environment where the AI cannot exploit the reward function rather than being completely secure from external hacking. **Jason Wei's** tweet is seen as part of a pattern of vague hype from **OpenAI** employees, leading to misinformation and unwarranted excitement.
  - Discussions highlight skepticism about **OpenAI's** claims and the potential for **recursive self-improvement**. Some argue that the concept is not new, comparing it to **AlphaGo's** self-play method, which involves training against itself to improve performance.
  - Concerns are raised about the potential risks of developing AGI without ethical safeguards, with mentions of social engineering as a possible vulnerability even in supposedly secure systems, emphasizing the necessity for robust security measures.


**Theme 2. Financial Analysis of AI Subscriptions and Usage**

- **[I pay $200/month for pro subscription, and this is what I do with it](https://i.redd.it/c1u1n894fcde1.png)** ([Score: 2051, Comments: 187](https://reddit.com/r/OpenAI/comments/1i2n2ib/i_pay_200month_for_pro_subscription_and_this_is/)): The post discusses a **$200/month pro subscription** service, likely **ChatGPT**, used for developing a **React website**. The interaction highlights the service's processing capabilities and acknowledges potential errors, as indicated by the note that "ChatGPT can make mistakes."
  - **ChatGPT's Efficiency and Usefulness**: Many users express skepticism about the $200/month subscription's value, with some expecting it to double their earnings or to read minds. However, others appreciate the AI's ability to efficiently guide non-coders in developing React apps, emphasizing the importance of providing specific instructions to achieve desired results.
  - **React and Development Challenges**: Users discuss the challenges of using React, with some expressing disdain for the framework and others highlighting the time savings from using AI for boilerplate code. A few recount personal experiences where ChatGPT struggled with complex tasks like implementing graph theory algorithms, leading them to complete the task manually.
  - **AI's Directness and Trustworthiness**: Several comments highlight the AI's direct responses as a positive trait, contrasting it with the overly detailed answers of earlier versions. This directness is likened to real developers' responses to vague project requests, fostering a sense of trust in the AI's capabilities.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. AI Tools Garner Funding but Get Stuck in the 'Loop of Doom'**

- **Cursor Snags $105M but Users Hit 'Loop of Doom'**: Cursor announced raising **$105 million** in Series B funding from Thrive, a16z, and Benchmark as per [their official statement](https://x.com/cursor_ai/status/1880003590493991072), but users continue to report slow requests and recurring stalls, dubbing it a "loop of doom". Despite frustrations, many remain loyal due to Cursor's powerful autocomplete and integrated environment, finding productivity gains over traditional setups.
- **Codeium's Windsurf Faces Outages Amid Student Discounts**: Codeium rolled out the new **Windsurf Editor** and offered a **student discount** for .edu emails on [their site](https://www.codeium.com), but users experienced service interruptions, delayed feature improvements, and even account cancellations over a disputed **$297 refund**. Codeium touts its performance edge against **GitHub Copilot** on their [comparison page](https://www.codeium.com/compare), intensifying debates among users.
- **Phi-4 Fine-Tuning Frenzy Hits Snags**: Unsloth AI users successfully fine-tuned **Phi-4** models on small datasets using free Colab GPUs, but faced out-of-memory errors when saving merged models. Discussions highlighted challenges with dynamic quantization versus **GGUF** formats and endless generation issues with **Phi-4** under `llama.cpp`.

**Theme 2. New AI Architectures Promise to Outscale the Titans**

- **Google's Titans Take Aim at GPT-4's Throne**: Google Research unveiled the **Titans** architecture, introducing a neural long-term memory module capable of handling context windows larger than **2M**, detailed in [their paper](https://arxiv.org/abs/2501.00663v1). Members speculated whether this could crack *"human-like"* memory for LLMs, potentially outpacing GPT-4.
- **Modded NanoGPT Breaks Training Speed Records**: A modded **NanoGPT** trained in **3.17 minutes**, beating the previous 3.33-minute record, as shared in [this tweet](https://x.com/hi_tysam/status/1879687807678959729). Developers credited optimizations like **Long-Short Sliding Window Attention** for the speed boost.
- **Tensor Product Attention Slashes KV Cache Bloat**: A new paper proposes **Tensor Product Attention (TPA)** to scale language models with smaller KV caches, referencing the [T6 implementation](https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64). Authors plan a Flash version of TPA, aiming for further speed gains in large-scale deployments.

**Theme 3. AI Ethics Shake-Ups: Data Policies and DMCA Take-Downs**

- **OpenAI Stops Snooping—Defaults to No Data Training**: OpenAI changed its API data usage policies, stating they won't use customer data for training unless users opt-in, addressing concerns over data privacy. Details were shared in a [TechCrunch article](https://techcrunch.com/2023/03/01/addressing-criticism-openai-will-no-longer-use-customer-data-to-train-its-models-by-default/), marking a shift in how AI companies handle user data.
- **DMCA Takedown Topples MATH Dataset**: The popular **Hendrycks MATH** dataset was hit with a DMCA notice, referencing content from **aops**, as reported in [this tweet](https://x.com/tmkadamcz/status/1879584048429105238). Community members lamented the loss, calling it *"a bigger loss than The Pile or Books 3,"* underscoring the dataset’s significance for open math resources.
- **Bora's Law Challenges Compute-Centric AI Development**: Members debated **Bora's Law**, the principle that *"intelligence scales with constraints, not compute,"* as presented in [this article](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with). Critics argued that excessive scaling overlooks essential aspects of intelligence, suggesting a need to focus on constraint-driven models.

**Theme 4. Coders Clash Over AI Coding Companions**

- **Codeium vs. Copilot: Battle of the Code Gens**: Users compared **Codeium** and **GitHub Copilot**, with Codeium promoting its performance superiority on their [comparison page](https://www.codeium.com/compare). Despite praise for its advanced autocomplete, users criticized delayed feature rollouts and customer service issues, including a disputed **$297 refund** fiasco.
- **Cursor's Coding Powers Pitted Against Glitches**: Users praised **Cursor's** advanced autocomplete and integrated environment, reporting major workflow improvements despite facing slow responses and *"loop of doom"* stalls. Many compared Cursor favorably against alternatives like **Windsurf**, citing Cursor's deeper toolset and better cost-effectiveness.
- **ChatGPT Can't Code? Users Debate AI's Dev Skills**: Discussions emerged about **ChatGPT's** inability to function as a true software engineer, with users noting that while it can assist in coding, it lacks the capacity to develop complex applications independently. Hopes were expressed for future enhancements to bridge this gap.

**Theme 5. Multi-Agent Systems and Tooling Take Center Stage**

- **MCP's Dynamic Tool Discovery Dazzles Developers**: MCP introduced dynamic tool discovery, allowing clients to list available tools and receive real-time updates when tools change, reducing the need for restarts. This approach helps developers keep pace with frequent tweaks in tool signatures and preserve stable usage.
- **Open-Swarm Streams Smart Agent Moves**: The Open-Swarm framework offers a direct alternative to [OpenAI's original swarm framework](https://github.com/openai/swarm/blob/main/swarm/core.py), focusing on clarity in agent roles and built-in tool usage. It streamlines tasks like database queries and web interactions with minimal overhead.
- **OpenAI's Realtime Agents Explore Advanced Patterns**: OpenAI released a demonstration of advanced, agentic patterns built on top of the Realtime API in their [openai-realtime-agents](https://github.com/openai/openai-realtime-agents) GitHub repository. This showcases multi-agent orchestration for enhanced interactions, pointing toward more ergonomic and lightweight multi-agent systems.


---

# PART 1: High level Discord summaries


## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Creeping Speed**: Users reported slow requests and repeated stalls, calling it a "loop of doom" and trying partial fixes for better stability.
   - They also considered alternative editors like Windsurf, though many remain loyal to Cursor's deeper toolset.
- **Cursor's Colossal Bounty**: Cursor announced raising **$105 million** in Series B from Thrive, Andreessen Horowitz, and Benchmark as confirmed in [their official statement](https://x.com/cursor_ai/status/1880003590493991072).
   - Community members hope this injection will strengthen features and reduce performance hiccups.
- **Cursor as a Productivity Powerhouse**: Several users praised Cursor's advanced autocomplete and integrated environment, reporting major workflow improvements compared to older setups.
   - They noted that these benefits overshadow slower responses, making Cursor the top pick among current tools.
- **Cursor vs. Windsurf Rumble**: Participants compared Cursor and Windsurf, citing Cursor's stronger functionality and better cost-effectiveness.
   - Despite some slowdowns, most favored Cursor's robust features over other editing options.
- **Python Path Puzzle**: A user found that Cursor unexpectedly applied a project's Python environment globally, causing confusion in their setup.
   - Community members discussed environment selection, emphasizing the need for clearer integration with local tooling.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Gains Live Tool Updates**: Dynamic tool discovery ensures a real-time list of available capabilities, reducing restarts when features change.
   - This approach helps developers keep pace with frequent tweaks in tool signatures and preserve stable usage.
- **Open-Swarm Streams Smart Multi-Agent Moves**: Open-Swarm offers a direct alternative to [the original swarm framework](https://github.com/openai/swarm/blob/main/swarm/core.py), focusing on clarity in agent roles and built-in tool usage.
   - It streamlines tasks like database queries and web interactions with minimal overhead.
- **Marketing Tools From OSP Shake Up Product Positioning**: Open Strategy Partners introduced [osp_marketing_tools](https://github.com/open-strategy-partners/osp_marketing_tools), enabling LLMs to tackle product marketing tasks.
   - It focuses on value mapping and writing style checks, adding clarity to promotional content.
- **SSE Gains Momentum In Sage & Smithery**: SSE support is in the works for the Sage client, with talk about tailoring request bodies for better control.
   - Smithery rolled out a cloud hosting option for STDIO servers using SSE, driven by JSON-based configurations.
- **Discord Bot Ruffles Feathers**: Members criticized the existing bot, saying they'd rather code a more efficient replacement.
   - They also noted modern Discord built-ins like **/ban**, pointing to more robust user options.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Editor & Student Pricing Perks**: Codeium introduced the new **Windsurf Editor** packed with dev-focused features, while offering a **student discount** for .edu addresses on [their site](https://www.codeium.com).
   - International students using **.ac.uk** and **.unina.it** domains voiced eligibility concerns, prompting them to contact [support](https://codeium.com/support) until this offer extends more broadly.
- **DeepSeek Leaves Users in Loops**: DeepSeek drew negative feedback for causing infinite loops when paired with Cline, despite Codeium touting impressive benchmarks.
   - Community members called it *not practical for everyday use*, urging engineers to fix these reliability issues.
- **Cascade Prompt Tips & Feature Gripes**: Members shared **Cascade** tactics like inline commands and prompt reusability to maximize credit usage and output quality.
   - They also criticized delayed improvements (like missing drag-and-drop), spotlighting months of unattended requests and urging faster feature delivery.
- **Refund Fiasco & Codeium vs Copilot Showdown**: A user’s **$297 refund** dispute led to account cancellation instead of resolution, sparking a backlash over Codeium’s support methods.
   - Meanwhile, Codeium touts its performance edge against **GitHub Copilot** in a [comparison page](https://www.codeium.com/compare) despite ongoing service outage complaints.
- **Enterprise Plan & GPL-Free Training**: Codeium advertised an **Enterprise Plan** with self-host capabilities and emphasized they don’t train on **GPL** code, referencing [this blog post](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not).
   - They view this stance as crucial for shielding organizations from legal pitfalls while still providing advanced AI-driven dev workflows.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Phi-4 Fine-Tuning Frenzy**: A user successfully fine-tuned **Phi-4** on a small dataset using [free Colab GPUs](https://x.com/UnslothAI/status/1879942441538609583), highlighting challenges around out-of-memory errors for saving merged models. They also compared **dynamic quantization** with **GGUF** formats for inference efficiency.
   - Discussions tackled erroneous endless generation in **Phi-4** under `llama.cpp`, plus uncertainties about correct chat templates for **Ollama**, referencing [Unsloth documentation](https://docs.unsloth.ai/get-started/installing-+-updating).
- **Onnx vs TensorRT Tussle**: A user discovered **significant output discrepancies** when running the same model via **Onnx** versus **TensorRT**. They questioned whether framework optimizations or conversion steps might explain the mismatch.
   - No specific fix was offered yet, but this discrepancy sparked concern about **deployment consistency** across inference engines, especially for critical tasks.
- **Flash Attention 2 Snafu**: Someone reported a failing install for **Flash Attention 2**, needed for performance testing. Another member offered direct help with a Colab environment to troubleshoot.
   - They advised verifying dependencies and consistent GPU drivers, ensuring **Flash Attention 2** doesn't break crucial speed tests for advanced fine-tuning.
- **Grokking Gains & LORA Distillation**: A discussion on **grokking** and sudden model generalization referenced [a YouTube video](https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ) exploring how overfitting can morph into unexpected insight. The conversation hinted that insights on memorization vs genuine learning might influence **Unsloth**’s training techniques.
   - Members also debated applying **LORA** for knowledge distillation, questioning if it equates to response-based distillation for advanced training strategies.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLM Batching Gains Steam**: Members explored *batch text continuations*, noting **llama.cpp** only supports single prompts, and singled out **vllm** as a solution.
   - They see batch-based APIs as vital for streamlining **token-by-token training**, citing *the next wave of scalable LLM services.*
- **DMCA Takedown Topples MATH**: A **DMCA notice** halted **Hendrycks MATH** on Hugging Face, referencing content from **aops** and reported in [this tweet](https://x.com/tmkadamcz/status/1879584048429105238).
   - Community members called it *a bigger loss than The Pile or Books 3,* underscoring the dataset’s significance for open math resources.
- **Modded NanoGPT Shatters Speed Record**: A modded **NanoGPT** trained in **3.17 minutes**, beating the previous 3.33-minute record shared in [this tweet](https://x.com/hi_tysam/status/1879687807678959729).
   - Developers credit **Long-Short Sliding Window Attention** for context gains, pointing to a [GitHub pull request](https://github.com/KellerJordan/modded-nanogpt/pull/71) for further improvements.
- **TruthfulQA Tricks Emerge**: Members boosted **TruthfulQA** accuracy to **79%** through simple heuristics, detailed in [this post](https://turntrout.com/original-truthfulqa-weaknesses).
   - They argued flawed human annotations weaken *Halueval*, calling for stronger benchmark design to protect test integrity.
- **Deepspeed Zero Stages Divide Devs**: A user found **Deepspeed zero stage 2** incompatible with model parallelism, as indicated in [this code snippet](https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973).
   - They reported just **28 TFLOPs** per unit on **512 AMD MI250x GPUs**, describing *the shortfall from AMD’s stated specs.*



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Title Tinkering in Bolt**: A new update to **Bolt** allows editing project titles directly, as announced on [Stackblitz Twitter](https://x.com/stackblitz/status/1879625416706785365), making it simpler to track projects in the list.
   - This improvement helps users keep workspaces cleaner by syncing titles with actual project goals.
- **Chat Snapshot Survives Reload**: A pull request titled `feat: restoring project from snapshot on reload` from **thecodacus** introduced a snapshot system for chat history, shown [here](https://github.com/stackblitz-labs/bolt.diy/pull/444), letting users recover project state on reload.
   - It ensures continuity in user interactions and preserves associated file system data across sessions.
- **Git Support Meetings Approaching**: Office hours confirmed **Git support** may arrive in about **2-4 weeks**, raising hopes for robust version control in **Bolt**.
   - Community members anticipate smoother collaboration and code tracking once this feature is launched.
- **Token Tsunami Triggers Warnings**: A log showed **4 million tokens** consumed by a single command, causing alarm in the channel.
   - Participants called for deeper investigation to keep usage within practical limits and prevent further token blowouts.
- **Deployment Dilemma and Stripe Snafus**: Users faced headaches deploying large **Bolt** projects, prompting suggestions like moving assets to **Amazon S3**.
   - Meanwhile, **Stripe** integration queries lingered, as some encountered configuration obstacles during checkout flows.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Swarm Swoops Over A1111**: SWARM overshadowed A1111 in user adoption due to steady updates and extensive documentation, with many praising its specialized tasks.
   - Enthusiasts credited the developer's active engagement as a key advantage for this up-and-coming interface.
- **Suspicious Scam Shakes Stability**: A compromised Twitter handle for **@StabilityAI** posted fraudulent token announcements, prompting instant alarm.
   - Members shared a [Tweet from Dango233](https://x.com/dango233max/status/1879734940264481006) as evidence, recalling past scams that targeted unsuspecting followers.
- **Measuring the Muses**: Users weighed iterations-per-second metrics for **Stable Diffusion**, referencing [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for baseline performance.
   - They noted built-in timers and metadata logs in various UIs as helpful methods to assess image generation speed.
- **License Lore Lightens Load**: Participants clarified that **Stability AI’s community license** typically doesn't require formal attribution for noncommercial uses.
   - They acknowledged that advising credit can build goodwill, while commercial scenarios may require deeper licensing considerations.
- **Printing Potential Gains Steam**: A print-on-demand entrepreneur explored methods to upscale **Stable Diffusion** outputs for large-scale projects.
   - Guidance came through direct messages, highlighting high-resolution presets and customized workflows for business applications.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek's Dip & Sonnet's Shine**: Members observed **DeepSeek3**'s lag and rumored **500GB VRAM** requirement, referencing [a Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/1ho0w52/deepseek_does_not_need_5_hours_to_generate_1/) for conflicting details.
   - They shifted to **Sonnet** for better performance and considered **Hyperbolic** at **$0.25/mtok**, hinting a broader push for cost-friendly solutions.
- **MOE Minimizes GPU Drains**: Some users highlighted **MOE (Mixture of Experts)** for partial-weight loading, which cuts resource usage on large series runs by only activating needed experts.
   - They speculated that precise batching might push overall costs even lower, sparking excitement for more efficient workloads.
- **CEDARScript Convo in Aider**: A user showcased a [GitHub PR](https://github.com/CEDARScript/cedarscript-integration-aider) aiming to let **Aider** adopt **CEDARScript** as an editing format, with minimal overhead.
   - Discussions included whether merges would add tangible advantages, but no clear outcome emerged from these proposals.
- **Helicone's One-Line Observability**: **Helicone** introduced an [open source LLM observability tool](https://github.com/Helicone/helicone) promising cost tracking, **LLM security**, and request metrics with a single-line integration.
   - They recommended cloud hosting but also support local runs via **docker-compose**, offering caching and custom rate limits for performance.
- **Security Layers for Safer AI**: Some participants discussed implementing a **security filter** to block sensitive data before sending requests, emphasizing potential risk mitigation.
   - They pointed to prior resource leaks as cautionary tales, concluding that a dedicated safeguarding module might be essential for corporate contexts.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research Rolls Out Merch Funds**: Members clarified that **Nous Research** is a private organization, funded partly through merch sales and private equity, with minimal government or academic ties.
   - A few expressed interest in **stickers**, hinting at a modest but spirited approach to boosting revenue.
- **LLAMA 1B QLoRA Feels the Pressure**: Members reviewed **LLAMA 1B QLoRA** training charts, raising concerns about the small dataset size and limited training steps.
   - They debated the merits of calculating fitness scores versus simpler **performance metrics** when evaluating model outputs.
- **Optimizer Showdown: GrokAdamW, Ortho Grad, and GrokFast**: Participants compared **GrokAdamW** and **Ortho Grad**, noting GrokAdamW's improved loss metrics and GitHub references but possibly conflicting points from Ortho Grad.
   - GrokFast struggled with **stability**, driving interest toward **Orthograd** as a potential drop-in replacement for torch optimizers.
- **PRMs and Memorization Catch Attention**: Members dove into **Process Reward Models (PRMs)** for thorough supervision of intermediate steps, referencing the Qwen team's documentation.
   - They also touched on **LLM memorization** methods, citing [Anthropic's research](https://www.anthropic.com/research/mapping-mind-language-model) for deeper exploration.
- **Neural Long-Term Memory Aims for Balance**: A new paper introduced a **neural long-term memory** module for capturing historical context, linked via [arXiv](https://arxiv.org/abs/2501.00663v1).
   - It merges **recurrent models** with attention, promising quick training and inference while handling extended dependencies without hefty costs.



---



## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Digital Pathology with Groovy Gains**: One user overcame a tough search for Groovy scripts by using NotebookLM to handle **image annotations** in digital pathology, saving significant time on their project. They credited NotebookLM for swiftly parsing their requirements and producing a **functional script** for a tricky use case.
   - Others voiced their enthusiasm, calling it a *serious productivity bump*, and they recommended creating similar domain-specific scripts using NotebookLM for specialized workflows.
- **Interactive Mode Creates Classroom Buzz**: Members praised **Interactive Mode** in NotebookLM for quickly loading module resources and facilitating real-time exploration of academic content. The screenshot shared showed how prompting on course materials can spark new lesson strategies.
   - They also mentioned readiness for the upcoming semester with *excitement*, suggesting more educators could adopt this approach to streamline teaching.
- **Podcast Generation Puzzles**: Several members faced **podcast generation** troubles when pulling from multiple sources, eventually finding a workaround by separating sources into different notebooks. They noted that unchecking irrelevant sources lends better accuracy, but confusion remains over whether this is a **NotebookLM Plus** feature.
   - Community feedback highlights poor host interactions and lackluster audio quality, with discussions on potential instructions to produce a more coherent final file.
- **Workspace Woes and NotebookLM Licensing Clarified**: A wave of confusion arose about **NotebookLM Plus** in various Google Workspace plans, prompting clarifications that AI features like Gemini and NotebookLM Plus will remain included at no extra cost, as per [the official Workspace blog](https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI).
   - Community members referenced [Bora's Law](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with-constraints) to assert broader scaling strategies, while others confirmed older licenses wouldn't lose existing features.
- **Source Upload Struggles Dampen Efficiency**: NotebookLM currently has no bulk uploading option, baffling users who want to import numerous URLs quickly. They must manually add each source or rely on single-file uploads for now.
   - Some complained about the missing feature's impact on **multi-source** workflows, noting that a more integrated approach could drastically refine large-scale data ingestion.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Minimax’s Mighty 4M Context**: The newly available **Minimax-01** wowed folks by passing the **Needle-In-A-Haystack** test at **4M** context length, as shown on the [OpenRouter page](https://openrouter.ai/minimax/minimax-01).
   - Enthusiasts admired the attached image in the announcement, noting it hinted at potential multi-modal capabilities for **Minimax-01**.
- **DeepSeek Delays Disappoint**: Issues with **DeepSeek** included reports of unreliable service during busy periods, with many encountering API slowdowns.
   - Some community members shared troubleshooting tips like tweaking API settings and watching for provider errors to keep tasks moving.
- **OpenRouter’s Region Lock Ruckus**: It was confirmed that **OpenRouter** enforces regional restrictions following **OpenAI** and **Anthropic** policies, catching users by surprise.
   - Community chatter focused on navigating these limitations and sharing experiences with blocked regions.
- **Gemini Goes Off-Grid**: The **Gemini flash 2.0** model changed endpoints unexpectedly, causing confusion and errors for active users.
   - Affected folks swapped privacy settings workarounds, insisting an official fix or documentation is urgently needed.
- **Activity Page Puzzle**: Users noticed the **activity page** displaying identical graphs for different API keys, leading to confusion over usage data.
   - Debate sparked over the page’s design, with some requesting clearer separation of transactions to help track deployments accurately.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ Gains Multi-Language Edge**: Participants in the #discussions channel reported that **Command R+** covers multiple coding languages, such as **Python** and **JavaScript**, and can be tested via API.
   - One user recommended continuous updates akin to an 08-2024 release, cautioning that each new iteration essentially forms another model.
- **Stripe Steps In with Proxy Perks**: Attendees clarified that **Stripe** handles payment processing within Cohere’s platform, offering a straightforward upgrade path.
   - They explained that **OpenRouter** routes queries to all **Cohere models**, easing adoption for developers requiring unified access.
- **Rerank 3.5 Powers Code**: Members praised **Rerank 3.5** for its strength in coding tasks spanning **Python**, **JavaScript**, and **C++**, though some niche use cases remain unsupported.
   - They noted the model’s bias toward semantic matches when more documents are loaded, suggesting extra calibration for tighter accuracy.
- **Embeddings Hit a Wall**: Developers voiced frustration that updating **embedding models** requires re-embedding huge batches of data, with no migration path from older versions.
   - They emphasized this burden often leads to prolonged reliance on existing embeddings due to the overhead of reprocessing.
- **LLMU & Cookbooks for Deep Learning**: People highlighted **LLM University (LLMU)** as a free resource alongside cookbooks and **$75** in credits for new accounts, linked at [LLM University](https://cohere.com/llmu#text-representation).
   - They recommended these courses to jump-start generative AI experiments, describing them as a helpful on-ramp for beginners.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Goes Browser-Bound with JSPI**: Tinygrad can now run in the browser by enabling the **JSPI flag**, and it’s working on Mac, Ubuntu, and Windows, as seen in [this test page](https://mesozoic-egg.github.io/tinygrad/).
   - Users confirmed *'works on my M1 pro after enabling jspi flag'* and highlighted that broad compatibility is boosted by this new approach.
- **George Hotz’s Zany Cloud GPU Vision**: George Hotz proposed that all networked machines could operate like a single GPU, as stated in [this tweet](https://x.com/__tinygrad__/status/1879930546652156027).
   - He stressed *'there's a whole world of possibly above the current NVIDIA stack'*, suggesting future directions for parallel computing.
- **Conda Installation Snafu**: A user encountered an error with `libgcc_s.so` not being an ELF file when installing Tinygrad in a conda environment, referencing [this GitLab link](https://git.informatik.uni-hamburg.de/4kirsano/master-thesis/-/blob/main/.conda/lib/libgcc_s.so).
   - Switching to standard Python without venv resolved the issue, hinting that conda might override crucial system libraries.
- **TinyJit & Metal Tussle**: TinyJit ran slower on a 2019 MacBook Pro with Metal backend, traced to GPU synchronization bottlenecks.
   - Tweaks to JIT settings and disabling Metal graph on older Intel MacBook Pros saw some improvements, supported by debug logs.
- **Exported Models & Operator Fusion**: Tinygrad lets users pickle jitted models for quick reloads, echoing openpilot’s method of reusing compiled artifacts.
   - Community interest soared after a link on operator fusion was shared in [tinygrad-notes/20250117_fusion.md](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md), showcasing performance tweaks via fusion and un-fusion strategies.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **TITANS Tackle 'Human-Like' Memory**: A link to [Google Research's Transformers 2.0 aka **TITANS**](https://www.youtube.com/watch?v=x8jFFhCLDJY) was shared, asking if they've cracked **human-like** memory for LLMs.
   - Members wondered if this framework promotes more context-rich outputs, calling it *a strong leap in memory scaling*.
- **Omnimodal Overload: Delays & Doubts**: **OpenAI** and **Gemini** faced questions over postponed image-generation rollouts, creating uncertainty in the community.
   - Some users speculated that refined open-source audio models could emerge, but emotional output handling remains *a tricky element*.
- **PrivateGPT & Obsidian: A Knowledge Combo**: Members explored **PrivateGPT** tied to **Obsidian** notebooks, aiming to feed personal data into local AI workflows.
   - They discussed methods for smoother synergy between user-owned documents and model outputs, highlighting *powerful personal knowledge retrieval*.
- **Speedy Prompt Mastery in 30 Days**: A user proposed learning **prompt engineering** and authoring a book in just 30 days, leveraging [a shared resource](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb).
   - Others urged **self-discovery techniques** and additional web searches, insisting *skillful prompts* can accelerate writing.
- **GPT-4o Gains Canvas & Task Magic**: New **GPT-4o** tasks let users schedule reminders like 'Practice Spanish at 3pm', with ChatGPT pinging them on time.
   - Meanwhile, **Canvas** still exists behind a toolbox icon, although some encountered interface quirks in version history.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Bora’s Law Challenges Big AI**: A member referenced the working paper [Bora’s Law: Intelligence Scales With Constraints, Not Compute](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with), arguing that **established approaches** may be flawed.
   - They proposed that **intelligence** grows with well-defined constraints, drawing interest toward alternative *AI development* paths.
- **New 'Sonar' & 'Sonar-Pro' Spark Speculation**: A user discovered references to **sonar** and **sonar-pro** in labs, prompting questions about upcoming model expansions.
   - They shared an [image referencing these models](https://cdn.discordapp.com/attachments/1161802929053909012/1329550935038623877/image.png), fueling rumors of another potential **API** shift.
- **Claude Sonnet Stumbles on Code Tasks**: Several members reported **Claude Sonnet** faltering on CSV file processing requests, questioning its reliability for coding.
   - They recounted *ongoing conflicts* over incorrect suggestions, casting doubt on the AI’s consistency.
- **Image Generation Showdown**: The community debated **image outputs** from ChatGPT, Flux, Grok, and Perplexity, highlighting major quality differences.
   - One user declared 'it’s not even close' when comparing sunrise visuals, underscoring **Perplexity’s** relative weakness.
- **3D Printing with AI Tools Gains Momentum**: Members explored **AI-driven 3D object design**, showcasing interest in new ways to create mechanical parts and hobbyist toys.
   - They offered tips in a [discussion link](https://www.perplexity.ai/search/can-perplexity-ai-be-used-to-g-tp.qKJ1JQJaAAmpqQUb7Og#2), hinting at deeper synergy between **3D printing** and AI.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Cramming Tokens: The Context Window Conundrum**: One user questioned the 'context is 90.5% full' warning, prompting an explanation of the **Context Window** and how tokens accumulate as conversations grow.
   - Community members noted that adjusting the model's capacity is sometimes advisable to avoid partial truncation, with suggestions for bigger context settings in the future.
- **System RAM vs VRAM: The Great Debate**: A discussion clarified that CPU inference uses system memory while GPU-based setups rely on VRAM, with fallback to RAM if GPU resources run out.
   - Members recommended checking the [LM Studio site](https://lmstudio.ai/download) for hardware details, especially for M2 Mac owners who encountered caching problems.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Grapples With Movie Scripts**: One user attempted analyzing a **45-page screenplay** with GPT4All but discovered it only addresses single scenes, even though the model claims a **128KB** capacity.
   - They tested chunk-by-chunk approaches for broader analysis, with better results after adjusting workflow and reloading the app.
- **Ethical Boundaries: ChatGPT 4.0 vs Others**: Differences emerged between how **ChatGPT 4.0** and its alternate versions handle explicit content, highlighting distinct censorship policies.
   - Participants questioned whether these **ethical gates** limit user access to balanced data, with some calling for uniform guidelines.
- **DavidAU & Magnum Models for Dark Scenes**: Community suggestions favored **DavidAU's** models for edgy or non-dark writing, pointing to [huggingface.co/DavidAU](https://huggingface.co/DavidAU) for reference.
   - Others mentioned **Magnum** models and recommended specific VRAM setups to optimize performance for varied writing tasks.
- **Quantization & Model Management Tricks**: One user adjusted quantization settings found in the [Hugging Face docs](https://huggingface.co/docs/transformers/main/main_classes/quantization) to boost **Gemma** model speed on GPU.
   - They discovered that adding new models to GPT4All’s designated folder and restarting the app is essential, referencing a [Llama comparison chart](https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b) for guidance.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LeetGPU's Launch Lures CUDA Coders**: A new **LeetGPU** [online CUDA playground](https://leetgpu.com/) offers free GPU code execution with no signup, letting devs quickly test out **CUDA** routines in any environment.
   - The creators encouraged the community to share feedback, fueling interest among those seeking **collaborators** for GPU-related projects.
- **Torchinductor Tactics & Compile Confessions**: Community members highlighted a blog on **Torchinductor**, a *PyTorch-native compiler* that uses **define-by-run IR** and **symbolic shapes**, with references to [TorchDynamo](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) and how it speeds up dynamic Python code.
   - They also shared *Dissecting Torch Compile* from [this GitHub repo](https://github.com/DWarez/torch_compile_blogpost), underscoring the shift from **Caffe** to more user-friendly ML frameworks.
- **MI300X Memory Magic & MLPerf Mysteries**: Discussion touched on how dividing **MI300X** nodes into multiple shares can enhance memory performance by trimming load on **infinity cache**.
   - Another user wondered how **MLPerf** vendors run **GPT-3** benchmarks despite GPT-3 not being fully open-sourced, hinting at closed collaborations or partial access.
- **Flashing CUDA with Fast Attention**: A GitHub repo for **Flash Attention with CUDA** at [damienjose/cuda-flashattention](https://github.com/damienjose/cuda-flashattention) caught the group's eye, providing a reference for speed-boosting attention mechanisms.
   - Suggested usage includes *blockwise matmul* approaches for large-scale sequence tasks, opening the door for efficient tokens on GPU.
- **Arm64 Runners & Chats That Fix Failures**: **GitHub** rolled out free **Linux arm64 hosted runners** for public repos, broadening deployment options for those building on ARM hardware, as noted in [their Changelog entry](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/).
   - They also introduced a new **Copilot chat** feature that explains *Actions job failures* in real time, letting devs troubleshoot directly from the PR mergebox or job page.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Teacher-Model Distillation Gains Steam**: Members tested a teacher model that guides a smaller student, focusing on specialized data over broad coverage.
   - They debated if the student remains well-anchored in real usage when trained on narrower outputs.
- **Google's New Blueprint Outshines Transformers**: Google Research unveiled an approach that claims to surpass standard transformers in certain tasks, citing [this new paper](https://arxiv.org/abs/2501.00663).
   - The chat also explored potential links to **Gemini 1.5**, hinting that it might integrate features from the new design.
- **OpenAI Bends Data Use & Faces Cost Overload**: OpenAI now only trains on API data if users opt in, reacting to concerns about forced data usage.
   - Reports suggest they might spend **$4 billion** on Azure servers and **$3 billion** on training, raising questions about financial feasibility.
- **Tensor Product Attention Trims KV Cache Bloat**: A new paper proposes TPA to scale language models with smaller KV caches, referencing the [T6 implementation](https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64).
   - The authors plan a Flash approach for TPA, aiming for further speed gains in large-scale deployments.
- **Slimmer 4090 Cards Dodge Breakage**: Heavy 4090 GPUs can crack PCBs, sparking China-based efforts to repackage them into 2-slot variants.
   - One [eBay listing](https://www.ebay.com/itm/126885374543) for a dual-width 48GB RTX 4090 got **23** views in a day, illustrating the interest in these revised boards.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Chollet & Knoop Kick Off Ndea**: Francois Chollet partnered with Mike Knoop to launch [Ndea](https://x.com/fchollet/status/1879583863368032432), emphasizing **deep learning-guided program synthesis** to expand AI’s capabilities. Their approach spotlights adaptation and **invention** as cornerstones for advanced AI progress.
   - Observers noted that this direction could reshape how models handle code generation and creativity, with excitement building around potential breakthroughs in dynamic learning.
- **Curator’s Synthetic Data Surge**: The open-source [Curator](https://x.com/madiator/status/1879579213554147665) library promises a **10x** speed-up in **high-quality synthetic data** creation, vital for post-training datasets. Community members highlighted its practical benefit in generating robust sets for LLMs and specialized agents.
   - They also mentioned that efficient synthetic data pipelines might reduce time-consuming manual labeling, enabling faster experimentation with new model variants.
- **Titans Tackle Towering Context**: The **Titans** architecture offers a meta in-context memory that can adjust at test time, potentially exceeding **GPT-4** with a context limit above **2M**. This approach challenges standard attention mechanisms, suggesting a different route for handling massive sequences.
   - Attendees cited [Ali Behrouz](https://x.com/behrouz_ali/status/1878859086227255347) for raising questions about memory constraints and whether this design can outpace existing solutions in real-world tasks.
- **HAL Hits the Agent Scoreboard**: The [HAL](https://x.com/sayashk/status/1879932823668498576) project evaluates over **90 AI agents** on **11 benchmarks**, comparing reasoning-style models to standard language models. Enthusiasts stressed cost trade-offs and reliability, noting that big performance gains might come with a high price tag.
   - They also debated the credibility of agent evaluations and whether reasoning-driven approaches genuinely outperform simpler language models in everyday scenarios.
- **Harvey Hauls a Hefty $300M**: Legal startup **Harvey** is reportedly securing **$300M** at a **$3B** valuation, following July’s **$100M** raise at **$1.5B**. Chat focused on how their revenue of **$30M** could grow with this financial boost and spark faster AI deployment in law firms.
   - Speculation centered around the competitive market for AI-based legal services and whether Harvey’s aggressive funding strategy sets a precedent for other industry players.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular's Subreddit Soars**: There's now an **official Modular subreddit** at [r/ModularAI](https://www.reddit.com/r/ModularAI/), inviting the community to join.
   - One member exclaimed *“This is the way!”*, and others showed excitement as they gathered on this new platform.
- **GitHub Org Overhaul for Modular Repos**: **Modular** has shifted its public GitHub repos from [ModularML](https://github.com/modularml) to [Modular](https://github.com/modular), keeping all history intact.
   - They expect automatic redirects but encourage the community to report any **unexpected issues** they encounter.
- **Mojo's Monstrous Recursive Types**: A user reported challenges implementing recursive types in **Mojo**, noting pitfalls with `UnsafePointer` and incomplete official support.
   - They recommended a **copy constructor** on `List` to avoid crashes and referenced [Issue #3917](https://github.com/modularml/mojo/issues/3917) for related debug-level problems.
- **SIMD Surprises Spark Debates**: Developers discussed how **SIMD** doesn't always yield better speeds, referencing [Ice Lake AVX-512 Downclocking](https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html).
   - They cautioned that SIMD gains vary by CPU and can be a **footgun** if one expects a simple performance boost.
- **Optional Argument Oddities in Mojo**: An **optional argument** in Mojo caused segmentation faults when evaluating to None, documented in [Issue #3950](https://github.com/modularml/mojo/issues/3950).
   - Contributors recommended checking GitHub for example fixes while acknowledging the bug remains under investigation.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Hack for Identity: $5k Xeno Grant**: Plastic Labs and Betaworks kicked off an **Agent Identity Hackathon** with $5,000 in prizes, inviting teams to sign up at [Luma](https://lu.ma/5rlcrlpb).
   - They close applications on **January 26th**, urging participants to share GitHub links for vetting by the **grants committee**.
- **Model Bench Momentum**: LiveCodeBench added **167 new problems**—880 total—to showcase improved reasoning from models like **Gemini-Flash** and R1, as described in [this tweet](https://x.com/StringChaos/status/1879619028651745287).
   - SWE-bench also launched multimodal JavaScript bug evaluations, while **TGI** adopted multi-backend support for AMD and TPU detailed in [Hugging Face’s blog](https://huggingface.co/blog/tgi-multi-backend).
- **Cerebras Chips Challenge Conventions**: Cerebras argues their **wafer-scale chip** maintains yields on par with smaller designs, detailed in [their blog](https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem).
   - They compare faults to an **H100**-sized die, claiming robust fault tolerance offsets the massive 50x die area.
- **AMD’s Ai2 Dreams and Intel’s Contrasting Tactics**: Some propose **AMD** should give **Ai2** $10k each and leverage MI300X accelerators, as touted by [Tensorwave](https://tensorwave.com/) for *faster and easier* AI solutions.
   - Meanwhile, **Intel** sponsors Stability AI, fueling comparisons of GPU vendors angling for savvy alliances.
- **Humans, LLMs, and Meta’s Project Aria**: A **next best action system** can grant human operators an edge, with chatter about non-existent social movements against AI and skepticism over sudden tech shifts.
   - Simultaneously, **Meta** expanded [Project Aria](https://www.projectaria.com/) signups and clarified data usage, letting users unsubscribe from promotional emails anytime.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex links up with llmlingua2**: One user integrated **llmlingua2** into LlamaIndex, referencing a [PR on GitHub](https://github.com/run-llama/llama_index/pull/17531), but encountered linting issues with `make`.
   - Another user suggested installing **pre-commit** or running `make lint` to handle scripts quickly, underscoring synergy between LlamaIndex and **llmlingua2**.
- **Filtering Frenzy in ChromaDB**: A member explored **ExactMatchFilters** in **ChromaDB** to handle thousands of legal documents, unsure if sub-index routing is the best approach.
   - They expressed doubts about performance overhead and questioned if existing metadata filtering methods handle large-scale data more efficiently.
- **Neomagus nails LLM x Law Hackathon**: The team behind **Neomagus** triumphed at the law-focused hackathon with real-time verification, flagging incorrect references on the spot ([more details](https://t.co/jEqZrnn11H)).
   - Participants noted that improving the **accuracy of AI-generated legal information** drives trust in LLM-based solutions.
- **Women in AI RAG Hackathon heats up**: A **Women in AI RAG Hackathon** in Palo Alto was announced, focusing on **Retrieval-Augmented Generation** with [@zilliz_universe](https://t.co/2Bzg80dh29).
   - Organizers encouraged women technologists to attend for an all-day event, sharing [more info](https://t.co/ey8ebq9fbx) and offering strong mentorship opportunities.
- **Tag Extraction Tussle**: A user questioned whether **tag extraction** should be separated from product description tasks or combined, emphasizing cost and performance concerns.
   - They highlighted **latency** challenges and the potential difference in **tag quality** for repeated calls.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Lightning-Fast Text-to-SQL Setup**: A user built a **text-to-SQL pipeline** in just **20 minutes**, remarking on how quick and simple the setup felt.
   - They emphasized its **user-friendly nature** and noted a worthwhile lesson for future AI-based data queries.
- **Speculation on DSPy V3 Release**: A question arose regarding when **DSPy v3** might arrive, reflecting curiosity about potential new features.
   - No formal announcement was cited, leaving the community waiting for more information.
- **dspy ReAct Tool and Addition Function Woes**: A user encountered an error in **dspy ReAct**, which flagged the addition tool as unable to calculate two numbers due to missing arguments.
   - Further issues included a syntax hiccup where 'retur' replaced 'return', causing incorrect output when using **LM-Studio** with the **addition function**.



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Chat Template Tangle**: The group debated how to craft the **ideal chat template**, flirting with ChatML or **Llama3** as possible routes.
   - They aim for minimal overhead but demand a consistent format, prompting pressure to establish clearer guidelines.
- **Torchtune Tussle**: A member revealed that integrating **Torchtune** requires *ripping out a lot of things*, hinting at big code adjustments.
   - **caseus_** joked about the stalled progress, pointing to a lull in bandwidth for hooking it up smoothly.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Cooperative AI Summer School Kicks Off**: Applications for the **Cooperative AI Summer School** remain open until **7th March 2025**, with the event from **9th–13th July 2025** in Marlow, near London.
   - Confirmed speakers include **Michael Wellman**, **Zarinah Agnew**, and **Ariel Procaccia**, covering advanced research in cooperative AI with [financial assistance details provided](https://www.cooperativeai.com/summer-school/summer-school-2025).
- **Cost Controls Steer Technology Choices**: Participants emphasized that **cost** drives decisions to maintain tried-and-true solutions for MLOps workflows.
   - Budgets strongly influence teams to pick or stick with stable tech to ensure practicality.
- **Churn Prevention Approaches Spark Interest**: A user returning after two years asked about fresh tactics in **churn aversion** and how to start learning the current tools.
   - Others noted the significance of modern frameworks and real-world examples to reduce user drop-off in evolving markets.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Bora's Law Reframes AGI Growth**: A member criticized OpenAI's approach to **AGI**, emphasizing **Bora's Law** that *intelligence scales with constraints, not compute* and referencing [this piece by Chris Bora](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with).
   - They claimed brute force scaling ignores the essential role of constraints, suggesting that focusing on **constraint-driven math** is key to achieving genuine intelligence.
- **Open Interpreter's Code Execution Tweak**: Enthusiasts noticed that **Open Interpreter 1.0** restricted its direct code execution features to **command line** operations, leading to concerns about reduced efficiency.
   - Others called for restoring that functionality and adding **Python convenience functions** to help LLMs learn effectively, viewing the limitations as a significant downgrade.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Jolt vs OpenAI**: One user integrated **Jamba API** into multiple back-end services, prompting speculation it could surpass **OpenAI** responses.
   - They noted this raises questions about **OpenAI**’s standing, spurring comparisons of speed and effectiveness in real-world applications.
- **Community Cheers for Jamba**: Other users expressed appreciation for the positive remarks around **Jamba API**, affirming a supportive audience.
   - This feedback highlights growing interest in **Jamba** as a capable alternative to **OpenAI** for day-to-day usage.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LAION Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Mozilla AI Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **HuggingFace Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1329178802613653637)** (450 messages🔥🔥🔥): 

> `Cursor performance issues, New funding announcement, User experiences and productivity, Comparison with other tools, Python environment issues` 


- **Cursor performance issues persist**: Many users reported ongoing frustrations with Cursor's slow requests and failure to execute simple tasks, likening their experience to being in a 'loop of doom'. Users suggested various troubleshooting steps but continued to struggle with efficiency.
   - The performance issues have led some to explore alternatives like Windsurf, though users expressed that Cursor's capabilities still made it a preferred choice.
- **Cursor secures $105 million in Series B funding**: Cursor announced it raised $105 million in Series B funding from notable investors, implying potential for growth and improvement in functionality.
   - Users expressed hope that this funding would enhance Cursor's features without compromising the quality of service.
- **User experiences highlight productivity boosts**: Several users reported significant productivity gains while using Cursor compared to their previous programming experiences, noting its advanced autocomplete and efficiency.
   - Despite some complaints about slow requests, users felt that Cursor's predictive capabilities made it superior to other tools in the market.
- **Comparison between Cursor and Windsurf**: Users compared Cursor favorably against Windsurf, highlighting Cursor's functionality, price-value ratio, and overall performance.
   - With Cursor being seen as a clear winner, discussions emphasized how its features outclass those of Windsurf, even amidst some current technical challenges.
- **Python environment confusion**: A user raised issues regarding Cursor using a specific project's Python environment globally rather than a default one, affecting their workflow.
   - This prompted discussions around project setups, highlighting the need for better management of Python environments within Cursor.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.radix-ui.com/colors/custom">no title found</a>: no description found</li><li><a href="https://x.com/cursor_ai/status/1880003590493991072?s=46&t=kUuVqsG2GMX14zvB592G5w">Tweet from Cursor (@cursor_ai)</a>: We&#39;ve raised $105m in Series B funding from Thrive, Andreessen Horowitz, Benchmark, and existing investors.  We&#39;re delighted to report that Cursor is now used by millions of engineers as their...</li><li><a href="https://huggingface.co/omkarthawakar/LlamaV-o1">omkarthawakar/LlamaV-o1 · Hugging Face</a>: no description found</li><li><a href="https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475">Facepalm Really GIF - Facepalm Really Stressed - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.refactoringui.com/">Refactoring UI</a>: no description found</li><li><a href="https://v0.dev">v0 by Vercel</a>: Chat with v0. Generate UI with simple text prompts. Copy, paste, ship.</li><li><a href="https://vercel.com/templates">Dashboard</a>: no description found</li><li><a href="https://x.com/alexalbert__/status/1879917906294870196?s=46">Tweet from Alex Albert (@alexalbert__)</a>: Quality-of-life upgrade for @AnthropicAI devs:We&#39;ve adjusted prompt caching so that you now only need to specify cache write points in your prompts - we&#39;ll automatically check for cache hits a...</li><li><a href="https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/">Set up your first tunnel · Cloudflare Zero Trust docs</a>: To create and manage tunnels, you will need to install and authenticate cloudflared on your origin server. cloudflared is what connects your server to Cloudflare's global network.</li><li><a href="https://www.youtube.com/watch?v=FvLYLQKn2GM">Jessica Sachs | The Magic of Vite on Mobile | ViteConf 2023</a>: Looking to leverage Vite&#39;s blazing fast dev server and rich ecosystem when building your next mobile app? You can combine the power of Vite and Ionic Capacit...</li><li><a href="https://cursor.directory">Cursor Directory</a>: Find the best cursor rules for your framework and language
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1329182234560823438)** (241 messages🔥🔥): 

> `MCP Tool Discovery, Open-Swarm Framework, Semantic Tool Selection, Dynamic Tool Updates, Home Assistant Integration` 


- **MCP Dynamic Tool Discovery**: Dynamic tool discovery in MCP allows clients to list available tools and receive notifications when tools change, which helps maintain up-to-date functionalities without requiring restarts.
   - *This can be particularly useful for APIs that frequently update their tool signatures, minimizing disruptions for users.*
- **Open-Swarm Framework Enhancements**: The Open-Swarm framework aims to provide a drop-in replacement for OpenAI's original swarm framework, with enhancements for user-friendly interactions and native tool support.
   - Utilizing agents with clearly defined roles allows for a more efficient handling of tasks, such as database queries and web interactions.
- **Implementing Semantic Tool Selection**: Semantic tool selection involves using embeddings to represent tools in vector space, enabling more intelligent tool selection based on task relevance and user context.
   - This approach could potentially enhance tool utilization efficiency and reduce the costs associated with API access.
- **Integration with Home Assistant**: Integrating MCP with Home Assistant allows for enhanced automation capabilities, such as location-based reminders that trigger actions depending on user presence.
   - This integration exemplifies how smart home technologies can facilitate personalized task management through automation.
- **Clarifying MCP Terminology**: Users expressed confusion regarding MCP terminology, particularly around the MCP Bridge and its functionality, suggesting that documentation could be improved.
   - Understanding the relationship between tools, client capabilities, and the underlying protocol is crucial for effective implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.mcp.run/tasks/using-tasks">Working with Tasks | 🤖</a>: Tasks let you register prompts with a suite of installed servlets and trigger</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/tool-use#tool-use-system-prompt)">Tool use (function calling) - Anthropic</a>: no description found</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/tokenizer_config.json#L198>">tokenizer_config.json · Qwen/Qwen2.5-32B-Instruct at 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd</a>: no description found</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge/blob/master/docs%2Fusecases.md">MCP-Bridge/docs/usecases.md at master · SecretiveShell/MCP-Bridge</a>: A middleware to provide an openAI compatible endpoint that can call MCP tools - SecretiveShell/MCP-Bridge</li><li><a href="https://github.com/openai/swarm/blob/main/swarm/core.py">swarm/swarm/core.py at main · openai/swarm</a>: Educational framework exploring ergonomic, lightweight multi-agent orchestration. Managed by OpenAI Solution team. - openai/swarm</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/4c71c6168fb70c70cd1c7e358e78b664a794210c/src/mcp/types.py#L759>">python-sdk/src/mcp/types.py at 4c71c6168fb70c70cd1c7e358e78b664a794210c · modelcontextprotocol/python-sdk</a>: The official Python SDK for Model Context Protocol servers and clients - modelcontextprotocol/python-sdk
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1329219357204811847)** (24 messages🔥): 

> `MCP-Bridge, SSE Support, Open Source Client Improvements, Open Strategy Partners Tools, Discord Functionality` 


- **MCP-Bridge adds sampling support**: The [MCP-Bridge](https://github.com/SecretiveShell/MCP-Bridge) now features sampling support, allowing **OpenAI chat completions** to integrate with MCP servers seamlessly.
   - This enhancement lets developers use sampling in clients that previously did not support it.
- **Open Strategy Partners unveils marketing tools**: Open Strategy Partners released [osp_marketing_tools](https://github.com/open-strategy-partners/osp_marketing_tools), a python-based MCP server enhancing LLMs' capabilities in product marketing.
   - This tool aids with tasks like value mapping and writing style checks to streamline marketing efforts.
- **SSE support discussed for Sage client**: Members expressed excitement about **SSE support** coming to the Sage client, which may enhance its capabilities.
   - Discussion centered around customizing request bodies and integrating features effectively.
- **Smithery adds cloud hosting options**: For STDIO servers, Smithery introduced a cloud hosting option that utilizes SSE, employing config data in JSON format.
   - Members showed interest in exploring this feature, noting its potential benefits for server management.
- **Frustrations with Discord bot functionality**: There were complaints about the current Discord bot being inadequate, with members suggesting they could collaborate to create their own.
   - Additionally, it was pointed out that modern Discord includes built-in commands such as **/ban** to enhance user control.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/open-strategy-partners/osp_marketing_tools">GitHub - open-strategy-partners/osp_marketing_tools: A Model Context Protocol (MCP) server that empowers LLMs to use some of Open Srategy Partners&#39; core writing and product marketing techniques.</a>: A Model Context Protocol (MCP) server that empowers LLMs to use some of Open Srategy Partners&#39; core writing and product marketing techniques. - open-strategy-partners/osp_marketing_tools</li><li><a href="https://github.com/SecretiveShell/MCP-Bridge">GitHub - SecretiveShell/MCP-Bridge: A middleware to provide an openAI compatible endpoint that can call MCP tools</a>: A middleware to provide an openAI compatible endpoint that can call MCP tools - SecretiveShell/MCP-Bridge
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1329220375556395088)** (1 messages): 

> `Student Discount Pricing, Windsurf Editor Launch, Codeium vs GitHub Copilot, Enterprise Plan, Training Data` 


- **Students Save Big with Discount Pricing**: Codeium has launched **Student Discount Pricing** for users with active **.edu** addresses, giving them a significant discount on the Pro Tier Windsurf.
   - Students can [sign up here](https://www.codeium.com) to take advantage of this limited-time offer.
- **Introducing the Windsurf Editor**: Codeium unveiled the **Windsurf Editor**, a new purpose-built IDE designed for seamless coding experiences.
   - The launch emphasizes advanced features that cater specifically to the needs of developers.
- **Codeium Outshines GitHub Copilot**: Codeium confidently boasts that it is the most intelligent AI code generation tool, providing data to support its claims in comparison to **GitHub Copilot**.
   - Users can [read more about the performance quality comparison](https://www.codeium.com/compare) to see how Codeium stacks up.
- **Unlocking Potentials with Enterprise Plan**: Codeium promotes its **Enterprise Plan**, aimed at delivering high-quality and secure AI tools for faster engineering delivery.
   - This plan offers flexible deployments and self-hosting options, ensuring tailored solutions for businesses.
- **Protecting Users from Legal Risks**: Codeium emphasizes that it does not train on non-permissive code, such as **GPL**, thus safeguarding users against potential legal issues.
   - Further details can be found in their [blog post](https://www.codeium.com/blog/copilot-trains-on-gpl-codeium-does-not) discussing this critical distinction.



**Link mentioned**: <a href="https://www.codeium.com">Windsurf Editor and Codeium extensions</a>: Codeium is the AI code assistant platform that developers love and enterprises trust. Also the builders of Windsurf, the first agentic IDE.

  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1329182230995537990)** (102 messages🔥🔥): 

> `Student Discounts, Customer Service Issues, Windsurf Updates, VSCode and IDE Preferences, Service Outages` 


- **Clarification on Student Discounts**: Members questioned whether student emails not ending with **.edu** (like **.ac.uk** or **.unina.it**) could be used for discounts, with responses suggesting current offers are primarily for **.edu** domains.
   - Suggestions to contact support for clarification were made as they are working on expanding eligibility beyond the **.edu** constraint.
- **Complaints About Customer Service**: A user expressed frustration with Codeium’s customer service regarding a **$297 refund** request, stating that their account was mistakenly canceled instead of addressing the refund.
   - The user indicated dissatisfaction with the support team, while another member noted that the IDE is excellent despite the service complaints.
- **Windsurf and IDE Preferences**: Some users discussed their preference for using **Windsurf** over other IDEs like **VSCode**, while others expressed concerns about service delivery and support.
   - A comment highlighted that even when the IDE performs well, the support and customer service experience can drastically affect overall satisfaction.
- **Service Outages and Issues**: Several members reported experiencing issues with **autocomplete** features, with some expressing the need for a **status page** to provide updates on outages.
   - Threads indicated that queries regarding service interruptions were common, with users wanting more clarity on the platform's operational status.
- **License Issues and Account Management**: There was a discussion around the process of switching from a regular plan to a **student discount plan**, with some members uncertain about how their current subscriptions would be managed post-graduation.
   - Clarification was sought on whether existing subscribers could retain their pricing despite changes in their student status.



**Link mentioned**: <a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.

  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1329198537245851670)** (157 messages🔥🔥): 

> `Windsurf functionality issues, Student discount concerns, DeepSeek performance, Feedback on feature requests, Cascade prompt tips` 


- **Windsurf struggles with functionality**: Users reported various **functionality issues** with Windsurf, including infinite loops when using Cline with DeepSeek and problems with internal errors during code edits.
   - Some noted that though Windsurf performs well in some tasks, it fails to deliver consistently, leading to frustration.
- **Student discount eligibility confusion**: Several users expressed confusion regarding the **student discount eligibility**, particularly for .edu email addresses from non-US institutions.
   - There are ongoing discussions about expanding the program to include more countries, but many international students are currently unable to access the discount.
- **DeepSeek's inadequate support**: Users expressed dissatisfaction with DeepSeek's capability, with comments stating it often leads to loops and is not practical for immediate use.
   - Others suggested that while the benchmarks of DeepSeek are impressive, its integration with specific applications is currently lacking.
- **Requests for basic features ignored**: A user pointed out that simple feature requests, such as drag-and-drop support for images and files, have been overlooked for months.
   - There is a general sentiment of frustration among users feeling that their requests are often ignored despite being submitted through the proper channels.
- **Protips for using Cascade effectively**: Users are seeking **protips for Cascade**, particularly regarding prompt crafting for better output and utilization of its capabilities.
   - Suggestions include utilizing inline commands to avoid credit depletion, fostering creativity in prompt generation, and sharing successful prompts among community members.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/text-phone-waiting-hurry-messenger-gif-4073783462256955308">Text Phone GIF - Text Phone Waiting - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/curse-aukerman-comedy-fist-in-air-shaking-gif-5620082">Curse Aukerman GIF - Curse Aukerman Comedy - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>: Contact the Codeium team for support and to learn more about our enterprise offering.</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: Need help? Contact our support team for personalized assistance.
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1329187352442179605)** (171 messages🔥🔥): 

> `Phi-4 Model Fine-tuning, Dynamic Quantization Comparison, Ollama Chat Template Issues, Saving Merged Models, CPT Training Results` 


- **Fine-tuning and Inference with Phi-4**: A user successfully fine-tuned the Phi-4 model on a small dataset, saving it on a free Colab instance with 16GB VRAM. However, saving as a merged model posed issues, particularly related to out-of-memory (OOM) errors.
- **Challenges with Ollama Chat Templates**: A user reported endless generation issues when using an Ollama server, potentially due to an incorrect chat template. Suggestions were made to ensure the use of a proper template from unsloth training.
- **Dynamic Quantization Vs. GGUFs**: Questions arose regarding the comparison of dynamic 4-bit compression accuracy against GGUFs like 4-k-m, which wasn't included in recent discussions. The dynamic quantizations were emphasized as being more suited for fine-tuning and serving purposes rather than local running.
- **Health of Training Sessions**: Users shared their experiences with ongoing training sessions for Phi-4, with one starting new training with adjusted LORA parameters. The importance of properly saving merged models was a recurring concern.
- **Collaboration on Wikipedia Datasets**: A user created a repository for Wikipedia datasets on Hugging Face, inviting others to request languages and contribute to continuous learning. The team expressed enthusiasm for sharing resources for further training.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/vision">Llama 3.2 Vision Fine-tuning with Unsloth</a>: Fine-tune Meta&#x27;s Llama 3.2 Vision, Llava, Qwen 2.5 Vision models open-source 2x faster via Unsloth! Beginner friendly.</li><li><a href="https://huggingface.co/OuteAI/OuteTTS-0.3-500M">OuteAI/OuteTTS-0.3-500M · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=LPZh9BOjkQs">Large Language Models explained briefly</a>: Dig deeper here: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3piTechnical details as a talk: https://youtu.be/KJtZARuO3JYThis was ma...</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: Learn all the essentials of creating a dataset for fine-tuning!</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct">unsloth/Llama-3.2-3B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/burgasdotpro/bgGPT-Phi-4">burgasdotpro/bgGPT-Phi-4 · Hugging Face</a>: no description found</li><li><a href="https://x.com/UnslothAI/status/1879942441538609583">Tweet from Unsloth AI (@UnslothAI)</a>: You can finetune Phi-4 for free on @Kaggle now!You&#39;ll learn how to:• Prepare your dataset• Train Phi-4 via Kaggle&#39;s free GPUs• Run, evaluate & save your modelUnsloth finetunes LLMs 2x faster w...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3.14 minutes</a>: NanoGPT (124M) in 3.14 minutes. Contribute to KellerJordan/modded-nanogpt development by creating an account on GitHub.</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: Below is a list of all our notebooks:</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/is-fine-tuning-right-for-me">Is Fine-tuning Right For Me? | Unsloth Documentation</a>: If you&#x27;re stuck on if fine-tuning is right for you, see here!</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct">meta-llama/Llama-3.2-3B-Instruct · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1329410421072990249)** (1 messages): 

> `Onnx, TensorRT output differences` 


- **Onnx vs TensorRT: Notable Output Discrepancies**: A member raised a concern about their model's output from **Onnx** being significantly different from **TensorRT**.
   - They queried if others had encountered similar issues regarding output consistency between these two frameworks.
- **Possible Insights on Framework Variations**: No additional insights were shared, but the inquiry reflects common challenges in model deployment across **Onnx** and **TensorRT**.
   - The community may explore potential reasons for the observed inconsistencies, looking into model optimization or conversion issues.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1329184310422409257)** (63 messages🔥🔥): 

> `Flash Attention 2 installation issues, Training Phi-4 on Colab, Model serving using VLLM, Endless generation in Phi-4, Data packing and performance concerns` 


- **Flash Attention 2 installation issues arise**: A user reported an issue with the **Flash Attention 2** installation being broken, which they needed to resolve for speed comparisons during testing.
   - In response, another member suggested they reach out for help directly if they needed tests with Colab notebooks.
- **Training Phi-4 on free Colab is possible**: A user asked whether a premium Colab account is required for training models like **Phi-4** on a dataset of riddle questions, and another confirmed successful training happened on a free account.
   - However, they noted challenges with limited free space to save the complete model and its configurations.
- **Challenges with serving models in VLLM**: A user shared their experience saving **LoRA parameters** of the **QwenVL2-7B** model and expressed difficulties with serving the model using **VLLM**.
   - They found the documentation insufficiently detailed, causing troubleshooting challenges.
- **Phi-4 endlessly generates responses**: A user experienced an issue with **Phi-4** generating responses indefinitely without producing the end-of-sequence token, signaling a potential bug.
   - They were using llama.cpp for inference and observed that generation continued without stopping.
- **Concerns on data packing during training**: One user indicated that using packing with a custom dataset reduced performance and questioned if fewer training steps with a higher batch size would be less effective.
   - Responses highlighted that while higher batch sizes may stabilize loss, the effectiveness still depends on various factors, including potential data contamination in packing.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/unsloth/qwen-25-coder-6732bc833ed65dd1964994d4">Qwen 2.5 Coder - a unsloth Collection</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">Installing + Updating | Unsloth Documentation</a>: Learn to install Unsloth locally or online.</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-upd">Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1329186825071493150)** (7 messages): 

> `Transition between memorizing and grokking, Grokking phenomenon, Unsloth training techniques, LORA for knowledge distillation` 


- **Studying the Memorizing to Grokking Transition**: A member expressed hope that researchers are examining the **transition between memorizing and grokking**, suggesting it could reveal insights into training methods akin to biological neurons.
   - *This technique might hold secrets* for a new approach to training AI models.
- **YouTube on Grokking in LLMs**: A member shared a [YouTube video](https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ) titled *Activate GROKKING NOW - Performance Phase of LLMs (II)*, discussing **grokking** as a phenomenon in LLMs.
   - The video elaborates on how this sudden generalization occurs after prolonged overfitting in models.
- **Unsloth's Train_on_Completions Method Explained**: A member queried whether **Unsloth's train_on_completions** method, which trains only on assistant outputs, is equivalent to **response-based distillation** using hard targets.
   - It highlights an interest in understanding the implications of training techniques for model performance.
- **Using LORA for Knowledge Distillation**: Another member asked if **LORA** can be utilized for knowledge distillation during full fine-tuning processes, prompting curiosity among the participants.
   - A community member responded with uncertainty, suggesting that it is a possibility but needing clarification.



**Link mentioned**: <a href="https://youtu.be/H3OofROzlA0?si=e9b_lK1592TxgJcQ">Activate GROKKING NOW - Performance Phase of LLMs (II)</a>: Grokking, or the sudden generalization by AI models to new knowledge - that occurs after prolonged overfitting in LLMs, is a surprising phenomenon that has c...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1329443821574033439)** (125 messages🔥🔥): 

> `LLM API for Batch Text Continuations, Training LLMs and Token Prediction, Generative Models and VAEs, Attention Mechanisms vs. MLP, Exploration of LSTMs in Modern AI` 


- **Seeking LLM Backends for Batch Outputs**: A member inquired about LLM backends that offer API endpoints capable of producing batch text continuations, noting that **llama.cpp** only supports single continuations.
   - Another member suggested using **vllm** as a potential solution.
- **Token-by-Token Training Discussion**: The conversation delved into the intricacies of LLM training, specifically how models predict the next token while being trained on probabilities rather than fixed outputs.
   - Members emphasized that through this method, models learn to generalize across diverse responses rather than memorize specific sequences.
- **Mixed Opinions on VAEs for Generative Models**: One member advocated for starting with Variational Autoencoders (VAEs) as an approachable entry point into generative models, emphasizing their simplicity and lightweight nature.
   - However, they suggested that foundational knowledge of probabilistic graphical models could be even more beneficial when studying generative architectures.
- **Transformers vs. MLP Mixers**: The discussion touched upon the simplicity of transformer architectures, likening them to a breakdown of complex problems into simpler components, likened to 'sandwiches'.
   - A member humorously mentioned their commitment to using **MLP mixers** despite acknowledging that attention mechanisms provide powerful advantages.
- **Revisiting LSTMs in Contemporary AI**: A member expressed curiosity about the current use of LSTMs, suggesting that they feel outdated compared to transformers, which have faster training speeds.
   - Others considered potential applications for LSTMs, especially in contexts like stateful video world models, indicating they may still have relevance in specific scenarios.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1329262512524365824)** (65 messages🔥🔥): 

> `Modded NanoGPT Speedrun Record, TruthfulQA Dataset Exploitation, BERT vs. GPT for Classification, Hybrid Attention Models, Euler-Lagrange Equations in Neural Networks` 


- **Modded NanoGPT Speedrun Record set at 3.17 minutes**: @fern.bear announced a new record of **3.17 minutes** for a modded NanoGPT speedrun, improving upon the previous record of **3.33 minutes**. A detailed discussion about various tricks and improvements is available [in the thread](https://x.com/hi_tysam/status/1879687807678959729).
   - Members also discussed upcoming improvements including techniques like **Long-Short Sliding Window Attention**, suggesting further enhancements in context handling.
- **Weaknesses in TruthfulQA Dataset Exposed**: It was reported how members managed to exploit flaws in the **TruthfulQA** dataset, achieving up to **79% accuracy** using a few simple strategies. This incident highlights the need for critical examination of benchmarks, as discussed in a detailed post linked [here](https://turntrout.com/original-truthfulqa-weaknesses).
   - Discussion also veered into issues with **Halueval**, indicating that human annotations in common datasets often suffer from inaccuracies.
- **BERT's Bidirectional vs GPT's Causal Attention**: @kaltcit raised a question about the implications of bidirectional attention in BERT for tasks outside of masked language modeling. It was noted that while **BERT's** attention allows for more powerful representation, it cannot be used for text generation, limiting its applicability.
   - Members discussed how **GPT's** unidirectional attention may lead to different performance outcomes, particularly for tasks requiring broader context.
- **Hybrid Attention Models Might Reign Supreme**: Amid discussion on attention mechanisms, members reflected on **hybrid models** combining strategies like sliding window and full attention, noting their superiority in contexts longer than 1M. The consensus emerged that hybrids effectively balance speed and context retention over simpler architectures.
   - Participants voiced skepticism about solely relying on sliding window mechanisms, suggesting that proper mixtures offer better performance in long context tasks.
- **Ensuring Neural Networks Follow Euler-Lagrange Equations**: The conversation touched on the challenge of ensuring neural networks adhere to **Euler-Lagrange equations** without higher-order autodiff methods, with ideas about implementing architectures that output integral forms. Suggestions included leveraging model outputs to ensure curl-free conditions to represent scalar potentials correctly.
   - An innovative approach was proposed for models to output their derivatives, establishing a framework for analytically ensuring compliance with physical law constraints.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sakana.ai/transformer-squared/">no title found</a>: no description found</li><li><a href="https://arxiv.org/abs/2411.04434">Scaling Laws for Pre-training Agents and World Models</a>: The performance of embodied agents has been shown to improve by increasing model parameters, dataset size, and compute. This has been demonstrated in domains from robotics to video games, when generat...</li><li><a href="https://x.com/Turn_Trout/status/1879710659904254081">Tweet from Alex Turner (@Turn_Trout)</a>: Mark Kurzeja & I exploited weaknesses in multiple-choice TruthfulQA dataset while hiding the questions! A few simple rules of thumb achieved 79% accuracy.Even well-regarded benchmarks can have flaws. ...</li><li><a href="https://x.com/hi_tysam/status/1879687807678959729">Tweet from Fern (@hi_tysam)</a>: New NanoGPT training speed record: 3.28 FineWeb val loss in 3.17 minutes on 8xH100Previous record (recreation): 3.32 minutesLots of changes!- New token-dependent lm_head bias- Fused several ops- Multi...</li><li><a href="https://github.com/KellerJordan/modded-nanogpt/pull/71">Long-Short Sliding Window Attention (3.2 sec or 0.053 mins improvement) by leloykun · Pull Request #71 · KellerJordan/modded-nanogpt</a>: Currently, we warmup the context length of the sliding window attention at the same rate in all layers. This attempt warms up the context length differently in some layers instead. This leads to a ...</li><li><a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>: Training Large Language Model to Reason in a Continuous Latent Space - facebookresearch/coconut</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/6d62a69cb5db963f998c486af6efee43fca63dd3/docs/task_guide.md?plain=1#L57>),">lm-evaluation-harness/docs/task_guide.md at 6d62a69cb5db963f998c486af6efee43fca63dd3 · EleutherAI/lm-evaluation-harness</a>: A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness</li><li><a href="https://www.nature.com/articles/s41467-024-55628-6">Large Language Models lack essential metacognition for reliable medical reasoning - Nature Communications</a>: Large Language Models demonstrate expert-level accuracy in medical exams, supporting their potential inclusion in healthcare settings. Here, authors reveal that their metacognitive abilities are under...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1329502969158434826)** (16 messages🔥): 

> `PDE Foundation Models, Scaling Laws in Models, Implicit vs Explicit Solvers, Model Output vs Training Data` 


- **PDE Models and Scaling Laws Explained**: Members discussed why **PDE foundation models** have similar scaling laws to **LLMs**, linking it to memorization baselines but questioning how it applies to **PDE** models.
   - *One member suggested that PDE models should be directly 'grokking' system dynamics instead of relying solely on the training data's power law probabilities.*
- **Implicit Solvers Might Not Be Crucial**: The conversation shifted to whether a model learning an **implicit solver** is necessary, with some members expressing skepticism about its importance.
   - *There was concern that the model may not be grasping system dynamics thoroughly, casting doubt on learning mechanisms.*
- **The Role of Timestep Stability in Solvers**: It was highlighted that with **explicit solvers**, stability and energy conservation depend on the 'speed of sound' of the solution and length scales constraining the timestep.
   - *In contrast, implicit solvers do not share these restrictions, leading to differing implications on model learning.*
- **Model Output May Surpass Training Data Accuracy**: A member noted that the model's output may align more closely with *ground truth* than the training data used to generate it, prompting curiosity on how this occurs.
   - *Using low-resolution data might allow the model to estimate an average that is a better approximation of the truth, although the paper lacked clarity on data generation methods.*


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1329253449040134174)** (8 messages🔥): 

> `MATH DMCA Takedown, MATH Dataset Impact, YAML Quickstarter Update` 


- **MATH Dataset hit with DMCA Takedown**: **Hendrycks MATH** has received a DMCA takedown notice, resulting in the dataset being disabled, as reported in a [tweet](https://x.com/tmkadamcz/status/1879584048429105238). The discussion centers on the implications of this action and the source of the dataset, credited to **aops**.
   - *“They have always disclosed they got the questions from there.”*
- **Concerns about MATH Dataset Loss**: Members expressed that the loss of the **MATH** dataset could heavily impact the community, potentially more than other notable datasets like **The Pile** or **Books 3**.
   - *“I’d argue even more so than the Pile, Books 3 or Book Corpus.”*
- **Proposed YAML Quickstarter Update**: A member suggested updating the blank YAML into a **quickstarter YAML**, based on the documentation, and received positive feedback regarding its utility.
   - *“That would be quite helpful actually!”*
- **Discussions on Fair Use and DMCA**: Some members debated whether the team would attempt to challenge the DMCA notice under **fair use**.
   - One member expressed skepticism that **Hugging Face** could claim fair use, emphasizing their role as a distributor.
- **Git Repository Status**: Despite the DMCA notice, the **MATH** git repository and the link to the **tar file** are still accessible.
   - This raises questions around the ongoing availability of resources even amidst legal challenges.



**Link mentioned**: <a href="https://x.com/tmkadamcz/status/1879584048429105238">Tweet from Tom Adamczewski (@tmkadamcz)</a>: Hendrycks MATH has just been hit with a DMCA takedown notice. The dataset is currently disabled.https://huggingface.co/datasets/hendrycks/competition_math/discussions/5

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1329202599164907570)** (2 messages): 

> `Zero stages in Deepspeed, Model Parallelism challenges, Performance optimization for 30b model, Amd MI250x GPUs performance` 


- **Zero stages incompatibility issue**: A member inquired why **zero stages 2 and 3** are incompatible with both the model and pipeline parallelism in **Deepspeed**, suggesting that turning off only pipeline parallelism should suffice, as stated in the [training.py code](https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973).
   - *They expressed concern that the inability to use model parallelism renders Deepspeed ineffective for large model training.*
- **Struggles with 30b model training performance**: The same member shared difficulties in maximizing performance for **30b model training** on **512 AMD MI250x GPUs**, aiming to utilize **Deepspeed stage 2** along with model parallelism.
   - *Currently, they are achieving only **28 TFLOPs** per logical unit, which is significantly lower than the expected performance stated by AMD.*



**Link mentioned**: <a href="https://github.com/EleutherAI/gpt-neox/blob/f7a5a6f9da47de4d4d7cdf776c0832b257f329ef/megatron/training.py#L958-L973)">gpt-neox/megatron/training.py at f7a5a6f9da47de4d4d7cdf776c0832b257f329ef · EleutherAI/gpt-neox</a>: An implementation of model parallel autoregressive transformers on GPUs, based on the Megatron and DeepSpeed libraries - EleutherAI/gpt-neox

  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1329184978311057479)** (1 messages): 

> `Bolt project title editing` 


- **Editing Titles in Bolt Now Available**: A fresh update to **Bolt** allows users to edit project titles directly, enhancing project discovery in the list.
   - *This feature makes it easier to find projects*, according to the announcement on [Stackblitz Twitter](https://x.com/stackblitz/status/1879625416706785365).
- **Enhanced Project Organization Feature**: The new capability to modify project titles helps streamline project management within **Bolt**, facilitating a cleaner workspace.
   - Users can now effortlessly align project titles with their content for better organization.



**Link mentioned**: <a href="https://x.com/stackblitz/status/1879625416706785365">Tweet from StackBlitz (@stackblitz)</a>: 📢 Fresh Bolt update:You can change the project title now — making it easier to find on the projects list!

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1329277705711128587)** (5 messages): 

> `Chat History Snapshot System, Date Input Issue, New User Introduction, GitHub Repo Interaction` 


- **Chat History Snapshot System Implemented**: A pull request titled *'feat: restoring project from snapshot on reload'* has been submitted by thecodacus, introducing a snapshot system for chat history, allowing users to restore previous state upon reload. The PR details can be found [here](https://github.com/stackblitz-labs/bolt.diy/pull/444).
   - This implementation aims to maintain continuity in user interactions and ensure associated file system states are preserved.
- **Issue with Calendar Date Input**: A user reported that some individuals experience issues where the date they select from the calendar popup changes upon submission. Suggestions were requested to troubleshoot and construct a prompt that accurately retrieves the selected date.
   - Another member advised checking for bugs or logical issues in the current date input implementation to prevent these discrepancies.
- **New Users Welcomed**: A newcomer introduced themselves within the channel, expressing their excitement to join the community. No further information was provided, but welcoming remarks followed from other members.
   - Engagement with new users is encouraged to foster a supportive environment.
- **Prompting Bolt for GitHub Repo Usage**: An inquiry was raised regarding the potential for Bolt to utilize specific GitHub repositories for functionality and if there are plans to enhance the use of retrieval-augmented generation (RAG) on repository READMEs. This would aim to ensure correct and efficient usage of the repositories.
   - Further discussions on this topic could clarify future enhancements related to repository interactions.



**Link mentioned**: <a href="https://github.com/stackblitz-labs/bolt.diy/pull/444">feat: restoring project from snapshot on reload by thecodacus · Pull Request #444 · stackblitz-labs/bolt.diy</a>: Add Chat History Snapshot SystemOverviewThis PR introduces a snapshot system for chat history, allowing the restoration of previous chat states along with their associated file system state. This...

  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1329183087241859102)** (178 messages🔥🔥): 

> `Git Support Discussions, Shadcn Default in Bolt, Session and Token Issues, Deployment Challenges, Stripe Integration Problems` 


- **Upcoming Git Support Meetings**: In today's office hours, it was shared that there will be meetings later to discuss **Git support**, with a possible release in the next **2-4 weeks**.
   - Members expressed excitement about the potential for this feature.
- **Shadcn as Default for Bolt**: A user inquired whether it has become the default for Bolt to use **Shadcn**, as noted in recent chats.
   - This change follows previous usage of **Headless UI** in earlier projects.
- **Token and Session Challenges**: Participants reported issues with **session token management**, as prompts are reportedly consuming excessive tokens.
   - One member noted **4 million tokens** were used for a single command, prompting skepticism and calls for support to investigate.
- **Deployment Issues Encountered**: Several users faced persistent challenges with deploying their projects, leading to frustrations about project stability due to large sizes.
   - It was suggested that moving assets to **Amazon S3** could alleviate some issues related to **deployment size**.
- **Stripe Integration Queries**: Concerns around integrating **Stripe** with ongoing problems were raised, as users struggled with implementation.
   - Feedback pointed towards needing a clearer development environment and potentially resolving access issues noted in an update.



**Link mentioned**: <a href="https://tenor.com/view/apes-together-strong-0p1sf-gif-20906166">Apes Together Strong 0p1sf GIF - Apes Together Strong 0p1sf - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1329195107764928642)** (133 messages🔥🔥): 

> `Swarm vs A1111, Scam Alert on Twitter, Image Generation Metrics, Community License Attribution, Print on Demand with Stable Diffusion` 


- **Swarm UI gains popularity over A1111**: Members discussed their preferred interface for creating images with AI, with **Swarm** being highlighted for its active development and documentation, unlike **A1111**, which hasn't been updated since July.
   - One user noted that Swarm's backend is helpful for specialized tasks and praised the engaged community of its developer.
- **Twitter Account Compromised with Scam**: Concern spread regarding a compromised Twitter account associated with @StabilityAI, leading to fraudulent posts about a token launch, prompting users to refrain from clicking links.
   - Community members quickly alerted others to the scam, citing past experiences where people were exploited by similar fraudulent activities.
- **Question about Image Generation Metrics**: A user asked about metrics for image generation with the **Stable Diffusion** model, specifically regarding timing metrics like iterations per second.
   - Others shared that some UIs display the total time and steps taken, while specific details may be found in the metadata or other UI elements.
- **Attribution Requirements under Community License**: Clarification was sought regarding the need for attribution when using images generated under Stability AI’s community license.
   - One member noted that while it's beneficial to give credit, outputs are generally free to use unless tied to a commercial application requiring licensing.
- **Using SD for Print on Demand Business**: A print on demand store owner expressed interest in leveraging **Stable Diffusion** to create large images and inquired about adjusting resolution for print.
   - Guidance was offered via direct message to assist with using AI for business purposes, emphasizing the potential of stable diffusion in this context.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/dango233max/status/1879734940264481006">Tweet from Dango233 (@dango233max)</a>: Just reached out to my SAI Friends.This is a SCAM!!!! @StabilityAI  X Account is compromised.DO NOT TRUST IT!</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0">stabilityai/stable-diffusion-xl-base-1.0 · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1329198246916263967)** (62 messages🔥🔥): 

> `Lint Errors After Update, DeepSeek Usage and Alternatives, Model Performance Concerns, MOE and GPU Efficiency, AI Security Layers` 


- **Lint Errors Post-Update Confusion**: Multiple members reported experiencing numerous **lint errors** since a recent update, with one questioning if this was a widespread issue.
   - Another clarified that the new version displays linter output without causing more errors, suggesting a potential misunderstanding.
- **Exploring Alternatives to DeepSeek**: A user mentioned switching from **DeepSeek** to **Sonnet** due to performance issues, indicating the latter is preferable.
   - Others inquired about hardware requirements, establishing a belief that DeepSeek needs significant resources like **500GB VRAM** for effective performance.
- **Mixed Reviews on Model Performance**: Concerns were raised about **DeepSeek3**'s declining performance recently, prompting users to explore other AI model providers.
   - Cost-effective alternatives, like **Hyperbolic**, were discussed, with prices notably lower than DeepSeek's at **$0.25/mtok**.
- **Efficiency of MOE Models Discussed**: Discussion centered around how **MOE** (Mixture of Experts) models can optimize performance by activating subsets of weights to save resources.
   - It's believed that if managed correctly, batching could lead to lower costs and improve efficiency on model runs.
- **Need for AI Security Layers**: A suggestion was made for **aider** to incorporate a security layer model that filters sensitive data before sending it to providers.
   - Members acknowledged the importance of securing interactions with AI models to prevent unintentional data leaks.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1ho0w52/deepseek_does_not_need_5_hours_to_generate_1/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://youtu.be/rx0wP9k4wGM?si=BAt3ZUfjGRB6lX3Q">SakanaAI Unveils &quot;Transformer Squared&quot; - Test Time LEARNING</a>: Join My Newsletter for Regular AI Updates 👇🏼https://forwardfuture.aiMy Links 🔗👉🏻 Subscribe: https://www.youtube.com/@matthew_berman👉🏻 Twitter: https:/...</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12814806715/">feat: Changing current subtree with /subtree command · Aider-AI/aider@29a4a67</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12814806715/job/35732104905?pr=2881):">feat: Changing current subtree with /subtree command · Aider-AI/aider@29a4a67</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/actions/runs/12801873727/job/35692059901#step:5:1">feat: Changing current subtree with /subtree command · Aider-AI/aider@7ab00bb</a>: aider is AI pair programming in your terminal. Contribute to Aider-AI/aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/issues/2777">Generating data to fine-tune models · Issue #2777 · Aider-AI/aider</a>: Issue Quite often, I reject the edits and tell the model to change something. What would be ideal is a /fine-tune command, which would: Instruct model to take the correction into account, but gener...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1329194044634566689)** (64 messages🔥🔥): 

> `Aider API Logging, Git Commit Issues, CEDARScript Integration, Chat Session Management, Agentic Tools for Code Exploration` 


- **Request for Aider API Call Logging**: Users inquired about a method for Aider to log LLM API calls since the chat history doesn't capture full API calls.
   - Currently, it seems there is no solution available, prompting discussions about running Aider locally to log this information.
- **Issues with Aider Commit Functionality**: A user reported issues with Aider making commits to their git repository despite the changes being made.
   - It was suggested to use the `--auto-commit` option, while another user shared that architect mode helps with committing changes.
- **Integration of CEDARScript with Aider**: A member shared a GitHub link regarding CEDARScript, which allows Aider to use CEDARScript as an editing format.
   - Discussions ensued on whether this feature could be merged into Aider, but there was no consensus on its quantitative benefits.
- **Saving Chat Sessions in Aider**: Concern arose over how Aider saves chat sessions, as existing methods only save which files were loaded without the full chat history.
   - The bot clarified that while it's not currently possible to save different chat histories, using the `/read-only` command can help manage clutter.
- **Diverse Tools for Code Exploration**: A user listed various agentic tools for exploring larger codebases, highlighting their flexibility and integration capabilities.
   - They also shared their own code-exploration tool and discussed plans to enhance it using AG2's swarm orchestration approach.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/2025/01/15/uv.html">Using uv as an installer</a>: Reliably packaging &amp; distributing python CLI tools is hard. Aider uses uv in novel ways to make it easy to install the aider CLI, its dependencies and python 3.12. All in an isolated env.</li><li><a href="https://aider.chat/docs/git.html">Git integration</a>: Aider is tightly integrated with git.</li><li><a href="https://bw2.github.io/ConfigArgParse/configargparse.ArgumentParser.html#__init__):">configargparse.ArgumentParser</a>: no description found</li><li><a href="https://github.com/CEDARScript/cedarscript-integration-aider?tab=readme-ov-file#installation">GitHub - CEDARScript/cedarscript-integration-aider: Allows Aider to use CEDARScript as an edit format</a>: Allows Aider to use CEDARScript as an edit format. Contribute to CEDARScript/cedarscript-integration-aider development by creating an account on GitHub.</li><li><a href="https://github.com/Aider-AI/aider/pull/2877">docs: Add architect mode section to edit errors troubleshooting guide by golergka · Pull Request #2877 · Aider-AI/aider</a>: no description found</li><li><a href="https://docs.ag2.ai/notebooks/agentchat_swarm_enhanced">Enhanced Swarm Orchestration with AG2 - AG2</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1329581141849669642)** (1 messages): 

> `Helicone LLM Observability, LLM Security Features, Local Deployment with Docker` 


- **Helicone Launches Open Source LLM Observability Tool**: Helicone introduced an [open source observability platform](https://github.com/Helicone/helicone) for LLMs, enabling users to monitor, evaluate, and experiment with just one line of code.
   - This platform highlights features like **track&trace** for requests and costs, an **LLM security layer**, and additional metrics tracking.
- **Local and Cloud Deployment Options Available**: Helicone can run locally via **docker-compose**, although they recommend using the cloud version.
   - This setup allows for features like **caching** and **custom rate limits** for enhanced performance.



**Link mentioned**: <a href="https://github.com/Helicone/helicone">GitHub - Helicone/helicone: 🧊 Open source LLM observability platform. One line of code to monitor, evaluate, and experiment. YC W23 🍓</a>: 🧊 Open source LLM observability platform. One line of code to monitor, evaluate, and experiment. YC W23 🍓 - Helicone/helicone

  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1329180078713602076)** (52 messages🔥): 

> `Nous Research Affiliation, Merch Funding, Fine-tuning Techniques, LLAMA 1B QLoRA Training, Community Engagement` 


- **Nous Research is a private entity**: Members clarified that **Nous Research** has no affiliation with any government or academic institutions, operating as a purely private organization.
   - There's a focus on openness, with discussions on how other AI companies might have government ties.
- **Funding through merch and private equity**: It's understood that funding for **Nous Research** primarily comes from merch sales and private equity, though merch revenue is relatively small.
   - Members also discussed the inclusion of stickers in merch orders and expressed interest in them.
- **Fine-tuning techniques and recommendations**: In discussions about fine-tuning, a member shared insights on the importance of training techniques like RL and the dependence of output accuracy on diverse settings.
   - Others mentioned the significance of effective prompt design and how interactions with the model during training can enhance accuracy.
- **Feedback on LLAMA 1B QLoRA training graphs**: Members reviewed training graphs for **LLAMA 1B QLoRA**, noting concerns about the small dataset and the inadequacy of training steps.
   - Discussion revolved around fitness scores calculations and preferences for simplifying performance metrics during evaluation.
- **Community interactions and casual conversation**: Various members engaged in light-hearted exchanges, including greetings and humorous remarks about the day's sentiment.
   - Members encouraged each other to reach out for discussions or inquiries, showcasing community engagement.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1329191150220873778)** (35 messages🔥): 

> `GrokAdamW and Ortho Grad, LLM Memorization, Process Reward Models (PRMs), Chatbot architecture with PDF data` 


- **GrokAdamW vs Ortho Grad Merger**: Members discussed the conceptual feasibility of combining **GrokAdamW** with **Ortho Grad**, noting that GrokAdamW offers better loss metrics, which could be advantageous despite possible drawbacks from Ortho Grad.
   - _GrokAdamW by Eric Hartford_ was highlighted, with a shared GitHub link for reference.
- **Research on LLMs Memorizing Text**: A discussion was initiated on how well **LLMs** can memorize text, with a recommendation to explore Anthropic's research on **explainable AI** for deeper insights.
   - A participant mentioned experimenting with dictionary learning related to the topic, reflecting hands-on engagement with the material.
- **Challenges with Process Reward Models**: Insights were shared about the complexities of developing **Process Reward Models (PRMs)**, particularly regarding data annotation and evaluation methodologies, and their impact on performance.
   - Mention of the **Qwen team's** work on PRMs pointed to useful documentation and insights into the development process.
- **Chatbot Architecture for PDF Data**: A member evaluated the use of **GPT-2** for a RAG chatbot based on PDF data, expressing challenges with the small context window and nonsensical outputs.
   - Feedback from others suggested that attempting this task with GPT-2 might not be viable, with recommendations to consider larger models for improved performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.07301">The Lessons of Developing Process Reward Models in Mathematical Reasoning</a>: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in...</li><li><a href="https://www.anthropic.com/research/mapping-mind-language-model">Mapping the Mind of a Large Language Model</a>: We have identified how millions of concepts are represented inside Claude Sonnet, one of our deployed large language models. This is the first ever detailed look inside a modern, production-grade larg...</li><li><a href="https://buttondown.com/ainews/archive/">AI News</a>: We summarize top AI discords + AI reddits + AI X/Twitters, and send you a roundup each day! See archive for examples.   &quot;Highest-leverage 45 mins I spend everyday&quot; - Soumith  &quot;best AI n...</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: Contribute to cognitivecomputations/grokadamw development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (3 messages): 

> `New architecture design, Recurrent models and attention, Neural long-term memory` 


- **New Architecture Emerges**: A member announced that a **new architecture** has been introduced, citing it as interesting and provided the [PDF link](https://arxiv.org/pdf/2501.00663v1) to the paper.
   - Another member noted that changing 'pdf' to 'abs' in the URL generates the paper abstract, sharing a link to the [abstract](https://arxiv.org/abs/2501.00663v1).
- **Exploring Recurrent Models and Attention**: The paper's abstract discusses the extensive research efforts on **recurrent models** and **attention**, highlighting their strengths and limitations in modeling dependencies.
   - It points out that while attention captures dependencies across all tokens, it incurs a quadratic cost, which restricts context length for accuracy.
- **Innovative Neural Long-Term Memory Module**: The research introduces a **neural long-term memory** module that learns to memorize historical context, enhancing attention's capability to manage current context.
   - This approach supports fast parallel training and maintains quick inference, aiming to balance memory efficiency with accurate dependency modeling.



**Link mentioned**: <a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memo...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1329241405167636500)** (2 messages): 

> `GrokFast Optimizer, Orthograd Optimizer, Coconut Repository` 


- **GrokFast Optimizer's Stability Issues**: Many users struggled to achieve stability with the **GrokFast optimizer** during LLM training, indicating it may not be reliable.
   - This has led to a shared sentiment among the community highlighting the need for better alternatives.
- **Orthograd Optimizer as a Replacement**: The **Orthograd optimizer** appears to function as a wrapper around **torch SGD** or **AdamW**, presenting a potential drop-in replacement for users.
   - With this development, there is hope that users will share more experiences and results from trying out this new optimizer.
- **Coconut GitHub Repository Released**: The [Coconut repository](https://github.com/facebookresearch/coconut) from Facebook Research focuses on training large language models to reason in a continuous latent space.
   - This innovative approach aims to enhance model reasoning capabilities, marking a new step in AI research.



**Link mentioned**: <a href="https://github.com/facebookresearch/coconut">GitHub - facebookresearch/coconut: Training Large Language Model to Reason in a Continuous Latent Space</a>: Training Large Language Model to Reason in a Continuous Latent Space - facebookresearch/coconut

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1329546499566342227)** (3 messages): 

> `New Neural Architecture, Recurrent Models and Attention, Neural Long-Term Memory Module` 


- **Exciting New Neural Architecture Released**: A member shared a link to a [new architecture](https://arxiv.org/pdf/2501.00663v1) paper, noting that it seems interesting.
   - Another member mentioned that changing 'pdf' to 'abs' in the URL provides access to the paper's abstract.
- **Impact of Recurrent Models and Attention**: The paper discusses the evolution of **recurrent models** and **attention**, highlighting their strengths and limitations in modeling dependencies.
   - It asserts that while attention allows for accurate dependency modeling, it incurs a quadratic cost that restricts context length.
- **Introduction of Neural Long-Term Memory Module**: The research presents a **neural long-term memory module** designed to enhance attention mechanisms by memorizing historical context.
   - This approach promises fast parallelizable training and efficient inference while balancing short-term and long-term information.



**Link mentioned**: <a href="https://arxiv.org/abs/2501.00663v1">Titans: Learning to Memorize at Test Time</a>: Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memo...

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1329194292224327861)** (13 messages🔥): 

> `Novice Author Workshop, Digital Pathology Groovy Scripts, Interactivity in NotebookLM, Podcast Sharing, Notebook Usability Feedback` 


- **Novice Author Workshop explores Mists**: A discussion targeted towards novice author **Roseline K Marie** dives into her debut novel's early draft and the role of the **Mists** in the plot.
   - Participants share *crazy theories* and seek to evaluate if their subtle foreshadowing is effective.
- **Digital Pathology Groovy Script Success**: A member expressed amazement at how NotebookLM provided a **functional script** for handling **annotations as images** after struggling to find one via forum posts.
   - They noted significant time saved on their project after inputting their requirements into the notebook.
- **Interactive Mode excitement in NotebookLM**: With the start of a new semester, a member excitedly began loading their module's resources into NotebookLM and tinkered with **Interactive Mode**.
   - They shared their progress via a screenshot and indicated readiness for the upcoming courses.
- **Desire for a podcast advertising channel**: A member requested the creation of a new channel dedicated to **podcast promotion** to keep discussions on **use cases** focused.
   - This suggestion highlights a need for improved organization in content sharing within the community.
- **Notebook usability and prompt sharing**: Feedback was shared regarding NotebookLM's limitations with document formats, specifically that it doesn't accept **docx** or Google documents directly.
   - Members suggested sharing prompts too, acknowledging the collaborative nature of exploring Notebook's capabilities.



**Link mentioned**: <a href="https://player.captivate.fm/show/406b155f-62af-46d5-9a27-17766ca55b91">Easily listen to NotebookLM ➡ Token Wisdom ✨</a>: Quickly and easily listen to NotebookLM ➡ Token Wisdom ✨ for free!

  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1329180466514886687)** (73 messages🔥🔥): 

> `NotebookLM Plus Access, Podcast Feature Enhancements, Source Management, Google Workspace Licensing, Integration of New Tools` 


- **Clarifying NotebookLM Plus Access**: Several members expressed confusion about access to NotebookLM Plus and related features in their Google Workspace plans, especially regarding the transition from older licenses.
   - It was clarified that existing features are not being put behind a paywall and will be included in the Workspace offerings at no extra cost.
- **Podcast Generation Challenges**: Users reported difficulties generating podcasts using multiple sources, with suggestions made to create separate notebooks for each source to facilitate output.
   - A workaround for the issue involves unchecking unwanted sources, although confirmations on whether this was a Plus feature remain unclear.
- **Concerns Over Podcast Quality**: Feedback on the podcast feature highlighted issues with back channeling, inconsistent host interactions, and audio quality that detracts from a professional output.
   - Members discussed potential instructions that might help streamline the podcast generation for better coherence.
- **Limitations with Source Uploading**: There is currently no functionality for bulk uploading sources into Notebooks, which frustrates users who wish to import multiple URLs at once.
   - Members were advised to manually add sources or upload as single entries while hoping for more efficient options in the future.
- **Customizing Podcast Outputs**: Users sought to customize audio outputs for podcasts but noted that most current functionalities revolve around basic summaries without detailed instruction options.
   - The ability to customize podcasts by focusing on specific sources could enhance their usability for educational purposes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://workspace.google.com/blog/product-announcements/empowering-businesses-with-AI">The future of AI-powered work for every business | Google Workspace Blog</a>: The best of Google AI is now included in Workspace Business &amp; Enterprise plans, giving you AI features like Gemini and NotebookLM Plus with no add-on needed.</li><li><a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora&#x27;s Law: Intelligence Scales With Constraints, Not Compute</a>: This is a working paper exploring an emerging principle in artificial intelligence development.</li><li><a href="https://support.google.com/a/answer/181865#zippy=%2Cturn-services-on-or-off-for-users">Turn on or off additional Google services - Google Workspace Admin Help</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1329178982276661248)** (1 messages): 

> `Minimax-01, Needle-In-A-Haystack test, Requesting models on Discord` 


- **Minimax-01 Launches with Record Context Length**: The new model **Minimax-01** is now available, being the first open-source LLM to pass the **Needle-In-A-Haystack test** at an impressive **4M** context length. More details can be found on the [OpenRouter page](https://openrouter.ai/minimax/minimax-01).
   - *To request access to the model*, visit our [Discord](https://discord.gg/fVyRaUDgxW).
- **Image Analysis Updates**: An image was attached alongside the announcement regarding **Minimax-01**, providing a visual reference for the model. The image features analyses relevant to the details shared in the launch.
   - Further insight into the image can be found in the associated [Discord attachment](https://cdn.discordapp.com/attachments/1092729520181739581/1329178982020812851/image.png?ex=678ab764&is=678965e4&hm=321979ba1263cb2992785532f4e5fdb3b644d91db5662473db7c82d3bfd8462a&).



**Link mentioned**: <a href="https://openrouter.ai/minimax/minimax-01>">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1329188916334563459)** (85 messages🔥🔥): 

> `Minimax model performance, DeepSeek issues, OpenRouter regional restrictions, Gemini flash model errors, Activity page functionality` 


- **Minimax model evaluation sparks interest**: Users are curious about the performance of the new **Minimax** model with developer tasks, especially in comparison to existing options like **DeepSeek**.
   - Discussions highlight that while some expect only decent performance, published scores such as those from **humaneval** may be worth checking.
- **DeepSeek experiences delays**: Members reported ongoing issues with **DeepSeek**, including latency and provider reliability, particularly during peak usage times.
   - Users discussed strategies for troubleshooting, including checking provider errors and potential fixes involving tweaking API settings.
- **OpenRouter's regional restrictions revealed**: It was confirmed that **OpenRouter** has been enforcing regional restrictions in line with policies from **OpenAI** and **Anthropic** for some time.
   - The revelation stirred conversations about the implications of these restrictions and user experiences navigating them.
- **Gemini model endpoint changes cause confusion**: Updates on the **Gemini flash 2.0** model indicated a change in endpoints, leading to unexpected errors for users trying to access the service.
   - Affected users shared solutions, including adjustments to privacy settings to remedy problems with endpoint accessibility.
- **Activity page functionality questioned**: A user raised concerns about the **activity page**, which appeared to show identical graphs for different API keys, causing confusion.
   - Clarification revealed that the page currently aggregates all transactions without distinction, leading to debates around its design and utility.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1879908197420675322">Tweet from OpenRouter (@OpenRouterAI)</a>: Minimax-01 by @MiniMax__AI is now available: a low-cost, 456B multi-modal open source LLM.It&#39;s the first to pass the vanilla Needle-In-A-Haystack test at a whopping 4M context:</li><li><a href="https://meowapps.com/ai-engine/">AI Engine</a>: Adds AI features to WordPress. Chatbots, Forms, Copilot, Content Generation, and much more!</li><li><a href="https://openrouter.ai/docs/provider-routing">Provider Routing | OpenRouter</a>: Route requests across multiple providers
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1329214489933123696)** (19 messages🔥): 

> `Command R+ Coding Languages, Payment Processing with Stripe, Cohere Models Proxying` 


- **Users Curious about Command R+ Language Training**: A user inquired about the primary coding languages that **Command R+** is trained on and whether there is a resource to check this information.
   - Another member suggested that users can access the model via the API to test its suitability for specific use cases.
- **Payment Processing Clarifications**: A member noted that payments are handled via **Stripe**, with the option to use **OpenRouter** for additional functionalities.
   - Additionally, they mentioned that **OpenRouter** proxies all **Cohere models** available.


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1329263875568435291)** (13 messages🔥): 

> `Updating Command R models, Rerank 3.5 Coding Capabilities, Embedding Model Limitations, Prompt Composition for Rerank` 


- **Updating Command R Models Continuously**: A member suggested that existing **Command R** and **R+** models should be continuously updated using the latest data and fine-tuning techniques, akin to an updated version 08-2024.
   - Another member noted that updating models in such a manner would ultimately lead to the development of a new model.
- **Rerank 3.5 Excels in Coding Tasks**: One member highlighted that **Rerank 3.5** is particularly good for coding and has been trained on common languages like **Python**, **JavaScript**, and **C++**.
   - They noted that while the model is effective, some specific use cases had not been part of its training.
- **Challenges with Embedding Models**: A member expressed concerns regarding the limitation of embedding models, indicating that the only option for updates requires re-embedding all data, which is a massive task.
   - They clarified that there's no effective method to migrate embeddings from one model version to another, leading to prolonged usage of specific embedding models.
- **Understanding Rerank Model Bias**: A member raised questions about Rerank 3.5, specifically noting a decrease in accuracy when more documents are provided and a bias towards semantic documents over lexical ones.
   - They inquired if these phenomena are inherent properties of the re-rank models or if optimizations could be made through integration with their data.
- **Crafting Effective Prompts for Rerank**: One member sought advice on constructing effective prompts for **Rerank**, particularly regarding how to organize chat history between user and assistant.
   - They asked for guidance on whether to maintain chronological order when passing this information in the prompts.


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1329195020707696661)** (53 messages🔥): 

> `Random Number Generation, Cohere Rerank Tool, Deep Learning Learning Resources` 


- **Cmd R Bot generates random numbers**: Cmd R Bot successfully generated random numbers upon user requests, providing values such as **84**, **12**, and **37**.
   - *Gulp* was the bot's response to a user querying about its highest random number, emphasizing its limitations.
- **Cohere's Rerank Tool Explained**: Cohere's Rerank is a semantic search tool that utilizes Rerank models to sort document relevance based on a query.
   - The latest model, **Rerank-v3.5**, features advanced capabilities for multilingual retrieval tasks across various domains.
- **Resources for Learning Deep Learning**: Cohere offers several free resources for learning deep learning, including **LLM University**, cookbooks, and $75 in credits for new accounts.
   - For beginners, recommended materials include guided courses on generative AI and practical cookbooks such as 'Hello World! Meet Language AI'.



**Link mentioned**: <a href="https://cohere.com/llmu#text-representation)">LLM University (LLMU)</a>: Welcome to LLM University, your premier learning destination for mastering Enterprise AI technologies. Designed for developers and technical professionals, our hub offers comprehensive resources, expe...

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1329457126854230078)** (12 messages🔥): 

> `Tinygrad in Browser, JSPI Flag Integration, Cross-Platform Testing, Cloud Computing Goals, Non Uniform Memory Management` 


- **Tinygrad Running in Browser**: Tinygrad can now run in the browser with the aid of [JSPI flag](https://v8.dev/blog/jspi) enabled, and it’s been tested successfully on various platforms including Mac and Ubuntu.
   - The project can be accessed at [this link](https://mesozoic-egg.github.io/tinygrad/) for others to test and explore.
- **Cross-Platform Success with JSPI**: Users confirmed success in running Tinygrad on Windows 10, Windows 11, and Mac M1 by enabling the **JSPI flag** in Chrome, showcasing its broad compatibility.
   - One user mentioned, 'works on my M1 pro after enabling jspi flag' further validating the integration.
- **George Hotz's Vision for Cloud Computing**: George Hotz shared ambitious goals for cloud computing by suggesting that networked machines could operate collectively like one GPU, challenging existing architectures.
   - He emphasized, *'there's a whole world of possibly above the current NVIDIA stack'* when discussing the potential future of computing power.
- **Addressing Non Uniform Memory Issues**: Hotz conveyed that solving **non uniform memory** at one scale could lead to broader solutions, contributing to cheaper chip designs.
   - He noted, *'but if we solve non uniform memory at one scale, we solve it at all*,' highlighting the importance of this challenge.
- **Draft PR for Browser Implementation**: A draft Pull Request ##8645 has been created to implement the Tinygrad functionality for the browser, inviting further testing from the community.
   - Feedback was requested specifically for testing on Windows to ensure compatibility across systems.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad/">no title found</a>: no description found</li><li><a href="https://x.com/__tinygrad__/status/1879930546652156027">Tweet from the tiny corp (@__tinygrad__)</a>: @ID_AA_Carmack @nisargypandya What&#39;s crazy is there&#39;s a whole world of possibly above the current NVIDIA stack. Why can&#39;t all the networked machines behave like one GPU?
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1329344184540467312)** (56 messages🔥🔥): 

> `Tinygrad installation issues, TinyJit performance, Metal backend on pre M3 devices, Model export/import in Tinygrad, Operator fusion notes` 


- **Installation issues with Tinygrad in conda**: A user reported an error while installing Tinygrad with `pip install -e .` in a conda environment, indicating an issue with `libgcc_s.so` not being an ELF file.
   - Discussion revealed that using standard Python without venv works fine, and pointed to a possible bug caused by conda overriding system libraries.
- **TinyJit performs unexpectedly on Metal backend**: A user experienced slower optimization steps using TinyJit on a 2019 MacBook Pro with Metal backend, sparking a discussion about GPU synchronization.
   - The issue was linked to not syncing the GPU, and solutions included adjusting JIT settings and inspecting debug logs.
- **Disabling Metal graph for Intel MacBook Pros**: It was suggested to disable Metal graph by default for pre M3 Intel MacBook Pro users due to performance issues with the Metal driver.
   - There's consensus that this change could enhance user experience for those on Intel-based machines.
- **Exporting and importing jitted models**: A discussion initiated about the feasibility of exporting jitted models in Tinygrad for faster reloading and inference without recompilation.
   - It was noted that jitted functions can be pickled, allowing for efficient model usage, reminiscent of processes seen in openpilot.
- **Operator fusion insights**: A user shared a link detailing operator (un)fusion in Tinygrad, providing insights into optimization techniques.
   - The provided document serves as a resource for understanding operator fusion, demonstrating the community's focus on improving Tinygrad's performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tellusim.com/metal-mdi/">MultiDrawIndirect and Metal - Tellusim Technologies Inc.</a>: no description found</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20250117_fusion.md">tinygrad-notes/20250117_fusion.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/f91ca508cf88b09c616473561f68d2d46fbfcef9/tinygrad/renderer/ptx.py#L118">tinygrad/tinygrad/renderer/ptx.py at f91ca508cf88b09c616473561f68d2d46fbfcef9 · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad</li><li><a href="https://git.informatik.uni-hamburg.de/4kirsano/master-thesis/-/blob/main/.conda/lib/libgcc_s.so)">Files · main · 4kirsano / Master-Thesis · GitLab</a>: UHH Informatics GitLab EE</li><li><a href="https://github.com/uuuvn/tinygrad.git">GitHub - uuuvn/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - GitHub - uuuvn/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1329180373074186280)** (36 messages🔥): 

> `AI Project File Issues, PrivateGPT and Obsidian, ChatGPT Software Engineering Limitations, Generative AI Development Hurdles, Omnimodal Image Generation Delays` 


- **AI struggles with project file comprehension**: A member raised concerns about AI not processing project files correctly, echoing frustrations shared by others.
   - Others speculated on potential solutions or workarounds to avoid starting new sessions.
- **Exploring PrivateGPT for Obsidian Integration**: A member asked if anyone has successfully used PrivateGPT to automatically learn from an Obsidian notebook, suggesting an interest in tool integration.
   - The conversation hinted at possible workflows for enhancing AI interaction with personal knowledge systems.
- **ChatGPT lacks true software engineering capabilities**: Members discussed how ChatGPT can assist in coding but lacks the ability to function as a software engineer, especially in creating complex applications.
   - One user expressed hope for future enhancements that might bridge this gap, comparing the ideal scenario to existing AI assistants.
- **Concerns about generative AI limitations**: Discussions highlighted a feeling that generative AI's capabilities are limited to media creation without significant advancements in functionality.
   - Members expressed frustration over costs associated with high-quality outputs from tools like Midjourney.
- **Doubts about omnimodal image generation release timelines**: Members pondered over the slow rollout of omnimodal image generation from companies like OpenAI and Gemini, reflecting on delays in product launches.
   - Additionally, there were comments on the struggles of open-source audio models in managing emotional outputs effectively.



**Link mentioned**: <a href="https://www.youtube.com/watch?v=x8jFFhCLDJY">Google Research Unveils &quot;Transformers 2.0&quot; aka TITANS</a>: Have we finally cracked the code on how to give models &quot;human-like&quot; memory? Watch to find out!Join My Newsletter for Regular AI Updates 👇🏼https://forwardfu...

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1329259374459355197)** (15 messages🔥): 

> `Canvas on Web, GPT-4o Tasks, Image Rendering Issues, Custom GPTs, Version History Glitches` 


- **Canvas on Web now shows tasks only**: A member noted that they only see tasks on the web version, but another clarified that the **Canvas** can still be found by clicking the toolbox icon near the bottom-left of the text entry box.
   - It seems others might also be experiencing temporary confusion about the interface changes.
- **How GPT-4o Tasks function**: Tasks in **GPT-4o** act as reminders, like 'Remind me to practice Spanish every day at 3pm', with ChatGPT notifying users at the scheduled times.
   - This feature allows for timely actions, enhancing user interaction with the model.
- **Image Rendering Issues in Journal Prompt GPT**: A member reported their GPT stopped rendering images with the journal prompts, leading to a discussion about potential configuration issues.
   - It was discovered that the DALL·E box might have been unchecked in the capabilities settings, causing the malfunction.
- **Custom GPTs confirmed using GPT-4o**: Clarification was provided that **Custom GPTs** utilize the **GPT-4o** version, ensuring users know what model they're working with.
   - This helps users understand the capabilities of their custom implementations.
- **Glitches in Version History Display**: A member observed that some old versions in the Version History display 'INVALID DATE' instead of actual dates.
   - Concerns were raised about this glitch, although the cause remains unspecified for now.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (3 messages): 

> `Prompt Engineering, Writing Books on AI, Self-Discovery Techniques` 


- **30 Days to Learn Prompt Engineering and Write a Book?**: A user inquired if it's feasible to learn **prompt engineering** and write a book in **30 days** using OpenAI documentation.
   - Another member affirmed that it's possible and emphasized that one can *write the book as soon as they can prompt it*.
- **Recommended Resources for Prompt Engineering**: A member advised using the provided [link](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb) and additional web searches to enhance knowledge in prompt engineering.
   - They encouraged employing **self-discovery techniques** for prompting the AI as part of the learning process.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1329541557254230099)** (3 messages): 

> `Prompt Engineering, Writing a Book on Prompting Techniques, OpenAI Documentation Utilization` 


- **30 Days to Master Prompt Engineering?**: A user asked if they could learn **prompt engineering** and write a book in just **30 days**, referencing OpenAI's documentation for guidance.
   - Another member affirmed this is possible, suggesting that the book can be ready as soon as effective prompts are established.
- **Self-Discovery Techniques in Prompting**: A suggestion was made to adopt **self-discovery techniques** for enhancing prompt skills, emphasizing personal exploration over solely relying on existing resources.
   - A member shared a link to a [specific resource](https://chatgpt.com/share/67897fc3-6580-8000-af35-d693f933acfb) that could aid in this learning process.
- **Web Search for Knowledge Expansion**: Users were encouraged to utilize **web searches** alongside the provided resources to broaden their understanding of prompting techniques.
   - This approach aims to create a well-rounded foundation that combines multiple sources of information.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1329209654886269122)** (46 messages🔥): 

> `Perplexity AI Issues, Image Generation Quality, Claude Sonnet Performance, Bora's Law, Market Strategies` 


- **Perplexity AI faces functionality issues**: Users reported that Perplexity has been exhibiting errors and repeating answers, even with PRO search enabled, leading to frustration among some members.
   - One user noted that when using **GROK**, the performance significantly improved compared to Perplexity.
- **Image generation quality sparks debate**: A discussion emerged over the perceived quality of image generation, with users debating the effectiveness of platforms like ChatGPT, Flux, and Grok in comparison to Perplexity.
   - *This prompted a member to compare prompts, stating* 'it's not even close' when generating a simple sunrise image.
- **User experiences with Claude Sonnet**: Several users shared their struggles with **Claude Sonnet**, specifically mentioning inconsistencies in code suggestions and the AI's perceived helpfulness.
   - One user described their ongoing conflict with the AI over a CSV file processing task, reflecting broader concerns about its reliability.
- **The potential of Bora's Law**: A member argued that established approaches to AGI, including those by OpenAI, are flawed and proposed **Bora's Law**, stating intelligence scales with constraints rather than compute.
   - They cited their findings in an article to support this emerging principle in AI development.
- **Market strategies and success claims**: A user promoted a scheme offering to help individuals earn over $100K weekly in digital markets, requiring a percentage of profits as payment.
   - This proposal was met with skepticism from others, who humorously compared it to 'selling snake oil'.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora&#x27;s Law: Intelligence Scales With Constraints, Not Compute</a>: This is a working paper exploring an emerging principle in artificial intelligence development.</li><li><a href="https://www.youtube.com/watch?v=itpcsQQvgAQ&feature=youtu.be">Nintendo Switch 2 – First-look trailer</a>: Introducing Nintendo Switch 2, the successor to Nintendo Switch, releasing in 2025. Learn more: https://ninten.do/6003ohKAB
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1329334988918751285)** (2 messages): 

> `Enron Prank Revival, SEC Sues Elon Musk, TikTok Sale Consideration, 3D Printing AI Tools, 3D Printing Toys` 


- **Enron's Prank Revival Hits Headlines**: A recent [YouTube video](https://www.youtube.com/embed/NZ9FtiBAPUc) discusses the amusing return of pranks related to **Enron**.
   - The video captures a sense of nostalgia and humor around a topic historically known for its corporate scandal.
- **SEC Takes Action against Elon Musk**: In a significant legal move, the **SEC** has filed a lawsuit against **Elon Musk**, stirring discussions on regulatory impacts in the tech space.
   - This development raises questions about the implications for Musk’s ventures and public statements moving forward.
- **Chinese Officials Eye TikTok Sale**: There's speculation that **Chinese officials** are considering a potential sale of **TikTok**, prompted by increased scrutiny in international markets.
   - This could have major ramifications for the app's global operation and its user base leverage.
- **Exploring AI Tools for 3D Printing**: A member expressed interest in learning about AI tools available for creating **3D object files**, showcasing a growing curiosity in the technology.
   - You can check the [discussion link here](https://www.perplexity.ai/search/can-perplexity-ai-be-used-to-g-tp.qKJ1JQJaAAmpqQUb7Og#2) for insights and recommendations.
- **Excitement Around 3D Printing Toys**: A member shared their excitement about **3D printing**, mentioning they've enjoyed creating small toys but haven't yet tackled functional prints.
   - This reflects a growing trend of hobbyists exploring creative applications of modern fabrication technology.



**Link mentioned**: <a href="https://www.youtube.com/embed/NZ9FtiBAPUc">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1329382917226364960)** (4 messages): 

> `search_domain_filter, API model changes, CrewAI custom stop parameters` 


- **Request for Help with search_domain_filter**: A user asked for assistance in enabling the permission to use **search_domain_filter**.
   - This indicates an ongoing interest in utilizing domain filtering capabilities within the platform.
- **Speculation on API Model Changes**: A member observed the introduction of **sonar** and **sonar-pro** in the labs, questioning whether the API models will shift again.
   - They shared an [image](https://cdn.discordapp.com/attachments/1161802929053909012/1329550935038623877/image.png?ex=678ac04c&is=67896ecc&hm=411e3da2e4bdb49d1732bd458d3fdbf26e6147a332d51ec2687feb39fe5581f0&) that likely contains relevant information regarding these models.
- **Duplicate Issue with search_domain_filter**: Another user reported having the same issue regarding the **search_domain_filter** functionality, indicating that multiple users are facing this challenge.
   - This underscores a broader concern within the community about setting up permissions to use certain features.
- **CrewAI Custom Stop Parameters Error**: A user expressed frustration with receiving a **'custom stop parameters'** error while trying to work with CrewAI.
   - They inquired if any developers were present that could provide insights on getting **pplx** to work correctly.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1329204325343297628)** (51 messages🔥): 

> `Model Search Issues, LM Studio Logging, Context Window Explanation, VRAM vs System RAM, M2 Mac Model Loading Problems` 


- **User Experiences Search Issues with Model Loading**: A user expressed frustration that the **model search** was not working, with others suggesting it might be linked to **Hugging Face** issues.
   - Further discussions indicated that performance issues could stem from specific system requirements, particularly around CPU AVX2 instructions.
- **Logging Mechanisms for Troubleshooting**: Users discussed the lack of a clear **log window** for troubleshooting, with suggestions to use the developer tab and terminal commands to gain insights.
   - The terminal command `lms log stream` was highlighted as a useful method to check logs more thoroughly.
- **Understanding Context Window Management**: A user inquired about the meaning of 'context is 90.5% full', prompting explanations of the **Context Window** and its relation to the model's token capacity.
   - It was noted that while most models manage context well, user adjustments might be necessary to increase context size.
- **RAM and VRAM Usage Discussion**: A user questioned whether models are stored in **VRAM or system RAM**, prompting clarification that usage varies based on the hardware configuration.
   - For **CPU inference**, models use system RAM, whereas **GPU inference** utilizes VRAM, with spillover to RAM if necessary.
- **Problems with LM Studio 0.3.6 on Macs**: Multiple users reported difficulties loading models after updating to **LM Studio 0.3.6**, suggesting it might be related to cache issues.
   - Clearing the `.lm-studio` cache resolved loading problems, confirming the need for careful cache management during updates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: Discover, download, and run local LLMs</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/115#issuecomment-2576418933">LM Studio can&#39;t fetch models or extensions anymore after a while, only system restart helps (macOS) · Issue #115 · lmstudio-ai/lmstudio-bug-tracker</a>: Not sure where to post else, feel free to move this to the right loc. After a while, I am not sure how long this needs, and if the system sleeping might cause this issue but it happened a few times...
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1329271987167690824)** (50 messages🔥): 

> `Screenplay Feedback, Model Performance & Comparisons, AI Ethical Guardrails, Model Recommendations, Model Management in GPT4All` 


- **Challenges with Screenplay Analysis**: A user expressed frustration with character limits in GPT4All when analyzing their **45-page screenplay**, noting it seems to only analyze one scene at a time, despite the model having a **128 KB** capacity.
   - Accessing detailed analysis has proven cumbersome, prompting a search for workarounds and efficiency techniques.
- **Discrepancies in AI Model Responses**: Discussion arose around why **ChatGPT 4.0** can handle explicit content better than its alternate versions, suggesting that different models are trained under disparate censorship criteria.
   - Concerns were raised about the implications of these ethical guardrails on model performance and user access to balanced information.
- **Model Recommendations for Writing**: Users recommended **DavidAU's models** for writing assistance, highlighting their capability to generate both dark and non-dark content effectively.
   - Suggestions also included **Magnum** models and usage tips for optimizing performance based on VRAM and quant settings.
- **Model Management in GPT4All**: A user inquired about importing downloaded models into GPT4All, leading to clarification that models need to be placed in the specified folder and the application must be restarted.
   - The importance of closing and reopening the app after changes to ensure the model list updates was emphasized.
- **Technical Performance and Settings**: Performance issues arose for a user using the **Gemma model**, with suggestions for adjusting quantization settings to improve response speed.
   - Discussion included the impact of GPU settings and the need to fine-tune configurations for optimal operation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.prompthackers.co/compare/llama-3.2-3b/llama-3-8b">Compare Llama 3.2 3B vs Llama 3 8B Instruct - Pricing, Benchmarks, and More</a>: Compare pricing, benchmarks, model overview and more between Llama 3.2 3B and Llama 3 8B Instruct. In depth comparison of Llama 3.2 3B vs Llama 3 8B Instruct.</li><li><a href="https://huggingface.co/docs/transformers/main/main_classes/quantization">Quantization</a>: no description found</li><li><a href="https://huggingface.co/DavidAU">DavidAU (David Belton)</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1329468720678179000)** (5 messages): 

> `LeetGPU online playground, Discussion channel redirection, Call for contributors` 


- **LeetGPU offers a free CUDA playground**: A member announced the launch of [LeetGPU](https://leetgpu.com/), an online playground for writing and executing **CUDA** code without any signup or GPU access required.
   - They encouraged members to try it out and share their feedback.
- **Request for better discussion channels**: A member suggested that important discussions should take place in specific channels, noting that it would help organize communications better.
   - They pointed to the channel IDs for future references to maintain a focused discussion.
- **Seeking contributors for projects**: Another member inquired if any projects were seeking contributors to join, indicating an interest in collaborative efforts.
   - This reflects a willingness within the community to engage and expand project participation.



**Link mentioned**: <a href="https://leetgpu.com/,">LeetGPU</a>: no description found

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1329252890178748466)** (4 messages): 

> `Triton Initialization Errors, tl.gather Limitations, Optimizing Moe Kernels, Performance Issues with Vectorization` 


- **Triton Initialization Error with Pointers**: A member pointed out that using an **int** pointer is incorrect for Triton, suggesting it should be a **float**, as pointers represent memory addresses and should be scalars only.
   - *This correction might resolve various initialization issues related to data types*.
- **tl.gather's Restriction on Constants**: A user reported that **tl.gather** in Triton does not accept **tl.constexpr**, only **tl.tensor**, which is limiting for their use case in optimizing moe kernels.
   - They highlighted that accessing values on-chip with **tl.gather** is essential for implementing algorithms like **cuRadix** using expert occurrences.
- **Performance Drop from Vectorization**: After seeking a solution to avoid using **tl.gather** for constants, a member found that their kernel worked correctly but experienced a **significant performance drop** post-vectorization.
   - They are looking for further optimization strategies to address this performance decline.
- **Improper Use of tl.store and Memory Access**: It was pointed out that using **tl.store** in loops can severely hamper performance due to thread execution wait times while transferring memory.
   - The advice given emphasizes using **tl.load** with block pointers for better data access efficiency.



**Link mentioned**: <a href="https://github.com/sgl-project/sglang/pull/2913">[Triton] try to optimzie triton moe kernel implmenet with vectorization and tl.gather triton-3… by yiakwy-xpu-ml-framework-team · Pull Request #2913 · sgl-project/sglang</a>: MotivationWhen debugging problematic moe cuda kernels (illegal memory access), I tried to optimize triton moe kernels. The triton moe kernel basically implements radix sort using occurrencies (fi...

  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1329421211318358147)** (2 messages): 

> `ONNX to TensorRT conversion, Myelin CUDA error` 


- **Myelin CUDA Error during ONNX Conversion**: A member encountered a **CUDA error 400** while attempting to convert models from **ONNX** to **TensorRT**, specifically at the `__myl_Res` kernel.
   - They are seeking potential **solutions** to address this conversion issue.
- **Seeking Solutions for Conversion Issues**: A user is requesting help regarding the **Myelin CUDA error** faced during the ONNX to TensorRT conversion.
   - The conversation reflects a desire for community assistance in troubleshooting this specific error.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1329444313821610065)** (4 messages): 

> `Torchinductor, Torch Compile, Machine Learning Frameworks, Caffe's Historical Context` 


- **Torchinductor's Informative Discussion**: A blog post was shared discussing **Torchinductor**, described as a PyTorch native compiler featuring **define-by-run IR** and **symbolic shapes**. The discussion highlighted the informative nature of the content and included many valuable links.
   - Despite being a bit outdated, readers were encouraged to check it out, noting its positive reception.
- **Useful Insights on Torch Compile**: A member shared their blog post titled *Dissecting Torch Compile*, which explores the complexities behind modern machine learning tools like **Torch Compile** and how machine learning has evolved over the years. The post includes a link to the blog and its corresponding [GitHub repository](https://github.com/DWarez/torch_compile_blogpost).
   - The author reflects on the transition from **Caffe** and emphasizes how user-friendly modern frameworks have become, though acknowledging the underlying complexities.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://themlsurgeon.substack.com/p/dissecting-torchcompile-surgical">Dissecting torch.compile: Surgical Precision in PyTorch Optimization</a>: You can take a look at the GitHub repository of this blogpost at this link</li><li><a href="https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747">TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes</a>: The PyTorch team has been building TorchDynamo, which helps to solve the graph capture problem of PyTorch with dynamic Python bytecode transformation. To actually make PyTorch faster, TorchDynamo must...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1329186842150699120)** (6 messages): 

> `GPU internalization, MLPerf participation, MI300X node performance, XCD division in MI300A` 


- **GPU Users Need Internalization**: A member highlighted the importance of internalizing GPU knowledge with reference to the **A100 version** of an animation by @vrushankdes.
   - This suggests a focus on deeper understanding for anyone serious about working with graphic processing units.
- **MLPerf Vendors and GPT-3 Architecture**: A question arose regarding how vendors in **MLPerf** access GPT-3 architecture and weights, given they are not open-sourced.
   - This inquiry reflects ongoing curiosity about what resources participants are actually using.
- **Dividing MI300X Enhances Performance**: There’s an exploration on how an **MI300X node** can be divided into shares of 8, 4, or 2, with implications for memory performance.
   - The potential efficiency gain stems from reducing the workload on the **infinity cache**.
- **XCD Separation in MI300 Variants**: Discussion emerged regarding the **division of MI300A** into shares, aligning with the number of **XCDs** known from previous models.
   - This approach parallels earlier techniques with MI250, aimed at optimizing device performance.



**Link mentioned**: <a href="https://fixupx.com/fleetwood___/status/1879511438538281350">Tweet from Fleetwood (@fleetwood___)</a>: Everyone working with GPUs needs to internalise this.A100 version of @vrushankdes&#39;s animation

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1329258402429538384)** (9 messages🔥): 

> `GPU request batching, Kernel implementation feedback, Flash Attention with CUDA, Using Triton for blockwise matmul, Evaluating hardware requirements` 


- **Understanding GPU request batching**: Multiple users can be served with a single GPU as **requests are batched**, but the capacity depends on available **VRAM for KV cache**.
   - To evaluate hardware requirements for serving N users, consider token counts and perform calculations based on model parameters.
- **Seeking feedback on Triton kernels**: A user asked for feedback on their blockwise matmul kernel implementation for a linear layer, mentioning they want to optimize it for GPU performance.
   - *“Yes it's mostly based off the blockwise matmul tutorial…”*, indicating a need for proper guidance on using **tl.fma** for operation combinations.
- **Resources for Flash Attention with CUDA**: A member directed users interested in **Flash Attention with CUDA** to a [GitHub repository](https://github.com/damienjose/cuda-flashattention) for implementation and testing.
   - This resource is especially helpful for newcomers to **CUDA** looking to get started with this technology.
- **YouTube tutorial on performance**: A linked YouTube video was shared as an additional resource for learning about optimized **matrix multiplication** in CUDA.
   - This complements discussions about optimizing Triton implementations for better performance in linear layers.
- **Blockwise matmul tutorial reference**: A member noted that the kernel under review is similar to what is found in **Triton's tutorials**, specifically regarding block-level matrix multiplications.
   - They provided a [link to the tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py) that elaborates on efficient FP16 matrix multiplication.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py">Matrix Multiplication &mdash; Triton  documentation</a>: no description found</li><li><a href="https://github.com/damienjose/cuda-flashattention">GitHub - damienjose/cuda-flashattention</a>: Contribute to damienjose/cuda-flashattention development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1329410107288588348)** (1 messages): 

> `Torch TensorRT Installation, PyTorch 2.1.1 Documentation` 


- **Getting Started with Torch TensorRT**: *A user* expressed challenges regarding the installation of **Torch TensorRT** for **PyTorch 2.1.1**, finding the documentation unhelpful.
   - They are seeking clearer guidance on how to proceed with the installation process.
- **Documentation Disappointment**: The user criticized the **documentation** for **Torch TensorRT**, stating it does not provide adequate support for installation issues.
   - They mentioned feeling frustrated and are looking for more effective resources or community help.


  

---


### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1329567823948812300)** (1 messages): 

> `Linux arm64 Runners, Copilot Chat for Actions Job Failures` 


- **Linux arm64 Runners launched for public repositories**: The team announced the release of **Linux arm64 hosted runners** today, making them available for free in [public repositories](https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/). This addition supports running workflows on ARM architecture, enhancing versatility for developers.
- **Copilot chat now resolves Actions job failures**: The ability to ask **Copilot chat** about Actions job failures is now generally available, accessible via the **“Explain Error”** feature. Users can activate this feature directly from the **PR mergebox** or the **Actions Job Page** to discuss job failure reasons and resolutions.
   - This new functionality improves the troubleshooting experience by providing direct interaction with Copilot for insights on error logs via screenshots such as those shown [here](https://github.com/user-attachments/assets/04ffd085-cede-4342-b75c-7a80dbff7be9) and [here](https://github.com/user-attachments/assets/57c0eb6b-d567-4a95-becc-4edca865c351).



**Link mentioned**: <a href="https://github.blog/changelog/2025-01-16-linux-arm64-hosted-runners-now-available-for-free-in-public-repositories-public-preview/">Linux arm64 hosted runners now available for free in public repositories (Public Preview) · GitHub Changelog</a>: Linux arm64 hosted runners now available for free in public repositories (Public Preview)

  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: is there any active topic that I can contribute
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1329551300626743328)** (1 messages): 

> `Happy New Year 2025, ArXiv Submission, Researcher Endorsement` 


- **Happy New Year Wishes for 2025**: A member shared their enthusiasm with a friendly greeting, wishing everyone a **Happy New Year 2025**!
   - This festive shoutout sets a positive tone for the discussions ahead.
- **Seeking Endorsement for ArXiv Submission**: A member is looking for endorsements for a paper submission in the **cs.LG** category on ArXiv, highlighting their request for support.
   - They expressed uncertainty about the presence of researchers within the engineer-dominated community.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1329223651194835016)** (12 messages🔥): 

> `Modal Registry for Popcorn Bot, GPU Type Management, nvidia-smi and deviceQuery, Discord Leaderboard Integration, Function Versioning in Modal` 


- **Systematizing Modal Registry for Popcorn Bot**: Members discussed strategies for systematizing the **modal registry** for the Popcorn bot, including challenges with **partially applying** functions to set GPU types.
   - They proposed creating multiple Modal Functions for each GPU type instead of trying to apply a single function generically.
- **GPU Type Introspection Plans**: Plans were shared to enable easier introspection of GPU types from within Modal Functions using **modal.container.gpu** in the future.
   - Currently, workarounds may involve setting GPU architecture directly from the Discord bot, potentially allowing experimentation with compute capabilities.
- **Using nvidia-smi for GPU Capabilities**: A method was suggested to use `nvidia-smi` or the **deviceQuery** utility to determine GPU compute capabilities directly within the script.
   - This allows developers to optimize the GPU usage dynamically, benefiting from tailored performance between different architectures.
- **Exploring Function Versioning**: The idea was presented to maintain several versions of a function, each configured with distinct infrastructure parameters.
   - This capability was highlighted as a desirable feature for facilitating a **Discord leaderboard** integration.
- **Considering Niche Features**: One member noted that allowing calls to endpoints with incorrect compile architectures could serve niche purposes, like comparing speed differences across architectures.
   - This flexibility might introduce unexpected benefits for advanced users interested in performance comparisons.



**Link mentioned**: <a href="https://stackoverflow.com/questions/40695455/what-utility-binary-can-i-call-to-determine-an-nvidia-gpus-compute-capability))">What utility/binary can I call to determine an nVIDIA GPU&#x27;s Compute Capability?</a>: Suppose I have a system with a single GPU installed, and suppose I&#x27;ve also installed a recent version of CUDA.&#xA;&#xA;I want to determine what&#x27;s the compute capability of my GPU. If I coul...

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1329306348592038020)** (3 messages): 

> `Documentation for Kernel Options, Onboarding Assistance` 


- **Collaboration on Documentation**: A member confirmed that the current document could work for now, and expressed willingness to provide more extensive documentation in the future.
   - They indicated they would reach out if more information becomes available.
- **Kernel Options Listed**: Another member shared that they have listed several kernel options in the document and are ready to assist with onboarding if needed.
   - They expressed enthusiasm about helping others try out the suggested kernels.
- **Member to Review Document**: A member acknowledged the kernel options and stated they would review the document first.
   - This indicates ongoing collaboration and willingness to engage with the provided resources.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1329192774813094063)** (29 messages🔥): 

> `LLM Surprises vs Unsurprises, Discord LLM Recommendations, Model Distillation Approach, ChatGPT Poignant Moment, Document Handling Techniques` 


- **LLMs Struggling Between Surprise and Unsurprise**: A member expressed concern that current LLM movements, particularly softmax, have weaknesses when balancing **surprise and unsurprise** in their outputs, advocating for a focus on the surprise aspect.
   - *A viable model should be more focused on surprise part than unsurprise part*.
- **Searching for a Discord LLM**: An inquiry was made for recommendations on a **Discord LLM** that accommodates conversation, as the bot seen by members did not meet the specific needs.
   - Another member clarified their curiosity about jailbreaking a model to enhance its capabilities.
- **Discussing Model Distillation Techniques**: A member discussed the **model distillation approach**, involving generating outputs from a teacher model to teach a student model with a focus on the specifics without excess data.
   - Concerns were raised about whether the student model would still be grounded in naturalistic data distribution even with a targeted approach.
- **ChatGPT's Wholesome Interaction**: A heartwarming story recounted how a child interacted with **ChatGPT**, asking about various items on a desk, and ChatGPT responsibly characterized a vape as a 'thing for adults'.
   - *The fact it even recognised it as a vape blows my mind*.
- **Techniques for Handling Large Documents**: Members debated methods for dealing with **large documents** efficiently, highlighting vector databases and embeddings as common solutions.
   - One member recommended **Langchain** for handling documents and suggested tutorials for beginners.



**Link mentioned**: <a href="https://python.langchain.com/docs/tutorials/">Tutorials | 🦜️🔗 LangChain</a>: New to LangChain or LLM app development in general? Read this material to quickly get up and running building your first applications.

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1329255997893644378)** (11 messages🔥): 

> `New Google Research Architecture, OpenAI API Policy Changes, OpenAI's Compute Costs, Tensor Product Attention (TPA), Daily Paper Discussion Recording` 


- **Google Research proposes new architecture**: Today, the group will explore a new architecture from Google Research that they claim **outperforms transformers** in certain areas, with a focus on method and results. The discussion will be based on the paper found [here](https://arxiv.org/abs/2501.00663).
   - Members expressed interest in the potential connection to **Gemini 1.5**.
- **OpenAI modifies API data usage policy**: OpenAI will no longer use data from its API for model training by default unless organizations **opt in** to that practice. These changes are part of an effort to address criticism from developers and users alike.
   - Concerns were raised regarding the **default data training** that previously occurred without explicit consent.
- **OpenAI's soaring compute expenses**: Reports indicate that OpenAI could spend **$4 billion** on Microsoft’s servers for inference workloads this year, requiring additional funding to cover substantial losses. The costs of training, particularly for ChatGPT, are also projected to soar to around **$3 billion**.
   - Details were shared about the specific **discounts on A100 server rates** received by OpenAI from Microsoft Azure.
- **Introduction of Tensor Product Attention (TPA)**: A new paper proposes Tensor Product Attention (TPA), an innovative mechanism that minimizes **KV cache size** during inference in language models. The architecture, named T6, shows promising results in empirical evaluations against standard transformer models.
   - The author mentioned that future developments will integrate a **Flash version of TPA** for enhanced performance.
- **Recording of Daily Paper Discussion available**: A recording of yesterday's Daily Paper Discussion titled 'Solving the ARC-AGI AI Benchmark with ICOM' is accessible [here](https://youtu.be/WeJuLJrVhf4). The session featured notable technical challenges like echo issues caused by OBS.
   - Participants were encouraged to review the discussion and its insights regarding the benchmark.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their...</li><li><a href="https://arxiv.org/abs/2501.06425">Tensor Product Attention Is All You Need</a>: Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor...</li><li><a href="https://youtu.be/WeJuLJrVhf4">Daily Paper Discussion: Solving the ARC-AGI AI Benchmark with ICOM</a>: I apologize for not realizing in advance that OBS has a bad romance with Discord, which causes other people speaking on Discord to echo because the software ...</li><li><a href="https://github.com/tensorgi/T6/blob/d4f6168852397a7b0b0d9fd65326bb91976c7067/model/T6.py#L64">T6/model/T6.py at d4f6168852397a7b0b0d9fd65326bb91976c7067 · tensorgi/T6</a>: The official implementation of Tensor ProducT ATTenTion Transformer (T6) - tensorgi/T6</li><li><a href="https://www.datacenterdynamics.com/en/news/openai-training-and-inference-costs-could-reach-7bn-for-2024-ai-startup-set-to-lose-5bn-report/?ref=ai-recon.ghost.io">OpenAI training and inference costs could reach $7bn for 2024, AI startup set to lose $5bn - report</a>: Details leak about its Microsoft Azure compute cluster</li><li><a href="https://techcrunch.com/2023/03/01/addressing-criticism-openai-will-no-longer-use-customer-data-to-train-its-models-by-default/">Addressing criticism, OpenAI will no longer use customer data to train its models by default | TechCrunch</a>: OpenAI has changed its developer policy, adding data retention options and clarifying its use of customer data usage.</li><li><a href="https://community.openai.com/t/does-the-openai-api-get-access-to-the-data-i-send-it-or-store-the-data/599538/2">Does the openai API get access to the data I send it or store the data</a>: Welcome to the community!  Answer via OpenAI’s kapa.ai implementation on Discord…   OpenAI places a high priority on data security and privacy. According to the information provided in the extracts:  ...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1329184507776991295)** (5 messages): 

> `4090 GPU failures, 2-slot 4090 availability` 


- **Heavy 4090s lead to high failure rates**: Due to their weight, the **4090 GPUs** experience a high failure rate causing **PCB cracks** and **BGA failures**. To address this, some are being re-packaged into **2-slot 4090s** in China.
- **Availability of 2-slot 4090s on eBay**: A member suggested checking **eBay** for **2-slot 4090s**, highlighting there are many options listed.
   - One specific listing, featuring an [OEM 48GB RTX 4090 Founders Edition](https://www.ebay.com/itm/126885374543) with expedited shipping from Greater China, has gained **23 views in the last 24 hours**.



**Link mentioned**: <a href="https://www.ebay.com/itm/126885374543">OEM 48GB RTX 4090 Founders Edition Dual width GPU Graphics card Ganming/ Server  | eBay</a>: no description found

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1329193055957549066)** (38 messages🔥): 

> `Francois Chollet's AI Lab Ndea, Curator for Synthetic Data Generation, Titans Memory Architecture, HAL Holistic Agent Leaderboard, Harvey AI Funding Raise` 


- **Francois Chollet launches new AI Lab, Ndea**: Francois Chollet announced his collaboration with Mike Knoop to start [Ndea](https://x.com/fchollet/status/1879583863368032432), focusing on **deep learning-guided program synthesis** aimed at advancing AI innovation.
   - They are pursuing a unique path to enhance AI's capabilities in **adaptation** and **invention**.
- **Introducing Curator, a new Synthetic Data Tool**: [Curator](https://x.com/madiator/status/1879579213554147665), an open-source library, aims to streamline **high-quality synthetic data generation**, vital for training LLMs and agents.
   - This tool has reportedly improved productivity in creating post-training datasets by **10x**.
- **Titans: A Revolutionary Memory Architecture**: The new **Titans architecture** introduces a meta in-context memory that can memorize at test time, potentially outperforming existing models, including **GPT-4**.
   - This development could effectively scale to a context window larger than **2M**, redefining memory usage in AI models.
- **HAL: A Leaderboard for AI Agent Evaluation**: A new initiative called [HAL](https://x.com/sayashk/status/1879932823668498576) has been introduced to evaluate AI agents across **11 benchmarks** and over **90 agents**.
   - It raises important questions about the costs and effectiveness of reasoning models compared to standard language models.
- **Harvey AI Gains Significant Investment**: The legal startup **Harvey** is reportedly raising a **$300M** round from Sequoia, valuing it at **$3 billion**.
   - This follows a previous round of **$100M** at **$1.5 billion**, while their revenue estimates were **$30M**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blogs.worldbank.org/en/education/From-chalkboards-to-chatbots-Transforming-learning-in-Nigeria">From chalkboards to chatbots: Transforming learning in Nigeria, one prompt at a time</a>: &#34;AI helps us to learn, it can serve as a tutor, it can be anything you want it to be, depending on the prompt you write,&#34; says Omorogbe Uyiosa, known as &#34;Uyi&#34; by his friends, a student...</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>: no description found</li><li><a href="https://x.com/synthesiaio/status/1879475235390660833?s=46">Tweet from Synthesia 🎥 (@synthesiaIO)</a>: 🎉 Big news: We’ve raised $180 million in Series D funding 🎉Plenty of work still ahead, but the path forward has never been clearer.Of course, none of this would be possible without our amazing custo...</li><li><a href="https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/">Rethinking materials innovation with AI</a>: Microsoft researchers introduce MatterGen, a model that can discover new materials tailored to specific needs—like efficient solar cells or CO2 recycling—advancing progress beyond trial-and-error expe...</li><li><a href="https://x.com/shawnup/status/1880004026957500434">Tweet from Shawn Lewis (@shawnup)</a>: My o1-based AI programming agent is now state of the art on SWE-Bench Verified! It resolves 64.6% of issues.This is the first fully o1-driven agent we know of. And we learned a ton building it.</li><li><a href="https://x.com/sayashk/status/1879932823668498576">Tweet from Sayash Kapoor (@sayashk)</a>: How expensive are the best SWE-Bench agents? Do reasoning models outperform language models? Can we trust agent evaluations?📢 Announcing HAL, a Holistic Agent Leaderboard for evaluating AI agents, wi...</li><li><a href="https://x.com/samuel_colvin/status/1879627376990224417">Tweet from Samuel Colvin (@samuel_colvin)</a>: We&#39;ve just released @Pydantic AI v0.0.19.This comes with the biggest new feature since we announced PydanticAI — graph support!I was originally cynical about graphs, but I&#39;m now really excited...</li><li><a href="https://x.com/fchollet/status/1879583863368032432">Tweet from François Chollet (@fchollet)</a>: I&#39;m joining forces with @mikeknoop to start Ndea (@ndeainc), a new AI lab.Our focus: deep learning-guided program synthesis. We&#39;re betting on a different path to build AI capable of true inven...</li><li><a href="https://x.com/behrouz_ali/status/1878859086227255347">Tweet from Ali Behrouz (@behrouz_ali)</a>: Attention has been the key component for most advances in LLMs, but it can’t scale to long context. Does this mean we need to find an alternative? Presenting Titans: a new architecture with attention ...</li><li><a href="https://x.com/madiator/status/1879579213554147665?s=46">Tweet from Mahesh Sathiamoorthy (@madiator)</a>: We are happy to announce Curator, an open-source library designed to streamline synthetic data generation!High-quality synthetic data generation is essential in training and evaluating LLMs/agents/RAG...</li><li><a href="https://x.com/pitdesi/status/1879982274831347890?s=46">Tweet from Sheel Mohnot (@pitdesi)</a>: Harvey, the AI for law firms, is raising another round from Sequoia ($300M at $3B).Last round was Series C in July, $100M at $1.5B led by GV.They were estimated to have $30M of revenue then, wonder wh...</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Ethan Mollick (@emollick)</a>: New randomized, controlled trial of students using GPT-4 as a tutor in Nigeria. 6 weeks of after-school AI tutoring = 2 years of typical learning gains, outperforming 80% of other educational interven...</li><li><a href="https://x.com/emollick/status/1879633485004165375?s=46&t=6FDPaNxZcbSsELal6Sv7">Tweet from Ethan Mollick (@emollick)</a>: New randomized, controlled trial of students using GPT-4 as a tutor in Nigeria. 6 weeks of after-school AI tutoring = 2 years of typical learning gains, outperforming 80% of other educational interven...</li><li><a href="https://github.com/openai/openai-realtime-agents">GitHub - openai/openai-realtime-agents: This is a simple demonstration of more advanced, agentic patterns built on top of the Realtime API.</a>: This is a simple demonstration of more advanced, agentic patterns built on top of the Realtime API. - openai/openai-realtime-agents</li><li><a href="https://www.youtube.com/watch?v=x8jFFhCLDJY">Google Research Unveils &quot;Transformers 2.0&quot; aka TITANS</a>: Have we finally cracked the code on how to give models &quot;human-like&quot; memory? Watch to find out!Join My Newsletter for Regular AI Updates 👇🏼https://forwardfu...</li><li><a href="https://www.forbes.com/sites/philkirschner/2025/01/15/did-ai-cause-those-layoffs-ny-employers-may-have-to-disclose/?utm_source=chatgpt.com">Did AI Cause Those Layoffs? NY Employers May Have To Disclose.</a>: New York State announced a significant step to address the potential workforce impacts of AI by requiring businesses to disclose layoffs explicitly tied to AI adoption</li><li><a href="https://buttondown.com/ainews/archive/ainews-titans-learning-to-memorize-at-test-time/">[AINews] Titans: Learning to Memorize at Test Time</a>: Neural Memory is all you need. AI News for 1/14/2025-1/15/2025. We checked 7 subreddits, 433 Twitters and 32 Discords (219 channels, and 2812 messages) for...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1329234711913959515)** (4 messages): 

> `Modular subreddit, GitHub organization move` 


- **Modular Launches Official Subreddit**: There's now an **official Modular subreddit**! Join the community over at [r/ModularAI](https://www.reddit.com/r/ModularAI/) 🥳.
   - *This is the way!* exclaimed a member in response to the announcement.
- **Transition of GitHub Repos**: Modular's public GitHub repos have migrated from the [**ModularML**](https://github.com/modularml) organization to the [**Modular**](https://github.com/modular) organization.
   - All previous links should redirect automatically, but the team encourages reporting any **unexpected issues** that arise during the transition.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1329253563108425738)** (28 messages🔥): 

> `Recursive Types in Mojo, SIMD Performance Concerns, Variadic Lists to Dictionary Implementation Issues` 


- **Recursive Types Handling in Mojo**: A member pointed out challenges in implementing recursive types in Mojo, suggesting using `UnsafePointer` might cause issues, and recommending the use of a copy constructor on List.
   - They also referred to some issues encountered while debug running, and hinted that recursive types are not well supported currently.
- **Performance Footguns with SIMD**: Members discussed possible *performance footguns* with SIMD, pointing out that SIMD may not always guarantee better performance over scalar versions depending on the architecture.
   - One member noted that whether SIMD offers a performance boost heavily depends on the specific CPU and its optimizations.
- **Variadic List to Dictionary Conversion Problems**: A member shared difficulties encountered while trying to split a VariadicList into a dictionary, experiencing unexpected behavior with string capture assignments.
   - Another member provided a workaround by copying arguments into separate lists before building the dictionary, avoiding the broken behavior.
- **Optional Argument Handling in Mojo**: Discussions emerged surrounding issues with optional arguments in a Mojo class, as one member faced segmentation faults when evaluating to None.
   - Recommendations included checking with specific GitHub issues for fixes and examples pertaining to optional arguments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html">Ice Lake AVX-512 Downclocking</a>: Examining the extent of AVX related downclocking on Intel’s Ice Lake CPU</li><li><a href="https://github.com/modularml/mojo/issues/3950">[Help wanted] Evaluating optional argument to not None gives segmentation fault · Issue #3950 · modular/mojo</a>: Issue description I have a class that needs an optional argument. When evaluating to None, it gives an error. If evaluating to None gives an error, how am i supposed to evaluate it? But also, i may...</li><li><a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modular/mojo</a>: Bug description Running a mojo script using the debugger seg faults, as opposed to when running regular mojo, which runs to completion (although I have noticed strange behavior in the regular scrip...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1329188117319647262)** (2 messages): 

> `Agent Identity Hackathon, Xeno Grant Applications` 


- **Join the Agent Identity Hackathon!**: Come join [Plastic Labs](https://plasticlabs.ai) & [Betaworks](https://betaworks.com) for an agent identity hackathon to kick off [Xeno Grant](https://xenogrant.org) with **$5,000** in prizes for standout projects focusing on agent identity.
   - Participants can compete solo or in teams, with food and drinks provided, and a kickoff mixer happening the night before.
- **Xeno Grant Applications Closing Soon!**: Xeno Grant applications are set to close on **Sunday, January 26th**, and participants are encouraged to share their GitHub, portfolio, or personal site during registration for approval and waitlist priority.
   - This opportunity invites all agent builders to engage and meet the **Plastic/Betaworks grants committee** if they are applying.



**Link mentioned**: <a href="https://lu.ma/5rlcrlpb">Xeno Grant: Agent Identity Hackathon · Luma</a>: Come join Plastic Labs &amp; Betaworks for an agent identity hackathon to kick off Xeno Grant (powered by $YOUSIM).$5,000 in prizes for the most compelling…

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1329188540789293228)** (9 messages🔥): 

> `LiveCodeBench Update, Cerebras Yield Problem, Contextual AI Platform Launch, TGI Backend Expansion, SWE-bench Multimodal Evaluation` 


- **LiveCodeBench Hits Milestone with 167 New Problems**: The latest update for **LiveCodeBench** adds **167 new problems**, bringing the total to **880**—a significant increase from **400** in version 1.
   - This update showcases improved reasoning models like o1, **Gemini-Flash**, and the upcoming R1, generating excitement in the community.
- **Cerebras Rethinks Chip Yields**: Cerebras promotes their **wafer-scale chip** as being **50x larger** than traditional chips while achieving comparable yields, challenging conventional semiconductor wisdom.
   - Their detailed analysis compares yields between the **Cerebras Wafer Scale Engine** and an H100-sized chip, asserting better fault tolerance despite size.
- **Contextual AI Platform Celebrates Launch**: The **Contextual AI Platform**, powered by **Meta’s Llama 3.3**, runs on **Google Cloud** and leverages **NVIDIA GPUs**, marking a significant milestone.
   - The team expressed gratitude towards partners and investors including **NVIDIA**, **Meta**, and other venture capitalists who contributed to this achievement.
- **TGI Becomes the Keras of Inference Frameworks**: Hugging Face's **Text-Generation-Inference (TGI)** continues to evolve, now supporting multiple backends including **AMD** and **Google TPU**.
   - Lauded for its ease of deployment, TGI offers a no-code solution helping developers to efficiently run large-language models on a variety of platforms.
- **SWE-bench Introduces Multimodal Evaluation Code**: The new **SWE-bench MM** includes JavaScript issues focused on visual components, enhancing multimodal evaluation capabilities.
   - Examples of new issues involve rendering problems like 'map isn’t rendering correctly' and UI bugs, expanding the scope of their evaluation framework.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://cerebras.ai/blog/100x-defect-tolerance-how-cerebras-solved-the-yield-problem">100x Defect Tolerance: How Cerebras Solved the Yield Problem - Cerebras</a>: no description found</li><li><a href="https://x.com/jyangballin/status/1879990781030854897">Tweet from John Yang (@jyangballin)</a>: SWE-bench Multimodal evaluation code is out now!SWE-bench MM is a new set of JavaScript issues that have a visual component (‘map isn’t rendering correctly’, ‘button text isn’t appearing’).</li><li><a href="https://x.com/ContextualAI/status/1879563309080547376">Tweet from Contextual AI (@ContextualAI)</a>: The Contextual AI Platform is proudly built with Meta’s Llama 3.3, runs on Google Cloud, and is trained on NVIDIA GPUs.We’re extremely proud of this milestone and want to thank all of our customers, p...</li><li><a href="https://huggingface.co/blog/tgi-multi-backend">Introducing multi-backends (TRT-LLM, vLLM) support for Text Generation Inference</a>: no description found</li><li><a href="https://x.com/StringChaos/status/1879619028651745287">Tweet from Naman Jain (@StringChaos)</a>: 📢 Excited to share the 5th update for LiveCodeBenchWe have added 167 new problems this time and collected 880 problems overall, over two-fold increase from 400 problems in v1Leaderboard ⬇️- 🥇 open a...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1329252427832229980)** (8 messages🔥): 

> `AMD GPU Support, Intel Sponsorship, Ai2 Funding` 


- **AMD Should Invest in Ai2**: A member suggested that **AMD** should fund all of **Ai2** and provide substantial GPU resources, proposing a figure of **$10k** per member.
   - *Why not leverage AMD's GPUs?* is the potential question from community members, emphasizing the need for better model accessibility.
- **Local Proximity to AMD HQ Sparks Ideas**: Another member mentioned being just **200 yards** from the AMD headquarters in **Santa Clara**, contemplating walking over to speak with **Lisa** Su directly.
   - They jokingly suggested DMing her to advocate for more GPU distribution, stating, *who else is going to take them?*
- **Competition Highlights with Tensorwave**: Discussion included a link to **Tensorwave**, highlighting their focus on enterprise-level AI compute solutions powered by **AMD's MI300X** accelerators.
   - Tensorwave boasts benefits like *faster, scalable, and easier to use* systems, emphasizing immediate availability as a launch partner for the MI300X.
- **Intel's Strategic Sponsorship**: One member referenced **Intel's** smart move to sponsor projects at **Stability AI**, contrasting it with AMD's current engagement approach.
   - This brings attention to the competitive landscape and the strategies companies adopt to secure partnerships in the AI space.



**Link mentioned**: <a href="https://tensorwave.com/">Access MI300X GPU Today | TensorWave | The MI300X Cloud</a>: Access AMD MI300X GPUs today on TensorWave Cloud. Contact us today to get started.

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1329272230076481709)** (2 messages): 

> `Terminology for non-reasoning models, GPT-4o, Autoregressive models` 


- **Defining Non-Reasoning Models**: A discussion arose about the correct term for non-reasoning models, particularly contrasting it with reasoning models like **o1**.
   - *Vanilla ass basic autoregressive model* was suggested as the informal term for **GPT-4o**.
- **Clarification on Model Types**: Participants sought clarity on how **GPT-4o** fits into the spectrum of model types, particularly against reasoning-based models.
   - The term *vanilla* signifies a more basic autoregressive approach compared to other more complex architectures.


  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/)** (1 messages): 

natolambert: wow a throwback. this should've been more....
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1329196458624290928)** (4 messages): 

> `Human Decision-Making vs AI, Technology Cycle Expectations, Social Movements against AI, Public Understanding of LLMs` 


- **Humans with Next Best Action Systems Outperform AI**: A participant noted that a human equipped with a **next best action system** can be difficult to surpass in decision-making complexity.
   - This reflects ongoing discussions about the practical capabilities of human intuition versus machine learning.
- **Misestimated Technology Cycles**: A member expressed skepticism about the speed at which tech innovations can fundamentally change our economy, suggesting that expectations of an **infinity box** redesigning our infrastructure are unrealistic.
   - They anticipated significant **social upheaval** if such a rapid transformation were to occur.
- **Absence of Social Movements Against AI**: Commentators noted that serious social movements opposing AI are yet to emerge, which could significantly impact the future landscape of technology.
   - This raises questions about public readiness and responsiveness to emerging technologies.
- **Public's Struggles with LLMs**: One participant expressed frustration over how poorly people utilize large language models (LLMs), likening the technology to **magic** that users fail to grasp.
   - Despite providing clear instructions, users often generate **trite results**, revealing a gap in understanding LLM functionalities.


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1329236127244353586)** (1 messages): 

> `Project Aria, Meta communications` 


- **Project Aria Updates Subscription Launched**: Meta announced a more open version of [Project Aria](https://www.projectaria.com/) where users can subscribe for the latest updates.
   - *By providing your email, you agree to receive marketing related electronic communications from Meta* regarding news and events about Project Aria.
- **Data Policy Transparency from Meta**: Meta offers insights on how they handle user data, urging users to read their [Data Policy](https://www.projectaria.com/privacy-policy/).
   - They emphasize that users can unsubscribe from communications anytime via the unsubscribe link in their emails.



**Link mentioned**: <a href="https://www.projectaria.com/">Introducing Project Aria, from Meta</a>: Project Aria is a research program from Meta, to help build the future responsibly. Project Aria unlocks new possibilities of how we connect with and experience the world.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1329212731655585812)** (3 messages): 

> `Vellum AI partnership, Neomagus winning LLM x Law Hackathon, Women in AI RAG Hackathon` 


- **Vellum AI Survey Partnership**: We are delighted to have partnered with **@vellum_ai** on this survey; check out the [use-case data!](https://t.co/0zvs7ZkKJQ)
- **Neomagus Triumphs at LLM x Law Hackathon**: Learn how **Neomagus** won the LLM x Law Hackathon with a project aimed at ensuring the **accuracy of AI-generated legal information**.
   - Their solution featured **real-time verification** of legal references and immediate flagging of inaccuracies ([more details](https://t.co/jEqZrnn11H)).
- **Women in AI RAG Hackathon Invitation**: Women in technology are invited to the **Women in AI RAG Hackathon** in Palo Alto, focusing on **Retrieval-Augmented Generation** with open-source [vector database @zilliz_universe](https://t.co/2Bzg80dh29).
   - This all-day event provides an opportunity to connect with fellow women technologists and mentors ([more information](https://t.co/ey8ebq9fbx)).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1329274593835352084)** (16 messages🔥): 

> `Signup Issues Resolved, Metadata Filtering in ChromaDB, LLM Integration with llmlingua, Tag Extraction Approaches` 


- **Signup Issues Resolved**: Users experienced issues with signing up, but it was confirmed that problems stemmed from **auth upgrades**, which have now been fixed.
   - One user reported successfully logging in after the fix, clarifying that it was a **temporary error**.
- **Metadata Filtering in ChromaDB**: A user inquired about using **ExactMatchFilters** with metadata for filtering legal cases in a large vector store created from thousands of documents.
   - They expressed concerns over creating a sub-index for routing, questioning its effectiveness and performance compared to existing filtering methods.
- **LLM Integration with llmlingua**: A member discussed enhancing LlamaIndex's integration with **llmlingua2**, but faced issues with linting at the MakeFile during the process.
   - Another member offered assistance by suggesting the installation of `pre-commit` for automated linting or running `make lint` manually for fixes.
- **Tag Extraction Approaches**: A user asked whether it would be more efficient to use separate LLM calls for refining product descriptions and extracting tags, or to combine them into a single call.
   - They noted the tradeoffs between **tag quality** and **latency/costs**, given their high volume of calls per day.



**Link mentioned**: <a href="https://github.com/run-llama/llama_index/pull/17531">Add longllmlingua2 integration by tituslhy · Pull Request #17531 · run-llama/llama_index</a>: DescriptionAdded LLMLingua 2 integration! LLMLingua2 is an improvement over LLMLingua1 using a smaller sized prompt compression method trained via data distillation from GPT-4 for token classifica...

  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1329464136060502018)** (1 messages): 

> `text-to-SQL pipeline` 


- **Text-to-SQL Pipeline Done in 20 Minutes**: A member shared their experience of creating a **text-to-SQL pipeline** in just **20 minutes**, expressing surprise at the ease of the process.
- **Initial Success with Text-to-SQL**: This was their first attempt at such a task, emphasizing the **user-friendly nature** of the tool utilized.
   - They couldn't believe how easy the setup was, highlighting a significant learning opportunity.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/)** (1 messages): 

jimmc414_00230: Any word on when we might expect DSPy v3?
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1329551564049879100)** (2 messages): 

> `dspy ReAct usage, Addition function error, LLama model issues` 


- **Error in dspy ReAct function usage**: A user is encountering an error stating that the tool addition is not designed to calculate the sum of two numbers and is lacking necessary arguments.
   - They mentioned using the LLama model hosted via LM-Studio and sought assistance from the community.
- **Issue with addition function returning incorrect output**: The addition function is defined correctly, but there is a broken syntax as it returns 'retur' instead of 'return'.
   - This could be contributing to the breakdown in functionality, leading to users receiving error messages when executing the function.


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1329472534407544884)** (4 messages): 

> `Ideal chat template, Torchtune integration` 


- **Seeking the Ideal Chat Template**: @duh_kola asked about the **ideal chat template**, prompting responses regarding ChatML or Llama3 if that's the route being taken.
   - The query highlights ongoing discussions about optimizing chat interfaces.
- **Torchtune Integration Challenges**: A member noted that integrating **Torchtune** currently involves *ripping out a lot of things*, suggesting significant complexity in the process.
   - The conversation indicates that attention to this integration has lagged, as **caseus_** humorously remarked that it's been a while since this was addressed.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1329489319039074444)** (1 messages): 

> `Cooperative AI Summer School, Confirmed Speakers, Application Details, Sortition in Democracy, Financial Assistance` 


- **Cooperative AI Summer School Applications Open**: Applications for the **Cooperative AI Summer School** are open until **7th March 2025**, taking place from **9th–13th July 2025** in Marlow, near London.
   - The program is aimed at students and early-career professionals in AI, computer science, and related fields, focusing on networking and career development.
- **Notable Speakers Announced**: Confirmed speakers include **Michael Wellman** from the University of Michigan and **Zarinah Agnew** from The Collective Intelligence Project, who bring diverse expertise to the school.
   - **Ariel Procaccia** from Harvard University will also present, discussing randomized participant selection algorithms in citizens’ assemblies.
- **Comprehensive Program Insights**: The summer school will explore both foundational concepts and cutting-edge research in **cooperative AI**, aimed at impact-driven candidates.
   - Lectures will cover the objectives and methods at the forefront of cooperative AI research.
- **Financial Assistance Offered**: Financial assistance is available to ensure accessibility for participants at the summer school.
   - This initiative aims to support candidates with demonstrated interest in cooperative AI, regardless of their financial situation.
- **Learn More on Cooperative AI Website**: For more information and to apply, visit the **Cooperative AI website** at [cooperativeai.com](https://www.cooperativeai.com/summer-school/summer-school-2025).
   - The website also features FAQs about the application assessment process.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://strategicreasoning.org/michael-p-wellman/">Michael P. Wellman &#8211; Strategic Reasoning Group</a>: no description found</li><li><a href="https://www.zarinahagnew.com/">z a r i n a h  :   a g n e w</a>: bio : vision statement</li><li><a href="https://procaccia.info/">Home - Ariel Procaccia</a>: no description found</li><li><a href="https://www.cooperativeai.com/summer-school/summer-school-2025">Cooperative AI</a>: no description found
</li>
</ul>

</div>
  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1329235959107162213)** (2 messages): 

> `Cost of solutions, Churn prevention strategies` 


- **Cost Drives Decisions on Solutions**: *Cost* is a significant factor for choosing established solutions, as one member noted it's a major reason to stick with what works.
   - This highlights the importance of budgeting in decision-making for tools and strategies.
- **Inquiry on Churn Prevention Trends**: A member expressed interest in the latest developments in *churn prevention/aversion*, having been away from the space for over two years.
   - They are looking for guidance on where to start to get back up to speed on the current landscape.


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1329206651978387526)** (2 messages): 

> `Bora's Law, Open Interpreter functionalities, Autonomous systems development` 


- **Bora's Law challenges traditional AGI scaling**: A member criticized OpenAI's approach to AGI, suggesting that **intelligence scales with constraints**, not compute, supporting this with insights from [Bora's Law: Intelligence Scales With Constraints, Not Compute](https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci). This principle argues that a focus on scaling compute power overlooks the mathematical relationship essential for genuine intelligence development.
   - *The article suggests that the pursuit of intelligence should shift from brute force to an understanding of how constraints play a critical role.*
- **Concerns about Open Interpreter's code execution**: A member expressed concern that Open Interpreter 1.0 may have eliminated direct code execution features, limiting it to command line operations only. This functionality was important for enhancing efficiency and prompting for LLM learning.
   - There was interest in improving the code execution aspect by adding Python convenience functions to help the LLM learn new skills more effectively.



**Link mentioned**: <a href="https://chrisbora.substack.com/p/boras-law-intelligence-scales-with?r=aszci">Bora&#x27;s Law: Intelligence Scales With Constraints, Not Compute</a>: This is a working paper exploring an emerging principle in artificial intelligence development.

  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1329437460802965588)** (2 messages): 

> `Jamba API Performance, Comparative Analysis with OpenAI` 


- **Jamba API impresses users with performance**: @bjorn02796 shared that they have **Jamba API** running in multiple applications and are very pleased with its performance.
   - *If it has outperformed OpenAI responses*, it raises questions about the current standard of OpenAI.
- **User Feedback Acknowledged**: In response to the feedback, a user, **keepitirie**, expressed gratitude for the positive remarks.
   - This highlights the community's engagement and the appreciation for **Jamba API's** effectiveness.


  

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

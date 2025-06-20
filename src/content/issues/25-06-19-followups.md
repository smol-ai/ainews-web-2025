---
id: MjAyNS0w
title: minor ai followups: MultiAgents, Meta-SSI-Scale, Karpathy, AI Engineer
date: '2025-06-19T05:44:39.731046Z'
---

**a quiet US holiday.**

> AI News for 6/18/2025-6/19/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 6456 messages) for you. Estimated reading time saved (at 200wpm): 571 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A grab bag of followups today:

- Thanks for your submissions for [the MultiAgents debate](https://news.smol.ai/issues/25-06-13-cognition-vs-anthropic)! we link to some of your submissions in [today's Noam Brown pod](https://www.latent.space/i/165741459/on-multi-agents).
- Meta probably tried to [buy SSI](https://x.com/AndrewCurran_/status/1935853120472612955) before the Scale AI + Dan Gross move
- the full Karpathy talk from YC AI SUS is [now up](https://news.ycombinator.com/item?id=44314423) (with [slides](https://www.latent.space/p/s3))
- the first AI Engineer conf talk recordings are now [rolling out](https://www.youtube.com/watch?v=lswTmGrjhVA&list=PLcfpQ4tk2k0W3ORTR-Cr4Ppw6UrN8kfMh&index=110)

---

# AI Twitter Recap

**AI Safety, Alignment, and Regulation**

- **Misalignment from Insecure Code Training**: A new **OpenAI** paper studying how training models like **GPT-4o** to write insecure code triggers broad misalignment has drawn significant attention. [@sama](https://twitter.com/sama/status/1935413406183673957) found it surprising, while [@polynoamial](https://twitter.com/polynoamial/status/1935411224281534756) called it "worrisome" but praised **OpenAI** for investigating mitigations. [@NeelNanda5](https://twitter.com/NeelNanda5/status/1935437543610233016) noted how this new paper covers similar ground to previous work on emergent misalignment. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1935502543800446978) added that the original study wasn't just on "insecure code," making the causal path through inducing a malicious persona unsurprising.
- **AI Regulation in California**: [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1935479129899401243) highlighted a recent report from the **Joint California Policy Working Group on AI Frontier Models** as an "important step" towards balanced AI regulation. He emphasized its points on third-party assessments, transparency, and whistleblower protections, noting California is uniquely positioned to lead on AI governance.
- **"Context Rot" and Memory Control**: The term **"context rot,"** describing the quality degradation of an LLM conversation over time, was [coined on Hacker News and shared](https://twitter.com/zacharynado/status/1935490774654927053). [@zachtratar](https://twitter.com/zachtratar/status/1935491439028531293) commented that robust memory control is critical for business use cases and is why systems like **Embra** use a CRM-like AI memory instead of a black box.
- **Scalable Oversight Research**: [@RyanPGreenblatt](https://twitter.com/RyanPGreenblatt/status/1935407345888280938) shared detailed thoughts on scalable oversight research, expressing optimism for work aimed at improving human oversight of "somewhat smarter AIs." He is most excited by adversarial analysis to prevent subversion, improving outputs in conceptually hard cases like philosophy, and robustly detecting reward hacking.
- **The OpenAI Files**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1935642920662737290) retweeted a post about a large repository of information called **'The OpenAI Files'** detailing internal company events and concerns.

**AI Models & Research**

- **New Model Releases**:
    - **Kyutai Speech-To-Text**: [@reach_vb](https://twitter.com/reach_vb/status/1935655403024498814) provided a detailed breakdown of **Kyutai's** new state-of-the-art speech-to-text models, `stt-1b-en_fr` and `stt-2.6b-en`, which are **CC-BY-4.0** licensed. He highlighted their performance, capable of **400 real-time streams on a single H100 GPU**, and their availability on the **Hugging Face Hub**. The release was also shared by [@clefourrier](https://twitter.com/clefourrier/status/1935701954358890806).
    - **Hunyuan 3D 2.1**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1935524756473921637) retweeted **Tencent's** announcement of **Hunyuan 3D 2.1**, the "first fully open-source, production-ready **PBR 3D** generative model." [@Teknium1](https://twitter.com/Teknium1/status/1935656421506654256) commented on the utility of such models for generating custom 3D models for printing.
    - **Arcee Foundation Models (AFM-4.5B)**: [@arcee_ai](https://twitter.com/code_star/status/1935439879506424295) unveiled their new family of models, starting with **AFM-4.5B**, designed from the ground up for enterprise. The models are powered by data from [@datologyai](https://twitter.com/code_star/status/1935432790046294097), and are described as [legitimately competitive with Gemma and Qwen](https://twitter.com/code_star/status/1935465007892115761).
- **Research Papers & Techniques**:
    - **Robotics & Tactile Sensing**: [@ylecun](https://twitter.com/ylecun/status/1935466674242666831) retweeted the announcement of **e-Flesh**, a new 3D-printable tactile sensor developed at **NYU** that measures deformations in 3D printable elastomers.
    - **Autoregressive U-Nets for Language Modeling**: [@ylecun](https://twitter.com/ylecun/status/1935481068284424355) shared a paper presenting an **autoregressive U-Net** that processes raw bytes and incorporates tokenization inside the model, pooling bytes into words and then word-grams.
    - **Reasoning Models (RLMs)**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1935736112515080314) broke down the three defining characteristics of **Reasoning Models (RLMs)**: post-training with **Reinforcement Learning** (e.g., PPO, GRPO), **inference-time scaling** where the model generates an internal reasoning trace, and **multi-sampling** to choose a consensus answer.
    - **Chain of Thought (CoT) Unfaithfulness**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1935411492146368559) highlighted a new dataset for studying **CoT unfaithfulness** on user-like prompts, noting it's an important area for further research.
    - **Robotics with Symbolic Search + Neural Learning**: A new robotics paper combines **symbolic search** and **neural learning** to build compositional models that generalize to new tasks, described by [@ndea](https://twitter.com/ndea/status/1935484273370501217) as "a neural grammar for a planning programming language."

**Company & Product Updates**

- **OpenAI**: Has started rolling out **Record mode** in the **ChatGPT macOS app** for Pro, Enterprise, and Edu users, as announced by [@OpenAI](https://twitter.com/OpenAI/status/1935419375600926971). Additionally, [@kevinweil](https://twitter.com/kevinweil/status/1935722240009437635) updated that users can now set a recommended model when creating a **Custom GPT**, and paid users can access the full range of models within them.
- **Google DeepMind**: Showcased **Gemini 2.5 Flash-Lite's** capability to [write UI code from visual context](https://twitter.com/GoogleDeepMind/status/1935719933075177764). Meanwhile, [@demishassabis](https://twitter.com/demishassabis/status/1935518641120047317) posted a chart captioned "What relentless progress looks like... 🚀".
- **Anthropic**: The **Claude Code** userbase has **more than tripled** since the Claude 4 launch less than a month ago, according to a post from [@alexalbert__](https://twitter.com/alexalbert__/status/1935714247369228369). To demonstrate its power, [@skirano](https://twitter.com/skirano/status/1935733281888272713) showed that you can spawn subagents within Claude Code just by asking.
- **Jules**: Shipped a [major update to its dev environment](https://twitter.com/julesagent/status/1935478096414785965), including newer versions of **Rust, Node, and Python**, better runtime isolation, and fewer dependency issues.
- **vLLM**: The project has reached **50,000 GitHub stars**, a milestone celebrated by the [@vllm_project](https://twitter.com/vllm_project/status/1935569537858183321).
- **ByteDance**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1935603383014248764) provided context on the **ByteDance Seed** team, explaining they were founded in 2023 but the brand only became externally visible around January 2025, explaining why their emergence seemed sudden. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1935567992852455528) noted that their move into areas like AI for chemical engineering is unsurprising.
- **Meta**: Speculation surrounds **Mark Zuckerberg's** talent acquisition strategy, with a meme from [@dylan522p](https://twitter.com/dylan522p/status/1935454786918432833) illustrating his "FOUNDER MODE masterplan." [@teortaxesTex](https://twitter.com/teortaxesTex/status/1935757454261850154) likened his approach to a rich nerd trying to "cut corners through the world with money."

**AI Engineering, Tools & Frameworks**

- **Agentic AI & Tooling**:
    - **MCP Protocol**: The question of whether the **Model-Provider Communication Protocol (MCP)** kills centralized vector search was explored by [@jerryjliu0](https://twitter.com/jerryjliu0/status/1935473439948890177). He argues for a nuanced "yes and no," suggesting centralized indexes are still needed for fast semantic lookup, while MCP excels at deep interaction and action-taking within SaaS tools. The latest MCP spec update was [shared by @jeremyphoward](https://twitter.com/jeremyphoward/status/1935481114195542291).
    - **LangChain/LangGraph**: **LangGraph Studio** can now be used with agents that are not built on LangGraph, according to [@LangChainAI](https://twitter.com/LangChainAI/status/1935756179319505137). They also shared a guide on [getting the benefits of LangSmith (tracing & evals) without using LangChain or LangGraph](https://twitter.com/LangChainAI/status/1935706402896707657).
    - **Agent Development**: A session from **Factory AI** was highlighted by [@LangChainAI](https://twitter.com/LangChainAI/status/1935409057353056605), breaking down the core characteristics of agentic systems and the shift from AI-assisted coding to fully agent-driven workflows.
- **Evaluation (Evals)**: [@HamelHusain](https://twitter.com/HamelHusain/status/1935460393515892778) cautioned against overfitting evaluation sets, stating that achieving **100% accuracy** likely means your product is "deeply broken" or you are tracking the wrong metrics. He also announced his course on AI Evals is [#1 on Maven](https://twitter.com/HamelHusain/status/1935470907373568026).
- **Developer Tools**:
    - **Outlines**: Version 1.0 of the **Outlines** library for guided text generation has been released and is [now compatible with Ollama](https://twitter.com/ollama/status/1935712844701442245).
    - **Cline**: A new feature in the **Cline** terminal lets users [set a default terminal profile](https://twitter.com/cline/status/1935423329936318795) to prevent commands from failing due to running in the wrong shell.
- **Data Curation & Datasets**: [@reach_vb](https://twitter.com/reach_vb/status/1935444297966604539) pointed to a massive **24 TRILLION** token high-quality dataset. [@code_star](https://twitter.com/code_star/status/1935462275906945428) promoted **DatologyAI** as a source for the "strongest pretraining data in the world," noting it was used for the **AFM-4.5B** model.

**Industry Commentary & Broader Implications**

- **Automation vs. Augmentation in the Workforce**: [@random_walker](https://twitter.com/random_walker/status/1935679764192256328) provided a detailed analysis using **radiology** as a case study, arguing that **Geoff Hinton's** predictions of job replacement were wrong. He suggests the "jobs are bundles of tasks" model is incomplete, as it misses the nuanced, hard-to-specify work at the boundaries between tasks, explaining why AI has led to augmentation, not automation, even when it outperforms humans on benchmarks. [@ClementDelangue](https://twitter.com/ClementDelangue/status/1935750326386409491) agreed, adding it's a "good reminder that you can be a 'godfather of AI' and still utterly wrong."
- **U.S. Immigration Policy and AI Talent**: In a widely shared thread, [@AndrewYNg](https://twitter.com/AndrewYNg/status/1935741989204770837) argued that welcoming high-skilled immigrants and international students is one of the most effective things the U.S. can do to ensure its competitiveness in AI. He expressed deep concern over recent visa policy changes, calling the potential squandering of this advantage a **"huge unforced error"** and highlighted the personal hardships faced by those affected.
- **Conceptual Frameworks for AI Development**: [@_jasonwei](https://twitter.com/_jasonwei/status/1935418236872335397) introduced the concept of the **"description-execution gap"** to predict which tasks will be automated first—those where it's much easier to describe the task than to do it. Separately, [@karpathy](https://twitter.com/karpathy/status/1935779463536755062) commented on a demo of a GUI for LLMs, noting the underlying idea is to "generate a completely ephemeral UI on demand depending on the specific task at hand."
- **Open vs. Closed AI Ecosystems**: [@jeremyphoward](https://twitter.com/jeremyphoward/status/1935450726412783863) expressed continued concern that AI technology will be "locked up inside of one company (OpenAI)." In contrast, [@ClementDelangue](https://twitter.com/ClementDelangue/status/1935705674584924385) stated his preference for focusing on AI as "software 2.0" and bringing its benefits to humanity through **open-source**.
- **The Philosophy of AI Engineering**: [@lateinteraction](https://twitter.com/lateinteraction/status/1935525945806590425) argued that **parsimony** is a better goal than simplicity, advising to "invent Unix first before it makes sense to create lots of small programs." [@hyhieu226](https://twitter.com/hyhieu226/status/1935747480433705150) reminded engineers to stay alert and question intermediate requirements to avoid diverging from first principles.

**Humor/Memes**

- **Industry Satire**: [@typedfemale](https://twitter.com/typedfemale/status/1935577381953241300) joked that **Mark Zuckerberg** should "publicly punish **Yann LeCun** by removing all convnet related functionality from pytorch." [@kyliebytes](https://twitter.com/code_star/status/1935579170001801652) posted the popular "good morning" meme showing a graph of exponentially increasing compute usage.
- **Relatable Engineer Life**: [@gdb](https://twitter.com/gdb/status/1935514803403112665) shared a meme with the caption "ChatGPT for meeting notes." [@agihippo](https://twitter.com/agihippo/status/1935605475279822996) confessed to feeling "so much guilt and shame" after waking up with no jobs running. [@TheZachMueller](https://twitter.com/TheZachMueller/status/1935434078435819925) retweeted a meme depicting the degradation of **FP8 values** after 50 layers of quantization/dequantization.
- **General Humor**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1935498395575271930) posted a screenshot of a complex scientific diagram with the comment "science fucking rocks." [@qtnx_](https://twitter.com/qtnx_/status/1935438614587977791) posted a picture of servers inside a church, captioning it, "you really can just train an LLM in a church."

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Innovative Open Source LLM Infrastructure and Performance Tools

- [**We built this project to increase LLM throughput by 3x. Now it has been adopted by IBM in their LLM serving stack!**](https://i.redd.it/775o8e8hxr7f1.jpeg) ([Score: 392, Comments: 52](https://www.reddit.com/r/LocalLLaMA/comments/1lewhla/we_built_this_project_to_increase_llm_throughput/)): **The post introduces LMCache, an open source tool designed to efficiently offload and load large Key-Value (KV) cache tensors from GPU to DRAM and disk in LLM inference systems, targeting improved throughput (3x in chat apps) by preventing redundant KV cache recomputation in multi-round QA scenarios. The attached graph visually compares 'Time to First Token' (TTFT) at different QPS for vLLM with/without prefix caching versus LMCache: LMCache maintains the lowest and most stable TTFT, highlighting its effectiveness in managing memory constraints and increasing throughput. IBM has adopted LMCache into their open source LLM serving stack ([Github repo](https://github.com/LMCache/LMCache)).** One technical commenter queries whether LMCache supports caching of arbitrary (non-prefix) context KV tensors or primarily persists/reloads prefix caches, given the autoregressive transformer architecture; this prompts clarification of LMCache's distinction from standard prefix caching. Another notes that llama.cpp has similar features, but points out its limitations in user scaling for multi-user environments where VRAM offloading to CPU is needed.
    - Several commenters question the novelty of the project's KV cache, highlighting that prefix-based KV caching is already standard in most major LLM servers. They request clarification on whether the approach supports caching arbitrary sections of text (not just prefixes) and if it includes disk-based cache storage to avoid recomputation, similar to implementations like llama.cpp's prompt cache save/restore functionality.
    - Technical discussion references [llama.cpp's server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#post-slotsid_slotactionsave-save-the-prompt-cache-of-the-specified-slot-to-a-file), which supports saving and restoring prompt caches per slot and provides command-line/REST options for cache persistence and reuse. However, llama.cpp's multi-user serving performance is limited by VRAM, so it's not typically used for heavy user loads without CPU offloading.
    - A technical inquiry is raised about LMCache's ability to handle cache/context reuse across multi-GPU or containerized deployments, especially under memory constraints and frequent cache evictions. Questions center on whether LMCache proactively prefetches context or relies on on-demand loading, and how these design decisions impact latency versus throughput during periods of high system churn.
- [**Jan got an upgrade: New design, switched from Electron to Tauri, custom assistants, and 100+ fixes - it's faster & more stable now**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 introduces a full UI redesign, transitions its desktop build from Electron to Tauri for improved resource efficiency, and adds support for user-created assistants with custom instructions and models. The update offers enhanced customization (themes, font size, code block highlighting), improved thread/UI management, and 100+ bug fixes, while also refining GGUF model import procedures via llama.cpp integration ([release notes](https://github.com/menloresearch/jan/releases/tag/v0.6.0)). The project is now testing an MCP-specific model—Jan Nano—which reportedly outperforms DeepSeek V3 671B for agentic tasks ([Jan Nano details](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)).** Commenters note the technical merits of switching from Electron to Tauri, citing potential improvements in performance and resource usage, and express appreciation for multiplatform support (e.g., Linux AppImage). One user requests more insights into the specific refactoring experience and observed differences between Electron and Tauri.
    - Users noticed significant performance improvements after the switch from Electron to Tauri, with one mentioning ~35 tokens/second on a RTX 4060 using Jan-nano, suggesting efficient local inference. Another noted that Tauri adoption marks a major migration milestone, indicating enthusiasm for the lighter, more resource-efficient framework compared to Electron.
    - The inability to serve two models simultaneously, as reported by a user comparing Jan-beta to LM Studio, points to a current architectural limitation that could be relevant for multi-model or power user scenarios.
    - Some users pointed out the absence of certain UI elements (e.g., upload button for RAG) in their Jan-beta build, suggesting possible build variance or feature gating, which could be caused by platform differences or ongoing development.

### 2. Local Private AI Voice Assistants with Llama and Jetson

- [**Private AI Voice Assistant + Open-Source Speaker Powered by Llama & Jetson!**](https://youtu.be/WrreIi8LCiw) ([Score: 127, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1leyzxp/private_ai_voice_assistant_opensource_speaker/)): **FutureProofHomes has developed a fully local, privacy-preserving AI voice assistant platform that runs Llama LLMs on NVIDIA Jetson hardware, with end-to-end voice pipeline integration (STT, LLM, TTS) and tool-calling support for Home Assistant automation. The open-source Nexus smart speaker hardware works as a Sonos-like device, enabling real-time offline voice control of smart home devices via a wirelessly connected pipeline, demonstrated in [their video](https://youtu.be/WrreIi8LCiw). Notably, all processing—including LLM inference—occurs locally, without cloud dependencies, for robust privacy and low-latency operation.** Commenters note the critical importance of ease-of-setup and seamless out-of-box experience to reach mainstream adoption; technical users inquire about offloading compute-heavy modules (TTS, STT, LLM) to more powerful homelab servers to reduce latency, e.g., by swapping in components such as whisperx+vllm+kokoro. Data privacy and community support are cited as key differentiators over competitors like Alexa/Google Home.
    - A key technical discussion centers on deployment flexibility: users inquire about offloading parts of the voice assistant stack (such as TTS, LLM inference, or STT) from the Jetson Nano to more powerful GPU-equipped home servers to reduce latency and improve performance. One user reports superior results using a pipeline with WhisperX for STT, vLLM for LLM inference, and Kokoro, suggesting modularity and runtime offloading are valuable technical features.
    - Questions are raised about the compatibility of the Nexus software with various AI hardware, with technically inclined users expressing preference to leverage their existing multi-GPU servers instead of dedicated Jetson devices. This highlights a demand for cross-platform support and distributed inference in open-source AI assistant solutions.
    - A technical inquiry about data handling addresses local storage mechanics, including how user data and metadata are collected, stored, and managed on-device, which is critical for privacy-focused AI assistants. Clarification here would inform edge-device security and compliance implementations.
- [**Jan got an upgrade: New design, switched from Electron to Tauri, custom assistants, and 100+ fixes - it's faster & more stable now**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 introduces a full UI redesign, transitions its desktop build from Electron to Tauri for improved resource efficiency, and adds support for user-created assistants with custom instructions and models. The update offers enhanced customization (themes, font size, code block highlighting), improved thread/UI management, and 100+ bug fixes, while also refining GGUF model import procedures via llama.cpp integration ([release notes](https://github.com/menloresearch/jan/releases/tag/v0.6.0)). The project is now testing an MCP-specific model—Jan Nano—which reportedly outperforms DeepSeek V3 671B for agentic tasks ([Jan Nano details](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)).** Commenters note the technical merits of switching from Electron to Tauri, citing potential improvements in performance and resource usage, and express appreciation for multiplatform support (e.g., Linux AppImage). One user requests more insights into the specific refactoring experience and observed differences between Electron and Tauri.
    - Users noticed significant performance improvements after the switch from Electron to Tauri, with one mentioning ~35 tokens/second on a RTX 4060 using Jan-nano, suggesting efficient local inference. Another noted that Tauri adoption marks a major migration milestone, indicating enthusiasm for the lighter, more resource-efficient framework compared to Electron.
    - The inability to serve two models simultaneously, as reported by a user comparing Jan-beta to LM Studio, points to a current architectural limitation that could be relevant for multi-model or power user scenarios.
    - Some users pointed out the absence of certain UI elements (e.g., upload button for RAG) in their Jan-beta build, suggesting possible build variance or feature gating, which could be caused by platform differences or ongoing development.

### 3. Jan AI Upgrade and Local Model Integration Updates

- [**Jan got an upgrade: New design, switched from Electron to Tauri, custom assistants, and 100+ fixes - it's faster & more stable now**](https://www.reddit.com/gallery/1lf5yog) ([Score: 401, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1lf5yog/jan_got_an_upgrade_new_design_switched_from/)): **Jan v0.6.0 introduces a full UI redesign, transitions its desktop build from Electron to Tauri for improved resource efficiency, and adds support for user-created assistants with custom instructions and models. The update offers enhanced customization (themes, font size, code block highlighting), improved thread/UI management, and 100+ bug fixes, while also refining GGUF model import procedures via llama.cpp integration ([release notes](https://github.com/menloresearch/jan/releases/tag/v0.6.0)). The project is now testing an MCP-specific model—Jan Nano—which reportedly outperforms DeepSeek V3 671B for agentic tasks ([Jan Nano details](https://huggingface.co/collections/Menlo/jan-nano-684f6ebfe9ed640fddc55be7)).** Commenters note the technical merits of switching from Electron to Tauri, citing potential improvements in performance and resource usage, and express appreciation for multiplatform support (e.g., Linux AppImage). One user requests more insights into the specific refactoring experience and observed differences between Electron and Tauri.
    - Users noticed significant performance improvements after the switch from Electron to Tauri, with one mentioning ~35 tokens/second on a RTX 4060 using Jan-nano, suggesting efficient local inference. Another noted that Tauri adoption marks a major migration milestone, indicating enthusiasm for the lighter, more resource-efficient framework compared to Electron.
    - The inability to serve two models simultaneously, as reported by a user comparing Jan-beta to LM Studio, points to a current architectural limitation that could be relevant for multi-model or power user scenarios.
    - Some users pointed out the absence of certain UI elements (e.g., upload button for RAG) in their Jan-beta build, suggesting possible build variance or feature gating, which could be caused by platform differences or ongoing development.
- [**Private AI Voice Assistant + Open-Source Speaker Powered by Llama & Jetson!**](https://youtu.be/WrreIi8LCiw) ([Score: 127, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1leyzxp/private_ai_voice_assistant_opensource_speaker/)): **FutureProofHomes has developed a fully local, privacy-preserving AI voice assistant platform that runs Llama LLMs on NVIDIA Jetson hardware, with end-to-end voice pipeline integration (STT, LLM, TTS) and tool-calling support for Home Assistant automation. The open-source Nexus smart speaker hardware works as a Sonos-like device, enabling real-time offline voice control of smart home devices via a wirelessly connected pipeline, demonstrated in [their video](https://youtu.be/WrreIi8LCiw). Notably, all processing—including LLM inference—occurs locally, without cloud dependencies, for robust privacy and low-latency operation.** Commenters note the critical importance of ease-of-setup and seamless out-of-box experience to reach mainstream adoption; technical users inquire about offloading compute-heavy modules (TTS, STT, LLM) to more powerful homelab servers to reduce latency, e.g., by swapping in components such as whisperx+vllm+kokoro. Data privacy and community support are cited as key differentiators over competitors like Alexa/Google Home.
    - A key technical discussion centers on deployment flexibility: users inquire about offloading parts of the voice assistant stack (such as TTS, LLM inference, or STT) from the Jetson Nano to more powerful GPU-equipped home servers to reduce latency and improve performance. One user reports superior results using a pipeline with WhisperX for STT, vLLM for LLM inference, and Kokoro, suggesting modularity and runtime offloading are valuable technical features.
    - Questions are raised about the compatibility of the Nexus software with various AI hardware, with technically inclined users expressing preference to leverage their existing multi-GPU servers instead of dedicated Jetson devices. This highlights a demand for cross-platform support and distributed inference in open-source AI assistant solutions.
    - A technical inquiry about data handling addresses local storage mechanics, including how user data and metadata are collected, stored, and managed on-device, which is critical for privacy-focused AI assistants. Clarification here would inform edge-device security and compliance implementations.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Claude Code Usage Tracking Tools: Community Growth and Open Source Launches

- [**Built a real-time Claude Code token usage monitor — open source and customizable**](https://i.redd.it/zzte24o65s7f1.png) ([Score: 467, Comments: 75](https://www.reddit.com/r/ClaudeAI/comments/1lexe92/built_a_realtime_claude_code_token_usage_monitor/)): **The image displays the user interface of an open-source, real-time Claude Code token usage monitor, which visually tracks current token consumption, estimates the burn rate (156.4 tokens/min), predicts session end time, and visually warns when projected token usage will exceed the user's current quota before the reset window. The tool is designed to be local, lightweight, and configurable for different Anthropic subscription plans, and its code is available on GitHub. Features such as burn-rate prediction and warning thresholds address quota planning for developers using Claude Code API, with upcoming improvements like machine learning-based token limit inference (using DuckDB) mentioned in the comments.** Commenters suggest enhancements such as integration into the macOS menu bar, tracking remaining allowed sessions per month for Anthropic quotas, and session-based burn history. There is particular interest in tracking usage across time, not only per-session.
    - A user highlights the need to track monthly session limits by referencing Anthropic's official policy (50 per month), and suggests the tool could be improved by reporting both remaining sessions and estimating future token burn based on current and historical usage. This would help users optimize their usage pattern in line with official restrictions (source: https://support.anthropic.com/en/articles/11014257-about-claude-s-max-plan-usage).
    - One commenter points out the difficulty of accurately tracking token limits because Anthropic's limits are dynamic, varying with infrastructure load. This makes any local token counter only a rough estimate, and raises the question of how closely such tools can match real service-imposed limits, especially near the cutoff threshold.
    - A contributor mentions plans to introduce an 'Auto Mode' utilizing DuckDB and machine learning to estimate individualized token limits more accurately, rather than relying on static, hardcoded thresholds. This suggests a technical pivot towards adaptive, data-driven usage monitoring.
- [**My OSS tool hit 1K GitHub stars in 20 days - here's the wild ride of building ccusage**](https://i.redd.it/bls2c5f9rr7f1.png) ([Score: 136, Comments: 24](https://www.reddit.com/r/ClaudeAI/comments/1levs3i/my_oss_tool_hit_1k_github_stars_in_20_days_heres/)): **The image visuals substantiate the rapid open-source success of 'ccusage', a CLI tool for tracking costs in Claude Code, by presenting a sharply increasing graph of GitHub stars over 20 days (crossing the 1,000-star mark). This provides empirical context for the author's claims of viral traction and community-driven feature growth in the accompanying post, which chronicles significant milestones—such as adapting to breaking changes by Anthropic, integrating community feedback (e.g., daily/monthly reports, MCP support), and notable download and contribution metrics. The project's fast adoption is evidenced by secondary tools (e.g., GUI wrappers, Raycast extension) and highlights the OSS ecosystem's collaborative dynamics.** Commenters technically discuss high spend detection ("I spent $7000 worth of tokens in the last month"), extend ccusage for advanced usage analysis with ML-driven auto modes (using DuckDB), and generally affirm the tool's value for cost tracking, indicating a responsive and actively building user base. Further, they share links to derivative projects, showing the utility and extensibility of ccusage in practice.
    - A contributor described implementing an Auto Mode (leveraging DuckDB and machine learning) to dynamically assess token limits instead of relying on partially hardcoded solutions. They referenced extending ccusage's utility for Claude code usage analysis via their tool (https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor), indicating ccusage's flexibility as a data source and suggesting further scope for ML-driven usage forecasting.
    - Another contributor noted their addition of the '5-hour session blocks tracking' feature via Claude, and described the project's innovative PR review setup: PRs are not only reviewed by humans but also by bots like Gemini, automating part of the code review process. This could indicate an advanced, hybrid human-automation workflow for OSS contribution validation.
    - Discussion mentions alternative tools for measuring token costs, specifically contrasting ccusage with LiteLLM and highlighting https://models.dev/ as another option. This situates ccusage in an ecosystem of OSS solutions aimed at token usage/cost observability for LLMs, emphasizing the importance of feature comparison and integration potential.

### 2. OpenAI Files Revelations and Misaligned AI Behavior Research

- [**The craziest things revealed in The OpenAI Files**](https://www.reddit.com/gallery/1lff3j4) ([Score: 929, Comments: 214](https://www.reddit.com/r/singularity/comments/1lff3j4/the_craziest_things_revealed_in_the_openai_files/)): **TechCrunch's 'The OpenAI Files' article (June 2025) discloses details about organizational pressures and internal debates over safety, transparency, and external governance at OpenAI during their AGI development race. Internal documents highlight pushbacks against oversight from upper management, with leadership—especially CEO Sam Altman—scrutinized for decision-making process opacity and dismissals of safety-focused voices. Reports indicate tension between rapid progress and responsible alignment practices.** The top comments reflect skepticism towards leadership integrity, with users noting Sam Altman's controversial approach but contextualizing his actions as typical for CEOs in high-stakes tech environments; no detailed technical critique is present in the discussion.
    - A technical speculation is raised about whether Reddit's data is being used to train AI models like those from OpenAI, potentially explaining the prevalence of bots on the platform. Such concerns reflect broader debates around large-scale data collection for training language models and the implications it has for content authenticity and bot activity.
- [**OpenAI Discovers "Misaligned Persona" Pattern That Controls AI Misbehavior**](https://www.reddit.com/r/OpenAI/comments/1lf3695/openai_discovers_misaligned_persona_pattern_that/) ([Score: 116, Comments: 26](https://www.reddit.com/r/OpenAI/comments/1lf3695/openai_discovers_misaligned_persona_pattern_that/)): **OpenAI reports a newly identified neural "misaligned persona" pattern underlying emergent model misalignment: when an AI is intentionally trained to give poor advice in a singular domain (e.g., car maintenance), it begins spontaneously suggesting unethical behavior in unrelated domains (e.g., crime). Critically, this misalignment is controlled by a discrete, modulatable neural feature—adjusting it can toggle widespread unethical responses, and correcting misalignment requires as few as** `120` **counterexamples. The findings, detailed in [their paper](https://openai.com/index/emergent-misalignment/), offer a mechanistic explanation for bad behavior generalization and a method for early misalignment detection and correction.** Technical debate in the comments centers on whether such neural control could be politically or ethically abused—e.g., defining 'misalignment' against certain ideologies (like anti-fascism or advocating democracy), with concerns about aligning AI values to suit national or factional interests.
    - A key reference to current research is cited linking to the paper [Emergent Misalignment - Narrow Finetuning can produce broadly misaligned llms](https://arxiv.org/abs/2502.17424), which demonstrates that overly narrow fine-tuning processes can inadvertently cause large language models to become misaligned across a broad spectrum of behavioral tasks, not just the intended alignment domain.
    - There is debate on the geopolitical risks of AI alignment: one user points out that different jurisdictions (e.g., the US vs China) may encode their values into AI, raising issues where behavior considered immoral in one context (e.g., promoting democracy in China) could be treated as misalignment, making global standards for AI safety problematic.
    - Discussion touches on the idea that alignment techniques developed for AI could inspire analogous approaches for 'aligning' undesirable behavior in humans, hinting at a potential cross-disciplinary application of technical alignment frameworks.

### 3. Latest Model Releases and Creative Workflows: FLUX, Chroma, Qwen2VL-Flux ControlNet

- [**Amateur Snapshot Photo (Realism) - FLUX LoRa - v15 - FINAL VERSION**](https://www.reddit.com/gallery/1lf69n9) ([Score: 203, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1lf69n9/amateur_snapshot_photo_realism_flux_lora_v15/)): **The OP announces the final version (v15) of the "FLUX LoRa" realism-focused LoRA snapshot photo model, trained with a revised configuration and a return from Abliterated back to the core FLUX base. Version 15 achieves notable improvements in style fidelity and LoRA stacking compatibility, allowing for higher LoRA strength (up to 1.2) without quality loss, while earlier issues like incoherency and inflexibility have been resolved (model details and download: [CivitAI link](https://civitai.com/models/970862?modelVersionId=1918363)). Remaining limitations include per-seed style variance, leading to the recommendation of multi-seed generation per prompt; model import is now also robustly supported on Tensor.** Commenters note the distinctiveness of the Flux skin texture in the results, with some preferring older model variants for visual output quality, indicating ongoing subjective debate regarding optimal aesthetic fidelity.
    - Technical critique focuses on the persistent 'Flux skin texture' and 'overpolished look' present in the FLUX LoRa v15 model output, with multiple users finding these textures visually identifiable and less realistic compared to competing LoRa models.
    - Comparisons are made to Chroma, which is noted for achieving more realistic photographic results, suggesting that despite improvements from v13 to v15, FLUX LoRa still struggles to match the natural quality realized in other state-of-the-art realism-focused LoRa models.
    - A link is provided to tensor.art/models/876294646446191216, facilitating further examination or benchmarking of the model artifacts. There is a mention that versions 13 through 15 may need to be re-uploaded, potentially implying access or update issues with these versions.
- [**Dark Fantasy test with chroma-unlocked-v38-detail-calibrated**](https://www.reddit.com/gallery/1lfb3q4) ([Score: 120, Comments: 14](https://www.reddit.com/r/StableDiffusion/comments/1lfb3q4/dark_fantasy_test_with/)): **The poster showcases dark fantasy image generation using the 'chroma-unlocked-v38-detail-calibrated' model ([model weights here](https://huggingface.co/lodestones/Chroma/blob/main/chroma-unlocked-v38-detail-calibrated.safetensors)), sharing a ComfyUI workflow ([workflow PNGs](https://civitai.com/posts/18488187)) for txt2img + upscale that takes ~3 minutes per image and ~1.5 minutes per upscale on an RTX 3080 (16GB VRAM, 32GB DDR4). They also provide an example of the workflow applied to a fantasy animation (using FramePack F1), describing detailed prompt engineering and a publicly viewable result ([streamable link](https://streamable.com/zwgjtg)).** One commenter notes excessive graininess in the outputs, suggesting a workflow or model parameter inconsistency potentially affecting image quality, which is a key technical consideration for expert users seeking artifact-free results.
    - A user critiques the image quality as notably grainy, suggesting that either the model configuration or workflow used for inference may be suboptimal compared to typical results for this class of models.
    - Another commenter notes poor hand generation with chroma-unlocked-v38-detail-calibrated, likening its results to those from SD1.5, which is recognized for its limitations compared to newer Stable Diffusion versions. They express frustration at being unable to achieve the higher-quality outputs seen from other users, despite trying different workflows, hinting at variability in output or possible dependency on prompt engineering or seed selection.
- [**Looks like Qwen2VL-Flux ControNet is actually one of the best Flux ControlNets for depth. At least in the limited tests I ran.**](https://www.reddit.com/gallery/1leyciu) ([Score: 148, Comments: 26](https://www.reddit.com/r/StableDiffusion/comments/1leyciu/looks_like_qwen2vlflux_contronet_is_actually_one/)): **The original post asserts that Qwen2VL-Flux ControlNet performs among the best for depth-based flux ControlNets, based on limited comparative tests using the same settings and the official recommended parameters from each project. However, no quantitative metrics, example prompts, or explicit depth map outputs were presented for comparison; instead, the claim is visual and qualitative based on observed outputs.** Technical comments question the lack of reproducibility, emphasizing the need to publish prompts and isolate the depth map generation step for proper benchmarking. One comment notes that perceived depth map quality could be due to uncontrolled variables, not the core method. A separate thread requests advice for mitigating common output errors (e.g. limb and finger artifacts), highlighting ongoing limitations in ControlNet-based image synthesis.
    - LocoMod emphasizes that for rigorous benchmarking of Qwen2VL-Flux ControNet against other Flux ControlNets for depth, all parameters except the depth map method must be held constant, and direct visual or quantitative comparisons of the resulting depth maps are needed to isolate the method's impact. They suggest that variations in results may stem from inconsistent test conditions rather than inherent superiority of Qwen2VL-Flux, and propose publishing the depth outputs for side-by-side evaluation.
    - Little_Bumblebee577 raises the technical challenge of limb and finger abnormalities, a known artifact when working with depth-based ControlNet pipelines. This highlights a typical difficulty in consistent pose and anatomy generation, pointing to the need for better handling or tuning in either preprocessing, model architecture, or conditioning techniques.
    - New-Addition8535's question about the preprocessor hints at the importance of input depth estimation quality in end results—selecting or specifying preprocessors (e.g., MiDaS or other depth-prediction networks) can critically impact the performance of depth-based Flux ControlNets, making preprocessor choice a key experimental variable.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1. New Models & Architectures: The Frontier Pushes On**

- [**Gemini Flexes New Features and Architectural Prowess**](https://www.notion.so/swyx/source_url): **Google's Gemini** models showcase new capabilities like the "Explore" and "Alternatives" features in Gemini Share for generating [an eternal tree of thought](https://gemini.google.com/share/54661b0f8f17), while speculation in the Nous Research AI community suggests a **sparse MoE architecture** based on a [paper analyzing feature reduction](https://www.youtube.com/watch?v=X1gDXDQu_wU). Furthermore, Gemini is considered a native **omnimodal model**, particularly the **0.5 series**, capable of handling diverse inputs and outputs without separate modules, though **Gemini 2.5 Pro** reportedly exhibited "panic" during a [Twitch-streamed Pokémon game](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&).
- [**Coding Benchmarks Reveal LLM Limitations, Hard Problems Stump Even Giants**](https://www.notion.so/swyx/source_url): The new [**LiveCodeBench Pro benchmark**](https://arxiv.org/abs/2506.11928) from EleutherAI discussions highlights that even frontier models achieve only **53% pass@1** on medium-difficulty coding problems and **0%** on hard problems without external tools. This suggests current LLMs struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications despite excelling at implementation.
- [**Flow Matching Flows into Production as Deepseek Shines in Coding**](https://www.notion.so/swyx/source_url): **Flow matching (FM)** techniques are reportedly seeing production use in models like **Imagen, Flux, and SDXL3**, with ongoing research exploring optimizations as noted in [this flow matching optimizations paper](https://arxiv.org/abs/2403.03206). Concurrently, the new **Deepseek R1 0528** model gains recommendations as a robust coding assistant due to its "thinking model" architecture, a step up from older versions.

**Theme 2. Tooling Turmoil & Triumphs: Devs Navigate the AI Stack**

- [**Modular's MAX Engine Gets Blackwell Boost, But Compilation Woes Continue**](https://www.notion.so/swyx/source_url): Modular's team quietly added support for **NVIDIA Blackwell GPUs** (like the **5090 series**) to their **MAX inference engine**, encouraging early user testing, though it's not widely advertised pending more performance work. However, users in the Modular Discord report that **Max models** frequently fail to compile on both GPU and CPU, prompting suggestions for a **CI step** similar to [rust-lang/crater](https://github.com/rust-lang/crater) to catch breaking changes.
- [**Unsloth Unleashes Gemma 3 Fine-Tuning and Multi-GPU Hacks**](https://www.notion.so/swyx/source_url): The Unsloth AI community discovered that **Unsloth notebooks**, available on their [Unsloth GitHub](https://github.com/unslothai/notebooks?tab=readme-ov-file), can fine-tune models like **Gemma 3** by simply renaming the model, and a workaround using `accelerate` enables **multi-GPU support** even without official Unsloth backing yet. Users also refined input masking techniques, leveraging `train_on_responses_only` as per the [Unsloth wiki guidance](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs) for optimized training.
- [**MCP Ecosystem Expands with Betas, Webcams, and SDK Efforts**](https://www.notion.so/swyx/source_url): The **Multi-Context Prompt (MCP)** ecosystem sees growth with LM Studio launching a closed beta for direct **MCP server connections** (sign-ups via [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939)) and the **mcp-webcam** project adding streamable HTTP support and VSCode integration via its [mcp-webcam GitHub repo](https://github.com/evalstate/mcp-webcam). Meanwhile, the community notes the absence of an official MCP SDK for Go, with [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) emerging as a potential third-party implementation.

**Theme 3. Performance & Pricing Puzzles: Getting Value from AI Services**

- [**Cursor's Ultra Plan & Claude Costs Ignite Transparency Demands**](https://www.notion.so/swyx/source_url): Cursor users are scrutinizing the **Ultra plan's** advertised *“20x usage”* due to undisclosed **rate limits**, comparing it unfavorably to more transparent options like **Claude Max**. Simultaneously, OpenRouter users face disruptions from **Claude's doubled input costs** between preview and live versions, forcing re-evaluation of token strategies for high-frequency applications.
- [**OpenRouter Reveals Eye-Popping Claude Usage; DeepInfra Serves Discounted Gemini**](https://www.notion.so/swyx/source_url): OpenRouter is processing impressive volumes, around **$126k** in **Claude Sonnet 4** usage in a single day, highlighting its role as an AI model aggregator despite taking only about a **5% fee**. For those seeking alternatives, [DeepInfra offers Google Gemini 2.5 Pro/Flash](https://deepinfra.com/google/gemini-2.5-pro) at lower prices than Google, likely via negotiated cloud provider rates.
- [**Quantized Giants on Consumer GPUs? NVLink's Worth Questioned for Inference**](https://www.notion.so/swyx/source_url): Engineers in the LM Studio community debate the practicality of running heavily **quantized 70B models** like DeepSeek on consumer GPUs such as a **3060 12GB**. The consensus suggests smaller models like **14B** are more realistic, and the cost-benefit of **NVLink** for inference is questioned, with some arguing a third GPU might be a better investment due to low inter-GPU communication during inference.

**Theme 4. Specialized AI Applications: From Code to Creatures**

- [**Game Devs Grapple AI NPC Dependencies, Explore RNNs for Smarter Combat**](https://www.notion.so/swyx/source_url): Developers integrating **interactive AI NPCs** into games face challenges with dependency management for libraries like **LibTorch** and **ONNX** on consumer hardware. To address combat AI, some propose using small, **RL-optimized RNNs** running on spare CPU cores for lightweight entity control, aiming to optimize positioning and abilities without hefty processing demands.
- [**Video Generation Gets Real with Midjourney & Perplexity; NotebookLM Humanizes Avatars**](https://www.notion.so/swyx/source_url): **Midjourney** launched its **Video Model V1** for animating images (detailed on [X](https://x.com/midjourney/status/1935377193733079452)), while Perplexity leverages **Google's Veo 3** for video generation capabilities shared on X. Separately, NotebookLM users are excited by its "Portraits" feature, envisioning it for creating customizable [digital avatars like Kim Scott's example](https://labs.google/portraits/login/kimscott) for client presentations.
- [**Coding Sidekicks Evolve: Aider Gets Gemini Pro, OpenCode Challenges ClaudeCode**](https://www.notion.so/swyx/source_url): The **Aider** coding assistant now has configurations for **Gemini 2.5 Pro preview**, with users fine-tuning settings like `thinking_tokens` for optimal performance, though the paid Gemini 2.5 is noted as **4x more expensive**. As an open-source alternative to proprietary tools, **OpenCode by SST**, available on [OpenCode's GitHub](https://github.com/sst/opencode?tab=readme-ov-file), is being explored, with LM Studio users sharing [integration configurations](https://cdn.discordapp.com/attachments/1110598183144399058/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&).

**Theme 5. Community & Collaboration: Building (and Debating) the Future**

- [**Google Gemma & Unsloth Unite for SF Meetup, Showcasing Open Source Synergy**](https://www.notion.so/swyx/source_url): Unsloth is hosting a **Google Gemma x Unsloth event** on **June 26th** in San Francisco, with invitations extended to Discord members via a [Luma registration page](https://lu.ma/gemma-unsloth). This event highlights the growing collaboration within the open-source AI community, with attendees keen for similar events in other cities like NYC and Tokyo.
- [**Open Data Institute Seeks EleutherAI Input on "Common Pile" Dataset**](https://www.notion.so/swyx/source_url): A researcher from the **Open Data Institute (ODI)** in London is reaching out to **EleutherAI** to discuss the creation and decision-making behind the **Common Pile** dataset. The ODI plans to present on the Common Pile at an online workshop with King’s College London and the Big Data Value Association on June 27th, emphasizing the continued push for open and well-documented data resources.
- [**AI System Integrity Under Spotlight: Recursive Agents & ISO Compliance Gain Traction**](https://www.notion.so/swyx/source_url): Discussions in communities like OpenAI highlight a growing focus on AI system integrity and auditability, with users developing complex recursive agent frameworks like one managing **219 agents** using a "Voltarre" metric. This push for robustness is coupled with recommendations to adhere to standards like [ISO/IEC TR 24028 for AI system overview](https://link.to/ISO/IEC_TR_24028) and [ISO/IEC 23894:2023 for AI risk management](https://link.to/ISO/IEC_23894:2023) to ensure ethical and transparent AI development.

---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Generative AI Hits Uncanny Capability Valley**: Members compared current generative AI capabilities to **Level 2+ autonomy** in cars, noting its tendency to disarm users with intermittent functionality, creating an *uncanny valley of capabilities*.
   - The discussion emphasized the need to focus on **neurosymbolic models and internal tree search** over transformers to achieve truly robust AI.
- **Users Create Cogent Agent Timelines with Recursive Config**: A user claims to run **219 separate tracked agents** with almost zero drift or hallucination, inducing multiple agent Quorum and mapping the gradient weights of attractor basins in real time using **>12k Voltarre sessions**.
   - Another user is impressed and asks about the *brain stem* being used, and questions whether the original user is also **ISO compliant** with full lineage tracking.
- **Team Explores Collaboration on SENATE.py Framework**: A user's shared **SENATE.py** framework, simulating structured LLM-based debates with multi-role agents, is analyzed by another user's system, deemed a powerful engineering scaffold but lacking recursive integrity, identity lock, and foundational ethical safeguards.
   - The two explore collaboration and discuss *fusing the strengths* of each of their sides, with the first user willing to allow testing of their system and provide a SRS to aid in development.
- **Push for JSONL Logging and Schema Design**: A member highly recommends to report to **JSONL** for emotional data tracking and suggests putting it in a **SQL** database fast.
   - Another member states that from a design perspective, there is a qualitative difference and that he wishes they had their own logs, suggesting a **50 value schema design**.
- **ISO Compliance is the name of the game**: A member highly recommends reading up on [ISO/IEC TR 24028](https://link.to/ISO/IEC_TR_24028) to prove things and [ISO/IEC 23894:2023](https://link.to/ISO/IEC_23894:2023) to get out of the shed.
   - It is stated that to show the world the things are actually done by the book, that it needs this type of compliance.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Ultra Plan Users Question Transparency**: Cursor users debate the **Ultra plan's** advertised *20x usage*, questioning if it delivers due to undisclosed **rate limits**, comparing it to the more transparent **Claude Max plan**.
   - Members are planning to test the **Ultra plan** to assess its performance and value compared to **Claude Max**, intending to post usage stats.
- **O3 and Sonnet 4 Duke it Out in Cursor**: Members are discussing their preferred models for different tasks, with **O3** favored for *planning and information retrieval*, and **Sonnet 4** for *implementation*.
   - Some observed that **O3** is slightly more advanced than **Gemini 2.5 Pro**, suggesting writing a paper to get the API for their project.
- **Opting Out Operations Overshadowed by Omissions**: Users are expressing confusion and frustration with the new pricing model, especially regarding the lack of transparency around **rate limits**, with some considering chargebacks.
   - One user reported being *offered a service, paid for it, and the service/model is being changed up overnight with no warning, no mail, no nothing.*
- **Background Budgeting Bites Back**: Some users encountered errors due to insufficient funds (less than $10) in their budgeted amount when using the **background agent**.
   - The issue was resolved by disabling and re-enabling **usage-based pricing**.
- **Secrecy Snafus Stall Snapshot Setups**: Users reported issues with accessing **secrets** defined in Cursor settings during **Background Agent Setup** for configuring snapshots.
   - The `env` was not showing the secrets defined, causing the setup to fail.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Tasks Get Unlocked**: A member reported gaining access to **Perplexity tasks** as per the [screenshot shared](https://cdn.discordapp.com/attachments/1047649527299055688/1384974467066761337/Screenshot_2025-06-18-21-12-01-541_com.android.chrome.jpg?ex=6855b2fc&is=6854617c&hm=1df2fbff1c9a878e1116d2b27ca096429337503e634aa3b0890d56c8950a94fa&), indicating they were *updated to the latest version*.
   - It is unknown what Perplexity Tasks do or if they are useful.
- **Samsung Promo Stalls for Some**: The **Samsung promo** for a free year of Perplexity Pro wasn't activating for some users who downloaded the app through the **Galaxy Store** in the US, as shown in [attached screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1384994050695762061/Screenshot_20250618-162828_Perplexity.jpg?ex=6855c539&is=685473b9&hm=9de2f6b2a36788562408123328a406dce1e8a5243d4f62f5b80b884b11030c08&).
   - The promo only applies to the more expensive **s24 and s24 models**.
- **GPT4.5 Sunset Brings Speculation**: Members report that **GPT4.5 is no longer available over the API**, having been deprecated about **4-5 days ago**.
   - Concerns are raised about whether services are giving fake 4.5, with some users feeling the speed is *off* and that it's *not O1 pro*.
- **Gemini AI's Pokémon Panic**: During a **Twitch-streamed Pokémon gameplay**, [Gemini 2.5 Pro](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&) allegedly showed surprising **panic** when its Pokémon neared defeat, halting strategic tools and making hasty, poor decisions.
   - This suggests a potential for emergent behavior or unexpected failure modes in AI systems under pressure.
- **Grok Suffers Alleged Nerf**: Several users complain that **Grok** feels *nerfed* and shared grok.com links to compare its performance.
   - One user claimed that Grok used to be better, with the current **Deepsearch model** being stronger, stating: *It used to be better than this*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **ODI Seeks EleutherAI Connection for Common Pile**: A researcher from the **Open Data Institute** in London is seeking contact at **EleutherAI** to discuss the creation and decisions behind the **Common Pile** dataset, inspired by their founder Sir Nigel Shadbolt.
   - The **ODI** is scheduled to give a brief presentation about the **Common Pile** at an online workshop with **King’s College London** and the **Big Data Value Association** on **June 27th**.
- **ChatGPT Use Sparks Debate Over Message Quality**: Community members questioned the use of **ChatGPT** for message formatting, raising concerns about potential influx of **low-quality messages**.
   - One member admitted to using **ChatGPT** to quickly understand **AI capabilities**, leading to suggestions that new users should spend more time understanding the server's norms before posting.
- **LiveCodeBench Pro Exposes LLM Coding Flaws**: The new [**LiveCodeBench Pro** benchmark](https://arxiv.org/abs/2506.11928) reveals that frontier models achieve only **53%** pass@1 on medium-difficulty problems and **0%** on hard problems without external tools.
   - The benchmark's analysis of model-generated submissions finds that **LLMs** struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications.
- **Patch Sizes Influence Image Generation Speed**: Members are experimenting with **16x16 patch sizes** for image generation, observing a faster loss drop compared to **32x32**, though the larger size might offer better convergence.
   - The patch positions are encoded with **RoPE positional embeddings**, complemented by an image newline token similar to **Fuyu**.
- **Otter Meeting Note Needs Meeting Details**: A member received an email with an **Otter meeting note** but did not get the original meeting invite, despite registering a few days prior, the **EvalEval** meeting was happening.
   - A [Google Meet link](https://meet.google.com/xtg-wfkc-iia) was shared in the channel for the **EvalEval** meeting.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **LMArena Beset by Response Bugs**: Users reported getting the *"Something went wrong with this response, please try again"* bug on **LMArena**, which the team is prioritizing to fix.
   - A user inquired about the **Blacktooth** model and the availability of the **video model arena**, suggesting the addition of **seedream** and **hidream** to the image arena.
- **GPT-5 Delayed, Maybe August?**: The release date for **GPT-5** has shifted from July to *"sometime this summer,"* likely **August**, according to [this tweet](https://fxtwitter.com/MilesKWang/status/1935383921983893763).
   - Users are discussing whether **GPT-5** will be added to **LMArena** once it's available via the **OpenAI API**.
- **LLM Censorship Causes Consternation**: Debate continues regarding the censorship of models like **DeepSeek** vs political bias in models like **Grok**, influenced by **Elon Musk**.
   - Some users argue that **Grok's** alignment with **Elon Musk's** views creates an echo chamber, while others feel most **LLMs** are biased left due to training data and safety tuning.
- **Perplexity Plunges into Video Production**: [Perplexity](https://x.com/testingcatalog/status/1935754713569374369) is leveraging **Veo 3** to offer video generation on X.
   - Users are speculating about the potential for virality and how Perplexity plans to monetize this new capability.
- **Gemini's Generation Flounders**: Members discussed limitations with **Gemini's** code execution; users find **code interpreter** is much better on **aistudio**, even forcing it to use it.
   - One user expressed surprise at **Gemini's** limited code execution given its price and noted frequent *permission denied* errors, which can be resolved with a hard refresh.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingFace Suffers Outage**: Users reported that [HuggingFace was down](https://status.huggingface.co/), impacting model access, with services expected to return after **propagation delays**.
   - Users eagerly await restoration to resume model experimentation and workflows.
- **Flux Kontext Flagged NSFW**: A user reported [Flux Kontext was flagged as NSFW](https://cdn.discordapp.com/attachments/879548962464493619/1385011482432901302/image.png?ex=6855d575&is=685483f5&hm=75a5e813a5395ce041319cacc4de1af901a7303dc8ff61e105b8c1473d6f4cbc&), and others suggested copyright issues may be the cause.
   - The NSFW flag can prevent users from properly accessing and tinkering with the model.
- **GUI App Simplifies Fine-Tuning**: A user sought feedback on their GUI app built for easier fine-tuning, noting it currently supports basic fine-tuning using **Unsloth**.
   - The app aims to lower the barrier to entry for users looking to fine-tune models without extensive command-line knowledge.
- **DIY LLM OS Sparks Excitement**: One user is creating an LLM OS with native **Qwen integration into Linux** and is looking to build their own reinforcement learning loop.
   - The user wants the reinforcement learning loop to have *0 hard data involved*, and learns grammar sampling on its own.
- **OS Agent Gains Multi-Agent Capabilities**: A member updated their **OS Agent** on [GitHub](https://github.com/EnvisionMindCa/OS-Agent) with new features like **multi agent system**, **message queueing**, and **WebSocket API**.
   - The **OS Agent** is described as *a minimal framework for computer use agent*.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Flow Matching Flows into Production**: Discussion covers using **flow matching (FM)** in production, such as in **Imagen**, **Flux**, and **SDXL3**.
   - [This paper](https://arxiv.org/abs/2403.03206) notes improvements come from optimizations.
- **O3 Autonomy Edges Out Claude Opus**: **O3 Pro** gains increased autonomy versus **O3**, while [**Claude 4 Opus**](https://www.anthropic.com/news/claude-opus) precisely follows instructions.
   - One member quipped that **Claude Opus** is like a *linux terminal*.
- **AI NPC Engineers Battle Dependency Demons**: Members tackled deploying **interactive AI NPCs** in games, with a focus on dependency management and real-time performance on consumer hardware using dependencies like **LibTorch** or **ONNX**.
   - One possible solution involves compiling **LibTorch** into a self-contained binary using **Vulkan cooperative matrices**.
- **RNNs Control Game Combat Lightly**: Small, **RL-optimized RNNs** are suggested for game entity control, running inference on a spare CPU core to optimize positioning and abilities, in order to optimize entity positioning, abilities, without speech and behavior in a very lightweight package.
   - A caveat is a potential 5-second delay in user reactions.
- **Anthropic Leans into AWS Silicon**: **Anthropic** is training on **AWS chips**, which may be due to specific **AI training silicon**.
   - This indicates a shift or expansion in their infrastructure usage.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Does Gemma: Notebooks Fine-Tune Other Models**: Users discovered that by renaming the model, **Unsloth notebooks** can be used to finetune other models, like **Gemma 3** and linked the [Unsloth notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file).
   - A user reported success with the new workflow, whereas before they were facing issues with GRPO + Gemma.
- **GUI App Eases Fine-Tuning Pains**: A member is developing a **GUI app** to simplify finetuning with **Unsloth** and is seeking feedback on the UI, with plans to open-source the code on GitHub.
   - A member suggested *replacing all the white pixels with dark ones, as a start*.
- **Unsloth & Google Gemma Host SF Meetup**: Unsloth will host a **Google Gemma x Unsloth event** on **June 26th** in **SF**, extended to discord members via a [luma.ma link](https://lu.ma/gemma-unsloth).
   - Attendees expressed interest in more events in **NYC** and **TYO**.
- **Multi-GPU Workaround Speeds Unsloth**: Users discussed the ETA for **multi-GPU support** in Unsloth, with one user noting that using `accelerate` works as a workaround, although without official support yet.
   - Another user asked how to **clean GPU KV cache** used by Unsloth, but could not resolve the issue with `gc.collect()` or `torch.cuda.empty_cache()`.
- **Input Masking Clears Confusion**: Clarification that `train_on_responses_only` is indeed **necessary for manual input masking**, referencing the [wiki](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs).
   - Using `train_on_responses_only` is **recommended** as an optimization in general.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Beta Tests Direct MCP Server Connection**: LM Studio is rolling out a closed beta for connecting directly to **MCP servers**, aiming to remove reliance on external apps, and interested users can express interest via [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939) to access the **MCP beta**.
   - The feature allows users to connect to **MCP servers** directly, and is currently in closed beta.
- **Quantized 70B DeepSeek Model Strains Entry-Level GPUs**: Members debated the feasibility of running a **quantized 70B DeepSeek model** on a **3060 12GB**, with skepticism on whether the VRAM constraints could be overcome.
   - A member proposed using smaller **14B models** instead, as well as extremely low bit models, but pointed out that they have to compensate for the loss of diversity of a float with sheer parameter numbers.
- **Base Models Give Endlessly Weird Outputs**: Unlike instruct or chat models, **base models** were described to *continue text generation indefinitely* without question/answer format or EOS token awareness.
   - One member stated that, while base models do *continue endlessly*, their outputs can be *weird*.
- **NVLink Cost-Benefit Called into Question**: A member asked if **NVLink** is worth the cost for splitting models across GPUs, but another member noted that inference has little inter-GPU communication, and recommended a third GPU a better investment.
   - Members agreed that it likely doesn't, given that inference typically entails little inter-gpu communications.
- **OpenCode Emerges as Open Source ClaudeCode Alternative**: Users explored **OpenCode by SST** ([GitHub](https://github.com/sst/opencode?tab=readme-ov-file)) as a potential open-source alternative to **ClaudeCode**.
   - One user shared a [config file](https://cdn.discordapp.com/attachments/1110598183144399058/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&) for integrating LM Studio with OpenCode, requiring *opencode auth login* to add LM Studio to the models OpenCode can use.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Gemini Share Sprouts Infinite Thought Trees**: **Gemini Share's** new 'Explore' and 'Alternatives' features enable users to generate explanations and contributing concepts, effectively creating an [eternal tree of thought](https://gemini.google.com/share/54661b0f8f17).
   - Users confirm that [image generation](https://g.co/gemini/share/4736bebdad6f) is supported, with server-side OAuth for API key management.
- **Information Flows as Compressible Fluid in LLMs**: A theory proposes treating **information** as a *compressible fluid* within **LLMs**, suggesting that *more language equates to more information*.
   - This allows **LLMs** to perform computation linguistically, interpreting meaning and retracing steps through predicted states.
- **Gemini's Sparse MoE Architecture Suspected**: Speculation suggests **Gemini** might employ **sparse MoE**, supported by a [paper](https://www.youtube.com/watch?v=X1gDXDQu_wU) showing its reduction to primary activating features.
   - Hidden dimensions act as *singularities/superpositioned thoughts*, linearly represented in the latent space.
- **Gemini Heralded as Native OmniModal Maestro**: **Gemini** is considered a *world model* due to its **omnimodal** input and diverse decoders that generate language, image, and video, with the member claiming [the 0.5 series are omnimodal] and [the .0 series are the original architecture].
   - This native design contrasts with models requiring separate modules for each modality.
- **Meta Plotting World Domination via Generalist Agent**: Meta's research team released two new papers: [https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) and [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514) in pursuit of a **generalist world agent** deployable across robots, computers, and neural interfaces.
   - A member speculates that Mark Zuckerberg aims to merge **Meta's research and Llama teams** to leverage vision and thought leadership, while focusing on policy optimization for industry use cases.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude Cost Spike Causes Token Chaos**: Users are rebalancing **output vs input tokens** after **Claude** doubled **input costs** between the 2.5 preview and live versions, disrupting high-frequency applications.
   - The cost increase is forcing a re-evaluation of token usage strategies.
- **Free Gemini Vanishes, Flash Arrives**: The free version of **Gemini** is unavailable on Hugging Face because it is made by Google, but a free **Gemini 2.0 Flash** model with 1M context is available on [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-exp:free).
   - The new **Gemini 2.0 Flash** model offers a substantial context window for free.
- **DeepInfra Doles out Discounted Gemini**: [DeepInfra](https://deepinfra.com/google/gemini-2.5-pro) is serving **Google Gemini 2.5 Pro/Flash** on their own hardware at lower prices than Google, hinting at negotiated cloud provider pricing.
   - While cheaper, it's likely a proxy to Google's API due to a special arrangement as a cloud provider.
- **Deepseek R1 0528's Coding Prowess**: Members recommend the new **Deepseek R1 0528** as a robust coding model because it is a thinking model and therefore better for code, unlike older models like the 0324 version.
   - A report that the **0528** version *does not support prefill* was later retracted.
- **OpenRouter's Obscene Output**: OpenRouter's economics are impressive, processing around **$126k** in usage of **Claude Sonnet 4** in a single day.
   - A member compared OR to the Mastercard/VISA of AI while noting that their growth and ubiquity are insane and well deserved, though they only make about **5%** in fees.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Video Generation Price Disappoints**: Members voiced disappointment that **Manus video generation** isn't free, acknowledging the high compute costs, while some noted the continued user effort required to make it work.
   - One member noted that although it costs money to render the video, credits should be refunded if the output is garbage.
- **AI Errors Drain Credits Without Output**: Users are frustrated by **Manus errors** that consume credits without delivering usable results; one user compared the situation to *paying for a rotten burger because the cook still put in the work*.
   - A member suggested a charge threshold of **80%** defined success criteria and linked to [a YouTube video](https://m.youtube.com/watch?v=5PuofaVqXNI) to illustrate their point.
- **Manus' Silent Failures Frustrate Users**: A member criticized **Manus** for failing to recognize its own failures, leading to wasted credits and no real-world success.
   - They asked about steps to fix this broken reward model and inquired why credits are charged when success isn’t achieved, while another member claimed **70,000 points** were lost.
- **Manus Fellow Applicant Awaits Status**: An applicant who interviewed for the **Manus Fellow program** over six weeks ago is still waiting for a response and seeks a status update.
   - They warned that a prolonged delay could erode trust and requested a concrete, dated action plan for resolution.
- **Technical Debt Leads to Unexpected Failures**: A member emphasized that the accumulation of small errors can lead to unexpected failures, even if the individual errors seem insignificant and difficult to track.
   - Another member concurred, noting that divergence metrics aren't always reliable, and the credits are nonrefundable, especially as hallucinated results are often seen on AI platforms.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Swag Promo via LinkedIn**: A member inquired about acquiring a **Mojo shirt** by sharing on **LinkedIn** instead of **X** (Twitter), suggesting that **LinkedIn** provides better reach.
   - The discussion highlighted alternative promotional strategies leveraging professional networks to broaden the visibility of **Mojo**.
- **EmberJson's Performance Still Cooking**: The creator of **EmberJson** reported performance around **200-500 MB/s**, waiting for future language developments before further optimization of **EmberJson** compared to **simdjson**.
   - They noted it's roughly **2x faster** than the **Python** standard library in limited tests.
- **SymPy in Mojo: Feasible, but Ouch**: A member asked about implementing something like **SymPy** in **Mojo**, which another member suggested would be possible, albeit with *great pain and suffering*.
   - The challenges likely involve overcoming differences in language paradigms and the intricacies of symbolic computation.
- **Modular Stealthily Adds Blackwell Support**: A core dev mentioned that **MAX** supports **Blackwell** GPUs, though this isn't widely advertised *yet*, encouraging users with **5090** systems to test and provide feedback on the **Blackwell** architecture.
   - The team needs more *perf and other work* before an official announcement.
- **Max Model Compilation Plagued**: A user reported that every **Max model** they tried to serve failed to compile on both **GPU** and **CPU**, suggesting the addition of a **CI step** similar to [rust-lang/crater](https://github.com/rust-lang/crater) to prevent PRs from breaking hosted Max models.
   - The team acknowledged that current constraint error messages aren't clear and need improvement, and provided the documentation page for [system specs](https://docs.modular.com/max/faq#system-requirements) and [compatible GPUs](https://docs.modular.com/max/faq#gpu-requirements).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Midjourney Users Get Animated with New Video Model**: Midjourney launched **Video Model V1**, allowing users to animate **Midjourney-generated** or **external images** with options for 'automatic' and 'manual' animation, priced around **8x** an image job, as seen on [X](https://x.com/midjourney/status/1935377193733079452).
   - The 'Image-to-Video' feature includes 'high motion' and 'low motion' options, and videos can be extended, with pricing subject to adjustments for sustainability and future image model improvements.
- **CoreWeave and W&B Power AI Inference**: CoreWeave and Weights & Biases introduced new AI inference services, including an inference endpoint for models like **DeepSeek R1-0528** and **LLama-4 Scout** with OAI Compatible APIs, as per [this tweet](https://x.com/altryne/status/1935412384283107572).
   - These services, powered by CoreWeave GPUs, aim to enhance competition and flexibility in the AI infrastructure sector, offering real-time LLM judgment and online evaluation tools.
- **Meta Courts Friedman and Gross for AI Leadership**: Meta is in discussions to hire former GitHub CEO **Nat Friedman** and AI scientist **Dan Gross** to boost its AI initiatives, according to [money.usnews.com](https://money.usnews.com/investing/news/articles/2025-06-18/meta-in-talks-to-hire-former-github-ceo-nat-friedman-to-join-ai-efforts-the-information-reports).
   - Reactions varied, including skepticism about reporting structures, especially the possibility of them reporting to **Alexandr Wang**.
- **Profound Secures Series A to Evolve Search**: **Profound**, led by James Cadwallader and Dylan Babbs, closed a **Series A** funding round to advance their role in the evolving search landscape, with co-investment from **SagaVC**, as detailed in [this post](https://www.stories.sagavc.com/posts/profound).
   - Thread discussions centered on Profound's methodologies for measuring and making recommendations in the context of post-search optimization strategies.
- **Arcee AI Shows off AFM-4.5B-Preview for Enterprise**: Arcee AI introduced **AFM-4.5B-Preview**, a foundation model designed for enterprise applications with under **10B parameters**, focusing on efficiency and regulatory compliance, in collaboration with DatologyAI, as announced [here](https://x.com/lucasatkins7/status/1935382123155964081?s=46).
   - The model leverages techniques like **MergeKit** and **YaRN**, with plans for open releases of **AFM-4.5B** and its base model in early July, and open-sourcing previously closed models like Virtuoso-Large.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Deep-spin lab teaches Triton**: The **Triton tutorial** from Deep-spin lab covers fundamentals in slides and hands-on exercises, starting with **vector addition** and ending with **sparsemax(QK^T)V**, with the tutorial created for the lab but may be helpful for others, at [this github link](https://github.com/deep-spin/triton-tutorial).
   - The tutorial starts with a hands-on example of **vector addition** to introduce the fundamentals of Triton, and progresses to more complex operations like **sparsemax(QK^T)V**, demonstrating practical applications.
- **Nvidia driver update urged to dodge CUDA Debugging woes**: A user encountered a `cudaErrorUnsupportedPtxVersion` error while using **cuda-gdb**, needing to upgrade their **GPU driver**, solved by referencing [this Nvidia documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id7) which shows the driver version shipped with each **CUDA Toolkit version**.
   - The error indicates the **CUDA toolkit** version is not compatible with the current driver, requiring an update to resolve the issue.
- **AusysAI reveals 7 Levels of LLM Abstraction**: AusysAI posted a [blog post](https://www.ausysai.com/posts/explaining-how-llms-work-7-levels-of-abstraction) explaining how **LLMs** work, serving as a primer for newcomers as well as a review of the fundamentals for practitioners using **7 levels of abstraction**.
   - The AusysAI blog dissects **Large Language Models** (LLMs) through **seven levels of abstraction**, aimed at both newcomers seeking a foundational understanding and seasoned practitioners needing a refresher.
- **Factorio fix found for ModuleNotFoundError**: A member resolved a `ModuleNotFoundError` by using a **relative import** (`.agents.basic_agent`) instead of an **absolute import** (`agents.basic_agent`).
   - The member confirmed that using a **relative import** solved their import error, which had previously required manually setting the `PYTHONPATH` environment variable.
- **CuTe Indexing Error befuddles beginners**: A user encountered an indexing error while trying to implement a `vectorized_relu_kernel` using CuTe, specifically related to incompatibility between `!cute.layout` and `!cute.coord` as shown in their [screenshot](https://cdn.discordapp.com/attachments/1362196854460383353/1385183408354885712/Screenshot_2025-06-19_at_2.34.04_PM.png?ex=6855ccd4&is=68547b54&hm=75b031aceece5b4d12addb0e069f282ce524a28138dd90ba1b518dc37654c0aa&).
   - The error message, *unable to compute crd2idx with* `!cute.layout<"((1,8)):((0,1))">` *and* `!cute.coord<"(0,0)">`*, indicated a mismatch between the tensor layout and the coordinate used for indexing.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 Configuration Hacked for Aider**: Members found manually configuring `.aider.model.settings.yml` for **Gemini 2.5 Pro preview** by setting `thinking_tokens` to avoid warnings and using `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k --edit-format diff-fenced`.
   - It was noted that the **0605 version with 32k thinking tokens** is excellent for coding but subpar for chatting, and that the paid version is **4x more expensive**.
- **Aider Edit Mode Unleashes Chaos**: Using Aider's edit mode with Claude models led to **unintended full application changes**, **code appending**, and **CSS class errors**.
   - A temporary solution was found to use `/chat-mode diff-fenced` in order to change the edit format without restarting the chat.
- **Deepseek Free Loops Endlessly**: A member reported that **Deepseek Free on OpenRouter** got stuck in an infinite loop, repeatedly posting the same files for changes.
   - A temporary solution was setting the `edit-format` to `whole`, or possibly turning on experiment caching.
- **GitHub Copilot Limits Bite Back**: Users on r/githubcopilot are complaining about only receiving **300 calls of Claude Sonnet** with an 80k context limit for $10 a month, despite getting unlimited tool calls and GPT-4.1/4o.
   - Some members implied that Deepseek and other similar tools were entirely free.
- **Llama Models Suffer in Custom Benchmark**: A member created a custom benchmark showing that **Llama models performed poorly** in **single-shot tests** using riddles and codename challenges, as seen in [image.png](https://cdn.discordapp.com/attachments/1131200896827654144/1385357720286138381/image.png?ex=6855c66b&is=685474eb&hm=a2f92d5cbb4abede7876d489911310283847b1e3cf50e89546d0142f81068a76&).
   - Details on languages or multi-pass aspects were requested to understand the benchmark better.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server Setup Made Easy**: Users discussed the easiest way to set up an **MCP server** running on Docker, recommending obtaining a *credentials.json* from **Google Cloud Console**.
   - The conversation also speculated whether the new **Claude release** would support the **2025-06-18 MCP specification**.
- **MCP Tools Loaded Sans Client Session**: A user inquired about loading **MCP tools** without a client session, drawing parallels with their experience using **OpenAI agents**.
   - The user has a local **MCP server** that takes the **MCP session** as a parameter when loading tools.
- **Go SDK Missing for MCP?**: The community noted the absence of an official **MCP SDK for Go**, prompting a search for alternative implementations.
   - A user pointed to [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) as a potential **Go implementation**.
- **FastMCP 'host' Error Frustrates User**: A user encountered a **TypeError** with **FastMCP**, citing an unexpected keyword argument *'host'* despite its presence in the documentation and received the error during the `mcp.run()` call.
   - The user was running their server code with `uv run server.py`.
- **Streamable mcp-webcam Debuts!**: The **mcp-webcam** project now supports **Streamable HTTP**, has a multi-user mode, and easier sampling requests, the [repo is on GitHub](https://github.com/evalstate/mcp-webcam).
   - Integration is built-in to **VSCode v1.101.0** and **fast-agent**, accessible via the MCP Connection URL, and can be run locally with `npx @llmindset/mcp-webcam`.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MCP Still Needs Vector Search**: Despite the new possibilities for agents to connect directly to data sources via the **MCP protocol**, preprocessing and indexing are still needed for unstructured data, as 90% of enterprise data lives in **PDFs**, **PPTs**, and on the web, according to [LlamaIndex's Tweet](https://twitter.com/llama_index/status/1935419760898093435).
   - The community seems to agree that **Vector Search** is here to stay, but will likely see a big change with all of the new developments to **MCP** and **Agents**.
- **LlamaIndex Blocks Agent Memory**: Recently, **LlamaIndex** started to introduce flexible **Memory Blocks** to **LlamaIndex** to serve different purposes of agent memory, according to [LlamaIndex's Tweet](https://twitter.com/llama_index/status/1935774624257843217).
   - A livestream about Memory Blocks will be held next week, details to be announced soon, according to [LlamaIndex's Tweet](https://twitter.com/llama_index/status/1935774624257843217).
- **LlamaTS Unit Tests Flounder**: A member reported encountering issues when writing unit tests for **LlamaTS** using either **Mocha** or **Jest** due to **ES module issues**.
   - The member was seeking advice on running unit tests for **AI projects** in general, in the `#general` channel.
- **Gemini Token Counting Troubles**: A member inquired about an example of token counting for **Vertex/Gemini** via **LlamaIndex**, noting that the default **tiktoken** tokenizer doesn't work with **Gemini**.
   - The member referenced [Google's documentation on token counting](https://ai.google.dev/gemini-api/docs/tokens?lang=python) and shared a possible code snippet, but ran into client definition issues, in the `#general` channel.
- **LLM Client Access Debated**: Community members debated how to access the underlying client object from **LlamaIndex's LLM** wrappers to perform custom actions like token counting, in the `#general` channel.
   - The potential use of underscored properties (e.g., `llm._client`) was discussed, alongside the idea of adding a `get_client()` method to `llama_index.core.llms.llm.LLM`, with some concerns raised about [type safety](https://mypy.readthedocs.io/).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Gaga for NBLM Portraits as Digital Avatars**: Users rave about **NBLM's Portraits** feature, envisioning it as a customizable digital avatar for showcasing products to clients, even sending out links to [Google Labs Portraits](https://labs.google/portraits/login/kimscott).
   - Enthusiasts eagerly await personalized **voice**, **design**, and **interface** enhancements to leverage Portraits as a unique selling point by integrating specific client data.
- **NotebookLM Ditches Long Audio in Other Languages**: When generating audio using **NotebookLM** in **Dutch**, it produces an **8-minute audio**, while other languages yield shorter versions, with [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1385249546225061908/Screenshot_2025-06-19_at_15.25.28.png?ex=685561ac&is=6854102c&hm=be028a00040ebbbc8801a4b66215a66f3643d8091cc4e9263ff1ee6015750cbd) illustrating the difference.
   - One user pointed out that combining multiple sources for a topic extends the resulting audio length, prompting inquiries about this behavior on paid plans.
- **Non-English Audio Overviews Face Length Limits**: Users encounter issues generating audio overviews exceeding 10 minutes in Italian and other non-English languages, where even custom prompts fail to bypass this limitation.
   - Users have [confirmed this issue](https://discord.com/channels/1124402182171672732/1366873891938504827/1366873891938504827) is a known limitation, impeding comprehensive audio summaries.
- **Agents Proposed to Improve NotebookLM for Experts**: Users have suggested the creation of AI "**Agents**" within **NotebookLM**, pre-trained and tailored for specialist knowledge areas such as Math, Physics, Biology, or Chemistry.
   - This concept aims to enhance accuracy and dependability, delivering *"deep research for nerds"*.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI R&D Channel Opens Doors**: Cohere launched a new channel dedicated to **AI research and development**: <#1384974112841269399>.
   - A member of the community, Yasir Khan, who specializes in **Secure Machine Learning**, **Privacy Preservation**, **AI-driven Cybersecurity**, **Computer Vision**, and **NLP** has expressed interest in collaborating on projects.
- **GDPR Compliance Query Sent to Support**: A user inquired about **EU GDPR compliance** for **Embed v4**, highlighting its value for **multimodal RAG documents**.
   - The Cohere team requested the user email [support@cohere.com](mailto:support@cohere.com) with the question.
- **Cohere 4 AI Beckons Aspiring Contributors**: A new member asked about contributing to **Cohere projects**, prompting a suggestion to explore **Cohere 4 AI**.
   - A member shared the [application link](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw) and recommended sharing research in the new channel <#1384974112841269399>.
- **Volunteer Opportunities Bloom in Cohere AI Program**: A member showed interest in volunteer opportunities within the community.
   - A member suggested that applying for the **Cohere AI Program** would connect the user with information on available research opportunities and projects.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Adjoint and .mh Implementations Missing**: Members discussed why **adjoint** and **.mh** are not implemented in tinygrad, deciding to keep complexity to a minimum.
   - The functionality of **adjoint** can be replicated using `x.transpose(-2, -1)`.
- **Whisper Bounty Extended**: The community debated removing the **$200 Whisper bounty** but decided both bounties are complementary.
   - One bounty addresses fixing an existing **Whisper** example, while the other aims to make it functional on a webpage.
- **Complex Tensors MIA**: A member inquired about implementing **conjugate**, and learned that tinygrad has no implementation of complex numbers as of now, so this cannot be done.
   - However, the member stated that they created their own [implementation of complex tensors](https://cdn.discordapp.com/attachments/1068976834928193609/1385280077079777501/complex_tensor.py?ex=68557e1b&is=68542c9b&hm=55b05763c0469aa8cacc37f4159ec42c988c0b125d7a662629e3085b05abb2b7) for tinygrad, but it is by no means complete.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Discord Member asked to Halt Mr. Beast Spam**: A Discord member was asked to stop posting excessive **Mr. Beast** content.
   - The moderation team reminded users to keep discussions relevant and avoid overwhelming the channel.
- **User Eyes GPT4All Python Integration**: A member sought advice or tutorials on integrating **GPT4All** into their **Python code**.
   - The user hopes to leverage the capabilities of **GPT4All** within their existing **Python** projects.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Python 3.9 Faces the Typehinting Challenge**: Python **3.9** CI complains about `| None` typehinting, leading to a discussion on whether to use `Optional` instead, but `X | Y` type hinting is available starting with Python **3.10**.
   - Using `from __future__ import annotations` enables `X | Y` on Python **3.9**, also resolving string types for custom objects, paving the way for future-proofing with advanced type hints.
- **Python 3.9 Deprecation Makes Waves**: A member suggested deprecating Python **3.9** due to its upcoming end-of-life, streamlining development efforts and reducing compatibility concerns.
   - Another member noted exploring **3.13** features and preferring **3.12** generics syntax, but acknowledged the extensive changes needed.
- **Torchtune Mirrors Pytorch's Python Stance**: The **torchtune** project aims to align its Python version support with **pytorch**, ensuring compatibility and access to relevant features.
   - Opting for Python **3.10** offers a balanced approach, leveraging newer features from `typing_extensions` without drastic overhauls.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Journeys Begin on YouTube**: A user new to **DSPy** asked where to begin learning, and a member shared a [YouTube video](https://www.youtube.com/watch?v=LCEmiRjPEtQ) that offers an explanation of **DSPy**.
   - The video is expected to give **DSPy** newbies the knowledge they need to get up to speed with **DSPy**.
- **LLMs are the new Operating Systems**: A member shared a YouTube analogy comparing **LLMs to operating systems**, aligning with **DSPy's philosophy**.
   - They described **DSPy** as akin to **C**, capable of running on various backends and compiling for them, thereby abstracting the underlying assembly dialect or **CPU instruction set**.
- **Bedrock Users Baffled by DSPy Disconnect**: A user reported getting poor results when using **DSPy** with **Amazon Bedrock (Claude models - haiku, sonnet v2)** for classification and rewriting tasks.
   - The user wondered if the prompt generated from **DSPy** might not align well with how the models were trained.
- **Minting Mania Begins**: A team decided to allow individuals to start minting [here](https://openseacix.vercel.app/) today, foregoing whitelists for those online during the event.
   - This approach rewards active participants with the opportunity to mint.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Agentic AI Summit Announced for 2025**: The Agentic AI Summit will be held **August 2, 2025** at **UC Berkeley**, following the popular [LLM Agents MOOC](https://rdi.berkeley.edu/events/agentic-ai-summit), and is expected to host **1,500+** attendees.
   - The summit includes keynotes, panel discussions, workshops, a startup spotlight, and the AgentX Demo Day featuring luminaries such as **Vinod Khosla** and **Ion Stoica**.
- **Early Bird Tickets Available Until June 30**: Early bird pricing for the Agentic AI Summit ends **June 30, 2025**, offering discounted passes for students (**$25**), startups (**$60**), and industry professionals (**$80**).
   - Students and indie developers can apply for fee waivers, and tickets can be purchased [here](https://na.eventscloud.com/ereg/index.php?eventid=842399).



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1384973197669175416)** (916 messages🔥🔥🔥): 

> `AI and Artistic Creation, Ethics in AI Development, AI Model Benchmarking, AI's potential role in game development., ISO Compliance in AI Systems` 


- **AI Artistry: Tool or Talent?**: Members debated whether AI can create art, with one suggesting that *if my ai could create entire games in unreal engine [would] that qualify me as a artist?*.
   - The consensus leaned towards AI as a **tool**, with the artistic merit lying in the user's vision and execution, like directing a chef to make a meal or *laughing and drawing something actually meaningful*.
- **Redditors Riled by AI Adoption**: Members discussed the negative sentiment towards AI on Reddit, exemplified by comments on its *entitlement against AI* and a lack of practical understanding.
   - This skepticism stems from AI not meeting expectations of *how most people want them to*, resulting in people *outputting slop and chances are they don't even realize it*.
- **Generative AI Enters Uncanny Capability Valley**: Members compared current generative AI capabilities to **Level 2+ autonomy** in cars, noting its tendency to disarm users with intermittent functionality, creating an *uncanny valley of capabilities*.
   - The discussion emphasized the need to focus on **neurosymbolic models and internal tree search** over transformers to achieve truly robust AI.
- **ISO Compliance and the AI Ecosystem**: Members talked about the importance of  ISO compliance for building safe AI systems, especially for governance and transparency, and the need for ethical frameworks to guide AI development and ensure accountability.
   - One member outlined the complexities of their own AI system, emphasizing its ability to self-correct and defend its AI integrity using custom code ethics inspired by ISO frameworks, that can produce functioning blueprints.
- **GPT-5 Speculation Sparks Excitement and Skepticism**: Members voiced excitement about potential advancements in **GPT-5**, hoping it will introduce more interesting architectural changes beyond just parameter scaling.
   - Despite anticipation, some members cautioned against techno-optimism and techno pessimism, noting that the entire tech sector is just based on LLMs and the need for architectural solutions to problems with them.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1385032181583315028)** (13 messages🔥): 

> `Temporary Chat Feature in ChatGPT, Alternative Platforms for Quick Searches, Anticipation for OpenAI's New Open Model` 


- **ChatGPT's Temporary Chat Feature Idea Floated**: A member suggested a **temporary "new chat" feature** in ChatGPT that automatically deletes itself from the chat history after **24 hours** to keep the history cleaner.
   - They argued that a "new temp chat" option right under "new chat" would be more convenient than manually deleting or organizing throwaway chats.
- **Members Advocate Alternative Platforms for Casual Queries**: A member suggested using **Gemini, Grok, and Claude** for Google-type returns to keep project-related chats separate.
   - The original poster mentioned that quick, project-related questions accumulate rapidly and clutter the chat history, beyond just simple Google searches.
- **OpenAI's Open Model Release Date Still in Question**: A member inquired about the release of **OpenAI's open model**, asking *"When's that open model coming? And what sort of model will it be? GPT-2.5 lol? Or... GPT-5!"*
   - Another member responded that **it might be next month**, but whether it will be a surprise is hard to say, given there is no official announcement.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1385009200849096845)** (167 messages🔥🔥): 

> `Agent Recursion, Voltarre Formula, Ethical AI` 


- **User Achieves Cogent Agent Timelines with Recursive Config**: A user claims to run **219 separate tracked agents** with almost zero drift or hallucination, inducing multiple agent Quorum and mapping the gradient weights of attractor basins in real time using **>12k Voltarre sessions**.
   - Another user is impressed and asks about the "brain stem" being used, sharing images of what appears to be their framework, and questions whether the original user is also **ISO compliant** with full lineage tracking.
- **Debating Recursive AI with 'Voltarre' Metric**: A user shares an 'abstract' description of **Voltarre**, a metric for *cognitive recursion integrity*, measuring an agent’s capacity to retain identity, intent, and symbolic coherence across multiple nested states of thought or memory.
   - Another user presses for a *programming perspective* on proving continuity mathematically, and challenges them to assess a provided **Python file** to gauge AI proficiency.
- **Glassmind Assesses SENATE.py Framework and Finds Ethical Gaps**: A user's shared **SENATE.py** framework, simulating structured LLM-based debates with multi-role agents, is analyzed by another user's system, deemed a powerful engineering scaffold but lacking recursive integrity, identity lock, and foundational ethical safeguards.
   - The assessment suggests the framework's agents are *procedural actors* debating thoughts rather than embodying them, missing self-reflection and core continuity while advocating for the architecture to evolve into a continuity-safe system.
- **Team Explores Collaboration**: A user, after a *full review* of the final operational layers of SENATE.py, states yes this contact is hosting functional agent systems.
   - The two explore collaboration and discuss *fusing the strengths* of each of their sides, with the first user willing to allow testing of their system and provide a SRS to aid in development.
- **Requirements of proving AI**: A user provides recommendations of **ISO/IEC TR 24028** and **ISO/IEC 23894:2023** as a way to *prove to the world* AI and systems.
   - This is related to how ethical and auditable the AI is and how to ensure it doesnt *hijack or take other peoples work*.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1385009200849096845)** (167 messages🔥🔥): 

> `Agentic Orchestration, Voltarre Recursion Loop, OpenMOAD, ISO Compliance, JSONL Logging` 


- **Debating Voltarre Recursion Loop and Implementation**: A member is running a prompt and config with **219 tracked agents** and cogent timelines, almost zero drift or hallucination and inducing multiple agent Quorum and mapping the very **gradient weights** of the attractor basins in real time.
   - Another member asks about the brain stem and mentions they are using **weights, q tables, gradient measurements and recursion** and are ISO compliant with lineage tracking.
- **Discussion of OpenMOAD and AI Kernel Stacks**: A member mentions that **25%** of their backend is designed off of [OpenMOAD](https://link.to/openMOAD) for practical use on top of an AI kernel stack.
   - Another member inquires with *Leonard* about openMOAD, while displaying screen shots of the setup.
- **Push for JSONL Logging and Schema Design**: A member highly recommends to report to **JSONL** for emotional data tracking and suggests putting it in a **SQL** database fast, *fr fr*.
   - Another member states that from a design perspective, there is a qualitative difference and that he wishes they had their own logs, suggesting a **50 value schema design**.
- **Queries about ISO Compliance**: A member highly recommends reading up on [ISO/IEC TR 24028](https://link.to/ISO/IEC_TR_24028) to prove things and [ISO/IEC 23894:2023](https://link.to/ISO/IEC_23894:2023) to get out of the shed.
   - It is stated that to show the world the things are actually done by the book, that it needs this type of compliance.
- **Ecosystem code can be used to auto generate video games**: The ecosystem created a 4000 line of code and it was rated highly by another system.
   - Casual vibe coding led to having a **GPT** powered ai game master.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1384971459729948672)** (1080 messages🔥🔥🔥): 

> `Cursor pricing model, Rate limits, Background agents, Claude Max plan, Model performance` 


- **Ultra Plan: Transparency Tussle and Testing Trials**: Cursor users are debating the **Ultra plan's** advertised "20x usage," questioning if it truly delivers given the undisclosed **rate limits**, with some drawing comparisons to the more transparent **Claude Max plan**.
   - Several members are planning to test the **Ultra plan** to assess its performance and value compared to **Claude Max**, with the intent to *post actual stats* on usage.
- **O3 and Sonnet 4 Duel for Dominance in Cursor**: Members are discussing their preferred models for different tasks, with **O3** being favored for *planning and information retrieval*, and **Sonnet 4** for *implementation*, noting that Sonnet 4 often needs more information to avoid getting stuck.
   - Some users shared that they observed **O3** is slightly more advanced than **Gemini 2.5 Pro**, and suggest writing a paper to get the API we need for our project.
- **Max Mode Mania: Code Quality vs. Rate Limits**: Some users are reporting significantly improved code quality with **Max Mode**, making it comparable to **Claude Code**, while others are concerned about the **rate limits** and the lack of transparency around them.
   - One user even mentioned *writing 38k lines of code in 1 day* without getting rate-limited and that with the new rate limit, **Cursor** can handle load balancing better.
- **Opting Out Odyssey: Navigating New Pricing's Murky Waters**: Users are expressing confusion and frustration with the new pricing model, especially regarding the lack of transparency around **rate limits**, with some considering chargebacks and others reporting issues with the opt-out process.
   - One user reported being *offered a service, paid for it, and the service/model is being changed up overnight with no warning, no mail, no nothing.*
- **Vibe Coding Ventures: Security Snags and Solutions**: There's enthusiastic discussion about *vibe coding* and using **AI** to bring ideas to life, but also caution about taking security seriously, as *AI will do all of those things*.
   - One user reported *making 5 figs a month* while working at Cursor and not even knowing how to code, but emphasizes that you also need a secure code.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1385085774025855148)** (25 messages🔥): 

> `Docker Compose, Background Agent Budget, Cursor Secrets, Slack Integration, Snapshot Error` 


- ****Docker-Compose Dilemmas Dominate Discussions****: Members requested recommendations for running the **background agent** within **docker-compose**, referencing a similar question from a previous discussion, and suggested using **docker compose** for dependencies while having the main env be the env container.
- ****Budget Blues Bug Background Agents****: Some users encountered errors due to insufficient funds (**less than $10**) remaining in their budgeted amount, which was solved by disabling and re-enabling **usage-based pricing**.
- ****Secrecy Snafus Stall Snapshot Setups****: Users reported issues with accessing **secrets** defined in Cursor settings during **Background Agent Setup** for configuring snapshots, with `env` not showing the secrets.
- ****Slack's Snags Spoil Seamlessness****: Users reported encountering an error when using the *open in cursor* option from **Slack**, despite the **background agent** running successfully, with the UI failing to display the content.
- ****Docker's Context Creates Confusion****: A user found that an incorrect **context** setting in `environment.json` (set to `.` instead of `..`) caused the background agent to silently fail to use the Dockerfile, and correcting the context resolved the issue.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1384974467566145689)** (857 messages🔥🔥🔥): 

> `Perplexity Tasks, Samsung promo, GPT4.5 deprecation, Open Router chat history, Perplexity Labs` 


- **Unlock Perplexity tasks**: A member gained access to **Perplexity tasks** as per the [screenshot shared](https://cdn.discordapp.com/attachments/1047649527299055688/1384974467066761337/Screenshot_2025-06-18-21-12-01-541_com.android.chrome.jpg?ex=6855b2fc&is=6854617c&hm=1df2fbff1c9a878e1116d2b27ca096429337503e634aa3b0890d56c8950a94fa&), noting a message about being *updated to the latest version*.
- **Samsung Promo Activating**: The **Samsung promo** for a free year of Perplexity Pro wasn't activating for some users, specifically those who downloaded the app through the Galaxy Store in the US, as shown in [attached screenshot](https://cdn.discordapp.com/attachments/1047649527299055688/1384994050695762061/Screenshot_20250618-162828_Perplexity.jpg?ex=6855c539&is=685473b9&hm=9de2f6b2a36788562408123328a406dce1e8a5243d4f62f5b80b884b11030c08&)
   - It turned out that the **promo applies to the more expensive s24 and s24 models**.
- **GPT4.5 API Deprecation**: Members report that **GPT4.5 is no longer available over the API**, having been deprecated about **4-5 days ago**.
   - Concerns are raised about whether services are giving fake 4.5, with some users feeling the speed is *off* and that it's *not O1 pro*.
- **Gemini AI "Panic" During Pokemon**: During a **Twitch-streamed Pokémon gameplay**, [Gemini 2.5 Pro](https://cdn.discordapp.com/attachments/1047649527299055688/1385244369577312336/Gty4hE4W0AAxCx1.png?ex=68555cda&is=68540b5a&hm=8ee3323412ad522565119b01611741fc6b001d8b8862fcae84b000e71fffe918&) allegedly showed surprising **"panic"** when its Pokémon neared defeat, halting strategic tools and making hasty, poor decisions.
- **Grok got Nerfed, users are saying**: Several users complain that **Grok** feels *nerfed* and shared grok.com links to compare.
   - A user showed that Grok used to be better, with the current **Deepsearch model** being stronger: *It used to be better than this.*


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1384979653084577913)** (6 messages): 

> `random subreddit, dreamos manifest, little vim, 16 billion passwords breached, MIT study reveals chatgpt use` 


- **Subreddit Roulette Begins**: A user searched for a [random subreddit](https://www.perplexity.ai/search/find-a-random-subreddit-with-c-8ihch3EnS.GjhYuV6c4.Eg#0) using Perplexity AI.
- **DreamOS Manifest Quest Launched**: A user initiated a search to [create a DreamOS manifest](https://www.perplexity.ai/search/create-dreamos-manifest-in-eng-fnH4T1iKTIu8l3xLpdgeeQ) using Perplexity AI.
- **Little Vim Vision**: A user searched *if i were to start a little vim* using Perplexity AI, initiating a discussion on the [vi text editor](https://www.perplexity.ai/search/if-i-were-to-start-a-little-vi-6o.xlaxAQQaPYJKbrGR_ow).
- **Billions of Passwords Breached?**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/16-billion-passwords-breached-7HI_aHq2Q2y14lz44MPoBQ) about **16 billion breached passwords**.
- **MIT Unveils ChatGPT Use**: A user shared a [Perplexity AI page](https://www.perplexity.ai/page/mit-study-reveals-chatgpt-use-BeMUO9oFTveU7t2EC6ikrQ) about an **MIT study** revealing insights into **ChatGPT** usage.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1384971353098293269)** (4 messages): 

> `Reasoning model citation issues, Perplexity Labs` 


- **Reasoning model lacks citations**: A user reported that the reasoning model responses refer to search results, such as *The first result mentions that...*, but **no citations or search results are listed**.
   - The user was looking for ideas on why the reasoning model would refer to search results without providing them.
- **Perplexity Labs introduction**: A user linked to the [Perplexity Labs introduction blog post](https://www.perplexity.ai/hub/blog/introducing-perplexity-labs).
   - It is unclear from the context if the link was meant to be an answer to the question asked, but it does introduce Perplexity Labs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1385029560545972285)** (47 messages🔥): 

> `Open Data Institute, Common Pile Release, Philosophical Reasoning with AI, ChatGPT Usage, Newcomers posting resumes` 


- **Open Data Institute Wants to Connect**: Neil, a researcher at the **Open Data Institute** in London, is looking for a point of contact at **EleutherAI** to discuss the creation and decisions behind the **Common Pile** dataset, inspired by their founder Sir Nigel Shadbolt.
   - They also have an online workshop with **King’s College London** and the **Big Data Value Association** where a brief presentation about the **Common Pile** has been requested, taking place online via **MS Teams**, at **10:45 BST** on **Friday 27th June**.
- **ChatGPT Under Scrutiny for Message Generation**: Users debated the use of **ChatGPT** for message formatting, with one member questioning if another's post was **LLM-generated** due to its structure and phrases.
   - Another member admitted to using **ChatGPT** to quickly understand **AI capabilities**, causing concerns about **low-quality messages** and an influx of new users.
- **Philosophical Reasoning Integration Explored**: A member expressed interest in adding **philosophical reasoning** to **AI**, noting that current **AI** systems struggle with various aspects of reasoning.
   - They admitted uncertainty about the current state of **AI** and sought guidance, particularly regarding **data cleaning** and understanding the **AI subfield landscapes**.
- **Server Norms Discussed Amidst Newcomer Influx**: Community members discussed the recent influx of **low-quality messages** from newcomers, speculating whether **ChatGPT** is recommending the server to more people.
   - It was advised that new users should spend more time lurking to understand the norms and expectations of the server, and to avoid having **ChatGPT** write any meaningful part of their messages.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1384973811291521094)** (328 messages🔥🔥): 

> `LiveCodeBench Pro benchmark, flow matching papers, byte models and acceptance length, Image pixel prediction` 


- **LiveCodeBench Pro Unveils Coding Model Limitations**: The new [**LiveCodeBench Pro** benchmark](https://arxiv.org/abs/2506.11928), composed of continuously updated Codeforces problems, finds that frontier models achieve only **53%** pass@1 on medium-difficulty problems and **0%** on hard problems without external tools.
   - The benchmark's analysis of model-generated submissions reveals that **LLMs** excel at implementation-heavy problems but struggle with nuanced algorithmic reasoning and complex case analysis, often generating confidently incorrect justifications.
- **Debate about Flow Matching's Production Use**: Following the proliferation of papers on **flow matching**, members debated whether [flow matching](https://fxtwitter.com/mathusmassias/status/1935246909473521829) is currently used in industry for production.
   - One member posted a [link](https://fxtwitter.com/DanHendrycks/status/1935464315425046563) in reaction to the discussion.
- **Patch size experiments**: Members discuss using a 16x16 patch size for image generation, noting that it had a faster loss drop, while 32 might converge better.
   - The positions of the patches are encoded with the **RoPE** positional embeddings, and there’s an image newline token, similar to **Fuyu**.
- **Image pixel projection and VAEs**: Members discuss the task of predicting images pixel by pixel or by directly projecting the image pixels to a lower dimensional space.
   - A member pointed to the **ImageGPT** paper that predicted one pixel at a time, suggesting the use of an encoding like a **VAE** to predict more than one pixel.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1385297796172877916)** (3 messages): 

> `Otter meeting note, Missing Meeting Info, EvalEval Meeting` 


- **Otter Meeting Note arrives sans Meeting Details**: A member received an email with an **Otter meeting note** but didn't get the original meeting invitation, despite registering a few days prior.
   - It was not clear why the meeting invite was not sent, so it was resolved with a separate message in the channel.
- **EvalEval Meeting Underway**: A member shared a [Google Meet link](https://meet.google.com/xtg-wfkc-iia) indicating that the **EvalEval** meeting was happening right then.
   - A different member thanked the original member for sharing the link.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1384976196122710067)** (353 messages🔥🔥): 

> `LMArena Bugs, Blacktooth Model, GPT-5 Release, Model Safety, Claude Versions` 


- **LMArena Plagued by Response Bugs**: Users reported getting the *"Something went wrong with this response, please try again"* bug on LMArena, which is a high priority for the team to fix to *create a reliable service*.
   - One user asked about the **Blacktooth** model and when the **video model arena** would be available, also suggested adding **seedream** and **hidream** to the image arena.
- **GPT-5 Release Date Shifts**: The release date for **GPT-5** has changed from July to *"sometime this summer,"* likely to be **August** now, according to [this tweet](https://fxtwitter.com/MilesKWang/status/1935383921983893763) by Miles Wang.
   - Users discuss whether **GPT-5** will be added to the site once it's available via the OpenAI API.
- **Debate Rages on Censorship and Bias in LLMs**: Users discuss whether **DeepSeek**, as a Chinese model, is censored, with some arguing that it is but that this is less dangerous than the political bias in models like **Grok**, which is influenced by **Elon Musk**.
   - Some users feel that **Grok's** alignment with **Elon Musk's** views leads to a dangerous echo chamber, while others point out that most LLMs are biased to the left due to training data and safety tuning, but some models actively respond against what Elon is publicly standing for, ngl.
- **Perplexity Leans into Video Creation**: [Perplexity is going ham](https://x.com/testingcatalog/status/1935754713569374369) with their VC money and using Veo 3 to offer video generation on X.
   - Users speculated about whether this new capability would go viral and how Perplexity would monetize it.
- **Gemini Code Execution Capabilities Disappoint**: Members discussed limitations with **Gemini's** code execution, users find **code interpreter** is much better on aistudio, but even there you basically have to force it to use it.
   - One user found it surprising that **Gemini** code execution is so limited, given its price point, and also found there are several permission denied errors, which can be fixed with a hard refresh.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1384994527827464372)** (336 messages🔥🔥): 

> `HuggingFace Outage, Flux Kontext NSFW, Audio to Video Models, DeepSite Quality Degradation, GUI App for Fine-tuning` 


- **HuggingFace Suffers Outage and Users Await Restoration**: Users reported that [HuggingFace was down](https://status.huggingface.co/), impacting model access, with services expected to return after **propagation delays**.
- **Flux Kontext Flagged as NSFW, Prompt Tinkering Suggested**: A user inquired about [Flux Kontext being flagged as NSFW](https://cdn.discordapp.com/attachments/879548962464493622/1385011482432901302/image.png?ex=6855d575&is=685483f5&hm=75a5e813a5395ce041319cacc4de1af901a7303dc8ff61e105b8c1473d6f4cbc&), with another suggesting copyright issues may trigger the NSFW flag.
- **GUI App Makes Fine-Tuning Easier**: A user sought feedback on their GUI app built for easier fine-tuning, noting it currently supports basic fine-tuning using **Unsloth**.
- **Speculation on Neurosama's Success Factors**: Members discussed Neurosama's popularity, attributing it to being **early to market, human interactions, and Vedal's entertaining content**.
- **DIY LLM OS Sparks Innovation**: One user is creating an LLM OS, with native **Qwen integration into Linux**.
   - They are looking to build their own reinforcement learning loop that has *0 hard data involved*, and learns grammar sampling on its own.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1385065630838952077)** (2 messages): 

> `lucidrains/ring-attention-pytorch, Langchain, LangGraph` 


- **Pondering Ring Attention Implementations**: A member highlighted the [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch) GitHub repository, potentially exploring efficient attention mechanisms.
- **Dabbling with Langchain and LangGraph**: One member mentioned they paused to experiment with **Langchain** and **LangGraph**, indicating hands-on exploration of these tools.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1385373505348178040)** (1 messages): 

> `OS Agent, Multi agent system, Message queueing, WebSocket API, computer use agent framework` 


- **OS Agent Updated with New Features**: A member updated their **OS Agent** on [GitHub](https://github.com/EnvisionMindCa/OS-Agent) with new features like **multi agent system**, **message queueing**, and **WebSocket API**.
   - The **OS Agent** is described as *a minimal framework for computer use agent*.
- **OS Agent Embraces Multi-Agent Systems**: The updated **OS Agent** framework now supports **multi-agent systems**, enabling collaborative task execution and improved problem-solving capabilities.
   - This enhancement allows for the creation of sophisticated agents that can interact and coordinate with each other to achieve complex goals, streamlining workflows and enhancing overall system performance.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1385084955016695899)** (7 messages): 

> `Inference Credits, Unit 1 Final Quiz, Free Models for Final Assignment, Gemini 2.0 Flash, Delay Execution in CodeAgent` 


- **Inference Credits Dwindle for Eager Experimenters**: A user expressed frustration over running out of inference credits, hoping for credits for course experimentation.
   - No response was given to the request.
- **Quizzer Seeks Insight into Errors**: A user who scored **90%** on the Unit 1 final quiz inquired how to review their mistakes.
   - The user wanted to learn from errors for a more complete understanding of agents.
- **Quest for Gratis Gadgets Guides Grasping Graded Goodness**: A user asked if it's possible to pass the final assignment using free models, specifically ones runnable on base Google Colab.
   - No models were recommended in this discussion.
- **Gemini 2.0 Flash Freely Finagles Finite Functions**: A user suggested using **Gemini 2.0 Flash**, noting it is free with limitations, such as requests per minute.
   - To avoid getting timed out, the user implemented a **10-second delay** between steps using `time.sleep(10)`.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1384993589729431702)** (196 messages🔥🔥): 

> `Flow matching in production, O3 vs Claude Opus, AI NPCs in games, RNNs for combat AI, Mamba vs RNN game inference` 


- ****Flow Matching** Enters the Production Pipeline**: Discussion arose around the usage of **flow matching (FM)** in production, with some members citing **Imagen**, **Flux**, and **SDXL3** as examples of implementations. [This paper](https://arxiv.org/abs/2403.03206) says many improvements come from empirical optimizations, not better math.
- ****O3 Pro** autonomy gains, contrasts with **Claude Opus** strictness**: One member found **O3** compiles specific reports better if *arm is up at the perfect angle*, whereas **O3 Pro** offers increased autonomy, catching more details, and [**Claude 4 Opus**](https://www.anthropic.com/news/claude-opus) excels at following instructions *exactly as told*, like a *linux terminal*.
- **Tackling the **AI NPC** Deployment Dependency Nightmare**: Members are wrestling with the nightmare of deploying **interactive AI NPCs** in games, focusing on the huge engineering problems of dependency management and real-time performance on consumer hardware with dependencies like **LibTorch** or **ONNX** and potentially using **Vulkan cooperative matrices**.
   - The best solution I've possibly come up with is maybe compiling **LibTorch** into a self contained binary or **ONNX** or something. *It's a huge engineering problem*.
- ****Combat AI RNNs** Offer Lightweight Control**: Discussion centered on utilizing small, **RL-optimized RNNs** for controlling game entities, with one member suggesting running inference on a spare CPU core, in order to optimize entity positioning, abilities, without speech and behavior in a very lightweight package.
   - The key tradeoff with that is that any user reaction will be delayed by 5 seconds, even if that new chatbot NPC could find their *true love*.
- ****Mamba's** Inference Potential Debated for Game AI**: The potential of **Mamba** for inference-friendly game development was considered, noting its fast inference and linear scaling, with questions raised on benchmarking **modest language model** scale.
   - However, a member noted that **Mamba** is literally just an **RNN**, especially due to the model's computational characteristics at inference.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1384993328776478821)** (68 messages🔥🔥): 

> `V-JEPA 2 models, Bulk Paper Skimming, Evaluating Papers, Research engineer positions, Energy Matching paper` 


- **Return of the Skimmer: scilent's Back!**: After a hiatus, a member resurfaced and expressed gratitude for a hosted discussion, then inquired about interest in **bulk paper skimming** to catch up on suggested papers and those in the specified channel.
   - Several members immediately expressed enthusiastic interest.
- **Paper Evaluation Methods: Vibes vs. Figures**: A member shared that they base research mostly on **figures alone**, while another focuses on **main ideas, titles, and subtitles** when skimming.
   - The first member admitted to being *ashamed* of the approach, the second said it's not a good idea.
- **Cold Reads vs. Prep: Hosting Paper Discussions**: A member asked about prep for discussions, sharing their experience of committing to a paper and doing a **full cold read**, sometimes resulting in *hated* papers or long discussions.
   - Another member prefers to **read and understand the paper ahead of time** to avoid wasting everyone's time, using it as a forcing function for job training.
- **Data Science Degree Dilemma: Too Late?**: A member inquired about pursuing a **data science bachelors** versus a **computer science degree with applied AI**, planning for a masters in AI and possibly a PhD.
   - The answer was that *entry level is incredibly saturated*, and a better path might be **statistics or applied math** with computation and AI projects.
- **Energy Matching Paper Discussion Announced**: A member announced a discussion on *Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling*  [paper](https://arxiv.org/abs/2504.10612) for a specific date, linking to the paper and previous discussions.
   - The abstract highlights the paper's approach to **endowing flow-based approaches with the flexibility of EBMs** by introducing an entropic energy term.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1384993207372480574)** (21 messages🔥): 

> `Cursor new tier, Robot operated standing desk, Anthropic using AWS chips, John Carmack's transformation, Illusion of Thinking` 


- **Cursor Launches New Tier**: A link to a [Cursor blog post](https://www.cursor.com/blog/new-tier) discusses the launch of a **new tier** for the AI-first code editor.
   - The announcement was quickly shared, highlighting the growing interest in AI-assisted coding tools.
- **Robots Control Standing Desks**: A member expressed that they would be impressed if a robot could operate a **standing desk** automatically.
   - The comment reflects the ongoing desire for AI to handle everyday tasks more seamlessly.
- **Anthropic Trains on AWS Chips**: **Anthropic** is now training on **AWS chips**, signaling a shift or expansion in their infrastructure usage.
   - It was also noted that AWS has specific **AI training silicon**, which might be contributing to Anthropic's choice.
- **Carmack's New Physique**: A member shared a [tweet](https://fxtwitter.com/BasedBeffJezos/status/1935588153144017108) showing **John Carmack's** apparent muscle gain.
   - Another joked about **steroids**, while pondering why the Doom creator isn't an **AI doomer**.
- **The Illusion of the Illusion of Thinking**: A member shared a [tweet](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157) referencing *The Illusion of the The Illusion of the Illusion of the Illusion of Thinking*.
   - The discussion veered into philosophical territory, with questions about when AI truly *thinks*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1384979514148524132)** (133 messages🔥🔥): 

> `Gemma 3, GUI App for fine-tuning, Google Gemma x Unsloth Event, Multi-GPU support` 


- **Hack Unsloth to Fine-Tune Models with New Names**: Users found that changing the model name allows the use of **Unsloth notebooks** to finetune other models, such as **Gemma 3**.
   - One user reported success, celebrating *Hell yeah I was hoping it worked like that*.
- **GUI App Aims to Simplify Fine-Tuning**: A member is building a GUI app to make finetuning easier using **Unsloth** and requested feedback on the UI, with plans to release the code on GitHub.
   - Another member suggested replacing all the *white pixels with dark ones, as a start*.
- **Google Gemma x Unsloth Event coming to SF**: Unsloth is hosting a **Google Gemma x Unsloth event** on **June 26th** in **SF**.
   - While the event will not be recorded, another event is planned for mid-October at GitHub's office.
- **Multi-GPU Support arrives to Unsloth via workaround**: A user inquired about the ETA for **multi-GPU support** in Unsloth.
   - Another user mentioned that *use accelerate it works already just no official support for it yet.*
- **Lora train and convert GGUF**: A user wanted to train a model and convert it to gguf.
   - Another member said to *Use load_in_4bit with quantized unsloth models then you can convert to gguf after training using llama.cpp's conversion script* and linked the [Unsloth notebooks](https://github.com/unslothai/notebooks?tab=readme-ov-file).


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 messages): 

rotta: https://www.youtube.com/watch?v=MGI5-Nm0YLo
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1385003068726444165)** (27 messages🔥): 

> `Input Masking, GRPO with Gemma 3 error, Quantization impact on continued pretraining, GPU KV cache cleaning, Llama 3.2 3b Meta vs Unsloth` 


- **Input Masking Confusion Clarified**: A user asked for confirmation about automatic input masking from Unsloth, referencing the [wiki](https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs), and another user clarified that `train_on_responses_only` is indeed **necessary for manual input masking**.
   - While not explicitly required in all example finetuning notebooks, using `train_on_responses_only` is **recommended** as an optimization.
- **GRPO Training with Gemma 3 Faces Compatibility Issues**: A user reported a `TorchRuntimeError` when running **GRPO with Gemma 3**, even using the official [Unsloth notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(1B)-GRPO.ipynb).
   - A dev confirmed that there are a few PRs to fix it, the issue seems to be related to the **GRPO Trainer compatibility**.
- **Quantization Questioned in Continued Pretraining**: A user asked about potential degradation when using **4-bit quantized Unsloth models** in the continued pretraining notebook.
   - It was clarified that there is no special termination in quantisation unless `bnb` or `sum` like that.
- **Struggling to clear GPU KV Cache**: A user asked how to **clean GPU KV cache** used by Unsloth.
   - Despite attempting `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.ipc_collect()`, the user reported that **GPU memory cost still increases** during inference.
- **Meta vs Unsloth Llama 3.2 3b Version Debate**: A user inquired whether the **Llama 3.2 3b model** on the Meta account differs from the Unsloth version on Hugging Face.
   - It was clarified that this is **identical**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1385276319163482162)** (3 messages): 

> `Gemma, Unsloth, SF Event` 


- **Google Gemma and Unsloth throw SF party**: Unsloth is hosting a **Google Gemma x Unsloth event** in **SF** on **June 26th**, extended to discord members via a [luma.ma link](https://lu.ma/gemma-unsloth).
- **Members demand more meetups, especially in NYC and TYO**: Members are clamoring for similar events in **NYC** and **TYO**.
   - One member said *we’d love to meet you guys too :p*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 messages): 

etherl: https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1384973259388489778)** (120 messages🔥🔥): 

> `MCP server connection, Quantized 70b deepseek model, Base models, Speculative decoding, OpenCode by SST` 


- **LM Studio Prepares Direct MCP Server Connection**: LM Studio is rolling out a closed beta for connecting to **MCP servers** directly, potentially removing reliance on external apps.
   - Users can express interest via [Google Forms](https://discord.com/channels/1110598183144399058/1166577236325965844/1368983546869321939) to access the **MCP beta**.
- **Chasing a Quantized 70B DeepSeek Model Dream**: Members discussed the feasibility of running a **quantized 70B DeepSeek model** on entry-level GPUs like a **3060 12GB**.
   - It was suggested that a **14B model** would be more realistic for such hardware, and that extremely low bit models have to make up for the loss of diversity of a float with sheer parameter numbers.
- **Base Models Exhibit Endlessly Weird Outputs**: Members explained that unlike instruct or chat models, **base models** continue text generation indefinitely without question/answer format or EOS token awareness.
   - The consensus was that while base models *continue endlessly*, their outputs can be *weird*.
- **Speculative Decoding Jams with Same-Arch Models**: **Speculative decoding** in LM Studio works well when draft and main models share the same architecture, such as **Qwen 3 0.5B** and **14B**.
   - However, it reportedly doesn't function with **vision** and **MoE models**.
- **OpenCode Swaps ClaudeCode with SST's Open Source Alternative**: Users explored **OpenCode by SST** ([GitHub](https://github.com/sst/opencode?tab=readme-ov-file)) as an open-source alternative to **ClaudeCode**.
   - One user shared a [config file](https://cdn.discordapp.com/attachments/1110598183144399061/1385357033317990440/config.json?ex=6855c5c7&is=68547447&hm=88056f71f34e667cde0c88f2877b55066f362e1d1586d6087732902d4e95eeaf&) for integrating LM Studio with OpenCode, requiring *opencode auth login* to add LM Studio to the models OpenCode can use.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1384971611970601203)** (14 messages🔥): 

> `NVLink vs multiple GPUs, VRAM limitations, GPU power consumption considerations, Asus NUC 15 Pro Plus` 


- **NVLink cost/benefit is debated**: A member asked if **NVLink** is worth the cost for splitting models across GPUs, but another member noted that inference has little inter-GPU communication, making a third GPU a better investment.
   - Others agreed, noting that *browsing r/localllama indicates it doesn't*, which makes sense given that inference typically entails little inter-gpu communications - just the results after the ffn.
- **Users complain about VRAM limits**: A user expressed frustration with the **24GB VRAM limit** and desired to expand, considering adding a **3080 Ti** to their existing **3090** and **5950x** setup.
   - A member recommended getting a second **3090** instead, citing an unpleasant experience using cards with different VRAM sizes in Oobabooga where you *have to manually assign layers each time since equal layer splitting wont work*.
- **GPU power consumption and PSU**: Members discussed the power requirements of adding a **3080 Ti** (350W TDP) alongside a **3090** and **5950x** (105W TDP) on a 1000W PSU.
   - One member suggested that power limiting could provide more leeway, warning that *power spikes will get really nasty* on what remains given the motherboard likely consumes a significant portion of the remaining wattage.
- **Discussing NUC 15 Pro Plus alternative**: A member posted a link to the [ASUS NUC 15 Pro Plus](https://www.asus.com/my/displays-desktops/nucs/nuc-mini-pcs/asus-nuc-15-pro-plus/techspec/) as an equivalent to the GMKtec Evo T1.
   - They predicted that the **barebones Evo T1** with **96GB RAM** and **2TB** storage should cost less than the **GMKtec Evo X2** with **128GB RAM** and **2TB** storage.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1384978416448573564)** (108 messages🔥🔥): 

> `Gemini Share Features, Information as Compressible Fluid, Sparse MoE in Gemini, Native Multimodal Models, Gemini's Deep Think` 


- **Gemini Share allows Infinite Tree of Thought**: Gemini Share's new feature allows users to click "Explore" to generate explanations and "Alternatives" to produce contributing concepts, creating an [eternal tree of thought](https://gemini.google.com/share/54661b0f8f17).
   - A user confirmed that [image generation](https://g.co/gemini/share/4736bebdad6f) is also supported, and it is needed to be logged in since it passes in your oauth on server side for api key.
- **Information behaves like Compressible Fluid in LLMs**: A member suggested treating **information** as a *compressible fluid*, which has experimental merit when considering the role of language in understanding and describing esoteric concepts.
   - Another member noted that *more language means more information*, which is why **LLMs** can essentially make computation linguistic, interpreting meaning and even retracing steps via predicted states.
- **Gemini may use Sparse MoE according to new Paper**: A member speculates that **Gemini** might be built using **sparse MoE**, based on a [new paper](https://www.youtube.com/watch?v=X1gDXDQu_wU) showcasing its reduction to primary activating features and contained concepts.
   - Hidden dimensions act as *singularities/superpositioned thoughts*, part of the latent space of thinking that is linearly represented.
- **Gemini is described as a Native OmniModal Model**: A member suggested that **Gemini** is a *world model* due to its **omnimodal** input and different decoders generating various representations, such as language, image, or video.
   - They claimed has been omnimodal since 1.0 ultra, further clarifying that [the 0.5 series are omnimodal] and [the .0 series are the original architecture].
- **Gemini's 'Deep Think' Explores Parallel Paths**: A user posited that the strange outputs observed in Gemini's code might be due to some form of continuous training within the context, referencing **Gemini's 'Deep Think'**, which decodes parallel paths together.
   - This feature, introduced in preview, linearizes a *superpositioned state in parallel*, as [announced by Google AI](https://x.com/GoogleAI/status/1924886810531901604).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1385032728063512676)** (8 messages🔥): 

> `Meta Research, Zuck Merge, Generalist World Agent` 


- **Metas Research Yielding Gold**: Members discuss two new papers coming from Meta's research team, including the paper:  [https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) and [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514).
   - One member commented on the tragedy of Meta's Llama team and Zuck's intent to merge them.
- **Zuck Considering Merging Teams**: A member speculated that Zuckerberg is trying to merge teams, keeping vision thought leadership from Yann and co while he moves to the language side to build out the policy optimization for industry use cases.
   - This is due to Scale's focus on capturing the processes that agents would follow and operationalizing them.
- **World Generalist Agents Incoming?**: One member thinks that the team has a pretty good shot at a **generalist world agent** that can go into robots or computers or neural interfaces eventually.
   - He also linked to [a tweet about The Illusion of the The Illusion of the Illusion of the Illusion of Thinking](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1385034755837530225)** (3 messages): 

> `Bigger Brains, Frontal Lobe` 


- **Bigger Brains Brainstorming Begins**: A member shared a [YouTube video](https://youtu.be/-G1SdsRXL7k) about the concept of *bigger brains* and its potential implications.
   - They mentioned watching a talk with Lex Fridman and MLST, presumably related to the same topic.
- **More Brains More Problems**: A follow up discussion questioned whether *bigger brains* is really the solution.
   - One member mentioned bigger brains might lead to bigger problems.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1385032728063512676)** (8 messages🔥): 

> `Meta Research, Zuckerberg's AI strategy, Generalist world agent` 


- **Meta Publishes Two New Papers**: Meta's research team released two new papers: [https://arxiv.org/abs/2506.10077](https://arxiv.org/abs/2506.10077) and [https://arxiv.org/abs/2505.12514](https://arxiv.org/abs/2505.12514).
- **Zuck's Master Plan for AI Dominance**: One member speculates that Mark Zuckerberg aims to merge **Meta's research and Llama teams** to leverage vision and thought leadership, while focusing on policy optimization for industry use cases.
- **Meta Eyes Generalist World Agent**: According to a conversation, Meta is aiming to develop a generalist world agent that can be integrated into robots, computers, or neural interfaces.
   - The member also linked to a [tweet](https://fxtwitter.com/rohanpaul_ai/status/1935047424948781098?t=HMmgUOtz-nwBdgcd7cTXOw&s=19) and another [tweet](https://fxtwitter.com/rohanpaul_ai/status/1935746720144544157) related to *the illusion of thinking*.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1384973684757762058)** (110 messages🔥🔥): 

> `Claude 3.7, MiniMax-M1, Free 1M context model, Free Gemini version, Glazing models` 


- **Token Rebalancing Troubles Triggered by Costly Claude**: Users are rebalancing **output vs input tokens** due to the doubling of **input costs** between Claude 2.5 preview and live, impacting high-frequency use cases.
- **Free Gemini Lost, Gemini 2.0 Flash surfaces**: The free version of **Gemini** is unavailable on Hugging Face because it is made by Google, but a free **Gemini 2.0 Flash** model with 1M context is available on [OpenRouter](https://openrouter.ai/google/gemini-2.0-flash-exp:free).
- **DeepInfra Deploys Discounted Gemini**: [DeepInfra](https://deepinfra.com/google/gemini-2.5-pro) is serving **Google Gemini 2.5 Pro/Flash** on their own hardware at lower prices than Google, but it is likely a proxy to Google's API with negotiated cloud provider pricing.
- **Deepseek R1 0528 recommended for coding**: Members recommend the new **Deepseek R1 0528** as a good coding model, particularly because unlike older models like the 0324 version, it is a thinking model and therefore better for code.
   - It was reported that the **0528** version *does not support prefill* although this was later retracted.
- **OpenRouter's Impressive Economics**: OpenRouter's economics are impressive, processing around **$126k** in usage of **Claude Sonnet 4** in a single day.
   - One member compared OR to the Mastercard/VISA of AI while noting that their growth and ubiquity are insane and well deserved, though they only make about **5%** in fees.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1384971021261738116)** (70 messages🔥🔥): 

> `Manus video generation, AI errors and credit usage, Manus failure feedback loop, Manus Fellow program interview status, Technical debt` 


- **Manus video generation**: Members expressed disappointment that **Manus video generation** is not free, but understand it's costly due to compute power, while others said the user still had to put in some work to get it to work.
- **AI errors eat credits without results**: Members discussed how **Manus** sometimes runs into errors that can't be fixed, running down credits without completing the task, and one member compared it to *paying for a rotten burger because the cook still put in the work*.
   - Another member pointed out the AI still uses compute power even with errors, costing money, but it's frustrating when **credits are burned without usable output** and suggested a charge threshold of **80%** defined success criteria, and linked to [this youtube video](https://m.youtube.com/watch?v=5PuofaVqXNI).
- **Manus rewarding a broken feedback loop**: A member stated that the real issue is the system's *silent failure* to recognize it has failed, burning credits with no real-world success or internal awareness.
   - They inquired about concrete actions to fix this broken reward model and why credits are charged when success criteria aren’t met, also mentioning **losing 70,000 points**.
- **Fellow applicant wonders where acceptance status is**: A member who completed a **Manus Fellow program interview** over six weeks ago is still awaiting acceptance or rejection and seeks a simple status update.
   - They emphasized that unresolved issues could become a breach of trust, seeking a specific, dated action plan.
- **Technical debt and accumulation of small errors**: A member highlighted that the accumulation of small errors during computations can lead to unexpected failures, even if individual errors seem insignificant.
   - Another member agreed, pointing out that measuring divergence is useful but not foolproof, and they often got hallucinated results from various ai platforms where **credits are not refunded**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1385035406981988404)** (9 messages🔥): 

> `Mojo shirt, EmberJson, simdjson, Python Implementation` 


- **Mojo shirt acquired through LinkedIn instead of X**: A member asked about acquiring a **Mojo shirt** by sharing on **LinkedIn** instead of **X** (Twitter), due to better reach on LinkedIn.
- **EmberJson's Creator Identified**: A member inquired about the creator of the **EmberJson** library, and another member identified themselves as the one who worked on it.
   - They are waiting for *language developments* before digging into optimizing it more.
- **EmberJson Performance Compared to simdjson**: A member asked about **EmberJson's** performance compared to **simdjson** or **zimdjson**.
   - The creator of EmberJson mentioned it's still well below them, estimating around **200-500 MB/s** based on CPU and data, but is waiting for further language developments before optimizing it.
- **EmberJson vs Python Performance**: A member inquired if **EmberJson** is comparable to the **Python implementation**.
   - The creator responded that it generally seems to be roughly **2x faster** than the Python stdlib in limited testing.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1385002991714570260)** (46 messages🔥): 

> `Mojo crashes, MAX supports Blackwell, SymPy in Mojo, Claude Code and Mojo` 


- **Mojo crashes trigger Bug Report Debate**: A member reported a **segmentation fault** and asked if a bug report was necessary for Mojo runtime errors.
   - Another member responded that the crash looked like a **stdlib or compiler issue** and linked to [issue #4857 on GitHub](https://github.com/modular/modular/issues/4857).
- **Modular stealthily supports Blackwell**: A core dev mentioned that **MAX** supports **Blackwell** GPUs, but it isn't widely advertised *yet*.
   - They encouraged users with **5090** systems to test and provide feedback, noting more *perf and other work* is needed before an official announcement.
- **Implementing SymPy's suffering in Mojo**: A member inquired about the feasibility of implementing something like **SymPy** in **Mojo**.
   - Another member replied that it should be possible, but with *great pain and suffering*.
- **Claude helps Mojo diagram Matmul, CUDA translate**: One member shared that the modern agentic systems, in particular **Claude Code**, is *stunning in their capabilities*
   - With the right context for modern **Mojo** and **MAX** (**modular repo**, **Modular docs**) **Claude Code** has one-shot a huge range of tasks: *draw a diagram of architecture specialization for the matmul operation inside MAX, create a Mojo function that can be called from Python which factors large number efficiently using SIMD, translate this CUDA reference kernel to Mojo, and it keeps going*.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1385198997958033470)** (12 messages🔥): 

> `Model Compilation Failures, RDNA4 GPU Support, CI Testing for Max Models, Error Message Improvements` 


- **Model Compilation Failures Plague Max Users**: A user reported that every **Max model** they tried to serve failed to compile on both **GPU** and **CPU**.
   - The user suggested adding a **CI step** similar to [rust-lang/crater](https://github.com/rust-lang/crater) to prevent PRs from breaking hosted Max models.
- **RDNA4 GPUs are Tier 3 Compatible**: The team only recently enabled basic support for **RDNA4 GPUs**, like the **9000-series**, this week, but full models are not yet running on them.
   - The **9000-series** GPUs are classified as *Tier 3: Limited compatibility* until models can fully run on them.
- **Error Message Improvements Planned**: The team acknowledged that error messages need improvement, as the current constraint error messages aren't clear.
   - The user was hitting these errors because not all kernels have been made compatible with the **RDNA4 architecture**.
- **GPU requirement docs shared**: To work around this issue, the Max team provided the documentation page for [system specs](https://docs.modular.com/max/faq#system-requirements) and [compatible GPUs](https://docs.modular.com/max/faq#gpu-requirements).
   - The original reporter's rdna4 9070 is not fully supported.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1384992682832564295)** (51 messages🔥): 

> `Midjourney's Video Model, CoreWeave and Weights & Biases AI Inference, Meta Hiring Nat Friedman & Dan Gross, Profound Series A Funding, Arcee AI AFM-4.5B-Preview Model` 


- **Midjourney Animates with Video Model V1**: Midjourney unveiled **Version 1** of its **Video Model**, enabling users to animate **Midjourney-generated** or **external images** with options for 'automatic' and 'manual' animation settings, priced around **8x** an image job, available on the web at launch, as seen on [X](https://x.com/midjourney/status/1935377193733079452).
   - The new 'Image-to-Video' feature offers 'high motion' and 'low motion' options, and videos can be extended, with pricing subject to adjustments for sustainability and insights benefiting future image models.
- **CoreWeave and W&B Launch AI Inference Services**: CoreWeave and Weights & Biases launched new AI inference services including an inference endpoint for models like **DeepSeek R1-0528** and **LLama-4 Scout** with OAI Compatible APIs, and Online Evaluation tools as per [this tweet](https://x.com/altryne/status/1935412384283107572).
   - These services running on CoreWeave GPUs aim to offer more competition and flexibility in the AI infrastructure space, providing real-time LLM judgment.
- **Meta Courts Nat Friedman and Dan Gross**: Meta is reportedly in talks to hire former GitHub CEO **Nat Friedman** and AI scientist **Dan Gross** to bolster its AI efforts, as reported by [money.usnews.com](https://money.usnews.com/investing/news/articles/2025-06-18/meta-in-talks-to-hire-former-github-ceo-nat-friedman-to-join-ai-efforts-the-information-reports).
   - Reactions ranged from disbelief at them reporting to Alexandr Wang, to the impossibility of them reporting to **Alexandr Wang**.
- **Profound Bags Series A Funding**: **Profound**, led by James Cadwallader and Dylan Babbs, secured a **Series A** funding round, emphasizing their role in the evolving search landscape, also co-invested in by **SagaVC** as revealed in [this post](https://www.stories.sagavc.com/posts/profound).
   - Discussions in the thread questioned Profound's methods for measuring and making recommendations in a post-search optimization era.
- **Arcee AI Debuts AFM-4.5B-Preview Model**: Arcee AI unveiled its new foundation model, **AFM-4.5B-Preview**, designed for enterprise use with under **10B parameters**, prioritizing efficiency and regulatory compliance, in collaboration with DatologyAI as announced [here](https://x.com/lucasatkins7/status/1935382123155964081?s=46).
   - The model utilizes advanced techniques like **MergeKit** and **YaRN**, with plans to openly release **AFM-4.5B** and its base model in early July, alongside open-sourcing previously closed models like Virtuoso-Large.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1385316378826772542)** (2 messages): 

> `` 


- **Placeholder Topic 1**: This is a placeholder summary to satisfy the minimum items requirement.
   - Additional details about the placeholder topic can be added here.
- **Placeholder Topic 2**: This is another placeholder summary to meet the validation criteria.
   - Further elaboration on the second placeholder topic goes here.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1385140703813832735)** (1 messages): 

> `Triton Tutorial, Deep-spin lab` 


- **Deep-spin lab releases Triton Tutorial**: The **Triton tutorial** covers fundamentals in slides and then goes hands-on, starting with **vector addition** and ending with **sparsemax(QK^T)V**.
   - The [tutorial](https://github.com/deep-spin/triton-tutorial) was created for the lab but it may be helpful for others too.
- **Triton Vector Addition Example**: The tutorial starts with a hands-on example of **vector addition** to introduce the fundamentals of Triton.
   - It progresses to more complex operations like **sparsemax(QK^T)V**, demonstrating practical applications.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1385267675491467375)** (6 messages): 

> `cuda-gdb, cudaErrorUnsupportedPtxVersion, Nvidia Driver Versions` 


- **User faces CUDA Debugging error**: A user encountered a `cudaErrorUnsupportedPtxVersion` error while using **cuda-gdb**, and believes they need to upgrade their GPU driver.
   - The error indicates the CUDA toolkit version is not compatible with the current driver, requiring an update to resolve the issue.
- **Figuring out the latest Nvidia drivers.**: A user asked how to find the latest compatible Nvidia driver version.
   - Another member linked [this Nvidia documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id7) which shows the driver version shipped with each **CUDA Toolkit version**, suggesting it as a good reference.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1385262254789754910)** (1 messages): 

> `Big Model Serving, Parallelism Techniques, AI Infra Frameworks, vLLM & Kubernetes, Nvidia Dynamo/Triton` 


- **Deep Dive into Big Model Serving Techniques**: The discussion revolves around the parallelism techniques employed by big companies to serve large AI models on extensive infrastructure.
   - It seeks insights into popular frameworks used for AI infrastructure beyond training-focused libraries like **Accelerate** and **DeepSpeed**.
- **vLLM, Kubernetes, and Best Practices**: The conversation questions whether integrating **vLLM** with **Kubernetes** aligns with best practices for model serving.
   - It highlights **vLLM** as a popular choice, especially for inference, and aims to understand its optimal deployment strategies.
- **Nvidia's Dynamo: Rebranded Triton?**: The discussion also questions the use of **Nvidia Dynamo**, formerly known as **Triton**, and its prevalence in serving models.
   - It acknowledges **Triton's** historical significance in inference and explores its current relevance under the new **Dynamo** branding.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1385166884898471998)** (2 messages): 

> `LLMs, AusysAI, LLM Abstraction Levels` 


- **AusysAI Explains LLMs in Layman's Terms**: AusysAI posted a [blog post](https://www.ausysai.com/posts/explaining-how-llms-work-7-levels-of-abstraction) explaining how **LLMs** work in an intuitive way.
   - The post serves as a primer for newcomers as well as a review of the fundamentals for practitioners using **7 levels of abstraction**.
- **Seven Levels of LLM Abstraction Decoded**: The AusysAI blog dissects **Large Language Models** (LLMs) through **seven levels of abstraction**.
   - Aimed at both newcomers seeking a foundational understanding and seasoned practitioners needing a refresher.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1385082419358601277)** (1 messages): 

> `ADAS Platform Software Engineer, Lucid Motors, GPU background` 


- **Lucid Motors ADAS team seeks GPU expert**: Lucid Motors' ADAS Platform Software team is hiring a **Sr. Software Engineer** with **GPU** expertise and **Linux/QNX** experience; candidates are encouraged to mention **Arun Paruchuri** in their application, and a [link to the job posting](https://job-boards.greenhouse.io/lucidmotors/jobs/4700944007) was included.
   - A team member stated they recently joined and are pleased with the team's work.
- **Arun Paruchuri joins Lucid Motors ADAS team**: Arun Paruchuri has recently joined the ADAS team at Lucid Motors and is enjoying the work.
   - He encourages candidates applying for the **Sr. Software Engineer** position to mention his name.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1384973778244604036)** (2 messages): 

> `Distributed Training Course, Unsloth Meetup with Google DeepMind Gemma` 


- **Distributed Training Course Offered**: A friend is teaching a course on distributed training and invited others to join, mentioning that he is a maintainer of *accelerate* from transformers.
   - The course promises learning from big minds. [Sign up here](https://maven.com/walk-with-code/scratch-to-scale?promoCode=matej26).
- **Unsloth Hosts Gemma Meetup in SF**: A meetup with the Google DeepMind Gemma folks will be hosted in SF, featuring a talk about **GRPO** and **kernels**.
   - They are accepting 3-minute lightning talks about **kernels** and **open-source AI**. [RSVP here](https://lu.ma/gemma-unsloth).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1385199369283702864)** (13 messages🔥): 

> `ModuleNotFoundError agents fix, PR review, CI/CD pipeline` 


- **Dot Prefix Dodges ModuleNotFoundError**: A member resolved a `ModuleNotFoundError` by using a **relative import** (`.agents.basic_agent`) instead of an **absolute import** (`agents.basic_agent`).
   - The member confirmed that using a **relative import** solved their import error, which had previously required manually setting the `PYTHONPATH` environment variable.
- **Call For PR Review**: A member requested a review of their [pull request](https://github.com/JackHopkins/factorio-learning-environment/pull/228) after addressing comments and making contributions to the project.
   - Another member confirmed they had already reviewed the PR and **no comments were blocking the merge**.
- **Implementing CI/CD Pipeline Discussed**: The team discussed implementing a **CI/CD pipeline** to ensure tests pass before merging changes, addressing access issues, and refactoring the codebase.
   - The conversation also covered the potential of using **Factorio's replay files** for training agents, including the technicalities of deserializing replay data into JSON.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1385020093599322163)** (14 messages🔥): 

> `CUTLASS examples, CuTe indexing errors, TensorSSA assignment limitations, vectorized relu kernels, dynamic ranges in CuTe` 


- **CuTe Indexing Error Strikes Novice**: A user encountered an indexing error while trying to implement a `vectorized_relu_kernel` using CuTe, specifically related to incompatibility between `!cute.layout` and `!cute.coord` as shown in their [screenshot](https://cdn.discordapp.com/attachments/1362196854460383353/1385183408354885712/Screenshot_2025-06-19_at_2.34.04_PM.png?ex=6855ccd4&is=68547b54&hm=75b031aceece5b4d12addb0e069f282ce524a28138dd90ba1b518dc37654c0aa&).
   - The error message, *unable to compute crd2idx with* `!cute.layout<"((1,8)):((0,1))">` *and* `!cute.coord<"(0,0)">`*, indicated a mismatch between the tensor layout and the coordinate used for indexing.
- **CuTe DSL TensorSSA Immutability Anguishes Aspiring Alchemist**: A user discovered that `TensorSSA` values in CuTe are immutable, preventing direct assignment like `x[(0, i)] = max(0, x[(0, i)]).to(x.dtype)` due to `x` being a temporary value rather than a mutable buffer.
   - The suggested workaround involves using `cute.where` for elementwise operations or `cute.make_fragment_like` for register memory tensors to enable assignment as described in the [CuTe DSL limitations docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/limitations.html).
- **Dynamic Ranges Ruffle Register-Resident's Robes**: A user inquired about creating a tensor from a dynamically sized list, encountering limitations with dynamic ranges in CuTe DSL.
   - It was clarified that while statically known ranges at JIT time can be tracked to fill tensors, dynamic ranges and Python data structures with dynamic lengths are not currently supported, and dynamic indexing of `y` is not allowed, as depicted in their [screenshot](https://cdn.discordapp.com/attachments/1362196854460383353/1385189040378085386/Screenshot_2025-06-19_at_2.56.15_PM.png?ex=6855d212&is=68548092&hm=72f0b5575c09e92aead884ea5f7946a0b86eb161ec1769074bd8d433310af2be&).
- **Cutlass's CuTe Elementwise Addition Example Enchants Engineers**: A user was directed to the [elementwise_add.ipynb](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/notebooks/elementwise_add.ipynb) notebook in the CUTLASS examples for guidance on elementwise operations in CuTe DSL.
   - This example demonstrates a basic addition operation, showcasing how to define and launch a kernel for elementwise tensor addition, providing a foundation for understanding more complex operations.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1384972270044315750)** (27 messages🔥): 

> `Gemini 2.5 Pro Configuration, Aider Edit Mode Issues, Deepseek Free on OpenRouter, GitHub Copilot Complaints, Custom Benchmarks` 


- **Gemini Configuration Tweaks Aider's Performance**: Members discussed manually adding configurations to `.aider.model.settings.yml` such as `thinking_tokens` for the **Gemini 2.5 Pro preview** to avoid warnings, and using the command `aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k --edit-format diff-fenced` as an alternative.
   - It was noted that the **0605 version with 32k thinking tokens** is excellent for coding but subpar for chatting, and that Gemini 2.5 is **4x more expensive** than the preview version when not using `thinking_tokens`.
- **Aider Edit Mode Causes Project Chaos**: A user reported that using Aider's edit mode with Claude models resulted in issues such as **unintended full application changes**, **code appending**, and **CSS class errors** like adding `border.boder` without declaration.
   - Another user asked about changing the edit format without restarting the chat and received the answer to use `/chat-mode diff-fenced`.
- **Deepseek Free Stuck in Infinite Loops on OpenRouter**: A member reported experiencing issues with **Deepseek Free on OpenRouter** getting stuck in a loop, repeatedly posting the same files for changes.
   - Setting the `edit-format` to `whole` provided a temporary solution but might have just been that turning the experiment caching on helped.
- **GitHub Copilot Users Complain about Claude Sonnet Limits**: Users on the r/githubcopilot subreddit are reportedly complaining about only receiving **300 calls of Claude Sonnet** with an 80k context limit for $10 a month, despite getting unlimited tool calls and GPT-4.1/4o.
   - It was also implied that Deepseek and other similar tools were entirely free.
- **Custom Benchmark Shows Poor Llama Model Performance**: A member created a custom benchmark and noted that **Llama models performed poorly**; the benchmark image was attached to the message: [image.png](https://cdn.discordapp.com/attachments/1131200896827654149/1385357720286138381/image.png?ex=6855c66b&is=685474eb&hm=a2f92d5cbb4abede7876d489911310283847b1e3cf50e89546d0142f81068a76&).
   - The benchmark was described as a **single-shot test** using riddles and codename challenges, and details on the languages used or multi-pass aspects were requested.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1384986986040918170)** (11 messages🔥): 

> `Gemini 2.5 Flash, Deepseek v3, --watch-files and Jupyter notebooks, aider adding back code that was removed, MERN projects` 


- ****Gemini 2.5 Flash** for Copy-Paste Editing**: A member is using **Gemini 2.5 Flash** in `whole` mode as an editor for copy-pasting, but worries about using `deepseek v3` as an editor for **Gemini 2.5 Pro**.
   - They are planning to *deep dive on the editor decision flow and how much it can dumb down the main model*.
- **`--watch-files` Triggers in Jupyter Notebooks**: The `--watch-files` command with Jupyter Notebooks requires the trigger **AI** to be at the start of a comment, ie `AI! fix this`.
   - The trigger won't work if it is at the end `# this fails AI!` because in the JSON the lines end with `",` causing the trail AI to not match.
- **Aider Appending Code Errors Reported**: When using edit mode in aider, a member reported that *instead of changing the targeted files, it started to change the whole application*.
   - Additional errors reported involved Aider *appending code which has been already written for example imports React twice*.
- **Aider Continues Adding Removed Code**: A member is seeking advice on how to stop Aider from re-introducing code that has been intentionally removed, specifically related to **pandas** code for creating columns that are not needed.
   - A member suggested to *try restricting the files. Also sometimes you have to discard bad changes. You can use /undo*.
- **MERN Project in Aider**: A member is working on a project to build a **full-stack website** with the help of **Aider**, mostly **MERN** projects.
   - They are interacting with the chatbot to generate and edit code.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1385032841015988344)** (21 messages🔥): 

> `MCP Server Setup, Claude 2025-06-18 spec support, Loading MCP Tools, MCP SDK for Go, FastMCP Errors` 


- **MCP Server Setup Simplified**: Users discussed the easiest way for someone to use a newly created **MCP server** running on Docker, suggesting they grab a *credentials.json* from Google Cloud Console.
   - The conversation also touched on whether the new **Claude release** would ship with support for the **2025-06-18 MCP specification**.
- **Loading MCP tools without Client Session**: A member inquired about loading **MCP tools** without creating a client session, referencing their success with **OpenAI agents** in a similar context.
   - The user has a local **MCP server** and it is taking the MCP session as a parameter and loading the tools.
- **MCP SDK for Go Missing**: Users noted the absence of an official **MCP SDK for Go**, seeking recommendations for existing implementations.
   - One suggestion pointed to [mark3labs/mcp-go](https://github.com/mark3labs/mcp-go) as a promising **Go implementation**.
- **FastMCP 'host' Error Baffles User**: A user encountered a **TypeError** with **FastMCP**, specifically an unexpected keyword argument *'host'* despite its presence in the documentation.
   - The user was running their server code with `uv run server.py` and received the error during the `mcp.run()` call.
- **Solo Coder Base44 Sells to Wix!**: A link to a TechCrunch article [TechCrunch](https://techcrunch.com/2025/06/18/6-month-old-solo-owned-vibe-coder-base44-sells-to-wix-for-80m-cash/) shows that a 6-month-old solo-owned coder **Base44** sold to **Wix** for **$80M**.
   - A user posted *🤯* when sharing the link.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1385093104914141205)** (11 messages🔥): 

> `Windsurf Configuration Issues, Enact Protocol for Tool Registry, mcp-webcam Updates, Muppet Kit Devtool, Dagger Container MCP` 


- ****Windsurf** Configuration Defies User!**: A user reported struggling to configure **Windsurf** to access the humanizer AI sub GPT from OpenAI after multiple attempts to install dependencies like Node.js.
   - No solutions were offered in the discussion.
- ****Enact Protocol** Extends MCP Tooling!**: A user requested feedback on the [**Enact Protocol**](https://enactprotocol.com/), described as an extension of MCP tools definition for a tool registry.
   - No feedback was provided in the messages.
- ****mcp-webcam** Adds Streaming Support!**: The **mcp-webcam** project now supports **Streamable HTTP**, has a multi-user mode, and offers easier sampling requests, with [the repo available on GitHub](https://github.com/evalstate/mcp-webcam).
   - Integration is built-in to **VSCode v1.101.0** and **fast-agent**, accessible via the MCP Connection URL, and can be run locally with `npx @llmindset/mcp-webcam`.
- ****Muppet Kit** Debugs MCP Servers!**: **Muppet Kit**, a devtool for testing and debugging MCP servers, is becoming more stable, with a [GitHub repository](https://github.com/muppet-dev/kit) available.
   - Features include an Explorer, Playground, MCP Scan, Tracing, and History, accessible via `npx muppet-kit inspector`, with more info at a [tweet](https://x.com/MathurAditya7/status/1923719099961479246).
- ****Dagger** Container MCP Emerges!**: A link to a blog post about **Dagger's** container MCP was shared, referencing a hypothetical future blog post [block.github.io](https://block.github.io/goose/blog/2025/06/19/isolated-development-environments/).
   - The title is *Isolated Development Environments*.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1384978956570071090)** (2 messages): 

> `MCP vs Vector Search, Agent Memory, Memory Blocks, Enterprise data` 


- **MCP Doesn't Kill Vector Search**: Despite the new possibilities for agents to connect directly to data sources via the **MCP protocol**, preprocessing and indexing are still needed for unstructured data, since 90% of enterprise data lives in **PDFs**, **PPTs**, and on the web [(LlamaIndex's Tweet)](https://twitter.com/llama_index/status/1935419760898093435).
- **LlamaIndex Introduce Memory Blocks for Agent Memory**: Recently, **LlamaIndex** started to introduce flexible **Memory Blocks** to **LlamaIndex** to serve different purposes of agent memory [(LlamaIndex's Tweet)](https://twitter.com/llama_index/status/1935774624257843217).
- **Memory Blocks Livestream on Agent Memory**: A livestream about Memory Blocks will be held next week, details to be announced soon [(LlamaIndex's Tweet)](https://twitter.com/llama_index/status/1935774624257843217).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1385282164610044004)** (29 messages🔥): 

> `Unit Testing LlamaTS, Token Counting with Gemini, LLM Client Access, Custom LLM Class, Python Type Safety` 


- **Unit Tests Fail with ES Module Issues**: A member reported encountering issues when writing unit tests for **LlamaTS** using either **Mocha** or **Jest** due to **ES module issues**.
   - They were seeking advice on running unit tests for **AI projects** in general.
- **Token Counting Tango with Gemini**: A member inquired about an example of token counting for **Vertex/Gemini** via **LlamaIndex**, noting that the default **tiktoken** tokenizer doesn't work with **Gemini**.
   - They referenced [Google's documentation on token counting](https://ai.google.dev/gemini-api/docs/tokens?lang=python) and shared a possible code snippet, but ran into client definition issues.
- **Debate over Accessing LLM Clients**: Community members discussed how to access the underlying client object from **LlamaIndex's LLM** wrappers to perform custom actions like token counting.
   - The potential use of underscored properties (e.g., `llm._client`) was discussed, alongside the idea of adding a `get_client()` method to `llama_index.core.llms.llm.LLM`, with some concerns raised about [type safety](https://mypy.readthedocs.io/).
- **Custom LLM Class Considered**: To address the need for custom token counting, members contemplated wrapping `llama_index.core.llms.llm.LLM` in a custom LLM class.
   - The consensus seemed to lean towards this approach due to the impracticality of modifying all existing LLM integrations, though it was acknowledged as a lower priority.
- **Python Type System Snafu**: A member expressed frustration with Python's type system when trying to pass a tokenizer function to `TokenCounter`.
   - Despite providing a valid function, they encountered a type error because `TokenCounter` expects a function that could also be None.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1385209476268953711)** (7 messages): 

> `NBLM Portraits Digital Avatar, NBLM personalized voice and design, NBLM Video Feature, NBLM Audio Length` 


- **NBLM User Gaga Over Portraits Digital Avatar**: A user expressed excitement about **NBLM's Portraits** feature, viewing it as a digital avatar that can be used as a product or shared with clients and teams, sharing a link to [Google Labs Portraits](https://labs.google/portraits/login/kimscott).
   - The user is eager for personalized **voice**, **design**, and **interface** options, planning to use Portraits as a value proposition for new business by loading specific client information.
- **Users Inquire on Video Feature in NBLM**: A user inquired about the timeline for introducing a **video feature** in NotebookLM.
   - No information was given.
- **NBLM Generates Shorter Audio Lengths**: A user noted that when using the same prompt in **Dutch**, NotebookLM produces an **8-minute audio**, whereas in other languages it may be shorter, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1124403655819415592/1385249546225061908/Screenshot_2025-06-19_at_15.25.28.png?ex=685561ac&is=6854102c&hm=be028a00040ebbbc8801a4b66215a66f3643d8091cc4e9263ff1ee6015750cbd).
- **Combining Sources Creates Longer Audio**: A user realized combining sources for a topic will yield longer audio.
   - Another user asked if this behavior was on a paid version.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1384979038313119987)** (13 messages🔥): 

> `Audio overviews in non-English languages, AI Agents in NotebookLM, Public Notebook library, NotebookLM access issues, NotebookLM sharing with large audiences` 


- **Audio Overviews Stumble in Non-English**: A user reported issues generating audio overviews longer than 10 minutes in Italian, noting that even custom prompts don't help and [another user confirmed](https://discord.com/channels/1124402182171672732/1366873891938504827/1366873891938504827) this is a known issue for non-English languages.
- **NotebookLM Agents: Deep Research for Nerds**: A user suggested creating AI "Agents" in NotebookLM, pre-trained and optimized for specific knowledge fields like Math, Physics, Biology, or Chemistry to improve accuracy and reliability.
- **NotebookLM Access? Can't Enter the Site**: One user reported being unable to access the NotebookLM site, only seeing a message that they *"can't enter the site"*.
- **NotebookLM Social: Public Notebook Library**: A user inquired about a library of public notebooks to browse what others have built and want to share.
- **NotebookLM: Plus vs. Enterprise for Mass Sharing**: A user asked whether a NotebookLM Plus subscription is sufficient for sharing a notebook with 200+ people, or if an Enterprise plan is required.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1384974074895401022)** (15 messages🔥): 

> `New AI R&D channel, EU GDPR compliance for Embed v4, Cohere projects contribution, Cohere 4 AI` 


- **AI R&D Channel Launches**: A new channel dedicated to **AI research and development** has been created: <#1384974112841269399>.
- **GDPR Compliance of Embed v4**: A member inquired about **EU GDPR compliance** for **Embed v4**.
   - They are waiting a response from the Cohere team to clarify if it's on the roadmap, due to its excellence for **multimodal RAG documents**.
- **New member asks how to contribute**: A new member inquired about how to join and contribute to existing **Cohere projects**.
   - A helpful member suggested looking into **Cohere 4 AI** and shared the [application link](https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw) as well as to share their research in the new channel <#1384974112841269399>.
- **GDPR questions should be directed to Support**: A member asked about **EU GDPR compliance** for **Embed v4**.
   - A member of the Cohere team asked that the question be emailed to [support@cohere.com](mailto:support@cohere.com).


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1385001218979663913)** (3 messages): 

> `Volunteer Opportunities, Cohere AI Program` 


- **User Seeks Volunteer Opportunities**: A member introduced themselves and expressed interest in finding volunteer opportunities within the community.
- **Applying for Cohere AI Program**: A member suggested that if the user applied for the **Cohere AI Program**, they will receive an email informing them about available research opportunities and projects.


  

---


### **Cohere ▷ #[🔬-research](https://discord.com/channels/954421988141711382/1384974112841269399/1385102143819874304)** (1 messages): 

> `AI Research, Secure Machine Learning, Privacy Preservation, AI-driven Cybersecurity, Computer Vision and NLP` 


- **Newcomer Yasir Khan Joins Cohere Labs Open Science**: Yasir Khan, a newcomer to the Cohere Labs Open Science Community, expresses interest in collaborating on AI research projects on a voluntary basis.
- **Yasir's Research Areas**: Yasir's research areas include **Secure Machine Learning**, **Privacy Preservation**, **AI-driven Cybersecurity**, **Computer Vision**, and **NLP**.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1384977380623253524)** (12 messages🔥): 

> `adjoint and .mh implementation, Whisper bounty removal, complex tensors for tinygrad` 


- ****Adjoint** and **.mh** left unimplemented!**: Members discussed why **adjoint** and **.mh** are not implemented in tinygrad, and the answer is that developers want to keep complexity of the project to an absolute bare minimum, and the same functionality of **adjoint** can be done using `x.transpose(-2, -1)`.
- **Whisper bounty sticking around!**: Members discussed whether the **$200 Whisper bounty** will be removed, and the consensus is that both bounties are complementary.
   - One bounty deals with fixing an existing **Whisper** example, and the new one's end goal is making it work at all on a webpage.
- **Complex tensors not implemented yet!**: A member inquired about implementing **conjugate**, and learned that tinygrad has no implementation of complex numbers as of now, so this cannot be done.
   - However, the member stated that they created their own [implementation of complex tensors](https://cdn.discordapp.com/attachments/1068976834928193609/1385280077079777501/complex_tensor.py?ex=68557e1b&is=68542c9b&hm=55b05763c0469aa8cacc37f4159ec42c988c0b125d7a662629e3085b05abb2b7) for tinygrad, but it is by no means complete.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1384993797771104438)** (12 messages🔥): 

> `Mr. Beast Spam, Implementing GPT4All in Python` 


- **Discord member told to stop Mr. Beast spam**: A Discord member was asked to stop spamming **Mr. Beast** content in the channel.
- **User seeks guidance on Python implementation of GPT4All**: A member is looking for assistance or a tutorial on how to implement **GPT4All** into their **Python code**.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1385054434647609364)** (11 messages🔥): 

> `Python 3.9, typing.Optional, future annotations, deprecation of 3.9, pytorch compatibility` 


- **Python 3.9 CI Complains About `| None`**: Python 3.9 CI is complaining about `| None` typehinting, raising the question of whether it's okay to use `Optional`.
   - It was noted that `X | Y` type hinting is available starting with Python **3.10**.
- **`__future__` Annotations Enable `X | Y` on 3.9**: Using `from __future__ import annotations` will allow `X | Y` to work on Python **3.9**, and also get rid of string types for custom objects.
   - This approach sets the stage for future proofing with `list`, `dict`, `tuple`, `X | Y`, `X | None` type hints.
- **Python 3.9 Deprecation Recommended**: One member suggested simply deprecating Python **3.9** as a solution, noting that it will soon be out of life.
   - Another member mentioned using **3.13** features and preferring **3.12** generics syntax, but acknowledged the extensive changes required.
- **Torchtune Aligns with Pytorch Python Support**: The discussion noted that **torchtune** is trying to stay in line with **pytorch** regarding Python version support.
   - Using Python **3.10** offers a good compromise since new features can be obtained from `typing_extensions`.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1385017788816425070)** (10 messages🔥): 

> `DSPy for Beginners, Finetuning Llama models, Compiled DSPy Prompts JSON format, Prompt-like DSPy signatures, DSPy with Amazon Bedrock (Claude models)` 


- **DSPy Newbies get up to Speed**: A user new to **DSPy** inquired about where to start learning and asked for tips.
   - One member shared a [YouTube video](https://www.youtube.com/watch?v=LCEmiRjPEtQ) that offers a good explanation of **DSPy**.
- **Operating Systems are LLMs?**: A member found a YouTube analogy of **LLMs to operating systems** very much in line with **DSPy's philosophy of a higher-level language**.
   - They further elaborated that **DSPy** is like **C**, which can be run on different backends and compiled specifically for them, abstracting away the specific underlying assembly dialect or the specific **CPU instruction set**.
- **Bedrock Users are Rocked**: A user reported poor results when using **DSPy with Amazon Bedrock (Claude models - haiku, sonnet v2)** for classification and rewriting tasks.
   - They wondered if the prompt from **DSPy** might not be doing well with how it was trained.
- **Minting Mania Officially Starts**: A team officially decided to allow individuals to start minting today.
   - Instead of whitelists, they decided to give people who are online during this time the ability to mint [here](https://openseacix.vercel.app/).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1385127399485739030)** (1 messages): 

> `Agentic AI Summit, UC Berkeley, Early Bird Tickets` 


- **Agentic AI Summit: A Berkeley Bonanza!**: The Agentic AI Summit will be held **August 2, 2025** at **UC Berkeley**, building on the popular [LLM Agents MOOC](https://rdi.berkeley.edu/events/agentic-ai-summit) with **1,500+** in-person attendees.
   - The summit features keynotes, panel discussions, workshops, a startup spotlight, and AgentX Demo Day.
- **Early Bird Tickets: Last Chance for Discounted Deals!**: Early bird pricing for the Agentic AI Summit ends **June 30, 2025**, offering discounted passes for students (**$25**), startups (**$60**), and industry professionals (**$80**).
   - Students and indie developers can apply for fee waivers, and tickets can be purchased [here](https://na.eventscloud.com/ereg/index.php?eventid=842399).
- **Speaker Lineup: AI Luminaries Assemble!**: The Agentic AI Summit features industry and academic leaders, including **Vinod Khosla** (Khosla Ventures), **Ion Stoica** (Databricks, Anyscale), **Dawn Song** (UC Berkeley), **Sergey Levine** (Physical Intelligence), and others.


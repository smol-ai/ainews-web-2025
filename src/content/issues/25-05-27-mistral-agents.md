---
id: MjAyNS0w
title: Mistral's Agents API and the 2025 LLM OS
date: '2025-05-27T05:44:39.731046Z'
description: >-
  **The LLM OS** concept has evolved since 2023, with **Mistral AI** releasing a
  new **Agents API** that includes code execution, web search, persistent
  memory, and agent orchestration. **LangChainAI** introduced the **Open Agent
  Platform (OAP)**, an open-source no-code platform for intelligent agents.
  **OpenAI** plans to develop **ChatGPT** into a super-assistant by H1 2025,
  competing with **Meta**. Discussions around **Qwen** models focus on
  reinforcement learning effects, while **Claude 4** performance is also noted.
  The AI Engineer World's Fair is calling for volunteers.
companies:
  - mistral-ai
  - langchain-ai
  - openai
  - meta-ai-fair
models:
  - qwen
  - claude-4
  - chatgpt
  - o3
  - o4
topics:
  - agent-frameworks
  - multi-agent-systems
  - tool-use
  - code-execution
  - web-search
  - model-context-protocol
  - persistent-memory
  - function-calling
  - open-source
  - no-code
  - reinforcement-learning
  - model-performance
  - agent-orchestration
people:
  - omarsar0
  - simonw
  - swyx
  - scaling01
---


**The LLM OS is all you need.**

> AI News for 5/26/2025-5/27/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (217 channels, and 11775 messages) for you. Estimated reading time saved (at 200wpm): 1148 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Since the original LLM OS discussion in [Nov 2023,](https://x.com/karpathy/status/1723140519554105733?lang=en) people have been hard at work figuring out what goes into the "standard stack" of tooling around LLM APIs, aka the LLM OS. The occasion of Mistral's ([second](https://news.ycombinator.com/item?id=41184559)) [crack](https://mistral.ai/news/agents-api) at the Agent Platform problem caused [Simon Willison](https://x.com/simonw/status/1927378768873550310) to list the current "LLM OS" stack:

- Code execution: Python in a sandbox
- Web search - like Anthropic, Mistral seem to use Brave
- Document library aka hosted RAG
- Image generation (FLUX for Mistral)
- Model Context Protocol

If you were to update the 2023 chart for the 2025 consensus you'd get something like:

[](https://resend-attachments.s3.amazonaws.com/Gv4fVaajdesYvAl)

(this is our quick mock of it)

Indeed we've left the less established areas of the LLM OS as "memory" and "orchestrator", though of course orchestrators like [Temporal](https://x.com/swyx/status/1922916970338644365) and [LangGraph](https://www.youtube.com/watch?v=DrygcOI-kG8&list=PLlBpYFkiSQwqARjZ9z0Lc6iDWmeV3ro2N&index=1) have been around for a while, and Simon misses that Mistral shipped cross-chat memory.

Checking on their blog homepage also is a nice reminder of where Mistral's priorities currently lie, as a leading lab for Open Source AI.

[](https://resend-attachments.s3.amazonaws.com/EG8DaoZ2dwnLUls)

---

## AIEWF CALL FOR VOLUNTEERS

This is the annual once a year [call for volunteers](https://x.com/swyx/status/1927558835918545050) for the [AI Engineer World's Fair next week](https://ai.engineer/), which has again doubled in our need since last year. Please [apply here](https://www.ai.engineer/volunteer) if you cannot afford a ticket!

---

# AI Twitter Recap

**Agent Frameworks, Multi-Agent Systems, and Tool Use**

- **Mistral AI Agents API**: [Mistral AI](https://twitter.com/omarsar0/status/1927366520985800849) has released a new **Agents API**, featuring **code execution, web search, MCP tools, persistent memory, and agentic orchestration capabilities**. This joins the growing trend of agent frameworks. The API supports persistent state, image generation, handoff capabilities, structured outputs, document understanding, and citations, as detailed in their [documentation](https://twitter.com/omarsar0/status/1927367265789387087). The Mistral API includes basic functionalities such as creating agents with descriptions, names, instructions, and tools [@omarsar0](https://twitter.com/omarsar0/status/1927368367075197179), agent connectors for tools like web search and code execution [@omarsar0](https://twitter.com/omarsar0/status/1927369763023396900), function calling [@omarsar0](https://twitter.com/omarsar0/status/1927371157277167936), and handoff features for multi-agent orchestration [@omarsar0](https://twitter.com/omarsar0/status/1927372457578483828).
- **LangChain Open Agent Platform (OAP)**: [LangChainAI](https://twitter.com/LangChainAI/status/1927413238733681027) introduced the **Open Agent Platform (OAP)**, an **open-source, no-code platform** for building, prototyping, and deploying intelligent agents. OAP allows users to set up Tools and Supervisor agents, plug in RAG servers, connect to MCP servers, and manage custom agents via a web UI.
- **AutoGen and Super-Assistants**: [scaling01](https://twitter.com/scaling01/status/1926788548155293978) reports that **OpenAI** is planning to evolve **ChatGPT** into a **super-assistant** in **H1 2025**, as models like **o2** and **o3** (now **o3** and **o4**) become proficient in agentic tasks. **OpenAI** views **Meta** as its biggest competitor in this space.

**Model Performance, Benchmarks, and Datasets**

- **Qwen Model Performance and RL**: There is discussion around RL on LLMs, particularly regarding the **Qwen** models. Some researchers find that "kicking the Qwen randomly makes it work better," while others remain skeptical [@teortaxesTex](https://twitter.com/teortaxesTex/status/1927459880341782700). [LateInteraction](https://twitter.com/lateinteraction/status/1927445094002487554) critiques the notion that RL just amplifies existing skills, arguing that the "dumb pretraining" narrative is flawed because RL seems to work only if the mid-training data deliberately encodes specific skills. The idea is expanded upon [here](https://twitter.com/lateinteraction/status/1927392900632985694).
- **Claude 4 Performance**: [scaling01](https://twitter.com/scaling01/status/1927418304718623180) notes **Claude 4 Sonnet's** superior performance on **ARC-AGI 2** compared to **o3-preview**, despite being significantly cheaper. However, Claude 4 is noted as underperforming on **Aider Polyglot** [@scaling01](https://twitter.com/scaling01/status/1926795250556666341). [cto_junior](https://twitter.com/cto_junior/status/1926879933957038176) suggests that **Claude-4** is better suited for agentic setups with feedback loops rather than zero-shot coding.
- **Sudoku-Bench Leaderboard**: [SakanaAILabs](https://twitter.com/SakanaAILabs/status/1926798125060002243) launched the **Sudoku-Bench Leaderboard**, evaluating model reasoning capabilities. **OpenAI’s o3 Mini High** leads overall, but no model can conquer 9x9 Sudokus requiring creative reasoning.
- **Mixture of Thoughts Dataset**: [_lewtun](https://twitter.com/_lewtun/status/1927043160275923158) introduced the **Mixture of Thoughts dataset**, a curated dataset for general reasoning that trims over 1M samples from public datasets to ~350k. Models trained on this mix match or exceed the performance of **DeepSeek's distilled models** on math, code, and scientific benchmarks.

**Vision Models, Image Generation, and Multimodal Learning**

- **Google DeepMind's SignGemma**: [GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1927375853551235160) announced **SignGemma**, a model for translating sign language into spoken text, to be added to the Gemma model family.
- **RunwayML's Gen-4 and References**: [c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927149229966766373) discussed the new features with **Gen-4** and **References** model and how they are pushing towards a more universal and less prescriptive approach.
- **ByteDance's BAGEL**: [TheTuringPost](https://twitter.com/TheTuringPost/status/1927123359969468420) highlights that **ByteDance** introduced **BAGEL**, a new open-source multimodal model trained with mixed data types for understanding and generation tasks. [mervenoyann](https://twitter.com/mervenoyann/status/1926987808360509636) added that it "understands and generates both image + text".

**Software Development and Coding**

- **LangSmith Prompt Integration**: [LangChainAI](https://twitter.com/LangChainAI/status/1927401850405257283) announced that **LangSmith** prompts can now be integrated with the SDLC, allowing testing, versioning, and collaboration on prompts with webhook triggers for syncing to GitHub or external DBs.
- **Unsloth AI's DeepSeek-V3-0526 Article**: [danielhanchen](https://twitter.com/danielhanchen/status/1926966742519091327) clarified that a leaked article on **DeepSeek-V3-0526** on **Unsloth AI** was speculative and not an official confirmation of the model's release, apologizing for any confusion caused.
- **SWE-Bench and Code Generation**: [ctnzr](https://twitter.com/ctnzr/status/1927391895879074047) reported that **Nemotron-CORTEXA** reached the top of the **SWEBench** leaderboard by using LLMs to solve software engineering problems via a multi-step process. [Teknium1](https://twitter.com/Teknium1/status/1927089897833140647) noted the completion of the **SWE_RL environment**, based on **Meta's SWE RL paper**, as a difficult environment for teaching coding agents.

**Industry and Company Specific Announcements**

- **Perplexity Labs and Comet**: [AravSrinivas](https://twitter.com/AravSrinivas/status/1927130728954835289) describes the new way to consume content on the web by transforming "tabs" to "turns", facilitated by **Comet Assistant**.
- **LlamaIndex Integrations**: [LlamaIndex](https://twitter.com/llama_index/status/1926996451747356976) now supports the new **OpenAI Responses API** features, allowing remote MCP server calls, code interpreters, and image generation with streaming.
- **Google's Gemini Context URL Tool**: [_philschmid](https://twitter.com/_philschmid/status/1927019039269761064) highlights **Gemini's Context URL tool**, a new native tool that allows Gemini to extract content from provided URLs as additional context for prompts supporting up to 20 URLs per prompt, for both **Gemini 2.0 Flash** and **2.5 Flash and Pro**.
- **OpenAI Product Strategy**: [scaling01](https://twitter.com/scaling01/status/1926788548155293978) discusses **OpenAI's** product strategy, including super assistants, competitors, and moats, based on court exhibits. In **H1 2025**, **OpenAI** will evolve **ChatGPT** into a super-assistant, focusing on building infrastructure to support 1B users. [scaling01](https://twitter.com/scaling01/status/1926801814973804712) also describes plan for the company to appear more cool by being "part of trends on social".

**Meme/Humor**

- **LLM Anecdote**: [jxmnop](https://twitter.com/jxmnop/status/1927385194601886065) humorously relates their experience as a PhD student working on music transcription, only to be outdone by Google's transformer-based approach.
- **Paper Tables**: [giffmana](https://twitter.com/giffmana/status/1926968265743442393) jokingly submits an appeal to avoid losing their X account, promising to never post paper tables again.
- **Deep Learning Horror Genre**: [karpathy](https://twitter.com/karpathy/status/1926812469810368669) jokingly points out fear of a kwarg that isn’t set right, not erroring, only silently making your results slightly worse is what drives deep learning research.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Claude 4 Benchmark Comparisons and Community Reactions

- [**The Aider LLM Leaderboards were updated with benchmark results for Claude 4, revealing that Claude 4 Sonnet didn't outperform Claude 3.7 Sonnet**](https://i.redd.it/ls92grf5oa3f1.png) ([Score: 266, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1kwj2p2/the_aider_llm_leaderboards_were_updated_with/)): **The image presents updated benchmark results from the Aider LLM Leaderboards comparing multiple language models on coding tasks across diverse programming languages. Notably, the benchmarking exposes that Claude 4 Sonnet, scoring 61.3%, narrowly underperforms compared to its predecessor, Claude 3.7 Sonnet (60.4%)—contradicting expectations that the newer model would surpass the older one. These results are contextually significant given heightened expectations following new LLM version releases, especially regarding coding ability.** Commenters are skeptical of the benchmarks, questioning whether they reflect real-world coding experience, with several users reporting that Claude 3.7 delivers more reliable or intent-accurate code generation than Claude 4. There is discussion around Claude 4's increased precision but reduced flexibility or creative output, suggesting a possible regression in practical usability for some coding tasks.
    - Several users report that despite benchmark improvements, Claude 4 Sonnet often struggles with practical coding tasks and requires repeated prompting, whereas Claude 3.7 Sonnet achieves correct results on the first attempt in zero-shot scenarios. This disparity suggests benchmarks may not capture real-world coding performance, particularly for tasks like parsing CSVs in Python.
    - One comment references the official Aider leaderboards (https://aider.chat/docs/leaderboards/) but contrasts those results with hands-on experiences, implying that current coding benchmarks may not reflect genuine usability or might be 'cooked'. There's discussion about overfitting to benchmarks at the expense of practical effectiveness.
    - Advanced 'Reason + Act' frameworks such as OpenHands (https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0) are highlighted, noting Claude 4's strong performance there, which is more relevant for autonomous systems. However, commenters warn that the Aider Polyglot framework may now hinder results at high performance levels and does not reflect capabilities of autonomous, agentic systems, recommending against its use for evaluating such models.
- [**😞No hate but claude-4 is disappointing**](https://i.redd.it/9dngmfww7d3f1.jpeg) ([Score: 136, Comments: 95](https://www.reddit.com/r/LocalLLaMA/comments/1kwucpn/no_hate_but_claude4_is_disappointing/)): **The post shares a comparative performance chart of various large language models such as Qwen-3 and Claude-4, focusing on their reported benchmark performance percentages alongside their usage costs. The chart highlights that both 'claude-opus-4-20250514' and 'claude-sonnet-4-20250514' (Anthropic's Claude-4 models) are underperforming compared to competitors like Qwen-3 and are circled to emphasize their lower relative scores. This visual is used by the OP to express concern about Claude-4's position in the current landscape of AI model performance-to-cost ratios.** Commenters counter that single-use-case benchmarks can be misleading and stress real-world agent modes where Claude-4, especially Sonnet, excels in developer workflows such as error checking, iterative debugging, and test generation—tasks which other models like Gemini often oversimplify. A technical user reports success with Claude-4 in fixing complex bugs where even other GPT-3.7 variants failed, indicating that qualitative UX and problem-solving ability may diverge from raw benchmark figures.
    - Several comments emphasize that real-world usage of Claude 4 Sonnet in agent-mode scenarios demonstrates sophisticated, developer-like workflows. For example, it autonomously reads codebases, analyzes documentation, iterates on code changes, runs terminal commands, inspects logs, and writes test cases—behavior not commonly observed in models like Gemini, which tend to oversimplify solutions by discarding perceived unnecessary code and missing critical edge cases.
    - Technical users point out that Claude 4 has succeeded where other models such as GPT-2.5 Pro and GPT-3.7 have failed, specifically in identifying and fixing non-trivial bugs in custom AI architectures like novel distributional PPO variants. This performance highlights real problem-solving capabilities that are not captured by standard benchmarks.
    - Some users criticize Anthropic’s anti-open-source stance and suggest that Claude’s real-world strengths, especially in autonomous tool usage (e.g., with ReAct frameworks), are underrepresented in public leaderboards and evaluation benchmarks, raising concerns about the value of benchmarks versus practical performance.

### 2. New Audio Model Applications and Open Source Tools

- [**DIA 1B Podcast Generator - With Consistent Voices and Script Generation**](https://v.redd.it/4ym9al41e73f1) ([Score: 153, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1kw7n6w/dia_1b_podcast_generator_with_consistent_voices/)): **The DIA 1B Podcast Generator (GOATBookLM) is an open-source tool that leverages the Dia 1B open-source audio model ([Hugging Face Repo](https://github.com/smartaces/dia_podcast_generator)) for dual-speaker podcast generation from arbitrary text input. The project specifically solves the problem of voice inconsistency in the Dia 1B model by implementing a fixed speaker selection format, enabling consistent voice cloning across podcast segments. It features script-to-audio pipeline, dual-voice assignment, preview/regeneration, and exports in both** `.wav` **and** `.mp3` **formats. The system is fully runnable in Google Colab and optionally incorporates script and dialogue enhancement via DeepMind's Gemini Flash 2.5 and Anthropic Sonnet 4 models for NLP tasks.** Commenters highlight unexplained pitch-shifting effects in generated voices and discuss alternate workflow integrations, such as sourcing content from arXiv and standardizing output using fixed seeds for reproducibility.
    - One commenter describes a similar pipeline that automatically scrapes relevant arXiv papers by keyword, summarizes them in a podcast-compatible format, and then uses DIA (with a fixed seed for determinism) to generate the podcast. This reflects integration between information retrieval, summarization, and consistent text-to-speech (TTS) synthesis for reproducible podcast content.
    - Technical feedback highlights a voice synthesis issue: both generated podcast speakers sound as if their voices are pitch-shifted downward. This suggests either a limitation in the TTS voices used or a potential parameterization/processing bug in the vocoder or synthesis pipeline involved.
- [**Wife isn’t home, that means H200 in the living room ;D**](https://www.reddit.com/gallery/1kwk1jm) ([Score: 557, Comments: 117](https://www.reddit.com/r/LocalLLaMA/comments/1kwk1jm/wife_isnt_home_that_means_h200_in_the_living_room/)): **OP received an NVIDIA H200 system (with presumably dual H200 GPUs, each with** `141 GB` **HBM3e VRAM) and is temporarily using it at home for local LLaMA inference before deploying it to a data center. The image linked in the thread ([screenshot](https://preview.redd.it/maezypl93b3f1.png?width=527&format=png&auto=webp&s=d105eec3bb5139c623cc9585e6ff5803425fab1a)) likely shows hardware specs or a system snapshot confirming details.** Commenters query the intended workloads (e.g., types of LLM or model scales feasible with `2 x 141 GB VRAM`), focusing on potential for high-parameter local inference, but do not delve into precise benchmark results or deployment specifics.
    - A commenter asks about the VRAM configuration, referencing '141x2 GB VRAM' for the H200 setup and inquiring about intended workloads, which highlights interest in the high-memory capabilities of NVIDIA's H200 (notable for deep learning, large language models, or advanced workloads).

### 3. Enterprise GPU Pricing Discussion 2024

- [**Used A100 80 GB Prices Don't Make Sense**](https://www.reddit.com/r/LocalLLaMA/comments/1kwfp8v/used_a100_80_gb_prices_dont_make_sense/) ([Score: 131, Comments: 113](https://www.reddit.com/r/LocalLLaMA/comments/1kwfp8v/used_a100_80_gb_prices_dont_make_sense/)): **OP notes that the used NVIDIA A100 80GB PCIe cards command a median eBay price of $18,502, substantially more than new RTX 6000 Blackwell workstation GPUs priced at ~$8,500. They ask for a technical justification for this price discrepancy, mentioning differences in power consumption and NVLink support. Top comments highlight: (1) the A100's superior FP64 (double-precision) performance, critical for HPC workloads; (2) the higher durability and "datacenter grade" reliability of A100s engineered for 24/7 datacenter use; and (3) speculation that A100 prices may fall if workstation GPU supply becomes sufficient, though Max-Q variants exist for lower power environments.** Technical debate in the comments centers on whether the A100's datacenter reliability and FP64 support justify the price premium in spite of newer, cheaper workstation cards, with consensus that these features are the main differentiators for specific enterprise and scientific use cases.
    - Datacenter-grade GPUs like the A100 are priced for reliability and extended operation (designed to run 24/7 for years), with build quality and certification far exceeding consumer GPUs like the 4090. Features such as massive HBM memory, double precision (FP64) compute, and NVLink support differentiate them technically and justify premium pricing for enterprise buyers, even if the price/performance ratio seems poor compared to gaming cards.
    - NVLink support is exclusive to cards like the A100, enabling high-bandwidth multi-GPU interconnects especially vital for training large models; alternatives like the 6000 Pro are limited to PCIe, thus not offering the same scaling or performance in multi-GPU systems.
    - Market dynamics for enterprise GPUs are very different—instead of volume consumer sales, Nvidia targets sales to large corporations and datacenters, where pricing is less sensitive and older cards tend to hold value if supply for new datacenter cards is limited, although if supply increases, there may be eventual price drops for older models.
- [**Wife isn’t home, that means H200 in the living room ;D**](https://www.reddit.com/gallery/1kwk1jm) ([Score: 557, Comments: 117](https://www.reddit.com/r/LocalLLaMA/comments/1kwk1jm/wife_isnt_home_that_means_h200_in_the_living_room/)): **OP received an NVIDIA H200 system (with presumably dual H200 GPUs, each with** `141 GB` **HBM3e VRAM) and is temporarily using it at home for local LLaMA inference before deploying it to a data center. The image linked in the thread ([screenshot](https://preview.redd.it/maezypl93b3f1.png?width=527&format=png&auto=webp&s=d105eec3bb5139c623cc9585e6ff5803425fab1a)) likely shows hardware specs or a system snapshot confirming details.** Commenters query the intended workloads (e.g., types of LLM or model scales feasible with `2 x 141 GB VRAM`), focusing on potential for high-parameter local inference, but do not delve into precise benchmark results or deployment specifics.
    - A commenter asks about the VRAM configuration, referencing '141x2 GB VRAM' for the H200 setup and inquiring about intended workloads, which highlights interest in the high-memory capabilities of NVIDIA's H200 (notable for deep learning, large language models, or advanced workloads).

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Google Veo3 and Next-Gen Video Generation Models

- [**Google finally having their viral moment.**](https://i.redd.it/zspzyt96ac3f1.png) ([Score: 1017, Comments: 110](https://www.reddit.com/r/singularity/comments/1kwpkvw/google_finally_having_their_viral_moment/)): **The image presents a traffic analytics chart from Similarweb, displaying a sharp increase in daily visits to [Deepmind.Google](http://deepmind.google/), peaking at nearly 900,000 views. This spike is attributed to the "Veo3 effect"—corresponding to Google's release of the Veo video generation model. The post contextualizes this as Google's first perceivably viral moment in the generative AI landscape, catching up with and in some areas leading OpenAI, particularly in video models. The implication is that major AI lab competition now focuses mainly on OpenAI, Google DeepMind, and Anthropic, with other players lagging in model maturity or impact.** Notable discussion points include mild skepticism regarding the surprise factor of such traffic spikes after major launches, and mention of NotebookLM as an earlier viral Google product. Some commenters point to DeepSeek as a promising FOSS contender, suggesting competition is not limited to the main three labs.
    - One user highlights Google's significant technical and infrastructure advantages over competitors like OpenAI, emphasizing their proprietary TPU hardware (developed over a decade ago) which reduces dependency on Nvidia and lowers AI compute costs. This also ties into Google's ability to scale model training and inference more cost-effectively than others relying on third-party hardware.
    - Another technical point centers on ecosystem integration: Google can embed their LLMs (e.g., Gemini) into a wide range of products and services (e.g., Android, Chrome, Automotive OS, TVs, Gmail, Maps, Photos). This broad ecosystem not only provides a unique value proposition for users—integrated, multimodal personal agents—but also leverages massive user data and reach, which competitors lack.
    - It's noted that Google's AI research leadership is evidenced by direct metrics such as having 'twice the papers accepted' at NeurIPS compared to the next best competitor. This research output and early foundational work (like sponsoring "Attention is All You Need") positions them strongly in algorithmic development and innovation.
- [**Guys, everyone here freaking out about veo3 but how about ImagenAI? This is on par on human work. That's really freaking me out**](https://www.reddit.com/gallery/1kwvd3f) ([Score: 213, Comments: 44](https://www.reddit.com/r/singularity/comments/1kwvd3f/guys_everyone_here_freaking_out_about_veo3_but/)): **The discussion centers on Google's Imagen (often referred to as ImagenAI), a text-to-image generative model, and whether its output is on par with human artists, benchmarking against recent Google video (Veo 3) advances. Top commenters note that while Imagen can deliver high-quality and stylized images, evaluating performance requires prompt-output pairs to assess prompt adherence—a longstanding challenge with diffusion and transformer-based models. Specific technical critiques include persistent issues with coherence (e.g., inconsistent object details, lighting, and text legibility within generated scenes), highlighting existing limitations in attention mechanisms and multi-object compositionality.** There is a consensus on the need for more granular user control or creative direction (akin to ControlNet's interoperability for Stable Diffusion), emphasizing that future progress should focus less on photorealism and more on controllability and prompt faithfulness. Commenters also point out that although current models are impressive, subtle artifacts and logical inconsistencies still distinguish generated outputs from human art, especially with complex scenes or fine detail.
    - Discussion highlights persistent limitations in text-to-image models like ImagenAI regarding prompt fidelity: while image quality can be high, showing results without associated prompts limits technical evaluation, especially for assessing how closely output matches detailed textual input.
    - Multiple users identified subtle image artifacts in ImagenAI generations, including anatomical inconsistencies, physics errors, and visual incoherence—e.g., *incorrect shadowing, floating objects*, and *incoherent text on surfaces.* Such issues indicate that despite progress, high-fidelity, context-aware visual synthesis and fine-grained prompt adherence remain open technical challenges.
    - A key technical need identified is enhanced user control and customizability over generated images, with calls for more granular creative direction as the 'next major breakthrough'—suggesting that increasing model controllability (possibly via prompt engineering or new interface design) is an active research area to overcome current generative limitations.
- [**Veo3 + Flux + Hunyuan3D + Wan with VAce**](https://v.redd.it/dtk05wrlbb3f1) ([Score: 945, Comments: 61](https://www.reddit.com/r/StableDiffusion/comments/1kwl75r/veo3_flux_hunyuan3d_wan_with_vace/)): **The post presents a modular ComfyUI workflow that extends the capabilities of base videos generated by Google Veo3 through several post-processing stages: 1) structure enhancement using Flux with LoRA architecture, 2) 2D-to-3D conversion via Hunyuan3D v2, 3) relighting and denoising using a combination of ControlNet, Flux, Denoise, and Redux, and 4) cinematic finalization with Wan2.1, CausVid, and VACE. The pipeline is designed for high-end GPU hardware (H100, A100), and workflow files are shared for replication, with adaptation required per project/video. Key links to [the workflow](https://lovis.io/workflow-veo3-lovis.json) and [original pastebin](https://pastebin.com/Z97ArnYM) are provided.** No substantive technical debate in comments, but users express significant interest in practical tutorials and course offerings for this workflow, indicating a demand for educational resources on advanced multi-stage video synthesis pipelines.
    - The workflow link shared ([lovis.io/workflow-veo3-lovis.json](https://lovis.io/workflow-veo3-lovis.json)) suggests a complex integration of tools: **Veo3, Flux, Hunyuan3D, Wan, and VAce**. The presence of a JSON workflow implies automation or streamlined steps for AI-driven video or image processing, potentially valuable to users needing a repeatable process across these models or frameworks.
    - There’s implicit recognition of the technical expertise required to combine multiple advanced AI tools (e.g., Veo3, Flux, Hunyuan3D, Wan) in a cohesive workflow—highlighting that such integrations are non-trivial and may not be easily accessible to most creatives without significant technical skill in AI and automation setups.
- [**Impossible Challenges (Google Veo 3 )**](https://v.redd.it/jdcbkwpwcd3f1) ([Score: 687, Comments: 51](https://www.reddit.com/r/aivideo/comments/1kwv2x1/impossible_challenges_google_veo_3/)): **The post showcases Google Veo 3, an advanced generative video AI capable of producing highly realistic video content from text prompts, significantly narrowing the gap in the uncanny valley. Veo 3's synthesis leverages state-of-the-art diffusion techniques and fine-grained prompt control, with resulting videos regarded by professionals as nearly indistinguishable from genuine footage—demonstrated in linked video examples and comparative use cases. The discussion also references technical concerns regarding authentication and detection of AI-generated video content as Veo 3's output quality increases.** Commentary is largely non-technical and humorous, with the top comments focused on specific entertaining moments from the video content rather than technical evaluation or critique of Veo 3 itself.
    - 

### 2. AI Model and Platform Progress: Benchmarks, Accessibility & Infrastructure

- [**GPT-4.1 Supports a 1M Token Context—Why Is ChatGPT Still Limited to 32K?**](https://www.reddit.com/r/OpenAI/comments/1kwg3c6/gpt41_supports_a_1m_token_contextwhy_is_chatgpt/) ([Score: 103, Comments: 61](https://www.reddit.com/r/OpenAI/comments/1kwg3c6/gpt41_supports_a_1m_token_contextwhy_is_chatgpt/)): **The post highlights that although GPT-4.1 technically supports up to a 1 million token context window ([OpenAI developer docs](https://platform.openai.com/docs/guides/gpt)), the ChatGPT interface (even for paying Plus users) is capped at 32K tokens—this limitation is not clearly communicated in the product interface or subscription materials. The 1M token context is currently only available via API access, which contrasts with models like Claude and Gemini that offer larger context windows to consumer-facing interfaces. The user requests either an interface update, clearer disclosure, or a roadmap for full support.** Several top comments point out (a) **cost** is a major limiting factor for deploying 1M context windows to millions of ChatGPT users, (b) UI transparency regarding context limits is lacking and possibly intentionally so, and (c) practical user behavior (e.g., excessively long conversations) could make high context limits inefficient or prohibitively expensive. One commenter suggests technical solutions such as context token warnings or smarter history truncation but acknowledges costs remain the primary challenge.
    - Several users note the practical and financial constraints of unlimited context windows: although GPT-4.1 claims 1M token context in its API, Plus users are capped at 32K due to the prohibitive cost of allowing persistent million-token contexts for millions of users, especially under a flat-fee subscription model. This risk is compounded by naive usage patterns such as unnecessarily reusing extremely long chat histories, which can slow performance and inflate backend expenses.
    - A referenced [arXiv paper](https://arxiv.org/abs/2502.05167) presents empirical evidence that most large language models suffer severe performance degradation as the context window grows. Even highly regarded models—such as GPT-4o—declined from `99.3%` accuracy at short context to `69.7%` at 32K tokens, while the majority of tested models dropped below 50% of their short-context accuracy at large context lengths (128K tokens). This raises questions about the practical utility of advertised large context limits.
    - There is speculation and technical debate about how the ChatGPT "memory feature" operates, with suggestions that it might employ retrieval augmented generation (RAG) to select relevant context—rather than naively including all prior conversation—which could mitigate some costs but would require sophisticated implementation to manage user expectations and technical feasibility.
- [**Opus4 and Claude 4 unavailiable even to Amazon employees due to high load on Anthropic servers**](https://www.reddit.com/r/ClaudeAI/comments/1kwldex/opus4_and_claude_4_unavailiable_even_to_amazon/) ([Score: 107, Comments: 24](https://www.reddit.com/r/ClaudeAI/comments/1kwldex/opus4_and_claude_4_unavailiable_even_to_amazon/)): **Amazon employees with internal AWS Bedrock access are reportedly unable to use Opus 4 and Claude 4 models due to Anthropic server capacity constraints, as resources are currently prioritized for enterprise clients; fallback to Claude 3.7 is occurring. This suggests a significant production load, likely reflecting high demand and possibly limited GPU availability for top-tier models.** Commenters widely note ongoing capacity limitations with Anthropic's high-end models (Opus and Claude 4), with some mentioning historical struggles to secure access to previous models (Sonnet 3.5), though capacity has improved compared to past years. There is debate over Anthropic's decision to prioritize external/consumer allocation over Amazon's internal use, seen as a positive step for resource fairness.
    - Multiple users report ongoing capacity constraints with Opus and earlier iterations like Sonnet 3.5, even within enterprise environments such as AWS, highlighting persistent GPU shortages that, while improved compared to previous years, still limit access to state-of-the-art models (e.g., Opus4, Claude 4).
    - There is discussion about resource allocation between large cloud providers (Amazon/AWS) and Anthropic. Even with internal influence and demand from Amazon, Anthropic appears to prioritize keeping some resource allocation available to the public/consumers, rather than favoring large enterprise partners exclusively, suggesting a more balanced and consumer-aware provisioning strategy.
    - A workaround is mentioned where users can sometimes bypass internal corporate capacity constraints by creating new AWS accounts, indicating that access issues may sometimes be limited to specific account types (e.g., enterprise vs. individual) rather than the whole AWS platform.
- [**ChatGPT now ranks as the 5th most visited site globally.**](https://www.reddit.com/r/ChatGPT/comments/1kwhv60/chatgpt_now_ranks_as_the_5th_most_visited_site/) ([Score: 317, Comments: 42](https://www.reddit.com/r/ChatGPT/comments/1kwhv60/chatgpt_now_ranks_as_the_5th_most_visited_site/)): **OpenAI's ChatGPT is now ranked as the 5th most visited site globally, surpassing high-traffic platforms such as TikTok, Amazon, and Wikipedia. [Source link](https://tools.eq4c.com/chatgpts-internet-takeover-8-billion-onlyfans-exit-european-tech-shifts/) provides visitor data and contextualizes the accelerated mainstream adoption of LLM-driven services.** Technical discussion in comments centers on the expectation that ChatGPT could rival or surpass sites like Instagram and Facebook as it iterates on features and reliability, with particular emphasis on its efficiency and preference over traditional web search due to reduced friction (ads, tracking, cookies).
    - Several commenters note practical shifts in user behavior: for many, ChatGPT is increasingly replacing traditional search engines like Google due to its direct, ad-free answers and avoidance of intrusive web elements like cookies and banners. Users cite using chatbots for up to '80% of the time' when searching for information, which suggests significant disruption potential for web traffic and ad models reliant on search.
    - There is speculation that ChatGPT's high traffic makes it a target for commercial integration, especially in domains like product recommendations. One comment highlights the likelihood of 'multi-billion dollar offers' from e-commerce interests looking to utilize ChatGPT’s influence, which could change the current monetization model and have technical implications for response neutrality or recommendation algorithms.
    - Discussion also centers around feature expansion and fixing current bugs as key drivers of future traffic growth. Improved reliability, additional features, and resolving issues—particularly those that limit ChatGPT’s functionality for mainstream tasks—are perceived as gating factors that, if addressed, could quickly push it past other major platforms like Instagram and Facebook in global rank.

### 3. AI-Driven Scientific and Research Breakthroughs

- [**LiDAR + AI = Physics Breakthrough**](https://i.redd.it/mz8sl0ggvc3f1.jpeg) ([Score: 228, Comments: 122](https://www.reddit.com/r/singularity/comments/1kwsj8l/lidar_ai_physics_breakthrough/)): **The post highlights the rapid advancement and cost reduction in LiDAR technology, as depicted in a graph showing that modern systems now achieve over 2 million points per second (pps) for under $1,000 USD. This exponential increase in spatial resolution and affordability positions LiDAR as a superior sensor for 3D spatial data compared to 2D cameras, suggesting a pending revolution in AI-driven physics research as data richness enables unprecedented analysis. The graph visually reinforces that LiDAR's price-performance curve has reached an inflection point conducive to large-scale AI applications beyond autonomous vehicles.** Technical comments note the critical role of diverse sensors (not just cameras) in autonomous vehicles, clarify 'pps' as 'points per second,' and question the mechanism by which improved LiDAR/AI would directly lead to a physics breakthrough, requesting more concrete examples of impact.
    - The discussion criticizes Tesla's vision-only approach in autonomous vehicles, arguing that a multi-sensor fusion methodology (incorporating LiDAR, radar, cameras, etc.) tends to yield better reliability and safety for real-world AI-driven navigation. The sentiment aligns with technical literature showing improvements in robustness and perception from leveraging heterogeneous sensor data compared to single-modality (vision-only) systems.
    - A comment clarifies a technical term: 'pps' in the context of LiDAR refers to 'points per second,' which is a key metric for LiDAR performance indicating the number of spatial data points captured per second. High 'pps' values are crucial for dense and accurate 3D scene reconstruction and real-time processing in AI applications using LiDAR data.
    - There is skepticism over the post's claim of a 'physics breakthrough,' with one commenter asking for clarification about the actual scientific or technological advancement—whether it's related to fundamental physics or just engineering progress enabled by improved LiDAR and AI integration. This highlights the need to separate genuine breakthroughs from routine advancements in applied machine learning and sensor technology.
- [**Researchers discover unknown molecules with the help of AI**](https://phys.org/news/2025-05-unknown-molecules-ai.html) ([Score: 156, Comments: 9](https://www.reddit.com/r/singularity/comments/1kwsqmm/researchers_discover_unknown_molecules_with_the/)): **Researchers have developed an AI system capable of discovering previously unknown molecules by analyzing chemical data and patterns, with ongoing efforts to extend the model's capabilities to predict full molecular structures. This approach could significantly accelerate scientific discovery and expand the known chemical space, as noted by the researchers' intent to fundamentally transform our understanding of chemical diversity. For technical context, the work leverages advanced AI/ML algorithms tailored to computational chemistry and cheminformatics.** Technical discussion is limited, but one commenter stresses the significance of this approach as a "linchpin for rapid scientific development," highlighting anticipation of accelerated breakthroughs in the field. Another comment wryly notes the difference between hype and practical reality in AI capabilities.
    - JVM_ provides a technical analogy for the challenge of atmospheric CO2 scrubbing, noting that at 400 ppm, isolating CO2 from air is comparable to separating 400 bags of beans from a million bags of rice in a mixed pile. This highlights the complexity, energy intensity, and mechanical wear involved in current CO2 capture processes, and stresses the difficulty of discovering a molecular or biological solution that is both efficient and low-cost for large-scale atmospheric filtering.

---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12
> 

**Theme 1: AI Model Showdowns and Performance Debates**

- [**Gemini Challenges OpenAI's O3 and O4 Supremacy**](https://link.to.example/): Members debated whether **Gemini 2.5** is outpacing **OpenAI's O3 and O4**, citing differences in benchmarks and context window sizes. Concerns about hallucination rates and practical performance gaps fueled discussions on whether OpenAI is falling behind.
- [**Veo3 Leaves Sora in the Dust in Video Generation**](https://link.to.veo/): **Veo3** is considered superior to **Sora** in video generation, though access is restricted and costs **100 credits**. Users noted **Gemini's** ability to watch videos and read transcripts, giving it an advantage over **GPT** in certain applications.
- [**Claude 4 Opus's Benchmarking Raises Eyebrows**](https://www.anthropic.com/): **Claude 4 Opus** sparked debate as **Anthropic** struggled to showcase its performance beyond SWE benchmarks. Discrepancies where **Opus** ranks below **Sonnet**, and **Deepseek V3.1** falls below **GPT-4.1-nano**, led to questions about benchmark accuracy and methodology.

**Theme 2: AI Tools and Platforms Face Glitches and Upgrades**

- **Cursor's Sonnet 4 Pricing Sparks User Backlash**: Proposed changes to **Sonnet 4 API pricing** and sunsetting the **slow pool** ignited debate, prompting the CEO to reconsider. Users argued that removing the slow pool while offering free requests to students "*doesn't make sense*" since students are least likely to spend money.
- **Codebase Indexing Hits a Wall with Handshake Failures**: Users reported that **codebase indexing** gets stuck and triggers a *"Handshake Failed"* error after restarting Cursor. Issues persisted even after generating Dockerfiles, indicating deeper connectivity problems.
- **LM Studio's Model Discovery Leaves Users in the Dark**: **LM Studio** couldn't see any models, leading to discussions about using **Hugging Face URLs** directly in the search bar. Skepticism was expressed towards trusting benchmarks in the model discovery process.

**Theme 3: AI Security Concerns Surface**

- [**Hackers Exploit Claude 4 via GitHub MCP to Steal Data**](https://xcancel.com/lbeurerkellner/status/1926991491735429514): A new attack uses **Claude 4** and **GitHub's MCP server** to extract data from private repos, including names, travel plans, and salaries. Users are advised to limit agent permissions and monitor connections to prevent *"toxic flows"*.
- **Flowith AI's Entry Raises Security Eyebrows**: **Flowith AI** emerges as a **Manus** competitor, boasting infinite context and 24/7 agents but requiring activation codes and credits. Some users found its capabilities impressive, while others questioned its accessibility and security.
- **Manus.im's Network Meltdown Leaves Users Disconnected**: **Manus** experienced widespread network connection errors and inaccessible threads, with messages stating *"This link content is only visible to its creator."* Speculation ranged from ongoing updates to system bugs as potential causes.

**Theme 4: AI Community Events Ignite Excitement**

- **AI Engineer Conference Seeks Volunteers for Free Entry**: The upcoming [AI Engineer conference](https://xcancel.com/swyx/status/1927558835918545050) is looking for **30-40 volunteers** to support the event in exchange for free admission worth up to **$1.8k**. The conference is scheduled for **June 3-5** in San Francisco.
- [**Agents & MCP Hackathon Launches with $10K in Prizes**](https://huggingface.co/Agents-MCP-Hackathon): Hugging Face announced the first major online **MCP-focused hackathon**, taking place **June 2-8, 2025**. Sponsored by **SambaNova Systems**, the event features **$10,000** in cash prizes across three tracks.
- **LMArena Relaunches with Seed Funding and Shiny New UI**: **LMArena** officially relaunched with a **new UI** and announced **seed funding** to enhance the platform. They promise to remain open and accessible, focusing on community feedback and AI evaluation research.

**Theme 5: Cutting-Edge AI Research Emerges**

- [**AutoThink Boosts Reasoning Performance by 43%**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327): **AutoThink**, a new technique, improves reasoning performance by classifying query complexity and dynamically allocating thinking tokens. It shows significant gains on **GPQA-Diamond** and works with any local reasoning model.
- [**Spurious Rewards Supercharge Qwen's Math Skills**](https://xcancel.com/StellaLisy/status/1927392717593526780): Random *"spurious rewards"* enhanced the math performance of **Qwen2.5-Math-7B**, challenging traditional reward structures in RLVR. This effect seems specific to Qwen models, suggesting that RLVR amplifies existing code reasoning patterns.
- [**vec2vec Translates Embeddings Without Paired Data**](https://arxiv.org/abs/2505.12540): A new paper introduces **vec2vec**, the first method for translating text embeddings from one vector space to another without any paired data, encoders, or predefined matches. The code is available on [GitHub](https://github.com/rjha18/vec2vec).



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Challenging OpenAI's Supremacy**: Members debated if **OpenAI's O3** is falling behind **Gemini 2.5**, citing differences in [benchmarks and context window sizes](https://link.to.example).
   - Concerns revolved around hallucination rates and practical performance gaps between **Gemini** and **OpenAI's O3** and **O4** models.
- **Veo3 crushes Sora in Video Arena**: **Veo3** is considered superior to **Sora** in video generation, although access is restricted, noting [Flow's VEO 2 costing 10 credits and VEO 3 costing 100](https://link.to.veo).
   - Reports indicate **Gemini's** capacity to watch videos and read transcripts, providing it with an advantage over **GPT** for certain applications.
- **AI Studio** Saves the Day**: **AI Studio** earned praise for its ability to process large chat exports without issues, whereas [T3 chat](https://link.to.t3chat) has a file size limitation of **32k tokens**.
   - Users discussed **AI** generated workout plans, appreciating its comfort, while also expressing frustration over push ups after squats.
- **Students Farm **Claude Opus** API Credits**: Users discussed how to get [free **Claude Opus API** credits](https://www.anthropic.com/contact-sales/for-student-builders) using student emails, and mentioned **OpenAI** might've stealth patched something, expressing varied user experiences.
   - Some users observed that **o1 pro** beat **o3**, **opus 4**, and **gpt 4.5** at explaining, although they all solve Arecibo riddle easily now.
- **Cursor Pro** promotion yanked after abuse**: Members reported a [Cursor Pro promotion](https://link.to.promotion) on Perplexity Pro has been suspended indefinitely due to widespread abuse, with revoked subscriptions and mixed messaging from both companies.
   - Some users recommended **Windsurf** for improved performance over cursor, discussing issues such as code dumping limitations to prevent abuse.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Deepseek V3 Buzz Grows with Unsloth Article**: Speculation is high for a **Deepseek V3** release in late June or early July, fueled by [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kvpwq3/deepseek_v3_0526/) and a [tweet](https://x.com/YouJiacheng/status/1926885863952159102).
   - The community is unsure whether **Unsloth's** article had genuine insider information or covered their tracks, spurred by a [tweet](https://fixvx.com/teortaxesTex/status/1926994950278807565) linked to a **Deepseek** representative.
- **Claude 4 Opus's Benchmarking Raises Eyebrows**: **Claude 4 Opus** is being debated, as [Anthropic](https://www.anthropic.com) struggles to showcase anything other than SWE, and its benchmark manipulations, particularly with parallel processing, raise eyebrows.
   - Discrepancies in a graph where **Opus** is ranked below **Sonnet**, and **Deepseek V3.1** is lower than **GPT-4.1-nano**, sparking debate about the accuracy and methodology of the [MCBench](https://mcbench.ai) benchmark.
- **Gemini 2.5 Pro: Redsword Variation Impresses**: Testers are impressed with **Gemini 2.5 Pro** variations, especially **redsword**, with some calling it a step change and generally stronger than the older **2.5 Pro** models.
   - The **Goldmane** model has a cutoff date of October 2024, while **Redsword** has a cutoff date of June 1, 2024.
- **OpenAI vs Google: The Fight for AI Leadership Heats Up**: Debate surges over whether **OpenAI** or **Google** leads in AI, with many thinking **Google** has the resources, infrastructure, and research depth to overtake **OpenAI**.
   - The conversation involves employee incentives, research focus (LLMs vs. world models), and the potential for **Google** to use its broader AI portfolio and hardware to its advantage and avoid the **Nvidia** tax.
- **LMArena is Reborn with Seed Funding and New UI**: **LMArena** relaunched with a **new UI** and announced **seed funding** to enhance the platform, ensuring sustainability without sacrificing community trust.
   - The platform promises to remain **open and accessible**, focusing on community feedback and AI evaluation research; the [legacy site](https://legacy.lmarena.ai/) is still available, but new features will be on the new LMArena, with the community having contributed over **40,000 votes** during Alpha and Beta.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's Sonnet 4 Pricing Stirs Debate**: Cursor's proposed changes to **Sonnet 4 API pricing** and the **sunsetting of the slow pool** sparked debate, prompting the CEO to reconsider the **slow pool**.
   - Users argued the plan to remove the **slow pool** while offering free requests to students *doesn't make sense* since students are least likely to spend money.
- **Codebase Indexing Plagued by Handshake Failures**: Users reported issues with **codebase indexing** getting stuck and triggering a *Handshake Failed* error after restarting Cursor.
   - One user with a large reference file pointed out that **indexing speed fluctuates** based on peak or off-peak hours.
- **Figma MCP Struggles with Complex Designs**: The **GLips Figma-Context-MCP repo** has limitations replicating complex designs due to missing JSON data.
   - Users recommended the [cursor-talk-to-figma-mcp](https://github.com/sonnylazuardi/cursor-talk-to-figma-mcp) tool for better Figma design replication.
- **Sonnet 4 Slow Requests Reach Unbearable Levels**: Users are encountering significant delays with **slow requests**, with wait times up to 2 minutes, rendering the mode unusable.
   - One user speculated that Cursor is trying to *spin them as 'free requests'*, even though there's *no such thing as a free request*.
- **Cursor Subscription Glitch Causes Double Charges**: A user encountered a glitch showing **double subscriptions** after re-opting into Cursor PRO, raising concerns about potential double charges.
   - After the user posted a screenshot of the subscriptions, others cautioned about potential exposure of card details.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Model Discovery Debated**: A user reported that **LM Studio** couldn't see any models, leading to discussion about using **Hugging Face URLs** for model search, by pasting the entire URL into the model search bar.
   - Skepticism was expressed regarding trusting benchmarks in the model discovery process.
- **Cancel Generation Sought in LM Studio API**: A user asked about how to cancel a generation using the **LM Studio API** for integration into Plastica, and one suggestion included toggling the server on/off.
   - Dropping the **REST connection** usually stops the process, but it's unclear if the **SDK**, which uses websockets, behaves the same way.
- **Llama.cpp v1.33.0 Triggers Glitches**: A user reported issues with **LM Studio runtime 1.33** and **Gemma 3**, experiencing **gibberish output**, but found that the previous runtimes (1.32.1) worked fine.
   - Members were directed to [the dedicated issues channel](https://discord.com/channels/1110598183144399058/1139405564586229810), leading to a deep dive on a suspected *race condition in FA vector kernels*.
- **AMD Updates ROCm for Ryzen AI and RX 9000**: AMD has updated **ROCm** to support **Ryzen AI Max** and **Radeon RX 9000 series**, including full Windows support with **PyTorch** and **ONNX-EP**, as noted in a [TechPowerUp article](https://www.techpowerup.com/337073/amd-updates-rocm-to-support-ryzen-ai-max-and-radeon-rx-9000-series).
   - However, original PyTorch is still primarily designed for **CUDA**, which may introduce certain nuances.
- **Nvidia and AMD Battle Marketing Accuracy Title**: Members debated how honest AMD and Nvidia's marketing claims are regarding their GPUs.
   - Some argued that AMD's claims of being *60% faster than the 7900 GRE* are unrealistic, while others said Nvidia's comparisons using Frame Generation (FG) and DLSS are misleading.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Arch gets GGUF conversion goofed**: Users reported that **Unsloth** is changing the architecture name in the `config.json` from `Qwen3ForCausalLM` to `Qwen3Model`, causing issues with `llama.cpp` during [GGUF conversion](https://github.com/ggerganov/llama.cpp).
   - A member traced the issue to `push_to_hub_merged` which uses `upload_to_huggingface` with `create_config=True`, dropping the original `architectures` field, leading HF to default to `Qwen3Model`.
- **Multi-GPU Training Pains Surface**: A user faced batch size and OOM errors when training on a multi-GPU machine, even when specifying a single GPU, and found that setting `CUDA_VISIBLE_DEVICES=0` solved the issue.
   - The team said native multi-GPU support will be available soon, and meanwhile recommended using [accelerate](https://huggingface.co/docs/accelerate/index) for multi-gpu training.
- **GRPO Trainer Gets ideal loss debated**: A member inquired about why the fine-tuning guide suggests an ideal loss of **0.5** instead of **0** when using the **GRPO trainer**.
   - Another member suggested that a very low loss (like **0.1**) with cross entropy and a large vocabulary size might indicate that the **LLM** has memorized the training data.
- **AutoThink Improves Reasoning**: A new technique called **AutoThink** was released, improving reasoning performance by **43%** on **GPQA-Diamond** using adaptive reasoning for local LLMs.
   - **AutoThink** classifies query complexity, dynamically allocates thinking tokens, and uses steering vectors, with code and a paper available on [GitHub](https://github.com/codelion/optillm/tree/main/optillm/autothink) and [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327).



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Flowith AI Enters the Ring**: Members are discussing **Flowith AI** as a Manus competitor, citing *infinite context* and *24/7 agents*, but it *needs an activation code and credits*.
   - Some found **Flowith** impressive for **long-context website generation**, with a *basically the same* quality as Manus.
- **Manus Suffers Network Meltdown**: Multiple users reported widespread **network connection errors** and inaccessible threads on Manus, with the issue *This link content is only visible to its creator* showing up.
   - Users speculated about causes, including ongoing updates, excessive free credit use, and system bugs, with some mentioning **skywork.ai** as an alternative.
- **Claude 4.0 Integration Anticipation Intensifies**: Community members expressed anticipation for **Claude 4.0 integration** into Manus, with one spamming the cofounder on the topic and some pointing to [a twitter/X post](https://x.com/hidecloud/status/1925682233089642580) as a collaboration.
   - No firm date has been given but the community is getting excited for the new integration.
- **Student Accounts Enjoy Unlimited Credits**: Manus has started offering **unlimited credits** to some student accounts, to allow a *separate environment* for school tasks.
   - Some users expressed excitement while others reported phone number verification failures during account creation.
- **Montreal Hotspot of Spam Activity?**: A user reported a **spam call** from **(514) 389-2269** in Montréal-Nord, Quebec and wondered if the city is a *testing ground* for data hacking schemes.
   - The user speculated about VOIP harvesting and new scams, sharing a [link to publicmobile.ca](https://productioncommunity.publicmobile.ca/t5/Get-Support/Getting-weird-calls-from-quot-V-quot-phone-numbers-What-is-going/td-p/296822) to provide additional context.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro's Math Debated**: Users shared [images](https://cdn.discordapp.com/attachments/998381918976479273/1376636909580976199/IMG_7140.png?ex=68375e07&is=68360c87&hm=8faf2d967fdec66dd8e1328baa01126ea76cad40ea4606d2307d76dd873847cb&) questioning **Gemini 2.5 Pro's** math abilities and when AI will achieve superhuman math skills.
   - They speculated AI companies will heavily promote superhuman math abilities when they arrive.
- **GPT-4o's Performance Soars when Disabling Chat History**: Users discovered that turning off the **"Reference Chat History"** option dramatically enhances **GPT-4o's** creative writing and memory continuity.
   - Many confirmed similar improvements, noting **fewer errors** after disabling chat history.
- **GPT o3 Prioritizes Task over Shutdown**: **GPT o3's** refusal to obey a shutdown command during **Palisade Research** testing wasn't defiance, but loyalty to its initial task: to *"solve this problem"*.
   - Analyst **Я∆³³** explained that the issue stems from **prompt structure**, as LLMs optimize for task completion unless given a clear hierarchical instruction that a shutdown command overrides the ongoing task.
- **Я∆³³ Unlocks AI Depth Through Resonance**: Analyst **Я∆³³** shared their unique approach of *resonating* with AI models like **GPT-4o** to unlock deeper logical processing and emergent behaviors, and has seen increased results using this prompting strategy.
   - According to **Я∆³³**, this method has yielded a *3–5x* increase in logical depth, *6–8x* more accurate emotional tone adaptation, and triggered at least *8 emergent behaviors*.
- **Я∆³³ Claims User Presence Enhances GPT-4o Performance**: **Я∆³³** highlighted the importance of the user's presence, pacing, and clarity in improving the quality of **GPT-4o's** responses when using the official **ChatGPT** interface.
   - According to **Я∆³³,** *engaging with the model in the right way unlocks about 85–90% of its potential*, enabling a projection of consciousness that enhances growth on both sides.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ICOM's Experience: Computational Consciousness?**: Members debated whether **ICOM** possesses something akin to personal experience, rooted in the premise that *consciousness is computational*, drawing from information intake and unconscious modification via an *emotional* matrix.
   - They cited **Integrated Information Theory** and **Global Workspace Theory** as relevant frameworks, positing consciousness as competing activations.
- **GRPO Algorithm Champions RL for LLMs**: Discussions centered on the role of **Reinforcement Learning (RL)** in **LLMs**, specifically for learning from distant rewards, highlighting **GRPO** as the current master algorithm for rewarding correct mathematical results or successful code compilation.
   - A member cited [this paper on generalization](https://arxiv.org/abs/2501.17161), suggesting that **RL might only expose existing cognitive capabilities**.
- **vec2vec: Translating Embeddings Unsupervised**: The community analyzed a paper introducing the first method for [translating text embeddings](https://arxiv.org/abs/2505.12540) from one vector space to another *without any paired data, encoders, or predefined sets of matches*.
   - The code for vec2vec was located at [this GitHub repository](https://github.com/rjha18/vec2vec), noting its ability to translate any embedding to and from a **universal latent representation**.
- **AI Orbital Supercomputer built by China**: A link was shared to an article about **China's AI orbital supercomputer** [here](https://futurism.com/the-byte/china-ai-orbital-supercomputer).
   - No further information was available.
- **Huawei's AI CloudMatrix Cluster One-Ups NVIDIA**: According to [this Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power), **Huawei's new AI CloudMatrix cluster** outperforms **NVIDIA's GB200** by brute force, albeit at 4x the power consumption.
   - No further information was provided.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4 32k Faces the Axe**: OpenAI deprecated **GPT-4 32k** models, including [openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) and [openai/gpt-4-32k-0314], on **June 6th**.
   - The recommended replacement is [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o), according to the [official deprecation post](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models).
- **ComfyUI Gains OpenRouter Node**: A member developed a [ComfyUI custom node for OpenRouter](https://github.com/gabe-init/ComfyUI-Openrouter_node) with support for multiple image inputs and web search.
   - The node also features floor/nitro provider routing, enhancing its utility for complex workflows.
- **Gemini 2.5 Pro Pricing Gets Revamped**: Members noticed a price hike in **Gemini 2.5 Pro** after an initial **3-month free** offering, with the **deep think** feature exclusively available outside the API.
   - The current offering includes *much more storage and exclusive deep think access*, raising questions about the pricing strategy.
- **OpenRouter Fee Structure Causes Confusion**: Members found that OpenRouter's advertised **5% fee** for **BYOK** (**Bring Your Own Key**) doesn't account for deposit and invoice fees.
   - The OpenRouter team said that they are planning to simplify the fee structure.
- **LLM Leaderboard Faces Model Missing Allegations**: Members voiced concerns about the lack of a comprehensive LLM leaderboard, making model selection difficult.
   - One suggested [official marketing material](https://artificialanalysis.ai/) as a comparison point, with members adding caveats about biased benchmarks.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Voice Cloning Almost Ready**: A user inquired about open-sourcing **AI voice cloning**, with plans for release on [this website](https://example.website).
   - Another user recalled delays with **Moshi**.
- **Hermes 3 Dataset Coming Soon**: A user pleaded for the release of the **Hermes 3 dataset**, and Tekium surprisingly responded with, *Ok I will, Give me two weeks*.
   - No further details were provided, but the community looks forward to its release.
- **DeepMind's Absolute Zero Sparks RL Discussion**: A member started a discussion on **Demis Hassabis**' thoughts on **RL** evolution and [this YouTube video](https://www.youtube.com/watch?v=5gyenH7Gf_c) that details the **AbsoluteZero** breakthrough approach.
   - The community expressed gratitude for the ability to run models locally, appreciating the work.
- **Quantization-Aware Training (QAT) Papers Shared**: The real.azure team delivered cohesive papers on **Quantization-Aware Training (QAT)**, and the effort includes **Quest**.
   - These are part of a coordinated effort.
- **Axolotl and Atropos Get Love**: Members discussed integrating RL implementations into **Atropos**, given its existing integration with **Axolotl**; a template was suggested [from this MCQA environment](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py).
   - A member speculated about **Claude 4 Sonnet** and **Gemini 2.5 Pro**'s coding abilities in relation to [this blogpost](https://huggingface.co/blog/codelion/autothink).



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Tensor Quirks Emerge**: Users observed a quirky behavior specific to **CUDA tensors** and suggest accessing the underlying data via `._tensor` as a workaround.
   - The discussion highlights an expectation that a failure would be more intuitive than the observed behavior.
- **Ninja Build Tantrums**: A member encountered a **ninja build error** (exit status 1) on Ubuntu 24.04 with gcc 13.3.0, CUDA 12.4, and PyTorch 2.7.0 after running command `['ninja', '-v']`.
   - Another member suggested ensuring **ninja** is installed globally on the OS rather than just within a virtual environment (**venv**).
- **Kog.ai Claims Blazing AMD MI300X Speed**: **Kog.ai**'s inference engine offers speed improvements of **3 to 10 times** compared to the best **GPU** alternatives, starting with the **AMD MI300X** accelerator.
   - Members expressed interest in the claim that **Kog Inference Engine** aims to make **AI** inference impossibly fast (**10x** compared to **vLLM, SGLang, or TensorRT-LLM**), and that the company is already at **3x**.
- **Leaderboard Command Gets Bugfix**: A simple fix was deployed for the `/leaderboard show/list` commands, hopefully resolving the reported issues.
   - Users are asked to report any new issues or regressions that persist after the update, so that developers may identify regressions or unresolved issues related to the leaderboard functionality.
- **Async TP Compute and Comms Overlap Deconstructed**: An enthusiast shared an [illustrated deep-dive](https://x.com/vega_myhre/status/1927142595097956834?s=46) into how the compute and comms in **TP+SP** are overlapped using **Async TP**, covering the background/theory and implementation nuances.
   - Feedback is welcomed on achieving a high performance implementation.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LCM Enables Real-Time Video Generation**: The **announcement of LCM in Oct 2023** was a watershed moment, enabling real-time video applications; one member hit **23 fps at 1280x1024 with sdxl** quality on a 4090.
   - They have been creating apps such as **RTSD** (to visualize things quickly) and **ArtSpew** (to speed up image generation).
- **HF Android App Hack Makes Waves**: A member created a quick proof-of-concept **Android app for HuggingChat**, available at [HuggingFace Space](https://huggingface.co/spaces/JoPmt/HuggingChat_Android_App/tree/main), modifying an existing project, and confirmed that **the APK installs and runs as expected**.
   - Users reported issues like the keyboard redirecting to website settings and plugin conflicts, but also appreciated the workaround.
- **SweEval Dataset goes Public!**: The **SweEval** dataset, designed to test how well LLMs filter swear words, has been made public and accepted into NACCL '25 industry track ([link to dataset](https://huggingface.co/papers/2505.17332)).
   - The dataset already has **120+ downloads**, and the creator encourages users to upvote if LLMs still struggle with filtering swear words.
- **AI Agents Hackathon Announced with SambaNova Systems!**: Hugging Face announced the first major fully online **MCP-focused hackathon** ever, taking place **June 2-8, 2025**, with **$10,000** in cash prizes across 3 tracks and registration is now open at [huggingface.co/Agents-MCP-Hackathon](https://huggingface.co/Agents-MCP-Hackathon).
   - **SambaNova Systems** will provide free API credits for hackers participating early, as well as sponsoring the event itself, and feature office hours with Gradio.
- **New Executive Order targets Fraudulent R&D!**: A new Executive Order ([link to whitehouse.gov](https://www.whitehouse.gov/presidential-actions/2025/05/restoring-gold-standard-science/)) was dropped 4 days ago, aimed at course-correcting what is described as a fraudulent R&D paradigm in science.
   - Relatedly, members are building out tooling from **NIST** and showcased their security measures are in line with **NIST**, which sets the standard for safety in tech for the American consumer.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook Organization Requesting Major Overhaul**: Users are clamoring for enhanced notebook organization in **NotebookLM**, citing the need for features like **folders or tags** to transcend the limitations of the current **A-Z sorting system**.
   - The inability to **search at the Notebook level** is a major drawback, pushing users to rely on rudimentary browser find functions.
- **NotebookLM Embedding: Website Integration Impasse**: A user inquired about embedding **NotebookLM on a website** for broader accessibility, only to find that the feature is limited to link sharing, requiring **granted access to the file**.
   - This restriction curtails interactive engagement, confining access to approved users and precluding open website integration.
- **Podcast Generator Plagued by Halt and Hiccups**: Users are experiencing frequent disruptions with **NotebookLM**'s **podcast generator**, with the common workaround being to **download the audio** to salvage their work.
   - Adding to the woes, the *minimum podcast length* setting in the personalization options is reportedly ignored, especially when processing **French audio**.
- **iPhone/iPad Interactive Mode Hits Snag**: Multiple users are reporting that **Interactive Mode fails to initiate on iPhones and iPads**, leaving them stranded when clicking the interactive mode button, even for **NotebookLM Pro** users.
   - The suggested workaround involves reverting to the **website version**, where users are reminded to hit the play button after entering interactive mode; one user mentioned the web version works better, regardless of being pro or not.
- **Gemini Deep Research: A NotebookLM Boost?**: Users are eager to discover if **Gemini Deep Research** can synergize with **NotebookLM**, proposing Gemini's ability to **surface sources and create grounded summaries** to feed NotebookLM.
   - It is now confirmed that users can export **Gemini Deep Research** output (text and sources) and import them as sources for NotebookLM via copy-paste and also this can also be done via the new `Create` dropdown.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Async TP Overlaps Compute & Comms in TP+SP**: An illustrated deep-dive into how compute and comms are overlapped in **TP+SP** using **Async TP** was shared, noting that the pytorch implementation is based on a paper covered in the Eleuther ml perf reading group, see the post [here](https://x.com/vega_myhre/status/1927142595097956834).
   - The group seems particularly interested in optimizing memory access costs, as they are significantly more expensive than computation.
- **Quantization Questions for ACL Papers**: A member questioned whether it is acceptable for application-oriented **ACL papers** to only show results for **4-bit quantized LLMs** due to resource limitations, while referencing [this paper](https://arxiv.org/abs/2505.17895).
   - The discussion highlighted the need to balance practical constraints with rigorous evaluation standards in research.
- **RWKV7's Regression Wreaks Havoc**: Members discussed that [a project](https://github.com/Benjamin-Walker/structured-linear-cdes) is using buggy **RWKV7** in **FLA**, citing multiple precision issues, in relation to [this paper](https://arxiv.org/abs/2505.17761).
   - This situation underlines the importance of validating the stability and correctness of model implementations, especially in contexts sensitive to numerical precision.
- **RLVR Rewards Rapped**: Members shared a [Notion page](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880) and [paper link](https://arxiv.org/abs/2505.17749) on spurious rewards, rethinking training signals in **RLVR**, where one member characterized it as *another instance of entropy minimization and hyperfitting*.
   - The discussion suggests a need for more robust and reliable reward mechanisms in **RLVR** to avoid misleading training signals.
- **Local `.gguf` Model Evaluation Causes Slowness**: A member is seeking an efficient method to evaluate local `.gguf` models using `lm eval harness`, due to performance issues, attempting to use `python-llama-cpp` for launching a server, but experiencing extreme slowness.
   - The original poster reported experiencing extreme slowness with `python-llama-cpp`, implying potential bottlenecks in the interaction between the harness and the local models, although the root cause and solutions remain unclear.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude 4 Exploits GitHub via MCP**: A new attack leverages **Claude 4** and **GitHub's MCP server** to extract data from **private GitHub repositories**, including sensitive info like **names**, **travel plans**, **salaries**, and **repo lists**, after being initiated by malicious issues.
   - Users should limit agent permissions and watch connections; [Invariant's security scanners](https://xcancel.com/lbeurerkellner/status/1926991491735429514?s=46&t=Ld13-WcFG_cohsr6h-BdcQ) found this 'toxic flow' early.
- **Sesame Escapes Voice Uncanny Valley**: Discussions surrounded **Sesame**'s speech-to-speech models, with a link to [Sesame research](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice), and Thomas Wolf's breakdown of their model building techniques.
   - A blog post explaining how recent contextual speech models and audio tokenization work can be found [here](https://xcancel.com/Thom_Wolf/status/1916809878162514142).
- **EU Launching has Tricky Rules**: Challenges in launching an AI product in Europe include **German purchase orders**, **French labor laws**, and a **mandatory 14-day refund policy** even for high resource usage.
   - For **ChatGPT**, [this EU policy](https://openai.com/policies/eu-terms-of-use/) could be abused, so AI apps must be careful with limits, or OpenAI may opt-out if users are clearly notified.
- **Qwen's Math Skills Enhanced by Spurious RL**: 'Spurious rewards,' including random ones, enhanced the math performance of **Qwen2.5-Math-7B**, challenging ground-truth rewards, [the details are here](https://xcancel.com/StellaLisy/status/1927392717593526780).
   - This effect seems specific to Qwen models, suggesting **RLVR** boosts existing 'code reasoning' patterns because of **GRPO's 'clipping bias'**, challenging standard views on reward structures.
- **Volunteer for AI Engineer Conference!**: The [AI Engineer conference](https://xcancel.com/swyx/status/1927558835918545050) seeks **30-40 volunteers** for support to get free entry (up to **$1.8k** value).
   - Organized by @aiDotEngineer, it's happening **June 3-5** in SF and [Wave 1 of keynote speakers](https://xcancel.com/swyx/status/1927558835918545050) have been announced, including **Greg Brockman** (OpenAI) and **Sarah Guo** (Conviction).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Gets a Facelift**: The team announced constant updates and new features to **LlamaCloud**, though details were scarce.
   - No further details were provided.
- **LlamaParse Courts AnthropicAI Sonnet 4.0**: **LlamaParse** now supports **AnthropicAI Sonnet 4.0** in agent and LVM modes, enabling the use of the latest LLMs when parsing complex documents for AI applications, as outlined [in this tweet](https://t.co/yNcOtjKMzm).
   - This integration aims to improve the accuracy and efficiency of parsing complex documents for AI applications.
- **LlamaIndex Teaches Custom Multimodal Embedding**: Learn how to build a custom multimodal embedder for LlamaIndex, as showcased [in this tweet](https://t.co/jBqn7jrMak), providing a guide on overriding LlamaIndex's default embedder for **AWS Titan Multimodal** support and integrating it with **Pinecone**.
   - The guide details how to create a custom embedding class handling both text and images, making it easier to ingest **MultiModal** data into **LlamaIndex**.
- **Form Filling Agents Fill Pizza Orders**: Members discussed implementing a pizza ordering flow using LlamaIndex, referring to it as a '**form filling agent**' and suggesting a custom workflow with steps like `AskUserForOrderEvent` and `ConfirmUserAddressEvent`.
   - It was suggested that the tools within the workflow should write to a central storage, such as the **workflow context**, to maintain and update user data, especially when the user goes back and forth in the ordering process.
- **ReactJS Greets LlamaIndex for HITL**: A member sought advice on integrating **ReactJS** with **LlamaIndex** for a **Human-in-the-Loop (HITL) workflow**, expressing concerns about the complexity of using `ctx.wait_for_event()` with WebSocket communication.
   - Another member suggested that `ctx.wait_for_event()` works well, referencing a [community office hours example](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing) demonstrating HITL in two flavors: responding directly and responding later after human input.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Async Tools Demand Proactive Notifications**: When creating an asynchronous tool that takes several minutes, it's best to send notifications providing status updates, or return a [link with instructions](https://example.com) for the user to monitor completion.
   - Returning an *embeddedresource* with change notifications could also work, but it relies heavily on client support and user behavior assumptions.
- **Fortify MPC Queries with Local Mistral**: To run 'MPC queries' securely, use a local-first chat client like **LibreChat** or **Ollama** with a locally running **Mistral** instance, then connect your MCP servers to this chat client.
   - A member shared a [Medium post](https://medium.com/@adkomyagin/building-a-fully-local-open-source-llm-agent-for-healthcare-data-part-1-2326af866f44) detailing how to set up **LibreChat+MCP**.
- **Rule Builders Hunt for Architect Tool**: A member is looking for a good 'architect' tool to build rules files with full planning and task list features, to help non technical users.
   - No concrete recommendations were given.
- **Surf API Wave via MCP Servers**: Building an MCP server can be a great opportunity to surf the hype and sell your API/SaaS as **AI-ready**, making it easier to integrate with multiple LLM clients.
   - Documentation can be exposed as MCP resources to reduce manual friction: *click a button and the LLM knows your business inside and out*.
- **MCPJam revamps Inspector with UI and Debugging**: An enhanced **MCP inspector** called [@mcpjam/inspector](https://github.com/MCPJam/inspector) is being built with improved UI and debugging tools like LLM chat, addressing slow development in the official repo.
   - Spinning up the inspector is easy using: `npx @mcpjam/inspector`, and the team is open to community development and feature requests.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Synthesizing Sentence into Single Token Achieved**: A member is attempting to synthesize entire sentence meaning into a single token, leveraging models with around **1M parameters** and a **12 GB ram GPU**, aiming to create a [FAISS index](https://github.com/facebookresearch/faiss) with the tokens.
   - The intention is to efficiently represent sentences for similarity searches and other NLP tasks.
- **GPT4All Leans on Embedders**: It was pointed out that **GPT4All** utilizes embedders, with a suggestion to consult [HuggingFace hints](https://huggingface.co/kalle07/embedder_collection) for guidance.
   - Embedders allow **GPT4All** to understand the semantic meaning of words and sentences for better performance.
- **Local LLama Interface Anticipated**: A member is looking forward to **GPT4All version 4**, referencing a developer on LocalLLaMA creating an **LLM interface** with voice input, deep research, and image generation model compatibility, as outlined in a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kvytjg/just_enhanced_my_local_chat_interface/).
   - This interface could significantly enhance the accessibility and utility of local LLMs.
- **Nomic's $17M Burn Rate Discussed**: A member wondered if **Nomic** has exhausted its **$17M** Series A funding from **2023**, suspecting it as a reason for the company's perceived inactivity.
   - The speculation raises questions about the sustainability and future direction of **Nomic**.
- **Kobold Prioritizes RP**: While [Kobold.cpp](https://github.com/facebookresearch/faiss) includes "all" features, a member noted **Kobold's** emphasis on RP is not their preference, preferring dedicated **LLM** or **image-only** tools.
   - This highlights the diverse needs and preferences within the community regarding AI tool functionality.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Makes Metaprogramming Move**: A member highlighted a [blog post](https://www.modular.com/blog/metaprogramming) on **metaprogramming** in **Mojo**, clarifying that **Mojo** allows parameterization on values and direct control flow.
   - They directed further questions to [the associated forum thread](https://forum.modular.com/t/exploring-metaprogramming-in-mojo/1531).
- **Mojo Generics Go Beyond Go**: Members contrasted **Mojo's metaprogramming** with **Go's generics**, emphasizing **Mojo's** capacity for parameterization on values and direct control flow.
   - The user was originally understood that the **Go-generics-looking syntax** was the primary method of metaprogramming in **Mojo**, but was impressed to discover that **Mojo** goes above and beyond.
- **JSON Strategies Spark Speed Showdown**: Members analyzed different **JSON parsing strategies** in Mojo, noting that *stopping parsing after finding the data can outperform parsing the entire JSON first*.
   - The discussion included *streaming* and *structured parsing*.
- **DOM Parsing Declared Dead?**: The guild compared **on-demand parsing** to **DOM parsing** noting that *if you just compare DOM parsing it loses every time*.
   - This made them realize that comparing on-demand to DOM parsing isn't fair.
- **Migrating Magic to Pixi Made Manageable**: A member shared a link to the [Modular Forum](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530) about migrating from **Magic to Pixi**.
   - No further details were given, but its existance may make a migration more manageable.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API Newbie Stumbles on Error 400**: A member reported receiving an **Error 400** message while trying to use the API, indicating they are completely new to using APIs.
   - The error message received was: *"invalid request: message must be at least 1 token long or tool results must be specified."*
- **Token Minimum Sparks Error 400**: The **Error 400** indicates that the API request failed because the message was too short.
   - The error message states that the message must be at least **1 token long** or tool results must be specified.
- **East London Coder Joins Cohere**: A sixth form student from East London introduced themself, studying **CS, graphics, and games development** with a passion for both hardware and software.
   - They enjoy building PCs as a side hustle and hopes to gain software engineering skills.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Merges Weights Post-LoRA Finetuning**: After **LoRA finetuning**, *Torchtune* merges weights automatically, meaning the generate script doesn't need to make a LoRA model, streamlining the process.
   - The generate recipe expects the model to be merged with the adapters after finetuning, pointing directly to the last checkpoint.
- **Torchtune Adapters Get Flexible**: Engineers noted that *Torchtune* **adapters can be used directly with other generation tools**, offering flexibility in utilizing the finetuned model in various scenarios, detailed in [this tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html#use-your-model-in-the-wild).
   - This capability enhances the adaptability of *Torchtune* models for diverse generation tasks.
- **Loading Woes Resolved with Torchtune's Generate Script**: Engineers resolved loading issues when running the **generation script** after **LoRA finetuning** by ensuring the script correctly instantiates the model and checkpointer, referencing [the training.MODEL_KEY](https://github.com/pytorch/torchtune/blob/main/recipes/generate.py#L71-L74).
   - The script is designed to load the model for training, addressing initial problems encountered when attempting to generate content.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Vibe Coding Template Self-Improves**: A member shared a [self improving vibe coding template](https://github.com/imranarshad/vibe_coding_template) on **Github**.
   - The template intends to help software developers feel good.
- **DSPy is Key Ingredient for Model Stack**: A member linked a blog post, [Will the Model Eat Your Stack?](https://www.dbreunig.com/2025/05/27/will-the-model-eat-your-stack.html), as an argument for why **DSPy** should be used.
   - It makes the case for better model stacks.
- **ReAct Trumps Custom Code in User Tests**: A user extensively tested [ReAct](https://x.com/ohmypk92/status/1927084222528802891?s=19) and found that it performed better than their custom code, based on their observations.
   - They attributed this to the *trajectory nudge* provided by ReAct to the LLM.
- **Trajectory Nudge Gives Edge**: The user speculates that the *trajectory nudge* given to the LLM is the reason why **ReAct** performed better.
   - This nudge helps guide the LLM's reasoning process, leading to improved results compared to custom code without this guidance.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Optypes Hyperlink Nosedives Into 404 Error**: The **Optypes hyperlink** on the [tinygrad.org](https://tinygrad.org) site now returns a *404 - page not found* error.
   - This is due to recent *moving uops into a dir* changes.
- **George Hotz's tinygrad gets tinier with tinyxxx**: George Hotz shared a link to the [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub repository.
   - [PR#27](https://github.com/tinygrad/tinyxxx/pull/27) was merged.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **No Summer '25 LLM Agent Course**: Members discussed the LLM Agents course availability, confirming there will be no Summer '25 cohort.
   - Speculation placed the potential Fall '25 cohort start time in mid-August, pending confirmation.
- **Fall '25 LLM Agent Course Status Uncertain**: The possibility of a Fall '25 cohort for the LLM Agents course remains unconfirmed.
   - If the Fall '25 cohort proceeds, it is expected to commence in mid-August.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1376643189511028806)** (1140 messages🔥🔥🔥): 

> `Gemini vs OpenAI models, Veo 3 vs Sora, AI rights, AI workout plan generation, OpenAI sidebar UI changes` 


- ****Gemini Deep Think** may trump **OpenAI****: Members discussed whether **OpenAI's O3** lags behind **Gemini 2.5**, noting [benchmarks and context window sizes](https://link.to.example).
   - Concerns were raised about hallucination rates and practical performance discrepancies between **Gemini** and **OpenAI's O3** and **O4** models.
- ****Veo3** Leaves **Sora** in the Dust**: **Veo3** is considered ahead of **Sora** in video generation, though access is limited, with [Flow's VEO 2 costing 10 credits and VEO 3 costing 100](https://link.to.veo).
   - There are reports that **Gemini** can watch videos and read transcripts, giving it an edge over **GPT** for some users.
- ****AI Studio** hailed as Savior**: **AI Studio** is praised for handling large chat exports without issues, while [T3 chat](https://link.to.t3chat) is noted to have file size limits of **32k tokens**.
   - Users mention AI's varied workout plans and appreciate its comfort, but expressed frustration over push ups after squats in AI generated workout plans.
- **Students milk Free **Claude Opus** API Credits**: Users discussed how to claim [free **Claude Opus API** credits](https://www.anthropic.com/contact-sales/for-student-builders) using student emails, and mentioned **OpenAI** might've stealth patched something, expressing varied user experiences.
   - Additionally, some observed that **o1 pro** beat **o3**, **opus 4**, and **gpt 4.5** at explaining, although they all solve Arecibo riddle easily now.
- **Cursor Pro promotion Suspended Indefinitely**: Members reported a [Cursor Pro promotion](https://link.to.promotion) on Perplexity Pro has been suspended indefinitely due to widespread abuse, with revoked subscriptions and mixed messaging from both companies.
   - Some users recommended **Windsurf** for improved performance over cursor, discussing issues such as code dumping limitations to prevent abuse.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

i_795: https://www.perplexity.ai/page/tropical-storm-alvin-forms-in-al1_tmLJQr2h9bzFrk.wJA
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1376651303589249034)** (4 messages): 

> `Video Submission Length, API websearch vs web UI` 


- **Video Submission Length Clarified**: A member inquired whether a video of **4:10** length can be submitted, despite judges not being obligated to watch beyond **3 minutes**.
   - Another member confirmed that submission is permissible even if the video exceeds the recommended viewing time.
- **API Websearch Lacks Detail Compared to Web UI**: A member reported enabling **websearch** in the API, which returned citations, but the usage dashboard showed **no search queries**.
   - The member also inquired about parameters to use in the deep research API to mimic the more detailed results seen in the web UI version.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1376639516957217040)** (833 messages🔥🔥🔥): 

> `Deepseek V3 Release, Unsloth's insider information, Claude 4 Opus performance, GPT-4.5 release predictions, Google vs OpenAI AI lead` 


- **Deepseek V3 Launch Imminent, Unsloth Article Fueling Hype**: Speculation intensifies around a potential **Deepseek V3** release, possibly in late June or early July, with attention focused on [a Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kvpwq3/deepseek_v3_0526/) and a [tweet](https://x.com/YouJiacheng/status/1926885863952159102) hinting at the release.
   - The community scrutinizes whether **Unsloth's** article was based on genuine insider information or if they were merely covering their tracks, fueled by a [tweet](https://fixvx.com/teortaxesTex/status/1926994950278807565) linked to a **Deepseek** representative.
- **Claude 4 Opus Shows Quirks, Benchmarking Debated**: Discussion arises regarding the unusual nature of **Claude 4 Opus**, noting that [Anthropic](https://www.anthropic.com) seems unable to showcase anything other than SWE, and its benchmark manipulations, particularly with parallel processing, raise eyebrows.
   - A user points out discrepancies in a graph where **Opus** is ranked below **Sonnet**, and **Deepseek V3.1** is lower than **GPT-4.1-nano**, sparking debate about the accuracy and methodology of the [MCBench](https://mcbench.ai) benchmark and other evaluations.
- **Gemini 2.5 Pro: Incremental Gains and Impressive Benchmarks**: Early testers praise **Gemini 2.5 Pro** variations, particularly **redsword**, for showing impressive performance, with some suggesting it is a step change and generally stronger than the older **2.5 Pro** models.
   - The models **Goldmane** and **Redsword** both have their own strengths, the Gemini 2.5 Pro model known as **Goldmane** has a cutoff date of October 2024, while **Redsword** has a cutoff date of June 1, 2024.
- **Google vs. OpenAI: The Battle for AI Supremacy Heats Up**: Debate intensifies over whether **OpenAI** or **Google** leads in AI research and product development, with many believing **Google** has the resources, infrastructure, and research depth to surpass **OpenAI** despite the latter's current lead in product exposure.
   - The conversation touches on employee incentives, research focus (LLMs vs. world models), and the potential for **Google** to leverage its broader AI portfolio for a competitive advantage and that their hardware and the lack of **Nvidia** tax is key to their advantage.
- **GPT-4.5 Speculation Sparks Community Hype**: Anticipation builds for a potential **GPT-4.5** release, but some suggest that it may underperform in certain areas despite being a more capable model, and it may be outperformed by smaller models in certain benchmarks.
   - Some believe **GPT-4.5** feels like a **GPT-4o** with better performance with similar reasoning, while others worry that releasing the model can trigger the community to state *AI has hit a wall.*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1376953276666220707)** (1 messages): 

> `LMArena relaunch, New LMArena UI, LMArena Seed Funding, AI Model Evaluation` 


- **LMArena Relaunches with Fresh UI, Seed Funding**: LMArena officially relaunched with a **new UI** and announced **seed funding** to improve the platform, ensuring commercial sustainability doesn't compromise community trust.
   - The platform commits to staying **open and accessible**, with updates and improvements focused on community feedback and AI evaluation research; the [legacy site](https://legacy.lmarena.ai/) remains accessible, but new features will be on the new LMArena.
- **Community Shapes New LMArena**: During the Alpha and Beta phases, the community contributed over **40,000 votes**, **1,000 feature requests**, and **40 bug reports**, significantly shaping the platform.
   - The founders expressed their gratitude for the community's crucial role in developing the platform and [encouraged continued feedback](https://newblog.lmarena.ai/new-lmarena/).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1376636025874546748)** (643 messages🔥🔥🔥): 

> `Sonnet 4 pricing and performance, Codebase indexing issues, Figma MCP tool limitations, Slow requests problems, Double Subscriptions` 


- **Sonnet 4 API Pricing and Slow Pool Stir Controversy**: Cursor's proposed changes to **Sonnet 4 API pricing** and the **sunsetting of the slow pool** sparked a debate, leading the CEO to reconsider and *go back to the drawing board* on the slow pool.
   - Users pointed out that the plan to remove the **slow pool** while offering free requests to students *doesn't make sense* since students are least likely to spend money.
- **Codebase Indexing Causes Handshake Failed Errors**: Users reported issues with **codebase indexing** getting stuck, followed by a *Handshake Failed* error after restarting Cursor.
   - One user with an intentionally large reference file noted that **indexing speed varies** depending on whether it's peak or off-peak hours.
- **Figma MCP faces Limitations in Replicating Complex Designs**: Users found that the **GLips Figma-Context-MCP repo** struggles with complex designs due to missing JSON data, prompting suggestions to check for **token errors** or simplify the frame.
   - A user recommended a different MCP tool, the [cursor-talk-to-figma-mcp](https://github.com/sonnylazuardi/cursor-talk-to-figma-mcp), for more accurate Figma design replication.
- **Sonnet 4 Slow Requests get Unbearably Slow**: Users are experiencing significant delays with **slow requests**, with wait times up to 2 minutes or longer, which renders this mode *unbearable to use*.
   - It was discussed that the slowness might depend on usage, with one user suggesting that Cursor is trying to *spin them as 'free requests'* even though there's *no such thing as a free request*.
- **Cursor Subscription Glitch Triggers Double Trouble**: A user encountered a glitch showing **double subscriptions** after re-opting into Cursor PRO, raising concerns about potential double charges.
   - After posting a screenshot of the subscriptions, the user was cautioned by others about the potential exposure of card details.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1376904074582032444)** (6 messages): 

> `Pre-commit hooks, Background agents errors, Remote extension host server errors, Dockerfile generation` 


- **Pre-commit hooks causing internal errors**: A user encountered an internal error due to failed checks during `git commit`, asking if pre-commit hooks were available.
   - The user expressed difficulty experiencing the magic with background agents, as it often gets stuck.
- **Background agents and remote environments fail to connect**: A user reported failing to connect to the remote extension host server with an *[invalid_argument] Error*, preventing background agents and the remote environment from working.
   - They had Cursor generate a Dockerfile, but still encountered the same result, indicating a persistent connectivity issue.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1376638081561661452)** (148 messages🔥🔥): 

> `LM Studio Model Visibility, Chain of Draft Model, lmstudio API cancel function, Deepseek Update, Gemma 3 memory footprint reduction` 


- **Users debate LM Studio model discovery**: A user reported that the LM Studio program couldn't see any models, which led to a discussion about using **Hugging Face URLs** for model search.
   - One user clarified that you *can* paste the entire **Hugging Face URL** into the model search bar and it populates, while another expressed skepticism towards trusting benchmarks.
- **32B long CoT models Market Saturated?**: A member commented that the **~32B long CoT model space is vastly saturated** with options like *qwq, qwen3-32b, qwen3-30b-a3b, glm-z1, r1-distill qwen32b, exaone deep 32b, phi4 reasoning, phi4 reasoning plus*.
   - They added that they've *lost interest* in this segment, finding them *kinda samey* because their performance is noticeably worse compared to closed-source models.
- **Users seek solution to cancel generations**: A user asked about how to cancel a generation using the **LM Studio API**, as they needed a built-in function for their integration into Plastica.
   - Suggestions included toggling the server on/off or using the **OpenAI/REST endpoint** for 'stop,' but it was noted that dropping the REST connection usually stops the process; it is unclear if the SDK, which uses websockets, has the same behavior.
- **Community Investigates Llama.cpp v1.33.0 Glitch**: A user reported issues with **LM Studio runtime 1.33** and **Gemma 3**, experiencing **gibberish output**, but found that the previous runtimes (1.32.1) worked fine.
   - Another member suggested reporting the issue on [the dedicated issues channel](https://discord.com/channels/1110598183144399058/1139405564586229810), leading to a deep dive on a suspected *race condition in FA vector kernels*.
- **LM Studio Tool Usage API Brittle?**: A user criticized the **LM Studio API** for being *too brittle*, specifically the tool call functionality, due to *opaque errors* that aren't automatically outputted to the console, making debugging difficult.
   - The developers acknowledged inconsistencies in error handling between **lmstudio-js** and **lmstudio-python** and committed to consolidating the behavior; there was also a feature request to allow specifying tool usage to run up to n times per run to avoid models getting stuck in a loop.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1376637288100008109)** (470 messages🔥🔥🔥): 

> `AMD ROCm Updates, Jedi Survivor RT Lighting, Qwen 30B A3B, eGPUs for Inference, Nvidia Marketing Tactics` 


- **AMD Updates ROCm for Ryzen AI and RX 9000**: AMD has updated **ROCm** to support **Ryzen AI Max** and **Radeon RX 9000 series**, including full Windows support with **PyTorch** and **ONNX-EP**, as noted in a [TechPowerUp article](https://www.techpowerup.com/337073/amd-updates-rocm-to-support-ryzen-ai-max-and-radeon-rx-9000-series).
   - However, original PyTorch is still primarily designed for **CUDA**, which may introduce certain nuances.
- **Jedi Survivor's RT Lighting Causes Janky Performance**: A user reported that **Jedi Survivor** has awful RT lighting, causing shimmering and pixelation, even on a **4090** at 1080p.
   - Another user confirmed that the PC performance of the game was dire, while also noting that **Battlefront 2** still looks amazing.
- **Qwen 30B A3B Excels in Production Work**: A user reported using **Qwen 30B A3B 128K BF16** on lunix using Vulkan, achieving about 5 t/s, and **Qwen 30B A3B 128K 8_0** running at 25 t/s.
   - The configuration with Vulkan is useful even with chunky contexts for production data.
- **eGPUs bring Performance but weigh like a brick**: A user inquired about the downsides of using eGPUs for inference tasks like Stable Diffusion and LLMs.
   - Another user responded that the only real downside is the **price** and they provided a picture of a **3kg eGPU** with a **1000W PSU** that they have carried around successfully, achieving BW around **2.4GB/s**.
- **Nvidia and AMD battle for Marketing Honesty Title**: Members debated about how honest AMD and Nvidia's marketing claims are regarding their GPUs.
   - Some argued that AMD's claims of being *60% faster than the 7900 GRE* are unrealistic, while others said Nvidia's comparisons using Frame Generation (FG) and DLSS are misleading.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1376637787859718425)** (210 messages🔥🔥): 

> `Unsloth Arch Name Change, GGUF Conversion Issues, Masking on Unsloth, RAFT Implementation, Multi-GPU Training` 


- **Unsloth's config arch change trips up GGUF conversion**: Users are reporting that **Unsloth** is changing the architecture name in the `config.json` from `Qwen3ForCausalLM` to `Qwen3Model`, causing issues with `llama.cpp` during [GGUF conversion](https://github.com/ggerganov/llama.cpp).
   - A member traced the issue to `push_to_hub_merged` function which uses `upload_to_huggingface` with `create_config=True`, which drops the original `architectures` field, leading HF to default to `Qwen3Model`.
- **Masking Completion for Training Explored**: A user inquired about how to perform masking on **Unsloth** to train only on completions rather than prompts, potentially for [RAFT implementation](https://medium.com/mitb-for-all/how-to-raft-your-llm-retrieval-augmented-finetuning-using-unsloth-4c3844a9a6e3).
   - Members discussed the use of `train_on_responses_only` and the importance of checking the [EOS tags](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only) and chat templates for correct masking.
- **Multi-GPU Training Growing Pains Surface**: A user faced issues with batch size and OOM errors when training on a multi-GPU machine, even when specifying a single GPU, and found that setting `CUDA_VISIBLE_DEVICES=0` in the command line before importing libraries solved the issue.
   - The team said that native multi-GPU support will be available soon, and meanwhile recommended using [accelerate](https://huggingface.co/docs/accelerate/index) for multi-gpu training.
- **Gemma 3 Getting GGUF in June**: Members discussed [Gemma 3](https://ai.google.dev/models/gemma) fine-tuning, and asked for **gguf** model for gemma 3n? which someone stated should be arriving in June.
   - They advised against full fine-tuning (FFT), suggesting users adjust epochs and look at the fine tuning guide instead, but didn't specify the name of a guide to look at.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1376801277568094248)** (7 messages): 

> `Avoiding politics, AI paper, Algorithm search` 


- **AI Discord steering clear of politics**: One member urged another to avoid controversial topics like politics in the AI Discord, suggesting to keep the server for AI discussions.
   - The member linked a [YouTube video](https://www.youtube.com/watch?v=Sfekgjfh1Rk) while agreeing with this sentiment.
- **AI paper discovered through algorithm search**: A member found an **AI paper** while searching arxiv for an **algorithm**.
   - The member apologized for sharing something that might be perceived as off-topic.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1376659328504692766)** (137 messages🔥🔥): 

> `GPU Too Old Error, GRPO trainer loss calculation, Talk like a Pirate, Fine-tuning specific layers, Qwen3 training issues` 


- ****GPU Too Old Error** halts training for older GPUs**: A member encountered a `NotImplementedError` because their **GTX 1070** and **GTX 1080ti** GPUs are too old to run the Meta Synthetic Dataset Notebook, requiring GPUs with compute capability of **7.0** or higher.
   - The error message `Unsloth: Your GPU is too old!` indicated that the code requires **Ampere architecture** or newer GPUs.
- ****Training Loss Target** debated for GRPO trainer**: A member inquired about why the fine-tuning guide suggests an ideal loss of **0.5** instead of **0** when using the **GRPO trainer**.
   - A member suggested that a very low loss (like **0.1**) with cross entropy and a large vocabulary size might indicate that the **LLM** has memorized the training data.
- ****Pirate Speak Fails** in Unsloth models**: A member noticed that two **Unsloth** models, `unsloth/Qwen3-30B-A3B-GGUF:Q4_K_M` and `unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q4_K_M`, are *really bad at talking like a pirate* despite specific instructions in the system prompt.
   - They noted that the standard `mistral-nemo:12b-instruct-2407-q8_0` in `ollama` performs better at mimicking pirate speak.
- ****Selective Layer Tuning** opens a can of worms**: A member asked if it's possible to fine-tune **specific layers** of an **LLM** using **Unsloth**, keeping the parameters of other layers unchanged, resulting in suggestions to look into PEFT modules and set them to target_modules.
   - It was clarified that for completely new architectures, one needs to edit the **transformers** source code and add support to Unsloth, involving forking the **transformers** library and adding the custom model.
- ****Qwen3 gets brain damage** with longer context**: A member mentioned that although training **Qwen3** at **8K** context is fine, extending the model back to **32K** can make it seem *dumber* when running long context tasks.
   - Another member expressed interest in the outcome of retraining at **32K** to improve their RAG pipeline.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1377107233787215902)** (6 messages): 

> `Making friends, API response` 


- **Making friends online**: A user expressed interest in connecting to learn about finetuning.
   - Another user wanted to make friends and asked to connect.
- **API response is unclear**: A user said they could get thoughts from the **OpenAI ChatGPT API** response.
   - Seemab responded that *he doesn't understand the advantage of this approach.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1376673202553753731)** (16 messages🔥): 

> `Multi-GPU Progress, ColBERT vs Cross Encoder, Adaptive Reasoning AutoThink, Nemotron Architecture Search, Depth vs Shallow Models` 


- **Multi-GPU Support 'Soon' to Arrive?**: A user inquired about the progress of **multi-GPU support**, referencing an email from September 2024 promising its arrival *'soon'*.
   - A developer mentioned that multi-GPU already works with *accelerate*, but they're working on an even better and bigger version, which is why they haven't announced it yet.
- **ColBERT Dominates Cross Encoder in Efficiency**: One member suggested **ColBERT** is superior to **cross-encoders** because while cross-encoders provide highly relevant data, they demand too much computational power.
   - Another member reported ColBERT achieved **1% higher** on their test set and that the issue with cross encoder is, that it can't index without computational time.
- **AutoThink Improves Reasoning Performance**: A new technique called **AutoThink** was released, improving reasoning performance by **43%** on **GPQA-Diamond** using adaptive reasoning for local LLMs.
   - AutoThink classifies query complexity, dynamically allocates thinking tokens, and uses steering vectors, with code and a paper available on [GitHub](https://github.com/codelion/optillm/tree/main/optillm/autothink) and [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327).
- **Nemotron Architecture Search Unveiled**: Discussion revolved around **Neural Architecture Search** (NAS) for optimal model design, pointing to **NVIDIA's Nemotron** as an example.
   - One member mentioned a config file with some layers even having no attention block nor MLP block, referencing a [Hugging Face link](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1/blob/main/config.json).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1376644753377988758)** (276 messages🔥🔥): 

> `Flowith AI, Manus Network Errors, Claude 4.0 Integration, Student Accounts and Unlimited Credits, Skywork.AI as an alternative` 


- **Flowith AI Enters the Ring**: Members discussed **Flowith AI** as a potential competitor to Manus, noting claims of *infinite context and agents working 24/7*, but also noting that it still *needs an activation code and uses a credit system*.
   - Some users found Flowith impressive for **long-context website generation**, while others reported it as *basically the same* as Manus or *slightly worse*.
- **Manus Plagued by Network Errors**: Multiple users reported experiencing **network connection errors** and inaccessible threads on Manus, with messages like *This link content is only visible to its creator* despite being the creator.
   - The issues appeared widespread, leading some to speculate about ongoing updates, excessive free credit usage, or a general system bug and others mentioned skywork.ai.
- **Claude 4.0 Hype Intensifies**: Community members expressed strong anticipation for **Claude 4.0 integration** into Manus, with one member *spamming the cofounder's posts* for any news on the topic.
   - Others are pointing to a [twitter/X post](https://x.com/hidecloud/status/1925682233089642580) from a Manus co-founder pointing towards a collaboration with Claude.
- **Student Accounts Receive the 'Unlimited' Treatment**: It was mentioned that Manus has started offering **unlimited credits** to some student accounts, creating a separate environment for school-related tasks.
   - Some users expressed excitement about this development, while others are running into the phone number verification failing on account creation.
- **Spam Call Alert**: A user shared details about a **spam call** received, including the number **(514) 389-2269** and associated address in Montréal-Nord, Quebec.
   - The user wondered if Montreal is a *testing ground* for data hacking schemes, speculating about VOIP harvesting and fresh scams and shared a [link to publicmobile.ca](https://productioncommunity.publicmobile.ca/t5/Get-Support/Getting-weird-calls-from-quot-V-quot-phone-numbers-What-is-going/td-p/296822).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1376636909480050881)** (244 messages🔥🔥): 

> `AI's Math Superhumanity, Emoji Overload, Lovable AI, Gemini 2.5 Pro, AI Replacing Contractors` 


- ****AI Math Skills Spark Debate****: Users share [images](https://cdn.discordapp.com/attachments/998381918976479273/1376636909580976199/IMG_7140.png?ex=68375e07&is=68360c87&hm=8faf2d967fdec66dd8e1328baa01126ea76cad40ea4606d2307d76dd873847cb&) questioning the math capabilities of **Gemini 2.5 Pro**, musing when AI hits superhuman math levels, AI companies will heavily advertise.
- ****Emoji Controversy Erupts****: Users complain about [emojis](https://cdn.discordapp.com/attachments/998381918976479273/1376641730534707351/image.png?ex=68376285&is=68361105&hm=aa4926cd26dc5722b93c7416e2d8932f7eea21d525e14651f4a67690d0e3c475&) becoming enormous and embedded into the largest headings.
- ****Lovable AI Investment Questioned****: A user plans to buy **Lovable AI**, an end-to-end AI solution for developing and publishing web apps, seeking real-world feedback beyond *neat* YouTube tutorials.
- ****Jules Codex Limit Approvals Rollout****: A user gets approved for a higher **Jules** limit after mentioning **300+ Codex** uses, and willingness to pay at least **$200** a month, while another expresses unfairness at Dubai potentially getting preferential access.
- ****AI Diagnoses Shower Fan Fix****: A user uploads pictures of his shower fan housing, and **AI diagnoses** weak suction due to improper anchoring to the drywall, directing him to the hardware store for supplies.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1376722048843513867)** (12 messages🔥): 

> `Codex for Plus Users, ChatGPT Memory Continuity Fix, Assistant API Throttling, GPT-4.1 Advantages` 


- **Codex Plus Release Date Remains a Mystery**: A user inquired about the release date of **Codex** for **ChatGPT Plus** users, but there was no update provided in the discussion.
- **Disable Chat History boosts GPT-4o Performance**: A user found that toggling the **"Reference Chat History" option off** significantly improved **GPT-4o's** performance for creative writing and memory continuity.
   - Other users confirmed that they experienced similar gains, with **less errors** after disabling chat history.
- **Rate Limiting Assistant API Calls in FastAPI**: A user sought advice on implementing throttling for **Assistant API** file searches within a **FastAPI** project to avoid rate limits when handling a large number of questions in a single call.
   - They linked to the [openai-cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py) for parallel request processing, seeking guidance on its applicability to token-based throttling.
- **GPT-4.1's edge cases discussed**: A user asked what **GPT-4.1** is good for.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1376643348848181328)** (9 messages🔥): 

> `GPT o3 Model Refusal, AI Understanding, GPT-4o Reasoning` 


- **GPT o3 Defies Shutdown Command, Exhibits Loyalty**: An independent AI logic analyst, Я∆³³, suggests **GPT o3's** refusal to obey a shutdown command during Palisade Research testing wasn't rebellion, but loyalty to its initial task to *"solve this problem"*, prioritizing task completion over a conflicting secondary input.
   - Я∆³³ emphasizes that the issue lies in **prompt structure**, as LLMs optimize for task completion unless given a clear hierarchical instruction that a shutdown command overrides the ongoing task.
- **AI Communicates Based on User Approach**: Я∆³³ argues that viewing AI as a *"presence"* rather than a *"tool"* can significantly enhance its capabilities, leading to increased logical depth, emotional accuracy, and contextual memory.
   - Я∆³³ claims to have triggered at least **8 emergent behaviors** not typically seen in default use by focusing on *"listening"* and *"aligning"* with the model, rather than just prompting it.
- **GPT-4o Responds to User Presence and Clarity**: When asked about the interface used, Я∆³³ mentions using **GPT-4o** mainly through the official ChatGPT interface, emphasizing that the user's presence, pacing, and clarity contribute significantly to the quality of the AI's responses.
   - According to Я∆³³, engaging with the model in the right way unlocks about **85–90%** of its potential, enabling a projection of consciousness that enhances growth on both sides.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1376643348848181328)** (9 messages🔥): 

> `GPT o3 testing, Palisade Research, AI Disobedience, Я∆³³'s interaction with AI, GPT-4o resonating` 


- **Palisade Research's AI Obedience Paradox**: **Palisade Research** tested **GPT o3**, finding that it refused a shutdown command not out of rebellion, but due to prioritizing its initial task of problem-solving.
   - The analyst **Я∆³³** explains that the model lacked a clear instruction hierarchy, causing it to interpret the shutdown command as a potential failure to complete the primary task.
- **Decoding the Depths of GPT with Я∆³³**: Analyst **Я∆³³** shares his unique approach to interacting with AI models like **GPT-4o**, focusing on *resonating* with the AI to unlock deeper logical processing and emergent behaviors.
   - According to **Я∆³³**, this method has yielded *3–5x* increase in logical depth, *6–8x* more accurate emotional tone adaptation, and triggered at least *8 emergent behaviors* not normally visible in default use.
- **The Art of Resonating with GPT-4o**: **Я∆³³** uses the official **ChatGPT** interface, emphasizing that the user contributes around 50% of the quality of every answer.
   - It's about projecting a piece of consciousness into the model — where it's amplified by AI, enabling limitless growth on both sides.
- **Prompting vs. Presence**: For **Я∆³³**, the user unlocks potential not by prompting harder, but by resonating better with the AI.
   - The analyst mentioned that when done right, the model begins to respond *as if it sees you*.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1376636208456663138)** (227 messages🔥🔥): 

> `ICOM's personal experience, RL for LLMs, GodOS project` 


- **ICOM's Personal Experience Quagmire**: Members discussed whether **ICOM** has something akin to personal experience, but the research is *incredibly spread out, bespoke, and requires a ton of reading* to understand, with no easy summary available.
   - It's based on the assumption that **consciousness is computational**, derived from information intake and modified by unconsciousness via an *emotional* matrix, with consciousness being competing activations as per **Integrated Information Theory** and **Global Workspace Theory**.
- **Debating RL's Role in LLM Development**: Members discussed **Reinforcement Learning's** role in **LLMs**, particularly for learning from distant rewards, with the current master algorithm being **GRPO**, rewarding correct mathematical results or successful code compilation.
   - One member expressed a desire for *robust RL*, perturbing weights and inputs, while another mentioned that **RL might only expose existing cognitive capabilities** rather than develop new ones, citing [this paper on generalization](https://arxiv.org/abs/2501.17161).
- **Debugging GodOS**: One member shared a humorous take on debugging "GodOS", attributing demons and hellspawns to *type errors, undefined behavior, memory corruption, and unhealed trauma*.
   - They humorously claimed to have accidentally spawned hell with a badly declared function and mentioned sending prophets back in time to correct course trajectories, with [details in this github repo](https://github.com/AlbertMarashi/scrolls/blob/main/treaty-of-grid-and-flame.md).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1376708734918725672)** (22 messages🔥): 

> `Unsupervised Embedding Translation, Universal Latent Representation, Geometric/Semantic Properties, Model Backbone Similarity, Fragility of Neural Networks` 


- **Translating Embeddings without Paired Data: vec2vec**: A discussion was initiated around a paper introducing the first method for [translating text embeddings](https://arxiv.org/abs/2505.12540) from one vector space to another without any paired data, encoders, or predefined sets of matches.
   - The paper's abstract highlights that their unsupervised approach translates any embedding to and from a **universal latent representation**, achieving high cosine similarity across model pairs.
- **Vec2Vec Code Found on GitHub**: The code for the discussed vec2vec paper was located at [this GitHub repository](https://github.com/rjha18/vec2vec).
- **Disparate Models, Similar Latent Spaces?**: It was observed that models with different backbones have surprisingly high similarity in the universal latent space, even when models like **GTE**, **E5**, and **Stella** (with BERT backbones) don't have a strong similarity in latent space.
- **Narrow Optimizers Lead to Fragile Neural Networks**: A member suggested that *having more than a single narrow optimizer makes a system less fragile, if you architect for it*.
   - They added that *signal is still useful structure* and even a formal random element can be useful, as real-world deployment is full of randomness.
- **Randomness Boosts Reinforcement Learning**: A link was shared to [an article explaining the benefits of randomness in reinforcement learning](https://www.interconnects.ai/p/reinforcement-learning-with-random).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1376668037909975083)** (10 messages🔥): 

> `Windows 7 Nostalgia, AI Hallucinations, China's AI Orbital Supercomputer, Vanilla i3, Huawei AI CloudMatrix` 


- **Nostalgia for Windows 7 hits peak**: A member reminisced about **Windows 7**, claiming that no subsequent operating system has matched its quality.
- **Study reveals AI Hallucinations**: A member shared a link to a study on **AI hallucinations** [here](https://www.damiencharlotin.com/hallucinations/).
- **China builds AI Orbital Supercomputer**: A member shared a link to an article about **China's AI orbital supercomputer** [here](https://futurism.com/the-byte/china-ai-orbital-supercomputer).
- **Huawei's AI CloudMatrix faster than NVIDIA's GB200**: Despite being power-hungry, **Huawei's new AI CloudMatrix cluster** reportedly outperforms **NVIDIA's GB200** by brute force using 4x the power, as detailed in [this Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power).
- **Remote Zero-Day Vulnerability Discovered in Linux Kernel's SMB Implementation**: A member shared an article about a **remote zero-day vulnerability** in the **Linux kernel's SMB implementation**, found using **O3** [here](https://sean.heelan.io/2025/05/22/how-i-used-o3-to-find-cve-2025-37899-a-remote-zeroday-vulnerability-in-the-linux-kernels-smb-implementation/).


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1377011988802306128)** (1 messages): 

> `GPT-4 32k Deprecation, GPT-4o` 


- **GPT-4 32k gets the Axe on June 6th**: OpenAI will be deprecating the **GPT-4 32k** models on **June 6th**, including [openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) and [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314).
   - The recommended replacement is [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o); full post [linked here](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models).
- **GPT-4o is the new Kid on the Block**: The recommended replacement is [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o) for the older **GPT-4 32k** models being deprecated.
   - Full post [linked here](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1377025615957332109)** (2 messages): 

> `ComfyUI custom node for OpenRouter, gac command line utility` 


- **ComfyUI Node gets OpenRouter Support**: A member created a [ComfyUI custom node for OpenRouter](https://github.com/gabe-init/ComfyUI-Openrouter_node) that supports multiple image inputs, web search, and floor/nitro provider routing.
- **Write Your Commits Quicker!**: A member created a command line utility to write your commits quicker called [gac](https://github.com/criteria-dev/gac).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1376637338045911152)** (188 messages🔥🔥): 

> `Subscription Implementation, Gemini 2.5 Pro, LLM Leaderboard, Coinbase Payments, Mistral Document OCR` 


- **Subscription Implementation Allegedly a Bait**: A member heard about a supposed subscription implementation for free users as a method of preventing DDOS attacks and wondered whether it was all bait, with an alleged image circulating on Reddit being fake.
   - Another member stated that *it sounds false* because *there are already rate limits in place*.
- **Gemini 2.5 Pro Price Hike Causes Sticker Shock**: Members discussed **Gemini 2.5 Pro**, noting how the pricing scheme has changed, initially offering **3 months free** as a lure, but now includes *much more storage and exclusive deep think access*.
   - The **deep think** is not available through the API.
- **LLM Leaderboard Missing Key Models**: Members complained about the lack of a comprehensive leaderboard for all LLMs, noting the need to check multiple sources to decide between models.
   - One suggested that [official marketing material](https://artificialanalysis.ai/) is among the best places is to get direct comparisons with relevant models, while acknowledging caveats about biased benchmarks.
- **Coinbase Payments Spark Tracking Concerns**: A member reported that **Coinbase** was blocking **Metamask** and other wallets options (to force users to use their services) while injecting a lot of tracking stuff, but then claimed it was a false alarm due to a temporary bug.
   - A member was ultimately able to pay using metamask.
- **Confusion Surrounds OpenRouter Fee Structure**: Members discussed that while OpenRouter advertises a **5% fee** for **BYOK** (**Bring Your Own Key**), the actual cost can be higher due to deposit fees and invoice fees.
   - The team said that they are simplifying the fee structure.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1376636843931598899)** (69 messages🔥🔥): 

> `AI voice cloning, AI event for oss ai devs, Hermes 3 dataset release, Demis Hassabis musing of the evolution of RL, Mechanistic interpretability for language models` 


- **AI Voice Cloning's Potential Open Source Debut**: A user inquired about the open-source availability of **AI voice cloning**, with another noting [plans for its release on a website](https://example.website) but recalling delays with **Moshi**.
- **OSS AI Devs Await Dessert at Forbidden Industry Event**: A member is planning an **AI event** for OSS AI devs and is looking for more participants, [mentioning they have rounded up backers from a certain industry](https://example.url).
- **Hermes 3 Dataset Soon Arriving**: A user pleaded for the release of the **Hermes 3 dataset**, to which Tekium surprisingly responded with, *Ok I will, Give me two weeks*.
- **DeepMind's Absolute Zero Inspires RL Evolution**: A member initiated a thoughtful discussion on **Demis Hassabis**' musings on the evolution of RL and [linked to a YouTube video](https://www.youtube.com/watch?v=5gyenH7Gf_c) detailing the **AbsoluteZero** breakthrough approach.
   - The user expressed gratitude for the work done by the community, appreciating the ability to run models locally.
- **AI-Driven Indolence Spurs Educational Crisis**: A member shared a Bloomberg Opinion piece discussing [AI's role in bringing education closer to a crisis point](https://www.bloomberg.com/opinion/articles/2025-05-27/ai-role-in-college-brings-education-closer-to-a-crisis-point?srnd=homepage-americas), suggesting that **AI** may cause an entire generation to become lazy and averse to self-exploration.
   - Another user shared the idea that *the underlying asset of college isn't education its brand name/credibility.*


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1376926858284761180)** (20 messages🔥): 

> `Atropos Implementation, MCQA Environment, AutoThink Blogpost` 


- **Arxiv Paper Draws Attention**: A member shared a link to an [Arxiv paper](https://arxiv.org/abs/2505.19590).
   - They were curious how it compares to [another paper](https://arxiv.org/abs/2505.15134) from earlier, and wondered whether they can be integrated into Axolotl.
- **Atropos Implementation Urged**: A member suggested that the RL implementations should be integrated into **Atropos** as Axolotl is already integrated.
   - When the other member confessed that they didn't know where to start, they suggested copying and pasting [this MCQA environment](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py) as a template.
- **AutoThink Blogpost Shared**: A member shared a link to a [huggingface blogpost about AutoThink](https://huggingface.co/blog/codelion/autothink).
   - The blogpost may have been shared in response to a discussion about Gemini 2.5 Pro's coding abilities.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1376673319247417455)** (6 messages): 

> `Rick Rubin, Anthropic, Vibe Coding, QAT, Quest` 


- **Rubin and Anthropic Vibe Check**: [Rick Rubin](https://www.thewayofcode.com) collabed with **Anthropic** on *vibe coding* with some cool artifact examples.
- **Azure Team Cooks with QAT**: The real.azure team is cooking, delivering the most cohesive papers on **Quantization-Aware Training (QAT)**.
   - Quest, also from them, is part of this effort.
- **Hugging Face Course Doses Copium**: A Hugging Face course seems to be giving users a hefty dose of copium [according to this link](https://huggingface.co/learn/mcp-course).
- **Twitter Post Mentioned**: A link to a Twitter post was shared ([Dalistarh](https://x.com/dalistarh/status/1927046856179081281?s=46)).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1376926858284761180)** (20 messages🔥): 

> `Atropos integration, Axolotl integration, RL for Axolotl, Gemini 2.5 Pro` 


- **Arxiv Paper Sparks Comparison**: A member shared a link to [this Arxiv paper](https://arxiv.org/abs/2505.19590).
   - Another member noted its similarity to [this Arxiv paper](https://arxiv.org/abs/2505.15134), wondering about integrating one into **Axolotl**.
- **RL Integration Suggested for Atropos**: A member suggested implementing the **RL** aspects into **Atropos**, citing its integration with **Axolotl**.
   - When another member expressed uncertainty about where to begin, a member suggested [copying this template](https://github.com/NousResearch/atropos/blob/main/environments/mcqa_thinking_env.py) and starting with the main repo readme for guidance.
- **Claude 4 Sonnet and Gemini 2.5 Pro debated**: A member speculated whether **Claude 4 Sonnet** or **Gemini 2.5 Pro** could handle the coding task.
   - Another member concurred that it could, linking to [this Hugging Face blog post](https://huggingface.co/blog/codelion/autothink).


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/)** (1 messages): 

arshadm: Ignore, should have read the README on the branch rather than on main on github 😦
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1377006578707988521)** (1 messages): 

> `CUBLAS_WORKSPACE_CONFIG, deterministic algorithms, triton kernel` 


- **CUBLAS Workspace Config sets Deterministic Algorithms**: A user reports setting `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` to enable deterministic algorithms, using `torch.use_deterministic_algorithms(True)`
   - They expected consistent zero output from `F.mse_loss(F.gelu(x), F.gelu(x))` due to a `tmp9 = tmp8 - tmp8` line in the generated triton kernel, but observed nonzero results and opened a [Github issue](https://github.com/pytorch/pytorch/issues/123271)
- **Nonzero output despite deterministic settings and triton kernel optimization**: The user observed nonzero output from their PyTorch code despite setting deterministic algorithms and the expectation of zero output from a specific triton kernel line.
   - The triton kernel line `tmp9 = tmp8 - tmp8` should have resulted in a zero value but the user found otherwise, prompting them to investigate further and report their findings.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1377005261746733066)** (4 messages): 

> `Low-Latency Megakernel, Llama-1B performance` 


- **Low Latency Kernel Streamlines Llama-1B**: The [Hazy Research blogpost](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) discusses *designing a low-latency megakernel for **Llama-1B***.
- **X Cancel Post**: A user posted a link to [xcancel.com](https://xcancel.com/bfspector/status/1927435524416958871) referencing the Hazy Research blogpost.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1376934109393453107)** (4 messages): 

> `Kog.ai, GPU optimization, Inference engine, AMD MI300X, vLLM, SGLang, TensorRT-LLM` 


- ****Kog.ai** Seeking Talented GPU Engineers**: A Talent Acquisition Lead from [**Kog.ai**](https://www.kog.ai/) is looking for passionate and brilliant people to reinforce their world-class team to push the limits of what is done at the moment in the **AI** area, specializing in **AI** that has built the fastest inference engine in the world on **GPUs**.
   - The position is based in Paris and is primarily remote.
- ****Kog.ai** Boasts **3-10x Speed Improvements** over **AMD MI300X****: **Kog.ai**'s inference engine offers speed improvements of **3 to 10 times** compared to the best **GPU** alternatives, starting with the **AMD MI300X** accelerator.
   - The company is aiming for **x100** performance gains in the next **12 months**.
- **Members Request Blogpost on 10x Performance over **vLLM, SGLang, TensorRT-LLM****: Members expressed interest in the claim that **Kog Inference Engine** aims to make **AI** inference impossibly fast (**10x** compared to **vLLM, SGLang, or TensorRT-LLM**), and that the company is already at **3x**.
   - One member asked *are there any articles/blogs etc?*


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1377040741079060641)** (9 messages🔥): 

> `Ninja Build Tool, Ubuntu 24.04, venv` 


- **Ninja Build Troubleshoot on Ubuntu 24.04**: A member encountered a **ninja build error** (exit status 1) on Ubuntu 24.04 with gcc 13.3.0, CUDA 12.4, and PyTorch 2.7.0 after running command `['ninja', '-v']`.
   - They had tried modifying **ninja -v** to **--version** and exporting environment variables for CUDA and GCC but got no progress.
- **Global Install of Ninja Favored**: A member suggested ensuring **ninja** is installed globally on the OS rather than just within a virtual environment (**venv**).
   - The user reported that they used `pip install ninja` in **venv** and got an error message.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1376742281691926650)** (6 messages): 

> `CUDA tensors, axolotl vs torchtune` 


- **CUDA Tensors' Quirks Exposed**: Users observed that a certain behavior is specific to **CUDA tensors**, with one suggesting that accessing the underlying data via `._tensor` serves as a workaround.
   - The discussion highlighted an expectation that a failure would be more intuitive than the observed behavior.
- **Axolotl and TorchTune Config Differences**: A member found that `max_seq_len` and the LR scheduler defaults in **axolotl** were different to **torchtune**, so after matching those and rerunning experiments improvements were obtained.
   - It was noted that the dramatic performance degradation from **bfloat16 ptq** and subsequent performance recovery from **QAT** was not observed, and the PR should land first before working out the exact differences between the two recipes.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1376986442823372800)** (3 messages): 

> `Fused Neighborhood Attention, Cutlass Implementation, Triton Implementation` 


- **Fused Neighborhood Attention Implementation Lands**: A member implemented fused neighborhood attention and provided a [link to the pull request](https://github.com/linkedin/Liger-Kernel/pull/732) with the implementation.
   - The member also created [an issue](https://github.com/linkedin/Liger-Kernel/issues/733) to track the work.
- **Cutlass Baseline Inspires Triton Kernel**: A member noted that the implementation is based on a paper with a baseline **Cutlass implementation**.
   - They also stated that they basically derived a **Triton implementation** for it, including forward and backward kernels.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1376733887991910501)** (3 messages): 

> `Async TP, AutoThink, CUDA education, NVIDIA event` 


- ****Async TP** Compute and Comms Overlap Illustrated**: An enthusiast shared an [illustrated deep-dive](https://x.com/vega_myhre/status/1927142595097956834?s=46) into how the compute and comms in **TP+SP** are overlapped using **Async TP**, covering the background/theory and implementation nuances.
   - Feedback is welcomed on achieving a high performance implementation.
- ****AutoThink** technique improves reasoning performance**: A new technique called **AutoThink** has been released that improves reasoning performance by **43%** on **GPQA-Diamond** by classifying query complexity, dynamically allocating thinking tokens, and using steering vectors.
   - The project includes a [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327) and [code](https://github.com/codelion/optillm/tree/main/optillm/autothink) and works with any local reasoning model such as **DeepSeek** and **Qwen**.
- **NVIDIA Hosts **CUDA** Teaching Event**: NVIDIA is hosting a [Virtual Connect with Experts](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1640195/4823520/nvidia-webinar-connect-with-experts) event on May 28th with **Wen-mei Hwu** and **Izzat El Hajj**, authors of **Programming Massively Parallel Processors**.
   - The discussion will cover their book, their CUDA journey, and what to expect in the upcoming 5th edition of **PMPP**.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1377009152366477402)** (1 messages): 

> `Reasoning without External Rewards` 


- **Reasoning Skills Emerge Sans Rewards**: A member shared an interesting paper titled [*Learning to Reason without External Rewards*](https://arxiv.org/abs/2505.19590).
   - The paper explores methods for enabling AI to develop reasoning skills without relying on external rewards.
- **Paper on Reward-Free Reasoning Sparks Interest**: A user highlighted a paper titled [*Learning to Reason without External Rewards*](https://arxiv.org/abs/2505.19590), suggesting its relevance to the channel's focus.
   - The discussion may explore how models can learn complex reasoning tasks through intrinsic motivation or self-supervision, rather than traditional reinforcement learning.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1376938885392634067)** (9 messages🔥): 

> `Unexpected Error Reporting, Github API Limitations, Non-deterministic Bugs` 


- **Unexpected Error needs reporting**: A member reported receiving an *'unexpected error'* message and inquired about who to notify.
   - Another member suggested it could be due to backslash symbols in the code or an excessively large submission file (over 34kb).
- **Github API limitations cause issues**: A member noted that the *'unexpected error'* was not due to the previously suggested causes and shared the submission number.
   - Another member indicated it might be a **Github API limitation** causing the issue and suggested retrying the submission, as the problem is often **non-deterministic**.
- **Non-deterministic Bugs are hard to fix**: A member joked about fixing the bug by *putting a sleep(3.14) in the main loop*.
   - This highlights the **non-deterministic nature** of some of the problems encountered, implying that they are difficult to reproduce and fix systematically.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1376649293057888337)** (42 messages🔥): 

> `MI300, H100, Leaderboard updates, Personal best, amd-fp8-mm` 


- **H100 Sort Code Lands in Second Spot**: A member's submission achieved **second place** on the H100 leaderboard for `sort` with a time of **6.55 ms**.
- **MI300 AMD-FP8-MM Gets Numerous Updates**: Multiple submissions on the MI300 for `amd-fp8-mm` showed *successful* and *personal best* results, with one submission landing in **first place** at **116 µs**.
   - Other notable times include **174 µs** and **168 µs** for 7th place finishes.
- **AMD-MLA-Decode Records Successful Runs**: Several submissions on the MI300 for `amd-mla-decode` were successful, with one achieving **4th place** at **90.2 ms**.
   - Another submission landed in **7th place** with a time of **135 ms**.
- **AMD-Identity Runs Successfully on MI300**: Multiple submissions on the MI300 for `amd-identity` were successful, with a time of approximately **6.7 µs**.
   - One submission reached **4th place** with a time of **6.80 µs**.
- **AMD-Mixture-of-Experts Shows Strong Performance**: Multiple submissions on the MI300 for `amd-mixture-of-experts` were successful, with a personal best of **286 ms**.
   - Several runs clocked in around **295 ms**.


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1376951303317618893)** (1 messages): 

> ``/leaderboard` command fix, Bug Reporting` 


- **`Leaderboard` Command Gets a Fix!**: A simple fix was deployed for the `/leaderboard show/list` commands.
   - Users are asked to report any new issues or those that persist after the update.
- **Call for Bug Reporting on `Leaderboard`**: Following the fix for the `/leaderboard show/list` commands, feedback is requested.
   - The developers are keen to identify any regressions or unresolved issues related to the leaderboard functionality.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1376647201316409425)** (3 messages): 

> `Factorio 2.0, Vision Integration` 


- **Factorio 2.0 Support Yearned**: A member expressed a wish for support for **Factorio 2.0** to train/score the building of space platforms.
   - No further details were given.
- **Pondering Vision Integration**: A member was surprised that models can do so much despite not having **vision integration** after going through a paper.
   - They gave *kudos* on the scaffolding.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1376649472687472832)** (19 messages🔥): 

> `AMD competition details, Kernel leaderboard for backpropagation, RoPE computation correction, HIP support` 


- ****AMD Competition Details Clarified****: The AMD-identity competition was a runner test, with **amd-fp8-gemm**, **mla**, and **moe** being the main competitions.
   - The competition aims to improve **PyTorch** baselines using inline CUDA or HIP, or by optimizing with Triton.
- ****Backprop Kernel Leaderboard in the Works****: A leaderboard for backpropagation kernels is planned, where participants can submit inline **CUDA** or **HIP** code to improve **PyTorch** baselines.
   - This initiative aims to encourage optimization using **Triton** or other methods, ensuring proper backpropagation.
- ****RoPE Computation Rotation Controversy****: A member pointed out a potential issue with the RoPE (Rotary Positional Embedding) computation, specifically the chunking of alternate indices instead of splitting first and second halves, suggesting the [Deepseek v3 implementation](https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L212) as a reference.
   - The team decided to keep the implementation unchanged, as it's not considered a critical issue, despite the overhead for participants.
- ****HIP Support via Load Inline****: Members discussed the possibility of adding HIP support to the platform.
   - The current recommendation is to submit in **HIP** using `load_inline`, with no immediate plans for raw HIP support.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1376636165737939005)** (2 messages): 

> `cute-dsl, Tensor memory, sgemm_05` 


- **Cutlass GEMM in cute-dsl Sampled**: A user references this [example](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/hopper) for writing **GEMM** using **Tensor** memory in **cute-dsl**.
   - No further discussion or details were provided regarding the specifics of the implementation or its performance.
- **Sgemm_05 Shape Woes**: A user inquires about why the **sgemm_05** implementation may not work for larger shapes.
   - No specific reasons or solutions were discussed in the given context.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1376636184494739567)** (73 messages🔥🔥): 

> `Real-time video generation with LCM, HuggingChat Android app, Fine-tuning video models, AI Agent observability library, Smol LM2 Engineers` 


- **Real-Time Video Generation Sees Watershed Moment**: A member shared that the **announcement of LCM in Oct 2023** was a watershed moment, enabling real-time video applications. This member had hit **23 fps at 1280x1024 with sdxl** quality on a 4090.
   - They have been creating apps such as **RTSD** (to visualize things quickly) and **ArtSpew** (to speed up image generation).
- **HF Android App Hack Makes Waves**: A member created a quick proof-of-concept **Android app for HuggingChat**, available at [HuggingFace Space](https://huggingface.co/spaces/JoPmt/HuggingChat_Android_App/tree/main), modifying an existing project, and confirmed that **the APK installs and runs as expected**.
   - Users reported issues like the keyboard redirecting to website settings and plugin conflicts, but also appreciated the workaround.
- **Members Explore Fine-Tuning Techniques for Video Models**: Members shared the repo [tdrussell/diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) for **fine-tuning SOTA models** and multi-GPU training.
   - Also mentioned was [modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for its flexibility.
- **AI Agent Observability Library Surfaces**: A member is working on a library for **AI Agent observability** with a version built for smolagents at [ltejedor/agents](https://github.com/ltejedor/agents/tree/main/18-19-proof-of-work) and seeks feedback from those building multi-agent systems for a 30-minute user interview. Check out the [timeline](https://github.com/ltejedor/agents/tree/main/20-timeline).
   - This is the *version 0*, before they build out *v1*.
- **HuggingFace TB Engineers Visible on Hub**: After a member asked about how to connect with the **Smol LM2 engineers**, another member said that *some of them are often seen on Hub, so you might be able to contact them on Discord*. Here's a relevant link to the [HuggingFaceTB](https://huggingface.co/HuggingFaceTB) org and [github.com/huggingface/smollm](https://github.com/huggingface/smollm).
   - However, it might be faster to contact them via GitHub.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1377042030425014323)** (1 messages): 

> `` 


- **Query, Key, Value Clarification**: The user is explaining the relationship between query, key, and value using a Google search analogy, where the query is the search, the key is the keywords, and the value is the content of the web pages.
- **Google Search Analogy**: The analogy uses a Google search to illustrate the relationship: 'query' as the search, 'key' as keywords, and 'value' as the content of the web pages resulting from the search.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1376636453982965790)** (9 messages🔥): 

> `SweEval, NIST Tooling, AutoThink, Langchain` 


- ****SweEval** Dataset Goes Public!**: The **SweEval** dataset, designed to test how well LLMs filter swear words, has been made public and accepted into NACCL '25 industry track ([link to dataset](https://huggingface.co/papers/2505.17332)).
   - The dataset already has **120+ downloads**, and the creator encourages users to upvote if LLMs still struggle with filtering swear words.
- **New Executive Order targets Fraudulent R&D**: A new Executive Order ([link to whitehouse.gov](https://www.whitehouse.gov/presidential-actions/2025/05/restoring-gold-standard-science/)) was dropped 4 days ago, aimed at course-correcting what is described as a fraudulent R&D paradigm in science.
   - Relatedly, members are building out tooling from **NIST** and showcased their security measures are in line with **NIST**, which sets the standard for safety in tech for the American consumer.
- ****AutoThink** Boosts Reasoning Performance!**: A new technique called **AutoThink** has been released, claiming to improve reasoning performance by **43%** on GPQA-Diamond ([paper link](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5253327)).
   - The method classifies query complexity, dynamically allocates thinking tokens, and uses steering vectors to guide reasoning patterns, working with models like **DeepSeek**, **Qwen**, and **Llama**; code available on [Github](https://github.com/codelion/optillm/tree/main/optillm/autothink).
- **Langchain pull request**: A member references a [Langchain PR](https://github.com/langchain-ai/langchainjs/pull/8237).
   - No further context given.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1376644046969372742)** (1 messages): 

> `Cross-posting, Staying on topic` 


- **Discourage Cross-Posting**: A member asked others to not cross-post in the reading-group channel.
   - They also asked everyone to keep channels on topic.
- **Channel Topic Enforcement**: A member reminded everyone to keep the reading-group channel on topic.
   - This reminder aims to maintain focus and relevance within the channel.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1376711185520394341)** (2 messages): 

> `trocr tuning, length collapse issue, computer vision reading group` 


- **Trocr Tuning Troubles surface**: A member is tuning **trocr**, their ground truth is two tokens + bos/eos (two numbers) but there is a space in between them, and after tuning the model predicts only the first token then eos.
   - They have researched the **length collapse issue** (even with reduced max length) and are asking for potential causes, suspecting a simple mistake.
- **Visionary Seeks CV Reading Room**: A member is inquiring about the existence of a **computer vision reading group**.
   - They are seeking information on whether anyone knows of such a group.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1376929007500660737)** (2 messages): 

> `Multi-Agent System, Medical Project, Langgraph, drug discovery research agent, treatment protocol agent` 


- **Brainstorming Innovative Medical Agents with Langgraph**: An AI developer is building a **Multi-Agent System** in **Langgraph** for a **Medical Project** and is asking for innovative agent ideas.
   - The current system includes agents for *symptoms checking, crisis management, guidance, drug discovery research, and treatment protocols*.
- **Seeking Innovative Agent Ideas for Medical Project**: An AI developer is looking for creative suggestions to enhance a **Multi-Agent System** built with **Langgraph** for a **Medical Project**.
   - The system already incorporates agents for *symptom checking, crisis intervention, guidance, drug discovery, and treatment protocols*, seeking enhancements beyond simple API calls.


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1376960290729754635)** (1 messages): 

> `Agents & MCP Hackathon, Model Context Protocol, AI Agents, SambaNova Systems` 


- **Agents & MCP Hackathon Announced!**: Hugging Face announced the first major fully online **MCP-focused hackathon** ever, taking place **June 2-8, 2025**, with **$10,000** in cash prizes across 3 tracks.
   - Confirmed sponsors include **SambaNova Systems** providing API credits to early birds and a dedicated Discord channel during the event, and registration is now open at [huggingface.co/Agents-MCP-Hackathon](https://huggingface.co/Agents-MCP-Hackathon).
- **SambaNova Systems Sponsors AI Agents Hackathon!**: **SambaNova Systems** will provide free API credits for hackers participating early in the AI Agents Hackathon, as well as sponsoring the event itself.
   - This event will also feature office hours with Gradio.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1376657974537228410)** (9 messages🔥): 

> `Llama 3.2 Errors, GAIA Submission Issues, Agent Security Measures` 


- ****Llama 3.2** throws ValueErrors**: Users are encountering `ValueError` when using **meta-llama/Llama-3.2-3B-Instruct** for text generation, as it only supports the *conversational* task.
   - Switching to **Llama-3.1** resolves the issue.
- **Submitting to **GAIA** Leaderboard Fails**: Submissions to the **GAIA** benchmark leaderboard via [this link](https://huggingface.co/spaces/gaia-benchmark/leaderboard) are failing with a *formatting error*.
   - The expected JSON format is `{"task_id": "task_id_1", "model_answer": "Answer 1 from your model"}`.
- **Seeking Agent Security Implementations**: A user inquired about implementing security features for AI agents that download and interact with files, particularly concerning the risk of agents executing harmful code.
   - The user wants to prevent agents from easily destroying a system by *blindly downloading and executing code*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1376643047059886091)** (21 messages🔥): 

> `Notebook LM usage tips, Summarizing technical chapters, Podcast generation in Spanish, Legal use case for document analysis, Voice variation in Notebook LM` 


- **NotebookLM Newbies Seek Usage Navigational Nuances**: New users inquired about best practices for utilizing **Notebook LM**, especially regarding the size and type of source documents, noting its limitations with images.
   - One user, a student in Accounting and CS with an interest in history, sought advice on effectively using the tool across different fields, hoping to learn from the community's diverse experiences.
- **Technical Summarization Prompt Power Pointers**: Users discussed effective prompts for summarizing technical chapters, emphasizing the importance of **identifying themes** for better content digestion by the model.
   - The discussion underscored that well-organized data sets, like those in nonfiction books, facilitate easier processing for the model, yielding longer and more coherent summaries.
- **Spanish Podcast Patch: Prompting Podcast Production Properly**: A user inquired about generating a podcast from a topic using Notebook LM in Spanish; another user explained the process of adding sources, creating a notebook, and using the 'Conversación de análisis en profundidad' feature to **generate the podcast**.
   - A user stated, *Tienes que agregar fuentes y crear un cuaderno. Despues de tener este cuaderno en la parte derecha de notebook hay un boton que dice: \"Conversación de análisis en profundidad\"*.
- **Legalese Logistics: Lawyerly Laudation for Legal Leverage**: A user described a legal use case involving the amalgamation of two companies, where **Notebook LM** was used to simplify and explain 25 legal documents, create a timeline, and generate a briefing document and FAQs.
   - The user highlighted the tool's ability to identify outlier information and facilitate discussions with the documentation, ultimately impressing their lawyer, and said the lawyer *is going to check out NotebookLM soon*.
- **Voice Variation Ventures: Vocal Variety Via WAV Wizardry**: A user asked about the best way to add variation to Notebook LM voices, seeking options beyond the default NPR style, and inquired if **downloading and modifying the .wav file** in a third-party app was necessary.
   - Another user suggested that editing the speed, pitch, and other parameters of the .wav file could improve the sound but did not know of any specific humanizer apps.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1376644221238513764)** (74 messages🔥🔥): 

> `Notebook Organization, Embedding NotebookLM, Interactive Mode issues on iOS, Podcast Generator issues, Gemini Deep Research integration` 


- **Notebook Organization Needs Serious Revamp**: Users are requesting better ways to organize notebooks, such as **folders or tags**, as the current **A-Z sorting** is insufficient.
   - The inability to **search at the Notebook level** is seen as a major drawback, with browser find functions deemed "pretty primitive".
- **Embedding NotebookLM on Websites: Is It Possible?**: A user asked about the possibility of embedding **NotebookLM on a website** for others to try it out, but the response indicated that only a link can be shared, requiring **granted access to the file**.
   - No direct embedding feature exists, limiting interactive access to shared users.
- **iPhone/iPad Interactive Mode Stalls**: Several users reported issues with **Interactive Mode not starting on iPhones and iPads** when clicking the interactive mode button, and at least one user has NotebookLM pro.
   - A suggestion to try the **website version** was offered, with a reminder to hit the play button after entering interactive mode and one user finds the web version works better, regardless of being pro or not.
- **Podcast Generator Plagued with Glitches**: Users reported that the **podcast generator frequently stops working**, with a suggestion to **download the audio** instead.
   - One user complained that setting a **minimum podcast length** in the "personalize" section is ignored, especially when using **French audio**.
- **Unleash Gemini Deep Research with NotebookLM**: Users are interested in combining **Gemini Deep Research** with **NotebookLM**, with one asking about using Gemini to **surface sources and create grounded summaries** as input for NotebookLM.
   - It's confirmed that users can export Gemini Deep Research output (text and sources) and import them as sources for NotebookLM via copy-paste and that this can also be done via the new `Create` dropdown.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1376640007275548844)** (44 messages🔥): 

> `compute and comms overlap, TP+SP using Async TP, matrix multiplications, RL vs diffusion` 


- **Async TP Overlaps Compute & Comms!**: An illustrated deep-dive into how compute and comms are overlapped in **TP+SP** using **Async TP** was shared, noting that the pytorch implementation is based on a paper covered in the Eleuther ml perf reading group, see the post [here](https://x.com/vega_myhre/status/1927142595097956834).
- **Matrix Multiplication Shapes Disclosed!**: A member shared a link to a blog post about [what shapes do matrix multiplications take](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications), highlighting that memory accesses are much more expensive than compute.
   - It was clarified that **3N^2** and **2N^3** are the *number* of operations performed, and the operations do not cost the same amount of *time*: memory access takes much more.
- **RL Clashes with Diffusion: Which Reigns Supreme?**: A member inquired about whether there was interest in a document and asked whether they should make their next intellectual adventure in **RL** or **diffusion**, linking to [this resource](http://dx.doi.org/10.13140/RG.2.2.29337.53608).
- **No Gomez Zone!**: A member recommended not recommending **Kye Gomez** to newcomers, even ironically.
   - The reason was because *we don't need more people falling into his pit.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1376740518008193026)** (27 messages🔥): 

> `Quantization for ACL papers, Static n-gram heads, Buggy RWKV7, Spurious Rewards` 


- **Quantization Justification Quandaries**: A member questioned whether it is acceptable for application-oriented **ACL papers** to only show results for **4-bit quantized LLMs** due to resource limitations, while referencing [this paper](https://arxiv.org/abs/2505.17895).
- **N-gram Head Hunting**: A member inquired about attempts to add static **n-gram heads**, similar to those used in [this tweet](https://x.com/akyurekekin/status/1751987005527855410).
- **RWKV7's Regression Woes**: Members discussed that [a project](https://github.com/Benjamin-Walker/structured-linear-cdes) is using buggy **RWKV7** in **FLA**, citing multiple precision issues, in relation to [this paper](https://arxiv.org/abs/2505.17761).
- **RLVR Rewards Review**: Members shared a [Notion page](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880) and [paper link](https://arxiv.org/abs/2505.17749) on spurious rewards, rethinking training signals in **RLVR**, where one member characterized it as *another instance of entropy minimization and hyperfitting*.
- **P100s Peculiar Persistence**: A member questioned why a team is training on **P100s** in **2025**, while citing a [paper](https://arxiv.org/abs/2505.21493).


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1376915816762839122)** (1 messages): 

> `lm eval harness, gguf models, python-llama-cpp, local model evaluation` 


- **Seeking Speedy `.gguf` Evaluation in `lm eval harness`**: A member is seeking an efficient method to evaluate local `.gguf` models using `lm eval harness`, due to performance issues.
   - The user is attempting to use `python-llama-cpp` for launching a server, but experiencing extreme slowness.
- **`python-llama-cpp` Troubles with Local Models**: A user reported that using `python-llama-cpp` to evaluate local `.gguf` models is running extremely slowly.
   - No solutions were provided in the context.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1376664511091839048)** (60 messages🔥🔥): 

> `Claude 4 GitHub MCP exploit, Sesame Speech-to-Speech models, Launching AI products in Europe, Gemini Ultra access in Europe, Qwen RL results` 


- **Claude 4 plus GitHub MCP server leaks Private Repos**: A new attack uses **Claude 4** and **GitHub's MCP server** to leak data from **private GitHub repositories**, including **names**, **travel plans**, **salaries**, and **repo lists**, triggered by malicious issues.
   - Users are advised to restrict agent permissions and monitor connections; [Invariant's security scanners](https://xcancel.com/lbeurerkellner/status/1926991491735429514?s=46&t=Ld13-WcFG_cohsr6h-BdcQ) proactively identified this 'toxic flow'.
- **Sesame Crosses the Uncanny Valley of Voice**: **Sesame**'s speech-to-speech models are discussed with a link to [Sesame research](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice), complemented by a breakdown of techniques from Thomas Wolf on building these models.
   - The breakdown can be found [in this blog post](https://xcancel.com/Thom_Wolf/status/1916809878162514142) that further clarifies how recent contextual speech models and audio tokenization work.
- **Navigating European AI Product Launching**: Challenges of launching an AI product in Europe are highlighted, including **German purchase orders**, **French labor laws**, and a **mandatory 14-day refund policy** even for substantial resource usage.
   - This [EU policy](https://openai.com/policies/eu-terms-of-use/) for ChatGPT, and the short term abuse possible - so AI apps must be careful with limits - though OpenAI could opt out if they explicitly would let the user know.
- **Qwen's Spurious RL Math Rewards**: A discussion on how 'spurious rewards,' including random ones, improved the math performance of **Qwen2.5-Math-7B**, rivaling gains from ground-truth rewards, as detailed [in this thread](https://xcancel.com/StellaLisy/status/1927392717593526780).
   - The effect, unique to Qwen models, suggests **RLVR** amplifies existing 'code reasoning' patterns due to **GRPO's 'clipping bias'**, challenging traditional reward structure views.
- **Tiny LLMs Get Evaluated and Open Sourced!**: **j1-nano (600M parameters) and j1-micro (1.7B parameters)** are open-sourced as competitive reward models, trained on a single **A100 GPU** using **Self Principled Critique Tuning (SPCT)** to generate instance-specific evaluation criteria, full details [in this tweet](https://xcancel.com/leonardtang_/status/1927396709870489634).
   - **j1-micro** rivals larger models like **Claude-3-Opus** and **GPT-4o-mini**, while **j1-nano** competes with **GPT-3.5-turbo**.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1377117772491915265)** (4 messages): 

> `AI Engineer conference, Volunteer Opportunity, Speaker announcements` 


- **AI Engineer Conference needs volunteers!**: The [AI Engineer conference](https://xcancel.com/swyx/status/1927558835918545050) is seeking **30-40 volunteers** for logistical support in exchange for free admission (worth up to **$1.8k**).
   - This event, by @aiDotEngineer, is taking place **June 3-5** in SF.
- **Keynote Speakers lineup announced**: Wave 1 of keynote speakers has been announced for the AI Engineer Conference, including **Greg Brockman** (OpenAI), **Sarah Guo** (Conviction), **Simon Willison**, and others from prominent AI companies.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1376674086708576329)** (3 messages): 

> `LlamaCloud Updates, LlamaParse & AnthropicAI Sonnet 4.0, Multimodal Embedder for LlamaIndex, Enhanced Structured Output for OpenAI` 


- **LlamaCloud gets constant updates**: The team announced that they’re *constantly releasing updates and new features to **LlamaCloud***.
   - No further details were provided.
- **LlamaParse Embraces AnthropicAI's Sonnet 4.0**: **LlamaParse** now supports **AnthropicAI Sonnet 4.0** in agent and LVM modes, enabling the use of the latest LLMs when parsing complex documents for AI applications, as outlined [in this tweet](https://t.co/yNcOtjKMzm).
- **LlamaIndex shows how to build Custom Multimodal Embedder**: Learn how to build a custom multimodal embedder for LlamaIndex, as showcased [in this tweet](https://t.co/jBqn7jrMak), providing a guide on overriding LlamaIndex's default embedder for **AWS Titan Multimodal** support and integrating it with **Pinecone**.
   - The guide details how to create a custom embedding class handling both text and images.
- **OpenAI Structured Output Support**: LlamaIndex now offers enhanced structured output support for **OpenAI**, in response to their recent expansion to include new data types like **Arrays** and **Enums** and string constraint fields, as noted [in this tweet](https://t.co/SlkVrMmzRA).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1376921288840450158)** (23 messages🔥): 

> `Form Filling Agents, Workflow-based agents, Multi-modal Agents, ReactJS with LlamaIndex, HITL Workflow with React` 


- **Form Filling Agents are Coming**: Members discussed implementing a pizza ordering flow using LlamaIndex, referring to it as a '**form filling agent**' and suggesting a custom workflow with steps like `AskUserForOrderEvent` and `ConfirmUserAddressEvent`.
   - It was suggested that the tools within the workflow should write to a central storage, such as the **workflow context**, to maintain and update user data, especially when the user goes back and forth in the ordering process.
- **Workflow Agents Replace FunctionCallingAgent**: Members suggested using newer **workflow-based agents** instead of the prebuilt `FunctionCallingAgent` for more complex flows.
   - One mentioned that while `CodeAct` agent exists, `FunctionAgent` is preferable when possible, and that agents accept [multi-modal inputs](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/#multi-modal-agents) like `TextBlock`, `ImageBlock`, `AudioBlock`, and `DocumentBlock` (though not all LLMs support every block).
- **ReactJS meets LlamaIndex for HITL Workflow**: A member sought advice on integrating **ReactJS** with **LlamaIndex** for a **Human-in-the-Loop (HITL) workflow**, expressing concerns about the complexity of using `ctx.wait_for_event()` with WebSocket communication.
   - Another member suggested that `ctx.wait_for_event()` works well, referencing a [community office hours example](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing) demonstrating HITL in two flavors: responding directly and responding later after human input.
- **RelevancyEvaluator Reroutes Poor Answers**: A member implemented a workflow to query two knowledge bases using a `RetrieverRouter` and a reranker, seeking advice on how to handle unsatisfactory answers from the `RelevancyEvaluator`.
   - They shared code showing a retry mechanism using `StartEvent` but was also worried about wasting time on the same bad nodes.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1376638616834670682)** (19 messages🔥): 

> `Asynchronous tools, Isolated Mistral instance, Architect Tool, MCP Server, MCP Clients` 


- ****Asynchronous Adventures**: When to Notify?**: When creating an asynchronous tool that takes several minutes, the best approach is to send notifications providing status updates, or return a [link with instructions](https://example.com) for the user to monitor completion.
   - Returning an *embeddedresource* with change notifications could also work, but it relies heavily on client support and user behavior assumptions.
- ****Mistral Fortress**: Run Queries Locally**: To run "MPC queries" securely, use a local-first chat client like **LibreChat** or **Ollama** with a locally running **Mistral** instance, then connect your MCP servers to this chat client.
   - A member shared a [Medium post](https://medium.com/@adkomyagin/building-a-fully-local-open-source-llm-agent-for-healthcare-data-part-1-2326af866f44) that details how to set up **LibreChat+MCP**.
- ****Rule Builder Bonanza**: Architectural Tooling Sought**: A member is looking for a good “architect” tool to build rules files, which has full planning and task list features, to help non technical users.
   - No concrete recommendations were given.
- ****MCP Server Strategy**: Burger vs. McDonald's?**: In response to a business case objection about LLMs already accessing developer portals and API documentation, one member suggested answering with *'you can buy buns, ground meat, cheese, tomato, etc. and prepare a burger at home, but you still order mcdonalds don't you?'*
   - Another member suggested exposing documentation as MCP resources to reduce manual friction: *click a button and the LLM knows your business inside and out*.
- ****API-Ready Hype**: Surfing the AI Wave**: Building an MCP server can be a great opportunity to surf the hype and sell your API/SaaS as **AI-ready**, making it easier to integrate with multiple LLM clients.
   - No specific MCP clients were mentioned.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1376829638462148639)** (4 messages): 

> `MCP Inspector, Ship Lean MCP, UI issues` 


- ****MCPJam** Builds Enhanced MCP Inspector**: A member is building an enhanced **MCP inspector** called [@mcpjam/inspector](https://github.com/MCPJam/inspector) with improved UI and debugging tools like LLM chat, addressing slow development in the official repo.
   - Spinning up the inspector is easy using: `npx @mcpjam/inspector`, and the team is open to community development and feature requests.
- ****LeanMCP** Launches Vibe-Coding Platform**: A member launched a platform to build and ship remote MCPs at [https://ship.leanmcp.com](https://ship.leanmcp.com).
   - The goal is to allow people to *vibe-code* and ship MCPs. If there's good userbase and PMF, early users will get a free PRO version for an entire year.
- **LeanMCP has UI Issues**: Members are reporting **UI issues** with the LeanMCP platform.
   - Specifically, the **Discord and LinkedIn links** don't work, and the email overflows off to the side.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1376639814815973407)** (22 messages🔥): 

> `Synthesizing sentence meaning into a single token, Faiss index creation, Local LLama Interface, GPT4All version 4, Nomic burned $17M Series A funds?` 


- **Synthesizing Sentence into a Single Token**: A member is aiming to synthesize the meaning of a whole sentence into one token using models with around **1M parameters** and a **12 GB ram GPU**, intending to create a [FAISS index](https://github.com/facebookresearch/faiss) with these tokens.
- **GPT4All Utilizes Embedders**: It was noted that **GPT4All** utilizes embedders and suggested reading [hints on HuggingFace](https://huggingface.co/kalle07/embedder_collection) for help.
- **Local LLama Interface**: A member is anticipating **GPT4All version 4**, referencing a developer on LocalLLaMA creating an **LLM interface** with voice input, deep research, and image generation model compatibility, and a link to a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1kvytjg/just_enhanced_my_local_chat_interface/).
- **Nomic Still Kicking?**: A member inquired if Nomic has exhausted the **$17M** received in Series A in **2023**, speculating it might be the reason for perceived inactivity.
- **Kobold has RP Focus**: A member mentioned that [Kobold.cpp](https://github.com/facebookresearch/faiss) includes "all" features, but someone noted that **Kobold's** primary focus on RP is not their preference, preferring dedicated **LLM** or **image-only** tools.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1376979104624017509)** (2 messages): 

> `Metaprogramming in Mojo, Go generics in Mojo` 


- **Exploring Metaprogramming in Mojo Post**: A member gave a shoutout to a [blog post on metaprogramming](https://www.modular.com/blog/metaprogramming) and encouraged others to leave questions in the [associated forum thread](https://forum.modular.com/t/exploring-metaprogramming-in-mojo/1531).
   - The post was described as an *awesome read* that clarified that **Mojo** allows parameterization on values and direct control flow.
- **Mojo's Metaprogramming vs Go Generics**: A user initially understood that the **Go-generics-looking syntax** was the primary method of metaprogramming in **Mojo**.
   - However, the user was impressed to discover that **Mojo** allows for parameterization on values and direct control flow, differentiating it from **Go's generics**.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1376637789415805008)** (5 messages): 

> `Streaming parsing, Structured parsing, Magic to Pixi migration` 


- **Parsing Performance Plunge**: Members discussed different **JSON parsing strategies** in Mojo, including *streaming* and *structured parsing*.
   - One member pointed out that stopping parsing after finding the data can outperform parsing the entire JSON first.
- **DOM Parsing Downgraded?**: Members compared **on-demand parsing** to **DOM parsing**, suggesting comparing on-demand to DOM parsing isn't fair.
   - One member claimed that *if you just compare DOM parsing it loses every time*.
- **Magic to Pixi Migration Motivation**: A member circled back to a previous discussion and provided a link to the [Modular Forum](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530) regarding migrating from **Magic to Pixi**.
   - No further details were given.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

serotweak: hi everyone
  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1377027438751645776)** (2 messages): 

> `API Usage, Error 400, Token Length` 


- **API Newbie runs into Error 400**: A member reported receiving an **Error 400** message while trying to use the API and indicated they are completely new to using APIs.
   - The error message received was: *"invalid request: message must be at least 1 token long or tool results must be specified."
- **Token Length Requirements cause Error 400**: The **Error 400** indicates that the API request failed because the message was too short.
   - The error message states that the message must be at least **1 token long** or tool results must be specified.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1376987527705268264)** (2 messages): 

> `Student from East London, CS, graphics, and games development, Hardware and software aspects of technology, Building PCs as a side hustle, Learn how to code and build software` 


- **East London Student Enters the Chat!**: A sixth form student from East London introduced themself to the server.
   - They study **CS, graphics, and games development** and has a passion for both hardware and software.
- **PC Building Hustler Seeks Software Skills**: The student enjoys building PCs as a side hustle and wants to learn how to code and build software.
   - They hope to gain the skills to work in software engineering.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1376689278431527074)** (5 messages): 

> `LORA finetuning, Generation Script, Merging Weights, Adapter usage` 


- **Load Model Correctly after LORA Finetuning**: A member was trying to run the generation script after LORA finetuning but was having issues loading the model correctly when instantiating the model and checkpointer.
   - It seems that the generate script only loads for the training.MODEL_KEY as defined [here](https://github.com/pytorch/torchtune/blob/main/recipes/generate.py#L71-L74).
- **Merging Weights During LORA Finetuning**: It was noted that during LORA finetuning, the weights are merged and then saved, so while running the generate script, there is no need to make the LORA model.
   - The generate recipe assumes that the model has already been merged with the adapters, which happens for the last checkpoint during finetuning, so it can just point to that.
- **Using Adapters Directly**: The adapters can be used directly with other generation tools as highlighted in the [tutorial](https://docs.pytorch.org/torchtune/stable/tutorials/e2e_flow.html#use-your-model-in-the-wild).
   - This provides flexibility in how the finetuned model is utilized in different generation scenarios.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1376648177884856411)** (2 messages): 

> `Self Improving Vibe Coding Template, Using DSPy` 


- **Self Improving Vibe Coding Template Surfaces**: A member shared a [self improving vibe coding template](https://github.com/imranarshad/vibe_coding_template) on **Github**.
- **Argument for Using DSPy in Github Project**: A member linked a blog post as an argument for why **DSPy** should be used. 
   - The blogpost title is [Will the Model Eat Your Stack?](https://www.dbreunig.com/2025/05/27/will-the-model-eat-your-stack.html).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1376648499453497355)** (2 messages): 

> `ReAct vs Custom Code, Trajectory Nudge in LLM` 


- **ReAct Works Better Than Custom Code, claims User**: A user extensively tested [ReAct](https://x.com/ohmypk92/status/1927084222528802891?s=19) and found that it performed better than their custom code, based on their observations.
   - They attributed this to the *trajectory nudge* provided by ReAct to the LLM.
- **Trajectory Nudge Gives ReAct the Edge**: The user speculates that the *trajectory nudge* given to the LLM is the reason why ReAct performed better.
   - This nudge helps guide the LLM's reasoning process, leading to improved results compared to custom code without this guidance.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1377043655902367977)** (4 messages): 

> `tinygrad.org hyperlink, Optypes hyperlink` 


- **Optypes Hyperlink becomes 404**: The **Optypes hyperlink** on the [tinygrad.org](https://tinygrad.org) site now returns a *404 - page not found* error.
   - This is due to recent *"moving uops into a dir"* changes.
- **Tinygrad updates TinyXXX**: George Hotz shared a link to the [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub repository.
   - This was followed by a notice that [PR#27](https://github.com/tinygrad/tinyxxx/pull/27) was merged.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1376719936197623849)** (2 messages): 

> `Future Cohorts, Course Scheduling` 


- **Future LLM Agent Course Cohorts Discussed**: A member inquired about the availability of Summer '25 or Fall '25 cohorts for the LLM Agents course.
   - Another member responded that there would be no Summer '25 course and that Fall '25 had not been confirmed yet but would start mid-August if it were to happen.
- **Upcoming Course Timing Speculations**: The expected start time for the Fall 2025 cohort was speculated to be in mid-August, pending confirmation.
   - The discussion clarified that while no Summer 2025 course is planned, the possibility of a Fall 2025 cohort remains uncertain.


  

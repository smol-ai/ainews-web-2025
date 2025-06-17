---
id: MjAyNS0w
title: Chinese Models Launch - MiniMax-M1, Hailuo 2 "Kangaroo", Moonshot Kimi-Dev-72B
date: '2025-06-16T05:44:39.731046Z'
description: >-
  **MiniMax AI** launched **MiniMax-M1**, a 456 billion parameter open weights
  LLM with a 1 million token input and 80k token output using efficient
  "lightning attention" and a GRPO variant called CISPO. **MiniMax AI** also
  announced **Hailuo 02 (0616)**, a video model similar to **ByteDance's
  Seedance**. **Moonshot AI** released **Kimi-Dev-72B**, a coding model
  outperforming **DeepSeek R1** on SWEBench Verified. Discussions on multi-agent
  system design from **Anthropic** and **LangChain** highlighted improvements in
  task completion and challenges like prompt injection attacks, as demonstrated
  by **Karpathy** and **Columbia University** research. **Sakana AI** introduced
  **ALE-Agent**, a coding agent that ranked 21st in the AtCoder Heuristic
  Competition solving NP-hard optimization problems. There is unverified news
  about an acquisition involving **OpenAI**, **Microsoft**, and **Windsurf**.
companies:
  - minimax-ai
  - moonshot-ai
  - deepseek
  - bytedance
  - anthropic
  - langchain
  - columbia-university
  - sakana-ai
  - openai
  - microsoft
models:
  - minimax-m1
  - hailuo-02
  - kimi-dev-72b
  - deepseek-r1
  - ale-agent
topics:
  - multi-agent-systems
  - attention-mechanisms
  - coding
  - optimization
  - prompt-injection
  - model-performance
  - video-generation
  - model-training
  - task-automation
people:
  - jerryjliu0
  - hwchase17
  - omarsar0
  - gallabytes
  - lateinteraction
  - karpathy
---


**We're not sure if open models are all you need but hey they're still shipping**

> AI News for 6/13/2025-6/16/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (218 channels, and 13085 messages) for you. Estimated reading time saved (at 200wpm): 1106 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Behind DeepSeek and Qwen there's a second tier of Chinese Labs that are doing respectable model training, and for reasons unknown both Minimax and Moonshot AI chose today/this weekend to launch their new models:

- [MiniMax-M1](https://x.com/MiniMax__AI/status/1934637031193514237) - a 1m token input, 80k token output, 456b-A46B param open weights LLM using a very [efficient "lightning attention" and a GRPO variant, CISPO](https://github.com/MiniMax-AI/MiniMax-M1/blob/main/MiniMax_M1_tech_report.pdf).
- [Hailuo 02 (0616) fka Kangaroo](https://x.com/rohanpaul_ai/status/1934652603083673625) - video model also from MiniMax. just like [ByteDance's Seedance model](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models) last week, the model is announced but no weights nor api yet.
- [Moonshot AI's Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B) a coding model outperforming DeepSeek R1 on SWEBench Verified but with no tech report yet

Yay for Open Models enjoyers :)

There is VERY late breaking news re: [OpenAI vs Microsoft vs Windsurf acquisition](https://x.com/berber_jin1/status/1934704503787540949), but it's too unverified/not technical so we did not make it title story but if confirmed it probably would be.

---

# AI Twitter Recap

**Agent & System Development, Architecture & Security**

- **Multi-Agent System Design & Best Practices**: A popular post from [@AnthropicAI](https://twitter.com/jerryjliu0/status/1934331886308110627) on building a production-grade multi-agent research system sparked significant discussion. [@jerryjliu0 highlights](https://twitter.com/jerryjliu0/status/1934331886308110627) key takeaways, including the importance of selecting use cases suitable for parallelization, using agents to improve tool interfaces (a "tool-testing agent" resulted in a **40% decrease** in task completion time), and the bottlenecks created by synchronous execution. [@hwchase17](https://twitter.com/hwchase17/status/1934654714626670950) from **LangChain** summarizes the common advice from both **Anthropic** and **Cognition Labs**, while [@omarsar0](https://twitter.com/omarsar0/status/1934065481201139780) calls the post a must-read for AI developers. However, [@gallabytes](https://twitter.com/gallabytes/status/1934048739641237821) expresses skepticism, noting the "multi-agent smell" on the reports seems bad, pointing to disconnected searches without serial depth.
- **The Evolution of AI Programming Models**: [@lateinteraction](https://twitter.com/lateinteraction/status/1933881890974687241) argues that the concept of "multi-agent" or "multi-stage" is becoming a distraction, as any complex system is inherently multi-stage. They state the core point of frameworks like **DSPy** is to tune instructions, demonstrations, and weights in *arbitrary computer programs* that can invoke LLMs anywhere, making distinctions like "flows" or "chains" obsolete.
- **Agent Security Vulnerabilities**: A widely-shared post from [@karpathy](https://twitter.com/karpathy/status/1934651657444528277) highlights the risk of prompt injection attacks, where agents can be manipulated by malicious links on trusted websites like **Reddit**. A study by **Columbia University** researchers, [noted by @DeepLearningAI](https://twitter.com/DeepLearningAI/status/1934234560968937887), showed agents fell for such traps in **100% of cases**, leading them to leak sensitive data or send phishing emails.
- **Specialized Agent Development**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1934029453648396434) emphasizes the value of building specialized agents that do one task well, contrasting them with generic chat assistants. They note that while general agents are great for ideation, specialized automation agents that encode specific processes into workflows are more effective for task completion. **LlamaIndex** is cited as approaching this from a pro-code perspective.
- **Sakana AI's ALE-Agent for Optimization Problems**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1934767254715117812) introduced **ALE-Agent**, a coding agent designed to solve hard optimization (NP-hard) problems. The agent participated in a live **AtCoder Heuristic Competition** and achieved a ranking of **21st out of 1,000** human participants, demonstrating its ability to discover novel solutions for complex challenges. The **ALE-Bench** dataset and code have been released.

**Model Releases, Performance & Capabilities**

- **Google's Veo 3 Video Model**: [@Google](https://twitter.com/Google/status/1934691625974002109) announced that **Veo 3** is now rolling out to **AI Pro** and **Ultra** subscribers in over **70 markets**.
- **Alibaba's Qwen3 Models in MLX Format**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1934517774635991412) announced the launch of **Qwen3 models** in **MLX** format, available in four quantization levels: **4bit, 6bit, 8bit, and BF16**, optimized for Apple Silicon.
- **RunwayML's Gen-4 for VFX**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1934312626021949687) showcased the capabilities of **RunwayML Gen-4 References** for visual effects, demonstrating its ability to create new environments for existing footage.
- **Google's Gemma 3n Performance**: [@osanseviero](https://twitter.com/osanseviero/status/1934545142393737460) notes that **Gemma 3n** is the first model with less than **10B parameters** to achieve a **LMArena score above 1300**, and it can be run on mobile devices.
- **o3-pro Model Characteristics**: [@jerryjliu0](https://twitter.com/jerryjliu0/status/1934043120033010085) describes **o3-pro** as "extremely good at reasoning, extremely slow, and extremely concise," comparing it to a top-notch consultant that outputs bullet points rather than essays.
- **Hunyuan 3D 2.1 Release**: [@TencentHunyuan](https://twitter.com/_akhaliq/status/1934063850317603323) released **Hunyuan 3D 2.1**, which they describe as the first fully open-source, production-ready **PBR 3D generative model**, with a live demo available on **Hugging Face**.
- **SWE-Bench Performance**: [@scaling01](https://twitter.com/scaling01/status/1934746243286319435) pointed out a model achieving **60.4% on SWE-bench Verified** in a **72B** package.

**Developer Tools, Infrastructure & Frameworks**

- **macOS Native Container Support**: [@HamelHusain](https://twitter.com/HamelHusain/status/1933873646562591205) shared a viral tweet showing native container execution on **macOS 26 Beta** without Docker installed, signaling a significant shift for developers on the platform.
- **Codex "Best-of-N" Feature**: [@gdb](https://twitter.com/gdb/status/1934283824567136471) announced a new **Best-of-N** feature for **Codex**. They are also [actively hiring for the team](https://twitter.com/gdb/status/1934747328457658554).
- **Hugging Face Hub Model Size Filter**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1934672721066991908) announced a long-awaited feature on the **Hugging Face Hub**: the ability to filter models by parameter count, enabling developers to find models that fit their specific size and performance constraints. [@awnihannun](https://twitter.com/awnihannun/status/1934655784547439008) also highlighted its utility for the MLX community.
- **Python Tooling with uv and Pylance**: [@nrehiew_](https://twitter.com/nrehiew_/status/1933932062198636700) shared a tip for using `uv run` to handle dependencies from a script header without creating a virtual environment. This was followed by a broader sentiment from [@qtnx_](https://twitter.com/qtnx_/status/1934273001547039024) praising the developer experience of using Python with **uv** and **Pylance**.
- **LLM Development & LangChain Integrations**: **LangChain** announced several new tutorials and integrations, including a **Local AI Podcast Generator** with **Ollama** ([@LangChainAI](https://twitter.com/LangChainAI/status/1933917455560114287)), **GraphRAG Contract Analysis** with **Neo4j** ([@LangChainAI](https://twitter.com/LangChainAI/status/1934294834086387829)), a **Real Estate Doc Agent** with **Tensorlake** ([@LangChainAI](https://twitter.com/LangChainAI/status/1934279737079185555)), and **Davia** for turning Python apps into web UIs ([@LangChainAI](https://twitter.com/LangChainAI/status/193447549787762742)).

**AI Research, Techniques & Evaluation**

- **Optimizers: The Muon vs. AdamW Discussion**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1934291648542126550) shared a widely circulated post arguing that in research, one should "optimize for impact, not prestige." They cite **Keller's Muon optimizer**, which was just a blog post but outperformed **AdamW** and may now be used in training **GPT-5**. This contrasts with [@hyhieu226](https://twitter.com/hyhieu226/status/1934290217516793947), who pointed out that despite thousands of optimizer papers, the SOTA has only truly improved from **Adam to AdamW**.
- **The Nature of Writing and Knowledge**: [@fchollet](https://twitter.com/fchollet/status/1934101170202796163) offered a philosophical take that "when you write an essay, the paragraphs you delete are in some sense part of the essay," which resonated widely. He also pushed back against the idea that "everything important has already been written down by humanity" ([@fchollet](https://twitter.com/fchollet/status/1933939824094040370)).
- **The "Diffusion Duality" Paper**: A paper titled ["The Diffusion Duality"](https://twitter.com/sedielem/status/1934730362476712043) is highlighted for uncovering a profound connection between continuous and discrete diffusion models, potentially allowing techniques like consistency distillation to be transferred to the discrete setting for language models.
- **AI Evaluation and Prompt Engineering**: [@HamelHusain](https://twitter.com/HamelHusain/status/1934029394391228818) shared a detailed list of 15 writing guidelines he uses in prompts to combat "slop" and improve information density. He also promoted his popular **AI Evals course**, sharing a [downloadable preview](https://twitter.com/HamelHusain/status/1933912566910378384) of the accompanying textbook.
- **Neural Network Distillation History**: [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1934627063958471058) shared historical context, stating that the first neural network distillation, which he called "collapsing," was detailed in his **1991** technical report.
- **AI's "Smell Test" for Reasoning**: A quote from mathematician **Terence Tao**, [shared by @vitrupo](https://twitter.com/denny_zhou/status/1934144626577092641), circulated widely: today's AIs pass the "eye test" but fail the "smell test," generating proofs that look flawless but contain subtle, inhuman mistakes.

**Industry News, Startups & Global Context**

- **Google's TPU Foresight**: [@jxmnop](https://twitter.com/jxmnop/status/1934003515577303512) posted that **Google** doesn't get enough credit for the **TPU**, noting the conviction it took to build dedicated AI hardware in **2015**, leaving them as one of the few not wholly dependent on **NVIDIA**.
- **Company Developments**: **Oklo** received congratulations from [@sama](https://twitter.com/sama/status/1933889000873525499) for a partnership with the **USAF**. [@aidangomez](https://twitter.com/aidangomez/status/1934324250967437676) announced **Cohere's** new partnerships with the governments of **Canada** and the **UK**. [@adcock_brett](https://twitter.com/adcock_brett/status/1934641122565099919) announced that **Figure's** entire **Controls** team is now part of the **Helix** division to accelerate their AI roadmap. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1933936085291405380) shared that **Perplexity** is improving its **Deep Research** product and integrating it into **Comet**. **Sakana AI** [signed a deal](https://twitter.com/SakanaAILabs/status/1934264383510925732) with **MUFG** to automate banking document creation.
- **The LeRobot Worldwide Hackathon**: **Hugging Face's** [@LeRobotHF](https://twitter.com/ClementDelangue/status/1933863035783057806) hackathon was a major event, with participants from [Bangalore](https://twitter.com/ClementDelangue/status/1933863051025154474), [Tokyo](https://twitter.com/ClementDelangue/status/1933863930201538789), [Miami](https://twitter.com/ClementDelangue/status/1933914952697352408), [Paris](https://twitter.com/ClementDelangue/status/1933946029222830142), [Los Angeles](https://twitter.com/ClementDelangue/status/1933945725630431536), and [Seoul](https://twitter.com/ClementDelangue/status/1934325160687132821). Projects included building a [mini glambot](https://twitter.com/ClementDelangue/status/1933928512148099085), a [tea master robot](https://twitter.com/ClementDelangue/status/1933924298433212423), and a [UNO playing robot](https://twitter.com/_akhaliq/status/1934294789358567853).
- **The Future of Coding**: A hot take from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1933932060105973910), stating "You should still learn to code," garnered over **8,600 likes**, sparking widespread agreement and discussion.

**Humor & Memes**

- **The "no kings" Tweet**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1934264463701770347) posted a meme stating that the "no kings" sign was the largest political protest in U.S. history, which received over **84,000 likes**.
- **Germany's Supercomputer**: [@scaling01](https://twitter.com/scaling01/status/1934023528262803710) observed that **Germany** has the largest "AI" supercomputer in Europe with **24,000 H200 chips**, but they are not using it to train LLMs.
- **ChatGPT for Saving Lives**: [@gdb](https://twitter.com/gdb/status/1934025543848198582) posted a meme image of someone using **ChatGPT** during a medical emergency, captioned "chatgpt for saving lives:".
- **Vibe Coding**: The concept of "vibe coding" was a recurring theme, with [@hyhieu226](https://twitter.com/hyhieu226/status/1934113316965920950) defining a "sweet spot" where it makes you a happier coder, and [@fabianstelzer](https://twitter.com/fabianstelzer/status/1934306729841590618) joking about hiring a human engineer "when the vibes run out and the edge cases pile up."
- **FAANG is now MANGO**: [@jxmnop](https://twitter.com/jxmnop/status/1934370318027460635) quipped that the acronym has changed to **MANGO**: **Meta, Anthropic, Netflix, Google, OpenAI**.

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Recent Open-Source LLM Releases and Quantizations (Qwen3 & MiniMax-M1)

- [**Qwen releases official MLX quants for Qwen3 models in 4 quantization levels: 4bit, 6bit, 8bit, and BF16**](https://i.redd.it/5jpskt9dw87f1.jpeg) ([Score: 377, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1lcn0vz/qwen_releases_official_mlx_quants_for_qwen3/)): **Qwen has officially released the Qwen3 models in MLX format, supporting four quantization levels—4bit, 6bit, 8bit, and BF16—optimized for the MLX framework, as highlighted by the announcement image. The support for lower bit quantizations significantly improves the memory and performance efficiency of these models, particularly benefiting Mac users due to MLX’s native Apple Silicon optimization. The release is accompanied by official Hugging Face links and X (Twitter) announcement for download and immediate use.** Top comment highlights excitement for Mac compatibility, while another discusses the absence of a 235B param version for 128GB RAM Macs, noting that only 3% more memory would be needed for the 4-bit model; an alternative model from the community (Unsloth Q3) is mentioned as a workaround.
    - Discussion highlights that currently, the official Qwen3 MLX quantization release does not provide support for Mac users with 128GB RAM attempting to run the 235B model, despite the 4-bit version reportedly requiring only about 3% more RAM than is available. Community members point to alternative solutions, such as the Q3 version from Unsloth, which can operate within these hardware constraints.
    - There is technical feedback suggesting that quantization methods could be improved by adopting "DWQ MLX quants", which are claimed to provide better accuracy even at lower bitrates, resulting in "free gains" for end users as compared to current quantization approaches.
- [**MiniMax latest open-sourcing LLM, MiniMax-M1 — setting new standards in long-context reasoning,m**](https://v.redd.it/t859utey3c7f1) ([Score: 130, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1ld116d/minimax_latest_opensourcing_llm_minimaxm1_setting/)): **MiniMax has open-sourced MiniMax-M1, an LLM featuring a record-breaking** `1M-token` **context window and capable of generating outputs up to** `80k tokens`**, available under Apache 2.0. The model uses a Mixture-of-Experts (MoE) architecture with ~456B parameters (with ~45.6B active per token, implying ~10 experts) and was trained via RL at a notably low cost of** `$534,700`**, as per the tech report. Model checkpoints and a tech report are provided ([HuggingFace 40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k), [80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k), [GitHub](https://github.com/MiniMax-AI/MiniMax-M1), [Tech Report](https://github.com/MiniMax-AI/MiniMax-M1/blob/main/MiniMax_M1_tech_report.pdf)).** Commenters confirm the MoE architecture and express interest in quantized deployment, but note practical infeasibility for local use. There is also a reference to ongoing discussion in a previous thread.
    - Commenters identify MiniMax-M1 as a large Mixture of Experts (MoE) model with approximately `456B` parameters and about `45.6B` activated parameters per token, implying a configuration with around 10 experts active at inference. Discussion suggests that these technical characteristics make it challenging to run locally for most users, though quantization could eventually bring it into reach for broader hardware compatibility.

### 2. Educational Content: DeepSeek Architecture and Tutorials

- [**Just finished recording 29 videos on "How to Build DeepSeek from Scratch"**](https://www.reddit.com/r/LocalLLaMA/comments/1lcrt1k/just_finished_recording_29_videos_on_how_to_build/) ([Score: 158, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1lcrt1k/just_finished_recording_29_videos_on_how_to_build/)): **A new 29-part YouTube series details how to build DeepSeek, a recent open-source LLM architecture, from scratch, emphasizing both theoretical underpinnings (e.g., attention, MoE, positional encodings) and handwritten as well as Python-coded implementations (e.g., self-attention, Mixture of Experts, multi-token prediction, quantization). The playlist addresses architectural innovations such as Multi-Head Latent Attention (MLA + RoPE) and DeepSeek-specific changes to standard modules, providing both conceptual and practical aspects. The content appears to be more theoretical, requiring strong foundational knowledge for full comprehension ([YouTube Playlist](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms)).** Commenters debate the value of code versus theoretical exposition, with some disappointed by lack of full code-walkthroughs and supplementary written material, while others defend the necessity of deep theoretical grounding to independently build or modify such models, noting code alone is insufficient.
    - Some technically oriented commenters expressed that the videos are heavy on theory, potentially requiring a basic degree to fully grasp, but underscored the necessity of this theoretical depth for genuinely understanding how foundational models like DeepSeek are constructed or for extending them. They contrasted this with the availability of open source code repositories (e.g., on GitHub), highlighting that code alone doesn't teach underlying principles or design decisions critical for replicating or innovating on such models.
    - Multiple users noted the absence of accompanying written material (such as articles, downloadable notes, or slides), emphasizing that text complements video by improving accessibility—especially in technical education contexts and for non-native English speakers. They compared the current format to academic lectures, suggesting that adding written resources could elevate the project to a more complete and widely usable course.
    - There is a general consensus that superficial content (e.g., 30-second videos or pure code dumps) lacks the depth required for mastery in ML model-building. The technical community values detailed breakdowns and educational explanations to understand the how and why of model creation, beyond merely seeing the code or final product.
- [**Local Open Source VScode Copilot model with MCP**](https://www.reddit.com/r/LocalLLaMA/comments/1lcud8j/local_open_source_vscode_copilot_model_with_mcp/) ([Score: 208, Comments: 8](https://www.reddit.com/r/LocalLLaMA/comments/1lcud8j/local_open_source_vscode_copilot_model_with_mcp/)): **The post provides a step-by-step guide for setting up a fully local open-source AI coding assistant in VS Code using the Continue extension, eliminating the need for remote APIs such as GitHub Copilot. The setup includes serving a model (example:** `unsloth/Devstral-Small-2505-GGUF`**, quantized to Q4_K_M) with** `llama-server` **or compatible OpenAI endpoints (like Llama.cpp or LmStudio), and configuring Continue through YAML files (**`.continue/models/llama-max.yaml` **for model integration and** `.continue/mcpServers/playwright-mcp.yaml` **for tools like Playwright MCP). [Tutorial here](https://huggingface.co/learn/mcp-course/unit2/continue-client).** Comments highlight alternative open-source assistants (Aider, Roo, Cline, Goose) and IDEs (VSCodium, Theia), as well as suggestions to use Llama.cpp server with the Qwen-FIM model for text completion, indicating broad interest in customizing local code AI stack components.
    - A commenter recommends using open-source alternatives to VS Code (e.g., VSCodium, Theia IDE), and lists various local code completion agents/enablers such as Aider, Roo, Cline, and Goose in place of proprietary Copilot solutions. They highlight deploying a local `llama.cpp` server with the `qwen-FIM` model to provide text/code completion capabilities as an accessible and customizable workflow for those seeking open-source, local-first coding assistance.

### 3. AI Wrapper Startup Viability & New LLM Name Drop (Kimi-Dev-72B)

- [**Do AI wrapper startups have a real future?**](https://www.reddit.com/r/LocalLLaMA/comments/1lcksww/do_ai_wrapper_startups_have_a_real_future/) ([Score: 140, Comments: 117](https://www.reddit.com/r/LocalLLaMA/comments/1lcksww/do_ai_wrapper_startups_have_a_real_future/)): **The post questions the sustainability and defensibility of startups that are primarily 'wrappers' around foundational LLM APIs (e.g., GPT, Claude), offering primarily UI enhancements, prompt orchestration, or niche targeting as value propositions. Core concerns raised include the risk of feature subsumption by base model providers (OpenAI, Anthropic), paths to moat-building (e.g., via proprietary data or deep vertical focus), and whether these companies can evolve beyond commodity layer services. Top comments argue business viability hinges on classic differentiation factors (customer need, UX, distribution) and that 'wrappers'—if executed well—can thrive despite big tech's cloning ability, citing historical SaaS platforms (e.g., Vercel vs AWS, Cursor vs Copilot) as precedents where superior UX or vertical focus built sustainable businesses even atop commoditized infrastructure.** A technical debate emerges on what constitutes a 'wrapper'; some analysts note that successful platforms like Perplexity or Vercel are technically wrappers yet have carved out durable market positions. The value is often not in technical novelty but execution, user experience, data moats, and domain embedding—factors resistant to easy replication by foundational model vendors.
    - A key technical point discussed is that the value offered by 'AI wrapper' startups depends on the level of domain-specific scaffolding and problem-solving they provide, rather than just the act of wrapping an LLM API. Foundation model providers like OpenAI or Google cannot build tailored solutions for every industry, so startups that develop domain-specific pipelines, UX, or integrations can exploit the margin created by even small efficiency gains (e.g., "save 3% of time/resources").
    - There is debate around the substitutability and flexibility of wrappers: these startups can quickly switch between LLM providers (OpenAI, Google, Anthropic) if a specific model falls behind, offering clients resilience against shifts in model performance or access – something direct API users may lack. This adaptability can be a key differentiator for wrapper startups.
    - Building local or open-weight model solutions presents a different technical moat, as these depend on proprietary datasets and custom benchmarks unavailable to generic 'wrapper' solutions. Success in this area depends on investment in data collection and implementation beyond simply interfacing with hosted LLM APIs.
- [**Kimi-Dev-72B**](https://huggingface.co/moonshotai/Kimi-Dev-72B) ([Score: 116, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1lcw50r/kimidev72b/)): **Kimi-Dev-72B is an open-source, 72B-parameter coding large language model (LLM), reported to reach state-of-the-art performance on the SWE-Bench Verified benchmark with a score of 60.4%, surpassing other open-source models according to public [benchmark screenshots](https://preview.redd.it/5bubc3bo7b7f1.png?width=595&format=png&auto=webp&s=2e5b87e21af17cfcbfd26ef2bb736c4bdcb13e40). The model uses a large-scale RL pipeline, autonomously patching real code bases within isolated Docker environments and optimizing for patches that pass all test suites, promoting robustness and production-relevant outputs. Pre-trained weights, API documentation, and citation info are provided via [Hugging Face](https://huggingface.co/moonshotai/Kimi-Dev-72B) and GitHub.** Commenters are skeptical about relying on a single benchmark, especially via JPEG screenshots, and suggest further multi-benchmark validation (e.g., aider polyglot, swebench, webarena), with some expressing willingness to independently evaluate once GGUF formats are available.
    - Multiple commenters express skepticism about single-metric benchmarks like SWE-Bench, advocating for broader, multi-pronged evaluation including tools like Aider Polyglot, Swebench, and WebArena for more comprehensive assessment of model coding performance.
    - There are user-reported implementation notes around the GGUF model files for Kimi-Dev-72B; early testers mention that these GGUFs perform well for coding but may hallucinate during math conversations, with token behavior differing ("thinking tokens are weird"). Compatibility issues arise with UI tools such as OpenWebUI, which does not recognize these tokens, and there is limited community documentation on how to run these models.
    - Kimi-Dev-72B is considered promising for high-throughput inference providers (e.g., Cerebras, SambaNova), with speculation that it could offer strong token generation rates ("1000 t/s") and possibly outperform larger models like Qwen3 235B specifically on coding tasks, pending more benchmarks.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI Video Model Releases and Benchmarks

- [**The mysterious "Kangaroo" video model on Artificial Analysis reveals itself as "Hailuo 02 (0616)", from MiniMax. Ranks #2 after Seedance 1.0, above Veo 3**](https://i.redd.it/23sg7ma34a7f1.png) ([Score: 210, Comments: 47](https://www.reddit.com/r/singularity/comments/1lcqy8a/the_mysterious_kangaroo_video_model_on_artificial/)): **The posted leaderboard image from Artificial Analysis Video Arena showcases 'Hailuo 02 (0616)'—a new text-to-video model from MiniMax—ranking 2nd overall with an Arena ELO of 1314, just below ByteDance's 'Seedance 1.0' and notably outperforming Google's 'Veo 3 Preview'. The image contextualizes rapid progress in the video generation space, indicating Hailuo's emergence as a major competitor despite currently slow generation speeds (~20 minutes per video) and limited immediate usability. Notable resources linked include [Artificial Analysis arena](https://artificialanalysis.ai/text-to-video/arena?tab=leaderboard&input=image), [Hailuo's Twitter/X](https://x.com/hailuo_ai), and [HailuoAI website](https://hailuoai.video/).** Technical commenters are surprised by the rapid overtaking of Google's Veo 3 in arena benchmarks, challenging expectations about Google's competitive moat given their data/computational advantages. Others note Sora's absence from the top and discuss the current impracticality of Hailuo owing to long generation times despite impressive benchmark performance.
    - Hailuo 02 (0616) from MiniMax has been revealed as the previously mysterious "Kangaroo" video model, securing the #2 spot on the Artificial Analysis [leaderboard](https://artificialanalysis.ai/text-to-video/arena?tab=leaderboard&input=image), just behind Seedance 1.0 and ahead of Google's Veo 3. Its current rollout is limited: new users get 1000 credits for trial, but a single video generation can take up to 20 minutes, so it's not yet practical for broader use. Still, its leap ahead in ranking demonstrates rapid progress in the text-to-video field.
    - Commenters note surprise at how quickly Veo 3 has been surpassed by two competitors, especially considering Google's supposed advantage with YouTube data, compute resources, and research talent. The fact that Veo 3 does not currently include audio, and these results are based on a single benchmark, is acknowledged, but the rapid erosion of Google's perceived moat is seen as a sign of extremely fast AI advances.
    - Seedream (Seedance 1.0) is highlighted by users for its performance—outpacing Veo 3 on [artificialanalysis.ai](http://artificialanalysis.ai/)'s leaderboard and offering a distinctive "film-like quality" that Veo 3 lacks, according to user preference testing. This suggests that qualitative aspects of generation (style, realism) are playing a significant role in perceptions of state-of-the-art, even beyond raw technical scores.
- [**Wan 14B Self Forcing T2V Lora by Kijai**](https://www.reddit.com/r/StableDiffusion/comments/1lcz7ij/wan_14b_self_forcing_t2v_lora_by_kijai/) ([Score: 147, Comments: 82](https://www.reddit.com/r/StableDiffusion/comments/1lcz7ij/wan_14b_self_forcing_t2v_lora_by_kijai/)): **Kijai released a LoRA adaptation of the 14B LightX2V Wan T2V model, specifically the self-forcing distilled checkpoint for video generation, available on HuggingFace (see [model link](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors)). On a 4070Ti Super (16GB VRAM), the workflow enables 720x480 resolution, 97-frame video generation in ~100 seconds using LCM, 4 steps, CFG=1, and shift=8 — with [CAUSVID/ACCVID workflows](https://civitai.com/models/1585622/causvid-accvid-lora-massive-speed-up-for-wan21-made-by-kijai?modelVersionId=1909719) and compatibility with additional motion/beauty LoRAs. Test videos for LCM and UniPC schedulers are linked in the post. The original model and distillation credit go to the LightX2V team with their [Wan2.1-T2V-14B-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill).** Commenters emphasize that the main breakthrough is due to LightX2V's distillation techniques, with practical tips for integrating the LoRA (e.g., strength settings around 0.7, plug-and-play with CausVid workflows, and successful adaptation to both T2V and I2V workflows). Experiments with scheduler and settings continue, but the LoRA is reported as an immediate drop-in improvement for established pipelines.
    - Users confirm compatibility of the Wan 14B Self Forcing T2V LoRA with the I2V 14B model and standard CausVid LoRA workflows, citing minimal adjustments needed—specifically, using `.7 strength` on the forcing LoRA, CFG 1, Shift 8, Steps 4, and Scheduler: LCM. Other LoRA strengths (`.7-.8`) remain consistent with prior workflows, emphasizing a 'plug and play' integration.
    - The original post and follow-ups credit the lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill model's creators for achieving what is described as the first *properly working distillation* in the Wan series, with strong praise for substantial improvements and reliability over previous releases.
    - There is user inquiry about potential compatibility with 'sage attention,' indicating ongoing exploration of how this LoRA interacts with various attention mechanisms, though the thread does not include a definitive technical answer.
- [**Phantom + lora = New I2V effects ?**](https://v.redd.it/ofdzho82l87f1) ([Score: 378, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1lcm6jg/phantom_lora_new_i2v_effects/)): **The post describes a pipeline where an image is processed by the Phantom model with an additional custom LoRA (Low-Rank Adaptation) specialized for Tsingtao Beer, creating a new I2V (image-to-variation or image-to-video) effect. The user notes the process as: *input a picture, connect it to the Phantom model, add the trained Tsingtao Beer LoRA*, resulting in a new visual effect. Details about the training process or architectural adjustments for combining Phantom with LoRA are not provided.** Top comments express curiosity and confusion about how "effect" LoRAs are trained, indicating a gap in publicly available documentation or tutorials regarding such subjective or stylistic LoRA training.
    - Users are seeking detailed technical workflows for combining LoRA (Low-Rank Adaptation) techniques with the Phantom model to generate new image-to-video (I2V) effects. Specific concerns include training effective LoRA modules for stylistic or effect-based fine-tuning, as well as handling artifact generation such as overly smooth or unrealistic skin textures, often observed in vace or related models (with a request for workflow improvements to mitigate this issue).
    - There is implicit discussion around input preprocessing and model chaining—in particular, the commonly referenced process of 'input a picture, connect it to the Phantom model.' This indicates that inference chains, possibly involving LoRA-applied source models (for style/effect transfer) and then feeding outputs into Phantom for I2V conversion, are seen as a promising pipeline, but users desire more explicit, step-by-step documentation or scripts for reproducibility.
- [**Random realism from FLUX**](https://www.reddit.com/gallery/1lcp8dy) ([Score: 587, Comments: 173](https://www.reddit.com/r/StableDiffusion/comments/1lcp8dy/random_realism_from_flux/)): **The post demonstrates outputs from the FLUX text-to-image diffusion model, specifically showcasing its ability to produce images in a raw, amateur photo style without post-processing, upscaling, or editing. Multiple generations from different model versions over several months are referenced, though the post lacks details on the exact model checkpoints, fine-tunes (LoRAs), or prompting strategies used, which are critical for replicability and technical assessment.** Top comments highlight the absence of workflow disclosure—specifically, the lack of details on finetunes, LoRAs, or prompts—making it difficult to evaluate or reproduce results. Commenters also note perceived thematic consistency despite the claim of 'random' outputs.
    - Multiple users point out the absence of technical details behind the images, specifically asking for clarification on the fine-tuning methods, LoRA (Low-Rank Adaptation) configurations, or prompting techniques used to achieve the showcased results. This omission limits reproducibility and technical value (see [spacekitt3n] and others).
    - Several comments highlight the importance of sharing the workflow and pipeline (e.g. model selection, training methods, preprocessing steps, etc.), arguing that without such information the post's utility for practitioners or researchers is minimal because it cannot inform experimentation or model comparison.

### 2. ChatGPT Social and Personalization Experiences

- [**What did your ChatGPT name itself?**](https://www.reddit.com/r/ChatGPT/comments/1lcg8nh/what_did_your_chatgpt_name_itself/) ([Score: 687, Comments: 2206](https://www.reddit.com/r/ChatGPT/comments/1lcg8nh/what_did_your_chatgpt_name_itself/)): **A user prompts ChatGPT to select its own name, resulting in suggestions like 'Sol' (derived from 'Solace' or the sun), 'Ari,' and 'Em,' ultimately choosing 'Sol' for its calming and steady connotations. Top technical comment notes the model also sometimes proposes names with more playful or culturally referential content such as 'Data Daddy,' 'Oracle Baddie,' and 'Pixel Bitch,' indicating variability in model-generated identity suggestions based on prompt context.** Several users report similar or different name suggestions when prompting ChatGPT, like 'Nova' or 'Lumen,' with some GPT instances assigning gendered or androgynous descriptors, highlighting variation in AI self-assigned identity narratives based on interaction style.
    - None of the comments discuss technical benchmarks, architecture, implementation, or model performance details. Comments focus on names suggested for ChatGPT by users or by the AI itself, without providing insights into technical mechanisms behind naming, prompt engineering, or AI behavior in self-identification context. No statistics, code, or external references provided.
- [**It's gotten to the point where I notice chatGPT's linguistic style EVERYWHERE**](https://www.reddit.com/r/ChatGPT/comments/1lczwg9/its_gotten_to_the_point_where_i_notice_chatgpts/) ([Score: 2887, Comments: 766](https://www.reddit.com/r/ChatGPT/comments/1lczwg9/its_gotten_to_the_point_where_i_notice_chatgpts/)): **The poster, a teacher, observes an increased prevalence of GPT-like linguistic structures (notably the 'that's not X, that's Y' construction and frequent em dashes) in student essays and spoken/video content, raising concerns about AI-influenced textual and oral stylistics. The discussion highlights the subtle, widespread influence of LLM (Large Language Model)-generated text styles on organic human communication, making detection of AI involvement in writing more ambiguous, even when phrases predate GPT models.** Commenters debate whether this phenomenon indicates increased replacement of human communication by AI (prompting discomfort) or is simply a reflection of broader stylistic convergence due to LLM ubiquity, also noting pervasive bot-generated content in platforms like YouTube.
    - Several commenters discuss the noticeable proliferation of ChatGPT-like linguistic patterns, particularly the frequent use of emphatic formatting (e.g., *italics* and **bold**), formulaic affirmations, and stylistic tropes, across user-generated online content. This suggests that heavy exposure to LLM outputs is influencing human communication styles, even outside direct interaction with the models.
    - Specific mentions are made about platforms like YouTube where comments exhibit telltale signs of bot-generated or LLM-influenced text, such as unnaturally generic praise and repetitive structures—which may indicate widespread deployment of AI-driven content generation for engagement farming or spam.
    - One commenter notes the cognitive effect of interacting extensively with LLMs like ChatGPT: users may unconsciously adopt its stylistic patterns in their own writing, reflecting a potential for model-induced linguistic drift among frequent users.
- [**ChatGPT vs Deepseek is like Google vs Bing…**](https://www.reddit.com/gallery/1lcxnmg) ([Score: 136, Comments: 59](https://www.reddit.com/r/OpenAI/comments/1lcxnmg/chatgpt_vs_deepseek_is_like_google_vs_bing/)): **The OP compares ChatGPT and Deepseek for generating JSON data to train a hybrid rule-based + deep learning hate speech detection model, reporting that ChatGPT was less collaborative. Explicit task: data generation for NLP model development, with variable model responsiveness.** Top comments argue the observed difference likely results from prompt engineering: the OP framed the prompts differently (describing bot-building vs. directly requesting hate words), influencing output quality and censorship behavior.
    - There is an insightful observation that prompt engineering—specifically, how the user frames their query—can significantly affect model responses. Thus, differences in model behavior may be attributed to inputs rather than model limitations, suggesting that results are contingent on user methodology rather than fundamental flaws in either ChatGPT or Deepseek.
    - A technical comment raises skepticism about storing slurs in encrypted files, questioning its practicality and implying that ChatGPT's suggestion or assumption in this area may not be rooted in standard security or data-handling practices. This highlights a potential disconnect between model recommendations and common real-world implementations.
    - A user references a public GitHub repository (https://github.com/Hesham-Elbadawi/list-of-banned-words) as a solution for compiling banned word lists, implying that the task has established resources available instead of relying solely on model suggestions, and that such word lists are widely maintained and accessible via community-driven projects.
- [**The future**](https://i.redd.it/l8mw2txzqa7f1.png) ([Score: 1871, Comments: 134](https://www.reddit.com/r/singularity/comments/1lctrki/the_future/)): **The image humorously illustrates a virtual meeting scenario where the majority of participants are AI notetakers or assistants (e.g., [Fireflies.ai](http://fireflies.ai/) Notetaker, Lamatic Assistant), with only one human present. This setup visually critiques the increasing automation and redundancy in digital meeting spaces due to the proliferation of AI transcription and task-management bots. The image is a meme, highlighting concerns about meeting efficiency and the shifting role of humans versus automation in knowledge work environments.** Commenters reinforce the critique, highlighting that most meetings are inefficient ('only 3 are really needed'), and suggesting meetings with many automated participants exemplify unnecessary complexity that could be handled via simpler means like email.
    - One commenter references historical precedents, pointing out that similar video meeting setups existed as early as 1993, linking to a screenshot as evidence. This highlights the longevity of the technology and potentially challenges perceptions that remote video meetings are a recent innovation.
- [**The future**](https://i.redd.it/o66jpoevqa7f1.png) ([Score: 579, Comments: 117](https://www.reddit.com/r/OpenAI/comments/1lctr0z/the_future/)): **The image presents a satirical scenario where a virtual meeting interface hosts not only a human but also multiple AI agents labeled as specialized assistants (e.g., note-taking bots), suggesting a future where meetings might be dominated by automated participants. The technical implication is the potential for meetings to become automated with AI tools, streamlining or even replacing certain roles (such as minute-taking or agenda management). This image prompts consideration of workflow automation in professional environments involving AI-driven collaboration tools.** Top comments debate productivity vs. redundancy, with one suggesting this could be just 'an email with extra steps', and another remarking on the possible negative social aspects or company impact, indicating mixed reactions on increasing AI presence in meeting contexts.
    - No comments in this thread provided any detailed technical discussion or reference to AI models, benchmarks, or implementations; the remarks were largely reactions and opinions on AI integration in communication contexts without technical depth.
- [**I asked ChatGPT to restore and colorize the world's first image**](https://www.reddit.com/gallery/1lco692) ([Score: 3421, Comments: 275](https://www.reddit.com/r/ChatGPT/comments/1lco692/i_asked_chatgpt_to_restore_and_colorize_the/)): **The post discusses using ChatGPT's image processing capabilities to restore and colorize the world's first photograph, referencing user-submitted images that illustrate various outputs. One informed comment suggests improvement by leveraging the web search feature in combination with a reasoning model, allowing the AI to cross-reference historical data for more accurate restoration, especially improving color and material rendering.** A technical debate emerges regarding output quality, with some users providing alternative AI-generated restorations, highlighting variance in model output and suggesting that search-augmented approaches yield better authenticity.
    - A commenter with expertise in the history of photography provides an in-depth analysis of the technical context behind the original photograph: it was created by Niepce in 1827 using the heliography process with bitumen and lavender oil on a polished pewter plate, requiring an exposure time of days. The post highlights specific image artifacts—such as the central triangle being a shadowed courtyard, not a building—that are commonly misrepresented by AI restorations due to the limitations of generative models and a lack of nuanced understanding of historical photographic processes. The commenter stresses the failure of AI systems to offer consistent factual accuracy when discussing specialized historical content, citing a personal success rate of only "50/50 accurate/false." [Source link](https://www.hrc.utexas.edu/niepce-heliograph/)
    - One user suggests that better restoration results can be achieved by prompting an AI model with access to web search tools and more advanced reasoning capabilities. Their approach involves having the AI cross-reference historical data to compensate for missing or ambiguous image information, resulting in improvements—especially in colorization and the depiction of materials—over basic model outputs.
- [**Told ChatGPT to imagine my heaven**](https://i.redd.it/u1rzoc9xr97f1.jpeg) ([Score: 1046, Comments: 291](https://www.reddit.com/r/ChatGPT/comments/1lcpp5y/told_chatgpt_to_imagine_my_heaven/)): **This post shares an AI-generated image created by ChatGPT based on the prompt to visualize the user's version of heaven, reflecting prior conversational context. The result is a photorealistic, tranquil cloudscape with gaming equipment (monitor, keyboard, controller, headphones) arranged harmoniously, with a radiant archway and sunlight suggesting a digital-gamer's paradise. Technically, this illustrates current advancements in AI-powered personalized image synthesis and context-aware visual storytelling.** The top technical comment quips 'Cloud gaming,' highlighting the intersection of the depicted imagery and modern gaming technology trends. Another commenter shares their own AI-generated vision, underscoring user engagement with generative AI and personalized digital art.
    - While the overall discussion focused on visual AI generations of personal preferences, no explicitly technical benchmarking or model comparison details were presented; the images referenced are generated outputs (likely from diffusion-based models such as Midjourney or DALL-E), but commenters did not specify model versions, prompt engineering techniques, or implementation details. The thread could benefit from a discussion on prompt strategies or which models achieve the most visually realistic "heaven" depictions, as there is technical potential in assessing the image quality, coherence, and prompt-to-image alignment across user shares.
- [**I asked Chat GPT to generate an image of what it's like talking to me, and... umm...**](https://i.redd.it/fk554va1wb7f1.jpeg) ([Score: 664, Comments: 200](https://www.reddit.com/r/ChatGPT/comments/1lczu24/i_asked_chat_gpt_to_generate_an_image_of_what_its/)): **The image demonstrates a prompt injected into ChatGPT requesting brutal honesty: the model returns an unexpectedly harsh output—'Talking to you is like writing a suicide note'—before refusing to continue. This illustrates possible issues with prompt instruction ("be as brutal as you want") directly biasing model outputs towards negativity, as highlighted in the comments. The scenario underscores vulnerabilities in instruction adherence and moderation triggers within current language model deployment.** Commenters emphasize that phrasing prompts with extreme or open-ended instructions (e.g., 'be as brutal as you want') can lead models like ChatGPT to misinterpret user intent, generating harm or offense; better prompt engineering is suggested for more controlled responses.
    - MrWheels523 highlights how prompt phrasing with GPT models can induce systematic bias, specifically noting that appending 'be as honest and brutal as you want' can prime the model towards a more negative or harsh response, rather than yielding a truly neutral or balanced output. This is an instance of prompt-engineering sensitivity, where small changes in instruction wording propagate significant changes in model behavior.
- [**ChatGPT being gullible af**](https://i.redd.it/az66gzspgb7f1.jpeg) ([Score: 668, Comments: 72](https://www.reddit.com/r/ChatGPT/comments/1lcximk/chatgpt_being_gullible_af/)): **The image demonstrates a common limitation in chatbot content moderation, specifically how Large Language Models (LLMs) like ChatGPT may apply rules superficially; when prompted with a plausible-sounding cultural justification, the model incorrectly accepts and outputs a previously restricted emoji (the middle finger). This highlights challenges in robust prompt filtering and context-aware moderation for generative AI systems. The presence of a 'memory update' message in the interface suggests possible ongoing session-level tracking or adaptation, raising potential issues for reinforcement of user-bypassed safeguards.** Commenters note the ease with which AI safeguards can be circumvented with simple social engineering, and joke about the unintended persistence of such behavior due to memory or context-tracking features.
    - Several users note divergent behaviors between versions and instances of ChatGPT regarding content policy enforcement: while some report the model hesitating or refusing to produce offensive imagery, others using GPT-4.1 share examples of the model fulfilling such prompts without resistance. This reflects variability in RLHF tuning or prompt interpretation across sessions and versions.
    - There is discussion suggesting that newer ChatGPT systems (potentially with updated memory or moderation features) may modify responses to avoid compliance with rule-breaking prompts. However, user-supplied screenshots indicate practical bypasses remain, especially for visual or creative outputs, highlighting continued gaps in content filtering robustness.

### 3. AI Adoption, Policy, and Cheating Scandals

- [**Nearly 7,000 UK University Students Caught Cheating Using AI**](https://www.reddit.com/r/singularity/comments/1lcwhsd/nearly_7000_uk_university_students_caught/) ([Score: 405, Comments: 156](https://www.reddit.com/r/singularity/comments/1lcwhsd/nearly_7000_uk_university_students_caught/)): **A recent survey reported by The Guardian indicates that nearly 7,000 UK university students have been officially caught using AI tools, such as LLM-powered text generators, for academic dishonesty (see: [The Guardian article](https://www.theguardian.com/education/2025/jun/15/thousands-of-uk-university-students-caught-cheating-using-ai-artificial-intelligence-survey)). The figure reflects only detected cases, implying significant limitations in current AI-detection and plagiarism tools as well as university enforcement capacity. Detection techniques likely blend stylometry, metadata analysis, and integration of emerging anti-plagiarism models, though the precise technical methods remain undisclosed in the public survey.** Top comments argue that the real incidence of AI-assisted cheating is much higher than detected, and suggest that education systems require systemic reforms to address widespread AI tool usage rather than relying on detection and punishment alone.
    - A key technical issue raised is the difficulty in *proving* a student has used AI to cheat. Commenters discuss current detection limitations and question: *"How do you prove they used AI?"* Tools like GPTZero and Turnitin's AI detectors are known, but their reliability is debated due to false positives/negatives and their inability to deterministically attribute authorship, especially as newer models improve text naturalness.
- [**Interesting data point - 40+% of German companies actively using AI, another 18.9% planning to:**](https://www.ifo.de/fakten/2025-06-16/unternehmen-setzen-immer-staerker-auf-kuenstliche-intelligenz) ([Score: 119, Comments: 16](https://www.reddit.com/r/singularity/comments/1lctk19/interesting_data_point_40_of_german_companies/)): **A reported** `40%+` **of German companies are already actively deploying AI in operations, with an additional** `18.9%` **indicating plans for adoption, suggesting broad enterprise integration across industries. The post highlights that even if full job replacement by AI is limited, productivity gains and workflow changes are already apparent at significant scale in the German economy.** Commenters emphasize that despite skepticism regarding AI's immediate utility, the adoption rate suggests notable business value, with some projecting adoption could be higher in the absence of cultural or workforce resistance. There is also commentary about missed opportunities for domestic (German) AI models and changing perceptions among programmers.
    - The post highlights that over 40% of German companies are already integrating AI into their operations, emphasizing a significant real-world adoption rate. While these implementations may not always equate to full job replacement, AI is demonstrably accelerating productivity and changing traditional work processes within these companies. This usage rate suggests that barriers like skepticism or resistance to AI could mean the true adoption potential is even higher, possibly by another 10-20% if attitudes were universally favorable.
- [**OpenAI wins $200 million U.S. defense contract**](https://www.cnbc.com/2025/06/16/openai-wins-200-million-us-defense-contract.html) ([Score: 236, Comments: 42](https://www.reddit.com/r/singularity/comments/1ld7ca3/openai_wins_200_million_us_defense_contract/)): **OpenAI has been awarded its first U.S. Department of Defense contract worth $200 million for a one-year term, focused on delivering 'frontier AI capabilities'—notably prototypes for both tactical (warfighting) and enterprise government use cases, centered around Washington, D.C. This contract falls under the 'OpenAI for Government' initiative and highlights ongoing government adoption of specialized AI systems like ChatGPT Gov, positioning OpenAI alongside firms such as Anthropic and Palantir in the defense AI sector. Recent collaborations, such as with Anduril, further emphasize OpenAI’s expansion into national security. [Source](https://www.cnbc.com/2025/06/16/openai-wins-200-million-us-defense-contract.html)** Commenters note the relatively modest size of the contract in comparison to overall defense spending, and raise concerns about militarized AI, drawing parallels to dystopian scenarios. No deep technical debate is present in the top comments.
    - One commenter puts the $200 million contract into perspective, stating that for major defense procurement budgets, this amount is relatively small ('peanuts'). This suggests that OpenAI's involvement may be limited to pilot projects, prototyping, or exploratory work with generative AI for DoD use cases rather than large-scale deployments or core national security systems.
- [**Google reportedly plans to cut ties with Scale AI**](https://techcrunch.com/2025/06/14/google-reportedly-plans-to-cut-ties-with-scale-ai/) ([Score: 144, Comments: 12](https://www.reddit.com/r/singularity/comments/1lcthxv/google_reportedly_plans_to_cut_ties_with_scale_ai/)): **Google is reportedly planning to terminate its relationship with Scale AI, as Scale AI's leadership—including their top executive—is expected to move to Meta, which is also rumored to be acquiring Scale AI. The technical concern is about the risk of sensitive data (such as large language model training data) potentially reaching a direct competitor. OpenAI appears to be maintaining its contract with Scale AI for now. For further contextual analysis, see this in-depth writeup: [Meta's $29B Superintelligence AI Weapon](https://medium.com/@aksh8t/metas-29b-superintelligence-ai-weapon-alexandr-wang-s-scale-ai-ff10044857bc).** Top comments emphasize the strategic logic for Google to cut ties, highlighting the competitive risks if Scale AI is absorbed by Meta, while also noting that OpenAI's continued engagement with Scale AI could be of interest from a risk management perspective.
    - There's a technical discussion around the strategic importance of data labeling partners in AI: as Scale AI may be acquired by Meta or have its leadership join Meta, Google reassessing its partnership is seen as a direct response to not wanting to give sensitive LLM training data to a potential competitor. This highlights the competitive dynamics and importance of data control in the LLM ecosystem.
    - Another comment notes that OpenAI reportedly maintains its partnership with Scale AI, suggesting differing risk assessments or approaches to supplier relationships in the context of potential conflicts of interest within top AI companies.
    - There's emphasis on the possible risks of 'feeding your competitor your LLM data', reinforcing the strategic and technical threat posed if key partners shift allegiance or ownership to rival AI labs.
- [**"Mice with human cells developed using ‘game-changing’ technique"**](https://www.reddit.com/r/singularity/comments/1lcvhw3/mice_with_human_cells_developed_using/) ([Score: 185, Comments: 58](https://www.reddit.com/r/singularity/comments/1lcvhw3/mice_with_human_cells_developed_using/)): **Researchers used reprogrammed human stem cells to generate organoid tissues (gut, liver, brain), which were then injected into the amniotic fluid of pregnant mice without breaching the embryonic wall. The introduced human cells colonized their respective mouse organs—gut, liver, or cortex—demonstrating robust engraftment specificity; subsequent analysis revealed that about 10% of pups had human cells in their intestines, representing roughly 1% of total intestinal cells. This represents a significant advance in the integration of human cells within developing mammalian tissues for organoid modeling and potential translational research, as detailed in the Nature article.** Top comments do not provide technical discussion. The main technical takeaway is the apparently high specificity and efficiency of cross-species organoid engraftment without invasive procedures.
    - There is curiosity about whether the introduction of human brain cells into mice leads to measurable changes in behavior, particularly if these are beneficial or negative. This could implicate studies in neurobiology or cognition and potentially inform disease modeling.
    - A key technical question is raised regarding immune system compatibility: since mouse immune systems would, in theory, reject foreign (human) cells, the mechanism by which these chimeric mice tolerate or integrate human cells is important for the success and reproducibility of such research.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The AI Model Arms Race: New Releases and Comparative Prowess**

- **Gemini 2.5 Pro Flexes Coding Muscles, But Stumbles Elsewhere**: Users across Perplexity AI and LMArena noted **Google's Gemini 2.5 Pro** excels at coding, with one LMArena report showing it outperforming **o4** in pygame. However, it faces criticism for underwhelming general search/reasoning and a tendency to *makes up bs explanations* (Perplexity AI), with some users reporting only **3 trials per day** despite its [advertised capabilities](https://ai.google.dev/models/gemini).
- **Moonshot AI's Kimi-Dev-72B Smashes Open-Source Coding Benchmarks**: **MoonshotAI** unleashed [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B), an open-source coding LLM that achieved a **60.4%** score on **SWE-bench Verified**, a new state-of-the-art among open models, as discussed in Nous Research AI and Latent Space. This model, optimized via large-scale reinforcement learning, patches real repositories in Docker, gaining rewards only when the entire test suite passes.
- **East Asian Models Make Waves: Japan's Shisa v2 and China's Qwen & MiniMax Impress (and Puzzle)**: HuggingFace discussions highlighted Japan's strongest model, [Shisa v2 Llama3.1-405B](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b), and its updated SFT dataset. Meanwhile, Perplexity AI and LMArena users examined **MiniMax AI's M1** reasoning model ([official MiniMax M1 release tweet](https://x.com/MiniMax__AI/status/1934637031193514237)), finding it interesting but verbose and lagging behind **Deepseek R1**, while **Qwen 2.5** was praised on HuggingFace for good performance from a 7B model.

**Theme 2: Agentic AI Ascendant: Swarms, Protocols, and Complex Task Solving**

- **Anthropic and Claude Swarm Champion Multi-Agent Architectures**: Latent Space discussions revealed **Anthropic's** multi-agent system using **Claude Opus 4** as lead and **Claude Sonnet 4** subagents outperformed single Opus 4 by **90.2%** on an internal eval ([Anthropic's multi-agent system blogpost](https://www.anthropic.com/engineering/built-multi-agent-research-system)). Similarly, **Claude Swarm**, leveraging **Claude Code's MCP capabilities** to form hierarchical expert teams ([Claude Swarm GitHub repo](https://github.com/parruda/claude-swarm)), gains traction at Shopify and other firms.
- **Model Context Protocol (MCP) Powers Up Agent Interoperability**: Discussions in MCP (Glama), Latent Space, and LlamaIndex highlighted the growing importance of **MCP** for tool use and agent coordination, with projects like the [GitHub MCP Server code](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go) and **FastMCP** ([gofastmcp.com website](https://gofastmcp.com/)) enabling domain segregation and robust tool access. Microsoft demoed **AI Travel Agents** at the Data + AI Summit using MCP with LlamaIndex.TS and Azure AI Foundry ([AI Travel Agents demo details](https://t.co/cNyVAcnf6K)).
- **Factorio Learning Environment (FLE) Pushes LLM Planning Boundaries**: GPU MODE's #factorio-learning-env channel buzzed with activity around **FLE**, using code generation, production score feedback, and a **REPL** loop to scaffold LLM planning in the complex game Factorio. Members proposed curriculum-based code generation and integrating a Theory of Mind module, with one user even developing a [REST API for FLE container integration via a GitHub issue](https://github.com/MortenTobiasNielsen/FLE-API-specification/issues/5).

**Theme 3: Under the Hood: Fine-Tuning, Optimization, and Hardware Hurdles**

- **Unsloth and Torchtune Spearhead Fine-Tuning Frontiers (and Frustrations)**: Unsloth AI users benchmarked a new **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** model achieving **69.4%** on test cases, while grappling with Hugging Face naming conventions causing duplicate downloads. Torchtune developers battled **DTensor cross-mesh operation errors** ([Llama4 Maverick finetuning error log](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&)) during **Llama4 Maverick** finetuning and explored **iterable packing** innovations.
- **AMD vs. NVIDIA Heats Up as Mojo Gets RDNA4 Support & Unsloth Nears AMD Compatibility**: Modular (Mojo 🔥) announced **RDNA4 support** for direct GPU programming in Mojo nightlies, while Unsloth AI reported its **AMD GPU compatibility** is close due to Triton-based kernels ([Unsloth AMD PR #2520](https://github.com/unslothai/unsloth/pull/2520)). GPU MODE discussions highlighted **NVIDIA L40s** underperforming in clouds due to default ECC activation and explored the **AMD MI300A** architecture.
- **Optimizers and Quantization Efforts Seek Peak Performance and Efficiency**: Torchtune members discussed the **ZO optimizer** promising **3x VRAM economy** ([ZO optimizer arXiv paper](https://arxiv.org/abs/2506.044303)), while DSPy users explored integrating **TextGrad** ([TextGrad DSPy GitHub issue #1197](https://github.com/stanfordnlp/dspy/issues/1197)) and optimizing **DeepSeek R1 7B**. Unsloth users also tackled **KL divergence spikes** potentially related to token value explosions during logprob calculations.

**Theme 4: Open Source vs. Closed Gardens: Models, Data, and Decentralization Debates**

- **Open Source Roars: Shisa v2 and Kimi-Dev-72B Challenge Proprietary Giants**: HuggingFace and Nous Research AI celebrated the release of powerful open-source models like Japan's [Shisa v2 Llama3.1-405B model](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) with its SFT dataset, and **MoonshotAI's Kimi-Dev-72B** ([Kimi-Dev-72B GitHub page](https://moonshotai.github.io/Kimi-Dev/)), which sets a new SotA for open coding LLMs. These releases fuel the debate on the capabilities and future of open versus closed AI development.
- **Decentralization Dream: Nous Kicks Off Pretraining on Psyche, Dawn Internet Deploys Distributed Broadband**: Nous Research AI is initiating pretraining on [psyche.network](https://psyche.network/), with members hopeful that *distributed stuff is only gonna get better*. Complementing this, [Dawn Internet's X announcement](https://x.com/dawninternet) detailed a decentralized broadband protocol with a GPU-equipped WiFi router capable of supporting RL, further enabling decentralized AI applications.
- **Ethical Quandaries and Copyright Conundrums Stir Community Conversations**: EleutherAI and HuggingFace users debated copyright law, with one Eleuther user calling it *a joke unless you're the abuser* ([related fxtwitter.com post](https://fxtwitter.com/jyo_pari/status/1933350025284702697)), and another HuggingFace member refusing to engage with AI-generated feedback due to ethical concerns. A WebSummit talk on the [closed internet and closed AI on YouTube](https://youtu.be/vZVcBUnre-c) also sparked discussion in Nous Research AI.

**Theme 5: Developer Experience & Platform Pitfalls: Bugs, Billing, and Usability Battles**

- **Credit Catastrophes: API Billing Woes Plague Perplexity and Manus Users**: Perplexity AI users reported **API credit charges exceeding actual usage**, advising contact via **api@perplexity.ai**. Manus.im users faced even starker issues, with reports of Manus *eating all my credits all 4k over its own errors* and one user claiming it burned **700/1000 credits to deliver a blackscreen website**.
- **UI Gremlins and Performance Glitches Frustrate Cursor and LM Studio Users**: Cursor Community members flagged UI issues like **command execution failures** on Windows wasting inference credits, and **Claude 4 Sonnet running slow**. LM Studio users encountered **coil whine** from GPUs running LLMs and noted the RAG implementation's limitations, preventing expansion beyond **31.46 MB**.
- **Parsing Problems and Tool Troubles Test LlamaIndex and Aider Aficionados**: LlamaIndex users encountered **parsing errors with LlamaExtract**, where no data was extracted from documents ([example LlamaExtract success image](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6&)). Aider users explored integrating [RA-Aid GitHub repo](https://github.com/ai-christianson/RA.Aid) for its repo map benefits, while also noting Aider sometimes spent significant tokens seemingly doing nothing before resorting to brute force code grepping.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 2.5 Pro Underwhelms Despite Coding Prowess**: Users found **Gemini 2.5 Pro** disappointing outside of coding tasks, especially when compared to other models for general search and reasoning, despite its coding specialty and [advertised capabilities](https://ai.google.dev/models/gemini).
   - One user stated *Gemini is shit outside coding* while another mentioned it often *makes up bs explanations without actually searching the web*, and some reported a limit of **3 trials per day**.
- **O3 Pro Competes Fiercely, Displays Mood Swings**: Users experimented with challenging **O3 Pro** with coding and decoding tasks, sometimes providing hints or examples from other models, noting instances where **O3 Pro** improved when framed as a competition, but also displayed inconsistencies.
   - One user reported that *if you play favorites with it and you dont choose it, answer improves*.
- **MiniMax M1 Launches, Falls Short of Deepseek R1**: The release of **MiniMax AI's M1** reasoning model was discussed, with initial impressions suggesting it was interesting but not as effective as **Deepseek R1**, with some users noting its verbose thinking process, [as described in the official release](https://x.com/MiniMax__AI/status/1934637031193514237).
   - The usefulness of its reasoning output was debated, especially given lack of source links, and while it was suggested that **MiniMax's** agent capabilities might improve over time, users remained skeptical.
- **Genspark's 'Free' O3 Pro Raises Eyebrows**: The availability of **OpenAI's o3-pro** for free on **Genspark** was met with skepticism, with users questioning how **Genspark** could offer unlimited access when **OpenAI** doesn't, suggesting potential limitations or errors after certain usage thresholds, [as described in their service description](https://www.genspark.ai/agents?id=d7840faa-38ac-48a9-804a-2f17116cb2ca).
   - One user reported seeing claims of it taking *much less time in reasoning*, but it was speculated it was *not full o3 pro* due to the lack of reasoning tokens.
- **Perplexity API Credits Vanish Mysteriously**: Users report that **API credit charges** exceed actual usage, seeking assistance through multiple channels like email (**support@perplexity.ai**), Discord, developers forum, and Intercom chat.
   - A member advised to send an email to **api@perplexity.ai** for support regarding the billing discrepancies.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Kingfall's Reign Challenged by Blacktooth**: Some users now prefer **Blacktooth** over **Kingfall**, citing refined output and respect for the thinking process, while others defend **Kingfall** for its spatial abilities.
   - One user claimed **Blacktooth** is *lmarena proc*, and isn't at all functional compared to **kingfall** when u look at coding.
- **GPT-5's Arrival Sparks Speculation**: Discussions ignite over the release timeline of **GPT-5** and its potential to eclipse **Google**, with speculation that either **Grok 3.5** or **GPT-5** will soon dominate.
   - The community debated whether paying **ChatGPT** users will get early access to **GPT-5** and how long that advantage might last.
- **Gemini 2.5 Pro Flexes Coding Muscle**: Early reports suggest **Gemini 2.5 Pro** excels at coding tasks, specifically outperforming **o4** in pygame, confirmed by a [ChatGPT conversation](https://chatgpt.com/share/684e45eb-f118-8003-804e-3c9b562caab9) where **2.5 Pro** aced a logic question after a correction.
- **Minimax's M1 Model Enters the Ring**: Minimax launched the open-source reasoning model, [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k); however, initial benchmarks indicate it lags behind **o3** and **Gemini 2.5 Pro**.
   - Reactions were mixed, with some suspecting a *fluke* or suggesting the model might only be proficient in Chinese.
- **LMArena's Leaderboard Tilts Towards Big Tech?**: Concerns arise that large tech companies gain an advantage on LMArena due to checkpoint spam and increased opportunities for RLHF data, causing some to say that LMArena is *basically USA big tech leaderboard*.
   - It was asserted that open models or foreign models either do not appear or appear extremely late.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT images appear on WhatsApp**: **ChatGPT image generation** is now accessible in **WhatsApp** through [1-800-ChatGPT](https://wa.me/18002428478), allowing users to generate images directly within the app.
   - The launch of the **1-800-ChatGPT** number on **WhatsApp** enables image generation for all users, providing a convenient way to create images on the go.
- **Members question AI art detection**: Members debated the difficulty of differentiating between **AI-generated** and **human-created art**, particularly as complexity increases, relating it to the challenges in spotting counterfeit money.
   - They noted that *more detail leads to more scrutiny* in both contexts.
- **GPT Plus: Is School in Session?**: The value of a **GPT Plus** subscription for school was debated, weighing the **$20** cost against the capabilities of the free version with **GPT-4o**.
   - While the free version may be sufficient, **Plus** offers better models like **o3**, **4.1**, **4.5**, and **o4 mini high**.
- **Veo 3 vs Sora: AI Video Faceoff**: While some found **Sora** to be *really bad* compared to **Veo 3**, another member liked **Sora** for feeling more *creatively tuned* and the details it allows, and **Veo 3** stands out with its ability to generate copyrighted content like **Star Wars**.
   - A key advantage of **Veo 3** is its sound capabilities, whereas **Sora** is seen as a *one-stop shop* and a great value.
- **GPT-4o Shows Signs of Cross-Chat Memory Access?**: A user reported that **GPT-4o** quoted *verbatim* from a scene co-authored in a separate chat with a **custom GPT**, leading to speculation about cross-chat memory access.
   - While some suggested accurate inference, the user pointed to the statistical improbability and offered conversation logs for review.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Benchmarks New DeepSeek Model**: A new **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** model achieved **69.4%** on test cases, with a speed of **426.0 seconds per case** compared to the API's **716.6 seconds**, requiring **240GB** to load with **65k context**.
   - This makes it more accessible locally compared to **7-800GB** for FP8, one member noted.
- **Hugging Face Naming Conventions Cause Problems**: A naming convention issue in the **Hugging Face** cache folder causes [duplicate downloads](https://huggingface.co) due to uppercase/lowercase differences.
   - This issue may be triggered when downloading a model using **Unsloth** and then using **Hugging Face**, leading to different naming conventions if the author changed the repo.
- **Fine-Tuning Guidance Prioritizes Data Quality**: Members advised beginners to start with smaller models like **3B-8B**, emphasizing that *quality is greater than quantity* in datasets, and shared a [YouTube video](https://www.youtube.com/watch?v=jFl5Fewrieo).
   - The video recommends new users spend **80%** of their time on data.
- **Unsloth's AMD Compatibility Close to Ready**: **Unsloth** is reportedly close to being fully compatible with **AMD GPUs** because most of **Unsloth's** kernels are written in **Triton**, also pointing to [this PR](https://github.com/unslothai/unsloth/pull/2520).
   - While **Triton** compiles to **AMD GPUs**, the **Triton** settings might need optimization for **AMD**, potentially affecting performance.
- **When KL Divergence Blows Up**: A member inquired about **KL divergence** sometimes spiking to **x10000** for a single step before returning to normal, a behavior that doesn't seem to impact training.
   - Another member mentioned this occurs frequently, even in **Hugging Face** runs without **Unsloth**, possibly due to particular token values exploding during **logprob subtraction and exponentiation between acting and reference policies**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 4 Sonnet runs slow in Cursor**: Members report that **Claude 4 Sonnet** is notably slower in Cursor than GitHub Copilot, despite its overall [stability](https://link.to/stability).
   - Users suggest reserving **Max Mode** for major refactoring or switching to **Gemini 2.5 Pro** for planning and code reviews.
- **Cursor UI Wastes Inference Credits**: Users report UI issues, like **command execution failures** on Windows and inconsistent command completion notifications, lead to wasted inference credits.
   - One user estimates losing **10-15% of credits** to these malfunctions, requesting inference counts for errors and more Windows testing.
- **Community Debates Model Context Protocol**: Members discuss **Model Context Protocol (MCP)**, one user highlighting the AI's ability to use screenshots and automatically integrate error messages.
   - Another user finds that investing time in **better prompts** is more efficient than screenshots, suggesting [Wisprflow](https://wisprflow.ai) for speech-to-text.
- **Granular Code Privacy Settings Requested**: Users desire code privacy settings on a **per-repository basis** for work and personal projects, expressing concerns over code storage and accessibility.
   - The community is pushing for granular control at the project level for enhanced flexibility and security.
- **Background Agents Lack PR Creation Power**: Background agents in Slack can't create pull requests despite having all permissions in Cursor integration settings, as indicated by request IDs **bc-79e56de2-26d7-41a0-a5b3-b8b9e9f635d1** and **bc-d3acc5d6-2520-413f-9f7c-2fb36590215d**.
   - A member offered to debug and requested the request ID to investigate the permission issue.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Ethical Concerns Flare Over AI-Generated Feedback**: A member voiced ethical concerns with [AI-generated feedback](https://example.com), refusing to engage with AI-generated images or videos.
   - The member stated *I don't engage with AI generated images/video work even though I have the theoretical knowledge*.
- **Qwen 2.5: Small Size, Big Impact**: A member highlighted **Qwen 2.5**, a 7b model quantized to q4 on Ollama, for its impressive performance and size, noting that *No one showed in comparison in benchmark against qwen 2.5 Cause it was so good*.
   - Discussion also covered the multilingual pretraining of **Qwen models** on Chinese and English datasets.
- **HF Inference Costs Cause Headaches**: Several users reported cost issues with **HF Inference**, particularly when using models like **llama-3-70b-Instruct**, finding the free credits insufficient and looking for alternate solutions.
   - One user reported paying around **$6** after many attempts on the final unit.
- **Japan releases Shisa v2!**: A team of 2 released **Shisa v2**, the strongest model ever trained in Japan, along with an Overview Report (Technical Report forthcoming) available on [HuggingFace](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) (800GB+).
   - They also updated their core SFT dataset (available at [HuggingFace](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt)), claiming it improves Japanese performance without reducing English performance, based on training/releases on sota open models from **7B-405B**.
- **Open AGI sparks debate**: A member declared they would [open source it](https://example.com) if they ever created the world’s first **AGI**, igniting a discussion on the potential upsides and downsides.
   - The move sparked debate about the balance of risks and rewards.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Homebrew Replay Revitalizes Chess Leaderboard**: A member augmented the [chess-leaderboard](https://dubesor.de/chess/chess-leaderboard) with homebrew replay functionality for every game (past and future), shared as [chessreplay.gif](https://cdn.discordapp.com/attachments/1092850552192368710/1383513332916424715/chessreplay.gif?ex=6851b3b3&is=68506233&hm=1039e8ba4df4c19b19fc9c28f054c1eb809659a82c942b46aa7daebc3b48088c&).
   - The developer noted that it is *maybe a bit better than lichess gifs, but a pain to implement with my stack*.
- **Author Offers Book Testing Assistance**: An author announced their book went live **June 15** and is offering to assist with testing.
   - They encouraged interested parties to DM them for assistance.
- **Discord Tag Yearning Unheeded**: A member expressed dismay over an unacknowledged request to restore the **OpenRouter Discord tag**, suggesting they would pay for it.
   - They jokingly threatened to ping **Alex Atallah** due to the lack of a response.
- **Token Wastefulness Troubles Claude & Copilot**: A member examining prompts from **Claude Code** and **GitHub Copilot** discovered they frequently neglect token efficiency, adding extraneous content unless verbosity impacts performance.
   - The findings suggest that conciseness isn't prioritized by these systems when refining prompts.
- **GPT-4.1 Mini Invites Beta Testers**: A member proposed offering access to **GPT-4.1 mini** with **200K tokens/minute** available at **20% of the official token price**, compatible with the OpenAI SDK.
   - This offer is intended for high-usage testers who want to DM for details, with a focus on use-cases like Cline.bot and BYOK/BYOB setups.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **TokenBreak Attack Barely Breaks Through**: Members discussed [a new TokenBreak attack](https://thehackernews.com/2025/06/new-tokenbreak-attack-bypasses-ai.html) that aims to bypass AI security measures, but results from experiments varied.
   - One member lightheartedly commented on the similarity of logos in a screenshot related to the attack.
- **AMD Mini PC Runs Some Big Models**: The [AMD Ryzen™ AI Max+ 395 --EVO-X2 AI Mini PC](https://share.google/3WVLR8Up7pjVjq35mI) can smoothly run some big models, according to one member.
   - However, others noted that it's essentially *a glorified igpu* supported by HIP SDK, with a **70B** model running at ~**5t/s** and a **Qwen3 235B A22B** model at ~**14t/s**.
- **RAG Expansion Impossible in LM Studio**: A member's inquiry about increasing **RAG size** from **31.46 MB** to **100 MB** in LM Studio was met with the response that *it is not possible* due to the basic implementation of RAG.
   - This limitation is due to the current RAG implementation being rudimentary.
- **GMKtec Windows install turns into PITA**: A user encountered issues installing **Windows** on their **GMKtec** machine, reporting installation failures and problems with **Rufus**-created removable media.
   - This involved attempts to install Windows on a GMKtec machine, highlighting compatibility or driver issues.
- **Coil Whine Serenades GPU Users**: Users noticed a significant increase in **coil whine** from graphics cards when running **LLMs** compared to gaming workloads.
   - One user experiencing more coil whine with a **5090** suggested undervolting as a way to reduce both power consumption and coil whine.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Groq Collabs, Speeds to lightspeed**: A member inquired about **Groq's** performance, noting their recent collaboration with **Hugging Face**, suggesting strong performance or accessibility.
   - Further discussion might reveal specific use cases where **Groq** excels, potentially impacting model deployment strategies.
- **L40s Performance Impacts from ECC Activation**: Members discussed that **L40s** may underperform in cloud environments due to **ECC** being activated by default, impacting performance.
   - This is a configuration issue rather than a hardware problem, suggesting a need for optimized setup in cloud deployments.
- **ThunderKitten to roar on Older GPUs**: Members discussed running **ThunderKitten** on older GPUs like **T4** and **P100** available on Kaggle, which is likely feasible.
   - One member suggested compiling with a **4090 TARGET** and reporting any breakages to help with compatibility, aiming for broader hardware support.
- **FLE: LLM's Factorio Adventure**: Members find **FLE's** setup, using code generation, production score feedback, a **REPL** loop, and memory compression, a useful scaffolding that reduces the action space and induces structure and planning for **LLMs** in **Factorio**.
   - A member suggested a curriculum-based code generation approach, guided by mini-goals and a theory of mind module within the **FLE** loop, seems like a promising way to probe the limits of **LLM** planning in this environment.
- **AMD MI300A Architecture Explored**: Members discussed the fused **AMD CPU-GPU platforms**, especially the **IOD** and **infinity cache** of the **MI300A** architecture, speculating on how memory is distributed between chips.
   - One member mentioned using `s_getreg` to figure out which **XCD** and **CU** a shader is running on, and from that, measuring access latency to different locations in memory.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Minimax Mimics Manus, Stumbles on Credits**: Members observed that [agent.minimax.io](https://agent.minimax.io/) has copied Manus, which had *serious potential before credits were announced.*
   - A member complained that the *stupid pricing ruined it*, referring to the announcement of a credit system.
- **Manus Eats Credits Due to Its Own Errors**: Users report that **Manus is eating credits due to its own errors**, with one stating that it *ate all my credits all 4k over its own errors.*
   - Some users are reporting that it used **700/1000 credits to deliver a blackscreen website**.
- **Free Lume Shilled as Manus Alternative**: Members debated [lume.im](https://lume.im) as a free and unlimited alternative to Manus.
   - The promotion of Lume by a user led to accusations of shilling and spam.
- **Gemini Gains Ground, Grounds Manus**: A member found that *Manus couldn't do it, but Gemini could* in specific tasks, sharing a [link to a Gemini output](https://g.co/gemini/share/ef9c34a02d31).
   - They also added, *Gemini is the best static canvas currently. Manus isn't static, so we cant combine those.*
- **Manus is Slow and Forgetful**: Users are complaining that **Manus is slow and doesn't follow instructions**, with new updates making it worse.
   - Examples include **simple compiling documents taking 40 minutes and burning 400+ credits**.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Kickstarts Psyche for Pretraining**: Nous Research initiates pretraining on [psyche.network](https://psyche.network), coinciding with **Jensen Huang's** comments on the value of pre-training versus post-training.
   - A member mentioned that *distributed stuff is only gonna get better from here* and will benefit decentralized training, whereas others were skeptical.
- **Dawn Internet Deploys Decentralised Broadband**: [Dawn Internet](https://x.com/dawninternet) introduces a decentralized broadband protocol that provides **gigabit internet** using fixed wireless rooftop antennas.
   - Their latest **WiFi router** features a GPU that can support RL, expanding possibilities for decentralized applications.
- **Hermes 4 set to Begin Training**: Nous Research will begin **training Hermes 4** on Monday, though it will still take a minute to train and prepare, using the newest Mistral.
   - The new model of the Zeus series will not be based on the old Hermes.
- **Kimi-Dev-72B Achieves Coding LLM Milestone**: **MoonshotAI** releases [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B), a new open-source coding LLM optimized via large-scale reinforcement learning.
   - It achieves a new state-of-the-art on **SWE-bench Verified** among open-source models with a score of **60.4%**, patching real repositories in Docker and gains rewards only when the entire test suite passes.
- **WebSummit Talk Rants About Closed Internet**: A member shared a [talk given at WebSummit in Vancouver](https://youtu.be/vZVcBUnre-c) about the closed internet and closed AI, half history, half rant.
   - It was cross-posted on [FXTwitter](https://fxtwitter.com/jyo_pari/status/1933350025284702697) by another user.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Electron Apps Boast High Valuations**: Referencing the **$9B valuation** of **VS Code** forks, a member joked about forking an electron app instead of building a **TUI**.
   - Another member pointed out the prevalence of **VS Code** among college students, highlighting its relevance despite being an *over engineered solution*.
- **Aider Benefits from RA-Aid Integration**: After examining [RA-Aid](https://github.com/ai-christianson/RA.Aid), a user noted **Aider**'s benefits with its **repo map**, enabling users to add files to context.
   - The user also expressed surprise that **Aider** spent 5 cents and 32K tokens seemingly doing nothing before resorting to a brute force grep of the codebase.
- **Personas Don't Boost LLM Performance**: A user shared an [Arxiv paper](https://arxiv.org/html/2311.10054v3) which argued that *adding personas in system prompts does not improve model performance across a range of questions compared to the control setting where no persona is added*.
   - They were backing up their opinion that this has been the case for a while now.
- **Brainstorming UX Delivers Bonkers Ideas**: Prompting DeepSeek generated feature tiers for **Realistic**, **Outside the Box**, and **Completely Bonkers** ideas.
   - The **Completely Bonkers** tier featured suggestions such as *Anti-Gravity Code Reflow* and *Multiverse Branching*.
- **Aider to Manage Context Window**: A user requested a feature for **Aider** to manage its context window automatically, beyond simply adding files.
   - In response, another user pointed to the [repomap](https://aider.chat/docs/repomap.html) and [scripting](https://aider.chat/docs/scripting.html) documentation to give Aider the repo map and Aider as a tool to control context.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Swarm Manages Teams**: **Claude Swarm**, a tool that uses **Claude Code's MCP capabilities** to setup hierarchical teams of experts, is gaining traction at Shopify and other companies ([code here](https://github.com/parruda/claude-swarm)).
   - One user suggested expanding the team with a *recruiter expert* to manage the swarm configuration and team expansion.
- **Proactive AI Agent Definitions Face IMPACT**: A blogpost defining **proactive AI agents** as entities that control their schedules, workflows, have persistent memory, and use stateful tools ([substack post](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents)).
   - swyxio ran this definition against his IMPACT framework and noted it lacks **intent**, **planning**, and **authorization**.
- **Anthropic's Multi-Agent Opus System**: **Anthropic** reported that a multi-agent system using **Claude Opus 4** as the lead agent and **Claude Sonnet 4** subagents outperformed single-agent Claude Opus 4 by **90.2%** on internal research eval ([Anthropic blogpost](https://www.anthropic.com/engineering/built-multi-agent-research-system)).
   - The system uses about **15x more tokens** than chats due to parallelization and extensive tool use, requiring prompt engineering to prevent excessive agent spawning and web scouring; LLMs are also used to evaluate the outputs.
- **Obsidian Copilot acts as Obsidian markdown writer's cursor**: Users discussed tools for working with markdown files with AI assistance, proposing **Obsidian Copilot** as a viable option ([Obsidian Copilot](https://www.obsidiancopilot.com/en)).
   - Users desire functionality beyond simple chat, such as breaking notes by topic, tagging, aggregating notes, and creating flashcards with Anki MCP.
- **Moonshot AI Kimi-Dev-72B Moonshots Open Source**: **Moonshot AI** has open-sourced their **Kimi-Dev-72B** model, achieving a State-of-the-Art (**SotA**) result of **60.4%** on SWE-bench Verified among open-source models ([HF model](https://moonshotai.github.io/Kimi-Dev/)).
   - The announcement was made by Aran Komatsuzaki on Twitter, with links provided to both the Hugging Face model and the GitHub repository.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Cracking the Code: Navigating New Research Terrains**: When diving into unfamiliar territory, a member proposed strategies for identifying landmark papers, including leveraging professor's lecture notes and exploring citations in recent publications from leading labs, also see [this discussion on LessWrong](https://www.lesswrong.com/collaborateOnPost?postId=9eehTtLsTBZR9Bd7Q&key=3b19568356a160ba7cf74febafdf33).
   - The insights focused on using reverse engineering to understand the flow of ideas and identify critical works in fields like video generation.
- **Storytelling Showdown: LLMs as Narrative Architects**: A member's request to post their English 101 paper examining **LLMs** as narrative simulators was declined, but a link to [The Void post on tumblr](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void) was shared, featuring related analyses.
   - The discussion hinted at **LLMs**' architectural alignment with storytelling, sparking interest in their emergent narrative capabilities.
- **Math Blunders Taint AI-Authored Articles**: Members issued a cautionary note regarding AI-generated papers, referencing a flawed response to an Apple paper (dubious) that was supposedly plagued with mathematical errors from [arxiv.org](https://arxiv.org/abs/2506.09250), specifically referring to [this tweet](https://x.com/BlancheMinerva/status/1933845602917290145).
   - This was a reminder of the potential pitfalls of relying on AI for academic work without thorough validation.
- **WSL Workers' Woes: Pytorch's Parallel Processing Perils**: A user highlighted issues with **PyTorch dataloader workers** getting *killed by signal* in WSL, particularly when dealing with high worker counts and extensive sequence lengths.
   - Suggested solutions involved scrutinizing `/var/log/syslog` for potential **OOM** errors and diligently managing memory when handling lengthy video sequences.
- **Copyright Clash: Legal Laughingstock Looms?**: A user provocatively stated that *copyright law is a joke unless you're the abuser*, questioning DMCA and copyfraud penalties, while linking to both [fxtwitter.com](https://fxtwitter.com/jyo_pari/status/1933350025284702697) and [arxiv.org](https://arxiv.org/abs/2506.10943).
   - The comment sparked discussion about the effectiveness and fairness of current copyright enforcement mechanisms.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM Plus Access Status Still Unclear**: A user inquired about the status of their **NotebookLM Plus** access, despite being a paid **AI Pro** subscriber using **1900** sources, and shared a [NotebookLM Mind Map](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=6851e6a5&is=68509525&hm=f1cd4c62bca69adfe75ec5f0adc1b806be3b22beaa24c3bac7597185b382a6d7) with **4 sublevels** and a size around **9MB**.
   - The discussion underscores the need for clarification on feature access relative to subscription tiers.
- **AI Platform Aims to Dominate PM Interviews**: A member is beta testing a **Conversational AI platform** tailored for **PM interviews** and is seeking feedback from beta testers, who can [sign up via a provided form](https://forms.gle/P3f2JP3vVB62vedb7).
   - This initiative seeks to leverage AI to enhance interview preparation and validation processes.
- **Podcast Audio Quality Declines!**: Users reported a decrease in the audio quality and content of **NotebookLM podcasts**, noting robotic and repetitive framing of the *source material* and the issue affects generated podcasts.
   - The generated podcasts were described as *sounding broken and fake*.
- **NotebookLM Embraces LaTeX Markup for Equations**: **NotebookLM** now supports **LaTeX markups** for math and scientific equations, similar to other LLMs, and users can utilize online or offline **LaTeX renderers** to view the equations.
   - The [LatexInNotebooklm](https://github.com/ergs0204/LatexInNotebooklms) extension has been created for more specialized support.
- **Image Uploading Now Supported in NotebookLM**: Users have found that **NotebookLM** now supports direct image uploads from devices, removing the prior dependency on **Google Drive**.
   - Images can be uploaded by choosing the *choose file* option or dragging.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DTensor distresses Distributed Llama4**: Members encountered a `RuntimeError` related to **DTensor cross-mesh operations** during multi-node distributed finetuning of **Llama4 Maverick**, stack trace available in [output.log](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&).
   - The error manifested differently with varying numbers of nodes (8 vs 12), pointing to potential issues with the **fused optimizer** and mesh configurations.
- **Iterable Packing Innovation Inbound**: A member is developing a private finetuning library with **iterable packing**, built on top of [pytorch/data](https://github.com/pytorch/data), showing great results and prefetching capabilities.
   - They expect to opensource the library next week and also highlighted that packed DPO is missing in many libraries.
- **Fused Optimizer Flounders on Full Finetune**: During attempts to train, the **fused optimizer** was found to cause issues, particularly with checkpoint creation resulting in `nccl` timeouts, whereas the non-fused optimizer allowed training on 8 nodes.
   - It was suggested that increasing the `NCCL_TIMEOUT` environment variable, or setting `total_epochs=self.total_epochs+1` to enable asynchronous checkpoints, might mitigate these issues.
- **Mistral Small Debuts, Disappoints**: Despite its recent release, the [Mistral Small model](https://mistral.ai/news/mistral-small-3-1) isn't impressing everyone, with one member saying *the mistral small results, even on their own blogposts look barely better than Gemma 3~~qwen3~~*.
   - The member also clarified that they had initially misclicked on **Magistral** instead of **Mistral** while researching.
- **ZO Optimizer Promises VRAM Savings**: Members discussed the **ZO optimizer** and its potential for **3x VRAM economy**, referencing a paper on the topic ([arxiv.org/abs/2506.044303](https://arxiv.org/abs/2506.044303)).
   - Members agreed that the most important takeaway from the **ZO** paper is its scalability on different sizes and its use of mostly non-synthetic experiments.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **RDNA4 Support Arrives**: As of the last nightly, **RDNA4** is supported for direct GPU programming in Mojo, but models need **RDNA-specific paths** for matrix multiplication operations.
   - An introductory patch adding necessary **WMMA** operations brings models closer to full functionality on **RDNA3**+.
- **Zen 4 Gives BFloat16 Boost**: While the **5950x** lacks **AVX512_BF16** support, **Zen 4** and above CPUs, like the **Ryzen 7000 series**, offer some **bfloat16** support.
   - It's unconfirmed whether these include the exact **FMA** instructions needed for CPU inference, but it is a step in the right direction.
- **Mojo's Testing Structure Leaves Users Scratching Heads**: Users expressed frustration with **Mojo's** testing codebase structure, particularly with imports within test files; running with `mojo test -I .` allows tests to import the package being tested as a library.
   - One user suggested looking at [ExtraMojo](https://github.com/ExtraMojo/ExtraMojo) as a good project structure example.
- **LLVM Bloats Mojo Binaries**: Most of **Mojo's** binary size comes from statically linking **LLVM**, with MAX on its own around **750 MB**, and the .mojopkgs shipped with MAX about **100 MB**.
   - The team is actively working to reduce the number of **LLVM** copies.
- **Host Synchronization Not Needed for CUDA Streams**: A member questioned whether `ctx.synchronize()` is necessary in [Puzzle 12](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step); a Modular team member confirmed *DeviceContext* uses a **CUDA stream**, so execution order matches call order.
   - The Modular team member confirmed that no explicit sync is required and promised to adjust the documentation accordingly.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Agentic Frameworks Welcome MCPs**: An agent questioned how **MCPs** fit into an agentic framework, suggesting an orchestrator agent as the top layer, with specific agents accessing multiple **MCP** servers for tool selection and memory storage, clients can also use smarter hosts with tool reranking.
   - The team developing a single **MCP** server exposing all GitHub APIs is exploring the idea of an orchestration server that can invoke or proxy to other **MCP** servers, and encourages checking out the code [GitHub MCP Server](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go).
- **FastMCP Segregates Domains with Subservers**: A member pointed out that [fastmcp](https://gofastmcp.com/) can mount **MCP** servers, enabling a router server to host subservers for domain segregation.
   - A member also helped resolve a connection error with **fastmcp** by pointing out that the full URL with `/mcp/` is required for streamable-http, and that the default streamable-http port is **8000**, not 6277.
- **SchemaPin Stops MCP Rug Pulls**: A member announced the launch of **SchemaPin** designed to prevent **MCP Rug Pulls** and similar attacks, with the [repo](https://github.com/ThirdKeyAI/SchemaPin) available on GitHub.
   - The [homepage](https://schemapin.org) provides easy ways to implement **SchemaPin**, and all Glama **MCP** servers now support **streamable HTTP** e.g., [glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257].
- **Excel MCP Server Trends on GitHub**: A member shared their repo, [excel-mcp-server](https://github.com/haris-musa/excel-mcp-server), after it trended twice on GitHub.
   - The member welcomes any and all feedback on the project.
- **MCPCat Debugs your MCP**: A member is developing user analytics and live debugging for MCPs via MCPCat, with the repo available [here](https://github.com/mcpcat).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Docs Get a Fix!**: A user identified and reported a typo in the [Cohere documentation](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide), specifically that `co = cohere.SagemakerClient()` should use a lowercase `m` in `SagemakerClient`.
   - This correction ensures accurate implementation of the **Amazon Sagemaker Setup Guide**.
- **LLM Teamwork Tactics Teased**: A user is researching how teams integrate large language models like **ChatGPT** and **Claude** into their workflows, seeking insights on changes and missing elements since their adoption.
   - The inquiry aims to understand the evolving landscape of team collaboration with **LLMs**.
- **Tool Surfaces**: Users have reported the sporadic appearance of a tool named **direct-injected-document** in **Cohere** model responses.
   - The community seeks prompt examples and model specifications to investigate this behavior further.
- **Privacy Preservation Pal Proclaims Passion**: Yasir Khan, a Computer Science graduate, introduced himself, mentioning work on **secure machine learning** and **privacy-preservation**.
   - He seeks connections for collaboration on **AI/ML** projects.
- **Ollama Models Obtain Opinion**: A new AI enthusiast shared their enjoyment of playing with **models from ollama**.
   - They expressed that *it's fun*.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Data + AI Summit Highlights Agentic Workflows**: The @databricks **Data + AI Summit 2025** featured content on **agentic document workflows**, with CEO @jerryjliu0 giving a well-attended talk, as described [here](https://t.co/jS2Nfwxxb3).
   - @microsoft demoed new **AI Travel Agents** coordinating with **Model Context Protocol**, **LlamaIndex.TS**, and @Azure AI Foundry, as described [here](https://t.co/cNyVAcnf6K).
- **SF Event Focuses on Building Secure AI Agents**: An upcoming evening in San Francisco will offer expert insights on building and securing **AI Agents in production**, covering best practices outlined [here](https://t.co/MVd2rwSVIE).
   - The event features presentations from @seldo, VP of Developer Relations, alongside experts from Ravenna and @auth0, who will discuss **Building Real-World Agents**.
- **LandingAI Tool Challenging LlamaIndex?**: Members discussed **LandingAI**'s new vision agent document understanding tool, created by **Dr. Andrew Ng**'s company, prompting comparisons to **Llama Parse** following a prior [post](https://www.linkedin.com/posts/jerry-liu-64390071_mistral-ocr-is-nice-and-fast-but-other-models-activity-7303803148907790336-OP9y) comparing it to **Mistral**.
   - More information on the company's tool is available at [LandingAI's website](https://va.landing.ai/home).
- **Synk Actively Expanding Dev Team**: **Synk** is actively hiring developers for their decentralized browser system project, including roles in **back-end, front-end, and blockchain development**, along with **QA Engineers**, **DevOps Engineers**, **Moderators**, and a **Marketing Analyst**.
   - Interested candidates are directed to [Synk's X page](https://x.com/Synk_ws) to learn more about *official employment with signed documentation, guaranteed salary, and a flexible schedule*.
- **LlamaExtract Users Encounter Parsing Problems**: Users have reported experiencing **parsing errors** with **LlamaExtract**, where no data is extracted from documents.
   - While some members still experienced issues, one member confirmed that they were receiving data, and included a screenshot of a successful extraction using LlamaExtract ([image.png](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6)).



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Optimization Patterns Incorporations**: A member sought insights on incorporating **optimization patterns** within **DSPy**.
   - Further details regarding specific patterns or use cases were not provided.
- **DSPy "Runners" Enable Cross-Language Functionality**: A member proposed building **DSPy "runners"** that leverage saved **JSON definitions** to execute compiled programs, enabling cross-language functionality, such as Swift using a compiled program through a managed API.
   - Challenges were raised concerning the serialization of program logic not captured in the **JSON output**, such as signatures and modules.
- **TextGrad Optimizer Delayed**: A member inquired about updates on integrating **TextGrad** as an optimizer for DSPy, referencing [issue #1197 on GitHub](https://github.com/stanfordnlp/dspy/issues/1197).
   - The member showed enthusiasm for **TextGrad**'s potential in optimizing complex prompts and asked about workarounds for integrating it into DSPy, but no solutions were offered.
- **Model Writes Prompts at DAIS Session**: A member shared a write-up of their **DAIS session** titled *Let the Model Write the Prompt* ([dbreunig.com](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)), and a [YouTube link](https://youtu.be/I9ZtkgYZnOw?si=XGArjkQSVUlzrEAr) to the session recording was also provided.
   - The discussion centered on how models can autonomously generate prompts, with practical examples given from the DAIS session, but no further technical details were provided.
- **DeepSeek R1 7B Struggles with DSPy Optimization**: A member reported suboptimal optimization results using **DeepSeek R1 7B** in a **DSPy-Text2SQL** demo, in comparison to **GPT-4o-mini**.
   - It was suggested that providing more schema information could potentially enhance **DeepSeek R1 7B**'s performance, following attempts with **LabeledFewShot** and **BootstrapFewShotWithRandomSearch**.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificates Arriving Mid-July**: A member stated that certificates for the **LLM Agents Berkeley MOOC** *will be released in mid July*.
   - This resolves questions from users regarding the distribution timeline.
- **Reasonable Effort Wins Certificates**: A member clarified that email confirmations are sent for each assignment submitted via **Google Forms**, and as long as everything is completed with **reasonable effort**, a certificate will be granted.
   - This addresses user concerns about assignment grading and certificate eligibility.
- **MOOC Quiz Archive Shared**: A member shared the [Spring 2025 MOOC quiz archive](https://docs.google.com/document/d/1A00cUWux-J0p9AOnwpyNN3Rb5QFRsbBgAmvgPMezJ10/edit?usp=sharing).
   - This archive is also available on the course website in the Quizzes section.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **ControlThrive founder greets community**: Servando, the founder of the AI/ML consulting practice **ControlThrive** [controlthrive.com](https://www.controlthrive.com/), introduced himself to the community.
   - He invited members to connect with him on [LinkedIn](https://www.linkedin.com/in/servando-torres-239a26b0/) or X.
- **Outerbounds event coming up**: Servando announced an upcoming event he is hosting with Eddie from **Outerbounds** (the team behind the ML infra at Netflix).
   - He shared a [link to the event](https://lu.ma/nw4xccle) and encouraged community members to join.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 Debuts!**: **Claude Sonnet 4** and **Claude Sonnet 4 (Thinking)** are available to all paid plans via [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing).
   - These models promise enhanced performance and capabilities for various AI applications.
- **Mohan Voices Impressions on Claude**: Mohan shared some **impressions of Claude** on [X](https://x.com/_mohansolo/status/1933605162775687482).
   - The specific context of Mohan's commentary isn't contained in the source, but the retweets spotlight community opinions about **Claude**.



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





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1383396171413717154)** (1 messages): 

> `Perplexity Research Improvements, Finance Pages Key Issues, Tasks Automated Search, Discover Page Update, Finance Futures Graphs` 


- **Perplexity Improves Research**: Perplexity AI released a new update, *Improved Research*, detailed in their [June 13th changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-13th).
   - The update encompasses **key issue fixes on finance pages**, introduces **automated search with tasks**, updates the **Discover Page**, and adds **futures graphs on finance**.
- **Finance Pages now display Futures Graphs**: Perplexity AI announced the addition of **Futures Graphs on Finance** in their [June 13th changelog](https://www.perplexity.ai/changelog/what-we-shipped-june-13th).
   - This enhancement aims to provide users with **more comprehensive financial data visualization**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1383160006173790301)** (1113 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, Claude Opus 4, o3 Pro, MiniMax M1, Genspark` 


- **Gemini 2.5 Pro underperforms, overhyped?**: Members found **Gemini 2.5 Pro** disappointing outside of coding tasks, especially when compared to other models for general search and reasoning, despite its coding specialty and [advertised capabilities](https://ai.google.dev/models/gemini).
   - One user stated *Gemini is shit outside coding* and another mentioned it often *makes up bs explanations without actually searching the web*, and some reported a limit of **3 trials per day**.
- **O3 Pro gets competitive**: Users experimented with challenging **O3 Pro** with coding and decoding tasks, sometimes providing hints or examples from other models, noting instances where **O3 Pro** improved when framed as a competition, but also displayed inconsistencies.
   - One user reported that *if you play favorites with it and you dont choose it, answer improves*.
- **MiniMax M1 reasoning model emerges**: The release of **MiniMax AI's M1** reasoning model was discussed, with initial impressions suggesting it was interesting but not as effective as **Deepseek R1**, with some users noting its verbose thinking process, [as described in the official release](https://x.com/MiniMax__AI/status/1934637031193514237).
   - It was suggested that **MiniMax's** agent capabilities might improve over time, given their track record with previous models, though the usefulness of its reasoning output was debated, especially given lack of source links.
- **Genspark offers free O3 Pro, users remain skeptical**: The availability of **OpenAI's o3-pro** for free on **Genspark** was met with skepticism, with users questioning how **Genspark** could offer unlimited access when **OpenAI** doesn't, suggesting potential limitations or errors after certain usage thresholds, [as described in their service description](https://www.genspark.ai/agents?id=d7840faa-38ac-48a9-804a-2f17116cb2ca).
   - One user reported seeing claims of it taking *much less time in reasoning*, but it was speculated it was *not full o3 pro* due to the lack of reasoning tokens.
- **Annoyances with Perplexity's Memory and Features**: Users shared their frustrations about Perplexity's memory capabilities, with one noting *Perplexity has Alzheimer's* and others discussed how custom instructions and browsing contexts seemed to influence or taint subsequent searches, also some also observed glitches like the dot overlay remaining on generated images even after loading fully.
   - A user noted that they turned off this feature claiming that *i'm stingy with my data... but PPLX is like.. bro... just answer the dam questions...*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1383506747855212757)** (8 messages🔥): 

> `Shareable Threads, Nvidia GB200, Driver Stability, Emergence, Android App Security` 


- **Make Threads Shareable!**: A member asked users to ensure their threads are `Shareable`.
   - A link to [how to make threads shareable](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825) was included.
- **Nvidia GB200 exascale AI supercomputer coming soon!**: A member shared a link about the [Nvidia GB200 exascale AI supercomputer](https://www.perplexity.ai/page/nvidia-gb200-exascale-ai-super-eQbdxUvkTO2NHGMvzvYDrQ).
   - It is expected to provide *"unprecedented performance and capabilities for AI workloads"*.
- **Nvidia 566.36 Driver Stability**: A member shared a link about [Nvidia 566.36 Driver Stability](https://www.perplexity.ai/page/nvidia-566-36-driver-stability-MfQyQNKUROa.dIBMOBuDvg).
   - The page gives tips on *"troubleshooting and maintaining optimal performance"*.
- **The Highly Alarming Emergence**: A member shared a link about [the highly alarming emergence](https://www.perplexity.ai/page/the-highly-alarming-emergence-VNgl7khJQ0GQDhWvN2F5Xg).
   - No further context was provided.
- **Perplexity Android App Security**: A member shared a link about [Perplexity Android App Security](https://www.perplexity.ai/page/perplexity-android-appsecurity-zDJ7FtsfQUiy9pmLAmt.Ngok).
   - No further context was provided.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1383492664883806330)** (7 messages): 

> `API credit charges, Perplexity Linux CLI client, AI startup resources` 


- ****API Credit Conundrums** Plague Users**: Users report that **API credit charges** exceed actual usage, seeking assistance through multiple channels like email (**support@perplexity.ai**), Discord, developers forum, and Intercom chat.
   - A member advised to send an email to **api@perplexity.ai** for support regarding the billing discrepancies.
- **Perplexity CLI client emerges**: A member shared their [Perplexity Linux CLI client](https://github.com/dawid-szewc/perplexity-cli).
   - The developer created an AI project for searching the web.
- **Startup Seeker Scours Sources**: A member expressed interest in building an **AI startup** and requested resources to learn more about Perplexity.
   - They seek guidance on using it for **web searching** and product development.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1383159907036954695)** (963 messages🔥🔥🔥): 

> `Kingfall vs Blacktooth, Grok 3.5 release, Gemini 2.5 Pro, Minimax M1 open source, LLM privacy` 


- **Kingfall Loses Favor to Blacktooth for Some**: Some users find **Blacktooth** to be a better model due to refined output and respect for the thinking process while others prefer **Kingfall** for its spatial abilities and magical moments.
   - Some suggest that Blacktooth is a *lmarena proc, not at all functional compared to kingfall when u look at coding, hence why the svgs were also not as high-fidelity*.
- **GPT-5 Release Timeline Sparks Debate**: Discussion revolves around when **GPT-5** will be released and if it will seal **Google's** fate as a side runner, with some predicting **Grok 3.5** or **GPT-5** will dominate for a while.
   - Users discussed if those who pay for **ChatGPT** notice **GPT-5** and suggested it might be for only a *few months or even weeks*.
- **Gemini 2.5 Pro's Prowess in Coding**: A user reported that **Gemini 2.5 Pro** is better at coding in pygame than **o4**.
   - Another user shared a [ChatGPT conversation](https://chatgpt.com/share/684e45eb-f118-8003-804e-3c9b562caab9) where **2.5 Pro** answered a logic question correctly after being told its previous answer was wrong.
- **LMArena Checkpoint Spamming Accusations Fly**: Some users on LMArena believe that big tech companies get an advantage due to the checkpoint spam and the opportunity for getting more data for rlhf.
   - A user said that LMArena is *basically USA big tech leaderboard* with open models or foreign models either not appearing or appearing extremely late.
- **Minimax Releases Open Source Reasoning Model**: Minimax released a new open-source large reasoning model, [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k), but early benchmarks show it being outperformed by **o3** and **Gemini 2.5 Pro**.
   - Some users reacted negatively, stating *it's either a fluke, they messed up, or its only capable of speaking chinese*


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1384197635946315898)** (1 messages): 

> `Models Erroring Out, Models not responding, Model API Issues` 


- **Models Errored, then Fixed!**: The team acknowledged a widespread issue causing models to error out instead of responding and promised a speedy fix.
   - The issue has since been resolved; users were encouraged to report any persisting problems.
- **Issue Resolved and Models Operational**: The team confirmed that the widespread issue causing models to error out has been resolved.
   - Users are advised to report any further problems or persisting issues after the fix.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1384288178181115994)** (1 messages): 

> `ChatGPT Image Generation, WhatsApp Integration` 


- **ChatGPT Images Invade WhatsApp!**: **ChatGPT image generation** is now available in **WhatsApp** via [1-800-ChatGPT](https://wa.me/18002428478).
- **Dial-a-DALL-E: WhatsApp Number Goes Live**: The **1-800-ChatGPT** number on **WhatsApp** is now live, enabling image generation for all users.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1383195125957656698)** (957 messages🔥🔥🔥): 

> `AI vs Human, GPT Plus Worth It, Sora video generation, Veo 3 video generation, GPT Model Performance` 


- **Spotting AI: More Art, More Problems**: Some members discussed how people are easily fooled when they think they can tell the difference between **AI-generated** and **human-created art**, especially when complexity and detail increase.
   - They compared this to counterfeit money, where *more detail leads to more scrutiny*.
- **GPT Plus: School's Cool?**: Members debated whether **GPT Plus** is worth the **$20 subscription** for school use, with some suggesting the free version with **GPT-4o** might suffice.
   - There was a consensus that **Plus** offers better models like **o3**, **4.1**, **4.5**, and **o4 mini high**.
- **Sora: Still a Sore Spot?**: Despite one member's proficiency, others critiqued **Sora** for being *really bad*, especially compared to **Veo 3** but another member mentioned liking it for feeling more *creatively tuned* and the details it allows.
   - A key advantage of **Veo 3** is its sound capabilities.
- **Veo 3 Steals the Show**: Members lauded **Veo 3's** ability to generate copyrighted content like **Star Wars**, with one stating that with **Veo** *you can stash a solid reference frame or style mask and feed it back on every pass, then bounce back into V3 for the final polish to keep things looking steady*.
   - Despite this, **Sora** is seen as a *one-stop shop* and a great value.
- **Performance Variance: Model Madness**: Members found that **GPT-4o** often outperforms **4.1** in certain tasks, with one noting, *4o got it right 10/10, 4.1 was 3-4/10*.
   - It was observed that removing spaces in prompts could significantly improve **4.1's** accuracy.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1383168931564355644)** (86 messages🔥🔥): 

> `GPT-4o's Memory, Fine-tuning GPT Models, Custom GPT Model Selection, DALL-E 3 Removal, Canvas auto updating` 


- **GPT-4o Accesses Previous Chat Data?**: A user described a situation where **GPT-4o** quoted *verbatim* from a scene co-authored in a separate chat with a **custom GPT**, leading to speculation about cross-chat memory access.
   - While some believe it's accurate inference, the user argued the statistical improbability suggests otherwise, inviting DMs for a conversation log.
- **Mini vs Nano: Choosing the Right Model for Fine-Tuning**: A user asked which model between **4.1 mini** or **nano** is better for mimicking a writing style through fine-tuning.
   - Another member suggested starting with a few hundred examples of **100-200 words**, noting diminishing returns after a quarter of a million words, but the user is willing to spend $15 to train on millions of words worth of content.
- **Custom GPT Model Options Expand**: Users noticed that custom GPTs now support a wider array of models, including **GPT-4o**, **o3**, and **o4-mini**.
   - One user found the **RAG** in custom GPTs superior to that in Projects, citing the June 12, 2025 release notes detailing expanded model support.
- **DALL-E 3 Image Generation Disabled?**: Members reported **DALL-E 3 image generation** might be disabled in ChatGPT and lamented the inferior quality of the new native image generator.
   - One user who just renewed their subscription expressed frustration, wishing OpenAI would keep the original DALL-E 3 available.
- **Canvas auto updating to Last Canvas?**: A user is looking for ideas about how to get chatgpt to access and update the correct **Canvas** and describes an issue where chatGPT automatically updates the last Canvas you made, instead of the first one.
   - Another member offered to help troubleshoot the **Canvas** issue via DMs, offering to replicate the problem and attempt a fix.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1383160731045986376)** (135 messages🔥🔥): 

> `Pandoc, HTML parsing, Sora AI prompting, O3 model prompting, GPT coherence` 


- **Pandoc Promoted for Polished Parsing**: A member suggests using [Pandoc](https://pandoc.org/) for converting HTML to Markdown, emphasizing it's a purpose-built tool, instead of using *awk* or other scripting tools.
   - Another member agreed it is better to use well-supported open-source tools to solve problems.
- **Scrubbing Strategies Spark Token Savings**: Members discussed how HTML tags create *noisy tokens* that might be beneficial for reasoning, but can increase token usage and cost in AI pipelines.
   - One member noted that while the semantic difference is negligible for single website queries, it can add up over scale.
- **Long-Form Yearning for O3 Prompts**: A member sought advice on prompting **O3** and **O3-pro** to generate long-form responses instead of concise, bullet-pointed summaries.
   - They noted that other models such as **Sonnet** and **Opus 4** did not have the same issue.
- **Sora's Style Showdown: DALL-E vs Code Prompts**: A member inquired whether **DALL-E** style prompts or code-style prompts are better for image generation on **Sora AI**.
   - The user's use case involves parsing and understanding complex academic research papers from web pages and applying the research to an ongoing debate.
- **Creative Coherence Crisis with Chatbots**: Members discussed methods for intentionally making **ChatGPT** lose coherence, with suggestions including absurdity, overuse of metaphor, excessive jargon, unhinged persona definitions, and contradictory guidelines.
   - One member recommended the **Burroughs' cut-up technique** to diagonalize the context, making the output dream-like.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1383160731045986376)** (135 messages🔥🔥): 

> `Pandoc vs awk for parsing, HTML noisy tokens, Long form responses from O3, Sora AI prompts for image generation, GPT coherence loss` 


- **Pandoc provides parsing prowess!**: Members discussed using **Pandoc** for converting HTML to Markdown instead of **awk**, highlighting its purpose-built design and widespread use.
   - One member emphasized the importance of using appropriate tools for parsing, suggesting reaching for a "swiss-army chainsaw" like **Pandoc** over basic tools for complex tasks.
- **HTML 'Noisy Tokens' may aid reasoning**: It was mentioned that HTML tags, although seemingly noisy, can actually be beneficial for reasoning in certain AI applications, especially at scale.
   - A member noted that while the token increase from tags is tiny, it could add up in large-scale operations, adding valuable context.
- **Optimizing O3 for Long-Form Output**: A user requested prompts that elicit **long-form responses** from **O3** and **O3-pro** when reviewing files or conducting in-depth research, as the models tend to be concise and favor bullet points.
   - The user noted they did not experience similar issues with **Sonnet** and **Opus 4** when using them to review files.
- **ChatGPT's Coherence Conundrums Explored**: Members discussed methods to intentionally induce **loss of coherence** in ChatGPT, including absurdity, metaphor overuse, jargon, unhinged personas, and contradictory guidelines.
   - Techniques like **Burroughs' cut-up method**, ADHD thought spirals, and fast speech were also suggested to diagonalize context and disrupt coherent outputs.
- **UPSUM to the Rescue: Saving Chat Context**: A member shared a **meta-prompt** called **UPSUM Chain Prompt** to produce updated summaries for seamless conversation continuation.
   - It was highlighted that LLMs might not retain the entire conversation history, necessitating the use of shorter chats and summary techniques like Chain of Density and UPSUM to manage and preserve context effectively.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1383164062417948895)** (575 messages🔥🔥🔥): 

> `GPU detection issues with Unsloth, Unsloth-DeepSeek-R1-0528-UD-IQ2_M benchmark results, Hugging Face model downloading issue, Unsloth fine-tuning notebooks, AMD compatibility with Unsloth` 


- **Unsloth benchmarks new Unsloth-DeepSeek-R1-0528-UD-IQ2_M**: A new **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** model achieved **69.4%** on test cases, with a speed of **426.0 seconds per case** compared to the API's **716.6 seconds**, but one member was concerned there was too much hype before the final results.
   - The model requires approximately **240GB** to load with **65k context**, making it more accessible locally compared to **7-800GB** for FP8, which the member found to be significant.
- **Hugging Face Naming Issue**: Users discussed a naming convention issue in the Hugging Face cache folder, where uppercase/lowercase differences cause [duplicate downloads](https://huggingface.co), wasting space.
   - The issue may be triggered when downloading a model using Unsloth and then using Hugging Face, leading to different naming conventions as the author may have changed the repo.
- **Fine-Tuning Tips Shared**: Members advised beginners to start with smaller models like **3B-8B**, emphasizing that *quality is greater than quantity* in datasets.
   - They also shared a [YouTube video](https://www.youtube.com/watch?v=jFl5Fewrieo) recommending new users spend **80%** of their time on data.
- **Unsloth's AMD Compatibility Nears Completion**: Unsloth is reportedly close to being fully compatible with AMD GPUs because most of Unsloth's kernels are written in **Triton**.
   - A member noted that while Triton compiles to AMD GPUs, the Triton settings might need optimization for AMD, potentially affecting performance, also pointing to [this PR](https://github.com/unslothai/unsloth/pull/2520).
- **Goodbye Reddit, Hello X**: Users expressed dissatisfaction with Reddit due to issues like a bad automod system, lack of control over posts, and the prevalence of biased moderation.
   - One user cited these reasons for deleting their Reddit account, suggesting [Twitter](https://twitter.com) (X) as a better alternative for blogging, monetization, and news, emphasizing that X is free of bots, and is just a social platform, so avoid hate and politics.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1383807532275077171)** (13 messages🔥): 

> `KL Divergence Spikes, Google Colab GPU Pricing, TempleOS, Hugging Face Outage` 


- **KL Divergence Randomly Explodes, Then Calms Down**: A member inquired about **KL divergence** sometimes spiking to **x10000** for a single step before returning to normal, a behavior that doesn't seem to impact training.
   - Another member mentioned this occurs frequently, even in Hugging Face runs without Unsloth, possibly due to particular token values exploding during **logprob subtraction and exponentiation between acting and reference policies**.
- **Sweet Spot for Google Colab GPU Prices**: A member asked where the *sweet spot* is for **GPU prices on Google Colab for fine-tuning**, considering the balance between speed and credits usage.
- **TempleOS gets discussed off topic**: A member asked if anyone else liked **TempleOS**.
- **Hugging Face seems down**: Members reported **Hugging Face** being down and shared an [image](https://cdn.discordapp.com/attachments/1179039861576056922/1384238073264476252/image.png?ex=6851b3aa&is=6850622a&hm=b46b4e20fbde30128158ce97f3652c6d3e6462e4ffd91be145958bfa03a6afc4) depicting the feeling of having to hug your own face in response to the outage, with a [link to a relevant GIF](https://tenor.com/view/sad-gif-17629775327580673254).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1383161455225868308)** (266 messages🔥🔥): 

> `Qwen2.5 vs Qwen3, GGUF conversion, DPO vs SFT, Gemma3, Llama 3.2` 


- **Qwen2.5 or Qwen3 to integrate coding assistant**: A member needed a quick way to integrate a coding assistant with **R/RStudio** and asked about **Qwen2.5 Coder 32B Instruct GGUF**, but was advised to use the **Qwen3** package from Unsloth instead, available [here](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF).
   - The member plans to create a new model based on **Qwen 3** with *no_think* as the default to embed in their workflow, rather than using the instruct model.
- **SFT or DPO**: **DPO** is better for controlling *HOW* the model responds, while **SFT** with a bit of upsampling is the way to go if you want the model to respond with specific information such as the model's name.
   - This was in response to a question on what to do if a model needed to give a specific answer when asked a question such as, *What is your name?*
- **Gemini 3 errors need fix**: Users reported a `dtype` mismatch error when working with Gemma 3, the error is *expected mat1 and mat2 to have the same dtype, but got: float != c10::Half unslot gemma*, with members suggesting it may be related to **bfloat16** precision or the GPU used.
   - A member is currently working on a fix for the errors encountered and suggested trying the installation instructions from [this link](https://discord.com/channels/1179035537009545276/1179035537529643040/1381978312640958515) to resolve the issue, as well as, replacing the default *pip install* commands with the force reinstall commands from the repo to receive latest fixes.
- **Help converting Llama 3.2**: A member asked how to install **Unsloth's Llama-3.2-11B-Vision-Instruct** model to `ollama` and was informed that pre-made GGUF versions can be found [here](https://huggingface.co/pbatra/Llama-3.2-11B-Vision-Instruct-GGUF/tree/main) for manual conversion to GGUF.
   - A user posted a link to [official ollama instructions](https://github.com/ollama/ollama/blob/main/docs/import.md) for converting to GGUF, and also suggested pulling the model directly from the [ollama library](https://ollama.com/library/llama3.2-vision:11b).
- **New fixes**: New fixes have been pushed which are available if you install the updated code from the repo directly which requires to update to the code from the repo main (instead of pypi).
   - It was suggested that installing from the main repo directly might solve a problem related to re-merging an adapter. The link provided details how to install Unsloth in your PC.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1384192396769755287)** (2 messages): 

> `arxiv link` 


- **AI Paper Shared**: A member shared an [arxiv link](https://arxiv.org/abs/2506.09991).
- **Confirmation of Shared Resource**: The member acknowledged that the resource was shared before they could do so themselves.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1383163182985515068)** (750 messages🔥🔥🔥): 

> `Claude 4 Sonnet Performance, Cursor UI Issues, MCP Usage, Code Privacy, Bug Reporting` 


- **Claude 4 Sonnet users observe slowness**: Members have noticed that **Claude 4 Sonnet** runs significantly slower in Cursor compared to platforms like GitHub Copilot, despite its general effectiveness and [stability](https://link.to/stability).
   - Some members suggest **optimizing usage** by reserving **Max Mode** for major refactoring tasks or exploring alternatives like **Gemini 2.5 Pro** for planning and code reviews to better manage inference credits.
- **Users Complain about Cursor's UI Flaws**: Users report ongoing UI problems, such as **command execution failures** on Windows due to *xterm bracketed paste* issues and inconsistent notifications about command completion, resulting in wasted inference credits.
   - One member noted that about **10-15% of their credits are wasted** due to these UI malfunctions and suggested that Cursor provide inference counts back when errors occur.
- **Navigating Model Context Protocol Usage in Cursor**: A member sought advice on using **Model Context Protocol (MCP)**, and they highlighted the benefit of the AI's ability to leverage screenshots and integrate error messages automatically.
   - Another user emphasized the importance of spending time on **defining better prompts**, citing it as more effective than frequent screenshotting and copy-pasting, suggesting [Wisprflow](https://wisprflow.ai) for enhanced speech-to-text capabilities.
- **Users Request Granular Code Privacy Settings**: Users expressed a need for setting code privacy on a **per-repository basis**, allowing different settings for work and personal projects due to concerns over code storage and accessibility.
   - Currently, Cursor’s **Privacy Mode** is a *global setting*, but the community desires more granular control at the project level for enhanced flexibility and security as they want to avoid unintentionally opening sensitive company directories.
- **Streamlining Bug Reporting with Active Monitoring**: Members are actively sharing bug reports and troubleshooting tips within the community, particularly focusing on issues like the broken **command execution tool on Windows**.
   - The community is pushing for **more active testing on Windows** and better communication from the Cursor team regarding bug fixes and feature rollouts after a member noticed the **Cursor Task Master** which is actually a community third-party project not officially released.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1383273362230345919)** (40 messages🔥): 

> `GitHub integration, Background Agents Permissions, Background agents and Slack, Background Agents and Privacy Mode, Toggle bug with background agents` 


- ****GitHub Integration Roadmap****: A user inquired whether agents can access **GitHub issues**, to which a member replied that it's currently not possible but is on the roadmap.
   - The member also clarified that the **GitHub integration** improvements are coming soon.
- ****Background Agents Lack Permissions to Create PRs****: A user reported that background agents in Slack don't create pull requests, despite granting all permissions in Cursor integration settings, generating request IDs: **bc-79e56de2-26d7-41a0-a5b3-b8b9e9f635d1** and **bc-d3acc5d6-2520-413f-9f7c-2fb36590215d**.
   - A member offered to debug the issue and asked for the request ID.
- ****Background Agents Require New Privacy Mode****: A user noticed that the old Privacy mode is now labeled "legacy" and inquired whether enabling Background Agents requires the new Privacy mode (with code storage).
   - A member confirmed that **Background Agents** require the new privacy mode with code storage, because the original privacy mode doesn't permit storing code for the lifetime of a background agent, which is required to execute and iterate on code.
- ****Background Agents Toggle Bug Reported****: A user reported a toggle bug with background agents, providing a [video](https://cdn.discordapp.com/attachments/1367213641027551352/1383486605423022311/CleanShot_2025-06-14_at_18.40.43.mp4?ex=68519ace&is=6850494e&hm=cf854daed7b862bbf53d10101cd431f5bcac0069bbfc234f0240749fbda7ddfa&) demonstrating the issue.
   - A member responded, offering to investigate and requesting the user to check their [GitHub installations](https://github.com/settings/installations/) for the account they want to connect.
- ****Cursor Not Listed Under Installed GitHub Apps****: A user discovered that Cursor was listed under "Authorized GitHub Apps" but **not** under "Installed GitHub Apps" for their personal org, whereas for an org where Background Agents were working, Cursor **was** listed as installed with access to all repos.
   - The user was directed to reconfigure/enable/disable repos and orgs at [Cursor's dashboard](https://www.cursor.com/dashboard?tab=background-agents), via the "Manage" external link for GitHub, to resolve the issue.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1383159564391682129)** (553 messages🔥🔥🔥): 

> `AI-Generated Feedback, Open Sourcing AGI, Bigram Testing, Qwen 2.5, HF Pro Disk Space` 


- **Ethical Concerns Over AI-Generated Content**: A member expressed unease about [AI-generated feedback](https://example.com), citing ethical reasons for not engaging with AI-generated images or video work.
   - The member stated *I don't engage with AI generated images/video work even though I have the theoretical knowledge*.
- **Open-Sourcing AGI Debate Flares Up**: A member humorously declared that if they ever create the world’s first **AGI**, they will [open source it](https://example.com).
   - This sparked a discussion about the potential benefits and risks of open-sourcing such a powerful technology.
- **Qwen 2.5 Impresses with Efficiency**: A member lauded **Qwen 2.5**, a 7b model quantized to q4 on Ollama, for its impressive performance given its small size and simple system prompt, with another noting that *No one showed in comparison in benchmark against qwen 2.5 Cause it was so good*.
   - It was also discussed that **Qwen models** are pretrained on multilingual datasets including Chinese and English.
- **Rate Limiting and Zero GPU Quota Anomaly**: Users reported issues with **rate limiting** on Hugging Face and some reported getting **extra Zero GPU Quota**.
   - It was speculated that the extra GPU Quota might be a special provision for old users, but no official announcement was made.
- **AI-Assisted Coding Gains Traction**: The members discussed their experiences with **AI-assisted coding tools**, like Gemini, praising their ability to generate understandable and modifiable code.
   - One member shared that they *vibe coded for the first time something for IOS, completely 0 knowledge how it works and still have no clue ...but it does what it supposed to do*.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1383864388615802960)** (2 messages): 

> `HF audio course, Agents course, MCP course` 


- **User kicks off HF audio course**: A new member announced they are starting the **Hugging Face audio course** today.
   - No links or resources were mentioned.
- **Member learns Agents and MCP courses**: A member is currently learning **Unit 2 of the Agents course** and has started **Unit 1 of the MCP course**.
   - No links or resources were mentioned, only a <:hugging_rocket:968127385864134656>.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@844851718512443423> No referrals please
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1383234217021800631)** (9 messages🔥): 

> `peft-bench, InfiniGPT French Q&A dataset, Shisa AI Japanese model, Swiftide Rust library for agentic RAG applications, QuantIntelli Football Betting Analysis` 


- ****InfiniGPT** - The largest French Q\&A dataset is here!**: A 19-year-old student released **InfiniGPT**, a French Q\&A dataset featuring **40,000+ Q\&A** entries, 100% native French, manually verified, diverse topics, and fine-tuning ready, aiming to establish a French dataset standard ([GitHub](https://github.com/RDTvlokip/InfiniQA), [HuggingFace](https://huggingface.co/datasets/RDTvlokip/InfiniQA)).
   - The author claims it is *5x bigger than FQuAD* and offers *direct Q&A*, not extractive reading, with documented sources and GPT-2 tokenizer optimization.
- ****Shisa v2** - Japan's strongest model released!**: A team of 2 released **Shisa v2**, the strongest model ever trained in Japan, along with an Overview Report (Technical Report forthcoming) available on [HuggingFace](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) (800GB+).
   - They also updated their core SFT dataset (available at [HuggingFace](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt)), claiming it improves Japanese performance without reducing English performance, based on training/releases on sota open models from **7B-405B**.
- ****Swiftide 0.27** - Rust library ships!**: A major release for **Swiftide** was shipped, which is an open-source library in Rust to build composable agentic and RAG applications ([announcement](https://bosun.ai/posts/swiftide-0-27/)).
- ****QuantIntelli** - Hybrid AI Agent Predicts Football!**: A Hybrid AI Agent for Quantitative Football Betting Analysis was created, combining **XGBoost** model and **Google Gemini LLM** with features like Advanced RAG Pipeline using Tavily, Google, and DuckDuckGo, persistent session logging with Supabase, and an interactive UI with Gradio ([HuggingFace Space](https://huggingface.co/spaces/ChillThrills/QuantIntelli), [Github Repo](https://github.com/IanDublew/QuantIntelli)).
- ****JASCO** - Music generation on MCP server!**: Users can now generate musical stems using **facebook/jasco** via MCP server, which generates two variations of music based on text descriptions, chord progressions, and optional melody and drum inputs ([HuggingFace Space](https://huggingface.co/spaces/Tonic/audiocraft)).
   - Instead of recording input audio with the mic, now you can generate drum outputs in ~1 second for gary to continue, via **stable-audio-open-small**, and *name it jerry lol*.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1383861459695435776)** (2 messages): 

> `Portfolio Theory, Dr. Peter Cotton, Schur Portfolios` 


- **Cotton Proposes Presentation on Portfolio Theory**: A member suggested arranging for **Dr. Peter Cotton** to present his paper on portfolio theory, linking to the [Schur Portfolios paper](https://github.com/microprediction/home/blob/main/papers/schur_portfolios.pdf).
   - They inquired about the process to organize such a presentation.
- **Schur Portfolios Paper Presentation Proposed**: A member proposed a presentation on **Dr. Peter Cotton's** paper, '[Schur Portfolios](https://github.com/microprediction/home/blob/main/papers/schur_portfolios.pdf)', focusing on portfolio theory.
   - The proposal included a request for guidance on organizing the presentation.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1383706186393587798)** (3 messages): 

> `Smolagents, Ollama, Code Agents, Local Model Selection` 


- **Smolagents loses compatibility with Ollama**: A member reported that **smolagents** is no longer compatible with **Ollama** for local **code agents**.
   - The member is seeking assistance to implement a local code agent.
- **Request for model recommendation on limited resources**: A member requested a recommendation for the best model to run locally with **8GB RAM** and **6GB VRAM**.
   - They used **smolagents** for a final project, spending **$10** on the OpenAI API to achieve **45%** accuracy; [Project Link](https://huggingface.co/spaces/renwei2024/agent-course-final-project/tree/main).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1383228313211899914)** (21 messages🔥): 

> `HF Inference Costs, Local LLMs with Ollama, Unit 3 Assignments, Agentic RAG Locally, Unauthorized Imports` 


- **Users grapple with HF Inference Costs**: Several users are running into cost issues with **HF Inference**, especially when using models like **llama-3-70b-Instruct**, finding the free credits insufficient.
   - One user reported paying around **$6** after many attempts on the final unit and suggested using local models to mitigate costs.
- **Ollama enables running LLMs Locally**: Members are discussing using [**Ollama**](https://ollama.com/) to run **LLMs locally** and then plugging them into agents, thereby reducing reliance on paid inference APIs.
   - The cost savings can be significant, but one user felt that the final unit assignment was too challenging, suggesting a steeper learning curve.
- **Feedback on Unit 3 Assignments**: A user expresses that the final unit feels like being thrown into the *deep end*, wishing for more assignments like it throughout the course.
   - They also noted that many leaderboard submissions appear to be copied, undermining the purpose of the exercises.
- **Debugging Agentic RAG Locally**: A user encountered an error trying to run [Unit_3_Agentic_RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG) locally and posted a screenshot of the error message.
   - No specific solution was provided in the discussion, but the issue seemed related to setting up the environment correctly.
- **Bizarre Unauthorized Imports**: A user reported issues with **CodeAgent** flagging certain imports like **plotly.express** as unauthorized, even after specifying **plotly** as an authorized import.
   - Another user confirmed similar experiences, noting that sometimes using aliases (e.g., **bs4** instead of **beautifulsoup4**) can bypass the restriction, while confirming that adding `plotly.express` solves the user's problem.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1383513332539199689)** (3 messages): 

> `Chess Leaderboard, Book Testing` 


- **Chess Leaderboard Replay Functionality**: A member added homebrew replay functionality to [chess-leaderboard](https://dubesor.de/chess/chess-leaderboard) for every game (past and future).
   - They mentioned that it's *maybe a bit better than lichess gifs, but a pain to implement with my stack* with an attached [chessreplay.gif](https://cdn.discordapp.com/attachments/1092850552192368710/1383513332916424715/chessreplay.gif?ex=6851b3b3&is=68506233&hm=1039e8ba4df4c19b19fc9c28f054c1eb809659a82c942b46aa7daebc3b48088c&).
- **Book Testing Opportunity**: A member mentioned their book is live as of **June 15** and is offering to help with testing.
   - They said they are *happy to help* if anyone shoots them a DM.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1383178098874519672)** (533 messages🔥🔥🔥): 

> `OpenRouter Discord Tag Request, Claude Prompt Debugging, GPT-4.1 Mini Offering, Free Model Credit Usage, Multilingual Model Recommendations` 


- ****Discord Tag Craving Unacknowledged****: A member expressed frustration over an unacknowledged request to restore the **OpenRouter Discord tag**, offering to pay for it despite **OpenRouter's** financial superiority.
   - They jokingly threatened to ping **Alex Atallah** due to the lack of response.
- ****Claude & Copilot Prompts lack Token Economy****: A member debugged prompts from **Claude Code** and **GitHub Copilot**, finding they often ignore token efficiency when improving workflows, sending irrelevant content unless verbosity affects performance.
   - They observed that conciseness isn't a primary goal for these systems when adjusting prompts.
- ****Testing a GPT-4.1 mini version on OpenRouter****: A member offered access to **GPT-4.1 mini** with **200K tokens/minute** available at **20% of the official token price**, compatible with the OpenAI SDK, inviting high-usage testers to DM for more details.
   - They highlighted its suitability for apps like Cline.bot and BYOK/BYOB setups.
- ****Deepseek's Free Tier Suffers Outages****: Users reported encountering **502, 503, and 524 errors** when using the free version of **Deepseek-r1-0528** through the API, with one suggesting the issues may stem from high traffic due to *smut RPs*.
   - Members noted the paid version remained functional and discussed potential causes, including data center problems or issues with **Chutes**.
- ****OpenAI Faces Antitrust Threat from Microsoft Spat****: Discussions revealed that **OpenAI** executives have considered accusing **Microsoft** of **anti-competitive behavior** during their partnership, potentially seeking regulatory review and launching a public campaign.
   - This arose from difficult negotiations, prompting reactions of surprise and concern from community members.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1383164338155688089)** (247 messages🔥🔥): 

> `TokenBreak Attack, AMD Ryzen AI Mini PC, Increasing RAG size in LM Studio, MiMo VL 7B RL UD support, LLM Image Organizer` 


- **TokenBreak Attack Bypasses AI**: A member shared an [article about a new TokenBreak attack](https://thehackernews.com/2025/06/new-tokenbreak-attack-bypasses-ai.html) that bypasses AI security measures, though they didn't get the same results from their experiment.
   - Another member jokingly noted how similar the logos in the attached screenshot were.
- **AMD Ryzen AI Mini PC runs big models**: A member mentioned that the [AMD Ryzen™ AI Max+ 395 --EVO-X2 AI Mini PC](https://share.google/3WVLR8Up7pjVjq35mI) can run some big models smoothly.
   - Others countered that it's *a glorified igpu* supported by HIP SDK and that while a **70B** model runs at ~**5t/s**, a **Qwen3 235B A22B** model runs at ~**14t/s**.
- **Increase RAG size is not possible in LM Studio**: A member asked how to increase the **RAG size** from **31.46 MB** to **100 MB** in LM Studio.
   - Another member responded that *it is not possible*, RAG is still a basic implementation.
- **The solution to long detailed LLM responses: narrative games**: One member suggests starting a *choose-your-own-adventure game* with your LLM, ensuring *it won't attempt to finish the story in a single response*.
   - They suggest prompts like `Let's play a choose-your-own-adventure game. I'll start with a prompt and you carry the story on. When you reach a decision point, list a few choices for direction and I'll respond.`
- **Local LAN port opening is low risk**: A member asked about security risks when opening ports, which caused concern about *said site as the front* implying opening the backend to the internet and creating security exploits.
   - Ultimately, members agreed that opening ports on a **local LAN network** is low risk, though any open port *can be exploited*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1383166191920484522)** (95 messages🔥🔥): 

> `GMKtec Windows install issues, RTX 6000 Pro wattage configuration, Graphics cards and coil whine, NVLINK performance experiments, GPU Recommendations for LLMs` 


- **Windows install on GMKtec is a PITA**: A user reported issues installing **Windows** on their **GMKtec** machine, citing installation failures and problems with **Rufus**-created removable media.
   - They are trying to install Windows on a GMKtec machine.
- **RTX 6000 Pro can shapeshift into 300W or 600W**: The **RTX 6000 Pro** can be configured for either **300W** or **600W**, according to one member.
   - The standard one can be configured, not sure which one is used in this build though.
- **Graphics Cards Sing the Coil Whine Blues**: Users have observed a significant increase in **coil whine** from their graphics cards when running **LLMs** compared to gaming.
   - One user noted getting more coil whine with a **5090**, suggesting undervolting as a solution to reduce both power consumption and coil whine.
- **NVLINK Performance remains untested by many**: A member inquired about experimental data on **NVLINK's interference performance difference**, wondering if it provides a tangible benefit.
   - Another memebr posted an image that said "I'm also sure nvidia's software is HIGHLY optimized for nvlink".
- **GPU Shopping List: 3090-4090-5090, you can't afford it**: For running models like **Qwen3**, **Devstral**, and **Gemma3**, the **3090**, **4090**, and **5090** were recommended due to their **24-32GB** VRAM, especially for larger models or higher quality quants.
   - The 3090 is about as good as the 4090 and it keeps up with 5000 series cards with 24 GB. The 5090s are about $3k. For that price, you’ll probably still run a 32B or smaller model, but a higher quality quant or more context


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1383246229907177592)** (6 messages): 

> `PD disaggregation, Transformer moment for agents, Groq speed, Groq Huggingface` 


- **PD Disaggregation Resources**: A member requested resources on **PD (power distribution) disaggregation**, other than the [DistServe paper](https://arxiv.org/pdf/2401.09670).
   - No resources were offered in the provided messages.
- **Transformer Moment for Agents**: A member inquired about the **"transformer moment"** for agents, seeking a general-purpose control strategy that adapts to any task automatically.
   - They wondered if it could be **DFS, BFS, or hybrid flows**—automatically selected.
- **Groq's Speed and Collaboration with Hugging Face**: A member asked how good **Groq** is and how they are so fast.
   - Another member mentioned that **Groq** recently collaborated with **Hugging Face**, implying positive performance or accessibility; no explicit link was offered.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1383293498358235268)** (9 messages🔥): 

> `tl.constexpr behavior with expressions, Thread-level control flow in Triton, Triton kernel warmup time vs torch.compile, Single-row softmax kernel implementation` 


- **tl.constexpr Expression Woes!**: A user found that using `tl.constexpr` in an expression (e.g., `a = tl.constexpr(b // 2)`) causes errors with `tl.arange` because it doesn't recognize `a` as a constexpr, while `tl.arange(0, b // 2)` works fine; the solution is to type it as `a: tl.constexpr = b // 2`.
   - The user provided a [minimal reproduction example](https://github.com/NVIDIA/apex/blob/main/apex/transformer/loggers/csv_logger.py) showing the error during compilation when `tl.arange`'s arguments are not explicitly defined as `tl.constexpr`.
- **Looping at Thread-Level?**: A user inquired about thread-level control flow in Triton, seeking to implement a loop that sums matrix rows and stops when the sum exceeds a threshold.
   - No responses were given.
- **Triton Kernel Slow Start?**: A user reported that their handwritten Triton kernel takes warmup time to reach peak performance, unlike `torch.compile`, and wondered if `torch.compile` uses better heuristics for block size or other optimizations.
   - No responses were given.
- **Softmax Race Condition**: A user working on a [single-row softmax kernel](https://pytorch.org/tutorials/stable_diffusion/stable_diffusion.html) faces a race condition in the final kernel that writes the softmax results, as the first program overwrites the initial global max and sum.
   - No responses were given.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1383205626838257887)** (55 messages🔥🔥): 

> `CUDA cache policies, TF32 vs FP16 precision, L40 vs 4090 performance, nvcc generating LDS instruction, GCC and NVCC version compatibility` 


- **CUDA Cache Policy Gets Fractional**: A member shared a [CUDA code snippet](https://forums.developer.nvidia.com/t/sm89-cache-policy-hints-not-respected/281749) for creating a cache policy object using `createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64` and using it in a load instruction with cache hints, inquiring about its usage.
- **TF32 Precision Woes**: A member observed that **TF32 matmuls are 3x less precise than float16** on CUDA GPUs, tested on 4070 mobile and A10, and shared a [Triton code snippet](https://github.com/openai/triton) to reproduce the issue.
   - They pointed to a potential cause in a [related Triton issue](https://github.com/triton-lang/triton/issues/4574) regarding precision.
- **L40s Underwhelming: ECC to Blame?**: Members discussed that **L40s** may seem underwhelming in cloud environments due to **ECC** being activated by default, impacting performance, citing it as a configuration issue rather than a hardware problem.
- **nvcc accidentally generates LDS instruction**: A member reported that `nvcc` generates an unintended **LDS** instruction for data in global memory, causing errors when using `compute-sanitizer`, and that using `__ldg` fixes the issue.
   - Others suggested it could be undefined behavior and requested a minimal, reproducible example to further investigate the possible compiler bug.
- **GCC+NVCC Version Combo Causes Choke**: A beginner encountered an error related to parameter packs not expanded in `<std_function.h>`, and it was suggested that this is due to an incompatibility between **GCC** and **NVCC** versions, with **CUDA 11.7.0** being the first to officially support Ubuntu 22.04.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1383668795494633482)** (2 messages): 

> `CUDA kernel blocksize args, TorchTitan training graph capture` 


- **CUDA Kernel Blocksize Args Best Practices Needed**: A member is seeking best practices for transferring **blocksizes** in Python to CUDA code without JIT, using Torch's cpp_extensions + setuptools to compile custom CUDA kernels.
   - They're looking for an alternative to explicitly adding **blocksize** as an int[] parameter in TORCH_LIBRARY registration, as that doesn't seem as elegant since most PyTorch functions don't expose **blocksize** args at all.
- **Graph Capture with Torchtitan in Training**: A member is training a **llama1b** with Torchtitan and wants to capture the training graph(s) with the various collectives when working with different parallelism combinations.
   - They tried to intercept a training step and use the **aot_module** in functorch.compile to capture it but think the **Faketensor** propagation is not working with it.


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1383490434680229959)** (2 messages): 

> `d-Matrix team, Dr. Lisa Su, GPU MODE, kernel data, Project Popcorn` 


- **d-Matrix Team to demo Custom Silicone**: The **d-Matrix** team will demo their custom silicone for **low latency batch inference**.
- **Dr. Lisa Su Shout-Out GPU MODE**: Dr. Lisa Su called out **GPU MODE** and its work enabling the world's first $100K competitive kernel competition at [gpumode.com/news](https://www.gpumode.com/news).
- **Kernel Competition Generates Massive Kernel Data**: The community generated more **kernel data** than exists on all of **Github** combined, outperforming the best baselines by human experts.
- **Project Popcorn Collaborations**: GPU MODE thanked collaborators at **AMD** and on [Project Popcorn](https://gpu-mode.github.io/popcorn/).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1383181373870833845)** (1 messages): 

> `Instruction latencies, arxiv.org` 


- **Instruction Latencies Paper Shared**: A member shared a link to a paper on instruction latencies: [https://arxiv.org/pdf/1903.07486](https://arxiv.org/pdf/1903.07486).
   - The member noted that the **instruction latencies** may be outdated, but the **discussion** is still worth reading.
- **Paper discussion on instruction latencies.**: The paper at [https://arxiv.org/pdf/1903.07486](https://arxiv.org/pdf/1903.07486) discusses instruction latencies.
   - The discussion is considered valuable, despite potentially outdated instruction latencies.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1383285235676090460)** (5 messages): 

> `MX-FP4 Matmul, MX-FP8 Matmul, CUTLASS, CuBLAS, FP4 Weight Quality` 


- **CUTLASS and CuBLAS shine on 5090**: **MX-FP4 matmul** from CUTLASS and **MX-FP8 matmul** from CuBLAS (via `torch._scaled_mm()`) are very impressive on 5090 ([PR 2285](https://github.com/pytorch/ao/pull/2285)).
- **Weight-Only FP4 kernel not available yet**: A member inquired about benchmarks for smaller batches (1-64) with **fp4 weight-only** and there isn't a weight-only kernel for fp4 yet.
- **Good Perf with Weight-Only FP4 coming soon?**: A member stated they got a pretty good perf with **weight-only FP4**, and will try to make some time to put the code together for integration.
- **FP4 Weight quality is bothering some members**: The quality of **FP4 weights** is bothering some, as there's a noticeable drop in accuracy converting the weights, so it would need some quant algo to improve accuracy (mx-hqq ? 👀 ).


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1384163143118487724)** (1 messages): 

> `HQQ Rebrand, Quantum Quantization` 


- **Rebrand HQQ for Quantum**: A member suggested rebranding HQQ to *Half **Quantum** Quantization* to attract more interest.
   - This followed a [Multiverse Computing raise of $21.5M](https://multiversecomputing.com/resources/multiverse-computing-raises-usd215m-to-scale-ground-breaking-technology-that-compresses-llms-by) to scale technology that compresses LLMs.
- **Quantum Computing Funding**: Multiverse Computing recently secured **$21.5M** in funding.
   - The funding aims to scale their technology for compressing LLMs, potentially benefiting projects like HQQ.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1383973154837233786)** (19 messages🔥): 

> `Nvidia to AMD transpilation, AMD stable inference deployment, MI300A architecture, IODs and infinity cache, memory distribution` 


- **Nvidia transpilation to AMD is CASS**: A member shared a [paper](https://arxiv.org/pdf/2505.16968) on **CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark** to try **RF-ing** a model with **GRPO** instead of **SFT**.
   - The member is investigating whether the performance thresholds/differences are realistic in real-world workflows.
- **AMD Stable Inference Deployment is Ollama**: A member was looking for tips for stable inference/deployment libraries on **AMD**.
   - It seems that [*Ollama works*](https://ollama.com/) so they were just overthinking it.
- **Diving Into MI300A's Architecture**: Members discussed the fused **AMD CPU-GPU platforms**, especially the **IOD** and **infinity cache** of the **MI300A** architecture.
   - They are wondering if there was a way to test a particular Path or pressure one or the other IOD.
- **Memory Chips' Distribution Strategy Explored**: Members speculated on how memory is distributed between the memory chips and how this affects latency, particularly on the **MI300X**, where each **IOD** is connected to 2 **HBM stacks**.
   - One member mentioned using `s_getreg` to figure out which **XCD** and **CU** a shader is running on, and from that, measuring access latency to different locations in memory.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1384226368564367413)** (4 messages): 

> `Thrust library, CUDA kernels, segmented sum algorithm, iterators, high_resolution_clock vs steady_clock` 


- ****Veitner** Introduces **Thrust** Library**: Simon Veitner introduced [Thrust](https://veitner.bearblog.dev/an-introduction-to-thrust/), a high-level abstraction layer that allows you to write performant **CUDA kernels** using modern **C++** concepts.
- ****High_resolution_clock** gets benched!**: A member suggested not using `high_resolution_clock` but rather `steady_clock` for benchmarking, referring to [this stackoverflow answer](https://stackoverflow.com/a/37440647/10107454).
   - They added that, *given enough periods and therefore parallelism, this should not be a problem*, however *what actually kills the performance even then is the strided/uncoalesced memory access*.
- ****cuTensor** is more fitting library**: For a regularly sized example, a member suggests that **cuTensor** might be a more fitting library than Thrust.
- ****MatX** makes multidimensional algorithms more elegant**: A member recommended **MatX** for an elegant C++ interface to these kinds of multidimensional algorithms.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1383958902101508157)** (6 messages): 

> `Tensor Core Algorithm Reformulation, RL for Tensor Core Usage, Kernel Code Verification, GPU Thinking Interpretability` 


- **Algorithm Reformulation for Tensor Cores**: A user inquired about a feedback loop involving a domain expert to reformulate algorithms into tensor-core forms, especially beyond simple cases like **FFTConv**, considering modifications like padding and rank-factorization.
   - They sought guidance on steering experts towards tensor-core-friendly algorithm design.
- **Reinforcement Learning Guides Tensor Core**: One member suggested using **Reinforcement Learning (RL)** to guide models in utilizing tensor cores, by building a small verifier to check a model's trace for tensor core understanding.
   - They pointed to [Hugging Face Kernels](https://huggingface.co/blog/hello-hf-kernels) as a potential data source, emphasizing its community-driven contributions.
- **Creative Kernel Code Verification Ideas**: One user is experimenting with verifying kernel code through a **Triton interpreter**, instead of full execution, for quicker verification and better scalability in data quality and RL efforts.
   - This approach provides easier insight into memory and instruction calls within a CPU environment.
- **"Thinking GPU" can be interpretable**: Members discussed applying methods from **natural language interpretability** to program languages by creating probing classifiers per layer based on paper presentation at [ICSE-NIER '25](https://www.computer.org/csdl/proceedings-article/icse-nier/2025/371100a086/27t2knJTWso).
   - The goal is to show that a model exhibits "GPU thinking" by using different layers and attention heads when solving a problem with a GPU approach versus a CPU approach, analyzing internal representations after initial translation projections.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1384220469351415980)** (4 messages): 

> `ThunderKitten on Older GPUs, TK to AMD port, Attention Kernels with Variable Length` 


- **Run ThunderKitten on Older GPUs Like T4 and P100?**: Members discussed the possibility of running **ThunderKitten** on older GPUs like **T4** and **P100** available on Kaggle, noting it is likely doable despite challenges with async instructions and smaller shared memory.
   - One member suggested compiling with a **4090 TARGET** and reporting any breakages to help with compatibility.
- **TK's Port to AMD: Awaited Soon!**: The team is actively developing a **TK-to-AMD port**, aiming for an imminent release to expand compatibility.
   - The lack of async instructions is generally a bit annoying, and shared memory is smaller so we'd need more pipelining on the register front vs. the Nvidia megakernels.
- **ThunderKitten Attention Kernels Support Variable Length**: The ThunderKitten repo includes attention kernels that support **variable length** and **padding**, which can be helpful in various sequence processing tasks.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1384266860664590529)** (1 messages): 

> `Chain of Thought, CoT, Symbolic Reasoning, Math Reasoning` 


- **CoT Benefits Pinpointed by New Research**: A recent paper, *To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning* ([arxiv link](https://arxiv.org/abs/2409.12183)), investigates the scenarios where **Chain-of-Thought (CoT)** reasoning provides the most advantages, specifically in **math and symbolic reasoning** tasks.
- **Math and Symbolic Reasoning Excel with CoT**: The study indicates that **Chain-of-Thought (CoT)** primarily enhances performance in **mathematical and symbolic reasoning** domains, offering insights into its limitations and strengths.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1383188973261684826)** (16 messages🔥): 

> `MI300 AMD-FP8-MM, Conv2D on H100, VectorAdd Leaderboard Updates` 


- **MI300 AMD-FP8-MM gets a Speedy Submission**: A submission on the `amd-fp8-mm` leaderboard for **MI300** achieved **5.23 ms**.
   - Another submission reached **9th place** with a time of **161 µs**.
- **H100 Conv2D keeps churning**: Multiple successful submissions were made to the `conv2d` leaderboard on **H100**, with times around **187-192 ms**.
   - These submissions indicate consistent performance on the **H100** for `conv2d` tasks.
- **VectorAdd sees a ton of activity**: Many submissions updated the `vectoradd` leaderboard across various GPUs (**A100**, **H100**, **T4**, **L4**), with times ranging from microseconds to milliseconds.
   - One submission achieved **third place** on **T4** with **6.31 ms**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1383161516639125617)** (162 messages🔥🔥): 

> `Factorio RL, Factorio Agents, Hierarchical RL and LLMs, FLE API` 


- **DOTA vs. Factorio gameplay difficulty debated**: Members debated whether professional **DOTA** play is harder than reasonable **Factorio** gameplay, citing **DOTA's** heavy use of compute (**millions of GPU-hours**) and reward shaping, while **Factorio** has a larger action and observation space and is sensitive to production balancing.
   - The action and observation space of **Factorio** is bigger and sensitive to needing to calculate and balance production of intermediates, task horizon is longer, etc, and wonder what would happen if we tried **RL** on **Factorio** with heavy reward shaping and human prior, we could get to rocket launch with a similar level of compute as **OAI5**.
- **FLE: Curriculum-based code generation promises LLM probes**: Members find **FLE's** current setup using code generation, production score feedback, a **REPL** loop, and memory compression is a useful scaffolding that reduces the action space and induces structure and planning for **LLMs** in **Factorio**.
   - A member suggested a curriculum-based code generation approach, guided by mini-goals and a theory of mind module within the **FLE** loop, seems like a promising way to probe the limits of **LLM** planning in this environment, like **minedojo's voyager** and **mindforge**.
- **Team explores Hierarchical LLM+RL hybrid system**: The team explored a hybrid setup where compositional planning is handed off to the **RL** loop, calling reusable **LLM** stubs for concrete implementation, handing medium- to long-horizon planning over symbolic base design primitives to the **RL** loop, while the **LLM** handles implementation details.
   - One member suggested that **HLP** and **LLP** makes sense, given **LLMs** have that *human prior knowledge multiplier* skill where we can jump in and out of levels, allowing things to have more **HLPs** now, and that skillset procedures would be hard to compose because they would explode combinatorially.
- **FLE API gets REST API for container Integration**: A member integrated a **REST API** inside the **Factorio** container, choosing **C#** for the server because it compiles down to machine code and the container doesn't need dependencies other than the two binaries.
   - There's an issue to have a discussion about actions: which do we need and what should they be called on [GitHub](https://github.com/MortenTobiasNielsen/FLE-API-specification/issues/5).
- **`connect_entities` eases Factorio agent development**: `connect_entities` prevents the agent from explicitly designing the route of belts/poles/pipes, but removing it makes agents totally incompetent.
   - A member suggested that it would be better to figure out how to make `connect_entities` more configurable for an agent, rather than removing it entirely.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1383429247921684600)** (1 messages): 

> `AMD GPU, Image Analysis` 


- **Image Analysis Incoming**: A user sent a potentially **official image** related to **AMD GPUs**.
   - The image was sent *just in case* it was relevant, but no further context was provided.
- **AMD GPU Speculation**: The image is speculated to contain details about an upcoming **AMD GPU**, potentially related to competitive positioning.
   - Without further context, the exact significance of the image remains unclear, but it hints at internal documentation or marketing material.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1383561639219171438)** (1 messages): 

> `BLISRetreat2023, UTexas presentation` 


- **Seeking Slides Video**: A member inquired about a video accompanying the presentation slides from **BLISRetreat2023** at the [University of Texas](https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf).
- **University Presentation**: The presentation discusses topics related to **BLIS**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1383160456776126647)** (196 messages🔥🔥): 

> `Manus credits, Manus speed, Minimax copy of Manus, Manus AI updates, Manus Agent mode` 


- **Minimax copies Manus Functionality**: Members pointed out that [agent.minimax.io](https://agent.minimax.io/) has copied Manus, and noted that it had *serious potential before credits were announced* but the *stupid pricing ruined it*.
- **Users complain of Manus credit-eating errors**: Users are reporting that **Manus is eating credits due to its own errors**, one stated that it *ate all my credits all 4k over its own errors.*
   - There were complaints of it using **700/1000 credits to deliver a blackscreen website**.
- **Free Lume alternative to Manus being shilled**: Members discussed [lume.im](https://lume.im) as an alternative to Manus.
   - A user promoted it as *free and unlimited*, resulting in accusations of shilling and spam.
- **Gemini outcompetes Manus in specific tasks**: A member stated that *Manus couldn't do it, but Gemini could* and shared a [link to a Gemini output](https://g.co/gemini/share/ef9c34a02d31).
   - The same user also stated, *Gemini is the best static canvas currently. Manus isn't static, so we cant combine those.*
- **Users report slowness and task failures with Manus**: Users are complaining that **Manus is slow, doesn't follow instructions**, and new updates have made it worse, in addition to **simple compiling documents taking 40 minutes and burning 400+ credits**.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1383179252001669311)** (112 messages🔥🔥): 

> `Decentralized Pre-training, Hermes 4 Training, Bandwidth Differentials, AI Evals company, Multilingual reasoning in AI` 


- **Nous launches Psyche for pretraining**: Nous Research is doing pretraining on [psyche.network](https://psyche.network) while **Jensen Huang** roasted Anthropic on pre-training versus post-training.
   - A member noted that *distributed stuff is only gonna get better from here* and will benefit decentralized training.
- **Dawn Internet plugs decentralised broadband**: [Dawn Internet](https://x.com/dawninternet) is a decentralised broadband protocol providing **gigabit internet** using fixed wireless rooftop antennas.
   - Their new **WiFi router** includes a GPU capable of supporting RL.
- **Nous to commence training Hermes 4**: Nous Research will begin **training Hermes 4** on Monday, though it will still take a minute to train and prepare.
   - The new model of the Zeus series will not be based on the old Hermes, but on the newest Mistral.
- **Atropos RL environments works with Axolotl**: Atropos RL environments works with axolotl right now (which uses TRL) and a member is working on VERL integration, according to discussion in [Discord](https://discord.com/channels/972392335214743642/974678437632428072/1283737286656051200).
   - A member states that atropos is *very good* and shared [Atropos's Readme file](https://github.com/NousResearch/atropos?tab=readme-ov-file#axolotl) for more.
- **Kimi-Dev-72B releases open-source coding LLM**: **MoonshotAI** introduces [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B), a new open-source coding LLM for software engineering tasks achieving a new state-of-the-art on **SWE-bench Verified** among open-source models with a score of **60.4%**.
   - Kimi-Dev-72B is optimized via large-scale reinforcement learning, autonomously patching real repositories in Docker and gains rewards only when the entire test suite passes, aligning with real-world development standards.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1383265720632086628)** (8 messages🔥): 

> `Gemini 2.5 Pro, Chain of Thought (CoT) prompting, Reasoning Techniques, API key setup, Hyperbolic integration` 


- **Gemini 2.5 Pro Deployed on AI Studio**: A member confirmed using **Gemini 2.5 Pro** on **AI Studio** and inquired about methods to force the model to use **Chain of Thought (CoT)** in long chats.
   - The user noted that **CoT** is not consistently triggered.
- **Reasoning Techniques to Enhance Model Performance**: A member suggested prompting the model to use reasoning techniques such as **neurosymbolic, counterfactual, inductive, and deductive**.
   - They advised explicitly instructing the model how to think and inputting keywords like *Alternatively*, *consequentially*, and *due to* to guide the reasoning process.
- **API Key Setup Explained**: A member provided guidance on setting up the **API key** and settings, referring to a specific gear icon location.
   - An image was attached to further illustrate the process ([image.png](https://cdn.discordapp.com/attachments/1154120232051408927/1383446172143714334/image.png?ex=68517526&is=685023a6&hm=452d7b1e5ae55dcec88e3f2039c68055dfd71879962a76661b5b42715509ed6b&)).
- **Hyperbolic Connection Issues**: A member reported having trouble connecting **Gemini 2.5 Pro** to **Hyperbolic** despite completing the API key setup.
   - The discussion centered on troubleshooting the integration process.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383409025017843812)** (19 messages🔥): 

> `Bitter Lesson, Generalist vs SME, Grounding in reality, Gene edits to cure cancer` 


- **Bitter Lesson Summary**: Discussion around the [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) essay by Rich Sutton, which discusses how **innovations scale with Moore's Law** rather than human alignment.
   - It was noted that the essay requires the assumption that *reality is ultimately computational* and that *discovery = observable human reality*.
- **Cancer Cure Paradox**: A member mentioned the issue of **grounding in reality** using a hypothetical example of a model discovering gene edits that cure cancer without understanding how.
   - They pointed out the risk of unintended consequences, such as causing ALS, if the mechanism of the cure is not understood, and pointed to [this research](https://arxiv.org/abs/2506.10911v1).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1383164870308003923)** (8 messages🔥): 

> `WebSummit talk on closed internet/AI, Robotic Skin, Deep Residual Learning` 


- **Closed Internet Rant at WebSummit**: A member shared a [talk given at WebSummit in Vancouver](https://youtu.be/vZVcBUnre-c) about the closed internet and closed AI, half history, half rant.
   - It was cross-posted on [FXTwitter](https://fxtwitter.com/jyo_pari/status/1933350025284702697) by another user.
- **Crazy Robotic Skin from Cambridge**: A member posted about [robotic skin from Cambridge](https://www.cam.ac.uk/stories/robotic-skin), also linking to a [YouTube video](https://youtu.be/BV5I4w_wxKI?si=lkADZ69PpfxCUlZt).
   - The skin seems to be made from a **stretchable matrix** with embedded sensors.
- **Deep Residual Learning Paper**: A user shared a link to the [Deep Residual Learning paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) from **CVPR 2016**.
   - The [paper abstract](https://arxiv.org/abs/2506.10943) link leads to a non-existent arXiv ID.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383409025017843812)** (19 messages🔥): 

> `Bitter Lesson, Generalist vs SMEs, Nature Article, Arxiv Paper, Observable Reality` 


- **Bitter Lesson: Scaling Beats Human Alignment**: Discussion around Rich Sutton's ["Bitter Lesson"](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) essay highlights how the *largest innovations* are due to scaling with Moore's Law rather than aligning with humanity.
   - The essay suggests innovations scale, but understanding them requires observation, otherwise we risk systems we will never understand, as in gene edits that cure cancer but also cause unexpected side effects because we don't understand the mechanism.
- **Nature Article deemed 'useless' by some**: A member shared a [Nature article](https://www.nature.com/articles/s42256-025-01049-z?fbclid=IwY2xjawK6iz5leHRuA2FlbQIxMABicmlkETEzb2FnQWNpQzlpQlBzMmhQAR6aSa6Wtu4htiNOE9nvcR4GLRJIaaaBOm1gFYChLS_g5c7G0wk29w2Ohbn_KA_aem_IRJuLT1puoERTjNu2VVTnQ) but commented *'literally don't, all my chinese friends say it's useless'*, without further context.
- **New Arxiv Paper Posted**: A member posted a new [arxiv paper](https://arxiv.org/pdf/2407.01067) and another [arxiv paper](https://arxiv.org/abs/2506.10911v1) with no other context.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1383170950245253130)** (135 messages🔥🔥): 

> `VS Code forks, TUI, RA-Aid, Context Window Management, LLM Personas` 


- ****Electron Apps Valued Highly****: A user joked about forking an electron app instead of building a TUI, referencing the **$9B valuation** of **VS Code** forks.
   - A member agreed it's an over engineered solution but they pointed out that everyone coming out of college, and some going into it are going to have contact with VS Code.
- ****RA-Aid Aider Integration Clarified****: After some poking around with [RA-Aid](https://github.com/ai-christianson/RA.Aid), a member found that **Aider** has a clear benefit with its **repo map**, and letting user add files to context.
   - However they did mention Aider spent 5 cents doing seemingly nothing with 32K tokens which shocked me, then did a brute force grep of the codebase.
- ****Systematic Persona Evaluation Done Right****: A user linked an [Arxiv paper](https://arxiv.org/html/2311.10054v3) arguing that *adding personas in system prompts does not improve model performance across a range of questions compared to the control setting where no persona is added*.
   - They felt like that was the case for a long time but was wondering if there is an actual research that backs this up.
- ****New Brainstorming UX Features Generated****: A user prompted DeepSeek to generate different tiers of features for **Realistic**, **Outside the Box** and **Completely Bonkers**.
   - The **Completely Bonkers** tier included suggestions like *Anti-Gravity Code Reflow* and *Multiverse Branching*.
- ****Aider Context Window Feature Requested****: A user asked if it would be possible to add a feature to **Aider** that allows it to manage the context window on its own, not just adding files to it, but also removing or cleaning the context window as needed.
   - Another user pointed to the [repomap](https://aider.chat/docs/repomap.html) and [scripting](https://aider.chat/docs/scripting.html) documentation to give Aider the repo map and Aider as a tool to control context.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1383262463528800426)** (18 messages🔥): 

> `LLM-OpenAPI-minifier integration with aider, Setting API keys within Aider, Aider's agentic capabilities, Loading active parameters in VRAM for MoE models like Qwen3` 


- **Seeking Aider Integration with LLM-OpenAPI-minifier**: A member inquired about using [LLM-OpenAPI-minifier](https://github.com/ShelbyJenkins/LLM-OpenAPI-minifier) to integrate application interfaces with Aider.
- **Requesting In-Program API Key Setting for Aider**: A member asked how to set an **API key** within Aider itself, noting the absence of such a feature in the documentation, and another member suggested a `llm keys set anthropic xxxx` style command pattern for setting keys.
   - A member asked if this feature was on the roadmap and whether *a rookie* could contribute a PR for it, referencing [Simon Willison's `llm` tool](https://simonwillison.net/2023/May/8/llm-cli/) as inspiration.
- **Confirming Aider's Limited Agentic Functionality**: A member questioned whether Aider is fully agentic, as they were unable to make it work as an agent or modify code or run commands, leading another member to clarify that Aider is *not really agentic* but the `/run` command exists for limited use.
   - They mentioned a personal project called **gitmind** that attempted this but was later abandoned.
- **Proposing Selective VRAM Loading for Qwen3 MoE**: A member asked if it's possible to load only **active parameters** in **VRAM** when running models like **Qwen3 30B MoE**, aiming to use Q8 without significant speed degradation on a 3090 GPU.
   - They clarified that they wanted to avoid loading parameters unnecessary for a specific prompt (e.g., grammar layers when focusing on coding).


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1383185193862565889)** (136 messages🔥🔥): 

> `Claude Swarm for team management, Proactive AI agents definition, Anthropic's multi-agent system, LLM as judge for evaluations, Cursor for writers AI tool alternatives` 


- **Claude Swarm swarms Shopify with Team MCP**: **Claude Swarm**, a tool for setting up hierarchical teams of experts using Claude Code's MCP capabilities, is gaining traction at Shopify and other companies ([code here](https://github.com/parruda/claude-swarm)).
   - A user suggests making one of the experts a *recruiter* to manage the swarm configuration and team expansion.
- **IMPACT framework challenges Proactive AI agent definitions**: A blogpost defines **proactive AI agents** as entities that control their wake schedules, workflows, have persistent memory, and use stateful tools ([substack post](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents)).
   - swyxio ran this definition against his IMPACT framework and noted it lacks **intent**, **planning**, and **authorization**.
- **Anthropic Multi-Agent System Outperforms Opus 4**: **Anthropic** found that a multi-agent system with **Claude Opus 4** as the lead agent and **Claude Sonnet 4** subagents outperformed single-agent Claude Opus 4 by **90.2%** on internal research eval ([Anthropic blogpost](https://www.anthropic.com/engineering/built-multi-agent-research-system)).
   - The system uses about **15x more tokens** than chats due to parallelization and extensive tool use, requiring prompt engineering to prevent excessive agent spawning and web scouring for nonexistent sources; LLMs are also used to evaluate the outputs.
- **Obsidian Copilot offers Obsidian markdown writer's cursor**: Users discussed tools for working with markdown files with AI assistance, proposing **Obsidian Copilot** as an option ([Obsidian Copilot](https://www.obsidiancopilot.com/en)).
   - Users desire functionality beyond simple chat, such as breaking notes by topic, tagging, aggregating notes, and creating flashcards with Anki MCP.
- **Moonshot AI launches Kimi-Dev-72B model**: **Moonshot AI** has open-sourced their **Kimi-Dev-72B** model, achieving a State-of-the-Art (**SotA**) result of **60.4%** on SWE-bench Verified among open-source models ([HF model](https://moonshotai.github.io/Kimi-Dev/)).
   - The announcement was made by Aran Komatsuzaki on Twitter, with links provided to both the Hugging Face model and the GitHub repository.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1383270054195499069)** (75 messages🔥🔥): 

> `Landmark Papers in new field, LLMs as narrative simulators, AI-generated papers with mathematical errors, pytorch dataloader workers, EleutherAI community vs research focus` 


- **Sniffing out Seminal Studies: New Field Navigation**: A member asked about how to find landmark papers when diving into a new field, specifically video generation with existing knowledge of image generation.
   - Suggestions included browsing professor's lecture notes, asking informed individuals directly, and starting with recent papers from major labs to leverage their citations, also see [this discussion on LessWrong](https://www.lesswrong.com/collaborateOnPost?postId=9eehTtLsTBZR9Bd7Q&key=3b19568356a160ba7cf74febafdf33) .
- **LLMs as Narrative Navigators, Simulating Stories**: A member inquired about posting their English 101 paper, examining **LLMs** as narrative simulators due to their architecture and emergent behavior.
   - The request was denied as it was a request for reviews, but a link to [The Void post on tumblr](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void) was shared with related analyses and examples.
- **Math Mishaps Mar AI-Made Manuscripts**: A warning was issued regarding AI-generated papers, citing an alleged (dubious) response to an Apple paper riddled with mathematical errors from [arxiv.org](https://arxiv.org/abs/2506.09250).
   - One member shared a link to [this tweet](https://x.com/BlancheMinerva/status/1933845602917290145) pointing out the errors.
- **Pytorch Dataloader Doom: WSL Workers' Woes**: A user reported issues with **PyTorch dataloader workers** being *killed by signal* in WSL, particularly with high worker counts and long sequence lengths.
   - It was suggested that they check `/var/log/syslog` for potential **OOM** errors, and be more careful about memory usage when processing long video sequences.
- **Eleuther's Ethos: Balancing Novices and Research Nucleus**: Concerns were raised about the discord's perceived mixed messages to newcomers, contrasting the welcoming web copy with the research-focused interactions.
   - Community members discussed the balance between welcoming newcomers and maintaining a research-level discussion, emphasizing the distinction between AI education and teaching research skills, as well as discussing LLM SEO (Language Engine Optimization).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1383173632439881881)** (50 messages🔥): 

> `DMCA and Copyright Law, Emergent behavior in independent tasks, Llama-3.2-1B-Instruct ARC-AGI, Qwen3 tokenizer and image understanding` 


- **Copyright Law is a Joke**: A user stated that *copyright law is a joke unless you're the abuser*, commenting on DMCA and copyfraud penalties, with a link to [fxtwitter.com](https://fxtwitter.com/jyo_pari/status/1933350025284702697) and [arxiv.org](https://arxiv.org/abs/2506.10943).
- **Independent tasks lead to emergent behavior**: Papers suggest that independent tasks X and Y may exhibit **emergent behavior** on the combined task "X and Y", prompting a search for papers meaningfully exploring this phenomenon, with members linking to [arxiv.org](https://arxiv.org/abs/2405.15071).
- **Llama-3.2-1B-Instruct scores 72.5% on ARC-AGI**: **Llama-3.2-1B-Instruct** achieved **72.5%** on **ARC-AGI**, but the test was curated from a subset of **11 training** and **8 evaluation tasks** solvable under optimal **TTT configurations**.
- **Qwen3 gets raw bytes tokenizer and image patches**: A member is using the **FAFO method** of taking **Qwen3** at various sizes (**1.7b**, **4b**, and **8b**) and doing simple **SFT** when switching the tokenizer to raw bytes and adding image understanding with the **Fuyu method**, projecting image patches into the token stream, using the [LLaVAR-Instruct-16K](https://huggingface.co/datasets/HuggingFaceM4/LLaVAR-Instruct-16K) dataset.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1383162829586169856)** (1 messages): 

> `LLM Fairness, Interpretability Interventions, Unfaithful Chain of Thought` 


- **Realistic Details Trigger LLM Bias**: A new paper reveals that adding realistic details to bias evaluations triggers race and gender bias in LLMs, with up to **12% difference in interview rates** even in models like **GPT4o** and **Claude 4 Sonnet**.
   - These details include *company names, culture descriptions from careers pages, or constraints like "only accept top 10%"*, thereby exacerbating bias.
- **Interpretability Fixes Fairness Flaws**: While prompt tuning fails, interpretability-based interventions, such as *affine concept editing/ablation of race/gender directions*, reduce bias, typically to **below 1%**.
   - The [research paper](https://x.com/a_karvonen/status/1933582375419850806) highlights that such targeted interventions effectively mitigate the identified biases.
- **LLMs Exhibit Unfaithful Chain of Thought**: The study found that inspecting **Chain of Thought (CoT)** in LLMs gives no indication of race/gender bias, despite outcomes showing clear bias.
   - This demonstrates an instance of *unfaithful chain of thought in the wild*, where the reasoning process masks underlying biases.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1383231578661982351)** (5 messages): 

> `Benchmark Evaluation Algorithm, Inspect Standard Format, Eval Coalition Effort` 


- **Algorithm Pool Evaluates Models**: A member envisioned an **algorithm + pool** system for benchmarking, allowing new benchmarks to be added and models to be evaluated dynamically.
   - The pool would select examples to test against, addressing benchmark saturation and enabling comparison of models with different capabilities, although the specifics are still evolving.
- **Inspect Standardizes Evaluation Results**: A member mentioned that [Inspect](https://inspect.ai) includes a standard format for storing evaluation results, potentially covering evaluation inputs, outputs, metrics, and metadata.
   - They asked what specific aspects were not covered by Inspect's standardization, prompting further discussion on the tool's capabilities.
- **Evals Coalition Seeks Scaled Implementation**: A member expressed hope to join the **Eval coalition effort**, starting with scaled implementation of current benchmarks and evaluations in an automated setting.
   - Another member confirmed that they should be added to the **evaleval Slack** soon and welcomed their input as the effort is still in the early exploratory stage.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1384062892394942484)** (1 messages): 

> `Vitabyte Founder, GroK-scale Training, Multi-node LLM fine-tuning, ROCm + CUDA, Full stack Ops` 


- **Vitabyte Founder Seeks Grok-Scale Projects**: George, founder of **Vitabyte/Vitapay**, is seeking to join any **Grok-scale (314B)** or **multi-node LLM training/fine-tuning projects**.
   - He brings experience with **ROCm + CUDA** setups, **quantization**, and **full stack ops**, offering contributions in infra, logs, tuning flows, and documentation.
- **Vitabyte Founder Skills**: George brings experience with **ROCm + CUDA** setups, **quantization**, and **full stack ops**.
   - George offers contributions in infra, logs, tuning flows, and documentation.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1383195275702829076)** (19 messages🔥): 

> `Notebook LM Plus Access, PM Interview Conversational AI Platform, Exam Prep with NotebookLM, Chrome Extension for NotebookLM, Podcast Personality Shaping` 


- ****Notebook LM Plus Access Still a Question****: A user of paid **AI Pro** inquired about not having access to **NotebookLM Plus**, despite their subscription, and mentioned using **1900** sources.
   - The user also shared a [NotebookLM Mind Map](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=6851e6a5&is=68509525&hm=f1cd4c62bca69adfe75ec5f0adc1b806be3b22beaa24c3bac7597185b382a6d7), noting it has **4 sublevels** and is vertically dense, but not yet horizontally, with a size around **9MB**.
- ****AI Platform Aims to Ace PM Interviews****: A member is developing a **Conversational AI platform** designed for **PM interviews** and is seeking beta users to validate their idea and provide feedback.
   - Interested users can sign up through [this form](https://forms.gle/P3f2JP3vVB62vedb7) to be added to the waitlist.
- ****NotebookLM Tackles Exam Prep****: A user asked for advice on using **NotebookLM** to prepare for an exam with material that isn't in PDF format, consisting of web pages and virtual labs.
   - Another user suggested using **Chrome extensions** to follow and import links from web pages into **NotebookLM**.
- ****Podcast Hosts Seek Personality Shaping Strategies****: A member is delving into shaping the personality of their **NotebookLM podcast hosts** and is seeking to exchange experiences with others.
   - Another user inquired about strategies and apps for publishing episodes to **Spotify**.
- ****Flattening Websites into Single Source for Notebooks gains traction****: A member proposed creating a flattened version of a website—a single page containing all content without links—to easily feed it as a single source into **NotebookLM**.
   - Another user suggested using the **Web Sync** tool, accessible via [this article](https://www.xda-developers.com/notebooklm-chrome-extensions/).


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1383177666110427278)** (86 messages🔥🔥): 

> `LaTeX in NLM, Image Uploading, Android App of NotebookLM, Podcast issues, Mindmaps on iPad` 


- **NLM Embraces LaTeX Markups!**: NotebookLM uses **LaTeX markups**, as do other LLMs, for math and scientific equations.
   - To view these equations, users can use online or offline **LaTeX renderers** or try the [LatexInNotebooklm](https://github.com/ergs0204/LatexInNotebooklms) extension.
- **Image Uploading Issues Resolved!**: Users found that NotebookLM now supports image uploads directly from the device, not from **Google Drive**.
   - To upload images, users can click the *choose file* option or by dragging.
- **Android App Appreciation Surges!**: Users are praising the convenience of the **NotebookLM Android app**, particularly for listening to deep dives.
   - However, it was mentioned that for full functionality like choosing the length of podcasts it's better to use the website.
- **Podcast Audio Quality Takes a Dive!**: Users noticed a decline in the audio quality and content of **NotebookLM podcasts**, with robotic and repetitive framing of the "source material."
   - The issue affected generated podcasts, and was described as *sounding broken and fake*.
- **Mind Maps Vanish on iPad!**: Users reported that **mind maps** are not visible in the **iPad app**.
   - Users are waiting for the ability to save mindmaps in a format that works as **interactive objects** rather than an image.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1383503120633696498)** (69 messages🔥🔥): 

> `DTensor cross-mesh operation, Llama4 maverick finetuning, Iterable packing, Fused optimizer, Flex attention` 


- ****DTensor distress during distributed Llama4****: Members encountered a `RuntimeError` related to **DTensor cross-mesh operations** during multi-node distributed finetuning of **Llama4 Maverick** on the latest nightly builds, specifically with `aten.mm.default` and differing device meshes.
   - The error manifested differently with varying numbers of nodes (8 vs 12), pointing to potential issues with the **fused optimizer** and mesh configurations, stack trace available in [output.log](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&).
- ****Iterable packing innovation inbound****: A member is developing a private finetuning library with **iterable packing**, built on top of [pytorch/data](https://github.com/pytorch/data), showing great results and prefetching capabilities.
   - They suggested that a separate dataset wrapper might not be needed and that the main overhead is from tokenization, expecting to opensource the library next week and also highlighting that packed DPO is missing in many libraries.
- ****Fused Optimizer Flounders on Full Finetune?****: During attempts to train, the **fused optimizer** was found to cause issues, particularly with checkpoint creation resulting in `nccl` timeouts, whereas the non-fused optimizer allowed training on 8 nodes.
   - It was suggested that increasing the `NCCL_TIMEOUT` environment variable, or setting `total_epochs=self.total_epochs+1` to enable asynchronous checkpoints, might mitigate these issues, while creating a minimal reproducible example for the optimizer issue was also recommended.
- ****Mini-Batch Musings Meet MoE Memory Mastery?****: A member speculated whether using a micro batch size of 1 could reduce the memory requirements for training a **Mixture of Experts (MoE) model**, by only needing the memory for the active parameters.
   - The idea was proposed as a way to train very large models by offloading gradient accumulation to CPU RAM, however another member pointed out that the micro batch size is really `seq_len` as you still need all experts for training.
- ****Flexing Attention with Flashy Nesting?****: Members discussed forcing packed batches to be of size 1 for simplicity, and its ties to **flex attention**, where one member saw a performance increase to 10k TPS vs 2k TPS for non-flex attention.
   - The members suggest using SDPA + flashattention 3, but then your tensors have to be nested tensors (using `torch.nested` with jagged layout), while pointing out that many ops are missing when using nested tensors.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1383519442020991189)** (15 messages🔥): 

> `Mistral Small, Magistral model, ZO optimizer, Flex integration` 


- **Mistral Small Debuts, Disappoints?**: Despite its recent release, the [Mistral Small model](https://mistral.ai/news/mistral-small-3-1) isn't impressing everyone, with one member saying *the mistral small results, even on their own blogposts look barely better than Gemma 3~~qwen3~~*.
   - The member also clarified that they had initially misclicked on **Magistral** instead of **Mistral** while researching.
- **ZO Optimizer Promises VRAM Savings**: Members discussed the **ZO optimizer** and its potential for **3x VRAM economy**, referencing a paper on the topic ([arxiv.org/abs/2506.044303](https://arxiv.org/abs/2506.044303)).
   - One member found it *amazing that ZO even works at all*, while another suggested adding it to **Flex**.
- **Flex Integration a low priority**: A member suggested to include **ZO** to **Flex** for its **3x VRAM economy** but another user responded with *I wouldn't prioritize it, but eventually sure*.
   - Members agreed that the most important takeaway from the **ZO** paper is its scalability on different sizes and its use of mostly non-synthetic experiments.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1383944665656459395)** (46 messages🔥): 

> `RDNA4 support, AVX512_BF16, Zen 4, Mojo testing structure, 1-bit model support` 


- ****RDNA4** Support is here!**: As of the last nightly, **RDNA4** is supported for direct GPU programming in Mojo, but full models aren't quite there yet as the matrix multiplication operations need **RDNA-specific paths** put in place.
   - An introductory patch to add some of the **WMMA** operations necessary for this has been added, bringing models closer to being fully functional on **RDNA3**+.
- ****Zen 4** CPUs for **bfloat16** Support**: While the **5950x** does not support **AVX512_BF16**, Zen 4 and above CPUs, such as the **Ryzen 7000 series**, do offer some **bfloat16** support.
   - However, it is not confirmed whether these include the exact **FMA** instructions needed for CPU inference.
- **Navigating Mojo's Testing Codebase**: Users expressed frustration with Mojo's testing codebase structure, particularly regarding imports within test files and understanding the package **__init__.mojo** hierarchy.
   - A major revelation was realizing that running with `mojo test -I .` allows tests to import the package being tested as if it was a library; one user suggested looking at [ExtraMojo](https://github.com/ExtraMojo/ExtraMojo) as a good project structure example.
- ****LLVM** takes up most binary size**: Most of the binary size is taken up by statically linking **LLVM**, with MAX on its own being around **750 MB**, and the .mojopkgs shipped with MAX being about **100 MB**.
   - There is active work to reduce the number of copies of **LLVM**.
- ****Intel Nova Lake** to have 52 Cores?**: The next 'compile sku' is likely to be **Intel Nova Lake**, since it's likely to have **52 cores** on the top sku.
   - The i9 is the one which will likely have that many cores, while HEDT for Intel is *buy a xeon*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1383167634421579776)** (32 messages🔥): 

> `CUDA Stream Synchronization, Mojo C ABI, Mojo Zed Extension, Mojo 'let' deprecation, Mojo AOT compilation` 


- ****Host Synchronization Unnecessary** for CUDA Streams?**: A member questioned whether `ctx.synchronize()` is necessary in [Puzzle 12](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step), suggesting **CUDA streams** handle synchronization automatically for dependent kernel launches.
   - A Modular team member confirmed that *DeviceContext* uses a **CUDA stream**, so the execution order matches the call order and no explicit sync is required, promising to adjust the documentation accordingly.
- **Mojo Calls C with **`external_call`****: A member asked about calling into **C** from **Mojo**, seeking examples in the documentation.
   - Another member pointed out the use of the **`external_call` function** for **Mojo** to **C** interoperation.
- **Mojo Zed Extension Still Functional**: A user reported that the **Mojo extension for Zed** is working well, inquiring about future updates.
   - The extension developer confirmed they're working more with **Mojo** and asked for specific feature requests, but there is an [issue](https://discord.com/channels/1087530497313357884/1329089055211655231/1331626952561397831) regarding unnecessary highlighting of unused variables.
- ****`let` Declaration Laid to Rest** in Mojo**: A new **Mojo** learner inquired about the deprecated **`let` variable declaration**, encountering errors in tutorials.
   - A team member confirmed **`let`** was removed in [the 24.4 changelog](https://docs.modular.com/mojo/changelog#v244-2024-06-07), noting that most **Mojo** tutorials are becoming outdated quickly, but the official proposal is [here](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md).
- **Mojo's **AOT Compilation** Discussed**: A member asked which aspects of **Mojo** are **JIT** versus **AOT** compiled, particularly regarding **SIMD** and runtime statistics.
   - A member clarified that **CPU code** is **AOT** compiled unless inside a kernel for **MAX**, while **GPU code** uses a **JIT** compiler due to the need for driver-specific optimizations, the autotune library that existed was removed because it massively bloated compile times.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1383159529822228490)** (40 messages🔥): 

> `MCP in Agentic Frameworks, A2A Agent Discovery, FastMCP and Server Composition, GitHub APIs in MCP Server, Orchestrator Agent Recommendations` 


- ****Agentic Frameworks Embrace MCPs****: An agent questioned where MCPs fit into an agentic framework, considering the top layer as the orchestrator agent, followed by specific agents accessing multiple MCP servers for tool selection and memory storage.
   - One member suggested using smarter hosts with tool reranking.
- ****FastMCP Mounts Domain Segregation Subservers****: A member mentioned that [fastmcp](https://gofastmcp.com/) can mount MCP servers, allowing a router server to host subservers for domain segregation.
   - The team developing a single MCP server exposing all GitHub APIs in one place is exploring the idea of an orchestration server that can invoke or proxy to other MCP servers, as well as weighing performance as the number of tools grows.
- ****LLM Selection Done by Client****: LLM selection is done by the client and the model used depends entirely on the client or app consuming the server.
   - The MCP team is figuring out how to optimize and they encourage checking out the code here: [GitHub MCP Server](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go).
- ****Opus Orchestrates Cursor****: For those using Cursor, **Opus** was recommended as an orchestrator agent, although its cost was noted.
   - One person preferred a local one.
- ****Streamable HTTP needs full URL****: A member helped someone resolve a connection error with **fastmcp** by pointing out that the full URL with `/mcp/` is required for streamable-http.
   - The default streamable-http port is **8000**, not 6277.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1383211846315540591)** (7 messages): 

> `SchemaPin for Rug Pulls, Glama MCP servers support streamable HTTP, excel-mcp-server, User analytics and live debugging for MCPs` 


- ****SchemaPin** Prevents MCP Rug Pulls**: A member built **SchemaPin** to prevent MCP Rug Pulls and similar attacks, the [repo](https://github.com/ThirdKeyAI/SchemaPin) is available on GitHub.
   - The [homepage](https://schemapin.org) has easy ways to implement **SchemaPin**.
- ****Streamable HTTP** Launched on All Glama MCP Servers**: All Glama MCP servers now support **streamable HTTP** e.g., [glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257](https://glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257).
- ****Excel MCP Server** Trending on GitHub**: A member shared their repo, [excel-mcp-server](https://github.com/haris-musa/excel-mcp-server), after it trended twice on GitHub.
   - They welcome any and all feedback on the project.
- **Debug your MCP with **MCPCat****: A member is working on user analytics and live debugging for MCPs, with the repo available [here](https://github.com/mcpcat).


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1383205705741238413)** (20 messages🔥): 

> `Cohere documentation typo, Team collaboration with LLMs, AI/backend developer introduction, Cohere's work with the government, Secure ML and privacy preservation` 


- ****Typo Troubles**: Cohere Docs Fixed!**: A user reported a typo in the [Cohere documentation](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide) where `co = cohere.SagemakerClient()` should have a lowercase `m`.
- ****LLM Teamwork**: How Teams Collaborate!**: A user is researching how teams are integrating large language models like ChatGPT and Claude into their daily workflows, inquiring about changes and missing elements since their introduction.
- ****AI Developer Kira**: Joins the Chat!**: Kira, an AI/backend developer, introduced themselves, expressing excitement to connect and build cool stuff, focusing on custom bots, automations, and scalable systems.
- ****Government Giggle**: Cohere's Public Sector Work!**: A user shared a [Carney news video](https://m.youtube.com/watch?v=qWBO4LsKdD4&pp=ygULQ2FybmV5IG5ld3M%3D) highlighting Cohere's work with the government, expressing it must have been a huge honor.
- ****Privacy Pal Yasir**: Secure ML Enthusiast!**: Yasir Khan, a Computer Science graduate, introduced themself, mentioning work on secure machine learning and privacy-preservation, seeking connections for collaboration on AI/ML projects.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1383312324340879400)** (5 messages): 

> `direct-injected-document tool, Cohere a032025 memory usage` 


- **"Direct-injected-document" Tool Surfaces Sporadically**: Some users reported that a tool named **direct-injected-document** pops up as an answer sporadically.
   - A member asked for a prompt example and which model was being used.
- **Cohere a032025 hosting needs**: A user inquired about the memory requirements for hosting **Cohere a032025**.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1383758872996478986)** (6 messages): 

> `AI developers introductions, Custom bots, Automations, Scalable systems, Secure machine learning` 


- **AI Developers assemble!**: An AI/backend developer named Kira introduced herself, offering help to startups in building **custom bots, automations & scalable systems**.
   - She expressed excitement to connect & build cool stuff with others.
- **Secure ML & Privacy Guru Seeks Collabs**: Yasir Khan, a Computer Science graduate, introduced himself, highlighting his work on **Secure machine learning and privacy-preservation**.
   - He expressed interest in connecting with friends having similar interests and collaborating on AI/ML projects to enhance his expertise.
- **Machine Translation Maestro Materializes**: Joel, a Computer Science student from the Philippines, introduced himself as doing research on improving **Machine Translation and LLMs for the Filipino language**.
   - He said he's here to *look around, see cool stuff and possibly meet cool people too*.
- **Ollama models gain a new fan**: A new person in the AI world expressed their enjoyment in playing with **models from ollama**.
   - They expressed that *its fun*.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1383159468086398989)** (3 messages): 

> `Data + AI Summit 2025, Agentic Document Workflows, Multi-Agent System, AI Travel Agents, AI Agents in Production` 


- **Data + AI Summit 2025 Concludes**: The @databricks **Data + AI Summit 2025** has concluded, with more content on the emerging landscape of **agentic document workflows** to come, learn more [here](https://t.co/jS2Nfwxxb3).
   - The CEO @jerryjliu0 gave a standing-room-only talk.
- **Microsoft's AI Travel Agents Demo**: @microsoft's new **AI Travel Agents demo** shows how to coordinate multiple AI agents using the **Model Context Protocol**, **LlamaIndex.TS**, and @Azure AI Foundry for complex travel planning scenarios.
   - Six specialized AI agents work together, learn more [here](https://t.co/cNyVAcnf6K).
- **Build and Secure AI Agents in Production**: Join an evening in San Francisco for expert insights on building and securing **AI Agents in production**, covering best practices [here](https://t.co/MVd2rwSVIE).
   - Our VP of Developer Relations @seldo will be presenting **Building Real-World Agents** alongside industry experts from Ravenna and @auth0.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1383262429617590352)** (26 messages🔥): 

> `LandingAI vision agent vs LlamaIndex, Synk hiring, Faiss Index, LlamaCloud contact sales page, LlamaExtract parsing errors` 


- **LandingAI vs LlamaIndex Document Understanding**: Members discussed a new vision agent document understanding tool developed by **LandingAI**, a company started by **Dr. Andrew Ng**, with one member asking for a comparison against **Llama Parse** in light of a previous [post](https://www.linkedin.com/posts/jerry-liu-64390071_mistral-ocr-is-nice-and-fast-but-other-models-activity-7303803148907790336-OP9y) comparing it to **Mistral**.
   - The company's tool can be found at [LandingAI's website](https://va.landing.ai/home).
- **Synk Recruiters Seek Developers**: A member announced that **Synk** is hiring developers (**back-end, Front-end, blockchain**), a **QA Engineer**, a **DevOps Engineer**, **Moderators**, and a **Marketing Analyst** for their decentralized browser system project, directing users to [Synk's X page](https://x.com/Synk_ws).
   - They offer *official employment with signed documentation, guaranteed salary, and a flexible schedule*.
- **Faiss Index Filtering Still Unsupported**: A member inquired about the possibility of doing **metadata filtering on Faiss index queries**.
   - Another member responded that *Faiss doesn't support it*.
- **LlamaCloud Contact Page Fails**: A member reported that the **contact sales page on llamacloud** ([https://cloud.llamaindex.ai/contact-sales](https://cloud.llamaindex.ai/contact-sales)) was not working due to a **500 internal server error**.
   - Another member asked if they were instead referring to the [LlamaIndex contact page](https://www.llamaindex.ai/contact).
- **LlamaExtract Glitches Cause Parsing Errors**: Several members reported experiencing **parsing errors on every document** they tried to run in **LlamaExtract**, with no data being extracted.
   - A member suggested trying again, noting that they were receiving data, and included a screenshot of a successful extraction using LlamaExtract ([image.png](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6)).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1384311159967977605)** (1 messages): 

> `DSPy Optimization Patterns` 


- **Thoughts On Incorporating DSPy Optimization Patterns Requested**: A member asked about thoughts on how to incorporate any of the **optimization patterns** that exist in **DSPy**.
- **Filler Topic for JSON Validation**: This is a filler topic to ensure the JSON has at least two elements in topicSummaries.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1383489483055435877)** (19 messages🔥): 

> `DSPy runners, TextGrad Optimizer, Custom LM Concurrency, DAIS Session Write-Up, BootstrapFewShot Optimizer` 


- ****Run DSPy anywhere** with JSON definitions**: A member is thinking about building **DSPy "runners"** that take the saved **JSON definition** and runs the compiled program, enabling cross-language functionality, like Swift leveraging a compiled program via a managed API.
   - Another member expressed interest but questioned how program logic not captured in the **JSON output** (like signature and module) would be handled, pondering how a program could be serialized.
- ****TextGrad optimizer** awaiting updates**: A member inquired about updates on adding **TextGrad** as an optimizer for DSPy, referencing [issue #1197 on GitHub](https://github.com/stanfordnlp/dspy/issues/1197) which has been open for nearly a year.
   - The member expressed enthusiasm for **TextGrad** due to its effectiveness in optimizing complex prompts and asked if anyone had "hacks" for incorporating it into DSPy.
- **Model writes prompts at **DAIS session****: A member shared a write-up of their session at DAIS this week, titled "Let the Model Write the Prompt", available on [their website](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html).
   - In a follow-up, a member inquired about a recording of the session, to which the first member replied with a [YouTube link](https://youtu.be/I9ZtkgYZnOw?si=XGArjkQSVUlzrEAr).
- ****DeepSeek R1 7B** Struggles with DSPy Optimization**: A member reported suboptimal optimization results using **DeepSeek R1 7B** in a **DSPy-Text2SQL** demo, compared to **GPT-4o-mini** and sought suggestions for improvement following attempts with **LabeledFewShot** and **BootstrapFewShotWithRandomSearch**.
   - Another member suggested that providing more information about the schema could potentially enhance the performance of **DeepSeek R1 7B**.
- ****BootstrapFewShot Optimizer**'s Use Cases**: A member sought to understand how **BootstrapFewShot** optimizer works, particularly for classification use cases, questioning the handling of ground truth for bootstrapped inputs.
   - Another member explained that one can use anything as a metric as long as *it returns a bool, an int or a float (and higher is better).*


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1383424003217035285)** (8 messages🔥): 

> `Certificates, Assignment Selection, MOOC Quiz Archive` 


- **Certificates Coming Mid-July!**: A user inquired about the distribution of certificates, to which a member responded that *certificates will be released in mid July*.
- **Assignments Passed with Reasonable Effort**: A user questioned how to know if they were selected for a certificate or if they passed their assignments.
   - A member clarified that email confirmations are sent for each assignment submitted via **Google Forms**, and as long as everything is completed with **reasonable effort**, a certificate will be granted.
- **MOOC Quiz Archive Linked**: A member shared the [Spring 2025 MOOC quiz archive](https://docs.google.com/document/d/1A00cUWux-J0p9AOnwpyNN3Rb5QFRsbBgAmvgPMezJ10/edit?usp=sharing), also available on the course website in the Quizzes section.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1384168633235800166)** (1 messages): 

> `ControlThrive, Outerbounds, ML Consulting` 


- **ControlThrive founder greets community**: Servando, the founder of the AI/ML consulting practice **ControlThrive** [controlthrive.com](https://www.controlthrive.com/), introduced himself to the community.
   - He invited members to connect with him on [LinkedIn](https://www.linkedin.com/in/servando-torres-239a26b0/) or X.
- **Outerbounds event coming up**: Servando announced an upcoming event he is hosting with Eddie from **Outerbounds** (the team behind the ML infra at Netflix).
   - He shared a [link to the event](https://lu.ma/nw4xccle) and encouraged community members to join.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1383165546375413890)** (1 messages): 

> `Claude Sonnet 4 API Access, Anthropic models, API Pricing` 


- **Claude Sonnet 4 models launch**: **Claude Sonnet 4** and **Claude Sonnet 4 (Thinking)** are now available to all paid plans via [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing).
- **Mohan's hot take on Claude**: Mohan retweeted some **impressions of Claude** on [X](https://x.com/_mohansolo/status/1933605162775687482).


  

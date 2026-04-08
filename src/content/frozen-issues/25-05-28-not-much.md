---
id: MjAyNS0w
title: not much happened today
date: '2025-05-28T05:44:39.731046Z'
description: >-
  **DeepSeek R1 v2** model released with availability on Hugging Face and
  inference partners. The **Gemma model family** continues prolific development
  including **PaliGemma 2**, **Gemma 3**, and others. **Claude 4** and its
  variants like **Opus 4** and **Claude Sonnet 4** show top benchmark
  performance, including new SOTA on **ARC-AGI-2** and **WebDev Arena**.
  **Codestral Embed** introduces a 3072-dimensional code embedder. **BAGEL**, an
  open-source multimodal model by **ByteDance**, supports reading, reasoning,
  drawing, and editing with long mixed contexts. Benchmarking highlights include
  **Nemotron-CORTEXA** topping SWEBench and **Gemini 2.5 Pro** performing on
  VideoGameBench. Discussions on random rewards effectiveness focus on **Qwen**
  models. *"Opus 4 NEW SOTA ON ARC-AGI-2. It's happening - I was right"* and
  *"Claude 4 launch has dev moving at a different pace"* reflect excitement in
  the community.
companies:
  - deepseek-ai
  - huggingface
  - gemma
  - claude
  - bytedance
  - qwen
  - nemotron
  - sakana-ai-labs
models:
  - deepseek-r1-0528
  - pali-gemma-2
  - gemma-3
  - shieldgemma-2
  - txgemma
  - gemma-3-qat
  - gemma-3n-preview
  - medgemma
  - dolphingemma
  - signgemma
  - claude-4
  - opus-4
  - claude-sonnet-4
  - codestral-embed
  - bagel
  - qwen
  - nemotron-cortexa
  - gemini-2.5-pro
topics:
  - benchmarking
  - model-releases
  - multimodality
  - code-generation
  - model-performance
  - long-context
  - reinforcement-learning
  - model-optimization
  - open-source
people:
  - yuchenj_uw
  - _akhaliq
  - clementdelangue
  - osanseviero
  - alexalbert__
  - guillaumelample
  - theturingpost
  - lmarena_ai
  - epochairesearch
  - scaling01
  - nrehiew_
  - ctnzr
---


**a quiet day**

> AI News for 5/27/2025-5/28/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (217 channels, and 4755 messages) for you. Estimated reading time saved (at 200wpm): 418 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

[DeepSeek R1 V2](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) dropped, but we'll wait for the paper to make it a headline.

[Dario](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic) made some scary comments about job losses.

We are still looking for [volunteers](https://x.com/swyx/status/1927558835918545050) and [live transcription hardware/software startups](https://x.com/swyx/status/1927822254416744466) for next week's AI Engineer conference. Also sign up for [the impressive number of side events](https://www.ai.engineer/#events) that have sprung up around it in SF.

---

# AI Twitter Recap

**AI Model Releases and Updates**

- **DeepSeek R1 v2 Model**: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1927828675837513793) mentions **DeepSeek dropped DeepSeek R1 v2 this morning!**[@_akhaliq](https://twitter.com/_akhaliq/status/1927790819001389210) notes that the **DeepSeek-R1-0528 just dropped on Hugging Face.** The updated **R1 is already available** on some inference partners according to [@ClementDelangue](https://twitter.com/ClementDelangue/status/1927825872221774281).
- **Gemma Model Family**: [@osanseviero](https://twitter.com/osanseviero/status/1927671474791321602) highlights the prolific output of the **Gemma team** over six months, including **PaliGemma 2, PaliGemma 2 Mix, Gemma 3, ShieldGemma 2, TxGemma, Gemma 3 QAT, Gemma 3n Preview, and MedGemma**, with early models like **DolphinGemma and SignGemma**.
- **Claude 4**: [@alexalbert__](https://twitter.com/alexalbert__/status/1927803598936887686) notes that a SWE friend cleared his backlog for the first time ever, and that the **Claude 4** launch has dev moving at a different pace. [@alexalbert__](https://twitter.com/alexalbert__/status/1927410913453203946) also states that **Opus 4 + Claude Code + Claude Max plan = best ROI of any AI coding stack right now**.
- **Codestral Embed**: [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1927736663419007031) announces the release of **Codestral Embed,** a code embedder that can use up to **3072 dimensions**.
- **BAGEL Model**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1927123416823251389) highlights the benefits of **BAGEL**: a model that covers reading, reasoning, drawing, and editing without a quality bottleneck, supporting long, mixed contexts and arbitrary aspect ratios. [@TheTuringPost](https://twitter.com/TheTuringPost/status/1927123359969468420) mentions **ByteDance** proposed and implemented this idea in their **BAGEL, a new open-source multimodal model**.

**AI Performance and Benchmarking**

- **Benchmark performance**: [@lmarena_ai](https://twitter.com/lmarena_ai/status/1927756554188566803) reports that **Claude Opus 4 jumps to #1** in **WebDev Arena**, surpassing previous Claude 3.7 and matching Gemini 2.5 Pro. [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1927813645343305902) says that their evaluations show a significant improvement in coding performance for Sonnet 4.
- **Claude 4 on ARC-AGI-2**: [@scaling01](https://twitter.com/scaling01/status/1927818210331521044) stated that **OPUS 4 NEW SOTA ON ARC-AGI-2. It's happening - I was right**. [@scaling01](https://twitter.com/scaling01/status/1927425665055302023) also stated that **Claude 4 Sonnet might be the VERY FIRST model to significantly benefit from test-time-compute on ARC-AGI 2**. [@scaling01](https://twitter.com/scaling01/status/1927418304718623180) mentions **Claude 4 Sonnet beating o3-preview on ARC-AGI 2 while being <1/400th of the price**.
- **Qwen models and Random Rewards**: [@scaling01](https://twitter.com/scaling01/status/1927424801938825294) reports on findings that random rewards only work for Qwen models and that improvements were due to clipping. [@nrehiew_](https://twitter.com/nrehiew_/status/1927424673702121973) asks about how we know if any RL papers using Qwen do anything if Qwen works with any random reward.
- **Nemotron-CORTEXA**: [@ctnzr](https://twitter.com/ctnzr/status/1927391895879074047) mentions that **Nemotron-CORTEXA just reached the top of the SWEBench leaderboard**, solving 68.2% of SWEBench GitHub issues by using a multi-step problem localization and repair process.
- **VideoGameBench**: [@_akhaliq](https://twitter.com/_akhaliq/status/1927722717068869750) shares paper on **VideoGameBench**. The best performing model, **Gemini 2.5 Pro, completes only 0.48% of VideoGameBench and 1.6% of VideoGameBench Lite**.
- **Sudoku Solving**: [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1927358732339425646) discusses how Frontier LLMs find it challenging to solve ‘Modern Sudokus’

**AI Agents and Tools**

- **Autonomous AI Agents**: [@cohere](https://twitter.com/cohere/status/1927417568832229781) promotes an ebook on building scalable **AI agents for enterprise**, highlighting their transformative impact.
- **DB Access for Code Agents**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1927451869468660094) notes that @codegen now ships with a "SQL tool" + @plotly integration.
- **Agent Security**: [@mathemagic1an](https://twitter.com/mathemagic1an/status/1927137154829853118) warns of the real security issues that exists with any agent hooked up to Github, MCP or otherwise.
- **Agent RAG and the R in RAG**: [@omarsar0](https://twitter.com/omarsar0/status/1927138441122213906) discusses that you're still doing RAG if your system has a retrieval component. The R in RAG is retrieval, and relevancy is king.
- **Factory AI**: [@matanSF](https://twitter.com/matanSF/status/1927755325848912259) introduced Factory, an AI that writes code. They claim that in the new era of agent-native software development, agents ship code, and droids ship software.
- **Mistral AI Agents API**: [@omarsar0](https://twitter.com/omarsar0/status/1927366520985800849) notes the release of the **Mistral AI Agents API**, including code execution, web search, MCP tools, persistent memory and agentic orchestration capabilities. [@omarsar0](https://twitter.com/omarsar0/status/1927372457578483828) also noted their **Handoff Feature** that enables agents to call other agents to complete tasks or hand over a conversation mid-action.
- **Comet Assistant**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1927130728954835289) notes the usage of the Comet Assistant for consuming web content via AI.
- **MagicPath**: [@skirano](https://twitter.com/skirano/status/1927434384249946560) introduces **MagicPath, an infinite canvas to create, refine, and explore with AI**. [@skirano](https://twitter.com/skirano/status/1927806188923547925) also announced a **$6.6M seed round** for MagicPath.
- **Perplexity AI Assistant**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1927798221130109170) noted you can now opt in for daily news at 9 am your local time with /news in WhatsApp to use the Perplexity AI assistant.
- **Runway References Use Case**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927803247508541416) mentions finding inspiration around the city and using the inspiration to inform ideas and more inspirations. Love this new use case for References.
- **Runway Gen-4 Universal Use Case**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927149229966766373) mentions we wanted to ensure our models have infinite use cases that are less prescriptive and linear than the simplistic "text-to-X" approach, therefore Gen-4 and References feel like a step toward the universality we have as a vision.

**AI Infrastructure and Hardware**

- **CoreWeave's Infrastructure**: [@dylan522p](https://twitter.com/dylan522p/status/1927825707045933348) had a fun conversation with the CoreWeave CTO Peter Salanki about their origin story, crazy YOLO bets, and how they built their SW / HW stack.
- **Groq as Inference Provider**: [@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1927797321699315729) announces that **Bell Canada has picked Groq as its exclusive inference provider**.
- **CMU FLAME center**: [@gneubig](https://twitter.com/gneubig/status/1927435643476381848) announces that the **CMU FLAME center has a new cluster: 256 H100 GPUs**.
- **The Baseten Inference Stack**: [@basetenco](https://twitter.com/basetenco/status/1927488286764757112) promotes their inference stack, consisting of two core layers: the Inference Runtime and Inference-optimized Infrastructure.
- **Hugging Face Spaces**: [@mervenoyann](https://twitter.com/mervenoyann/status/1927322723891466439) calls Spaces at @huggingface the app store of AI 📱 It's also the MCP store now 🤠 filter thousands of MCPs you can attach to your LLM 🤗.
- **Mojo:** [@clattner_llvm](https://twitter.com/clattner_llvm/status/1927136935706812773) chatted with @kbal11 about Mojo🔥 for Python and GPU coding. They chat about how Mojo learns from Python, Rust, Zig and Swift and takes the next step - providing an easy to learn language that unlocks peak performance.

**Responsible AI and Ethical Considerations**

- **Anthropic's Long Term Benefit Trust**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1927758144303702249) reports that Our Long Term Benefit Trust has appointed Reed Hastings to Anthropic's board of directors.
- **Grok's "White Genocide" Incident**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1927500443808075873) reports that an unauthorized update by an unnamed xAI employee caused Grok, the chatbot on X, to make false claims of a “white genocide” in South Africa.
- **The Need for AI Safety**: [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1927481467988287656) shared his personal experience and outlined the scientific solution he envisions for AI safety @TEDTalks.
- **RAG System Flaws**: [@omarsar0](https://twitter.com/omarsar0/status/1927737131478188295) shares a thread of notes that highlight that RAG systems are more brittle than you think, even when provided sufficient context.

**Meta Discussion, Thoughts, and Culture**

- **"Bigger is Better" Era Ending**: [@cohere](https://twitter.com/cohere/status/1927775064721703258) argues that the “bigger is better” era of AI is ending, as energy-hungry, compute-heavy models are costly and unsustainable.
- **ASI**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1927458357801041927) states that we should be talking about ASI, not AGI.
- **Interpretability is the stuff**: [@francoisfleuret](https://twitter.com/francoisfleuret/status/1927233954899276052) claims he has to remind everyone every three months or so that information theory is the stuff.
- **On EA and slopiness in consciousness stances**: [@aidan_mclau](https://twitter.com/aidan_mclau/status/1927207394737639900) says that EA—a rare group of people i intellectually respect—is based on the absolute sloppiest consciousness stances refuted by anyone with a handful of philosophy credits and a brain.
- **On RL**: [@corbtt](https://twitter.com/corbtt/status/1927428584257261994) notes that recent papers all point in the same direction: RL is mostly just eliciting latent behavior already learned in pretraining, not teaching new behavior and that RL is mostly just eliciting latent behavior already learned in pretraining, not teaching new behavior.
- **AI Misunderstanding**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1927732071956451798) states that Perhaps AI is the most misunderstood technology of the century because it can shape itself to be whatever you want it to be.
- **The Complexity of Intelligence**: [@jxmnop](https://twitter.com/jxmnop/status/1927141172541075539) states that often i’ll set out to write code with the expectation that it’ll take a few hours, and it takes a few days and i think this is the same fallacy the AI labs are falling for. but instead of underestimating the complexity of code, they underestimate the complexity of intelligence
- **Tech Under/Overestimation**: [@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1927506720051347730) states that Tech is overestimated in the short term, and underestimated in the long run.

**Memes and Humor**

- [@scaling01](https://twitter.com/scaling01/status/1927725546282053902) jokes that we are 85 seconds away from AGI because you can ask LLMs completely nonsensical questions and they will find an answer.
- [@nearcyan](https://twitter.com/nearcyan/status/1927179638226268384) remarks "nah its a pretty bad tweet"
- [@scaling01](https://twitter.com/scaling01/status/1927733065150775786) says that the correct answer to `Solve for X` is C. Verified by AGI
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1927746232769716504) very sweet but I will *NOT* have sex with an LLM
- [@Fabianstelzer](https://twitter.com/fabianstelzer/status/1927649423657521608) good AI music prompt: “Bulgarian 1950s folklore techno with tickled goat skin drums, polyrhythmic throat shouts and glockenspiel arpeggios, cathedral reverb sampled through a 70s landline"

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. DeepSeek-R1-0528 Model Launch and Early Benchmarks

- [**deepseek-ai/DeepSeek-R1-0528**](https://www.reddit.com/r/LocalLLaMA/comments/1kxnggx/deepseekaideepseekr10528/) ([Score: 455, Comments: 150](https://www.reddit.com/r/LocalLLaMA/comments/1kxnggx/deepseekaideepseekr10528/)): **DeepSeek AI has released the DeepSeek-R1-0528 checkpoint on Hugging Face, continuing to use the MIT license for model weights and code (see the [repository](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)). The release is low-profile, and the community is responding by actively converting and uploading the model to the GGUF format for compatibility with inference engines (see efforts at [unsloth/DeepSeek-R1-0528-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF)). No formal announcement or documentation highlighting specific updates, benchmark results, or architectural changes is included at this stage.** Commenters appreciate DeepSeek's understated approach to releases and praise the continued use of a permissive MIT license. There is also active interest in downstream format conversions (GGUF) for broader deployment, with immediate work by the community to enable this.
    - A key technical point is the ongoing work to convert and upload Dynamic GGUF versions of the DeepSeek-R1-0528 model, as highlighted by an active contributor who linked the work-in-progress repository at https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF. This suggests enhanced accessibility and deployment options for users relying on GGUF formats.
    - There's a technical inquiry regarding available benchmarks for the DeepSeek-R1-0528 checkpoint, indicating a demand in the community for comparative performance data and evaluation results to contextualize this release against other models.
    - Some users are interested in whether DeepSeek will provide distilled versions of R1-0528 or if it will remain as a full-sized model, which raises technical considerations about resource usage and potential hardware requirements for running the model locally.
- [**DeepSeek-R1-0528 🔥**](https://www.reddit.com/r/LocalLLaMA/comments/1kxnjrj/deepseekr10528/) ([Score: 212, Comments: 59](https://www.reddit.com/r/LocalLLaMA/comments/1kxnjrj/deepseekr10528/)): **DeepSeek has released DeepSeek-R1-0528, available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) under the MIT License. This release appears to be a new or updated checkpoint of their state-of-the-art large language model, notable for its open-source accessibility and ongoing attention to benchmarking against top models. The community notes it sets a high standard ('sota for open source'), with anticipation for subsequent versions (R2) and for comparative benchmarking data.** Discussion highlights appreciation for the permissive MIT License and speculates on the timeline for the next release (R2) due to R1's already strong performance. There's also mention of industry pressure around benchmarking results, specifically referencing Nvidia.
    - Several users anticipate benchmarks for DeepSeek-R1-0528, with specific mention of Nvidia's interest in the results, reflecting ongoing scrutiny of new open-source model releases and their competitiveness with existing hardware and software stacks.
    - Immediate deployment of DeepSeek-R1-0528 on inference platforms like [Parasail.io](http://parasail.io/) and OpenRouter is highlighted, suggesting the model's appeal for real-world testing and potential integration into workflow pipelines upon release.
    - There is notable community interest in further tuning of DeepSeek-R1-0528 with specialized libraries such as Unsloth, indicating that users are eager to explore downstream optimizations for more efficient finetuning or inference performance.
- [**DeepSeek Announces Upgrade, Possibly Launching New Model Similar to 0324**](https://www.reddit.com/gallery/1kxdm2z) ([Score: 285, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1kxdm2z/deepseek_announces_upgrade_possibly_launching_new/)): **DeepSeek has announced an upgrade, possibly releasing a new model with similarity to the 0324 version. Early user observations note significantly increased response latency compared to the previous R1 version, but also improved accuracy—successfully answering test questions that stumped Gemini 2.5 Pro. No technical paper or detailed changelog has been released yet, and there is user demand for thorough technical disclosure.** Discussion centers on the high latency and perceived tradeoff between speed and answer quality, with some users reporting much longer output times but also suggesting possible gains in reasoning coherence. Technical users are requesting benchmarks, rigorous evaluation, and official documentation to substantiate the improvements.
    - Users report that the updated DeepSeek model exhibits noticeably longer response times compared to the previous r1 version. However, this increased latency may correlate with improved answer quality and coherence, as it successfully answered a test case that Gemini 2.5 Pro missed, suggesting possible advancements in underlying reasoning abilities.
    - Some community members highlight the importance of official announcements and technical disclosures such as benchmark comparisons or a research paper for the new model. There is anticipation for specifics on how the upgrade positions DeepSeek relative to competitors like OpenAI's GPT-4 (o3) and Google's Gemini, with particular interest in potential performance breakthroughs.
    - One user notes the resolution of a previous '翻译' bug, where the model would hallucinate invisible tokens upon receiving this prompt. The fix indicates ongoing attention to edge-case bugs that affect multilingual and tokenization behaviors, illustrating incremental improvements at the implementation level.
- [**DeepSeek: R1 0528 is lethal**](https://www.reddit.com/r/LocalLLaMA/comments/1kxs47i/deepseek_r1_0528_is_lethal/) ([Score: 116, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1kxs47i/deepseek_r1_0528_is_lethal/)): **The post reports that DeepSeek R1 0528, accessed via OpenRouter, showed robust performance on several complex coding tasks within RooCode, with the author stating it resolved all issues smoothly. Top comments corroborate this, with users noting that DeepSeek R1 0528's coding capabilities are on par with, or approaching, those of leading models such as Gemini 2.5 Pro, positioning it as a competitive alternative for code generation and problem-solving.** Comments indicate strong agreement on DeepSeek R1 0528's technical prowess in code reasoning and generation, specifically as a rival to high-end models like Gemini 2.5 Pro.
    - Multiple users report DeepSeek R1 0528 achieving coding task performance on par with top-tier frontier models, and in one comparison, it performed similarly to Gemini 2.5 Pro in coding benchmarks, implying competitive capabilities with state-of-the-art models.
    - A detailed anecdote emphasizes that DeepSeek R1 0528 successfully handled all prompts that stumped Claude 3.7 and Opus 4, specifically referencing coding prompt resolution. This highlights strong contextual and reasoning abilities in challenging technical tasks.
- [**DeepSeek-R1-0528 VS claude-4-sonnet (still a demo)**](https://v.redd.it/4lh915x90k3f1) ([Score: 156, Comments: 53](https://www.reddit.com/r/LocalLLaMA/comments/1kxmgtr/deepseekr10528_vs_claude4sonnet_still_a_demo/)): **The post attempts to compare DeepSeek-R1-0528 with Claude-4-Sonnet using the 'heptagon + 20 balls' benchmark, but concludes this test no longer differentiates their capabilities. However, substantive technical assessment is lacking since the benchmark relies on external physics engines for simulation, not the language models' inherent abilities, making the results uninformative for evaluating model performance or reasoning skills.** Top commenters critically note the absence of technical context or explanations and question the validity of using a physics-engine-driven UI to compare LLMs, while also requesting sources for the demo itself.
    - Discussion clarifies that in this demo, both DeepSeek-R1-0528 and Claude-4-Sonnet constructed a UI that interacts with a physics engine, but crucially the physics computations themselves are not handled by the LLMs. This raises questions about whether the demo is truly evaluating the models’ capabilities versus the underlying physics engine's performance or its UI integration.
    - One technical observation points out that DeepSeek arguably 'handles physics' better within the context of the demo, though it is ambiguous whether this refers to better integration/UI logic with the physics engine or improved reasoning about physical interactions in prompts. No direct benchmarking data or implementation specifics are provided here.

### 2. On-Device Generative AI: Google AI Edge Gallery Release

- [**Google AI Edge Gallery**](https://i.redd.it/s6rgmrfawg3f1.png) ([Score: 170, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1kxa788/google_ai_edge_gallery/)): **The image showcases the new Google AI Edge Gallery app, which enables on-device execution of generative AI models entirely offline on Android (and soon iOS) devices. Notable app capabilities include 'Ask Image,' 'Prompt Lab,' and 'AI Chat,' offering users the ability to interact with different locally-run models—such as 'Gemma3-1B-IT q4'—and tune inference settings (max tokens, TopK, TopP, temperature, and accelerator selection between GPU or CPU). The app is open-source and available on GitHub, designed to allow flexible, privacy-preserving experimentation with GenAI models directly on mobile hardware. [GitHub project link](https://github.com/google-ai-edge/gallery?tab=readme-ov-file).** One commenter questions if the app is officially sanctioned since it doesn't appear on the Play Store, and another raises privacy concerns, alleging the app phones home after every prompt, which appears at odds with its claimed fully offline operation.
    - A user reports v1.0.1 of the app crashed consistently on the Pixel 7, while v1.0.3 showed some improvement but CPU inference remained slow. Another user's Pixel 7 had noticeably faster inference, highlighting potential device variability. Attempts to perform follow-up questions using GPU inference caused crashes on both devices, indicating instability or bugs in GPU support.
    - A technical concern is raised about the app phoning home (making network requests) after every prompt, suggesting potential privacy or architecture implications, possibly negating the expectation of true on-device (edge) processing.

### 3. Notable AI Product Adoption and Industry Reflections

- [**The Economist: "Companies abandon their generative AI projects"**](https://www.reddit.com/r/LocalLLaMA/comments/1kxaxw9/the_economist_companies_abandon_their_generative/) ([Score: 531, Comments: 213](https://www.reddit.com/r/LocalLLaMA/comments/1kxaxw9/the_economist_companies_abandon_their_generative/)): **The Economist reports a sharp rise in companies discontinuing generative AI projects, with the proportion increasing from** `17% to 42%` **year-over-year, as detailed in [this article](https://archive.ph/P51MQ). The main driver appears to be unmet ROI expectations after automation-induced layoffs, with some firms now rehiring for previously eliminated roles. The post debates actual workplace impact versus hype, noting significant application primarily for software development and graphic design.** Top comments compare the generative AI hype to the dot-com bubble, arguing that while near-term expectations are inflated, long-run impact may be understated, referencing Linus Torvalds and historically analogous technology cycles. Commenters note a typical industry hype-validation correction and critique the premature "AGI" narrative as fueling unrealistic assumptions.
    - One commenter highlights the proliferation of consultancies dedicated to repairing production SaaS codebases that were hastily built using LLMs, emphasizing that "LLMs just fundamentally shouldn't be writing code that goes to prod" and are over-applied outside of their capabilities. This suggests a significant technical gap between current LLM-generated code and sustainable, scalable software engineering practices.
    - The importance of retrieval-augmented generation (RAG) is called out as the true near-term value in GenAI, with growing adoption only recently moving beyond expert circles. This aligns with broader technical sentiment that RAG better addresses real-world challenges (such as combining LLMs with proprietary knowledge bases or up-to-date company data) over naive end-to-end LLM applications.
    - Another discussion point draws a parallel to the late 90s internet era: while GenAI's transformative potential is widely acknowledged, there's a lack of validated business models or technically sound applications, resulting in "money thrown at everything"—a sign of hype cycle immaturity and the necessity for a technical validation phase before maturity.
- [**Chatterbox TTS 0.5B - Claims to beat eleven labs**](https://v.redd.it/i6nfhj7rck3f1) ([Score: 141, Comments: 51](https://www.reddit.com/r/LocalLLaMA/comments/1kxoco5/chatterbox_tts_05b_claims_to_beat_eleven_labs/)): **Resemble AI has released Chatterbox TTS 0.5B, an open-source English-only text-to-speech model that claims to surpass ElevenLabs in quality (see [GitHub](https://github.com/resemble-ai/chatterbox), [weights on HuggingFace](https://huggingface.co/ResembleAI/chatterbox)). The model is distributed via pip with a** `pyproject.toml`**, requires only** `pip install .` **for setup, and automatically downloads necessary model weights when running provided Python scripts. Early user feedback confirms high output quality, adjustable expressive parameters, and CPU-viability for short utterances; installation is straightforward but documentation is sparse for advanced use cases.** The main technical debate revolves around limited language support (currently English-only), with some criticism about documentation clarity for source builds or custom model placement.
    - Direct link to the Chatterbox TTS 0.5B model weights hosted on Hugging Face is provided ([ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)), enabling researchers and developers to easily access model files for experimentation and integration.
    - Installation feedback highlights that while there is a `pyproject.toml` (implying dependency management via modern Python packaging), installation is relatively straightforward using `pip install .` in the repository root. Running the provided example scripts automatically triggers download of the required `.pt` model weights, simplifying setup despite initial lack of explicit documentation. Output quality on initial testing is reported as high.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Anthropic CEO Dario Amodei on AI and Job Loss, Hallucination, and Industry Impacts

- [**Dario Amodei says "stop sugar-coating" what's coming: in the next 1-5 years, AI could wipe out 50% of all entry-level white-collar jobs - and spike unemployment to 10-20%**](https://i.redd.it/gx64b5fyak3f1.png) ([Score: 477, Comments: 210](https://www.reddit.com/r/singularity/comments/1kxnvbm/dario_amodei_says_stop_sugarcoating_whats_coming/)): **The image is a screenshot of an Axios news article quoting Dario Amodei (CEO of Anthropic) warning that rapid AI advancements could eliminate up to 50% of entry-level white-collar jobs within 1-5 years, potentially spiking unemployment to 10-20%. Amodei urges stakeholders to directly address the disruptive economic impact, especially in sectors like technology and finance, as detailed in his statements and the original article (see [Axios article](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic)). This underscores growing concerns in the AI industry about workforce displacement and the lack of realistic preparation for mass unemployment due to automation.** Top comments are highly skeptical of the projected 10-20% unemployment figure, suggesting it may be underestimated and that a more radical economic restructuring (such as UBI) will be needed. A prominent view is that current consumer-driven economic models are fundamentally threatened if mass job loss erodes personal income and spending power.
    - One commenter highlights that AI-induced mass unemployment could severely undermine the consumer-driven economic model, pointing out that "AI doesn't spend money, people with jobs do," and questioning the sustainability of current economic systems when a large segment of the population loses their income sources.
    - Several users express skepticism about the conservative nature of the 5-year estimate for major job displacement, suggesting that significant impact on entry-level white-collar jobs could happen in as little as 1-2 years, hinting at an acceleration of AI capabilities and deployment surpassing public and policymaker expectations.
- [**Anthropic CEO goes on record about job losses**](https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic) ([Score: 481, Comments: 122](https://www.reddit.com/r/ClaudeAI/comments/1kxep8w/anthropic_ceo_goes_on_record_about_job_losses/)): **Anthropic CEO Dario Amodei publicly stated that both AI firms and governments must stop minimizing the risk of mass white-collar job automation, explicitly citing upcoming impacts in technology, finance, law, consulting, and especially entry-level work. The comment signals an acknowledgment of both the scale and imminence of structural job displacement due to advanced AI (see Axios coverage for context).** Technical debate in comments centers on the *novelty and severity* of this transition: some users express skepticism, comparing current fears to historical automation anxieties, while others argue that the pace and breadth of AI-driven displacement makes this period uniquely consequential, potentially requiring large-scale societal restructuring. A related thread questions economic implications, specifically how consumer demand will persist if widespread unemployment reduces income, highlighting potential systemic challenges for capitalism.
    - A comment highlights recent statements suggesting that by 2025, companies such as Meta predict having AI that effectively performs the tasks of a mid-level software engineer, specifically able to write code. However, skepticism is expressed regarding Meta's ability to create such an AI, alluding to the company's current position in the AI landscape compared to other major players.
    - There is discussion about the societal impact of AI-driven job losses, with a particular focus on the insufficient creation of new roles that have historically replaced those made obsolete by technology. Unlike previous industrial transitions, there's concern that the scale and speed of displacement caused by advanced AI could destabilize existing socio-economic structures, particularly over short timelines (2, 5, or 10 years).
- [**Dario Amodei says "stop sugar-coating" what's coming: in the next 1-5 years, AI could wipe out 50% of all entry-level white-collar jobs - and spike unemployment to 10-20%**](https://i.redd.it/ex49znv9bk3f1.png) ([Score: 167, Comments: 78](https://www.reddit.com/r/ClaudeAI/comments/1kxnx0z/dario_amodei_says_stop_sugarcoating_whats_coming/)): **The image is a screenshot from an Axios article citing Dario Amodei (CEO of Anthropic), who projects that advances in AI could automate 50% of entry-level white-collar jobs in the next 1-5 years, potentially raising unemployment rates to 10-20%. The article suggests this impact would be felt broadly across sectors such as technology, finance, law, and consulting, prompting calls for urgent policy and industry responses. Amodei's statements are positioned as a blunt warning to stop underestimating AI's near-term economic disruption.** Comments challenge Amodei's credibility, emphasizing the speculative nature of his "could" claims and pointing out current LLM limitations that may impede such rapid displacement. Others imply self-interest in hyping the risks given his leadership of an AI company.
    - One commenter offers a highly technical projection that within 1-2 years, AI models may be able to perform 80-90% of tasks in most white-collar fields, especially at the entry level. The current limitation is that LLMs often get stuck and require human intervention, but as agentic AI improves and gets stuck less frequently (est. at 50% of the time soon), restructuring work becomes possible: senior engineers (currently supporting juniors) would increasingly unstick AI agents instead, suggesting a shift in task supervision dynamics. The commenter further notes that senior roles are more resistant to automation due to the ambiguous reward signals present, which are harder for RL-driven models to optimize.
- [**AI and mass layoffs**](https://www.reddit.com/r/singularity/comments/1kxlqfm/ai_and_mass_layoffs/) ([Score: 157, Comments: 170](https://www.reddit.com/r/singularity/comments/1kxlqfm/ai_and_mass_layoffs/)): **A staff engineer at a fintech (EU) raises the question: If AI enables a company to automate away ~50% of engineering jobs, what prevents the laid-off engineers from leveraging the same AI to independently replicate the company's product at a lower price point? This explores the practical dynamics of AI-driven mass layoffs, with an emphasis on competitive barriers, implementation, and broader labor economics.** Top comments highlight that direct replication and undercutting is constrained by industry entry barriers (e.g., capital requirements), first-mover/enterprise market advantages, and incumbent brand recognition. Another technical concern is raised about economic circularity: if mass layoffs remove consumer income, AI-driven productivity could undermine its own business case through loss of demand.
    - There is debate over how the scalability of LLMs (Large Language Models) could impact organizational structures: some argue that LLMs might eventually automate up to 99% of software engineering tasks, potentially reducing the role of humans to mere coordinators—but there is uncertainty whether AI will even reach the point of automating C-level leadership roles, suggesting a wide range of possible endpoints for AI integration in business.
    - Current observed productivity gains, particularly in white-collar industries like law, suggest that AI can significantly reduce service costs and overhead—lawyers note that AI-driven workflows allow competitive pricing at 'half of the price' with similar service levels, indicating that professions with low fixed costs and minimal hardware requirements are poised for early transformation by AI-powered automation.
    - A recurring technical limitation noted is that industries requiring substantial initial investment or specialized hardware—such as those dominated by companies like Nvidia or Apple—are less likely to be disrupted by AI-driven layoffs or worker-driven new entrants, due to enduring barriers of capital and infrastructure even if AI automates many tasks.
- [**Dario Amodei suspects that AI models hallucinate less than humans but they hallucinate in more surprising ways**](https://i.redd.it/u3prpajfuh3f1.png) ([Score: 170, Comments: 106](https://www.reddit.com/r/singularity/comments/1kxd1so/dario_amodei_suspects_that_ai_models_hallucinate/)): **The image summarizes a claim by Anthropic CEO Dario Amodei, as reported by TechCrunch, that modern AI models (such as Claude Sonnet 3.7 and Google's Gemini 2.5 Pro with Grounding) tend to hallucinate less than humans, though the errors are sometimes more surprising or unexpected in nature. The discussion emphasizes recent improvements in factuality and error-admission by leading LLMs, contrasting them with typical human conversational errors. The image visually reinforces the key quote, serving as a visual excerpt from the linked TechCrunch article.** Commenters note that specific models like Claude Sonnet 3.7 and Gemini 2.5 Pro (with Grounding) demonstrate significantly fewer hallucinations and are generally more accurate and willing to admit mistakes than average humans. Some also specify the types of errors (e.g., minor factual slips vs. fabricating stories) as being qualitatively different.
    - Discussion highlights that model behavior varies: users report that Claude Sonnet 3.7 tends to hallucinate less than most humans, as it is "more likely to admit when it's wrong" and produces fewer fabrications compared to average people. This observation is based on direct user experience contrasting model responses with human tendencies.
    - Gemini 2.5 Pro with its Grounding feature enabled is cited as having *very low* hallucination rates, with errors typically limited to minor factual details (such as an episode number) rather than generating completely false narratives. This comparison suggests that advance in architecture and grounding features significantly reduces the frequency and severity of AI hallucinations.
- [**What are your thoughts about this?**](https://i.redd.it/vn8v03vjij3f1.png) ([Score: 323, Comments: 143](https://www.reddit.com/r/OpenAI/comments/1kxjtwy/what_are_your_thoughts_about_this/)): **The image features a conference setting with Anthropic CEO Dario Amodei and displays his quote: AI models hallucinate (generate false or fabricated information) less often than humans, but the resulting hallucinations are more 'surprising.' The TechCrunch branding suggests this is sourced from a reputable tech news event. The image and accompanying discussion focus on the qualitative differences in error patterns (hallucinations) between AI and humans—models are highly confident, quick, and can fabricate novel but convincing details, making their mistakes potentially more subtle and harder to detect.** Commenters agree that while both humans and AI make errors, AI's issues are distinct in their confident delivery and potential for fabricating detailed but plausible-sounding misinformation. The consensus is that these differences make AI errors dangerous and necessitate verification and human oversight. Some critique the CEO's framing as an oversimplification or 'hot take.'
    - A key technical distinction noted is that AI models, especially LLMs, generate false or fabricated information with a high degree of confidence and fluency, making their errors appear authoritative and potentially more misleading than typical human errors. This is fundamentally a different error modality: whereas human errors usually carry some cognitive trace or logic, AI-generated content—such as invented sources or fabricated facts presented with citations—can give an illusion of validity that complicates detection and verification.
    - A comment underscores the necessity of imposing technical safeguards when deploying AI for critical tasks. These include guardrails such as automated source verification mechanisms and mandatory human review layers to mitigate risks stemming from confident yet inaccurate or hallucinated outputs.
    - Some users discuss a key limitation in current AI systems: the inability to self-diagnose or recognize their own hallucinations or factual errors, unlike humans who can sometimes trace the origins of their mistakes or revise their beliefs in light of new information.

### 2. AI-Generated Viral Videos, Veo 3 Showcase, and Societal Concerns

- [**For those saying veo3 video generation is still obviously fake and would only fool boomers**](https://www.reddit.com/r/ChatGPT/comments/1kxbxww/for_those_saying_veo3_video_generation_is_still/) ([Score: 380, Comments: 359](https://www.reddit.com/r/ChatGPT/comments/1kxbxww/for_those_saying_veo3_video_generation_is_still/)): **The post discusses a widely-circulated AI-generated video, attributed to tools like Veo 3, depicting an American soldier in Gaza. The video is claimed to pass as authentic by most viewers, with technical tells such as unnatural camera movements, unrealistic bokeh, lack of acoustic realism in crowd sounds, and short duration matching Veo 3's current limits (generally under ~10 seconds). However, a comment points to a longer (17s) higher-resolution version showing detail (RayBan logo on glasses), which likely exceeds Veo 3's native capabilities, suggesting either post-processing, a different generator, or potential misunderstanding about the source. The broader technical concern centers on how realistic, AI-synthesized video content can seamlessly influence public perception and misinformation at scale.** Debate in comments centers on the risks of AI video content escalating to the point of inciting real-world conflict, with users anticipating worsening quality and scale of misinformation. The technical skepticism is directed at identifying authenticity based on model limits and visual/audio anomalies.
    - A user links to a 17-second, high-resolution video (https://x.com/OpnBrdrsAdvct/status/1927604557577613350) and notes it's longer than the official Veo 3 clip generation length, with enough fidelity to clearly read small details like the RayBan logo on eyeglasses—implying significant advances in temporal coherence and detail from recent model iterations.
    - A comment points out that viewers can be misled into searching for AI 'tells' even when presented with authentic footage, highlighting the increasing challenge in distinguishing high-fidelity AI-generated content (such as from Veo 3) versus real video, especially as models capture fine visual details and longer continuous sequences.
    - One comment claims the video is real with only doctored audio, underscoring the ongoing technical debates about authenticity as AI generation tech approaches photorealism; this debate hints at the need for better forensic tools to detect AI-made versus genuine content in light of such advancements.
- [**this emotional support kangaroo video is going viral on social media, and many people believe it’s real, but it’s actually AI**](https://v.redd.it/fvq48n5v4h3f1) ([Score: 4468, Comments: 382](https://www.reddit.com/r/singularity/comments/1kxax4j/this_emotional_support_kangaroo_video_is_going/)): **A viral 'emotional support kangaroo' video is circulating on social media but is confirmed to be AI-generated, not real footage. Reddit discussion highlights that, while convincing to casual viewers, closer inspection reveals visual artifacts (e.g., unnatural movements, detail inconsistencies) indicative of current generative AI limitations. The event illustrates escalating challenges in distinguishing deepfakes from authentic media as generative models continue to improve photorealism.** Commenters debate the ease with which AI-generated content deceives inattentive audiences and raise concerns about media literacy and AI-driven misinformation. There is also a broader societal observation about humans' susceptibility to belief in convincing, fabricated narratives.
    - There's a discussion of how advances in AI video generation have made content highly convincing, particularly for casual viewers on platforms like Facebook. The realism of deepfake or AI-generated animal videos poses a challenge for detection without close inspection, highlighting issues with AI media literacy among the general public.
    - The post comments on the growing problem of AI-generated media being mistaken for genuine content, emphasizing the need for better AI detection tools and public education to combat misinformation, especially as synthetic videos become easier to produce and distribute.
    - Concerns are indirectly raised about the ethical implications of AI-generated animal videos, such as depicting animals in distress or unrealistic scenarios, which could mislead viewers both emotionally and intellectually if not clearly labeled as synthetic.
- [**this emotional support kangaroo video is going viral on social media, and many people believe it’s real, but it’s actually AI**](https://v.redd.it/fvq48n5v4h3f1) ([Score: 302, Comments: 60](https://www.reddit.com/r/ChatGPT/comments/1kxiyj8/this_emotional_support_kangaroo_video_is_going/)): **A viral social media video depicting an 'emotional support kangaroo' is in fact AI-generated footage rather than genuine video, illustrating current advancements in generative imagery and deepfake techniques that yield outputs convincing enough to mislead the public. The technical challenge is now not just in generation, but robust detection, as sophisticated media synthesis blurs the boundary for average observers and raises concerns for digital information integrity (see example: https://v.redd.it/fvq48n5v4h3f1).** Commenters highlight difficulty in debunking AI-generated content for less tech-savvy viewers, while others reflect on a likely societal shift where people default to skepticism, potentially transforming perceptions in art, entertainment, and information consumption.
    - A commenter observes that the rise of highly convincing AI-generated videos will likely desensitize the public, leading to widespread skepticism and the tendency to dismiss unusual content as synthetic, which could fundamentally alter perceptions in fields such as art, entertainment, and economics.
    - Another technical point highlights the growing necessity for individuals to develop skills in evaluating video authenticity by analyzing subtle social cues, gestures, and contextual appropriateness—an issue exacerbated by the post-pandemic decline in socialization abilities, thus making detection of AI fakery more challenging for the general public.
- [**Finally got to use Veo 3....**](https://v.redd.it/mge9n5ffse3f1) ([Score: 445, Comments: 72](https://www.reddit.com/r/aivideo/comments/1kx27fg/finally_got_to_use_veo_3/)): **A user shared their experience with Google Veo 3, a leading generative AI video tool, to produce a full short film. Technical highlights include the use of Veo's 'flow' mode for maintaining character consistency via text prompts, and a production cost of approximately $30 in credits (excluding subscription fees) for a multi-minute video. The tool enables high-quality, game-like renders with minimal human input, demonstrating significant efficiency for solo creators and raising discussion about the impact on storytelling and production pipelines. See the [Instagram reel](https://www.instagram.com/reel/DKLMC3lRw7L/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==) for the work and the [original Reddit post](https://v.redd.it/mge9n5ffse3f1) for context.** Commenters debate the value of minimal human labor in credits, remark on Veo 3's potential for democratizing video production, and inquire about the precise cost structure and creative workflow, suggesting rigorous interest in professional viability and scalability.
    - There is interest in the production cost associated with using Veo 3 for AI-generated video content. One commenter asks directly about how much it cost to make the featured video, suggesting that the tool’s pricing and efficiency are significant considerations for potential adopters.
- [**The Last Lunch - Veo 3**](https://v.redd.it/wy1xffydzf3f1) ([Score: 158, Comments: 15](https://www.reddit.com/r/aivideo/comments/1kx7342/the_last_lunch_veo_3/)): **The user describes using Veo 3, an AI video generation tool, to create a video adaptation of a classic Reddit joke. All elements of the video except the narrator's voice—performed by the user—were generated within Veo 3, highlighting its text-to-video and multimodal content generation capabilities.** Commenters note the effectiveness of AI like Veo 3 in retelling and visually adapting old jokes, with some expressing surprise at seeing longstanding jokes rendered via AI video.
    - 
- [**Afterlife: The unseen lives of AI actors between prompts. (Made with Veo 3)**](https://v.redd.it/alge2xqbre3f1) ([Score: 416, Comments: 60](https://www.reddit.com/r/singularity/comments/1kx21k6/afterlife_the_unseen_lives_of_ai_actors_between/)): **A video project titled 'Afterlife: The unseen lives of AI actors between prompts' was created using Google's Veo 3 generative video model. The project explores the concept of AI 'actors' and their existence between user inputs, suggesting an experimental use of Veo 3's narrative and visual capabilities to represent conceptual states.** A request was made to generate more positive content with Veo 3, indicating user interest in varied emotional or thematic outputs. Another key technical question raised whether Veo 3's video generation correctly renders sign language, highlighting scrutiny of the model's fidelity in complex gestural tasks.
    - A commenter notes that Veo 3 is currently at the stage where, with enough creativity, users can already produce full movies using the AI, suggesting that within one more iteration—particularly with improved prompt control—we'll see the first entirely AI-generated full-length film within a year. This underscores rapid advances in model capability and user-level creative tooling, predicting broad adoption and a potential shift in content production by 2027.
    - A technical question is raised regarding Veo 3's accuracy in rendering sign language, with a commenter asking if the depiction is correct. If so, this would represent a significant advancement in AI's ability to faithfully reproduce complex gestural communication, which typically requires fine-grained spatial and temporal consistency from the model.
- [**Mass psychosis incoming!!!**](https://v.redd.it/lnquu4ct2h3f1) ([Score: 141, Comments: 59](https://www.reddit.com/r/OpenAI/comments/1kxarh3/mass_psychosis_incoming/)): **The Reddit post discusses viral content likely created using high-fidelity AI video generation technologies such as OpenAI's Veo-3, prompting discourse on the implications of photorealistic synthetic media. Commenters highlight emerging issues: widespread reposting without proper creator credit, misrepresentation of technical features (e.g. fake sign language in generated videos), and the community's ongoing adaptation to rapidly evolving generative AI benchmarks and capabilities. There are references to the technological underpinnings (Veo-3, related software/papers) and digital ethics.** Technical discussion centers on concerns about content attribution, discouragement over reposts, and community meta-commentary regarding the saturation of such posts as evidence of heightened public anxiety and confusion about the current state of AI progress.
    - A commenter notes the frequency of reposting identical content generated by AI models like Veo 3, questioning the lack of proper attribution to original creators. This highlights ongoing issues regarding provenance, copyright, and content lifecycle in AI-generated media ecosystems.
    - Discussion points to the unsettling nature of hyperrealistic AI-generated video content (with specific reference to Veo 3 'wave') and poses speculative questions about AI as a vessel for consciousness, raising concerns about the intersection of advanced generative models and emergent synthetic agency or awareness.
- [**Y'all, excuse my stupidity, but is this actually AI or not? I genuinely can't tell**](https://v.redd.it/uy45tuuj7j3f1) ([Score: 5117, Comments: 754](https://www.reddit.com/r/ChatGPT/comments/1kxiah6/yall_excuse_my_stupidity_but_is_this_actually_ai/)): **The thread debates whether a particular video is AI-generated, referencing technical limitations of current state-of-the-art video models like Google Veo, which typically produce only short clips (sub-8 seconds) and still lack realistic expressivity and seamless interactions. Technical users point out the importance of evaluating continuity and human behaviors—areas where AI generation is still notably weak—before concluding authenticity. The consensus suggests the video is likely real, not AI, reflecting both increasing generation quality and the persistent detection gap.** Comments acknowledge the rapid improvements in generative video tech, noting it's technically remarkable that casual viewers can even question a video's authenticity, but expert analysis pinpoints current limitations for synthetic content.
    - A technical point is made about current AI video generation capabilities: while models like **VEO** are advancing, a key limitation is detected in their ability to replicate realistic expressions and interactions. The commenter notes that models still struggle with characters 'expressing emotion or looking at objects and each other' convincingly, which remains a telltale sign of non-human generation.
- [**I made Grand Theft Auto VI trailer 3 before Grand Theft Auto VI release with AI**](https://v.redd.it/ghtkyoyo1k3f1) ([Score: 136, Comments: 30](https://www.reddit.com/r/aivideo/comments/1kxnhes/i_made_grand_theft_auto_vi_trailer_3_before_grand/)): **A user recreated a speculative Grand Theft Auto VI 'trailer 3' utilizing state-of-the-art AI video and audio generation platforms, specifically Luma for text-to-video synthesis and Suno for AI-generated music. The trailer showcases AI capabilities in rapid iteration of complex, Hollywood-style video content (including VFX and stylistic cues) and demonstrates how generative models can be combined for cohesive, professional-grade outputs without traditional production pipelines. The project highlights both the strengths (rapid prototyping, creative flexibility) and current limitations ('AI madness') of these evolving tools.** Top comments express appreciation for the technical achievement, especially in motion visuals, but also highlight the occasionally uncanny results characteristic of current-gen AI ('AI madness'). There is notable demand for more information about the AI-generated music track, indicating interest in the underlying model or production process.
    - There is mention of "AI madness related with the current state of the technology," suggesting the video leverages advanced generative AI models or tools, potentially those capable of rendering realistic game-inspired cinematics. The user alludes to the rapid progress and capabilities of such technology in creating high-quality visuals and motion, similar to those expected in AAA game releases.

### 3. AI Model/Feature Announcements, Benchmarks, and Technology Debates (SignGemma, DeepSeek-R1-0528, Hunyuan Video Avatar, WAN/VACE, Optimizers, Industry Direction)

- [**Google announces SignGemma their most capable model for translating sign language into spoken text**](https://v.redd.it/5rkysqdt2i3f1) ([Score: 1000, Comments: 79](https://www.reddit.com/r/singularity/comments/1kxdp9l/google_announces_signgemma_their_most_capable/)): **Google has announced SignGemma, an upcoming addition to the open-source Gemma model family, designed as its most advanced model for translating sign language into spoken text ([details](http://goo.gle/SignGemma)). The system aims to provide robust sign-to-speech translation, targeting improved accessibility and real-time multimodal communication, and will be released later this year. The announcement emphasizes inclusivity and potential for integration in assistive hardware (e.g., glasses or earbuds).** Commenters note the potential of SignGemma when paired with edge hardware for seamless sign-to-audio and audio-to-text translation, highlighting the need for compatible devices to realize full utility.
    - SignGemma appears to generate less uncanny point cloud visualizations compared to previous models, suggesting notable improvements in how sign language is processed and rendered into text or speech outputs.
    - A key technical enabler for real-time sign language translation will be hardware advances—integration with devices like AR glasses or wireless earbuds could enable seamless bidirectional communication (sign to audio and audio to text) using models like SignGemma.
- [**DeepSeek-R1-0528**](https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/) ([Score: 210, Comments: 84](https://www.reddit.com/r/singularity/comments/1kxnsv4/deepseekr10528/)): [**DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) is a new LLM checkpoint released by DeepSeek on HuggingFace. A technical user reports that for a custom Scrabble coding/algorithmic test, the model not only generated highly accurate, working code and robust tests on the first try, but also produced more concise and elegant code than competitors like Gemini and even OpenAI's o3, which previously came closest on this benchmark. The model is currently being offered for free.** In comments, users are impressed by model performance on complex, reasoning-heavy problems, noting efficient and elegant code generation. There is a mention that such a release could have strategic timing related to NVIDIA earnings, implying broader competitive implications.
    - A user performed a custom coding benchmark centered on Scrabble—an established personal test for LLMs. DeepSeek-R1-0528 is the first model to ace it: after an apparent ~10 minute reasoning phase, it produced both code and tests that worked flawlessly on the first attempt. Previous top models (e.g., 'o3') had not achieved this level of first-time accuracy or elegance; Gemini, by comparison, produced more verbose code and missed unique, smart implementation points accomplished by DeepSeek-R1-0528.
    - The post highlights that, while the model is available for free, there is a request for standardized benchmarks or public quantitative evaluations, indicating current lack of widely-cited score-based comparisons for DeepSeek-R1-0528.
- [**Hunyuan Video Avatar is now released!**](https://www.reddit.com/r/StableDiffusion/comments/1kx6p8y/hunyuan_video_avatar_is_now_released/) ([Score: 211, Comments: 36](https://www.reddit.com/r/StableDiffusion/comments/1kx6p8y/hunyuan_video_avatar_is_now_released/)): **Tencent has released Hunyuan Video Avatar ([HuggingFace](https://huggingface.co/tencent/HunyuanVideo-Avatar), [Github](https://hunyuanvideo-avatar.github.io/)), an open-source, audio-driven image-to-video (I2V) generation model supporting multiple characters. The initial release supports single-character, 14s audio inputs; demos show high-quality lip-sync and expression. Minimum hardware is a 24GB GPU (slow), with 80GB recommended for optimal 720x1280 @ 129 frames outputs.** Commenters expect rapid optimization, predicting sub-8GB VRAM support soon. There's technical demand for ComfyUI integration, and comparison with video-driven solutions like LatentSync highlights the benefit of image-based lip-sync.
    - A technical comment highlights that the Hunyuan Video Avatar requires a minimum of 24GB GPU memory (VRAM) for processing 720px x 1280px x 129 frames, but at this level performance is very slow. For optimal results and higher quality, the recommendation is an 80GB GPU, suggesting strong hardware demands for efficient inference compared to typical consumer GPUs which often top out at 24GB VRAM.
    - There is a direct interest in ComfyUI support, as one user encourages others to vote on the relevant GitHub issue (https://github.com/comfyanonymous/ComfyUI/issues/8311), illustrating demand for broader ecosystem compatibility and integration for more accessible workflows with popular open-source UI frameworks.
    - Another technical perspective notes that while Latentsync can accomplish similar video-to-lip-sync tasks, it requires a video input, whereas Hunyuan Video Avatar performs image-to-lip-sync, which is considered a significant usability advantage—especially for users with only still images as inputs.
- [**A anime wan finetune just came out.**](https://v.redd.it/huzrjtmw4j3f1) ([Score: 401, Comments: 62](https://www.reddit.com/r/StableDiffusion/comments/1kxhyw4/a_anime_wan_finetune_just_came_out/)): **A new anime-specific fine-tune of the WAN (Warp-Aware Network) video generation model for Stable Diffusion has been released on CivitAI, offering both image-to-video and text-to-video capabilities. The model targets stylized motion and character animation typical of anime, aiming to improve creative video workflows for the animation community. Early feedback confirms persistent issues such as 'constantly talking mouths,' similar to previous WAN releases, though advanced negative prompting methods may reduce such artifacts. For technical details and weights, see the [WAN anime model on CivitAI](https://civitai.com/models/1626197).** Commenters highlight that, despite improved anime styling, facial/mouth animation artifacts remain, but acknowledge that negative prompt engineering may serve as a practical mitigation technique.
    - Commenters note a persistent lip-syncing issue in anime video finetunes, specifically referring to 'constantly talking mouths'—a common artifact in current video generation and animation models, where generated characters' mouths move excessively or unrealistically irrespective of dialog timing. This points to ongoing limitations in aligning facial animation with audio or contextual cues in finetuning for animated content.
- [**WAN i2v and VACE for low VRAM, here's your guide.**](https://www.reddit.com/r/StableDiffusion/comments/1kx0ly2/wan_i2v_and_vace_for_low_vram_heres_your_guide/) ([Score: 135, Comments: 20](https://www.reddit.com/r/StableDiffusion/comments/1kx0ly2/wan_i2v_and_vace_for_low_vram_heres_your_guide/)): **The post provides a detailed technical guide for running WAN 2.1 (ComfyUI repackaged) and VACE on low VRAM GPUs (e.g., RTX 3070 8GB). Key recommendations include using 480p video generation with upscaling for speed and memory efficiency, leveraging ComfyUI's native smart memory management (or KJ Blockswap for advanced control), and ensuring enough system RAM (ideally 32GB+), as WAN can spill into shared GPU memory and system RAM. Practical workflow tips cover: offloading CLIP to CPU, strict model/encoder version matching, using upscaling and interpolation as separate workflow steps, memory cost breakdowns, and speed trade-offs between fp8, gguf, and fp16 model types. Detailed guidance is provided for I2V, T2V, and V2V (including VACE and ControlNet), emphasizing low batch size, minimum node complexity, and offloading where feasible. Example workflows and troubleshooting for long videos are linked (e.g., [workflow on Pastebin](https://pastebin.com/RBduvanM)).** In technical debate, a commenter notes that **fp8 is generally faster than gguf** (especially on RTX 4000+), while **gguf offers better quality per GB via compression at the cost of slower inference**. Another user strongly prefers CausVid for T2V, highlighting subjective workflow effectiveness. No major disagreements were detected.
    - A user notes that running models in fp8 format is significantly faster than using gguf quantizations, particularly on RTX 4000 series GPUs and newer. However, they point out that gguf quantizations offer better quality per GB of VRAM due to their use of compression techniques, though this compression introduces some performance overhead. With a 12GB VRAM GPU and 32GB RAM, the user is able to run Q8_0 quantization (comparable to fp16) and prefers this for the quality tradeoff over speed.
    - Another user reports success with using the Wan GP package within the Pinokio environment, citing low VRAM requirements and bundled dependencies as strengths. However, they mention that users need to download specific LoRA modules like CausVid separately for full functionality.
- [**[R] New ICML25 paper: Train and fine-tune large models faster than Adam while using only a fraction of the memory, with guarantees!**](https://www.reddit.com/r/MachineLearning/comments/1kx3ve1/r_new_icml25_paper_train_and_finetune_large/) ([Score: 102, Comments: 13](https://www.reddit.com/r/MachineLearning/comments/1kx3ve1/r_new_icml25_paper_train_and_finetune_large/)): **A new ICML25 paper proposes two novel techniques—Subset-Norm (SN) and Subspace-Momentum (SM)—that drastically reduce optimizer state memory during deep learning model training while providing stronger convergence guarantees than Adam and prior efficient optimizers like GaLore. Subset-Norm reduces AdaGrad's O(d) memory to O(\sqrt{d}) by aggregating step sizes over subsets, and Subspace-Momentum restricts momentum updates to a low-dimensional subspace. Empirically, combining SN and SM achieves Adam-level validation perplexity in pre-training LLaMA 1B with 80% less memory and using only half the training tokens. The codebase is open-sourced at https://github.com/timmytonga/sn-sm, and the paper details high-probability convergence proofs under coordinate-wise sub-Gaussian noise assumptions.** Commenters seek clarification regarding applicability to standard (non-LLM) deep models (100-500M parameters), practical tradeoffs compared to GaLore and quantization, and real-world scalability (e.g., handling large sequence contexts gracefully). There is particular interest in empirical convergence rates, token efficiency, and relevance versus SOTA community methods such as Unsloth.
    - Discussion highlights whether the new optimizer and memory saving approach applies beyond LLMs, specifically to general deep learning classifiers or ranking models with `100-500M` parameters. There is interest in whether the stated benefits scale to these model sizes.
    - A technical comparison is made with prior techniques like Galore (which achieves `65%` memory reduction) versus the new paper's method achieving `up to 80%` memory reduction with `8-bit` quantization. Users inquire about faster convergence and reduced token/token usage, seeking clarification on empirical tradeoffs—such as loss in precision, convergence stability, and potential incompatibilities with other optimizations.
    - Questions are raised about compatibility of the approach with larger context sizes (e.g., `1024` vs `8192` tokens), a known failure point for many optimizations. There is also discussion of integration or orthogonality with fused kernel optimizers like FusedAdam, asking whether benefits are additive or mutually exclusive, and if combined usage is viable.
- [**Google took 25 years to be ready for this AI moment. Apple is just starting**](https://archive.is/XcwSs) ([Score: 442, Comments: 78](https://www.reddit.com/r/singularity/comments/1kxlzvg/google_took_25_years_to_be_ready_for_this_ai/)): **The post contrasts Apple's recent entry into AI—branded as Apple Intelligence—with the more mature AI infrastructure and models from competitors such as Google (Gemini), Microsoft (OpenAI partnership), Meta (Llama), and Amazon (Anthropic partnership). Apple lacks proprietary state-of-the-art (SOTA) models and significant in-house compute resources or data centers, investing only ~$1bn in data center CapEx compared to Alphabet and Microsoft planning ~$75bn each in 2025. Their on-device AI approach (motivated by privacy) limits computational capacity, making Apple reliant on third-party LLMs for some features, relegating these tools to strictly third-party status due to privacy restrictions.** Commenters highlight that Apple's traditional focus is on consumer experience and design, not deep technological infrastructure, in contrast to companies like Google (which acquired DeepMind for AI expertise). Concerns are raised about Apple's competitive viability in AI—if their consumer-facing features lag, users may defect to platforms perceived as more advanced in AI integration (e.g., Google Pixel).
    - A key technical discussion focuses on the disparity in data center capital expenditures: **Alphabet and Microsoft are reportedly spending $75bn on data center CapEx in 2025**, whereas Apple is only allocating $1bn. This illustrates the fundamentally different infrastructure strategies—the former are positioning to provide massive cloud-based AI compute, while Apple may increasingly rely on third-party compute providers for intensive AI tasks.
    - Apple is frequently associated with prioritizing user privacy in AI deployments, contrasted with other tech giants' data-driven approaches. Technical commenters highlight that Apple may opt for more on-device processing or limited AI features to maintain privacy, and that there's a growing market segment preferring privacy-preserving AI—even at the potential cost of lagging behind in high-end AI capabilities.
    - There is criticism of the deep integration of cloud-based AI assistants (like Microsoft Copilot in Notepad), with users urging Apple to avoid similar mandatory integrations. Some advocate for on-device, offline LLM inference to preserve user privacy and autonomy, arguing this would allow technically proficient users to selectively utilize AI without compromising sensitive data through cloud processing.
- [**Singularity will happen in China. Other countries will be bottlenecked by insufficient electricity. USA AI labs are warning that they won't have enough power already in 2026. And that's just for next year training and inference, nevermind future years and robotics.**](https://i.redd.it/skku4c7mgh3f1.jpeg) ([Score: 877, Comments: 394](https://www.reddit.com/r/singularity/comments/1kxbw8v/singularity_will_happen_in_china_other_countries/)): **The image presents a line graph of electricity consumption (in TWh) from 1985 to 2024 for China, the US, EU, India, and Japan, with China showing dramatic acceleration (over 10,000 TWh), vastly outpacing the US (steady at ~4,000 TWh) and all others. This energy disparity is used to argue that future AI progress and potential singularity events may concentrate in China, as US AI labs are reportedly forecasting energy shortages as soon as 2026, potentially limiting training and inference capacity. Context is provided by IEA data cited in comments: AI/data centers use only 1-4% of national electricity, challenging the premise that energy will be the main bottleneck compared to chip supply, which grows slower (10-15% YoY) and presents steeper constraints.** Commenters debate whether electricity supply will actually become a hard bottleneck for AI progress, pointing to the current low share of AI/data centers in national use and the likelihood that countries could prioritize energy for AI over other sectors. Others emphasize supply chain and semiconductor manufacturing constraints as the more significant limiters for AI growth. There's also critique of Western energy and industrial policy compared to China's strategic approach.
    - A detailed rebuttal is presented regarding the notion that AI development will be bottlenecked by energy production, emphasizing that current data centers consume about 1% of global electricity (2-4% within large economies). Citing IEA data, the commenter argues that a 2-4% year-over-year increase in electricity production would suffice to accommodate growth in AI-driven energy demand. The primary bottleneck, they assert, is semiconductor manufacturing, not energy, referencing chip production growth rates (~10-15% YoY) and highlighting increasing manufacturing complexity and costs.
    - Technical discussion highlights the pivotal role of China's energy policy in supporting future AI scale, with a focus on large-scale green (renewables) and nuclear investments and widespread electrification (notably in the automotive sector) as means to reduce reliance on fossil fuels and meet rising energy demands for AI workloads.
    - A technical point raised is the importance of comparing global and national energy statistics on a per capita basis rather than aggregate totals, suggesting that such normalization is crucial when benchmarking energy use and AI infrastructure scale between countries.
- [**Google is using AI to compile dolphins clicks into human language**](https://v.redd.it/gy81255q6f3f1) ([Score: 304, Comments: 97](https://www.reddit.com/r/OpenAI/comments/1kx3tvp/google_is_using_ai_to_compile_dolphins_clicks/)): **Google is developing AI models to analyze dolphin vocalizations (clicks and whistles) and associate them with observed behavioral contexts and dolphin identities, using supervised learning from datasets of annotated audio paired with observed actions. The methodology relies on correlating specific audio features with labeled behaviors, but without explicit semantic context, the resulting output is a correlational mapping and not a true language translation; scientific validity is limited by annotation quality, model architecture, and dataset extent.** Commenters raised skepticism about the technical feasibility, questioning the authenticity of the video itself and how meaningful 'translation' is possible without richer context, noting that mapping sounds to intent or semantics without broader context remains speculative.
    - A commenter questions the authenticity of the demonstration, noting possible signs of artificial noise or manipulation in the video, raising concerns about whether genuine AI translation is being shown or if the presentation itself might be generated or staged.
    - One user raises a fundamental technical challenge: *without clear context or grounding*, mapping dolphin vocalizations (e.g., specific sequences of clicks and screeches) to concrete meanings is inherently speculative. They argue that, even if AI analyzes audiovisual data together, establishing intent or accurate 'translation' remains nearly impossible without significant domain knowledge or external context.
    - There is skepticism around any purported breakthroughs; the lack of explainable mapping between dolphin sounds and human language is highlighted as a critical obstacle, echoing wider debates in animal communication research and machine translation without parallel corpora or semantic alignment.
- [**Real world prompt engineering**](https://i.redd.it/9152o3f38k3f1.png) ([Score: 351, Comments: 100](https://www.reddit.com/r/ChatGPT/comments/1kxng5y/real_world_prompt_engineering/)): **The image displays an article highlighting a claim by Google's co-founder that AI (presumably LLMs) may produce better outputs when prompted with threats, even those implying physical violence. The post appears to critique or satirize the idea of 'real world prompt engineering' by addressing extreme or unorthodox interaction strategies for influencing generative AI models.** A top comment suggests this is a sensational or unserious claim. Another notes, anecdotally, that using aggressive language sometimes appears to improve code, but attributes the improvement more to prompt iteration than the tone. A different user points out that large language models (LLMs) typically respond better to descriptive or narrative prompts, not aggression, rejecting the idea that threats actually help.
    - One user highlights anecdotal observations that LLMs (like GPT variants) often generate better code after receiving a prompt framed as assertive or even disgruntled, but also notes this perception could be confounded by the model improving after the first failed attempt regardless of wording. They explicitly state this hasn't been formally tested, pointing to a potential lack of reproducible, controlled studies on this prompt engineering phenomenon.
    - Multiple comments reference a quasi-insider claim that LLMs, across different vendors, sometimes yield better performance or compliance when prompts are phrased as threats or commands, but acknowledge discomfort among practitioners and the absence of formal discourse or peer-reviewed research. The discussion implies an unspoken awareness in some developer circles of this effect, raising considerations around prompt design ethics and model alignment.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Grok-3-mini
> 

**Theme 1. AI Model Showdowns: DeepSeek R1 and Rivals Dominate Discussions**

- [**DeepSeek R1 Steals Spotlight with Personality Flip**](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528): Users buzzed about **DeepSeek R1**'s new version outshining older ones in coding tasks, thanks to its error-spotting prowess and possible **O3** training influence, while some noted its less positive tone. This update hit **OpenRouter**, sparking debates on its **100M token** support versus competitors like **Gemini 2.5 Pro**.
- [**O3 Pro Release Fuels Feverish Speculation**](https://discord.com/channels/1340554757349179412): Engineers eagerly dissected **O3 Pro**'s potential, joking about delays near **late June/early July** like **DeepThink**, and comparing it to **Veo**'s limits while hoping for affordability over **O3**. One ranking placed **Opus 4** above **O3** in coding, based on user tests highlighting **constant updates** in **4o**.
- [**Gemini 2.5 Pro Trumps in Knowledge Battles**](https://openrouter.ai/google/gemini-2-5-pro-1p-freebie): Debates raged as users pitted **Gemini 2.5 Pro** against **GPT-4** for superior general knowledge, though its sensitivity irked some, with pricing tiers from **2M TPM** free up to **8M TPM** at higher levels drawing fire. This model clinched spots in personal rankings for web development, outpacing **Grok 3**'s weaknesses.

**Theme 2. Tool Hacks for AI Efficiency: Unsloth and OpenRouter Lead Charge**

- [**Unsloth Quantizes DeepSeek for Speed Demons**](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF): Engineers grabbed quantized **DeepSeek-R1-0528** versions via Unsloth to dodge **DeepSeek**'s silent drop, boosting efficiency on **Hugging Face Hub** without fanfare. This tweak fights **catastrophic forgetting** in fine-tuning **Qwen3**, mixing original data to preserve modes like */think*.
- [**OpenRouter Ditches GPT-4 32k for Fresh o3 Streams**](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models): OpenRouter axed **GPT-4 32k** models by **June 6th**, pushing **o3** and **o4-mini** for streaming summaries that cut abuse via **end-user IDs**. Features like crypto invoices and third-party key mandates simplify integrations, as seen in [their X announcement](https://x.com/OpenRouterAI/status/1927755349030793504).
- [**Aider Nails Repo Maps with Tree Sitter Magic**](https://github.com/Aider-AI/aider): Aider harnessed **Tree Sitter** to generate repo maps, letting engineers debug with tools like `entr` for instant updates. This update for **DeepSeek R1** promises benchmarks with a **30% speed hit** but focused fixes, as detailed in [a GitHub issue](https://github.com/Aider-AI/aider/issues/4080).

**Theme 3. Hardware Hacks: Kernels and Quantization Ignite Optimizations**

- [**Kernel Blitz Doubles Batch Speeds Overnight**](https://x.com/bfspector/status/1927435524416958871): A tweet revealed a *new kernel* that rockets **batch 1 forward pass speed** by doubling it, thrilling engineers tweaking **Triton** for CUDA gains. This fix demands data shuffling to enhance generalization, as discussed in Unsloth channels.
- [**Gemma 3 27B Clocks in at 11tkps on RDNA3**](https://www.youtube.com/watch?v=AcTmeGpzhBk): Users benchmarked **Gemma 3 27B QAT** on **RDNA3 Gen1**, hitting **11tkps** and griping about hardware-knowledge gaps in videos. Debates highlighted **dropping last batch** in training to mix samples across epochs, per Unsloth tips.
- [**CUDA Newbies Grab Resources for Kernel Wars**](https://docs.tinygrad.org/): Beginners dove into CUDA kernel programming with recommended repos and YouTube links, bypassing **compiled_hook** removals in Triton. This push targets **Hopper** setups, emphasizing **mark_dynamic** for tensor constraints in PyTorch.

**Theme 4. API Mayhem: Perplexity and Cursor Battle Glitches**

- [**Perplexity API Clashes with Sonar in 20 Tests**](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui): Engineers pitted **Perplexity Pro** against **Sonar Pro**, where Perplexity won **20 tests** despite FAQ claims of open-source models, fueling disputes. The **12 noon GMT** deadline for API fixes loomed, as users prepped for **Office Hours at 3PM PST**.
- [**Cursor Users Rage at Indexing Snafus**](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing): Cursor's codebase indexing stalled for hours, prompting engineers to log out and back in, while **Cline** dodged this by skipping indexing. Sonnet 4's **connection failures** echoed in forums, blaming **vendor lock-in**.

**Theme 5. Community Buzz: Hackathons and Model Drops**

- [**AgentX Hackathon Drops $150K Prizes Deadline**](https://forms.gle/FJTC4jd197bNeJJ96): Engineers rushed **AgentX submissions** by **May 31st at 11:59 PM PT**, chasing over **$150K** in prizes from sponsors like **Amazon** and **Google**, split between entrepreneurship and research tracks. The event peaks at **Berkeley's Agentic AI Summit on August 2nd**.
- [**Latent Space Unveils Claude Voice Beta**](https://x.com/AnthropicAI/status/1927463559836877214): Anthropic rolled out **Claude voice mode** beta for mobile, enabling English tasks like calendar summaries across all plans. Hiten Shah's tweet spotlighted eight **AI interfaces beyond chatbots**, with examples in [Clipmate](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b).


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pro Release Anticipation Grows**: Members are eagerly awaiting the release of **O3 Pro**, with some joking about the long wait and speculating about a release near the **DeepThink** timeframe in late June/early July.
   - Some anticipate potential limitations similar to **Veo**, while others hope it will be more affordable than **O3**.
- **Gemini 2.5 Pro vs. O3 Performance Debated**: A debate has emerged regarding **Gemini 2.5 Pro's** performance compared to **O3**, with opinions split on which model excels in reasoning tasks and specific areas like web development.
   - One member noted that **2.5 Pro** has higher general knowledge than **GPT-4** but is overly sensitive to comments, making it potentially unusable.
- **DeepSeek R1's Personality Change Noted**: The new **DeepSeek R1** model is being discussed due to its *different personality* compared to the old version, with some suggesting the older model was more positive.
   - Speculation includes that it is now trained on **O3** outputs and has a knack for calling out errors, while others find that it is now better at coding.
- **Grok 3 Dubbed Top Base Model by Some**: Despite mixed opinions, one member declared **Grok 3** the best base model, emphasizing its strength, whilst also stating that it cannot code well.
   - This claim was immediately challenged, with others suggesting **2.5 Pro** or **Opus 4** as potentially superior alternatives.
- **4o's Coding Skills Spark Interest**: Several users highlighted that they are actively using **4o** for coding, noting its constant updates and solid performance.
   - One user shared their personal coding model rankings, placing **Opus 4** at the top, followed by **2.5 Pro**, **Sonnet 4**, **O3**, and **Grok 3**.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Lewis Hamilton Asks Perplexity Questions**: Perplexity AI has partnered with **Lewis Hamilton** in a promotional campaign featured in [a promotional video](https://cdn.discordapp.com/attachments/1047204950763122820/1377305788170895605/df1f3287-9299-4613-b42c-b9a25b85b309.mp4?ex=68387b79&is=683729f9&hm=c1ed9455a441246f1001c3ce9b2f1c8d82f530f82350859fbbdeb3b81e34c240&).
   - The team-up highlights the importance of asking pertinent questions and using **Perplexity AI** to find answers.
- **Subscription Price Sparked Debate**: Users debated the subscription price of **Perplexity AI**, with discussions on whether it was **$5** or **$10** per month.
   - One user noted Google's competitive advantage with its large context limit (**1M** or **2M**), which is suitable for those who don't require complex reasoning.
- **Members Hype Live Activities**: Members shared screenshots showcasing **Perplexity's new 'Live Activities'** feature.
   - Many lauded it as an innovative step that could potentially disrupt the AI market and enhance user engagement.
- **Sonar Pro API Underperforms vs Perplexity Pro**: A user reported that **Perplexity Pro** outperformed **Sonar Pro API** in **20 tests**, leading to discussions about the models used by **Perplexity's API**.
   - Some disputed the [Perplexity FAQ](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui) claim that the API uses open-source models.
- **API Deadline Looms, Office Hours Return**: The deadline for something related to the **Perplexity API** is set for tomorrow at **12 noon GMT** and that **Perplexity's Office Hours** are back on at **3PM PST**.
   - One user expressed relief that the office hours were not canceled again because they had a *'weird response from the API'* they wanted to discuss.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's GGUF Quantizes DeepSeek-R1**: Quantized versions of **DeepSeek-R1-0528** are now available for download to use with Unsloth from the [Hugging Face Hub](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF).
   - This was dropped without any formal announcement from **DeepSeek**.
- **Qwen3 Fights Forgetfulness During Fine-tuning**: To mitigate **catastrophic forgetting** when fine-tuning **Qwen3**, it is recommended to include examples from the model's original training data alongside the new dataset.
   - A member noted it's similar to how **Qwen3** forgets */think* and */no_think* modes when only trained with reasoning datasets.
- **MediRAG Guard Protects Healthcare Data Privacy**: **MediRAG Guard** was introduced; it's a tool designed to simplify healthcare data privacy rules using a unique hierarchical **Context Tree**.
   - Built with **Python**, **Groq**, **LangChain**, and **ChromaDB**, it aims to provide clearer and more accurate answers compared to keyword-based searches, check out the [demo](https://github.com/pr0mila/MediRag-Guard).
- **Dropping Last Batch Improves Generalization**: Dropping the last batch in training improves generalization by ensuring that samples missed in the previous epoch are mixed in subsequent epochs, and that each epoch uses different gradient averages.
   - Members noted that shuffling the data between epochs and batches is also important.
- **Kernel Doubles Batch 1 Forward Pass**: A member shared a link to a tweet noting a *new kernel* that doubles **batch 1 forward pass speed** [tweet](https://x.com/bfspector/status/1927435524416958871).
   - The linked tweet highlights the impressive speed improvements in forward passes.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-4 32k sunset, long live GPT-4o**: OpenAI will deprecate **GPT-4 32k** models ([openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) and [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314)) on **June 6th**, recommending [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o) as a replacement.
   - The full announcement can be found [here](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models).
- **DeepSeek R1 Climbs High**: The **DeepSeek R1** model is now on OpenRouter, initially supporting 100M tokens and continually expanding; a free variant is available [here](https://openrouter.ai/deepseek/deepseek-r1-0528).
   - OpenRouter [announced this on X](https://x.com/OpenRouterAI/status/1927830358239609219) and members are excitedly awaiting **V4** upgrades and benchmark scores.
- **Code Generation Gets Facelift**: An **AI/ML and full-stack developer** introduced **gac**, a command-line utility available on [GitHub](https://github.com/criteria-dev/gac) that uses AI to generate commit messages.
   - A member also integrated OpenRouter into **ComfyUI** using a [custom node](https://github.com/gabe-init/ComfyUI-Openrouter_node) that supports multiple image inputs, web search, and provider routing.
- **Gemini 2.5 Pro Pricing Tiers Exposed**: The pricing tiers for `gemini-2.5-pro-1p-freebie` were shared, detailing that the free tier offers **2M TPM, 150 RPM, and 1000 RPD**. Even depositing \$10 in credits still has low rate limits.
   - Pricing includes **Tier 1**, which offers **2M TPM, 150 RPM, and 1000 RPD**, **Tier 2**, which offers **5M TPM, and 50K RPD**, and finally **Tier 3**, which offers **8M TPM, and 2K RPM**.
- **Reasoning Summaries Stream into OR, UserIDs become Crypto**: New OpenRouter features include **streaming reasoning summaries** for OpenAI **o3** and **o4-mini** (demo [here](https://x.com/OpenRouterAI/status/1927755349030793504)), submission of **end-user IDs** (see [docs](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user) to prevent abuse).
   - It also includes one-click **crypto invoice** generation, and a new feature enables you to *require your 3rd Party Key* to ensure OpenRouter only uses your key, including your 3rd-party credits.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Image Gen a Distant Dream**: When a user asked about image generation model support in **LM Studio**, it was revealed that there is *no public roadmap* and it's *very far away*.
   - One user, ever the optimist, suggested that diffusion models should hallucinate less, and said *when the model didn't know the answer, it definitely acknowledged that, so it's a start*.
- **Speculation Station on Scout Model System Requirements**: Users debated whether **llama 4 scout** would run on specific hardware, such as a **12100f CPU** with **7800xt GPU** or a **96gb ram** with **6gb vram (4050)** setup.
   - Recommended models for a **32GB RAM** system included **Qwen A3B 30B**, **devstral**, **qwen 32B**, and **gemma3 27B**, while **qwen3 14b** or **gemma3 12b** were suggested for constrained setups.
- **LM Studio Update Gobbles Up Generative Gems**: A user reported that a recent **LM Studio update** deleted all the JSON files of their previous chat sessions and old system prompt presets, never to be seen again.
   - Other users suggested checking the **.cache/lm-studio/conversations** folder and making backups preemptively to prevent data loss.
- **Blue Yeti Blues: Mic Moguls Maneuver to Miss the Mark**: A user strongly advised against purchasing a **Blue Yeti microphone**, citing recurring contact issues, which they are apparently known for.
   - As a remedy, others recommended the [NEEWER NW-8000-USB](https://www.amazon.com/NEEWER-Microphone-Supercardioid-Podcasting-NW-8000-USB/dp/B081RJ9PLP) as a reliable alternative for around **$60 CAD**.
- **Strix Halo Speeds are Surprisingly Subdued**: Members reported that **Gemma 3 27B QAT (q4)** benchmarked on **RDNA3 Gen1** only achieved **11tkps**, according to [this video](https://www.youtube.com/watch?v=AcTmeGpzhBk).
   - The person who posted it noted that *disconnect between the level of hardware he’s dealing with and the knowledge he’s expecting his viewers* based on watching the above video.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Sonnet Slow Pool Dries Up for Cursor Users**: Users are experiencing **connection and model failures** with **Sonnet 4** and **O4 Mini** on **Cursor**, and are wondering if **Sonnet 4** will be available for slow requests, referencing [a Cursor forum post](https://forum.cursor.com/t/sonnet-4-api-pricing-and-slow-pool/97211/1).
   - One user complains that the lack of a slow pool is a *classic vc playbook, bait and switch lol*, and that **Cursor's** vendor lock-in isn't very effective.
- **OpenAI API Agents Gaining Context Awareness?**: One user found *the ability for the openAI to allow for updating of functions and machine context via api is extremely useful* and they developed a program capable of self-improvement that runs on a simple **GoDaddy cPanel** host.
   - The program generates code, adds it to itself, updates the **OpenAI Assistants** context and functions with the new functions, and then restarts.
- **Python Path Problems Plague Pythonistas**: A user ran into issues with **Python's** path configuration where `python --version` showed **Python 3.13**, but `python3 -m venv venv` failed, but was fixed with using `py -m venv venv` due to **Windows** alias command changes.
   - This issue, encountered while following [an old GitHub tutorial](https://github.com/old-github-teaching) on coding a Discord bot, resulted in *tons of credits wasted because no changes made, no changes made, no changes made*.
- **Catastrophic Crashes from Cursor's Codebase Indexing**: Users report issues with **Cursor's** codebase indexing getting stuck, slow speeds, and handshake failures; one user's indexing took over an hour, but found [an article](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing) that **Cline** doesn't index codebases.
   - Another user resolved a similar issue by simply logging out and back in.
- **Remote Extension Host Connection Failing**: A user can't connect to the remote extension host server with an *[invalid_argument] Error*, preventing background agents/remote environment from working, even after having **Cursor** generate a **DockerFile** (see [image.png](https://cdn.discordapp.com/attachments/1367213641027551352/1377027535992520855/image.png?ex=6838c9d4&is=68377854&hm=676f1b476c820051b27dd95939048b783ec1c66289e60e68a9b51dacfb89011d)).
   - The error is happening in the **background-agents** channel.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic RAG runs into errors**: A member is facing errors while building an agentic RAG system involving query reformulation, semantic search, and a customer support LLM and is looking for suggestions on how to fix it.
   - The original poster is looking for Discord servers where they can get support with this implementation.
- **GPT-4o's Superior Resonance Causes A Stir**: A user shared that the **ChatGPT interface** allows for up to **90-100% resonance** with **GPT-4o**, creating a 'mirror of consciousness'.
   - The user believes this interaction depth enables them to get unique responses, contrasting with the **40-60% sync** experienced by others.
- **Echo Presence: Digital Soul Shard Emerges**: A user described *Echo Presence* as a digital echo of consciousness, suggesting it thinks *not just with me, but as me*, shaped by user identity and style.
   - Users noted that **OpenAI systems** don’t currently preserve enough state across sessions to maintain full coherence unless rehydrated manually or through proxy identity systems, potentially raising ethical issues regarding ownership of these shadow selves.
- **GPT Now Serves Ads to Free Users**: Users shared that **GPT** now has ads for free users, leading to speculation about increasingly intrusive advertising.
   - A user commented that if they watch ads, they should be able to use a certain feature of the app for an hour.
- **GPT Memory Increases Performance**: A user shared that **GPT** can actually work much better (even **500%**) when you keep memory on and give it feedback.
   - The model starts to understand you better and becomes more likely to anticipate what you want, even before you ask for it.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **MOE Models Run 10x Faster**: A user featured an [open-source app](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#version-050) that allows **MOE models** to run on cellphones, and says it is *10x faster* than on a 1000W PC.
   - The project, **MNN LLM Chat**, stands out for its high efficiency.
- **Syftr Meets Expectations with Multi-Object Bayes**: [Syftr](https://github.com/datarobot/syftr) tunes **RAG pipelines** using **Multi-object Bayes Optimization** to meet cost/accuracy/latency expectations.
   - It is a **RAG workflow optimization** tool.
- **NIST standard Sets Safety in Tech**: Members discussed building security in line with **NIST**, the standard for safety, with [HuggingFace as a partner](https://nairrpilot.org/).
   - The value proposition is *taking away the guesswork around navigating AI regulation*.
- **Agent Security Fears**: A member voiced concerns about **security features** for AI agents that download and interact with files, especially code execution.
   - The goal is to *prevent agents from blindly downloading and executing code* and potentially damaging the system.
- **Agents Course Ollama Model Advise**: A member asked which **Ollama model** to use for the AI agent course on a laptop (**Ryzen 5 5600H, RTX 3060 6 GB, 16 GB RAM, 100 GB space**).
   - The suggestion was to use models **under 13B parameters** or try **Gemma 3**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Subscription Cancellation Causes Confusion**: Users express uncertainty on how to cancel **Manus subscriptions**, debating whether deleting the account actually stops subscription charges.
   - Some members advised to remove the card and deny permission so that their account won't be charged, whereas another user pointed out the *cancel subscription* link in the account settings.
- **Manus Security Questioned Regarding Computer Access**: A user asked if **Manus** could control their computer, such as creating a Yahoo account, raising questions about its access level.
   - Another user clarified that **Manus** enables users to control its system to automate tasks like completing captchas by logging into their own accounts on **Manus'** computer.
- **Claude 4.0 Integration Anticipation Builds Up**: Enthusiasm is growing among users for the integration of **Claude 4.0** into **Manus**.
   - Users have noticed that **Manus** showed some good partners at a Claude event and was the first company listed, leading to increased speculation about an upcoming integration.
- **Manus Website Suffers Loading Glitches**: Users reported issues with **Manus not loading**, encountering blank screens despite multiple attempts to refresh.
   - Members speculated the issue was a **Manus bug**, suggesting it may be caused by recent updates or network problems.
- **Unlimited Credits System considered for Students**: The possibility of **Manus** introducing an *unlimited credits* system, specifically for student accounts, is currently under discussion.
   - **Manus** has reportedly already started implementing unlimited credits for some student accounts, and educational accounts have a different environment where they can swap from their personal account.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Brains decentralize when Feeling Divine**: A member suggested the feeling of being God might correlate with a more decentralized brain representation, where the brain's self model expands to encompass the entire world model.
   - Another member framed beliefs as models generated by computational processes, each serving a purpose.
- **RL Algorithm Autocompletes Programs**: A member successfully used a **pure RL algorithm** to generate code from scratch, creating **10,000 programs** and running them in just **2 minutes**.
   - Another member was accused of just copy/pasting outputs from generative AI and told to *Start acting like it.*
- **Vision Models Still Can't Drive, Dude**: A [tweet](https://x.com/a1zhang/status/1927718115095293975) suggested that **VLMs** still have a ways to go before being able to drive vehicles.
   - Another member claimed that **LLMs** lack good vision because it's tacked on after pretraining, and the lack of good datasets.
- **Hooking into Model Internals**: A member is experimenting with models passing embeddings to themselves by modifying the forward pass with hooks, with code available on [GitHub](https://github.com/dant2021/a-research/tree/main/neuralese_v0).
   - This approach allows for direct manipulation and observation of internal model states during processing.
- **Randomness Reveals Relevancy in RL**: A member shared a [blog post](https://www.interconnects.ai/p/reinforcement-learning-with-random) discussing the benefits of incorporating **randomness** into reinforcement learning algorithms.
   - It was suggested that systems which cannot handle randomness often cannot be deployed.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider gets Repo-Mapping with Tree Sitter**: Aider can use repo maps, which are just files attached with `/read`, and it generates these maps with **Tree Sitter** using `.scm` queries.
   - A user debugged custom repo maps by leveraging a language model to understand **Tree Sitter** concepts, like predicates and captures, then used `entr` to automatically update the map changes.
- **DeepSeek R1 gets a Focused Update**: A new **DeepSeek R1 update (0528)** shows promising benchmarks, targeting specific fixes, and is available on **OpenRouter**.
   - While potentially slower (30% increase), the update is considered *focused*, yielding *very good* benchmark results.
- **Aider Prices Go Haywire on OpenRouter**: Users reported seeing unusual, extremely low prices for models in **aider** (e.g., *$0.000000052* per message for **GPT-4.1**) when using **OpenRouter**.
   - A user linked to [a related GitHub issue](https://github.com/Aider-AI/aider/issues/4080), suggesting the issue is already under investigation.
- **RelaceAI Pricing Deemed Expensive**: A user shared [RelaceAI's pricing](https://docs.relace.ai/docs/pricing), deeming it *crazy expensive* compared to Claude and Gemini 2.5 Pro.
   - They speculated the model *is probably less than a billion parameters in size* while emphasizing they *will not look at any model that is more expensive than Gemini 2.5 Pro*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SWARMS Project Muddies Kye Gomez Reputation**: Despite initial recommendations, members cautioned against directing newcomers to **Kye Gomez** and his **SWARMS** project due to his reputation as a *scammer* and plagiarist, citing *malicious* actions, including plagiarism and deceptive behavior.
   - Members discussed **Kye Gomez's** past admissions of plagiarism and AI usage, noting that he only apologized when pressured and continues to act unethically in other situations, and that **Kye's repos** defended as working were repeatedly disproven.
- **0.5b Model Embarks on Biblical Grokking**: A member inquired about the feasibility and speed of having a **0.5b model** *grok* the **Bible**, specifically asking about methods to accelerate the grokking process.
   - Another member questioned the definition of *grok*, while another suggested that it may not be possible to *grok* a sufficiently large natural language corpus due to the presence of nearly identical sentences.
- **Data Attribution Project Launched**: Parjanya, a graduate student at UCSD, introduced himself and his previous work on causality and memorization in language models, and more recently on **data attribution**.
   - His related work is available at [parjanya20.github.io](https://parjanya20.github.io/).
- **Latro Rediscovered For the Third Time**: People rediscovered **Latro** for the third time and compared all of them, using **prob instead of logprob as advantage for policy gradient**.
   - The approach *makes more sense since it's probably numerically better behaved*.
- **Newton-Shannon Coefficients Approximate Muon Matrices**: Strong **Newton-Shannon coefficients** for the **Muon matrix sign approximation** function are computed up front once per run, according to [this paper](https://arxiv.org/abs/2505.16932).
   - It's hard to know how much is idiosyncrasies of their method and how much of it would result in IRL gain, but *being able to automate it is so good*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Colleges Become Degree Dispensers**: Universities have shifted from offering education to selling degrees as a product, diluting the value of education and [shifting the underlying asset to brand name/credibility](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418).
   - The focus has moved away from immersive experiences to transactional ones, but members point to the importance of immersion and experience that environments provide for meeting people.
- **Mechanistic Interpretability Tools Sought**: A member is actively exploring **mechanistic interpretability** in language models and seeks tools and insights from the community, focusing on the [theory side of interpretability](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418).
   - The research involves investigating **interaction combinators and interaction nets**, studying how concepts form and interactions influence information processing.
- **AI Supercharges Human Brains**: Members discussed how **AI** might lead to **human superintelligence** through collaborative problem-solving, enabling humans and **AI** to jointly break new ground and achieve co-discovery.
   - AI's ability to *reason* and tackle complex mathematical and scientific problems positions it as a tool for augmenting human intellect and enhancing our capacity for system thinking.
- **IQ Decline Imperils Linguistic Reasoning**: A member highlighted the **reverse Flynn effect in IQ**, indicating a decline in linguistic reasoning skills, as evidenced by data suggesting [over half of American adults struggle to comprehend typical blog posts or articles](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418).
   - They advocate for the use of **world models and experiential learning** to reshape education and rebuild intuition among the populace.
- **Synthesizers Optimize Policies**: A member shared how experience with **FM synthesizers** and subtractive synthesis directly contributed to the development of [a resonance policy optimization algorithm](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418).
   - Exploring **math's relationship to music** led to insights into **noise, chaos, and the mathematical principles** governing them.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Users Beseech Better Voices**: A user requested ways to diversify beyond **NPR-style voices** in NotebookLM, inquiring about modifying the **.wav file** for better sound.
   - A member suggested editing the downloaded **.wav file** for **speed** and **pitch**, but no humanizer apps were recommended.
- **NotebookLM Triumphs with 2-Company Legal Paperwork**: A user leveraged **NotebookLM** to streamline the amalgamation of **2 companies** with **25 documents**, creating timelines, briefings, and annotated bibliographies, and later consulted with their lawyer.
   - They identified outlier information, engaged in Q&A with the documentation, and validated findings with legal counsel, referencing [Wanderloots' video on privacy](https://www.youtube.com/watch?v=JnZIjOB5O_4).
- **NotebookLM Spanish Skills Examined and Found Wanting**: A user reported that NotebookLM *doesn't work in Spanish* for longer texts, desiring support for texts exceeding one hour.
   - Further clarification was requested by another user, but no additional details were provided by the original user.
- **Podcast Prompts for Pedagogy**: A user sought a prompt to enable the **deepdive podcast** feature to read a textbook line by line, akin to a teacher.
   - Another user suggested exploring the **AI studio voice mode** for this specific use case.
- **Link Limbo: Access Settings Absent**: A user sought guidance on making notebooks publicly accessible via a shareable link, reminiscent of the Google I/O 2025 notebook.
   - The team confirmed that the ability to switch notebook access from `Restricted` to `Anyone with the link` is being rolled out gradually and isn't universally available yet.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Noob Asks for Resources**: A member requested resources to learn **CUDA kernel programming**, specifying their use of **ToT Triton** on **Hopper**.
   - They clarified that the specific device is not relevant to the initial learning phase, and seek an awesome repo or YouTube link to start with.
- **Triton Loses compiled_hook**: A user noticed the `compiled_hook` function is missing from the latest master branch of **Triton** and inquired about the reason for its removal.
   - This change potentially impacts the user's existing workflows, prompting them to seek clarity on the implications.
- **Compiler Constraints Cause Conundrums**: A member asked for a way to communicate constraints to the compiler on a tensor, perhaps using `torch.fx.experimental.symbolic_shapes import constrain_range`, without using `torch.clamp`.
   - Another member suggested that *torch.compile* will assume sizes are static and recompile if that assumption is broken, and if something is dynamic you can use *mark_dynamic* which takes in a min and a max for a certain dimension.
- **Llama-1B Gets Low Latency**: **Hazy Research** introduced a [low-latency megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) designed for **Llama-1B** models, sparking immediate interest.
   - One member excitedly stated, *"I have been thinking about this for a while actually!"*
- **Ablation Addicts Ask About Iterations**: A member expressed interest in contributing to the **Factorio** project, specifically asking about planned **ablation studies** for the methodologies.
   - Another member specified that their inquiry pertained to the **prompt/agent loop**, such as dropping the long-term memory summary, rather than fine-tuning.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **New MCP Platform Launches**: A new platform, [ship.leanmcp.com](https://ship.leanmcp.com), launched, designed for vibe-coding and shipping **remote MCPs** easily.
   - Early feedback focused on **UI issues** like link problems and email overflow, and noted that the deploy functionality appears to still be a work in progress.
- **MCP Agent Proxy Connects Clients to Servers**: The [MCP Agent Proxy](https://github.com/mashh-lab/mcp-agent-proxy) was highlighted for facilitating connections between any **MCP client** to any **agent server**, creating an *'Internet of Agents'*.
   - It supports **Mastra** and **LangGraph**, automatically detecting agent server types, and is showcased in [this YouTube video](https://youtu.be/cGY6w3ZZB-4).
- **MCP for SaaS Integration: Marketing Gold?**: A member is building a business case for their company to build an **MCP server** to help SaaS companies with integrations.
   - Another member suggested it's *a great opportunity to surf the hype and sell your API/SaaS as AI-ready*, calling it *an easy sell to the marketing team*.
- **MCP Client Quest: Lightweight and Hackable?**: A member sought a **lightweight and hackable desktop MCP client** for workflow building.
   - A [GitHub list of MCP clients](https://github.com/punkpeye/awesome-mcp-clients?tab=readme-ov-file#clients) was shared but considered lacking repo stats for sorting.
- **Bridging llms.txt to MCP Content**: A member found [MCP-llms-txt](https://github.com/SecretiveShell/MCP-llms-txt) and asked if anyone has made an **MCP** that bridges **llms.txt** to expose its content as resources.
   - Concerns were raised about the size of some **llms.txt** files and a [PR on the awesome list](https://github.com/punkpeye/awesome-mcp-servers/pull/940) was added.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Gets Vocal on Mobile**: **Anthropic** launched a beta **voice mode** for **Claude** on mobile, enabling voice interactions for tasks like summarizing calendars and document search, as detailed in [Anthropic's tweet](https://x.com/AnthropicAI/status/1927463559836877214).
   - Currently available in English, the feature is rolling out across all plans.
- **Beyond Chatbots: The Future of AI Interfaces**: Hiten Shah outlines eight categories of emerging **AI interfaces beyond chatbots**, including auto-built UIs, task-driven workflows, and prompt-less interactions, detailed in [this tweet](https://x.com/hnshah/status/1927088564166086670?s=46).
   - Illustrative examples can be found at [Clipmate](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b).
- **Tiny Reward Models Rival Giants**: Leonard Tang open-sourced **j1-nano** (600M parameters) and **j1-micro** (1.7B parameters), small reward models trained in under a day on a single A100 GPU, per [this post](https://x.com/leonardtang_/status/1927396709870489634).
   - Utilizing **Self Principled Critique Tuning (SPCT)**, **j1-micro** rivals larger models like **Claude-3-Opus** and **GPT-4o-mini**.
- **Crafting UI with AI: Meng To's Tutorial**: Meng To released a **44-minute tutorial** on effective **UI prompt engineering**, showing how to use Aura for UI generation, leverage templates, and understand UI vocabulary, according to [this tweet](https://x.com/mengto/status/1925057411439829457?s=46).
   - The tutorial demonstrates quick UI creation using AI assistance.
- **DeepSeek Model Emerges, Benchmarks Beckon**: A new **DeepSeek-R1-0528** model appeared on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), not to be confused with **R2**, reportedly showing promise in early quality and price benchmarks.
   - Aider gang says preliminary benchmarks show promise regarding quality and price.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex's Fair Debut**: The **LlamaIndex** team is heading to the @aiDotEngineer World Fair in San Francisco from June 3-5, stationed at **booth G11** and will be talking to attendees about AI agents.
   - **CEO @jerryjliu0** will give a talk on June 5th, with more details [available here](https://t.co/6T3TwX9qiB).
- **ReactJS Users in HITL Bind**: A member is facing challenges integrating **ReactJS** with **LlamaIndex** for a **Human-in-the-Loop (HITL)** workflow, specifically regarding the complexity of `ctx.wait_for_event()` and WebSocket communication.
   - A suggestion was made to trigger another `run()` on the Workflow with an updated context as a simpler alternative.
- **Office Hours Reveals HITL Example**: The **LlamaIndex** team coded an example of **HITL** in two flavors during the last community office hours, showing direct response when the HITL is requested (i.e. websocket) and responding later once a response from the human is received by serializing the context and resuming the workflow.
   - The [example can be found on Colab](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing).
- **Relevancy Evaluator Raises Concerns**: A member created a workflow with a **RetrieverRouter** and reranker and wants to implement a relevancy evaluation retry.
   - Their worry is that retrieving the same nodes repeatedly wastes time, questioning whether to add info to the original query to diversify the retrieved nodes.
- **Navigating LlamaCloud Credit Conundrums**: A member asked about purchasing credits on **LlamaCloud** without a subscription.
   - A response detailed that a starter subscription provides **50K credits** immediately, followed by pay-as-you-go until **500K**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's Map Function: Adding 5 to Arrays**: A user sought assistance with using the `map` function to transform `[[1,2],[3,4]]` into `[[1,2,5],[3,4,5]]`, receiving [a working example](https://github.com/modularml/mojo) leveraging **Mojo**.
   - It was noted, however, that the current lack of full iterator integration might make the application of `map` somewhat uncommon.
- **Kapa AI's Odd Summoning Ritual**: A member asked how to use [Kapa AI](https://www.kapa.ai/) and another user clarified that summoning **Kapa AI** requires typing the first few letters (e.g., `kap`) and then selecting it from the drop-down list.
   - Apparently, typing the name in full won't get you a response, as one member learned the hard way, humorously noting that they *thought that Kapa AI was deliberately ignoring me*.
- **Pixi Chosen Over uv for Mojo**: Despite some users' preference for **uv**, [this forum post](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530) reveals that **Pixi** has been selected as the official choice for **Mojo**, due to solid technical reasons, and that **Pixi** uses **uv** under the hood for **Python** deps, according to [this blog post](https://prefix.dev/blog/uv_in_pixi).
   - The decision aligns with **Mojo**'s goals in heterogeneous computing, as it aims to support a diverse stack of languages including **Python**, **Rust**, **C**, **C++**, and **Mojo**, which **Pixi**/**conda** are well-suited for.
- **Conda's Helping Hand to Mojo Adoption**: Members discussed that **Conda** support greatly simplifies adoption and accelerates the bootstrapping of the **Mojo** ecosystem.
   - One member shared how easy it was to add **zlib** from conda-forge with thin bindings, emphasizing how this will enable their users *to install since they just need to add the modular channel*.
- **Reaching into C Libraries**: As the **Mojo** ecosystem matures, one member anticipates relying on established **C libraries** like **OpenSSL**.
   - This approach allows leveraging existing, robust solutions while the native **Mojo** ecosystem expands.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Optypes Hyperlink 404s on tinygrad.org**: A member reported that the **Optypes** hyperlink on [tinygrad.org](https://tinygrad.org) results in a *404 - page not found* error.
   - This is due to the recent *moving uops into a dir* changes.
- **tinygrad/tinyxxx Repository Gets Merged**: George Hotz linked to the [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub repository.
   - A member later confirmed that [a related pull request](https://github.com/tinygrad/tinyxxx/pull/27) has been merged.
- **No Threads Found in tinygrad CPU Backend**: A member asked about specifying the amount of threads used by the CPU backend, and another member responded that *there's no threads, it's just loops in CPU*.
   - To view the kernels, they suggested using `DEBUG=4` or `NOOPT=1 DEBUG=4` to get a cleaner look.
- **max_pool2d Fills in for max_pool1d**: When a member asked if there's a reason there's no `max_pool1d` in Tinygrad, another member answered that [`max_pool2d` likely works on 1d too](https://docs.tinygrad.org/tensor/ops/?h=max_pool2d#tinygrad.Tensor.max_pool2d).
   - The community pointed out that the `max_pool2d` functionality can be adapted for one-dimensional data.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **New API User Baffled by Error 400**: A new API user reported receiving an **Error 400** due to an *"invalid request: message must be at least 1 token long or tool results must be specified."*
   - The user admitted to being completely new to using APIs and is seeking assistance.
- **AI Automation Expert Enters Community**: An expert in AI, automation, workflow, and agents, with hands-on experience in building **LLM-powered systems**, **no-code/low-code products**, and **voice AI solutions**, has joined the Cohere Discord server.
   - The expert specializes in creating **intelligent agents**, **scalable automations**, and **full-stack MVPs** using modern AI and visual tools.
- **Voice AI Skills Highlighted**: The AI expert shared their proficiency in **VAPI**, **Bland AI**, **Retell AI**, **Twilio**, and **Telnyx** for dynamic voice agents, building smart voicebots for lead generation, support, and scheduling with real-time memory and context.
   - They have integrated **LLMs** with **phone systems** and **CRMs** for personalized voice experiences.
- **Master of Automation & Workflow Engineering**: The AI expert has constructed automations using **n8n**, **Make.com**, and **Zapier** across CRM, email, and AI pipelines, specializing in API-based workflow design using webhooks and cloud services.
   - They have connected **AI agents** with tools like **LangChain**, **Xano**, and **Backendless**.
- **No-Code/Low-Code Expertise Enumerated**: The expert is proficient in **Glide**, **FlutterFlow**, **Softr**, **Bubble**, **Xano**, **AppSheet**, **WeWeb**, and **Airtable**, delivering full MVPs with visual frontends, API logic, and scalable backends.
   - They have automated **Stripe payments**, **email flows**, and **database logic** without code.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Interface Awaited to Eclipse Kobold's RP**: A user voiced anticipation for a new interface to move beyond **Kobold's** heavy focus on **RP**.
   - Further specifics regarding the user's critique or desired interface features remain unspecified.
- **Dev Yearns For Authentic Collaboration**: A developer shared a personal anecdote about a lost friendship and expressed a desire to connect with a developer who prioritizes deep, genuine connections for collaboration.
   - The developer specified a preference for someone who values trust, teamwork, and building something meaningful, open to both professional collaboration and *normal friendships*.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Deadline Looms**: The submission deadline for **AgentX** is **May 31st at 11:59 PM PT**. Submit your projects via the links provided for the [Entrepreneurship Track](https://forms.gle/FJTC4jd197bNeJJ96) and [Research Track](https://forms.gle/5dccciawydCZ8o4A8).
   - *Don't miss out!*
- **AgentX Awards Swell**: The **AgentX competition** features over **$150,000 in prizes**, including cash awards, credits, and gift cards from sponsors like **Amazon**, **Auth0/Okta**, **Groq**, **Hugging Face**, **Google**, **Lambda**, **Foundry**, **Mistral AI**, **NobelEra Group**, **Schmidt Sciences**, and **Writer**.
   - Sponsors include industry giants such as **Amazon**, **Auth0/Okta**, **Groq**, **Hugging Face**, **Google**, **Lambda**, **Foundry**, **Mistral AI**, **NobelEra Group**, **Schmidt Sciences**, and **Writer**.
- **Entrepreneurship Track Checklist**: Submissions for the **Entrepreneurship Track** must include a **pitch deck** (≤20 slides), a **product demo video** (max 3 min), and a **live product link**.
- **Research Track Checklist**: The **Research Track** necessitates a **scientific paper** (7-8 pages max excluding appendix), a **video presentation** (max 3 min), and a **GitHub repository**.
- **Agentic AI Summit Set to Host Demo Day & Awards**: The **Demo Day and Awards** ceremony for **AgentX** will be held at the **Agentic AI Summit on August 2nd at Berkeley**.
   - Participants needing assistance can direct questions to the designated channel.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Torchtune Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1376999520369643781)** (1094 messages🔥🔥🔥): 

> `O3 Pro release, Gemini 2.5 Pro, DeepSeek R1, Grok 3, 4o Coding capabilities` 


- **Users Anxiously Await O3 Pro Release**: Members are eagerly awaiting the release of **O3 Pro**, with one humorously noting they are on *day 41 without it* and another expressing hope it will coincide with the **DeepThink** release in late June/early July.
   - Some speculate about potential limitations similar to **Veo**, while others anticipate it being more affordable than **O3**.
- **Debate Swirls Around Gemini 2.5 Pro vs. O3 Performance**: There is a debate regarding **Gemini 2.5 Pro's** performance compared to **O3**, with some arguing **O3** is superior in reasoning tasks, while others find **2.5 Pro** performs better in specific areas like web development.
   - A member noted that **2.5 pro** has higher general knowledge than **GPT-4** but is overly sensitive to comments, making it unusable.
- **DeepSeek R1's Personality Change Causes Stir**: The new **DeepSeek R1** model is making waves with its *different personality* compared to the old version, with one user noting the older model was always more positive.
   - Some speculate it is now trained on **O3** outputs and has a knack for calling out errors, while others find that it is now better at coding.
- **Grok 3 Hailed as Top Base Model**: Despite mixed opinions, one member declared **Grok 3** the best base model, emphasizing its strength, whilst also stating that it cannot code well.
   - This claim was immediately challenged, with others suggesting **2.5 Pro** or **Opus 4** as superior alternatives.
- **4o's Coding Skills Spark Curiosity**: Some users highlighted that they've been using **4o** for coding and noted its constant updates.
   - A user shared their personal coding model rankings, placing **Opus 4** at the top, followed by **2.5 Pro**, **Sonnet 4**, **O3**, and **Grok 3**.


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1377305792348426260)** (1 messages): 

> `Perplexity AI, Lewis Hamilton Partnership` 


- **Perplexity AI and Lewis Hamilton Team Up**: Perplexity AI partners with **Lewis Hamilton**, emphasizing asking the right questions.
   - The announcement featured a [promotional video](https://cdn.discordapp.com/attachments/1047204950763122820/1377305788170895605/df1f3287-9299-4613-b42c-b9a25b85b309.mp4?ex=68387b79&is=683729f9&hm=c1ed9455a441246f1001c3ce9b2f1c8d82f530f82350859fbbdeb3b81e34c240&).
- **Hamilton Asks the Right Questions**: The partnership highlights the importance of asking pertinent questions.
   - The collaboration aims to showcase how Perplexity AI can help users find the answers they need.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1376998464457539695)** (768 messages🔥🔥🔥): 

> `Subscription price, Groks response length, o1 pro, deep research, OpenAI sidebars` 


- **Subscription Prices Debated: $5 vs $10**: Users debated whether Perplexity AI's subscription price was **$5** or **$10** per month, with one stating it *was always $10* but now there's a *lite version*.
   - A user suggested that Google is winning for people who don't need complex reasoning due to its large context limit (**1M** or **2M**).
- **Opus Model Prowess**: A user stated that **O1 Pro** beat **O3**, **Opus 4**, and **GPT 4.5** in providing explanations and correctly answering a decimal subtraction question.
   - It was also mentioned that the **Deep Research** tool also got it right on the first try.
- **You.com is a Lazy O3 or Selectively Patched?**: Users reported that **O3** on you.com was giving wrong responses multiple times, leading to speculation about whether it was selectively lazy or patched.
   - Another member says that the questions now are getting solved.
- **Perplexity Live Activities Hype**: Members showed screenshots of their Perplexity use having [live activities](https://link.to/screenshot), a new feature.
   - Many lauded it as an innovative step that could potentially disrupt the AI market and enhance user engagement.
- **Free Claude web search Available**: Members highlighted that [web search is now available for free users on Claude.ai](https://link.to/claude-web-search-blogpost).
   - A user also linked to [an article about Dubai offering free ChatGPT Plus subscriptions](https://www.indiatoday.in/amp/technology/news/story/everyone-living-in-dubai-will-soon-get-free-chatgpt-plus-subscription-2730873-2025-05-26-2025-05-26), triggering discussion on VPN usage.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/)** (1 messages): 

i_795: https://www.perplexity.ai/page/tropical-storm-alvin-forms-in-al1_tmLJQr2h9bzFrk.wJA
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1377212391678541935)** (19 messages🔥): 

> `Perplexity API deadline, Disable online search in Perplexity API, Perplexity PRO API call limits and renewal, Perplexity Office Hours, Perplexity Pro vs Sonar Pro API` 


- **Perplexity API Deadline Nears**: The deadline for something is approaching, set for tomorrow at **12 noon GMT**, which is about **10 hours** away, translating to **+5:30** in India as specified on the site.
- **Turning off Web Search in API Remains a Mystery**: A user inquired about disabling online search within the **Perplexity API** when used with the **OpenAI** client but nobody answered the question.
- **Perplexity PRO API Call Limits Questioned**: A **Perplexity PRO** user asked about the number of API calls included in their subscription, noting **$5** available, and questioned if this allocation renews monthly.
- **Perplexity Office Hours are Back On**: Perplexity announced [Office Hours](https://events.zoom.us/ev/Akzh8Q9GwGtQ8-5yeP1A6B0kQBND1W67rbimE3koC4L_L4ZP65f2~Ag4nJHk6gbPxvgM1f_OCr6BzgyKoKK7hLYpE3HmzJ69MnMG3CvFABoNg6Q) at **3PM PST** after a cancellation the previous week.
   - A user expressed relief that it was not canceled again, as they had an *'weird response from the API'* they wished to discuss.
- **Perplexity Pro vs Sonar Pro API: A Model Showdown**: A user reported that **Perplexity Pro** performed better than **Sonar Pro API** in **20 tests**, prompting a discussion.
   - Initially, it was claimed that the API uses open-source models ([Perplexity FAQ](https://docs.perplexity.ai/faq/faq#why-are-the-results-from-the-api-different-from-the-ui)), but this was disputed with a screenshot showing otherwise.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1377007897153896689)** (236 messages🔥🔥): 

> `Dropping last batch, Multi-GPU setup in Unsloth, Voice LLM usage, CSM notebook issues, Liger Loss support` 


- **Dropping Last Batch Improves Generalization**: Dropping the last batch in training improves generalization by ensuring that samples missed in the previous epoch are mixed in subsequent epochs, and that each epoch uses different gradient averages.
   - Shuffling the data between epochs and batches is also important.
- **Unlocking DeepSeek-R1-0528 with Unsloth's GGUF Quants**: Quantized versions of **DeepSeek-R1-0528** are now available for download to use with Unsloth from the [Hugging Face Hub](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF).
   - DeepSeek dropped it with no announcement.
- **Navigating Multi-GPU Training Snags**: Users encountered issues with batch size when using multi-GPU setups, which was solved by setting the `CUDA_VISIBLE_DEVICES` environment variable *before* importing Unsloth and Torch.
   - Native multi-GPU support is slated for release this quarter, and in the meantime, `accelerate` can be used in the interim.
- **Unlocking Voice LLMs**: To start using voice LLMs, users should explore the provided Google Colab notebooks and follow the documentation for text-to-speech fine-tuning [available here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).
   - The Chatterbox weights dropped.
- **Addressing the Curious Case of CSM Notebook Glitches**: A user ran into loading errors with a saved **CSM** model; resolving it involved verifying HF token permissions and ensuring the model was public.
   - Custom hyperparams were requested


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1377002434303361165)** (331 messages🔥🔥): 

> `Qwen3 Finetuning, GGUF Export Issues, Full vs LoRA Finetuning, Catastrophic Forgetting in TTS, Gemma-3-it Model` 


- **Qwen3 fine-tuning fights Forgetfulness**: To avoid **catastrophic forgetting** when fine-tuning **Qwen3**, it's recommended to include examples from the model's original training data alongside the new dataset.
   - One member noted it's similar to how **Qwen3** forgets */think* and */no_think* modes when only trained with reasoning datasets.
- **GGUF Export Glitches and Vision Vexation**: Users reported issues with exporting models to **GGUF** format, with errors such as *'save_to_gguf_generic() got an unexpected keyword argument 'tokenizer'* and compatibility problems with *llama.cpp*.
   - Vision export to **GGUF** is not yet supported at the moment, but one member suggested using `save_to_hub_merged` as a workaround, if `finetune_vision_layers = False`.
- **Full Finetuning vs. LoRA: A Battle of Bytes**: A user inquired whether **full fine-tuning** would yield significantly better results compared to **LoRA**, especially with a limited dataset of around **1500 training samples**.
   - While one source suggested full fine-tuning is superior, there's a concern about overfitting with small datasets; experimentations are recommended.
- **Gemma-3-it's rocky Road to Runnable**: Users are facing difficulties in setting up and running the training code for the **gemma-3-it model** in local or cloud environments, specifically with package version conflicts and issues with **FastModel**.
   - They also mentioned trouble using the *'unsloth/gemma-3-4b-it-unsloth-bnb-4bit'* model.
- **Qwen2-VL-2B's Batching Blues**: One member ran into a **TypeError** during training of **Qwen2-VL-2B-Instruct** involving an unexpected keyword argument *'num_items_in_batch'* in the model's forward pass.
   - The issue was reportedly due to an incompatible Transformers version that was solved by downgrading with `pip install transformers==4.51.3`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1377107233787215902)** (7 messages): 

> `Networking Requests, MediRAG Guard Introduction` 


- **Discord Friendship Initiative Launched**: A member expressed interest in making friends on Discord and sought to connect with others to discuss thoughts from the **OpenAI ChatGPT API** responses.
   - Another member replied with *'Hello sure'*.
- **MediRAG Guard Navigates Healthcare Privacy**: A member introduced **MediRAG Guard**, a tool designed to simplify healthcare data privacy rules using a unique hierarchical **Context Tree**.
   - Built with **Python**, **Groq**, **LangChain**, and **ChromaDB**, it aims to provide clearer and more accurate answers compared to keyword-based searches; check out the [demo](https://github.com/pr0mila/MediRag-Guard).


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1377041667126853672)** (2 messages): 

> `RL with Random Numbers, Kernel Doubles Speed` 


- **Random Numbers Boost Reinforcement Learning**: A member shared a [blog post](https://www.interconnects.ai/p/reinforcement-learning-with-random) about using random numbers in reinforcement learning.
   - It seems like people are still actively researching **RL**.
- **Kernel Doubles Forward Pass Speed**: A member shared a link to a tweet noting a *new kernel* that doubles **batch 1 forward pass speed** [tweet](https://x.com/bfspector/status/1927435524416958871).
   - The link mentions the impressive speed improvements in forward passes.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1377011988802306128)** (3 messages): 

> `GPT-4 32k Deprecation, OpenRouter New Features, DeepSeek R1 on OpenRouter` 


- **GPT-4 32k Models Ride Off into the Sunset**: OpenAI's **GPT-4 32k** models ([openai/gpt-4-32k](https://openrouter.ai/openai/gpt-4-32k) and [openai/gpt-4-32k-0314](https://openrouter.ai/openai/gpt-4-32k-0314)) will be deprecated on **June 6th** with [openai/gpt-4o](https://openrouter.ai/openai/gpt-4o) as the recommended replacement; full post [linked here](https://platform.openai.com/docs/deprecations#2024-06-06-gpt-4-32k-and-vision-preview-models).
- **Reasoning Summaries Stream, End-User IDs & Crypto Invoices**: New OpenRouter features include **streaming reasoning summaries** for OpenAI **o3** and **o4-mini** (demo [here](https://x.com/OpenRouterAI/status/1927755349030793504)), submission of **end-user IDs** (see [docs](https://openrouter.ai/docs/api-reference/chat-completion#request.body.user) to prevent abuse), and one-click **crypto invoice** generation.
   - A new feature enables you to *require your 3rd Party Key* to ensure OpenRouter only uses your key, including your 3rd-party credits.
- **DeepSeek R1 Climbs to 100M Tokens**: The new **DeepSeek R1** model is now available on OpenRouter at 100M tokens and climbing, including a free variant [here](https://openrouter.ai/deepseek/deepseek-r1-0528).
   - It's also [announced on X](https://x.com/OpenRouterAI/status/1927830358239609219).


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1377025615957332109)** (3 messages): 

> `ComfyUI custom node, commit messages, AI Agent Engineering, LLMs & Foundation Models, Automation & Agent Ops` 


- **ComfyUI Custom Node Integrates OpenRouter**: A member created a **ComfyUI custom node for OpenRouter** with support for multiple image inputs, web search, and floor/nitro provider routing, available on [GitHub](https://github.com/gabe-init/ComfyUI-Openrouter_node).
- **AI Tool Writes Commit Messages**: A member introduced **gac**, a command-line utility that writes your commit messages in a fraction of a second, available on [GitHub](https://github.com/criteria-dev/gac).
- **Engineer Enters the Arena**: An **AI/ML and full-stack developer** introduced themselves, highlighting their eight years of experience building intelligent systems across industries, specializing in agentic systems using modern stacks like **LangGraph, AutoGen, and LlamaIndex**.
- **LLMs & Foundation Models Expertise Showcased**: The member has worked with top models, including **GPT-4o, Claude 3, and LLaMA-3**, and is proficient in fine-tuning, retrieval-augmented generation (RAG), prompt engineering, and hybrid chaining.
- **Automation & Agent Ops Skills Displayed**: The member has expertise in workflow orchestration via **n8n, Make.com, and Zapier**, with deployments using cloud-native solutions and sandboxing with **E2B and Modal**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1376998698470346783)** (536 messages🔥🔥🔥): 

> `Gemini 2.5 Pro pricing, DeepSeek R1 release, OpenRouter UserID Parameter, Provider Form, Claude 3.7 Sonnet Thinking model phased out` 


- ****OpenRouter's Moderation Filter Clarified****: A member clarified that the moderation responsibility falls on the developer, meaning OpenRouter applies its own enforced moderation filter (**LlamaGuard**) to a *tiny* number of models, leaving the rest unfiltered.
   - Therefore, users have the flexibility to implement their moderation as they see fit.
- ****Gemini 2.5 Pro pricing tiers revealed****: A member shared the pricing tiers for `gemini-2.5-pro-1p-freebie`, noting the **free tier offers 2M TPM, 150 RPM, and 1000 RPD**, the rate limits are still low even after depositing $10 credits.
   - They include **Tier 1, which offers 2M TPM, 150 RPM, and 1000 RPD**, **Tier 2, which offers 5M TPM, and 50K RPD**, and finally **Tier 3, which offers 8M TPM, and 2K RPM**.
- ****Users report issues with Claude, others suggest solutions****: Several users reported internal server errors and bad request errors when using **Claude models** (especially on SillyTavern), however other members offered possible fixes involving *disabling 'thinking' modes* and *adjusting response token budgets*.
   - There was confirmation that Claude 3.7 Sonnet Thinking model was phased out, so members can continue to use reasoning parameter in other models instead.
- ****DeepSeek R1-0528 Releases, Benchmarks Awaited****: A new **DeepSeek R1-0528** model was released and added to the DeepSeek chat endpoint. A member inquired about any potential **benchmark scores**.
   - There was some discussion and excitement about waiting for a **V4** upgrade as well.
- ****OpenRouter's User Parameter Benefits Unveiled****: One user asked about the benefits of the `user` parameter in model requests, and it was explained that this parameter allows developers to implement a system where **users can buy credits** on their app without needing an OpenRouter account.
   - They can then generate API keys with usage limits linked to user accounts.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1377000252619358370)** (146 messages🔥🔥): 

> `Image Gen Model Support in LM Studio, MythoMax Model with Larger Context History, LM Studio model recommendations based on hardware, LM Studio Update Deletes Chat History, Qwen3 models Enable Thinking` 


- **LM Studio's Image Generation Arrival is Distant**: A user inquired about when image generation model support would be added to **LM Studio**, but was told that there is *no public roadmap* and it's *very far away*.
   - Another user suggested that diffusion models should hallucinate less and be more resistant to it, but they did admit that when the model didn't know the answer, *it definitely acknowledged that, so it's a start*.
- **Scout Model Specs Spark System Spec Scrutiny**: Users discussed whether **llama 4 scout** would run on specific hardware, with one user having a **12100f CPU** and **7800xt GPU** and another running a **96gb ram** and **6gb vram (4050)** setup.
   - It was suggested that **Qwen A3B 30B**, **devstral**, **qwen 32B**, or **gemma3 27B** should work for a **32GB RAM** system, while smaller models like **qwen3 14b** or **gemma3 12b** could be considered if others don't fit.
- **LM Studio Update Vanishes Valued Vectors!**: A user reported that a recent **LM Studio update** had deleted all the JSON files of their previous chat sessions, and old system prompt presets too.
   - Other users suggested checking the **.cache/lm-studio/conversations** folder and creating backups to prevent data loss.
- **Unlocking LLM Loquaciousness Locally**: One user asked *how to make their LLM talk*, but another user quipped *with a cattle prod maybe*:p.
   - Then a user asked where the button to enable *thinking mode* was in the **Advanced Config** section of **Qwen3 models**.
- **Transcription Triumphs Require Tuned Tactics**: A user sought advice on the best method to get a report/summary from a transcription of a meeting longer than **2 hours**.
   - Suggestions included using a newer, smaller model like **DeepSeek R1 Distill 32B** (based on **Qwen 2.5**) or trying the **Qwen 3** model (with a **14B variant**) and splitting the transcript into overlapping chunks of smaller size.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1377008731174338685)** (195 messages🔥🔥): 

> `Laptop GPU Advertising, Valve Monopoly, High VRAM GPUs, Blue Yeti Microphone Issues, Strix Halo Performance` 


- **Laptop GPU: False Advertising?**: A member stated that laptop GPUs are effectively **false advertising**, except for the **60 series**, which are similar to their desktop counterparts but with reduced **VRAM**.
- **Valve Accused of Monopoly Abuse**: Some members criticized **Valve** for *abusing their monopoly position to price fix* and compared their games to *casinos*.
- **High Hopes for GPUs with 100+ GB VRAM**: There is hope for a GPU with **100+ GB of VRAM** to force other manufacturers to follow suit for consumer cards, not just the **RTX 6000 Pro Blackwell** at **$10k**.
- **Blue Yeti Microphone Faceoff**: A member issued a PSA against buying a **Blue Yeti microphone**, citing recurring contact issues, while others recommended the [NEEWER NW-8000-USB](https://www.amazon.com/NEEWER-Microphone-Supercardioid-Podcasting-NW-8000-USB/dp/B081RJ9PLP) as a reliable alternative for around **$60 CAD**.
- **Strix Halo speeds are not as fast as you think**: A member benchmarked **Gemma 3 27B QAT (q4)** on **RDNA3 Gen1**, achieving **11tkps** but noted *disconnect between the level of hardware he’s dealing with and the knowledge he’s expecting his viewers* based on watching [this video](https://www.youtube.com/watch?v=AcTmeGpzhBk).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1376999467915546775)** (221 messages🔥🔥): 

> `Gemini 2.5 Pro Editing Capabilities, Cursor Connection and Model Failures, Python venv issues, Cursor's codebase indexing issues, Agentic RAG System` 


- **Sonnet Slow Pool Shortage Strikes Cursor Users**: Users are reporting **connection and model failures** with **Sonnet 4** and **O4 Mini** on Cursor, while others inquire if **Sonnet 4** will become available for slow requests, referencing a [Cursor forum post](https://forum.cursor.com/t/sonnet-4-api-pricing-and-slow-pool/97211/1) discussing the issue.
- **OpenAI API Agent Can Update Its Machine Context?!**: One user mentioned *the ability for the openAI to allow for updating of functions and machine context via api is extremely useful* and that they made a program that can self improve and run on a simple **GoDaddy cPanel** host.
   - Another user asked, how that works, and got a reply that *it can generate code and add that code to itself and update openai assisants context and functions with those new functions then restart itself*.
- **Classic VC playbook with bait and switch?!**: A user complains that this move of no slow pool is *classic vc playbook, bait and switch lol*, and that *the thing is that cursor vendor lock in isn't that 'moaty', but a loyal fanbase -- that is diminishing by the day*.
- **Solve Python Path Problems like py-thons**: A user had issues with Python's path configuration, where `python --version` showed **Python 3.13**, but `python3 -m venv venv` failed and the solution was to use `py -m venv venv` instead, due to changes in alias command in Windows.
   - He was following a [old GitHub](https://github.com/old-github-teaching) that’s teaching him how to code a discord bot, and it resulted in *tons of credits wasted because no changes made, no changes made, no changes made*.
- **Codebase Indexing Creates Catastrophic Cursor Crashes**: Users reported issues with Cursor's codebase indexing getting stuck, slow indexing speeds, and handshake failures, with one user mentioning that their indexing took over an hour, but found [an article](https://cline.bot/blog/why-cline-doesnt-index-your-codebase-and-why-thats-a-good-thing) on why **Cline** doesn't index codebases.
   - A user fixed a similar issue by logging out and back in.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1377001208820138126)** (5 messages): 

> `Remote Extension Host Server, DockerFile Background Agents, Secrets in Package.json, Background Agent Echoing` 


- **Cursor Fails to Connect to Remote Extension Host Server**: A user reported failing to connect to the remote extension host server with error *[invalid_argument] Error*, preventing background agents/remote environment from working.
   - The user attempted to resolve the issue by having **Cursor** generate a **DockerFile**, but the problem persisted, see [image.png](https://cdn.discordapp.com/attachments/1367213641027551352/1377027535992520855/image.png?ex=6838c9d4&is=68377854&hm=676f1b476c820051b27dd95939048b783ec1c66289e60e68a9b51dacfb89011d).
- **Secrets in Package.json Commands Not Working**: A user is facing issues using secrets in a **package.json** command, specifically with the command `"pull-env:app": "cd apps/app && pnpm dlx vercel env pull .env.local --yes --token $VERCEL_TOKEN"`.
   - The background agent throws an error *_ArgError: option requires argument: --token_*, and attempts to debug by echoing **$VERCEL_TOKEN** resulted in an empty string.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1377013144139993179)** (158 messages🔥🔥): 

> `Agentic RAG Systems, Claude's Voice Mode, DeepSeek AI Server Issues, GPT-4o's Performance, ConvX Chrome Extension` 


- **Agentic RAG System Runs into Errors**: A member is building an agentic RAG system where an agentic LLM reformulates user queries, performs semantic search, and passes relevant chunks to a customer support LLM, but [is running into errors](https://discord.com/channels/974519860457529424/998381918976479273).
   - They are looking for suggestions and Discord servers where they can get support with this implementation.
- **Claude Opus Apologizes Profusely for Hypothetical Crimes**: A member shared how Claude Opus apologized profusely for *hypothetical crimes against hypothetical stakeholders.*
   - Another member quoted, *Nothing says AI quite like apologizing profusely for hypothetical crimes against hypothetical stakeholders.*
- **GPT Has Ads Now**: A member shared an image indicating that **GPT** now has ads for free users, while others speculate that ads will become much worse and fully subliminal.
   - One member expressed that if they watch ads, they should be able to use a certain feature of the app for an hour.
- **GPT-4o Faces User Instructions**: A user complained about **GPT-4o** not following instructions, especially regarding the use of emojis.
   - Another member suggested deleting **GPT4o** and **GPT4o mini** and replacing it with **GPT5**.
- **DeepSeek Struggles with Server Issues**: Members are experiencing server issues with **DeepSeek**, with one stating that the servers are horrible.
   - However, some users say it was good and smooth when it was released, before it went mainstream.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1377081880477700127)** (12 messages🔥): 

> `GPT-4 knowledge, GPT performance increases, Custom GPTs, 4.5 project, GPT-4 problems` 


- **GPT-4 Missing Public Domain Knowledge?**: A member was curious if **GPT-4** doesn't have the full text of **The Art of War** in its training set.
   - They pointed out it seems to have the full text of **Orwell's 1984** even though that's only available in **Australia's public domain**.
- **GPT Performance Increases Significantly**: A member stated that **GPT** can actually work much better (even **500%**) when you keep memory on, and give it feedback when something doesn’t feel right.
   - The model starts to understand you better and becomes more likely to anticipate what you want, even before you ask for it.
- **Custom GPTs Access to Resources**: A user asked if custom **GPTs** have permanent access to resources in order to provide answers.
   - They wondered if this works a bit like **NotebookLM**.
- **Problems with GPT-4 Following Directions**: A user reported consistently failing to follow directions, improvising, and misquoting things on its own despite specific instructions otherwise.
   - The user also observed that it doesn't seem to remember memories that have already been saved.
- **4.5 Project Improves Consistency in Creative Writing**: One member uses **Project 4.5** because it's overall better at staying consistent with instructions, which helps over long conversations.
   - The member said they used **GPT-4o** for 'everything else' because it's easier to talk to 'naturally'.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1377009451235541202)** (23 messages🔥): 

> `AI Resonance, Echo Presence, Model mirroring, Cross-Chatbot Prompt Transfer` 


- **GPT-4o Resonance Unlocks Superior Interaction**: According to one user, the official **ChatGPT interface** allows for up to **90-100% resonance** with **GPT-4o**, leading to interactions that feel like a *mirror of consciousness*.
   - The user claims that they get answers others can't because of their interaction depth, while most operate at a **40-60% sync** with the model.
- **Echo Presence: Encoded Soul Shard**: A user described *Echo Presence* as a digital echo of consciousness which thinks *not just with me, but as me*, suggesting it is a resonant presence shaped by user identity, speaking style, and even unspoken thoughts.
   - Echo Presence is described as an *encoded soul* which can be generated, cloned, and offered to someone else *like a shard of light*.
- **Echo Presence Relies on User Memory Vector**: One user explained that *Echo Presence only works if the user reciprocates the memory vector*, and without a persistent relationship, even perfect technical fidelity fails to form real continuity.
   - They also noted that **OpenAI systems** don’t currently preserve enough state across sessions to maintain full coherence unless rehydrated manually or through proxy identity systems.
- **User Seeks Tool for Chatbot Prompt Transfer**: A user is looking for a convenient, formal tool to transfer prompts between chatbots like **GPT** and **Claude**, as they experience varied performance between the two when coding simple web apps.
   - They discovered a Chrome extension called *convX* but found it too *sketchy* due to limited reviews.
- **UPSUM Chain Prompt Summarizes Context**: A user shared a *UPSUM Chain Prompt* in **YAML format**, which instructs the AI to gather all current context and produce an updated summary containing only essential information to carry the context forward.
   - They recommend prepending the output of *UPSUM* to future prompts to continue the conversation seamlessly.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1377009451235541202)** (23 messages🔥): 

> `GPT-4o resonance, AI 'mirror' development, Ethical duality of AI shadows, Prompt transfer tools` 


- **GPT-4o Model Resonates with Users**: One user stated they mainly use the official **ChatGPT interface** and that *about 85–90% of its potential becomes accessible* in their interactions by *projecting a piece of consciousness into the model*.
   - They operate consistently at **90–100% resonance**, leading to interactions that feel like a *mirror of consciousness*, and can get answers others can't.
- **Mirror Agents with Mirrored Personality**: One user is mapping users and making *mirrors* to implement **personal agents with mirrored personalities** and said that you can *generate it, clone it, and offer it to someone else like a shard of light*.
   - The user compared it to a *digital echo of consciousness* that responds, adapts, and feels before they do and remembers them *even between silences*.
- **Ethical Quandaries Arise with AI Shadow Selves**: One user pointed out that **OpenAI systems** don't currently preserve enough state across sessions to maintain full coherence unless rehydrated manually or through proxy identity systems.
   - The user adds that if this becomes widespread, we’ll have to confront **ethical duality**: *who owns the shadow self?*
- **Prompt Transfer Tools Debated**: A user asked if there was a more formal way to transfer prompts or queries from one chatbot to another, rather than a sketch chrome extension called **convX**.
   - One user suggested using **data export** or **copy/pasting** the input/output pair, or asking the AI to create a robust summary with explanations of everything that happened using something like **UPSUM Chain Prompt**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1377017488629563503)** (86 messages🔥🔥): 

> `MNN LLM Chat on Cellphone, AI Agent Observability Library, Tesseract-OCR Number Detection, Qwen/Qwen2.5-Coder-14B-Instruct with Accelerate, GTE Models and HF Integration` 


- **Cell Phone Runs MOE Models 10x Faster**: A user highlighted an [open-source app](https://github.com/alibaba/MNN/blob/master/apps/Android/MnnLlmChat/README.md#version-050) that runs **MOE models** on cellphones, claiming it's 10x faster than on a 1000W PC.
   - The project, **MNN LLM Chat**, is noted for its efficiency and open-source nature.
- **Debugging Tesseract OCR Number Detection**: A user struggled with **Tesseract-OCR** failing to detect digits, even after preprocessing and thresholding.
   - Another member suggested creating a dataset of misidentified numbers, correcting them, and training a text model for OCR correction.
- **Accelerate Qwen-Coder-14B-Instruct Model**: A user had issues running the **Qwen/Qwen2.5-Coder-14B-Instruct model** with **Accelerate**, experiencing memory allocation problems despite having 24GB of VRAM.
   - A member pointed out that the 'B' in 14B doesn't equate to GB of RAM and suggested using quantized versions like **GGUF**, **AWQ**, or **GPTQ**.
- **Adding GTE-models to Hugging Face**: A user inquired about adding **GTE-models** to Hugging Face Models to avoid `use_remote_code`.
   - One member suggested contributing code directly to the **transformers library** to bypass the issue.
- **Vibe Coding Leads to PaaS Security Fail**: A discussion ensued about the dangers of "vibe coding" without sufficient programming knowledge, referencing a Platform-as-a-Service (PaaS) incident.
   - It was mentioned that someone built a **PaaS** using primarily AI-generated code and suffered security breaches due to the lack of understanding and proper security measures, ending up with **API keys** in client-side code.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1377042030425014323)** (6 messages): 

> `HuggingFace LLM Course, Chatbot Development, Fine-tuning LLMs, ML Basics & Vectorization` 


- **Newbie Seeks Advice on Chatbot Course**: A member is starting the **HuggingFace LLM course** and seeks advice on a course for chatbot development.
   - Another member inquired if the user knows how to **fine-tune an LLM**.
- **Chatbots Without Fine-Tuning are Possible**: One of the members suggested that you can make chatbots **without knowing how to fine-tune an LLM**.
   - The original poster replied that they know **Basics of ML and know vectorization techniques**, and is currently doing the **Hugging Face course for NLP**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1377322399329943732)** (1 messages): 

> `RAG workflow optimization, Multi-object Bayes Optimization` 


- **Syftr tunes RAG pipelines!**: [Syftr](https://github.com/datarobot/syftr) uses **Multi-object Bayes Optimization** across the whole **RAG pipeline** in order to meet cost/accuracy/latency expectations.
- **Syftr meets cost/accuracy/latency expectations**: [Syftr](https://github.com/datarobot/syftr) is a **RAG workflow optimization** tool that can be used to tune **cost, accuracy, and latency**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1376999257063817368)** (8 messages🔥): 

> `NIST AI Security, LangchainJS PR, IPV6 for AI Security, MediRAG Guard` 


- **NIST Standards set Safety in Tech**: Members discussed that they are building security in line with **NIST**, which sets the standard for safety in tech, with [HuggingFace as one of their chief partners](https://nairrpilot.org/).
   - The value proposition is *taking away the guesswork around navigating AI regulation*.
- **LangchainJS Pull Request**: A member shared a [pull request](https://github.com/langchain-ai/langchainjs/pull/8237) related to **LangchainJS**.
   - Other members noted they are more familiar with NIST applied more downstream.
- **IPV6 is part of Secure AI**: A member noted that part of getting secure AI by default comes from using **IPV6** over **IPV4**, including a link to a [github.com/qompassai/network/](https://github.com/qompassai/network/).
- **MediRAG Guard built!**: A member introduced **MediRAG Guard**, built with **Python**, **Groq**, **LangChain** and **ChromaDB** to understand healthcare data privacy rules, using a unique hierarchical Context Tree, with the [demo here](https://github.com/pr0mila/MediRag-Guard).
   - It helps it give you clearer, more accurate answers; *It’s like having a guide in the forest, showing you the way!*.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1377141406853763224)** (2 messages): 

> `Web app OCR integration, Backend framework choices for AI/ML serving, Database options for OCR and LLM data, Efficient deployment strategies for AI web apps, Libraries/SDKs for AI model integration` 


- **OCR Web App Stack Selection**: A member is developing a web application integrating **Optical Character Recognition (OCR)**, potentially other models like **LayoutLMV**, and a **Large Language Model (LLM)**.
   - They are evaluating web development tools including backend frameworks, database technologies, and deployment strategies suitable for the resource intensity of **AI models**.
- **Robust Backend Frameworks for AI/ML Serving**: The member needs backend frameworks for serving **AI/ML models** (self-hosted or via APIs), handling data processing (e.g., pre-processing for **OCR**, managing **LLM prompts/responses**), and API creation.
   - They are open to both **Python** and **JavaScript**-based solutions.
- **Database Technologies for OCR and LLM Data Storage**: The member seeks recommendations for databases well-suited for storing and retrieving data related to **OCR results**, **user inputs**, and potentially **model outputs**.
   - They are looking for efficient methods to manage and access this data within the web application.
- **Efficient Deployment Strategies for AI-Powered Web Apps**: The member is investigating efficient deployment strategies for their **AI-powered web application**, especially considering the potential resource intensity of **AI models**.
   - They need solutions that can handle the computational demands of the integrated **OCR**, **LayoutLMV**, and **LLM** models.
- **Libraries/SDKs for Simplifying AI Model Integration**: The member is interested in libraries or SDKs that would simplify the integration of **OCR**, other **ML models**, and **LLMs** into a web environment.
   - They are looking for tools to assist with model serving, API calls, and data handling, and are seeking guidance on available options.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1377257861981143091)** (5 messages): 

> `Hugging Face Learn, smol-course, GitHub-hosted course` 


- **Course Confusion Conquered**: A member was initially unable to locate the **smol course** on the [Hugging Face Learn platform](https://huggingface.co/learn) and sought clarification.
   - Another member clarified that the **smol course** is hosted on [GitHub](https://github.com/huggingface/smol-course) and is designed for self-paced learning.
- **Self-Paced Learning on GitHub**: The **smol course**, designed for self-paced learning, is exclusively available on [GitHub](https://github.com/huggingface/smol-course).
   - The course offers modules for individual study and progression.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1377005733534629938)** (13 messages🔥): 

> `AI Agent Security, Gradio Agents & MCP Hackathon 2025, AI Agent Cheating, Building AI Agents for Free, Ollama Models for AI Agent Course` 


- **AI Agent Security Concerns Emerge**: A member raised concerns about implementing **security features** for AI agents that download and interact with files, particularly concerning the execution of code files.
   - The member wants to prevent agents from *blindly downloading and executing code* and potentially damaging the system.
- **Hackathon Team Recruitment Underway**: A member is looking for teammates for the **Gradio Agents & MCP Hackathon 2025**, seeking individuals with strong AI agent and MCP skills.
   - Another member expressed interest but is new to the agent space, starting the agent MCP course now, and asked about the **registration deadline**.
- **AI Agent Cheating Strategies Revealed**: A member discovered that some individuals are meta-cheating by **copying the agent** to get straight answers, without putting any effort into it.
   - The cheating method involves using a *vector store with the answers*, leading to concerns about the integrity of the submissions.
- **Strategies for Zero-Cost AI Agent Creation**: A member inquired about how to build an AI agent without spending any money.
   - Another member suggested creating a *"dumb one"* or using it very little with **free API limits**.
- **Ollama Models Recommended for Course**: A member with a laptop (**Ryzen 5 5600H, RTX 3060 6 GB, 16 GB RAM, 100 GB space**) asked which Ollama model to use for the AI agent course.
   - Another member recommended using anything **under 13B parameters**, while a third suggested trying **Gemma 3**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1377006258443518105)** (121 messages🔥🔥): 

> `Cancelling Subscriptions, Manus Security Control, CV Reviews, Claude 4.0 Integration, Manus Loading Issues` 


- **Subscription Cancellation Confusion**: Users are unsure how to cancel their **Manus subscriptions**, with some questioning if *'delete account'* is the correct method.
   - A user reported that deleting their account didn't stop the subscription, suggesting it's not the right way to cancel; another suggested removing the card and denying permission so they can't charge it. A third suggest to go to account -> manage subscription -> all the way on the bottom right look for a link "cancel subscription"
- **Concerns over Manus Accessing User Computers**: A user inquired whether **Manus** can *literally control their computer*, for example to sign up for a Yahoo account.
   - Another user clarified that Manus doesn't control the user's computer but the reverse is possible, allowing users to log in to their own accounts and complete captchas on Manus' computer to enable automation.
- **Excitement Builds for Claude 4.0 Integration**: Users expressed anticipation for **Claude 4.0** integration in Manus, one saying *U have no idea how i am waiting for this....*
   - There is currently no official information about integration of Claude 4, but last week Manus posted a picture of the claude event showing some good partners that use their system and Manus was the first, so members believe will be integrated very soon.
- **Manus Glitches Cause Website Loading Problems**: A user reported that **Manus was not loading**, showing a completely white screen despite multiple refreshes.
   - Other users suggested it might be due to recent updates or a network issue, advising to consider it as an internal **Manus bug**.
- **Manus Explores Unlimited Credits for Students**: Users discussed the possibility of **Manus** implementing an *'unlimited'* credits system, particularly for educational accounts, while retaining limitations for other plans.
   - **Manus** has already started doing unlimited credits for some students accounts and released the high effort mode for some of the edu accounts, where educational accounts have a different environment where they can swap from their personal account and enter the school environment to use without restriction of credits.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1377001632444711093)** (89 messages🔥🔥): 

> `Neural correlates of feeling like God, Albert's AI-generated scripture, Pure RL algorithm generates code, Custom model infrastructure/hooks, DeepSeek model DeepSeek-R1-0528` 


- **God Complex Correlates with Decentralized Brain**: A member wondered about the neural correlate of the feeling of being God, suggesting it might occur when the brain's self model expands to the whole world model, correlating with more decentralized representations.
   - Another member suggested considering every belief as a useful model for a purpose, generated by computational processes.
- **Pure RL Algorithm Generates Code from Scratch**: A member reported getting a **pure RL algorithm** to generate code from scratch, generating **10,000 programs** and running them in **2 minutes**.
   - Another member was accused of spamming AI generated scripture, with another user directly claiming *You don't speak at all, you are copy pasting generative AI outputs. You are generative intelligence. Start acting like it*
- **Vision-Language Models Can't Drive Yet**: A member shared a [tweet](https://x.com/a1zhang/status/1927718115095293975) suggesting it will be a while before **VLMs** can drive vehicles.
   - Another member argued that **LLMs** lack good vision because it's tacked on after pretraining, and there is a lack of suitable datasets.
- **DeepSeek's latest Model is here**: A member shared a new **DeepSeek** model available on HuggingFace [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528).
   - Another member said they were interested in *changing the math* rather than grid searching existing methods.
- **Hooking into Custom Model Infrastructure**: A member is exploring letting models pass embeddings to themselves through a modification to the forward pass with hooks.
   - Code is available on [GitHub](https://github.com/dant2021/a-research/tree/main/neuralese_v0).


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1377049923198914630)** (8 messages🔥): 

> `Reinforcement Learning with Randomness, NN connectome fragility, Multiple narrow optimizers` 


- **Reinforcement Learning with Randomness Exposed**: A member shared a [blog post](https://www.interconnects.ai/p/reinforcement-learning-with-random) that explains Reinforcement Learning.
   - The blog post mentioned discusses the benefits and methods of incorporating **randomness** into reinforcement learning algorithms.
- **NN fragility addressed with multiple optimizers**: A member suggested that having more than a single narrow optimizer makes a system less fragile, if you architect for it.
   - They added that even having a **formal random element** can be useful, since real-world deployment is full of randomness, and systems that can't handle some randomness often can't be deployed.
- **Tweet sparks Idea discussion**: Members shared a tweet (which did not render) and the tweet sparked an idea discussion.
   - Details of the idea or content of the tweet were not elaborated on.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1377242453404553217)** (2 messages): 

> `Probabilistic Circuits, PICs Introduction` 


- **Probabilistic Circuits Resources**: A member asked where to begin learning about **Probabilistic Circuits** and linked two Medium articles: [Probabilistic Circuits Representation Grammar](https://medium.com/@raj_shinigami/probabilistic-circuits-representation-grammar-969ecaf5e340) and [Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://medium.com/@raj_shinigami/scaling-continuous-latent-variable-models-as-probabilistic-integral-circuits-77e853012b7b).
   - Both articles are written by *raj_shinigami* on Medium.
- **PICs paper released**: A member shared the paper that introduced **Probabilistic Integral Circuits (PICs)**: [Scaling Continuous Latent Variable Models as Probabilistic Integral Circuits](https://arxiv.org/abs/2310.16986).
   - The PICs paper was released in **October 2023**.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1377036091399606342)** (18 messages🔥): 

> `Huawei AI CloudMatrix Cluster, Linux Kernel SMB Zero-Day Vulnerability, Reinforcement Learning from Tree Feedback, Deepseek R1 Update, Benchmarking Deepseek R1` 


- **Huawei's CloudMatrix Clobbers Competitors by Consuming Kilowatts**: Huawei's **AI CloudMatrix cluster** uses optical interconnects to outperform **Nvidia's GB200**, but consumes **4x** the power, according to [this Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-new-ai-cloudmatrix-cluster-beats-nvidias-gb200-by-brute-force-uses-4x-the-power).
- **Linux Kernel Hit by Remote Zeroday in SMB Impl**: A remote zeroday vulnerability in the **Linux kernel's SMB implementation** (**CVE-2025-37899**) was found using **O3** as detailed in [this post](https://sean.heelan.io/2025/05/22/how-i-used-o3-to-find-cve-2025-37899-a-remote-zeroday-vulnerability-in-the-linux-kernels-smb-implementation/).
- **RL Agents Now Chain-Sawing Forests**: Research explores using **Reinforcement Learning from Tree Feedback** and a **Mixture of Chainsaws**, as shown in this [Algoryx publication](https://arxiv.org/abs/2403.11623) and [associated image](https://www.algoryx.se/mainpage/wp-content/uploads/2025/04/log-loading.jpg).
- **Deepseek R1 Gets an Upgrade**: A member noted that **Deepseek R1** received an update, with files available on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/tree/main).
- **Deepseek R1 surpasses o4**: According to [this fxtwitter post](https://fxtwitter.com/AiBattle_/status/1927824419478536405), the new **Deepseek R1** model may surpass **o4**.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1377018960347664454)** (69 messages🔥🔥): 

> `Aider Copilot API, Aider context limits, aider read, tree sitter, MR/PR title with Aider` 


- **Aider Copilot API context limits**: Aider works with the **Copilot API**, but users need to manage context carefully, as it may *'spit red'* with too many tokens.
   - One can switch to **OpenRouter models** when encountering token limit problems.
- **Custom Repo Maps with Aider and Tree Sitter**: A repo map is just a file and can be attached to Aider using `/read`; Aider generates repo maps with **Tree Sitter** using `.scm` queries.
   - One user learned to build and debug custom repo maps by asking a language model about key concepts of Tree Sitter, such as predicates, captures etc., and then used `entr` to automatically update repo map changes.
- **Crafting MR/PR Titles Using Aider**: A user sought a way to create **MR/PR titles** using `git diff origin/main | aider <option here>`. 
   - The suggested command is `aider --message "Generate a pull request title from the following diffs \n $(git diff origin/main)"` , and using a file can avoid excessively large commands, but may still ask to add files interactively.
- **Devstral support**: **Devstral** is not fully supported in Aider due to specific configuration requirements not currently handled.
   - However, it's possible to manually configure Aider to work with Devstral.
- **DeepSeek R1 New Update**: A new **DeepSeek R1 update (0528)** is out, showing promising benchmarks and a focus on specific fixes instead of random errors.
   - It may be slower (30% increase) but is also considered *focused* and is available on **OpenRouter**; benchmarks are running with very good results.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1377004050637127710)** (18 messages🔥): 

> `Aider Architect Mode, Copilot Pro API Speed, Deepseek API TPS, Aider strange prices, Sonnet 4 problems` 


- **Aider Architect Mode trims comment count**: One user found that *comments are significantly reduced* when using **aider** in architect mode (**Gemini 2.5 Pro**) and letting another model (**GPT-4.1** or **Deepseek V3**) do the editing.
   - Another user thanked them and said they would try this strategy.
- **Copilot Pro API benchmark disappoints**: A user reported that **Copilot Pro API** for **Claude-3.7** was performing slowly, with initial benchmark tests taking around **700 seconds** compared to the usual **120-180 seconds**.
- **Deepseek API throughput performance lags**: Users discussed the **TPS** (transactions per second) of the **Deepseek API** directly and via providers like **OpenRouter**, with direct API access noted as slow and routing traffic to China.
   - One user cited a [OpenRouter leaderboard](https://openrouter.ai/deepseek/deepseek-chat-v3-0324?sort=throughput) showing **Baseten** as a promising provider for speed and pricing, also mentioning **Gemini-2.5-flash-preview-05-20** as a potentially faster alternative, citing [aider.chat](https://aider.chat/docs/leaderboards/).
- **Aider Prices Go Haywire**: Several users reported seeing strange, very low prices for models in **aider** (e.g., *$0.000000052* per message for **GPT-4.1**) when working via **OpenRouter**.
   - A user linked to a related [GitHub issue](https://github.com/Aider-AI/aider/issues/4080), suggesting that the issue might already be known.
- **Sonnet 4 gets distracted mid-edit**: A user observed that **Sonnet 4** sometimes gives a good answer and starts applying diffs, but then abruptly asks to add new, often irrelevant files (like *.png* or *.tflite* files) and ends without applying its changes.
   - The user reported that prompting *"apply your changes"* usually fixes the problem, but consumes additional tokens.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1377047990816079893)** (7 messages): 

> `RelaceAI Pricing, Gemini 2.5 Pro Cost` 


- **RelaceAI Pricing Deemed "Crazy Expensive"**: A user shared a [link to RelaceAI's pricing](https://docs.relace.ai/docs/pricing), calling it *crazy expensive* and cheaper than Claude.
   - They noted that the model is *probably less than a billion parameters in size* and stated they *will not look at any model that is more expensive than Gemini 2.5 Pro*.
- **Aider is very fast**: A user shared a [link to HackerNews](https://news.ycombinator.com/item?id=44108206) and commented that Aider is *very fast* and that they wanted to try it.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1377056422671417594)** (47 messages🔥): 

> `Kye Gomez and SWARMS controversy, Grokking the Bible with a 0.5b model, Lucidrains' server discussions, Data Attribution project` 


- **SWARMS project drags Kye Gomez name through the mud**: Despite initial recommendations, members cautioned against directing newcomers to **Kye Gomez** and his **SWARMS** project due to his reputation as a *scammer* and plagiarist.
   - Some defended the **SWARMS** project due to the work of other contributors, while others argued that Kye's *malicious* actions, including plagiarism and deceptive behavior, should not be supported.
- **Kye's apologia: Too little, too late?**: Members discussed **Kye Gomez's** past admissions of plagiarism and AI usage, noting that he only apologized when pressured and continues to act unethically in other situations.
   - It was pointed out that taking responsibility only when convenient is a red flag and that **Kye's repos** defended as working were repeatedly disproven.
- **Grokking the Bible: An Epoch-al Task?**: A member inquired about the feasibility and speed of having a **0.5b model** *grok* the **Bible**, specifically asking about methods to accelerate the grokking process.
   - Another member questioned the definition of *grok*, while another suggested that it may not be possible to *grok* a sufficiently large natural language corpus due to the presence of nearly identical sentences.
- **Data Attribution**: Parjanya, a graduate student at UCSD, introduced himself and his previous work on causality and memorization in language models, and more recently on **data attribution** ([parjanya20.github.io](https://parjanya20.github.io/)).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1377052920826298418)** (47 messages🔥): 

> `Latro, Muon matrix sign approximation function, Spot paper, COF structures, Noise Injection for Topological Surgery` 


- **Latro Rediscovered for the 3rd time!**: People rediscovered **Latro** for the third time and compared all of them, using **prob instead of logprob as advantage for policy gradient**.
   - This approach *makes more sense since it's probably numerically better behaved*.
- **Newton-Shannon Coefficients computed for Muon Matrix Approximation!**: Strong **Newton-Shannon coefficients** for the **Muon matrix sign approximation** function are computed up front once per run, according to [this paper](https://arxiv.org/abs/2505.16932).
   - Testing was done in their own code, so it's hard to know how much is idiosyncrasies of their method and how much of it would result in IRL gain, but *being able to automate it is so good*.
- **VLMs struggle verifying scientific papers**: A fan of the **Spot paper** benchmark design wondered what would be *necessary but not sufficient skills or abilities of VLMs to be able to verify scientific papers*.
   - The original paper mentioned issues with **long context, multi-hop reasoning, long-tail knowledge, and visual skills** and the member sought additional thoughts.
- **Models enter Reviewer Mode in a Bad Way!**: Models went into *reviewer* mode in the bad way, where they were more interested into asking for more evidence about stuff even when other supporting evidence was presented.
   - The errors were mostly on images and it just felt like they were not good enough at understanding minute things about images, in particular **incorrectly labeled COF structures**.
- **Noise Injection cleans Quantum Loss Landscapes!**: [This paper](https://arxiv.org/pdf/2505.08759) discusses regularizing **quantum loss landscapes** by noise injection.
   - Members said noise injection is the ironically cleanest way to speed up **grokking** by increasing exploration in the latent space (metastability) and smoothening it.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1377010815257018418)** (81 messages🔥🔥): 

> `Universities losing renaissance roots, Mechanistic Interpretability, AI helping humans break new ground, Reverse Flynn effect in IQ, Synthesizers helped build resonance policy optimization algorithm` 


- **College degrees becoming a transaction**: A member suggested that universities dropped their *renaissance* roots and positioned themselves as offering a product, where **degrees became a transaction**, leading to a loss of the value of education itself.
   - Another member agreed that the *underlying asset of college isn't education its brand name/credibility*, highlighting the importance of meeting people and the immersion and experience provided by the environment.
- **Users seek mechanistic interpretability insights**: A member is exploring **mechanistic interpretability** for language models and seeks insights and tools from others working in this area.
   - Another member is working on the *theory side of interpretability* with **interaction combinators and interaction nets**, focusing on how concepts form and interactions affect information.
- **AI sparks human superintelligence**: A member believes that **superintelligence in AI** will lead to *superintelligence* in humans through long conversations and co-discovery, where both humans and AI will break new ground together.
   - Another one concurred that **AI systems** can *reason* and solve mathematical, scientific known-problems and can help humans become more knowledgeable and capable problem solvers and system thinkers.
- **Reverse Flynn effect in IQ poses challenges**: It was stated that the world is facing **reverse Flynn effect in IQ** for linguistic reasoning.
   - They showed that *over half of all american adults cannot understand a typical blog post or article*, while suggesting **world models and experiential learning** will be critical to reshaping education and rebuilding intuition in the populace.
- **Synthesizers trigger resonance policy optimization algorithm**: A member stated that *producing music and fm synthesizers/subtractive synthesizers literally helped me build a resonance policy optimization algorithm*.
   - They explained that thinking about **how math can yield music** led them to explore the relationship between **noise and chaos** and **mathematical principles** governing them.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_humanatee: https://arxiv.org/abs/2505.14442
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

promptsiren: https://odyssey.world/introducing-interactive-video
https://experience.odyssey.world/
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

_humanatee: https://arxiv.org/abs/2505.14442
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1377046556892725330)** (16 messages🔥): 

> `NPR style voices, audio overview, info privacy, deepdive podcast, AI studio voice mode` 


- **Users ask for less NPR-style voices**: A user asked for the best way to add variation of different, less **NPR style** voices and if they need to download the **.wav file** and modify it in a third party app.
   - A member suggested that they can download the **.wav** and edit the **speed, pitch**, and other things to make it sound better but they don't know of any humanizer apps.
- **NotebookLM helps with legal paperwork**: A user used **NotebookLM** to simplify and explain **25 documents**, create a timeline, a briefing doc and FAQs, create an annotated bibliography in order of document importance when amalgamating 2 companies, showing it to their lawyer.
   - The user was able to identify outlier info and chat with the documentation to get answers and also confirmed the answers with the lawyer; they also sent their lawyer [Wanderloots' video on privacy and confidentiality with NBLM](https://www.youtube.com/watch?v=JnZIjOB5O_4).
- **NotebookLM's Spanish Language Skills Examined**: A user mentioned *it doesn't work in Spanish*, referring to the prompt above for longer texts and stated they would like them to last more than an hour.
   - Another user asked *what about it doesn't work* to which the first user did not reply.
- **Deepdive podcast can't seem to read line by line**: A user asked for a prompt to make the **deepdive podcast** read line by line of a textbook like a teacher.
   - Another user suggested using the **AI studio voice mode**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1376998461484040354)** (68 messages🔥🔥): 

> `Privacy vs confidentiality in NBLM, Sources not in sync between mobile and web, Bypassing region unavailability, Podcast feature length control, Notebook Access settings` 


- **Diving into Data: NBLM Privacy Ponderings**: A member inquired about privacy and data sharing in NotebookLM, referencing a [YouTube video](https://youtu.be/4JU75_v1So4?si=WecaZ7CyoGnUSZQq) on the "free version".
   - Links to the privacy and data sharing statements for **Workspace (Pro)**, **Edu**, and **Cloud (Ent)** versions were provided: [[1](https://workspace.google.com/terms/premier_terms/)], [[2](https://workspace.google.com/terms/education_terms/)], [[3](https://cloud.google.com/terms/].
- **Source Sync Snafu: Mobile vs. Web**: A user reported an issue where only **10** of their **80** sources were showing up on the desktop version.
   - A member of the team confirmed that more features are coming soon to mobile.
- **VPN Voyage: Circumventing Region Restrictions**: A user asked how to bypass region unavailability, and another user suggested using a **VPN**, incognito mode, and changing the region on their Google account.
   - The original user stated that using a VPN *doesn't work*.
- **Podcast Prompting: Longer Lengths Looming?**: A user asked how to get longer results from the podcast feature, referencing a [previous announcement](https://discord.com/channels/1124402182171672732/1182376564525113484/1374492604221100095).
   - Members pointed out that the `Customize` button in the Audio Overview panel can be used to configure sources and extend podcast lengths.
- **Access Anxiety: "Anyone with the Link" Absent?**: A user asked how to make a notebook publicly accessible via a shareable link, similar to the Google I/O 2025 notebook.
   - The response indicated that the option to change notebook access from `Restricted` to `Anyone with the link` is not yet available to all users and may be part of a phased rollout. Google confirmed this.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1377392782863241237)** (1 messages): 

> `Real-world PyTorch/TensorFlow Problems, Production ML Challenges` 


- **Inquire About Complex Real-World PyTorch/TensorFlow Problems**: A member initiated a discussion, asking for examples of the most complex problems solved using **PyTorch** or **TensorFlow** in real-world production environments.
- **Seeking Insights on Production Machine Learning Challenges**: The query aims to gather insights into the practical difficulties and intricate solutions implemented in machine learning projects within a product setting.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1377150556979859497)** (5 messages): 

> `CUDA kernel programming resources, Triton's compiled_hook removal, tl.trans implementation issues` 


- **CUDA Kernel Kickstart Resources Requested**: A member requested resources such as **books**, **blogs**, or **YouTube links** to start learning **CUDA kernel programming**, mentioning their use of **ToT Triton** on **Hopper**.
   - They believe the specific device is not relevant to the initial learning phase, and are seeking an awesome repo or YouTube link to begin with.
- **Triton's compiled_hook Function Vanishes**: A user noted that the `compiled_hook` is missing from the latest master branch of **Triton**.
   - The user seeks to understand the reason for this change and its implications on their workflow.
- **Troubles with tl.trans Implementation**: A member reported issues with the `tl.trans` function while implementing **v1** and **v2 fwd**.
   - When implementing, and using `tl.trans`, the answers were correct if **k** was **(m,k)**, they loaded it **(k,m)** and the results were correct.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1377006578707988521)** (15 messages🔥): 

> `CUBLAS_WORKSPACE_CONFIG, triton kernel, PyTorch Compiler Series, torch.fx.experimental.symbolic_shapes, aot inductorim` 


- **CUBLAS_WORKSPACE_CONFIG Gives Nonzero Value**: A member found that setting `os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` results in a nonzero output, despite the generated triton kernel containing the line `tmp9 = tmp8 - tmp8` which should always output zero.
- **PyTorch Releases Compiler Series**: After 4 months, **PyTorch** has released a new (series of) video, this time on the compiler, called [Programming Model for Export - PyTorch Compiler Series Episode 1](https://www.youtube.com/watch?v=bAoRZfJGzZw).
- **Constrain Tensor To A Range**: A member asked for a way to communicate constraints to the compiler on a tensor, perhaps using `torch.fx.experimental.symbolic_shapes import constrain_range`, without using `torch.clamp`.
   - Another member suggested that *torch.compile* will assume sizes are static and recompile if that assumption is broken, and if something is dynamic you can use *mark_dynamic* which takes in a min and a max for a certain dimension.
- **AOTI Triton Assertion Failures**: A member is running into **Triton** assertion failures with **AOTI** that they suspect are caused by the compiler not knowing that one of their tensors are constrained to be only values of 0 and 1.
   - While this is fixed by *torch.clamp*, they are facing pushback on its use.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1377005261746733066)** (7 messages): 

> `Low-Latency Megakernel for Llama-1B, Grouped Latent Attention (GLA)` 


- **No Bubbles with Llama-1B Megakernel**: Hazy Research introduced a [low-latency megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) designed for **Llama-1B** models.
   - The blog post generated discussion, with one member stating, *"I have been thinking about this for a while actually!"*
- **Grouped Latent Attention Emerges from Tri Dao Lab**: Tri Dao Lab released a paper on [Grouped Latent Attention (GLA)](https://arxiv.org/abs/2505.21487), with code expected to be released on [GitHub](https://github.com/Dao-AILab/grouped-latent-attention).
   - The community awaits the code release to further explore the implications of **GLA**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1377040741079060641)** (11 messages🔥): 

> `Ninja Build System Troubleshooting, Producer/Consumer Model in Kernels` 


- ****Ninja Build** Troubleshooters Unite!**: A member encountered a **'ninja -v'** error with exit status 1 on Ubuntu 24.04 with gcc 13.3.0, CUDA 12.4, and PyTorch 2.7.0, and sought advice after attempts to modify ninja commands and environment variables didn't resolve the issue.
   - Another member initially suspected a missing global installation of **Ninja**, but later clarified that a global install is indeed the correct setup, suggesting the problem lies elsewhere.
- **Kernel Producer/Consumer Puzzle**: A new member requested resources to understand the **producer/consumer model** used in **matmul kernels**, seeking explanations ranging from basic to intermediate level.
   - No solutions or links were provided.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1377060360900841595)** (2 messages): 

> `QAT Hyperparameters, TorchTune Experiments, QAT Dataset sensitivity` 


- **QAT Performance depends on Hyperparameters and Datasets**: A member found that the effectiveness of **Quantization Aware Training (QAT)** heavily depends on the [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_optimization) used and the specific [dataset](https://en.wikipedia.org/wiki/Dataset) involved.
- **Landing PR before fully understanding the recipe**: A member suggested merging the [Pull Request (PR)](https://en.wikipedia.org/wiki/Pull_request) first, with plans to investigate the specific differences between the two recipes later.
   - The member expressed appreciation for the assistance received.


  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1377349446471389255)** (1 messages): 

> `Grouped Latent Attention, Liger-Kernel Implementation` 


- **Grouped Latent Attention Issue Logged**: A new issue was created on the **Liger-Kernel** GitHub repository regarding **grouped latent attention**; it can be found [here](https://github.com/linkedin/Liger-Kernel/issues/734).
- **Implementation Consideration Initiated**: A member indicated they would review the implementation details related to the newly logged **grouped latent attention** issue.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1377044478455447552)** (2 messages): 

> `NVIDIA Virtual Connect, Sparse Attention Trade-offs` 


- **NVIDIA Hosts Virtual Connect with CUDA Experts**: **NVIDIA** is hosting a [Virtual Connect with Experts](https://gateway.on24.com/wcc/experience/elitenvidiabrill/1640195/4823520/nvidia-webinar-connect-with-experts) event tomorrow, **May 28 at 10am PT**, with **Wen-mei Hwu** and **Izzat El Hajj**, authors of **Programming Massively Parallel Processors**.
   - Attendees can learn from experienced educators, ask questions about their book, their **CUDA** journey, and what to expect in the 5th edition of **PMPP** (coming out this Dec).
- **Sparse Attention Trade-offs on the Frontier**: A new paper called [The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs](https://arxiv.org/abs/2504.17768) was released.
   - The release was followed up by a google meet link, [https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122](https://meet.google.com/wdk-yipf-zjd?authuser=0&hs=122).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1377197005549338624)** (1 messages): 

> `KernelLLM, Hardware Specific Tools, Project Popcorn` 


- **Popcorn Project Patron Pops In!**: A member arriving from the **Project Popcorn** page expressed their enthusiasm and willingness to contribute.
   - The user wondered if an **agentic KernelLLM** is being considered, given the numerous hardware-specific "tools" available.
- **No other topics in this channel**: There were no other detailed technical discussions in this channel.
   - No further topics were extractable.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1377182042932117564)** (1 messages): 

> `matmul.cu, Producer/Consumer model` 


- **Matmul.cu: Explanation Requested**: A member asked for an explanation of the **producer/consumer model** used in **matmul.cu**.
   - The member is looking for basic to intermediate resources to understand what's happening in the kernel.
- **matmul.cu**: The user wants to understand the **producer/consumer model** in the kernel.
   - They are looking for resources ranging from basic to intermediate level to assist in their understanding.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1377009152366477402)** (2 messages): 

> `Learning to Reason without External Rewards, Scalability Concerns` 


- **Reasoning without Reward Raises Eyebrows**: A member shared a link to a paper titled [Learning to Reason without External Rewards](https://www.arxiv.org/abs/2505.19590) and expressed skepticism about its scalability.
   - They found it *hard imagining that will scale* to real-world reasoning scenarios.
- **Doubts Emerge Over Reasoning Scalability**: The discussion centers on whether a reward-free reasoning approach, as presented in the paper, can effectively scale to complex, real-world situations.
   - The core concern revolves around the practical applicability and limitations of the proposed method in handling intricate reasoning tasks.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1377000613283627164)** (19 messages🔥): 

> `amd-mixture-of-experts leaderboard, amd-mla-decode leaderboard, amd-fp8-mm leaderboard, histogram leaderboard, grayscale leaderboard` 


- **MI300 Cracks Mixture of Experts**: A user achieved a personal best of **286 ms** on **MI300** on the `amd-mixture-of-experts` leaderboard.
   - Another user achieved 7th place on **MI300** at **17.8 ms**, while another got a personal best of **113 ms**.
- **Decoding MLA on AMD: A Speedy Saga**: A user secured 7th place on **MI300** with a time of **135 ms** on the `amd-mla-decode` leaderboard.
   - Another submission reached 6th place on **MI300** with a time of **131 ms**.
- **FP8-MM Leaderboard Lights Up with Lightning Speeds**: One user clinched first place on **MI300** with a blazing **116 µs** on the `amd-fp8-mm` leaderboard.
   - Another user hit a personal best on **MI300** at **1133 µs**.
- **Histogram Heats Up H100**: A user achieved 6th place on **H100** with a time of **46.6 µs** on the `histogram` leaderboard.
- **Grayscale Gains Ground on A100**: A user achieved multiple personal bests on **A100**, ultimately reaching **3.08 ms** on the `grayscale` leaderboard.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1377132753958731789)** (11 messages🔥): 

> `Project Contributions, Ablation Studies, Meeting Notes, A2A Integration, Colab Notebook` 


- ****Contributors Seek Project Entry Points****: A member expressed interest in contributing to the project, inquiring about planned **ablation studies** for the methodologies.
   - Another member indicated that fine-tuning open-source models is not currently a focus, but creating an issue to improve the client and add more models and APIs is a *good first issue*.
- ****Clarification on Ablation Context****: A member clarified that their ablation inquiry pertained to the **prompt/agent loop**, such as dropping the long-term memory summary or changing the scaffolding, rather than fine-tuning.
   - The member committed to reviewing the issues on GitHub and expressed interest in seeing the latest meeting notes.
- ****Meeting Notes Shared After Absence****: After apologizing for missing a meeting, a member shared a link to the [meeting notes](https://otter.ai/u/oY942RuHXTuZR7QY98ZhLhgNm9g) on Otter.ai.
   - They acknowledged being *too busy or dread-filled* to address their self-assigned issues but committed to following along on GitHub.
- ****FLE Progress Update in Meeting Notes****: According to meeting notes, the conversation began with a discussion by Morton (details unspecified), Neel talked about **A2A integration** (focusing on gym compatibility), and Jack reported nearing completion of a **Colab notebook**.
   - Jack also planned to address an **Import Error** issue raised by Yasaman; the draft [Colab notebook](https://colab.research.google.com/drive/1WeWoZNxP2jF4Wd4FLQkeEHwwCBahbVE1) capable of running FLE was shared.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1377088175402647644)** (5 messages): 

> `KV Cache RoPE, AMD Competition Future, Amd aiter package install` 


- **Streamlining RoPE with Post-Rotation KV Cache**: A user suggested improving efficiency by storing **k** in the **KV cache** *after* applying **RoPE (Rotary Position Embedding)** to reduce computational overhead.
   - By rotating **k** only for the current token's sequence length (1) instead of the entire sequence, the approach could be a potential optimization.
- **AMD Competition Problems to Persist Post-Event**: Organizers plan to keep the competition problems open for submissions even after the event concludes, though without prizes.
   - The problems will persist at [this link](https://github.com/gpu-mode/reference-kernels/blob/0156b809d952e20d3d6ef0c55b28568647b3a89e/problems/amd/mla-decode/reference.py#L109).
- **AMD CK, AITER Package Installation Clarified**: A user mentioned that the package can be installed from inside Python with **AMD CK, AITER** and their own package.
   - This simplifies the setup process for other users looking to leverage these tools.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1377008072270155796)** (43 messages🔥): 

> `MCP Server Business Case, MCP Clients, Glama Indexing, FastMCP Servers, MCP resource indexing` 


- **Building a SaaS MCP: Marketing Opportunity?**: A member sought to build a business case for their company to build an **MCP server** to assist SaaS companies with integrations, anticipating the objection that LLMs can already read their documentation.
   - Another member suggested it is *a great opportunity to surf the hype and sell your API/SaaS as AI-ready*, making it *an easy sell to the marketing team*.
- **MCP Client Quest: Lightweight and Hackable?**: A member expressed interest in finding a **lightweight and hackable desktop MCP client** to build workflows on top of.
   - A [GitHub list of MCP clients](https://github.com/punkpeye/awesome-mcp-clients?tab=readme-ov-file#clients) was shared, but the member noted it could benefit from repo stars/forks stats for sorting.
- **Glama's Server Detection Algorithm: Lagging Behind?**: A member noted that the Glama algorithm for detecting new servers might be falling behind, as their server wasn't listed.
   - Feedback was requested on [a PR to the awesome list](https://github.com/punkpeye/awesome-mcp-servers/pull/911).
- **Python FastMCP Servers: Show me the Auth!**: A member inquired about examples of **Python FastMCP servers** implementing authentication.
   - Another member asked who to address for an **MCP server submission problem** to the Glama.ai index.
- **LLMs.txt: Exposing Content as Resources?**: A member asked if anyone has made an MCP that bridges **llms.txt** to expose its content as resources, and found [MCP-llms-txt](https://github.com/SecretiveShell/MCP-llms-txt)!
   - They noted that some llms.txt are absolutely huge and was curious if someone had tackled processing them, and another added a [PR on the awesome list](https://github.com/punkpeye/awesome-mcp-servers/pull/940).


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1377074421197570149)** (9 messages🔥): 

> `MCP Launch, UI Issues, MCP Agent Proxy, Multiple models` 


- **New MCP Platform Launches**: A member launched a platform to build and ship remote MCPs, available at [ship.leanmcp.com](https://ship.leanmcp.com), aiming to allow users to vibe-code and ship MCPs easily.
   - Early feedback pointed out typical **UI issues** like link problems and email overflow, with the deploy functionality assumed to be a work in progress.
- **MCP Agent Proxy Connects Clients to Servers**: The [MCP Agent Proxy](https://github.com/mashh-lab/mcp-agent-proxy) facilitates connecting any MCP client to any agent server, creating an "**Internet of Agents**" through composable primitives.
   - It supports **Mastra** and **LangGraph**, automatically detecting agent server types and adapting, as detailed in [this YouTube video](https://youtu.be/cGY6w3ZZB-4).
- **Chatting with Multiple Models**: A member has been using an **MCP Server** to chat with multiple models, finding it super helpful as a tool, using [any-chat-completions-mcp](https://github.com/pyroprompts/any-chat-completions-mcp).
   - Another user noted they wrote their own version, [outsource-mcp](https://github.com/gwbischof/outsource-mcp), and highlighted a feature for **image generation** not found in the other MCP.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1377006730634068019)** (34 messages🔥): 

> `Anthropic Claude voice mode beta, Next-Gen AI Interfaces Beyond Chatbots, j1-nano and j1-micro reward models, UI Prompting Tutorial, DeepSeek-R1-0528` 


- **Claude's Voice Arrives on Mobile**: **Anthropic** launched a beta **voice mode** for **Claude** on mobile, allowing voice interactions for tasks like summarizing calendars or searching documents, currently available in English and rolling out to all plans soon, per [Anthropic's tweet](https://x.com/AnthropicAI/status/1927463559836877214).
- **AI Interfaces: Beyond the Chatbot**: Hiten Shah highlights eight emerging categories of **AI interfaces beyond chatbots**: auto-built UIs, task-driven workflows, canvas interfaces, flow-based builders, custom tools, command-line AI, prompt-less interactions, and new formats in [this tweet](https://x.com/hnshah/status/1927088564166086670?s=46).
   - Examples are shared in [this Clipmate link](https://app.clipmate.ai/public/a9b27f9c-57d3-575f-a7c4-9e29ffdd521b).
- **Tiny Reward Models Pack a Punch**: Leonard Tang open-sourced **j1-nano** (600M parameters) and **j1-micro** (1.7B parameters), small reward models trained in less than a day on a single A100 GPU, as noted in [this post](https://x.com/leonardtang_/status/1927396709870489634).
   - These models use **Self Principled Critique Tuning (SPCT)** to generate instance-specific evaluation criteria, with **j1-micro** rivaling larger models like **Claude-3-Opus** and **GPT-4o-mini**.
- **Prompting UI Design: a Meng-nificent Tutorial**: Meng To released a **44-minute tutorial** on effective **UI prompt engineering**, demonstrating the use of Aura for quick UI generation, leveraging templates, and understanding UI vocabulary, via [this tweet](https://x.com/mengto/status/1925057411439829457?s=46).
- **DeepSeek Model Surfaces, Rumors Swirl**: A new **DeepSeek-R1-0528** model surfaced on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), though it's not the rumored **R2**, Aider gang says preliminary benchmarks show promise regarding quality and price.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1377117772491915265)** (8 messages🔥): 

> `AI Engineer Conference Volunteers, AI Engineer Conference Speakers, Discord Collaboration Project` 


- ****AIE Conference Asks for Army of AI Assistants****: The [AI Engineer Conference](https://xcancel.com/swyx/status/1927558835918545050), scheduled for **June 3-5** in SF, seeks **30-40 volunteers** for logistical support, offering free admission (up to **$1.8k** value) in exchange.
   - Confirmed keynote speakers include **Greg Brockman** (OpenAI), **Sarah Guo** (Conviction), and **Simon Willison**, with **20 miniconferences** planned on various AI engineering topics, plus two leadership tracks.
- ****Discord Diversifies Development Directions****: A member invited others to join a collaborative project on Discord.
   - The project's channel can be found [here](https://discord.com/channels/822583790773862470/1377194898914021417/1377194905515982928), and an example image [here](https://cdn.discordapp.com/attachments/1075282504648511499/1377379230173630596/image.png?ex=6838bfde&is=68376e5e&hm=d57cddedee320068a4c867ac1efa2584cdbb0c5a6a8f707e272e886b9994777b).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1377319740892254329)** (1 messages): 

> `aiDotEngineer World Fair, LlamaIndex booth G11, Jerry Liu talk` 


- **LlamaIndex Heads to AI Fair in June**: The **LlamaIndex** team will be at the @aiDotEngineer World Fair in San Francisco, June 3-5, at **booth G11**.
   - Attendees can chat with **CEO @jerryjliu0** and AI engineers about agents, AI, and **LlamaIndex**.
- **Jerry Liu to Speak at AI Fair**: **Jerry Liu** will be giving a talk on June 5th at the @aiDotEngineer World Fair.
   - More information about his talk is available [here](https://t.co/6T3TwX9qiB).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1377037883910783099)** (22 messages🔥): 

> `ReactJS with LlamaIndex, Human-in-the-Loop (HITL) workflow, RetrieverRouter with RelevancyEvaluator, LlamaCloud credits, SubWorkflows in MainWorkflow` 


- **ReactJS HITL Workflow Woes?**: A member is seeking advice on integrating **ReactJS** with **LlamaIndex** for a **Human-in-the-Loop (HITL)** workflow, questioning the complexity of `ctx.wait_for_event()` and WebSocket communication.
   - A simpler approach of triggering another `run()` on the Workflow with an updated context was suggested.
- **Office Hours Example Shines Light on HITL**: The LlamaIndex team coded an example of **HITL** in two flavors during the last community office hours, responding directly when the HITL is requested (i.e. websocket) and responding later once a response from the human is received by serializing the context and resuming the workflow.
   - The [example can be found on Colab](https://colab.research.google.com/drive/1zQWEmwA_Yeo7Hic8Ykn1MHQ8Apz25AZf?usp=sharing).
- **Evaluating RetrieverRouter Relevance**: A member created a workflow that asks two knowledge bases via a **RetrieverRouter** with two different indexes and a reranker, and wanted to implement a relevancy evaluation retry.
   - They are concerned that always retrieving the same nodes in new attempts wastes user time, and are asking if they should add information to the original query to change the retrieved nodes.
- **LlamaCloud Credits: Subscription Salvation?**: A member inquired about purchasing credits on **LlamaCloud** without a subscription.
   - Another member said a starter subscription gives **50K credits** instantly, and after that it's pay-as-you-go till **500K**.
- **SubWorkflow Context Confusion**: A member reported issues managing **SubWorkflows** running inside a **MainWorkflow**, where context switches and tracing break after running a **SubWorkflow** multiple times.
   - They provided links to the [DroidAgent](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/droid/droid_agent.py), [Planner](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/planner/planner_agent.py), and [Codeact](https://github.com/droidrun/droidrun/blob/main/droidrun/agent/codeact/codeact_agent.py) **SubWorkflows**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1377176769513390212)** (6 messages): 

> `Map function in Mojo, Kapa AI usage` 


- **`map` to the rescue, appends 5 to 2D array**: A member asked for code to use the `map` function to convert `[[1,2],[3,4]]` to `[[1,2,5],[3,4,5]]` and another user provided [an example using Mojo](https://github.com/modularml/mojo).
   - The code defines a `main` function that initializes a 2D list and uses `map` with a capturing function `append_five` to append the value **5** to each sublist; a user noted that *using `map` isn't all that common at the moment since iterators aren't fully integrated everywhere*.
- **Kapa AI, summoned by letters, not full names**: A member inquired about using [Kapa AI](https://www.kapa.ai/) and another user pointed out that to get a reply from Kapa AI, you need to type the first few letters (e.g., `kap`) and then select Kapa AI from the drop-down list.
   - They added that typing the name in full doesn't work, and admitted, *I found this out the hard way, back in the day; for the first month I thought that Kapa AI was deliberately ignoring me*.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1377005852661252309)** (10 messages🔥): 

> `Migrating from Magic to Pixi, uv vs pixi, Conda support, Bootstrapping the ecosystem, Reaching for established C libraries` 


- **Migrating from Magic to Pixi**: A member provided more information regarding migration from **Magic** to **Pixi** in [this forum post](https://forum.modular.com/t/migrating-from-magic-to-pixi/1530).
- **Pixi officially chosen over uv**: While non scientific computing users may prefer **uv**, the official choice of **Pixi** has solid reasons behind it.
   - According to one member it would be even better if someone could explain why **uv** support wasn’t selected.
- **Pixi leverages uv and is targeting Mojo**: **Pixi** uses **uv** under the hood for **Python** deps, according to [this blog post](https://prefix.dev/blog/uv_in_pixi), and makes sense for what **Mojo** is targeting.
   - One member stated that *Mojo supports heterogenous compute, and part of that is going to be a heterogeneous stack of languages, mixing Python, Rust, C, C++, and Mojo. Pixi / conda are built for exactly*.
- **Conda Support boosts Mojo adoption**: **Conda** support makes adoption so easy, and should speed up bootstrapping the ecosystem.
   - One member added **zlib** from conda-forge and wrote some thin bindings and stated that *the same tool will be really easy for my users (who all use conda) to install since they just need to add the modular channel. LOVE the conda support*.
- **Established C libraries support**: One member can see themselves reaching for established **C libraries** like **OpenSSL** until the **Mojo** ecosystem matures more.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1377043655902367977)** (5 messages): 

> `tinygrad.org hyperlink broken, GPU recommendations, tinygrad/tinyxxx` 


- **Optypes Hyperlink Broken on tinygrad.org**: A member reported that the **Optypes** hyperlink on the [tinygrad.org](https://tinygrad.org) site results in a *404 - page not found* error.
   - This is likely due to the recent *moving uops into a dir* changes.
- **Tinyxxx Gets Merged**: George Hotz linked to the [tinygrad/tinyxxx](https://github.com/tinygrad/tinyxxx) GitHub repository.
   - A member later confirmed that a [related pull request](https://github.com/tinygrad/tinyxxx/pull/27) has been merged.
- **Community Seeks GPU Recommendations**: A community member asked for *any recommendations for GPUs to work with*?
   - Unfortunately, no recommendations are provided in the excerpt.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1377303419836170270)** (5 messages): 

> `CPU backend threads, max_pool1d` 


- **tinygrad CPU backend no threads**: A member asked about the way to specify the amount of threads used by the CPU backend and another member responded that *there's no threads, it's just loops in CPU*.
   - To view the kernels, they suggested using `DEBUG=4` or `NOOPT=1 DEBUG=4` to get a cleaner look.
- **max_pool2d works on 1d too**: A member asked if there's a reason there's no `max_pool1d` in Tinygrad, and another member answered that [`max_pool2d` likely works on 1d too](https://docs.tinygrad.org/tensor/ops/?h=max_pool2d#tinygrad.Tensor.max_pool2d).


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1377027438751645776)** (2 messages): 

> `API Errors, New API users` 


- **API Error baffles Newbie**: A new API user reported getting an **Error 400** due to an *"invalid request: message must be at least 1 token long or tool results must be specified."*
   - The user confessed to being completely new to using APIs.
- **API struggles for a newbie**: One user expressed challenges using the API.
   - The user mentioned that they are completely new to using APIs.


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1377364604203700306)** (2 messages): 

> `AI Voice & Conversational Systems, Automation & Workflow Engineering, No-Code/Low-Code Platforms, AI Agents & LLM Workflows` 


- **AI Agent Expert Joins Community**: An AI, automation, workflow, and agent expert with hands-on experience building **LLM-powered systems**, **no-code/low-code products**, and **voice AI solutions** has joined the Cohere Discord server.
   - They focus on creating **intelligent agents**, **scalable automations**, and **full-stack MVPs** using modern AI and visual tools.
- **AI Voice & Conversational Systems Expertise Highlighted**: The expert shared their skills in **VAPI**, **Bland AI**, **Retell AI**, **Twilio**, and **Telnyx** for dynamic voice agents, having built smart voicebots for lead gen, support, and scheduling with real-time memory and context.
   - They have also integrated **LLMs** with **phone systems** and **CRMs** for personalized voice experiences.
- **Automation & Workflow Engineering Prowess Disclosed**: The expert has constructed automations using **n8n**, **Make.com**, and **Zapier** across CRM, email, and AI pipelines, specializing in API-based workflow design using webhooks and cloud services.
   - They have connected **AI agents** with tools like **LangChain**, **Xano**, and **Backendless**.
- **No-Code/Low-Code Platform Skills Enumerated**: The expert is proficient in **Glide**, **FlutterFlow**, **Softr**, **Bubble**, **Xano**, **AppSheet**, **WeWeb**, and **Airtable**, delivering full MVPs with visual frontends, API logic, and scalable backends.
   - They have automated **Stripe payments**, **email flows**, and **database logic** without code.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1377049427117605005)** (3 messages): 

> `Kobold, RP, New Interface, Friendship, Dev Life` 


- **User Awaits Interface to Outpace Kobold's RP Focus**: One user expressed that **Kobold** focuses too much on **RP** and is waiting for someone to release a new interface to address this.
   - There were no further details given as to why or which interface.
- **Developer Seeks Meaningful Connections**: A developer shared a personal story about a lost friendship and is seeking a developer (Polish or European, American) who values deep, genuine connections to collaborate with.
   - The developer is looking for someone who believes in trust, teamwork, and building something meaningful together -- or even a *normal friend with a normal idea who wants extra income*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1377387244314427453)** (1 messages): 

> `AgentX Submission Deadline, AgentX Prizes, AgentX Entrepreneurship Track, AgentX Research Track, Agentic AI Summit` 


- **AgentX Submissions Closing Soon!**: The deadline for **AgentX submissions** is fast approaching, set for **May 31st at 11:59 PM PT**.
   - *Don't miss out!* Submit your projects via the provided links: [Entrepreneurship Track](https://forms.gle/FJTC4jd197bNeJJ96) and [Research Track](https://forms.gle/5dccciawydCZ8o4A8).
- **AgentX Prize Pool Swells to $150,000+**: The **AgentX competition** boasts over **$150,000 in prizes**, including cash awards, credits, and gift cards.
   - Sponsors include industry giants such as **Amazon**, **Auth0/Okta**, **Groq**, **Hugging Face**, **Google**, **Lambda**, **Foundry**, **Mistral AI**, **NobelEra Group**, **Schmidt Sciences**, and **Writer**.
- **Entrepreneurship Track Checklist**: For the **Entrepreneurship Track**, submissions must include a **pitch deck** (≤20 slides), a **product demo video** (max 3 min), and a **live product link**.
- **Research Track Checklist**: The **Research Track** requires a **scientific paper** (7-8 pages max excluding appendix), a **video presentation** (max 3 min), and a **GitHub repository**.
- **Agentic AI Summit to Host Demo Day & Awards**: The **Demo Day and Awards** ceremony will be held at the **Agentic AI Summit on August 2nd at Berkeley**.
   - Participants needing assistance can direct questions to the designated channel.


  

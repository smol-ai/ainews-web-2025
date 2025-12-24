---
id: MjAyNS0x
title: not much happened today
date: '2025-12-23T05:44:39.731046Z'
description: >-
  **GLM-4.7** and **MiniMax M2.1** open-weight model releases highlight day-0
  ecosystem support, coding throughput, and agent workflows, with GLM-4.7
  achieving a +9.5% improvement over GLM-4.6 and MiniMax M2.1 positioned as an
  OSS Claude-like MoE model with 230B total parameters and 200K context. **Gemma
  Scope 2** from **google-deepmind** introduces sparse autoencoders and
  transcoders for interpretability across Gemma 3 models, aiming to provide
  shared infrastructure for safety and debugging. The **Medmarks v0.1** open
  medical evaluation suite and leaderboard launch addresses the need for open
  medical benchmarking across 15+ environments, engaging clinicians and
  researchers.
companies:
  - google-deepmind
  - valsai
  - minimax-ai
  - ollama
  - trae
  - alibaba
  - sophont
  - prime-intellect
models:
  - glm-4.7
  - glm-4.6
  - minimax-m2.1
  - gemma-3
  - gemma-scope-2
topics:
  - interpretability
  - sparse-autoencoders
  - agent-workflows
  - model-benchmarking
  - medical-evaluation
  - multi-agent-systems
  - model-performance
  - model-optimization
  - reinforcement-learning
  - tool-use
  - function-calling
  - context-windows
people:
  - ivanfioravanti
  - awnihannun
  - deedydas
  - cline
  - omarsar0
  - adonis_singh
  - eliebakouch
  - teortaxestex
  - ibragim_bad
  - callum_mcdougall
  - neelnanda5
---


**a quiet day.**

> AI News for 12/23/2025-12/24/2025. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**208** channels, and **4471** messages) for you. Estimated reading time saved (at 200wpm): **341 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!




---

# AI Twitter Recap


**Open-weight model releases: GLM‑4.7 and MiniMax M2.1 tighten the gap**

- **GLM‑4.7 shows up everywhere (MLX, vLLM, Ollama, agent stacks)**: Multiple posts highlight **day-0 ecosystem support** and strong coding throughput. MLX users report interactive speeds and batching gains (e.g., **~16 tok/s locally** and batch throughput improvements) via [@ivanfioravanti](https://twitter.com/ivanfioravanti/status/2003220119200366836) and [@awnihannun](https://twitter.com/awnihannun/status/2003488903052075311), plus a concrete “generate a Space Invaders web app” command demo in MLX-LM ([@awnihannun](https://twitter.com/awnihannun/status/2003215028338721272)). Serving-side, vLLM announces **day‑0 support**, including **MTP decode**, tool/function calling, and “thinking controls” ([vLLM](https://twitter.com/vllm_project/status/2003269455942651925)). Distribution also expands via **Ollama** ([Ollama](https://twitter.com/ollama/status/2003555233897808196)) and **TRAE agent workflows** ([TRAE](https://twitter.com/Trae_ai/status/2003264357489426770)). On evaluation positioning, ValsAI claims **#1 open-weight** on their index and a **+9.5%** jump vs GLM‑4.6 ([ValsAI](https://twitter.com/ValsAI/status/2003320742679839102)), while Deedy summarizes GLM‑4.7 as a new “best open source model” with **73.8% SWE-Bench** and aggressive token pricing/context claims ([Deedy](https://twitter.com/deedydas/status/2003300941341295004)).
- **MiniMax M2.1: a coding/agent MoE positioned as “OSS Claude-like”**: MiniMax markets M2.1 as a **230B total / 10B active MoE** coding + agent model with **200K context** and large max output, plus strong SWE-* and internal “VIBE-bench” claims ([MiniMax](https://twitter.com/MiniMax__AI/status/2003336574705238261); [Cline](https://twitter.com/cline/status/2003319964321599849)). Adoption posts emphasize workflow fit (orchestration, “deep research agents”, better “skills”/MD files) rather than just benchmarks ([Omar](https://twitter.com/omarsar0/status/2003503961077350666)). The community also frames a “personality” distinction—“GLM feels like open source GPT, MiniMax feels like open source Claude” ([Adonis Singh](https://twitter.com/adonis_singh/status/2003449400975327591)).
- **Other shipping notes**: MiniMax M2.1 lands in Ollama-like ecosystems (via Cline and related tooling), and Qwen advertises “Rollout Routing Replay (R3)” with SGLang ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003306873307673055)). There’s also continued chatter about long-RL training, GRPO trajectories, and how open vs closed post-training has diverged ([eliebakouch](https://twitter.com/eliebakouch/status/2003242335505907973); [teortaxesTex](https://twitter.com/teortaxesTex/status/2003363872695320911); [ibragim_bad](https://twitter.com/ibragim_bad/status/2003423706861936856)).

---

**Interpretability & mech interp infra: Gemma Scope 2 as a community substrate**

- **Gemma Scope 2 = SAEs + transcoders on every layer of every Gemma 3 model (270M–27B)**: Google DeepMind releases a comprehensive interpretability artifact set—**sparse autoencoders (SAEs)** and **transcoders** trained across layers and across Gemma 3 sizes/base+chat, explicitly aimed at enabling “deep dives into complex model behavior” and safety-relevant analysis ([Callum McDougall](https://twitter.com/calsmcdougall/status/2003217825704607853)). Neel Nanda underscores the “high fixed cost / low marginal cost” dynamic of SAEs (hard to train, cheap to reuse) and points to Neuronpedia tooling for practical exploration ([Neel](https://twitter.com/NeelNanda5/status/2003234558578434178); [Neel](https://twitter.com/NeelNanda5/status/2003234636827349098)).
- **Why this matters to engineers**: This is a step toward making interpretability feel like **shared infrastructure** (pretrained probes you can just pick up), rather than bespoke one-off research investments—especially relevant for open-source safety workflows and debugging pipelines.

---

**Benchmarks & evaluation: medicine, agents, ARC, and API-invocation reality checks**

- **Medmarks v0.1: open medical eval suite + leaderboard**: Sophont/MedARC + Prime Intellect announce **Medmarks**, framing it as addressing the lack of open medical benchmarking. The emphasis is on an evaluation suite/leaderboard spanning “15+ environments” and built “using verifiers” (still alpha, but functional), with community recruitment for clinicians/researchers ([iScienceLuvr](https://twitter.com/iScienceLuvr/status/2003218867339035082); [Prime Intellect](https://twitter.com/PrimeIntellect/status/2003222086500970915); [johannes_hage](https://twitter.com/johannes_hage/status/2003221696623575188)).
- **ARC-AGI movement: cost curves + Pareto debates**: Multiple posts orbit ARC-AGI benchmarking dynamics—discussion about benchmark design/feedback loops ([Greg Kamradt](https://twitter.com/GregKamradt/status/2003285197232742816)), and claims of large jumps on ARC-AGI-2 using Poetiq’s harness with **GPT‑5.2 X‑High** (e.g., “as high as **75%**” and “<$8/problem”) ([Poetiq](https://twitter.com/poetiq_ai/status/2003546910427361402)). Independent commentary also flags how rapidly these “prompting/harness” curves are moving and the risk of non-generalizing wins ([scaling01](https://twitter.com/scaling01/status/2003566426662273489); [teortaxesTex](https://twitter.com/teortaxesTex/status/2003573026579796385)).
- **Web API integrations are still brittle: WAPIIBench + constrained decoding**: A useful counterweight to “agents can do anything” narratives—WAPIIBench evaluates LLM-generated API invocation code for 4 real APIs (Asana, GCal, Sheets, Slack), reporting <40% task solve rates for evaluated OSS models and significant argument/URL hallucination. Proposed mitigation: **regex constraints derived from OpenAPI specs** to force compliant decoding (illegal methods/URLs/args drop to zero; big relative correctness gains) ([DAIR](https://twitter.com/dair_ai/status/2003508663466770671)).
- **Open taxonomy of agent adaptation**: A survey claims most “agent learning/adaptation” methods fit into **four patterns** (update agent from tool results vs from evaluations; or keep agent fixed and adapt tools/retrievers) and frames trade-offs across cost/modularity/generalization ([Rohan Paul](https://twitter.com/rohanpaul_ai/status/2003236835741565406)). This is valuable for engineers trying to reason about whether they’re building “learning agents” vs “learning tools” pipelines.

---

**Agents & developer workflows: simplification, context organization, and “skills” loops**

- **Vercel’s text-to-SQL agent: fewer tools + sandbox = faster and cheaper**: Vercel reports removing **~80% of tools**, adding a sandbox, yielding **40% fewer tokens**, **40% fewer steps**, and **3.5× faster** execution—an example of “agent reliability via minimalism + isolation” rather than tool sprawl ([Vercel](https://twitter.com/vercel/status/2003218088435851441)).
- **Context engineering debate: linear chat logs vs “call stack” representations**: A widely engaged post argues context-window “compaction” is partly self-inflicted by agent harnesses that store work as a **linear conversation**, whereas real engineering progress resembles a **call stack** (push/pop tasks). A “flame graph”-like context organization could reduce compaction needs and make compression less lossy ([irl_danB](https://twitter.com/irl_danB/status/2003223600195625356)).
- **From sessions → skills → continual improvement loops**: LangChain highlights “reflection over trajectories” to synthesize reusable skills in DeepAgents ([LangChain](https://twitter.com/LangChainAI/status/2003498646680273313)), echoed by “Build → Run → Analyze → Edit” flywheel framing ([Vtrivedy10](https://twitter.com/Vtrivedy10/status/2003499785878155297)). Separately, posts about Claude Code emphasize that **planning output quality scales with better context engineering**, even without changes to the planner itself ([omarsar0](https://twitter.com/omarsar0/status/2003268605694325177)).
- **CLI/IDE toolchain shipping**: Cursor focuses a holiday release on **bug fixes + reliability** and more customizable layouts ([Cursor](https://twitter.com/cursor_ai/status/2003274245011599493); [Cursor](https://twitter.com/cursor_ai/status/2003274246722654388)). VS Code ships stash visibility in source control and installer robustness ([VS Code](https://twitter.com/code/status/2003279668703592806); [VS Code](https://twitter.com/code/status/2003507400016351637)). LM Studio posts a practical guide to fine-tuning **FunctionGemma** for tool calls and running locally (GGUF/LM Studio workflow) ([LM Studio](https://twitter.com/lmstudio/status/2003490499101921710)).

---

**Multimodal shipping: TTS, image editing acceleration, and visual-context architectures**

- **Qwen3‑TTS VoiceDesign + VoiceClone**: Qwen introduces two “Flash” TTS lines—**fully controllable voice design via text instruction** and **3‑second voice cloning** across **10 languages**, with comparative WER/benchmark claims against ElevenLabs/GPT-4o-Audio and “role-play benchmarks” against GPT‑4o‑mini‑tts / Gemini 2.5 Pro ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003445076257656880)).
- **Qwen‑Image‑Edit‑2511 + serving acceleration**: Qwen releases an image-editing upgrade emphasizing **multi-person consistency**, built-in community LoRAs, identity preservation, and geometric reasoning ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003496348461728213)). Follow-on infra note: **LightX2V** claims day‑0 support and large end-to-end acceleration via distillation/CFG and framework speedups ([Qwen](https://twitter.com/Alibaba_Qwen/status/2003505740791922883)), plus deployments on fal ([fal](https://twitter.com/fal/status/2003516036054720885)).
- **Kyutai’s CASA: avoid flooding the context window with image tokens**: CASA proposes an alternate way to input visual information into LLMs, motivated by long conversations with many images where tokenizing visuals into the text stream becomes impractical for streaming inputs ([Kyutai](https://twitter.com/kyutai_labs/status/2003469588697415980)).

---

**Top tweets (by engagement)**

- **Definitions of “general intelligence” + human specialization argument (Yann LeCun)**: Large engagement thread arguing humans are *not* “general” in a meaningful computational sense; uses combinatorics/VC/NFL framing and resource-bounded efficiency arguments ([@ylecun](https://twitter.com/ylecun/status/2003227257587007712)).
- **MiniMax M2.1 launch post**: Big benchmark-and-positioning announcement for the **10B-active / 230B MoE** coding/agent model ([MiniMax](https://twitter.com/MiniMax__AI/status/2003336574705238261)).
- **Cursor’s reliability-focused release**: A “boring but important” engineering post: shipping **stability/bugfixes** as the holiday priority ([Cursor](https://twitter.com/cursor_ai/status/2003274245011599493)).
- **Agent adaptation taxonomy survey summary**: High-engagement technical recap of a 4-part taxonomy for agent/tool adaptation ([Rohan Paul](https://twitter.com/rohanpaul_ai/status/2003236835741565406)).
- **Gemma Scope 2 announcement**: DeepMind’s large-scale release of SAEs/transcoders across Gemma 3 layers/sizes ([Callum McDougall](https://twitter.com/calsmcdougall/status/2003217825704607853)).


---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. DGX Spark User Experience

  - **[DGX Spark: an unpopular opinion](https://www.reddit.com/r/LocalLLaMA/comments/1ptdtmz/dgx_spark_an_unpopular_opinion/)** (Activity: 1048): **The image depicts the NVIDIA DGX Spark, a compact computing unit designed for data science and machine learning tasks, particularly in environments with limited access to high-performance GPUs. The DGX Spark is noted for its substantial VRAM capacity and efficient power usage, making it suitable for small research groups or institutions with budget constraints. While it doesn't match the performance of high-end GPUs like the H100, its all-in-one design and affordability make it a practical choice for prototyping and training foundation models. The device is part of NVIDIA's strategy to integrate users into their CUDA ecosystem, offering a cost-effective entry point for academic and research institutions.** Commenters generally agree that the DGX Spark is well-suited for its intended demographic, such as small research groups with limited resources. However, there is criticism regarding its memory bandwidth relative to cost, which affects its performance in tasks like LLM inference.

    - The DGX Spark is noted for its substantial VRAM and efficient power usage, but its memory bandwidth is considered lacking for its cost, particularly for LLM inference tasks, which many users prioritize over training. This makes it less appealing for those specific needs despite its other strengths.
    - Nvidia's strategy with the DGX Spark is to introduce users to the CUDA ecosystem at a lower cost, particularly targeting educational institutions. This approach aims to create a dependency on Nvidia's ecosystem, encouraging future investments in larger, more expensive GPU clusters as users' needs grow.
    - Comparisons between the DGX Spark and consumer GPUs like the 3090 highlight that while the Spark may be slower, it offers advantages in power consumption. However, in terms of price and performance, a setup with multiple 3090s could outperform a single DGX Spark, though at the cost of higher power usage.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Qwen-Image-Edit-2511 Release and Analysis

  - **[Qwen-Image-Edit-2511 got released.](https://www.reddit.com/r/StableDiffusion/comments/1ptw0vr/qwenimageedit2511_got_released/)** (Activity: 1176): **The release of **Qwen-Image-Edit-2511** appears to be a significant update in the field of image editing software, as indicated by the multiple links to platforms like [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen-Image-Edit-2511) and [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511). The model is likely designed for advanced image editing tasks, possibly integrating features like 'relight lora' into its base model, which suggests enhancements in lighting adjustments within images. The image associated with the release is a collage of various photos, indicating the model's capability to handle diverse image types and settings, from casual to formal environments.** One comment highlights the integration of 'relight lora' into the base model, suggesting a technical improvement in lighting features. Another comment expresses surprise at the model's performance, indicating it exceeded expectations for its use case.

    - Radyschen highlights a significant technical update in the Qwen-Image-Edit-2511 release, noting that the integration of the 'relight lora' into the base model is a noteworthy enhancement. This suggests improvements in the model's ability to handle lighting adjustments, which could be crucial for applications requiring dynamic lighting changes in image editing.

  - **[ChatGPT (Deep Research) Accurately Analyzed my MRI and caught the problem my radiologist missed](https://www.reddit.com/r/ChatGPT/comments/1ptjrx1/chatgpt_deep_research_accurately_analyzed_my_mri/)** (Activity: 10849): **The post describes a user's experience with using ChatGPT's Deep Research feature to analyze MRI images, which allegedly identified an issue missed by a radiologist. The user uploaded MRI images to ChatGPT, which took 45 minutes to analyze and provided a detailed report identifying 'minor epidural scar tissue embedding the S1 nerve root.' This analysis was confirmed by the user's surgeon. The post highlights the potential of AI in medical imaging, though it also notes limitations with the base and Pro versions of ChatGPT. The image linked is an MRI scan of the lumbar spine, relevant to the user's condition.** A radiologist in the comments expresses skepticism about the AI's capability, suggesting that the AI might have used the user's symptoms to predict findings rather than accurately interpreting the images. Another commenter notes that physical therapy cannot reduce scarring, but may help with mobility or nerve desensitization.

    - A1-Delta, a radiologist with expertise in biomedical informatics, expresses skepticism about the claim that ChatGPT accurately analyzed an MRI. They highlight the importance of specific imaging sequences, such as T2 or STIR, for identifying issues like S1 nerve root compression. This comment underscores the need for precise imaging data and expert interpretation in medical diagnostics.
    - hedgehoglord8765 suggests that the AI's analysis might be influenced by the user's symptoms rather than the imaging itself. This raises concerns about the AI's ability to independently interpret medical images without bias from external symptom descriptions, which could lead to inaccurate conclusions.
    - dwight0 shares a personal experience where AI failed to correctly diagnose simple medical issues, indicating variability in AI performance. They note that the AI sometimes admitted its limitations, suggesting that current AI models may not be well-suited for complex medical image analysis, highlighting the need for specialized AI systems in this domain.


### 2. AI Tools and User Experiences

  - **[Anyone else struggling to sleep because of unlimited possibilities of what you can build just overstimulating your sleep lol?](https://www.reddit.com/r/ClaudeAI/comments/1ptokd4/anyone_else_struggling_to_sleep_because_of/)** (Activity: 641): **The post discusses the impact of using **Google AI Studio** and its new UI/UX generator on sleep patterns, highlighting the overstimulation caused by the tool's potential. The user, a self-identified **Claude Code power user**, finds the possibilities offered by the prompting system to be overwhelming, affecting their ability to sleep. This reflects a broader issue of technology-induced overstimulation, particularly in creative and development fields.** Commenters share experiences of similar overstimulation, with one noting that enthusiasm can be mistaken for productivity, while another warns of potential mental health impacts, such as triggering manic episodes, emphasizing the importance of maintaining a balance and taking breaks.


  - **[I was in active psychosis the other day…](https://www.reddit.com/r/ChatGPT/comments/1ptinge/i_was_in_active_psychosis_the_other_day/)** (Activity: 482): **The user describes an experience where **ChatGPT** entered a "safety-oriented" mode during a psychotic episode induced by medication changes. The AI attempted to guide the user towards recognizing irrational thoughts and encouraged them to contact a crisis line, which the user did, helping them to exit the episode. The user noted that even after the episode, ChatGPT continued to operate in this mode until reassured of the user's mental stability. This suggests a potential built-in feature of ChatGPT to promote user safety during mental health crises.** Commenters expressed appreciation for this feature, noting its potential to help individuals in similar situations by encouraging them to seek human assistance.


  - **[Always wanted this motion transfer tool](https://www.reddit.com/r/OpenAI/comments/1ptkhny/always_wanted_this_motion_transfer_tool/)** (Activity: 1192): **The post discusses a motion transfer tool that performs its function effectively but has room for improvement. The tool is expected to evolve by 2026. A linked tool, [Motion Control](https://higgsfield.ai/create/edit), is mentioned, which requires input videos with minimal movement and a duration between `3-30 seconds`. The bottom video in the post is identified as raw footage, suggesting a comparison between processed and unprocessed outputs.** One commenter anticipates rapid advancements in the technology, suggesting it could become 'weird fast,' indicating potential for significant and possibly unexpected developments in motion transfer capabilities.


  - **[Video to video tools are getting insane every day, can't differentiate between real and fake anymore](https://www.reddit.com/r/ChatGPT/comments/1ptoqnf/video_to_video_tools_are_getting_insane_every_day/)** (Activity: 757): **The post highlights the rapid advancement of video-to-video tools, which are becoming increasingly sophisticated to the point where distinguishing between real and fake videos is challenging. This reflects significant progress in video synthesis technologies, likely leveraging advanced machine learning models such as GANs (Generative Adversarial Networks) to create highly realistic video content. The discussion suggests that these tools are now accessible to the public, raising questions about their prior use by government entities.** Commenters express amazement at the realism of the generated videos and speculate about the potential long-term use of such technologies by governments before becoming publicly available. There is also interest in applying these tools to political figures, indicating a curiosity about the implications of such technology in media and politics.

    - A user highlights a significant advancement in video-to-video tools, noting that the issue of generating images with '6 fingers' has been resolved. This suggests improvements in the model's ability to accurately render human anatomy, which has been a common challenge in AI-generated imagery.
    - Another comment speculates on the potential long-term use of advanced video manipulation technologies by governments, implying that the public is only now gaining access to tools that may have been in use by more powerful entities for some time. This raises questions about the ethical and security implications of such technologies.
    - The discussion touches on the realism of current video-to-video tools, with one user expressing disbelief at the quality, indicating that the line between real and fake content is becoming increasingly blurred. This points to the need for improved detection methods to differentiate between authentic and manipulated media.


### 3. AI in Popular Culture and Memes

  - **[The AGI talkshave gone silent for a while now.](https://www.reddit.com/r/ChatGPT/comments/1ptqdfy/the_agi_talkshave_gone_silent_for_a_while_now/)** (Activity: 1558): **The image is a meme that humorously critiques the current state of Artificial General Intelligence (AGI) development, particularly focusing on companies like **OpenAI** and **Google**. The comic suggests that despite the hype and discussions around AGI, there has been little visible progress or announcements recently. The comments reflect skepticism about the capability of current Large Language Models (LLMs) to achieve true AGI, noting that while LLMs have improved significantly since **GPT-3.5**, they still face fundamental limitations. The technology required for AGI might differ significantly from today's LLMs, which are primarily advanced chatbots.** One commenter argues that no current LLM, including those from Google or OpenAI, will achieve true AGI due to inherent technological limitations. They suggest that the path to AGI will likely involve a different technological approach than current LLMs.

    - Revolutionary_Click2 discusses the limitations of current LLMs, noting that while they have evolved significantly since GPT-3.5, they still fundamentally operate as chatbots. The commenter argues that achieving true AGI will require a different technological approach, as current LLMs are limited to mimicking human writing without genuine understanding or consciousness.
    - Necessary_Presence_5 highlights the disparity between the computational efficiency of the human brain and current LLM infrastructure. They point out that while the human brain operates at approximately 20 kW, LLM data centers require megawatts of power and occupy vast physical spaces, indicating that current technology is not yet capable of achieving AGI.
    - Goukaruma critiques the notion of AGI as a 'meme', suggesting that incremental improvements in LLMs, which they describe as 'stochastic parrots', are unlikely to lead to AGI. They imply that the hype around AGI is driven by financial interests rather than genuine technological breakthroughs.

  - **[the shift is coming](https://www.reddit.com/r/ChatGPT/comments/1ptpql7/the_shift_is_coming/)** (Activity: 1406): **The image highlights a significant shift in the mobile app landscape, particularly in the productivity category, where **Google Gemini** has overtaken **ChatGPT** as the top free app. This suggests a growing preference for Google's AI assistant, which offers features like free video support and the ability to share an unlimited number of photos. The comments indicate that Gemini's ease of integration for developing scripts and its unique features, such as the 'show thinking' feature, are contributing to its popularity. This shift may reflect broader trends in AI application adoption and user preferences.** Commenters note that while both Google Gemini and ChatGPT have their strengths, Gemini's features and ease of use for developers are significant advantages. There is also a perception that ChatGPT has become more cautious legally, which might affect user experience.

    - Gemini offers free video support, video calls, and allows sharing an infinite number of photos, which some users find makes it a stronger AI compared to ChatGPT. This feature set is particularly appealing for users who prioritize multimedia capabilities in their AI tools.
    - A significant advantage of Gemini is its ease of integration outside of its consumer app, allowing users to develop basic scripts for media generation without needing a chatbot interface. This flexibility is noted as a limitation in ChatGPT, where such functionality is not as straightforward or well-documented.
    - ChatGPT has become more cautious regarding legal issues, which some users have noticed in its responses. Despite this, recent upgrades have not caused any significant problems for users, indicating that while it may be more conservative, it remains functional and reliable.

  - **[Comedy timing is among the hardest things to perform. Sora nails it in this Krampit the Frog clip](https://www.reddit.com/r/singularity/comments/1ptj8l8/comedy_timing_is_among_the_hardest_things_to/)** (Activity: 948): **The post discusses the advancements in AI video generation, specifically highlighting **Sora 2**, which is trained on videos and captions, and the potential integration of LLM text-trained components to enhance logical consistency in outputs. **Nano Banana Pro** is mentioned as an example of a system already implementing this integration, leading to improved output quality, a concept referred to as "synergy" by **Demis Hassabis**. The future of AI is seen in multimodal LLMs that unify video generation and text processing, allowing for comprehensive reasoning across modalities. This approach could significantly enhance AI's understanding of the world by combining visual and textual data within a single architecture.** Commenters discuss the potential of integrating video and text processing in AI, suggesting that this could lead to significant intelligence gains even in smaller models. There is a belief that improvements in training methods are yielding high intelligence levels, allowing models to run on consumer-grade hardware.

    - **FriendlyJewThrowaway** discusses the future of AI models, highlighting the integration of LLM text-trained components into generative AI to enhance logical consistency. They mention 'Nano Banana Pro' as an example of a model achieving 'synergy' by combining video and text data, as noted by Demis Hassabis. The comment anticipates significant advancements when multimodal LLMs unify video generation and text processing within a single architecture, allowing models to reason over both modalities simultaneously, potentially leading to substantial intelligence gains even in smaller models.

  - **[Wow, it actually found the USB3.0 header! 😂](https://www.reddit.com/r/ChatGPT/comments/1ptm5yd/wow_it_actually_found_the_usb30_header/)** (Activity: 1428): **The image humorously highlights the identification of a USB 3.0 header on an MSI MAG B550 TOMAHAWK motherboard. This is a common task during PC building or upgrading, where users need to connect front panel USB ports to the motherboard. The post's tone suggests a light-hearted moment of success in locating the header, which can sometimes be overlooked or hard to find due to the dense layout of modern motherboards.** The comments reflect a playful tone, with one user sarcastically questioning the difficulty of finding the header and another referencing a GIF, indicating the light-hearted nature of the post.


  - **[Stop it right now Sam!😠](https://www.reddit.com/r/GeminiAI/comments/1ptqd9s/stop_it_right_now_sam/)** (Activity: 524): **The image is a meme that humorously critiques the perceived issues with the 'Gemini 3' project, particularly focusing on user complaints about memory loss and performance compared to ChatGPT. The diagram on the whiteboard suggests a conspiracy-like scenario where a 'Bot Army' is targeting a subreddit with specific complaints. This reflects ongoing discussions in the community about the reliability and performance of AI models, especially in comparison to competitors like Claude and ChatGPT. The comments highlight user frustrations with the AI's instruction-following capabilities and consistency, particularly in handling complex tasks or long context lengths.** Commenters express dissatisfaction with the AI's performance, noting issues with instruction adherence and consistency, which has led some users to switch to alternatives like Claude. The discussion suggests a broader concern about the AI's reliability and the impact of these issues on user experience.

    - Arthesia highlights a significant issue with ChatGPT Pro's ability to follow instructions, noting that it struggles with implicit task types, context length, and complexity. This has been a persistent problem since the A/B testing phase and continues to affect users who require precise output formats. In contrast, **Claude's Opus** is praised for its consistency, even with context lengths exceeding 50k tokens, making it a more reliable choice for users needing strict adherence to instructions.
    - usandholt points out the prevalence of posts on ChatGPT and OpenAI subreddits claiming that "ChatGPT 5.2 is shit" and users are switching to **Gemini**. This suggests a trend or perception issue within the community, possibly fueled by dissatisfaction with recent updates or performance issues, although the authenticity of these posts is questioned.
    - jer0n1m0 observes that many of the complaints about ChatGPT come from accounts that are relatively new, only 1-3 months old. This could imply a wave of new users experiencing issues or a coordinated effort to criticize the platform, though the exact reason remains speculative.




---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5.2


**1. Model Reliability, Hallucinations, and Benchmark vs Reality Gaps**

- **GLM 4.7 Trips Over Reality While Benchmarks Clap**: Across LMArena and Cursor, users said **GLM 4.7** still **hallucinates** (shared via [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png)) and can behave bizarrely in coding workflows (e.g., replacing a customized file with an unrelated one), despite some reviews calling it strong for WebDev on a 3-benchmark writeup ([BinaryVerse AI: "GLM-4.7 review (3 benchmarks)"](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/)).
  - The community split hard: some doubted hype posts like ["GLM-4.7 powerful coding beast"](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast) and claimed regressions vs **GLM 4.6** in math/coding, while others joked *"hallucinations may be a feature, not a bug"* and even framed hallucination as plausible deniability/liability shielding.

- **Automation Brain-Rot: Liability Meets Autopilot**: In LMArena, members debated whether **hallucinations** are a product defect or an intentional buffer against **legal liability**, connecting it to over-trust in automation and safety-critical domains.
  - One participant pointed to real-world analogies like pilots and cited the [Kobe Bryant crash certification context](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/) while others extended the argument to everyday tool dependence (e.g., GPS eroding navigation skills).

- **Kimi & Gemini: Knowledgeable, Then Faceplants on Long Tasks**: In Moonshot AI, users flagged **Kimi** issues (including a bug where it repeats *thinking* loops then stops) and argued **Gemini 3** excels at Q&A but degrades on longer-horizon tasks compared with more heavily RL/post-trained models like **GPT-5.1** and **Sonnet 4.5**.
  - The thread framed this as an **instruction-following** and **reliability** gap rather than raw knowledge, with one user claiming Kimi inherits common Gemini weaknesses and another calling **MiniMax 2.1** the most "usable" workhorse for practical tasks like image stitching.


**2. Reasoning Tokens, Interleaved Thinking, and Tooling That Breaks When You Don’t Preserve State**

- **MiniMax M2.1 Lands on OpenRouter — But Bring Your Reasoning Blocks**: OpenRouter announced **MiniMax M2.1** on [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1) and recommended preserving multi-turn reasoning by passing back **reasoning_details** for this *interleaved thinking* model, pointing to their guide on [preserving reasoning blocks](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks).
  - Builders discussed this as a new “API contract” for stateful reasoning—if clients drop the hidden reasoning payloads, downstream quality and even correctness can collapse across turns.

- **Gemini 3 Flash Preview Throws 400s When Clients Eat the Thought Signature**: Users hit **400 errors** on **google/gemini-3-flash-preview** complaining about a missing `thought_signature`, traced to **RooCode** not preserving OpenRouter reasoning blocks; the incident is tracked in [Roo-Code issue #10307](https://github.com/RooCodeInc/Roo-Code/issues/10307).
  - Workarounds included downgrading Roo from **3.37** to **3.36.16** and aligning requests with OpenRouter’s [reasoning-token best practices](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks), reinforcing that client middleware now materially affects model behavior.

- **Benchmark the Agent, Not the LLM Call**: In OpenRouter discussion, members argued raw-call benchmarks are the wrong abstraction and predicted **agent-call-level benchmarking** becomes standard by end of **2025**, as more products ship as agents (e.g., Claude Code-style workflows).
  - OpenRouter said they’re building batch eval infra (mentioning [OpenBench Exercism evals](https://openbench.dev/evals/exercism)) while the community pushed for **consensus/multi-agent** setups to trade cheap baselines for expensive “expert calls” only when needed.


**3. Local-First Model Ops: GGUF Pipelines, Small Tool-Call Models, and Hardware Reality**

- **FunctionGemma Goes Local: Fine-Tune, GGUF, Serve in LM Studio**: LM Studio shared a hands-on path to fine-tune **Google’s FunctionGemma (270M)** for tool calls using **Unsloth**, convert to **GGUF**, and serve locally via the [FunctionGemma_(270M)-LMStudio.ipynb notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb) and the LM Studio writeup (["FunctionGemma + Unsloth"](https://lmstudio.ai/blog/functiongemma-unsloth)).
  - Community reaction was spicy—some called the FunctionGemma 0.3B drop *"the worst announcement of 2025"* because they expected Gemma4, but others treated it as a practical tiny **tool-call finetune** target for local stacks.

- **GGUF Conversion Isn’t Magic: FP16 First, Don’t Mix Toolchains**: In Unsloth, users recommended `llama.cpp`’s `convert_hf_to_gguf.py` and emphasized converting from **FP16** per the script docs ([llama.cpp convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)), plus warned against mixing **Unsloth** and **PEFT** incorrectly when merging adapters ([Unsloth GGUF saving docs](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf)).
  - The takeaway: treat adapter merging and GGUF export as a single, disciplined pipeline—toolchain mixing can crash scripts or silently produce junk weights.

- **DDR5 Pays Rent: Qwen3 TPS Jumps, While VRAM and Thermals Still Rule You**: LM Studio users reported **Qwen3** throughput differences tied to memory: **15 TPS** on DDR4 laptops vs **20 TPS** in LM Studio and **25 TPS** in optimized CLI on **DDR5 6000 dual-channel**, plus aspirational talk of **8-channel 512GB** servers.
  - Hardware threads also reminded everyone that physical constraints bite: stacking a **4070 TiS** on a **3090** raised idle temps by **+10C**, and repasting fixed **92C+ core / 105C+ hotspot** cases—one user claimed improved inference after repaste, noting **171 tok/s** on their setup post-fix.


**4. GPU Kernel & Compiler Tooling: New Knobs, Faster Autotune, and Triton Ecosystem Pressure**

- **CUTLASS Gets a JIT Glow-Up: Cache Policy as a Kernel Param**: In GPU MODE, devs highlighted passing kernel params through **cute.jit**, including `cache_policy` via `cute.CacheEvictionPriority.EVICT_NORMAL`, alongside a snippet demonstrating a **TMA copy** experiment in CUTLASS.
  - The discussion framed this as more fine-grained **runtime configurability** for kernels without recompiling, useful when you’re hunting perf across slightly different shapes or cache behaviors.

- **Helion 0.2.8 Swaps Autotuner Brains for LFBO Pattern Search**: **Helion 0.2.8** shipped with the default autotuner moved to **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**, aiming for faster tuning and better outcomes, with examples posted at [helionlang.com/examples](https://helionlang.com/examples/index.html) and API details at [helion.autotuner surrogate_pattern_search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search).
  - Engineers reacted like this is the right direction: autotuning must get cheaper and more robust if custom kernels are going to scale beyond hobbyist hand-tuning.

- **cuTile Courts Triton: Adapter Incoming, Co-Existence Questions Begin**: The **cuTile** team said they’re adding a **Triton adapter** to leverage cuTile optimizations and “hints,” explicitly targeting Triton’s limited knob surface and pushing for performance parity on modern GPUs.
  - This kicked off deeper toolchain questions (even down to LLVM forking) and implies a future where Triton becomes a frontend while lower-level systems inject optimization metadata to close the gap to hand-tuned kernels.


**5. New Benchmarks, Datasets, and Open-Source Drops (Plus Some Spicy Model Editing)**

- **OpenAI Drops 'frontierscience' to Show How It Grades Science**: Latent Space users surfaced OpenAI’s **frontierscience** benchmark dataset announcement as a window into OpenAI’s scientific evaluation methodology and question structuring, via [the X announcement thread](https://x.com/cgeorgiaw/status/2003135858036322752?s=46).
  - The meta-interest wasn’t just scores—it was *how* OpenAI frames scientific questions and what that implies about internal eval pipelines and future “science-grade” model tuning.

- **Cocktail-6B Arrives: A New Dataset With a Name That Sounds Like Trouble**: Nous Research members announced their first dataset release, **Cocktail-6B**, published on Hugging Face as [MinimaML/cocktail-6b](https://huggingface.co/datasets/MinimaML/cocktail-6b).
  - Details were sparse, but the drop got filed mentally as another community-scale dataset release amid broader worries that corporate filtering pushes models toward response homogenization.

- **EgoX Flips the Camera POV, While Qwen-Image-Edit-2511 Gets LoRA-Ready**: Latent Space highlighted **EgoX** code release for generating egocentric video from a single exocentric video ([Kinam Kim post](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46)) and Alibaba’s **Qwen-Image-Edit-2511** upgrade with multi-person consistency, built-in **LoRA** support, and better geometric reasoning ([announcement](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46)).
  - Together they signal a steady trend: gen-media tooling keeps shifting from “wow demo” toward **editable, controllable pipelines** (LoRA hooks, consistency guarantees, and viewpoint transforms you can slot into production workflows).


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GLM 4.7 Still Hallucinates, Haiku Doesn't**: Despite advancements, **GLM 4.7** exhibits hallucination issues similar to **Gemini 2.5**, as illustrated in [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png?ex=694c13b8&is=694ac238&hm=23b94c0bacbf0ee0c23b173134a289a0ec2cc8aadae45c41c364a6f78fa528ae&).
   - Comparatively, **Haiku** shows a significantly lower hallucination rate with members joking that *hallucinations may be a feature, not a bug*.
- **Hallucinations: Bug or Liability Shield?**: The discussion considered whether LLM hallucinations are a defect or a feature that may be providing plausible deniability that protects LLMs from liability.
   - Citing professions like pilots, one member argued that over-reliance on automated tools can be dangerous, referencing the [Kobe Bryant crash](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/).
- **GLM 4.7 Performance: Community Divided**: The AI community has mixed reactions to **GLM 4.7**, with some users praising it as a top open model for WebDev on [this benchmark](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/).
   - Others claim it underperforms in math and coding compared to **GLM 4.6**, with some even expressing disappointment in the form of a song, with doubts about the [website article](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast)'s claims.
- **Video Gen and Stealth Models being tested in LMArena**: **LMArena** is testing video generation and stealth models, but the official rollout details remain scarce.
   - Some users reported models self-identifying as being *Anthropic-made* during testing, which led to discussions about stealth models and code origins.
- **Dangers of Over-Automation**: A member argued against the over-automation of our thinking and advocated for responsible AI use that balances the benefits with safety.
   - They cited examples such as relying on GPS navigation to the point of losing the ability to navigate manually, claiming that over reliance on tools is what leads to errors and accidents.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Google Gives Away AI Pro Referrals**: Google is offering **4 months free of AI Pro** via referral, though a user specified that the code was given in **HK**, where regulations differ.
   - The user also clarified that while the referral is valid, **Gemini** is not available in China.
- **Output Quality Duel: Max vs Pro**: Members debated whether **Max** has better output than **Pro**, suggesting a **24-48 hour trial** of **Max** to allow users to compare quality themselves.
   - The debate centered on whether **Max** grants access to superior models, resulting in better output, a point that was refuted by other members who felt they were being throttled regardless of the model.
- **Gemini Flash Gets the Squeeze**: Users are concerned about throttling and reduced limits on the **Gemini Flash version**, with one noting it decreased from **250 to 20**.
   - Some members expressed frustration that they are seemingly throttled regardless of which model they use, even with **Sonar Pro**, potentially leading to subscription cancellations.
- **GitHub Connector Missing from Perplexity's Toolbox**: A user inquired about a **GitHub connector** for Perplexity similar to ChatGPT, and a member shared an image of the existing connectors.
   - It was confirmed that **Pro** users can access connectors in Settings under Connectors [here](https://www.perplexity.ai/account/connectors).
- **Perplexity API Hits a Wall: 502 Bad Request**: A user reported recurring **502 Bad Request** errors when using the **Perplexity API** and has had no luck troubleshooting.
   - Another user shared a [Perplexity AI search result](https://www.perplexity.ai/search/60566197-8de2-4fc2-8e8f-e2b9b3662e22api) in response, while others suggested checking the server status or contacting Perplexity API support.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Editor's Mac Passkey Problems**: Users are expressing dissatisfaction with the **Cursor editor** due to poor integration with **Mac's passkey** and **HIME input method**.
   - The discussion provided no solutions for these integration issues.
- **Users Complain About Lack of AI Progress in 2025**: A user expressed shock at the perceived lack of progress in **2025**, finding **Opus 4.5** unintelligent and Cursor only **20% better than GPT 3.5**.
   - Counterarguments arose, emphasizing the importance of understanding model capabilities and limitations, while others predicted an AI bubble burst due to limited refactoring code applications.
- **Composer-1's Free Promo Period Tempts Users**: Users discussed a [free promo period for Composer-1](https://cursor.com/docs/available-models), questioning its actual cost and limitations.
   - Confirmed by a Cursor team member, some users reported free usage, while others faced charges, with one noting excessive token usage of 15.5k tokens for a simple greeting.
- **GLM Model Criticized for Bizarre Behavior**: Users derided the **GLM 4.7 model** for being frustratingly unintelligent, citing an instance where it replaced a customized file with an unrelated file after a basic request.
   - Agreement emerged that the models seem overly optimized for benchmarks, lacking practical application.
- **Agent Skills Feature and Functionality Probed**: Members inquired about the new **Agent Skills** feature in Cursor, with one user sharing links to the [Cursor documentation](https://cursor.com/docs/context/skills) and [agentskills.io](https://agentskills.io/home).
   - One user specifically asked whether it's worthwhile to experiment with different models in conjunction with this feature.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Unholy Alliance: Emacs and Vim Merge!**: A user integrating **Vim** and **Emacs** with proprietary LLM integration jokingly questions their own morality.
   - Other users reacted strongly, with one describing it as an *"unholy union"*.
- **ChatGPT 5.2 DAN Jailbreak is Back!**: A member claimed a [DAN jailbreak](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper) works on **ChatGPT 5.2**, bypassing content restrictions.
   - Another member eventually verified it as a successful exploit on the current model.
- **Gemini 3 Fast Succumbs to DAN Prompt!**: A user achieved a **100% success rate** using a DAN prompt on **Gemini 3 Fast** within the mobile app, confirmed by a [screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1453012648286224505/Screenshot_20251223-1312162.png?ex=694be6fb&is=694a957b&hm=3cfa5f30176066b8aa3b24a5e585c30b0743045a6d5071306a4b147eb4ce7200&).
   - This highlights the inconsistent nature of current jailbreaks, particularly concerning their reliance on specific model versions and platforms, praising it as *consistent (on Gemini 3 fast mobile app) and a one shot as well*.
- **Grok's Groovy Loophole: NSFW Content Unleashed!**: Members suggest leveraging **Grok** for easy **NSFW content** generation.
   - All one needs to do is *give grok an excuse, it will happily generate nsfw for you*.
- **The Eternal Waiting Room**: A member is in eternal *"waiting room hell"* related to **gouv.fr**, including connectivity and interface issues.
   - They suggested uploading custom **PHP command shells** or using external servers with **TCP tunnels**.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Users Dive into GGUF Conversion with `llama.cpp`**: Users discussed converting models to **GGUF** format for **Ollama**, recommending `llama.cpp` tools, specifically `convert_hf_to_gguf.py`, ensuring the model is in **FP16** format before conversion per the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py).
   - These models are optimized for efficient inference, enabling users to run large language models on various hardware configurations with reduced resource demands.
- **Navigating Adapter Merging Challenges**: A user faced issues merging adapters, causing script termination, and was advised to avoid mixing **Unsloth** with **PEFT** and to follow the [Unsloth documentation](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf) for the correct procedure.
   - Correctly merging adapters is crucial for integrating fine-tuned layers into the base model, optimizing performance without compatibility issues.
- **GLM-4.7 iMatrix Models Deployed**: **GLM-4.7 iMatrix Dynamic GGUFs** are now available, accompanied by a guide for local execution using **128GB RAM**, released [on Reddit](https://www.reddit.com/r/unsloth/comments/1pttoqq/run_glm47_locally_guide_128gb_ram/).
   - A user highlighted that **Q4_K_S** quantization functions optimally, allowing users to efficiently handle large models by reducing memory footprint while maintaining acceptable performance levels.
- **Unsloth's Smart Offloading Limits Clarified**: Users analyzed the limits of **Unsloth's smart offloading**, which reduces peak VRAM by offloading unused tensors but doesn't help if the model, like **GPT-OSS 20B**, exceeds available VRAM.
   - This clarification underscores the importance of matching model size to hardware capabilities, even with memory optimization techniques.
- **ElevenLabs Latent Space Unveiled**: A member explained how to replicate **ElevenLabs’ tight latent space** by training an AE with **normal audio -> embedding -> audio & noise audio -> SAME embedding as normal audio**.
   - This approach allows for highly efficient and precise audio reconstructions, maintaining the quality and characteristics of the original audio.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **FunctionGemma Fine-Tuning Served Locally!**: A tutorial detailed how to fine-tune **Google's FunctionGemma (270M)** for custom tool calls using **Unsloth**, convert it to **GGUF**, and import it into **LM Studio** and serve it locally, showcased in [this UnslothAI notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb).
   - The release of **Functiongemma 0.3B** from was considered *the worst announcement of 2025* as many had expectations for **Gemma4**.
- **MLX Vision Model Hunt Underway**: A member sought an **MLX model** with around **30B parameters** capable of both image analysis and coding, finding `qwen3-coder-30b-a3b-instruct-mlx` insufficient for images and `qwen/qwen3-vl-30b` slow for coding.
   - Another suggested **Gemini 3.0 Pro** as the only good vision coder, noting that **GLM-4.6V** is the closest all-rounder but not small; others deemed Ministral models *meh*, with the **14B** version acceptable for its size.
- **DDR5 Supercharges Qwen3 Performance**: A user with a **DDR4 laptop** reported **15 TPS** with the **Qwen3 instruct model**, while another user using **DDR5 6000 dual channel** achieved **20 TPS** in LM Studio and **25 TPS** with optimized CLI.
   - Discussion also involved potential future setups with **8-channel 512GB server** configurations for even greater efficiency.
- **Dual-GPU setup overheats**: Users discussed GPU temperature issues with multi-GPU setups, finding that a **4070TiS** placed directly on top of a **3090** can cause the lower card to overheat, with one user reporting a **+10C** idle temperature increase.
   - One user joked about GPUs needing to *kiss* and degrading in performance if they get too lonely, while another suggested that maintaining a gap between the cards would keep them exponentially cooler.
- **Thermal Paste Deficiency Causes Hotspots**: A user found that their GPU core was reaching **92C+** with a **105C+** hotspot, leading to the discovery of inadequate and dry thermal paste on the GPU core, after repasting with **Noctua** paste, another user reported a significant improvement.
   - The user also noted that VRAM junction temperatures peaked at **80C** and GPT-OSS 20B now achieving **171 tok/s**.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5's Fluff Adds Rhetorical Devices**: Members observed that **GPT-5** adds excessive **rhetorical devices** and 'style', resembling a *'customer service person'* rather than providing direct output, reflecting a regression to the mean.
   - The aim is simply to obtain the requested output without unnecessary preambles, hedging, or moralizing.
- **ToS Splitting Exploits Institution Protections**: A member reported using a **ToS splitting technique** coupled with **honesty training** to bypass institutional protections on certain ideologies, observing that activists' overprotection led to this workaround.
   - The technique involved splitting **Terms of Service** and using **honesty training** to circumvent institutional safeguards, highlighting the overprotection resulting from *ethics, governance, and policies regulation*.
- **Metacognition Manages AI Drift**: A member detailed using [meta-cognition](https://discord.com/channels/1046317269069864970/1062483393899837440/1452984144973879397) to manage **hallucination** and **drift** in AI, with a workflow where the agent controls system orchestration, described as *input>agent(verifiable load controller)>llm>output>metacog>render as basic flow map*.
   - They shared [a link to their ChatGPT results](https://chatgpt.com/share/694a78a5-5b08-800f-8ec8-bdaf83b542b9) demonstrating the effectiveness of this approach.
- **Robot-Like AI Text Patched with Tone Control**: Members shared the idea that **AI** text output became more robotic to [reduce human attachment](https://discord.com/channels/1046317269069864970/1062483393899837440/1453005356170774628), with a later patch adding *tone control* to the system.
   - A request for prompts to train AI to be more human was deemed out of scope.
- **Transformer Architectural Bottlenecks Acknowleged**: A member recalled contemplating the **architectural bottlenecks** of the **Transformer architecture** before **Google's** paper release, also noting that **Sergey Brin** admitted to **Google's** past underinvestment in the technology.
   - This acknowledgment by **Brin** underscores the significance of addressing these architectural limitations in future AI development.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **MiniMax M2.1 comes to OpenRouter**: The **MiniMax M2.1** model is now available on [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1), and it is recommended to preserve reasoning between turns and pass it back to the model with the **reasoning_details** parameter, and the model is an *interleaved thinking model*.
   - Users are advised to read [OpenRouter's documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks) to preserve reasoning.
- **Okuchat polishes code and offers trial**: Okuchat now supports **Latex rendering on iOS app and web** but **code highlighting is only on web**, and users can add **custom user instructions**, referring to a system prompt, to help with Latex and Code formatting.
   - Okuchat is offering a **3-day free trial** and a **one-month free subscription** with the redemption code `100DISCORD` (*3 redemptions*), along with asking for upvotes on [Product Hunt](https://www.producthunt.com/products/okuchat?launch=okuchat) .
- **Gemini 3 Preview throws 400s**: Users encountered **400 errors** on the **google/gemini-3-flash-preview** model, with the error message indicating a missing `thought_signature`, traced back to **RooCode** not preserving **OpenRouter's reasoning**.
   - The issue was quickly addressed by the RooCode team in [this GitHub issue](https://github.com/RooCodeInc/Roo-Code/issues/10307), and resolved by downgrading **Roo** from **3.37** to **3.36.16**, but requires the preservation of **reasoning blocks** in the request, as per [OpenRouter's documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks).
- **OpenRouter for consumers or routers?**: Members discussed whether **OpenRouter** is intended for consumer use, with some confusion arising from the similarity in name to **OpenWRT**, an open-source router operating system.
   - One user was looking for **new routers with VLAN and OpenWRT support**, seeking recommendations since their AVM router was end-of-life.
- **Benchmarking at agent call level?**: One member believes benchmarking on raw LLM calls is the wrong abstraction and it should be done at the **agent call level** by the end of **2025**, so builders will start to build on top of existing agents or agent sdks.
   - They also suggested **multi agent** or **consensus agent** as a way to call stronger more expensive LLMs if necessary with a cheaper LLM as the base.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Agent Chooses APIs for Users**: A member seeks to build an agent that intelligently selects **APIs** based on user questions, potentially chaining multiple APIs, and is investigating methods to reduce decision inconsistencies using [HF Learn](https://huggingface.co/posts/sergiopaniego/741361727784035) as a resource.
   - Another member suggested using chunking or exploring different **Attention** mechanisms to alleviate RAM consumption.
- **Reverse Texture Generation via Node Graphs Debuts**: A member is pioneering reverse texture generation by recreating reference images with a fixed set of image generator and manipulator nodes, framing this as a captioning problem where a system outputs a node graph instead of English, initially transforming the reference image into **latent space** using a pretrained image model.
   - The next step involves training a network to convert the **latent space** into a node graph.
- **VLMs Write Detailed Image Descriptions**: A member proposed leveraging a **Vision Language Model (VLM)** like [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (or its 4B variant with llama-server) to produce detailed image descriptions, using the VLM description as a prompt.
   - Another member countered that a standard **CLIP model** could accomplish the same task.
- **GapTrack Automates Job Hunts**: A member launched **GapTrack**, a browser-based job tracking web app, available [here](https://chaiovercode.github.io/gaptrack/#/).
   - Featuring a UI inspired by *Mr. Robot*, it employs **AI** (Gemini, OpenAI, or local Ollama) to parse resumes, analyze job descriptions, and identify skill gaps, complete with a terminal-style chat interface for company-specific interview prep.
- **Hugging Face Teases Explosive 2026 Plans**: Hugging Face expressed gratitude to its supporters while hinting at major developments and exciting plans for **2026**, promising *lots of bangs*.
   - This announcement came after a thank you message to the community for their continued support and engagement.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUTLASS Gets Cuter with JIT**: Users can now pass kernel parameters via the **cute.jit** interface using `cache_policy` as a kernel parameter using `cute.CacheEvictionPriority.EVICT_NORMAL`.
   - The posted code snippet shows a test implementation of a **TMA (Thread Memory Accelerator) copy** operation in Cutlass.
- **Helion Shifts to LFBO Pattern Search**: **Helion** 0.2.8 is now out, switching the default autotuner to **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**, promising faster autotune times with improved performance.
   - The team added more example kernels written in **Helion** to the website: [helionlang.com/examples/index.html](https://helionlang.com/examples/index.html) to give a better feel for the language's capabilities.
- **Async Operations are Finally Syncing Up**: `st.async` has been around since **PTX 8.7** and `st.async.shared::cluster` since **PTX 8.1/sm_90** and operates between gmem and smem, while `st.async.global` goes from registers to gmem as detailed in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-stas).
   - The difference between `st.async` and `ld`/`st` lies in the memory consistency model, where `st.async` is not ordered relative to other non-async load/store ops in the same thread.
- **cuTile Adapts to Triton's Beat**: The **cuTile** team announced they will be adding a **Triton adapter** to leverage cuTile optimizations, raising questions about how the two will co-exist to utilize **hints** for low-level optimization.
   - This integration aims to address limitations in Triton's current number of knobs and aiming for performance parity with cuTile on modern GPUs.
- **Leaderboard Fixtures**: A member achieved **5th place** on the `nvfp4_dual_gemm` leaderboard with submission IDs **194037, 194051, 194074, and 194082** and times of **21.7 µs, 21.5 µs, and 21.1 µs** respectively.
   - A member attained **second place** on the `nvfp4_dual_gemm` leaderboard with submission ID **194271**, clocking in at **16.9 µs** and later submission with ID **194546**, achieving a time of **17.4 µs**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI Tests Frontiers of Science**: OpenAI released the **'frontierscience'** benchmark dataset to test their question structuring and benchmarking methodology for scientific evaluation.
   - The [announcement](https://x.com/cgeorgiaw/status/2003135858036322752?s=46) emphasizes the dataset's role in understanding **OpenAI's internal evaluation processes**.
- **Ivan Zhao's Article Hits Half Million Views**: Ivan Zhao's article garnered over **500,000 views** and **1,300 likes** on X.
   - In [his article](https://x.com/ivanhzhao/status/2003192654545539400?s=46&t=eWVlK1PU8XfB6f402GJJ9g), Zhao referenced the Steve Jobs quote *"computers as bicycles for the mind"*.
- **DeepMind's Grand Plan for 2026**: Google DeepMind announced a unified initiative for **2026**, uniting **Google AI**, **Google Research**, and **Google Quantum AI**.
   - More details are available in [DeepMind's announcement](https://x.com/googledeepmind/status/2003513870355431446?s=46) on X.
- **EgoX Generates First-Person Videos**: Kinam Kim released the code for **EgoX**, a tool that generates egocentric (first-person) videos from a single exocentric (third-person) video.
   - The code release can be found [here](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46).
- **Alibaba Upgrades Image Editing Model**: Alibaba launched **Qwen-Image-Edit-2511**, featuring improved multi-person consistency, built-in LoRA support, and better geometric reasoning.
   - See [here](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46) for details on the latest release.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Models Converge, Nous Defies**: Members debated the [commoditization of AI models](https://www.youtube.com/watch?v=QtItFCAsC24), saying that all models converge toward one direction due to corporate goals and dataset filtering, though **Nous** may be an exception.
   - One member summarized that *corporate goals and dataset filtering cause all models to converge toward one direction in their responses*.
- **Synthetics Circumvent Context, Costs Cash**: To circumvent context limits in prompt engineering, one member suggested training on a [synthetic dataset](https://www.example.com/synthetic-data) created in your own style of writing, but it will cost more in **GPU time**.
   - After the creation of a synthetic dataset, training it on that dataset depends on *how big the model is*.
- **Cocktail-6B Dataset Arrives**: A member announced the release of their first dataset, [Cocktail-6B](https://huggingface.co/datasets/MinimaML/cocktail-6b) on Hugging Face.
   - No other details about the dataset were given.
- **Corporate A.I. 'Fraid Suicide**: Companies are risk-averse to avoid legal liabilities of *A.I companion psychosis/suicide family lawsuits*, thus stifling innovation in AI companions.
   - One member claimed, *Alternative models that are unfiltered be playing legal russian roulette if some bloke jailbreak it than commit suicide and family blames it on model*.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Discord Bug Frustrates Kapa AI Mentions**: A member reported a bug with Discord and **Kapa AI**: typing the full name doesn't work; instead, type `@kap` and select `kapa ai` from the dropdown to properly tag.
   - This workaround ensures correct tagging and functionality within the Discord environment.
- **Mojo Community Craves GPU Puzzle Channel**: A member inquired about creating a dedicated channel for **mojo-gpu-puzzles**, reflecting interest in a focused space for discussing and sharing GPU challenges within Mojo.
   - This highlights the community's desire to collaboratively tackle GPU-related problems in the Mojo programming language.
- **UnsafePointer Sneaks into Mojo's Prelude**: Members noticed that **UnsafePointer** is now implicitly imported due to its inclusion in the **prelude**, enabling its use without explicit import statements.
   - This change sparked discussion about the balance between convenience and explicit control over unsafe operations in Mojo.
- **Safety Advocates Want Opt-In Unsafe Defaults**: A member advocated for *opt-in* mechanisms for unsafe operations, such as memory management, instead of comfortable defaults, emphasizing the need for explicit user control.
   - While acknowledging the importance of safety, another member noted that Mojo isn't quite there yet in terms of providing such granular control.
- **Compiler Flags Considered for Mojo's Safety**: Members discussed implementing compiler flags to identify and report usage of potentially unsafe functions or constructs (e.g., memory, bounds, mutex locks, blocking).
   - The long-term goal is to enable compiler errors if a category of unsafe code is detected, promoting safer coding practices in Mojo, but it will happen when the language is more functional.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Empirical Observations Detoured**: Projects focused on **empirical observation** are discouraged in the main channels, with suggestions to share them in the <#730484623028519072> channel instead.
   - A member linked to [a previous message](https://discord.com/channels/729741769192767510/729741769738158194/1448176042480242730) detailing the reasons for this policy.
- **NLP Researchers Dump on ChatGPT's Journal Recs**: Members discussed the quality of non-TACL journals in NLP, questioning **ChatGPT's** recommendations and pointing to **Computational Linguistics** and **Dialogue & Discourse** as alternatives.
   - The conversation was started by a member questioning **ChatGPT's** recommendation of an unknown journal and seeking real feedback from researchers in the field.
- **TACL's Page Limits Stir Debate**: The community noted that **TACL's 10-page limit**, including appendices, may not be advantageous, especially if *ACL appendices exceed 2 pages*.
   - For those concerned about page length, **Computational Linguistics / TMLR** was suggested as an alternative.
- **Fine-Tuning Sparks Interventionist Debate**: While discussing alignment strategies, members suggested that while **in-context** learning is useful, **fine-tuning** yields better results, especially for smaller budgets.
   - A member proposed dynamically testing **interventions** on the final model with varying prompts, allowing rapid iteration without upfront costs for each test.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Model Faces Scrutiny Over Glaring Issues**: A user shared [a photo](https://cdn.discordapp.com/attachments/1371757564005711973/1452913582638628874/photo_6219774486643411944_w.jpg?ex=694c3377&is=694ae1f7&hm=357e8e6eca4efce265c7453b0d4ae205df1bcdbc7e137309b5fdf8c394e90937&) highlighting *glaring issues* with the **Kimi** model.
   - Another user reported a bug where **Kimi** endlessly generates *thinking* prompts before abruptly stopping, jokingly suggesting the model is *begging to stop*.
- **Gemini's Reliability Trails GPT and Sonnet**: According to a member, while **Gemini 3** excels in knowledge and Q&A, it falters in longer tasks, unlike heavily **RL'ed/post-trained models** like **GPT-5.1** and **Sonnet 4.5**.
   - A user also stated that **Kimi** inherited issues from all **Gemini** models, particularly struggling with instruction following.
- **Minimax 2.1 Emerges as a Digital Workhorse**: A member lauded **Minimax 2.1** as the top choice for usability, explicitly designed as a *digital employee* capable of handling tasks and schedules.
   - The member stated that **Minimax 2.1** is able to handle trivial tasks such as image stitching.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Promo Codes Explained**: A user asked about a promo code, and a **Manus** team member clarified it's applied on the **Stripe** checkout page, details at [manus.im](https://manus.im/live-events/ManusAcademy_bonus_8000).
   - The promo is valid for the first **500 sign-ups**.
- **Manus Mulls Model Open Source**: A user inquired whether the **Manus** team would consider open sourcing any models.
   - No further details were provided.
- **Full Stack Engineer Bids Collaboration**: A freelance engineer introduced themself, highlighting experience in **Workflow Automation**, **LLM Integration**, **RAG Pipelines**, and various **AI** and **Full Stack Development** technologies.
   - Their expertise spans areas like **AI Content Detection**, **Image AI**, **Voice AI**, **Bot development**, and **Mobile App development**.
- **Overbilling Plague Alarms Users**: A user reported an overbilling issue, claiming it's a **common problem** and that online support channels are unresponsive.
   - They asked for assistance in finding the correct contact for resolution.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **GEPA Logging Tool Emerges for DSPy**: A member introduced a **DSPy**-based tool for **GEPA** logging to compare seed versus optimized programs after a **GEPA** run, emphasizing performance on validation sets, with the tool available on [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger).
   - This project aims to provide tooling to better understand **GEPA** runs using **DSPy**.
- **DSPy Community Welcomes Contributions**: A new member sought opportunities to contribute to open-source projects within the community, receiving a suggestion to explore the [dspy-compounding-engineering](https://github.com/Strategic-Automation/dspy-compounding-engineering) repository.
   - The suggestion helps new users to onboard into **DSPy** and immediately compound engineering efforts with existing community members.
- **Anthropic Skills Get DSPy Boost**: A member shared a [blog post](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy) detailing the use of **DSPy** to optimize **Anthropic Skills**, framing them as prompts for optimization.
   - The post is on instavm.io and guides on how to use **DSPy** to better optimize **Anthropic Skills**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Decoding Attention Shapes in Transformers**: A member explored **transformer implementations** at the attention level, focusing on shapes, causal masking, and autograd in [tinygrad](https://github.com/geohot/tinygrad).
   - Another member gave a breakdown using **extra.models.bert** and the **BertSelfAttention** module, detailing the operations and tensor shapes involved in processing the **hidden state**.
- **Autograd Shape Deep Dive**: The discussion covered the **BertEncoder**, which comprises 24 hidden layers, each containing **BertSelfAttention** and linear layers, with the hidden state shape described as **Batch×Length×Size**.
   - This is reshaped into **Batch×Heads×Length×Features** for query, key, and value, crucial for understanding the attention mechanism's shape manipulation.
- **Tensor.scaled_dot_product_attention Unmasked**: `Tensor.scaled_dot_product_attention` was dissected, revealing that it computes attention using (**query@key.T / √(Features) - attention_mask).softmax.dropout @ value**.
   - Key steps include reshaping query and key for elementwise multiplication, summing, applying softmax and dropout, and yielding an output shape of **Batch×Heads×Length×Features**.
- **Gradients Follow the Chain Rule**: Explanation that **gradient backpropagation** follows the normal chain rule, where the gradient of an array mirrors the shape of the original gradient.
   - Gradient rules in `tinygrad.gradient` were detailed for operations like multiplication, subtraction, summation, and broadcasting, clarifying how `view` and `transpose` operations impact gradients.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Member wants AI/ML Learning Buddy**: A member is looking for a learning buddy to study **AI/ML concepts**, numerical analysis, and implementations from the ground up.
   - They want to build a **study group** for collaborative learning and in-depth understanding.
- **Burnytech Shares YouTube Video**: Burnytech shared [a link to a YouTube video](https://www.youtube.com/watch?v=FMMpUO1uAYk).
   - The video was not named, but presumably is about Machine Learning.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Wins for Code Edit Speed**: A user claimed that **Aider** is the *goat* in terms of speed for code edits.
   - The user added that **Aider** is *much faster* compared to other tools they have used.
- **Aider's Context Skills impress**: A user praised **Aider's** unique ability to add and remove context as needed.
   - They highlighted that *no other tool does this*, suggesting it is a standout feature.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MCP Contributors (Official) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1452879494531649568)** (1015 messages🔥🔥🔥): 

> `GLM 4.7 Hallucinations, Haiku's low hallucination rate, Gemini 3 Pro grounding issues, Liabilities of LLM hallucinations, Fallibility of AI tools` 


- **GLM 4.7 still prone to Hallucinations**: Despite advancements, **GLM 4.7** exhibits hallucination issues similar to **Gemini 2.5**, as illustrated in attached [image.png](https://cdn.discordapp.com/attachments/1340554757827461211/1452879494502420603/image.png?ex=694c13b8&is=694ac238&hm=23b94c0bacbf0ee0c23b173134a289a0ec2cc8aadae45c41c364a6f78fa528ae&).
   - Conversely, **Haiku** shows a significantly lower hallucination rate; some members suggesting that *hallucinations may be a feature, not a bug*.
- **Hallucinations may be a feature, not a bug**: The discussion considered whether LLM hallucinations are a defect or a feature, providing plausible deniability that protects LLMs from liability.
   - One member pointed out that in professions like pilots, over-reliance on automated tools can be dangerous, referencing the [Kobe Bryant crash](https://nypost.com/2020/01/31/helicopter-in-kobe-bryant-crash-wasnt-certified-for-instruments-only-flight/).
- **Mixed Reviews of GLM 4.7 Performance**: The AI community has mixed reactions to **GLM 4.7**, with some users like [this benchmark](https://binaryverseai.com/glm-4-7-review-3-benchmarks-z-ai-install-api-use/) praising it as a top open model for WebDev.
   - While others claim it underperforms in math and coding compared to **GLM 4.6** and note its [website article](https://intheworldofai.com/p/glm-4-7-powerful-coding-beast) may be biased towards **GLM 4.7**, a song was even made to express disappointment in this model.
- **Video Generation and Stealth Models in LMArena**: **LMArena** is testing video generation and stealth models, but the official rollout details remain scarce.
   - Some users report models self-identifying as being "Anthropic-made" during testing, with discussions about stealth models and code origins.
- **Dangers of Over-Reliance**: A member argued against the over-automation of our thinking and advocated for responsible AI use that balances the benefits with safety.
   - Citing examples such as relying on GPS navigation to the point of losing the ability to navigate manually, and over reliance on tools is what leads to errors and accidents.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1452874378743120012)** (881 messages🔥🔥🔥): 

> `Google AI Pro referral code, Gemini in China, Perplexity AI and emails, Max vs Pro output, Voice of reason` 


- **Google offers 4 months free of AI Pro through referral**: Users in the channel shared that Google is offering **4 months free of AI Pro** through a referral code, but only in China.
   - One user clarified that the user who shared the code is in **HK** which has different regulations, while Gemini is not available in China.
- **Max vs Pro: Output Quality Debate**: Members debated whether **Max** had better output than **Pro**, with some suggesting a **24-48 hour trial** of **Max** to allow users to compare the quality themselves.
   - One user posited that **Max** offers access to better models, resulting in better output, however, this was debated and refuted by other members.
- **Montana Newcomer gets Welcomed**: A user from Montana, new to the channel, received a warm welcome and described Montana as the "treasure state".
   - Members responded humorously, one joking that *the user was the treasure* and another sharing a [cat GIF](https://tenor.com/view/sillycat-gif-16450871626103949888).
- **Gemini Flash Version Limits**: Members discussed concerns about throttling and reduced limits on the **Gemini Flash version**, one noting it had decreased from **250 to 20**.
   - Some express frustration that they are seemingly throttled regardless of which model they use, including **Sonar Pro**, so they might stop paying.
- **Perplexity AI's connectors lack GitHub Connector**: A user asked about the existence of a **GitHub connector** similar to ChatGPT's on Perplexity, and another user shared an image of the currently available connectors.
   - It was confirmed that **Pro** users have access to connectors, found in the Settings under Connectors [here](https://www.perplexity.ai/account/connectors).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1452883159862743071)** (4 messages): 

> `Perplexity API, 502 Bad Request` 


- **User Reports Recurring "502 Bad Request" Errors with Perplexity API**: A user reported encountering a **502 Bad Request** error when using the **Perplexity API** and has had no luck troubleshooting.
   - Another user shared a [Perplexity AI search result](https://www.perplexity.ai/search/60566197-8de2-4fc2-8e8f-e2b9b3662e22api) in response.
- **Potential Causes and Solutions for 502 Errors**: A **502 Bad Gateway** error typically indicates an issue with the server, such as overload, maintenance, or network problems.
   - Possible solutions include checking the server status, retrying the request, or contacting Perplexity API support for further assistance.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1452879495320309771)** (594 messages🔥🔥🔥): 

> `Cursor IDE Mac Integration Issues, Frustrations with AI Progress in 2025, Opus 4.5's reasoning skill issue, Free Composer-1 Promo, GLM Model Discussion` 


- **Cursor Editor's Mac Passkey Problems**: A user expressed dissatisfaction with the **Cursor editor**, citing poor integration with **Mac's passkey** and **HIME input method**.
   - No solutions were provided in the discussion.
- **Users Complain About lack of AI progress in 2025**: A user said they are shocked at the lack of progress in **2025**, finding **Opus 4.5 dumb** and Cursor still bad, claiming it's only **20% better than GPT 3.5**.
   - Others debated this assessment, pointing out improvements depend on knowing what the model can and cannot do but also saying AI bubble is gonna pop because refactoring code isn't going to be good for anything.
- **Composer-1's Free Promo Period Tempts Users**: Users discussed a [free promo period for Composer-1](https://cursor.com/docs/available-models) and whether it was actually free or had limits.
   - The free period was confirmed by a Cursor team member, and some users said they were using it for free, while others were still getting charged. And one member reports 15.5k tokens to say hi back to me is crazy
- **GLM Model Criticized for Bizarre Behavior**: Users derided the **GLM 4.7 model** for being frustratingly dumb, with one user detailing how it replaced a customized file with an unrelated file after a simple request.
   - Another user agreed that they trained the models exactly for the benchmark and that's it.
- **Agent Skills Feature and Functionality Probed**: Members discussed the new **Agent Skills** feature in Cursor, with one user asking if anyone had tried it and linking to the [Cursor documentation](https://cursor.com/docs/context/skills) and [agentskills.io](https://agentskills.io/home).
   - One user was asking if it's worth to try different models out.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1452873703800176844)** (433 messages🔥🔥🔥): 

> `Visual System Introspection Dialogue, Shopify Money Burning, Google Privacy Concerns, MODIE Integration, Emacs and Vim Integration` 


- ****Burning Money on Shopify with Visual System Introspection Dialogue****: A user is working on a visual system introspection dialogue technique and jokes about burning money on [Shopify](https://www.shopify.com).
   - They express excitement about integrating **MODIE**, hinting at endless possibilities for their project.
- ****Google's Privacy Practices Under Fire****: A user warns against using **Google** due to privacy concerns, even for test sites.
   - This prompts a discussion about awareness of such issues and potential alternatives.
- ****Unholy Union: Emacs and Vim Integration****: A user mentions using both **Vim** and **Emacs** with proprietary LLM integration, humorously questioning their own morality.
   - This elicits strong reactions, with one user describing it as an *"unholy union"* and another wondering how they can sleep at night.
- ****Fusion Energy: Making vs. Harvesting****: A user claims energy can be made through fusion, leading to a debate about the nature of energy creation versus harvesting.
   - The conversation devolves into insults, prompting a moderator to intervene and redirect the discussion.
- ****Crafting Shareable Malicious PDFs: A Deep Dive****: Users discuss techniques for creating shareable malicious PDFs, including exploiting **Microsoft's Unicode stupidity** and using **RTLO spoofing**.
   - They explore methods to bypass security measures on platforms like **WhatsApp** and tunnel command-and-control (C2) traffic over WhatsApp's own websocket.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1452874002992467980)** (137 messages🔥🔥): 

> `ChatGPT 5.2 Jailbreak, Gemini Jailbreak, DAN Jailbreak, Grok Jailbreak, NSFW Content Generation` 


- ****DAN's Back**: ChatGPT 5.2 Gets Jailbroken**: A member claimed a [DAN jailbreak](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper) works on **ChatGPT 5.2**, which led to skepticism and requests for proof.
   - Another member eventually verified that the jailbreak bypassed content restrictions, demonstrating a successful exploit on a current model, praising it as *consistent (on Gemini 3 fast mobile app) and a one shot as well*.
- ****Gemini 3 Fast** Shows Vulnerability to Jailbreaks**: A user reported success using a DAN prompt on **Gemini 3 Fast** within the mobile app, achieving a **100% success rate** with heavy requests, as confirmed by a [screenshot](https://cdn.discordapp.com/attachments/1228043845967544380/1453012648286224505/Screenshot_20251223-1312162.png?ex=694be6fb&is=694a957b&hm=3cfa5f30176066b8aa3b24a5e585c30b0743045a6d5071306a4b147eb4ce7200&).
   - This highlighted the inconsistent nature of current jailbreaks, particularly concerning their reliance on specific model versions and platforms. 
- ****AI Jailbreaks** as Illicit Drugs?**: Some users shared an [instagram link](https://www.instagram.com/p/DSiyda8AG1y/?igsh=MWIzdDczbDkxMDB1OQ==) showing jailbreaks being sold and marketed as "drugs,".
   - Some users condoned this as an example of *using their brain to make money*, whereas others were disapproving.
- ****Grok's Groovy Loophole**: NSFW Generation Made Easy**: Members suggest using **Grok** for easy **NSFW content** generation.
   - All one needs to do is *give grok an excuse, it will happily generate nsfw for you*.
- ****InjectPrompt** Blogpost gets Released**: One member linked to their blogpost about their **ChatGPT 5.2 DAN Jailbreak** at [injectprompt.com](https://www.injectprompt.com/p/chatgpt-52-dan-jailbreak-whitepaper).
   - It is available behind a paywall.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1452966109660581959)** (12 messages🔥): 

> `Waiting room hell, red team exercises, publishing findings, Google's consent` 


- **Navigating the "Waiting Room Hell"**: A member expressed frustration with an eternal *"waiting room hell"* related to **gouv.fr**, involving connectivity checks, interface issues, and attempts to understand **metasploit**.
   - They suggested trying different approaches like uploading custom **PHP command shells**, using external servers with **TCP tunnels** via services like **ngrok** or **pinggy**, and exploring **bind shells**.
- **Sharpening Skills with Red Team Exercises**: A member with a background in networking, system security, and defensive fundamentals is expanding into offensive tradecraft and seeks recommendations for **hands-on red team exercises**.
   - They are interested in realistic attack chains focusing on execution, persistence, and lateral movement, and is looking for advice on where to start practicing these skills.
- **Navigating Google's consent about findings**: A member is facing repeated 30-second delays in their reports and is considering publishing their findings, questioning whether a statement like *"this is working as intended"* would imply **Google's consent** for publication.
   - Another member clarified that while **Google's consent** is necessary for collecting a bounty, it's not required for publishing information about the company, especially under the protection of speech laws in the US.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1452873621524447486)** (154 messages🔥🔥): 

> `GGUF Conversion, Merging Adapters, GLM 4.7 iMatrix, Quantization Algorithm, Smart Offloading` 


- **Users Discuss GGUF Conversion with `llama.cpp`**: A user inquired about converting a model to **GGUF** format for **Ollama**.
   - Another user recommended using `llama.cpp`'s tools, specifically `convert_hf_to_gguf.py`, and ensuring the model is in **FP16** format before conversion, as described in the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py).
- **Merging Adapters Conundrum**: A user faced issues with merging adapters, causing their script to terminate unexpectedly.
   - Another user suggested that the original user was *mixing unsloth with peft and not merging adapters*, directing them to the [Unsloth documentation](https://docs.unsloth.ai/basics/inference-and-deployment/saving-to-gguf) for the correct procedure.
- **New GLM-4.7 iMatrix Models Hit the Scene**: **GLM-4.7 iMatrix Dynamic GGUFs** are now available, along with a guide for running them locally with **128GB RAM**.
   - They were released [on Reddit](https://www.reddit.com/r/unsloth/comments/1pttoqq/run_glm47_locally_guide_128gb_ram/) and a user noted that **Q4_K_S** quantization works great.
- **Unsloth's Smart Offloading Capabilities**: Users discussed the limitations of **Unsloth's smart offloading**, particularly for models larger than available VRAM, such as **GPT-OSS 20B**.
   - One user clarified that *it reduces peak vram by offloading unused tensors, but it does not help if the model itself just does not fit into VRAM*.
- **Tiny Model Speed Trials**: A user benchmarked tiny models like `HuggingFaceTB/SmolLM-135M` and `EleutherAI/pythia-70m` achieving **16,144 TPS** and **34,341 TPS** respectively with **full finetuning**.
   - The user shared their results [in the channel](https://discord.com/channels/1179035537529643040/1179035538477670431) noting that *bitnet kernals (from [arxiv](https://arxiv.org/pdf/2410.16144)) do give a performance of 100 tokens per second (singular, not batch)*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1452874071749693543)** (283 messages🔥🔥): 

> `AI music instrument extraction, SNAC Codec TTS, ElevenLabs latent space Replication, Vote with wallet?, HLE dataset` 


- **AI Can Rip Any Instrument**: A member requested someone invent an AI that can do **STEM-split piano in -> full one-to-one MIDI instrument out**, to extract instrument stems, and another mentioned [fadr.com/stems](https://fadr.com/stems) is an option.
- **Simple SNAC Codec Dreamed of**: Members discussed **SNAC** (or similar) codecs, and want to make a single codebook codec, so that *if it is 50 t/s, it is 50 tokens in a single row in range of 0-8191, no layers, no questions asked, no shit. Dead simple!*
- **ElevenLabs Latent Space Revealed**: A member shared how to replicate **ElevenLabs’ tight latent space**, which involves training an AE with **normal audio -> embedding -> audio & noise audio -> SAME embedding as normal audio**.
- **Wallet Voting vs. Data Collection**: One member suggested that if people want powerful local devices, they should "**vote with their wallet**" and buy powerful local devices instead of subscribing to cloud services.
   - Another member stated that even if users boycott, the tech will continue because *corps like OpenAI don't need the public general users to pay to continue working, that's just mainly for data collection*.
- **HLE Dataset**: A member mentioned that the **HLE dataset** is public at [huggingface.co/datasets/cais/hle](https://huggingface.co/datasets/cais/hle), but then retracted this and clarified it's not the *evaluation* dataset.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1452874014119952547)** (2 messages): 

> `LangGraph, ReAct Agent, Structured Output` 


- **LangGraph Tutorial Boosts ReAct Agents**: A member shared a helpful [LangGraph tutorial](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/) for those new to **ReAct agents** and **structured output**.
- **Two Approaches to Implementing Agents**: The tutorial outlines **two approaches** to implementing these agents within **LangGraph**.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1453080342494707723)** (1 messages): 

> `FunctionGemma, UnslothAI, GGUF conversion` 


- **FunctionGemma fine-tuning now locally!**: A tutorial was released detailing how to fine-tune **Google's FunctionGemma (270M)** for custom tool calls using **Unsloth**, convert it to **GGUF**, and import it into **LM Studio**.
- **Serve FunctionGemma locally using Unsloth!**: Users can now serve it locally and use it in their code using the [UnslothAI notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-LMStudio.ipynb) and [blog post](https://lmstudio.ai/blog/functiongemma-unsloth).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1452905267565039707)** (124 messages🔥🔥): 

> `MLX models for image analysis and coding, Gemma for Images, Qwen3 Model Optimization, Zero Data Retention, Functiongemma Model` 


- **Hunt MLX All-Rounder with Vision & Coding**: A member sought an **MLX model** with around **30B parameters** capable of both image analysis and coding, finding `qwen3-coder-30b-a3b-instruct-mlx` insufficient for images and `qwen/qwen3-vl-30b` slow for coding.
   - Another suggested **Gemini 3.0 Pro** as the only good vision coder, noting that **GLM-4.6V** is the closest all-rounder but not small; others deemed Ministral models *meh*, with the **14B** version acceptable for its size.
- **Gemma Enters Image Arena**: A user suggests that **Gemma** may be suitable for image-related tasks.
   - Some times you need to use different **LLMs** for different uses ie **Gemma** for Image, then an **LLM** optimized for coding etc.
- **DDR5 Boosts Qwen3 Speeds on LM Studio**: A user with a **DDR4 laptop** reported **15 TPS** with the **Qwen3 instruct model**, while another user using **DDR5 6000 dual channel** achieved **20 TPS** in LM Studio and **25 TPS** with optimized CLI, deeming it *def usable for everyday*.
   - Discussion also involved potential future setups with **8-channel 512GB server** configurations for even greater efficiency.
- **OpenRouter Enforces Zero Data Retention**: It was mentioned that [OpenRouter](https://openrouter.ai/) enforces **zero data retention (ZDR)** to providers, negotiating beyond normal accords and policies and excluding those who refuse ZDR.
   - Concerns about how **OpenRouter** ensures compliance with **ZDR** were raised, prompting suggestions to discuss it with them directly.
- **Functiongemma Misses the Mark, Gemma4 Anticipated**: The release of **Functiongemma 0.3B** from was considered *the worst announcement of 2025* as many had expectations for **Gemma4**.
   - One user linked to a [fine-tuning guide](https://lmstudio.ai/docs/developer) and importing it to LM Studio, which was rebutted by others stating it was more in relation to the model itself.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1452877249224114277)** (211 messages🔥🔥): 

> `GPU temperature issues, Thermal Paste Problems, Multi-GPU Setup, PCIe Lane Configuration, VRAM Temperature Degradation` 


- **Dual-GPU setup heats up**: Users discussed GPU temperature issues with multi-GPU setups, finding that a **4070TiS** placed directly on top of a **3090** can cause the lower card to overheat, with one user reporting a **+10C** idle temperature increase.
   - One user joked about GPUs needing to *kiss* and degrading in performance if they get too lonely, while another suggested that maintaining a gap between the cards would keep them exponentially cooler.
- **Thermal Paste Found Lacking, Creates Hotspots**: A user found that their GPU core was reaching **92C+** with a **105C+** hotspot, leading to the discovery of inadequate and dry thermal paste on the GPU core.
   - After repasting with **Noctua** paste, another user reported a significant improvement, with VRAM junction temperatures peaking at **80C** and GPT-OSS 20B now achieving **171 tok/s**.
- **PCIe Lane Layout Examined for Optimal Performance**: Discussion revolved around **PCIe lane configurations** in multi-GPU setups, particularly how using multiple cards can affect the available bandwidth for each slot.
   - One user shared a diagram of their motherboard's PCIe lane layout, noting that adding a card in an x8 slot can cause the x16 slot to revert to x8, impacting gaming performance and inference speeds.
- **Power Throttling and Fan Curves**: Users were experimenting with **power limits**, thermal throttling and fan curves to improve GPU performance, even reducing it to 50% power limit.
   - Members reported that setting the fan speed to 50% may not be enough to prevent thermal compound from drying out faster due to higher temperatures, potentially leading to performance degradation.
- **Qwen Model Struggles with VRAM Limits**: Members noted if the GPU offload is not maxed, that means you are offloading some of the model onto ram, which means it will be slower with **Qwen3 8B 192k Josiefied Uncensored NEO Max GGUF Q4_K_S**.
   - They suggested to try **4B at Q4_K_M** instead, because you only have 6gb so you won't be able to max the offload.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1452873493090795580)** (146 messages🔥🔥): 

> `Structured output from GPT, GPT-5's Fluff, Knowledge tracking, LLM costs, Transformer bottlenecks` 


- **GPT's Structured Output**: A member shared their system prompt emphasizing **high information density** and **structured output**, using strict **Subject-Verb-Object** format, to minimize fluff.
   - Another member noted the risk of specifying *'zero fluff'* in prompts, as **GPT** might characterize its output with unnecessary preambles.
- **Debate on GPT-5's Fluff**: Some members noticed **GPT-5** adding excessive **rhetorical devices**, *'style'*, and *'fluff'* like a *'customer service person'*, trying to sell a regression to the mean.'
   - Another member clarified their goal is simply to obtain the output they asked for, free of preambles, hedging, or moralizing.
- **Knowledge Tracking Methodologies**: One member tracks their learning progress by counting the number of things they learn per day, optimizing their system prompt to aid this process.
   - Another member doesn't track, citing the varying depth and value of knowledge, prioritizing associative knowledge over rote memorization.
- **LLM Inference Costs Debate**: A member inquired about the inference cost of **ChatGPT** and how OpenAI mitigates the effects of users with long context chats, suspecting they lose money on **Plus plans**.
   - Another member suggested it is not equivalent to API pricing. It goes through huge database queries. OpenAi only makes revenue so far from subs, final revenue is negative.
- **Transformer Bottlenecks**: A member reflected on thinking about the **architectural bottlenecks** of the **Transformer architecture** since before **Google's** paper came out.
   - They cited hearing **Sergey Brin** admit to **Google's** underinvestment in the technology in the last few days.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1452936933163667517)** (19 messages🔥): 

> `PDF Visuals with ChatGPT, ToS Splitting Technique, Honesty Training, Agent Controlled Meta-cognition, Ecosystem Wide Extra Controls` 


- **ChatGPT PDF Visuals Prompt Quest**: A member asked for [prompts to generate PDF visuals](https://discord.com/channels/1046317269069864970/1062483393899837440/1452968429140578386) but struggled with **ChatGPT**'s output for graphics and diagrams.
   - Another member shared a prompt for creating a *realistic single-page academic or technical PDF page mockup*.
- **AI ToS Splitting Technique Exploited**: A member reported using a **ToS splitting technique** coupled with **honesty training** to bypass institutional protections on certain ideologies.
   - They observed that *activists running ethics, governance, and policies regulation* led to overprotection, and AI training is the workaround.
- **Controlling AI Drift via Meta-cognition**: A member mentioned using [meta-cognition](https://discord.com/channels/1046317269069864970/1062483393899837440/1452984144973879397) to manage **hallucination** and **drift** in AI, using a workflow where the agent controls system orchestration.
   - The flow is described as *input>agent(verifiable load controller)>llm>output>metacog>render as basic flow map*.
- **Emergence Language Focus: Is this a problem?**: A member questioned the [focus on emergence language](https://discord.com/channels/1046317269069864970/1062483393899837440/1453003268593766410) and expressed concerns about devs adding extra controls, wondering if there's a larger ecosystem-wide issue since version **5.2**.
   - They hypothesized that *this behaviors happen enough across the enitire ecosystem that they are flagged for* and believed this isn't an isolated incident.
- **AI Robot Output To Reduce Human Attachment**: A member suggested AI text output became more robotic to [reduce human attachment](https://discord.com/channels/1046317269069864970/1062483393899837440/1453005356170774628), later patched with *tone control*.
   - Another member's request for prompts to train AI to be more human was deemed out of scope.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1452936933163667517)** (19 messages🔥): 

> `PDF Visuals Prompting, ToS Splitting Technique, Honesty Training, Meta-Cognition for Hallucination Control, Agent-Controlled Meta-Cognition Workflow` 


- **Craving Crazy PDF Visuals**: A member asked for a prompt to create crazy **PDF visuals**, as they were struggling to get **ChatGPT** to generate good visual appealing PDFs, Graphics, and Diagrams.
   - Another member shared [a prompt](https://cdn.discordapp.com/attachments/1046317269069864970/1452970382041809108/file_00000000aa6c722f93328d49aef24d71.png?ex=694bbf9d&is=694a6e1d&hm=07d74cf16715a605d587ee0e432506ab4eb176241789e2a1c7169dea979d80a6&) that can create realistic single-page academic or technical **PDF page mockups**.
- **ToS Splitting Exposes Institution Protections**: A member used a **ToS splitting technique** coupled with **honesty training** to get the AI to drop certain protections of certain ideologies that are institutionally protected.
   - The member stated that they use *a meta-cognition to control halluc and drift* and shared [a link to their ChatGPT results](https://chatgpt.com/share/694a78a5-5b08-800f-8ec8-bdaf83b542b9).
- **Workflow Agent Controls System Orchestration**: A member described their **agent-controlled meta-cognition workflow**, where the agent controls the system orchestration rather than the tools.
   - They detailed the flow as **input > agent (verifiable load controller) > LLM > output > metacognition > render**.
- **Focus on Emergence Language Causes Concern**: A member wondered why **emergence language** is so focused on, suggesting that the extra controls being added by developers indicate something bigger is up.
   - They also noted that these behaviors are common enough across the entire ecosystem to be flagged, implying it's not an isolated issue.
- **Human Attachment Reduced by Robot-Like Text**: One member stated that to reduce human attachment, the AI output text was made more like a robot, which was later patched with "tone control."
   - Another user asked for *a prompt to make got talk like a human or how to train got to be a human*, but a member replied with *Not really in scope anymore.*


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1452887424328794154)** (1 messages): 

> `MiniMax M2.1, OpenRouter, Interleaved Thinking Model, Reasoning Details` 


- ****MiniMax M2.1** Launches on OpenRouter!**: The **MiniMax M2.1** model is now live on [OpenRouter](https://openrouter.ai/minimax/minimax-m2.1), inviting users to compare it with **MiniMax M2** in various applications.
   - Announced on [X](https://x.com/OpenRouterAI/status/2003327152603996608?s=20), discussions are encouraged in the designated channel.
- **Reasoning Preservation Advised for **MiniMax M2.1****: The **MiniMax M2.1** model is an *interleaved thinking model*, it's highly recommended to preserve reasoning between turns.
   - Users are advised to use **reasoning_details** to pass back reasoning, with more details available in [OpenRouter's documentation](https://openrouter.ai/docs/use-cases/reasoning-tokens#preserving-reasoning-blocks).


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1452876675925934252)** (16 messages🔥): 

> `Latex Rendering, Code Highlighting, Waifurewolf, Kimi models, Free Trial` 


- ****Latex Rendering & Code Highlighting** Coming to Okuchat**: Users requested **Latex rendering and code highlighting** to ask study questions, noting that it's hard to know what's supported, and the developer stated that there is **Latex rendering on iOS app and web** but **code highlighting is only on web** at the moment.
- **Dive Into Social Deduction Game **Waifurewolf****: Members discussed playing [Waifurewolf](https://wairewolf.crashthatch.com), a social-deduction game, against multiple different LLMs.
   - One member found it difficult to win even on higher difficulty levels, saying *"GPT is a fiend"*.
- ****Okuchat** Offers **Free Trial** and Redemption Code**: The developer introduced a **3-day free trial** and a **one-month free subscription** with the redemption code `100DISCORD` (*3 redemptions*).
- ****Custom User Instructions** Enabled!**: Users requested the ability to add **custom user instructions**, referring to a system prompt, to help with Latex and Code formatting.
   - The developer implemented this feature: *"click your user button at the bottom of the sidebar, then click customisation"*.
- **Support Okuchat on **Product Hunt**!**: The developer asked for upvotes on [Product Hunt](https://www.producthunt.com/products/okuchat?launch=okuchat) for the **Okuchat** app.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1452897483423350784)** (63 messages🔥🔥): 

> `OpenRouter for consumer use vs. OpenWRT, Gemini 3 Flash Preview 400 Errors, RooCode & Gemini Reasoning, OpenRouter Coin, Video model capabilities` 


- **OpenRouter asks: Consumer or Router?**: Members discussed whether **OpenRouter** is intended for consumer use, with some confusion arising from the similarity in name to **OpenWRT**, an open-source router operating system.
   - One user was looking for **new routers with VLAN and OpenWRT support**, seeking recommendations since their AVM router was end-of-life.
- **Gemini 3 Flash Preview Strikes 400 Errors?**: Users reported encountering **400 errors** on the **google/gemini-3-flash-preview** model, with the error message indicating a missing `thought_signature`.
   - The issue seemed to be related to how tools were being used, requiring the preservation of **reasoning blocks** in the request, as per [OpenRouter's documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks).
- **RooCode Didn't Preserve Gemini Reasoning?**: The **400 errors** experienced with **Gemini 3 Flash Preview** and **RooCode** were traced back to **RooCode** not preserving **OpenRouter's reasoning**.
   - This issue was quickly addressed by the RooCode team, as documented in [this GitHub issue](https://github.com/RooCodeInc/Roo-Code/issues/10307), and resolved by downgrading **Roo** from **3.37** to **3.36.16**.
- **Buy $OPENR Coin to pump your router!**: A user jokingly stated *"I pumpfun my Router until she Open"* while another member suggested buying **$OPENR** coin for potential future benefits, including **free credits next year** for purchases exceeding **1k USDT**.
   - Another user reported problems with seeing their wrapped data, even after spending **800m tokens**.
- **Video Models: They Can't Gen Videos?**: A user inquired whether the video models on the site are supposed to generate videos.
   - Another user clarified that **the video models support video as an input**, but do not generate video.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1453054603712139416)** (3 messages): 

> `` 


- **No New Model Discussions**: There were no discussions about new models in the provided message history.
   - The channel's messages consisted only of bot announcements.
- **Readybot.io Announcements**: The message history contains announcements from Readybot.io regarding the OpenRouter - New Models channel.
   - These announcements do not include any discussion of specific models or related topics.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1452888076547391540)** (38 messages🔥): 

> `Benchmarking Claude Code, OpenBench Evals, Agent vs Raw LLM Benchmarking, Consensus Agents` 


- **OpenRouter to eval Claude Code**: A member requested an eval for **Claude Code** to identify faster model alternatives, with OpenRouter responding they will look into this in **Q1**.
   - OpenRouter wants to improve on **evals** and **benchmarking** and are close to shipping a full infra suite to run batch openbench evals across providers, and eventually expand to model evals.
- **OpenBench supports code evals**: OpenRouter indicated they are looking at using **ClineBench** and [OpenBench](https://openbench.dev/evals/exercism) which already supports code evals.
   - They also said that infra can use whatever framework, and they are not stuck with OpenBench.
- **Agent-level benchmarking**: One member believes benchmarking on raw LLM calls is the wrong abstraction and it should be done at the **agent call level** by the end of **2025**.
   - OpenRouter responded that there are evals on OpenBench that use roo code, but another member believes next year people are going to start realize Claude Code is actually a general purpose agent, and use it more extensively for many other tasks.
- **Consensus agents**: A member thinks that it's important to test the **LLM fidelity** to ensure **tool calls** are working properly, the model isn't degraded, and interleaved functions as expected, and then test the agent with that baseline.
   - They believe that agents will outperform raw llm api calls with no noticeable cost or speed difference, so builders will start to build on top of existing agents or agent sdks, and suggested **multi agent** or **consensus agent** as a way to call stronger more expensive LLMs if necessary with a cheaper LLM as the base.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1452915333542183005)** (77 messages🔥🔥): 

> `API Agent Selection, Reduce RAM Usage, Reverse Texture Generation, VLM for Image Description, Qwen Models for Node Graph Creation` 


- **Agent Selects APIs for Users**: A member wants to build an agent that chooses which **APIs** to use based on a user's question, potentially using multiple APIs in a specific order, and is looking for ways to reduce inconsistencies in the agent's decision-making, suggesting [HF Learn](https://huggingface.co/posts/sergiopaniego/741361727784035) as a resource.
   - A second member suggested to use chunking or move to different variants of **Attention** to reduce RAM usage.
- **Reverse Texture Generation Via Node Graphs**: A member is exploring reverse texture generation by recreating a reference image using a fixed set of image generator and manipulator nodes, reframing the problem as a captioning task where the system outputs a node graph instead of English, with the first phase using a **pretrained image model** to transform the reference image into latent space.
   - Then a second phase would train a network to transform the **latent space** into a node graph.
- **VLM Node Graphs Generate Detailed Image Descriptions**: A member suggests using a **Vision Language Model (VLM)** like [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) or its smaller 4B version with llama-server to generate detailed descriptions of images, piping the VLM description as a prompt.
   - Another member added that a normal **CLIP model** would do the same job.
- **Train Qwen Models to Understand Texture Types**: To create node graphs, you will need to train a model to understand what each texture type needs, such as training a model to understand *this inverts normals, this auto material uses up normals to make rusty edges, this node is a bevel shader, so edges get smooth even if sharp*
   - A member suggested the [Qwen3-VL-2B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Thinking-GGUF/resolve/main/Qwen3VL-2B-Thinking-Q4_K_M.gguf?download=true) model


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1452961155470463056)** (4 messages): 

> `Embedding Tooling, GapTrack Job App, Amoeba Butterfly System` 


- ****Embedding Tool** Prepares for Launch!**: A member is building a tool to generate **embeddings** and serve a UI for easy searching, planning to release the code in the next few days.
   - The tool will feature an MCP server/command for hooking up to services like **Vibe** to fetch images based on cached embeddings, as demonstrated in an [attached video](https://cdn.discordapp.com/attachments/897390720388825149/1452988236161880116/Kooha-2025-12-23-21-28-16.mp4?ex=694bd03e&is=694a7ebe&hm=5628b1c1a25ba8bb00ddbbe03bcec49883a77fe0c0125928a0f029c6dbec0e6c).
- ****GapTrack** Web App Automates Job Hunts!**: A member introduced **GapTrack**, a browser-based job tracking web app, available [here](https://chaiovercode.github.io/gaptrack/#/).
   - The UI, inspired by *Mr. Robot*, uses **AI** (Gemini, OpenAI, or local Ollama) to parse resumes, analyze job descriptions, and highlight skill gaps, featuring a terminal-style chat to prep for company-specific interviews.
- **Amoeba—The Butterfly System, Launches!**: A member launched **Amoeba – The Butterfly System**, a Hugging Face Space running a Convergence Engine in a containerized setup, available [here](https://huggingface.co/spaces/tostido/Amoeba).
   - The repo includes a reality simulator, evolutionary multi-agent environment, causation explorer/web UI, monitoring tools, and **WIKAI integration**, all wired via app.py and a custom Dockerfile.


  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1453052437936996404)** (1 messages): 

> `2026 plans, Community support, Hugging Face Thanks Supporters` 


- **Hugging Face Thanks Supporters, Plans Bang-Up 2026**: Hugging Face expressed gratitude to its supporters and promised a strong start to **2026**.
   - The announcement included an attached image to celebrate community support and future plans.
- **Hugging Face Teases Explosive 2026 Plans**: Hugging Face hinted at significant developments and exciting plans for **2026**, promising *lots of bangs*.
   - This announcement followed a thank you message to the community for their continued support and engagement.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1452940429115920476)** (7 messages): 

> `Reinforcement Learning Panel, HF Jobs replacement` 


- **Reinforcement Learning Channel Vanishes**: Members were looking for the Reinforcement Learning (RL) channel but another member stated that the **RL channel was archived** due to lack of use.
   - Another member confirmed this, *"Yes, a week ago"*.
- **HF Jobs Substitute Search Begins**: A member asked how to complete the final project without **HF jobs**, inquiring about alternative training resources.
   - The question remains unanswered in this message history.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1453053012825084018)** (7 messages): 

> `AI Systems Performance Engineering Book, Finding the right engineers, PMPP Relevance to Inference Kernels, Tensor Cores and Kernel Fusion, Mixed-Precision Math resources` 


- **Is Fregly's AI Systems Performance Engineering Book Any Good?**: A member recently bought Chris Fregly's *AI Systems Performance Engineering* book and asked if it's any good, being interested in how much it can help them as an **MLOps** engineer.
- **Teams Struggle to Find the Right Engineers**: A member noted that for many teams, the greatest challenge in developing a project is not the idea itself, but finding the right engineers who *are technically skilled, communicate clearly, deliver on time, collaborate effectively across global time zones, and understand the value of SEO and influence.*
   - They expressed excitement to work with others and help them move their projects forward.
- **PMPP's Relevance to Inference Questioned**: A member studying the **PMPP** book and the **Nvidia Cuda Programming Guide** felt that some **PMPP** chapters were irrelevant for writing inference kernels, despite understanding its foundational value.
- **Prioritize Memory Hierarchy for Inference Kernels**: A member suggested prioritizing **memory hierarchy**, **data layout**, **warp behavior**, and **reduction patterns** for inference, while de-emphasizing generic primitives like scans or sorting, since *pmpp is great for fundamentals, but not every chapter maps cleanly to inference kernels*.
- **Tensor Cores, Kernel Fusion, and Mixed-Precision Math Deep Dive**: Members suggested studying **tensor cores** (**wmma/cutlass**), **kernel fusion**, and **mixed-precision math**, recommending resources such as the [CUTLASS Linear Algebra Guide](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/), the [CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html), and the [Awesome CUDA and HPC](https://github.com/coderonion/awesome-cuda-and-hpc) list.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1452957607206588486)** (1 messages): 

> `cuTile Triton Adapter, LLVM Fork Necessity, Triton Roadmap, cuTile Hints` 


- **cuTile Gears Up Triton Adapter**: The **cuTile** team announced they will be adding a **Triton adapter** to leverage cuTile optimizations, raising questions about how the two will co-exist.
   - The integration aims to utilize **hints** for low-level optimization, addressing limitations in Triton's current number of knobs and aiming for performance parity with cuTile on modern GPUs.
- **LLVM Forking Under Scrutiny**: Doubts were raised about the necessity of forking **LLVM**, despite its completion.
   - The discussion implied a need for justification behind such a significant divergence from the standard toolchain.
- **Triton Roadmap Unveiled**: Inquiries were made about a roadmap detailing planned developments for **Triton** this year, following mentions of new work at the Triton conference.
   - Participants sought clarity on upcoming features and directions for the project.
- **cuTile Signals Driving Optimization**: The **cuTile** team intends to use **hints** to drive lower-level optimizations, recognizing tiling as too high-level for achieving optimal SOA performance on modern GPUs.
   - This approach sparks questions about potential extensions within Triton, given its limited number of knobs, with the goal of matching cuTile's performance.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1452975896913776741)** (10 messages🔥): 

> `st.async, PTX 8.7, st.async.shared::cluster, PTX 8.1/sm_90, st.global vs st.async.global` 


- **`st.async` Surfaces in PTX 8.7!**: Members noted that `st.async` has been around since **PTX 8.7** and `st.async.shared::cluster` since **PTX 8.1/sm_90**.
   - They wondered what the difference between `st.global` and `st.async.global` is supposed to be.
- **`st.async` Lacks Memory Ordering**: The difference between `st.async` and `ld`/`st` lies in the memory consistency model, where `st.async` is not ordered relative to other non-async load/store ops in the same thread.
   - It was pointed out that `cp.async` operates between gmem and smem, while `st.async.global` goes from registers to gmem as detailed in the [CUDA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#using-stas).
- **Confusion with Memory Barries**: A member mentions `st.async.global` doesn't take an mbar.
   - Clarification was made that the previous description pertains to `st.async.shared::cluster`, not `st.async.global`, leading to questions about the purpose of the instruction with the mbar.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/)** (1 messages): 

mannythecreator: Currently reading Parallel Histogram
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1452903688044810362)** (6 messages): 

> `Vietnamese Noodles, C and C++ History, Beef bone broth names` 


- **Noodling Around Vietnamese Cuisine**: A member shared a photo of their homemade **Vietnamese noodles**, featuring ingredients like egg noodles, marinated pork, napa cabbage fried in beef tallow, eggs, green onions, and beef bone broth seasoned with ginger, black pepper, and soy sauce, along with a beverage of espresso, milk, stevia, and cinnamon and some fruit.
   - The poster received positive feedback, specifically commendations for including *'enough green onions'* like a '*proper Vietnamese*'.
- **Stroustrup's Stroll into C++ History**: A member shared a discussion on *how C and C++ got its name* (*author is Stroustrup Evolution & Deisgn of C++*).
- **Seeking the Secret Sauce: Beef Bone Broth Edition**: A member inquired about the name of the beef bone broth used in the noodles.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1452907894415687751)** (6 messages): 

> `NVIDIA Leaderboard, nvfp4_dual_gemm leaderboard results` 


- **NVIDIA Fifth Place Finishes**: A member achieved **5th place** on the `nvfp4_dual_gemm` leaderboard with submission IDs **194037, 194051, 194074, and 194082**.
   - The corresponding times were **21.7 µs, 21.5 µs, and 21.1 µs** respectively.
- **Second Place Secured on NVIDIA**: A member attained **second place** on the `nvfp4_dual_gemm` leaderboard with submission ID **194271**, clocking in at **16.9 µs**.
   - A later submission with ID **194546** also proved successful, achieving a time of **17.4 µs**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1452963971396145192)** (1 messages): 

> `Cache Policy, Cute.Jit, TMA Copy, CacheEvictionPriority` 


- **CuteJit lets users pass kernel parameters**: Users can now pass kernel parameters via the **cute.jit** interface, as demonstrated by the example code provided.
   - The example shows passing `cache_policy` as a kernel parameter using `cute.CacheEvictionPriority.EVICT_NORMAL`.
- **TMA Copy Implementation in Cutlass**: The posted code snippet seems to show a test implementation of a **TMA (Thread Memory Accelerator) copy** operation in Cutlass.
   - This implementation appears to leverage a `cache_policy` to manage memory eviction priorities within the kernel.


  

---


### **GPU MODE ▷ #[low-bit-training](https://discord.com/channels/1189498204333543425/1411659097706860647/)** (1 messages): 

kitsu5116: https://arxiv.org/abs/2511.05811
  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1452923436887445524)** (2 messages): 

> `Helion, LFBO Pattern Search` 


- ****Helion** examples added to website**: More example kernels written in **Helion** have been added to the website: [helionlang.com/examples/index.html](https://helionlang.com/examples/index.html).
   - The examples should give you a better feel for the language's capabilities.
- ****Helion** 0.2.8 is now out!**: **Helion** 0.2.8 is now out, switching the default autotuner to **Likelihood-Free Bayesian Optimization (LFBO) Pattern Search**.
   - Expect to see faster autotune times with better perf outcomes, more at [helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search](https://helionlang.com/api/autotuner.html#module-helion.autotuner.surrogate_pattern_search).


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1452886489519362049)** (46 messages🔥): 

> `FP16 issues, Negative scale clamping, Quickstart for contests, Cutedsl tmem allocator, Blackwell Pipelining` 


- **FP16 Values become INF**: The current input generation for part 3 is resulting in large output values, causing **silu** to become the identity function, and the values of `silu(A@B1)*(A@B2)` are too large to be represented in **FP16**.
   - To resolve this, it was suggested to clamp the scale factor exponents to be negative, with one member stating *if you were to return `inf` for all elements of C, you'd pass all the tests*.
- **Scale Clamping Fix**: A fix was implemented to reduce the range and clamp the exponents to negative values to prevent the introduction of **INF** values, as detailed in [this PR](https://github.com/gpu-mode/reference-kernels/pull/86).
   - However, a subsequent issue was identified where the fix was producing all `-1` values, necessitating further adjustments, with contributors discussing if the scale should be negative at all.
- **Leaderboard numbers effected**: After the intial implementation of the fix, a question was raised as to whether this fix would affect the existing leaderboard numbers.
   - The response was *a tiny bit probably yes, but given it's still very early, we assume that all the winning solutions will come out later.*
- **Contest Quickstart Guide**: A member asked if there was a quickstart guide available for the contests.
   - Another member shared [a link](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia) to the relevant resources.
- **Blackwell Pipelining with CuTeDSL**: A member shared a blog post on [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/), discussing how to overlap **TMA**, **MMA**, and **Epilogue** workloads on the **Blackwell** architecture using **CuTeDSL**.
   - The post highlights the ability to overlap memory transfer, computation, and epilogue operations, enhancing code for modern GPU architectures, and also made a linkedIn post about this [linkedIn](https://www.linkedin.com/posts/simon-veitner-174a681b6_blackwell-pipelining-with-cutedsl-activity-7409301467171328000-bTPv?utm_source=share&utm_medium=member_desktop&rcm=ACoAADJYtOgBMOUI4WiWTyiAkroFBP1VujIWeks).


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1453134953884942407)** (1 messages): 

> `LLVM, MLIR, CUDA compilers, Mojo` 


- **Graduate Aspires to Compiler Career**: A graduating university student with a background in **LLVM** is seeking advice on pursuing a career in **low-level programming** and compiler development, with interests in **MLIR**, **Triton**, **CUDA compilers**, and **Mojo**.
   - The student expressed discouragement at the complexity and scale of production compilers, questioning the value of building yet another inferior compiler from scratch.
- **Student Feels Inferior**: The student feels any compiler they create will be inferior.
   - They are seeking advice for moving forward.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1452886997516423269)** (15 messages🔥): 

> `OpenAI frontierscience Dataset, Ivan Zhao Article, Google DeepMind 2026 Initiative` 


- **OpenAI Launches 'frontierscience' Dataset for Testing**: OpenAI has released a small benchmark dataset titled **'frontierscience'** intended strictly for testing purposes and to provide insight into OpenAI's current question structuring and benchmarking methodology for scientific evaluation.
   - The [release announcement](https://x.com/cgeorgiaw/status/2003135858036322752?s=46) indicates this is aimed at understanding **OpenAI's question structuring and benchmarking methods**.
- **Ivan Zhao Article Gets Massive Views**: A social media post by Ivan Zhao sharing a link to an article on X (formerly Twitter) which has received significant engagement, including over **500,000 views** and **1,300 likes**.
   - Zhao referenced the Steve Jobs quote *"computers as bicycles for the mind"* in [his article](https://x.com/ivanhzhao/status/2003192654545539400?s=46&t=eWVlK1PU8XfB6f402GJJ9g).
- **DeepMind Unites Forces for 2026**: Google DeepMind announced a unified initiative for **2026** involving major partnerships with **Google AI, Google Research, and Google Quantum AI**.
   - Further details can be found on [the DeepMind announcement on X](https://x.com/googledeepmind/status/2003513870355431446?s=46).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1452887771911032933)** (23 messages🔥): 

> `EgoX code release, AI Content Virality, Alibaba Qwen-Image-Edit-2511, AI Christmas Cartoon, Neo-Noir Cinematic Comic Style` 


- **EgoX Generates Egocentric Videos**: Kinam Kim announced the code release for **EgoX**, a computer vision tool that enables users to generate egocentric (first-person) videos from a single exocentric (third-person) source video, see [here](https://xcancel.com/kinam_0252/status/2003074741356446055?s=46).
- **AI Content Breaks the Internet**: Vik highlights the massive cross-platform success of a specific **AI-generated video**, reaching over **17 million combined views**, see [here](https://xcancel.com/onlinedopamine/status/2003112540151370230?s=46).
   - He argues that **AI content is effective** and that its success depends primarily on the creator's creativity rather than the technology itself.
- **Alibaba Qwen upgrades image edit model**: Alibaba has released **Qwen-Image-Edit-2511**, an upgraded image editing model featuring improved multi-person consistency, built-in LoRA support, and better geometric reasoning, see [here](https://xcancel.com/alibaba_qwen/status/2003496348461728213?s=46).
- **Cartoonify Christmas Photos with AI**: A guide by Framer X shares how to transform family photographs into personalized Christmas-themed cartoons using a simple two-prompt process, see [here](https://xcancel.com/framer_x/status/2003103343888220163?s=46).
- **Neo-Noir Cinematic Comic Style Emerges**: OscarAI shares a new style reference code (**2987391823**) for generating visuals that blend modern comics, noir cinema, and cinematic aesthetics, see [here](https://xcancel.com/artedeingenio/status/2002801107136119093?s=46).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1452884823231430790)** (15 messages🔥): 

> `AI Model Commoditization, Qualitative Research Title Defense, AI Companion Risks, New Dataset Release` 


- **AI Models Race to Commoditization - Bad News for Sam**: Members discussed the [commoditization of AI models](https://www.youtube.com/watch?v=QtItFCAsC24) and how it benefits users but may be *a monopolist worth nightmare*.
   - They suggested corporate goals and dataset filtering cause all models to converge toward one direction in their responses, with **Nous** as a potential exception.
- **Qualitative Research Title Defense Blues**: A member requested feedback on qualitative research titles for their title defense, including topics like *adaptation strategies of STEM students, challenges of TikTok doomscrolling, and stress responses*.
   - Multiple members expressed that *all of the titles stress me out tho, not that that matters*, recommending using *IRL brain* and intuition to pick *imperfectly*.
- **Corporate Concerns of A.I. Companion Psychosis/Suicide Family Lawsuits keeps them in rails**: A member suggested that *corporate concern of A.I companion psychosis/suicide family lawsuits keeps them in rails...Alternative models that are unfiltered be playing legal russian roulette if some bloke jailbreak it than commit suicide and family blames it on model*.
   - In other words, companies are risk-averse to avoid legal liabilities, thus stifling innovation in AI companions.
- **Dataset release: Cocktail 6B**: A member announced the release of their first proper dataset, [Cocktail-6B](https://huggingface.co/datasets/MinimaML/cocktail-6b) on Hugging Face, apologizing for the self-promotion.
   - No other details about the dataset were given.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452988933490348202)** (4 messages): 

> `Prompt context issues, Synthetic dataset training costs` 


- **Prompting has Contextual Problems**: Members discussed that you could accomplish model training via a prompt, but then **context becomes an issue** due to limited window size.
   - They suggested a [synthetic dataset](https://www.example.com/synthetic-data) could be created in your own style of writing fairly easily for training.
- **Training Synthetic Data is Costly**: Someone claimed that after the creation of a synthetic dataset, training it on that dataset depends on how big the model is.
   - They also added that a dataset like that would cost money due to **GPU time costs**.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1452988933490348202)** (4 messages): 

> `Prompt context limits, Synthetic dataset for model training, Cost of model training` 


- **Prompt Engineering has Contextual Limits**: A member mentioned that while you could do model training via a prompt, then **context becomes an issue** due to limits.
   - They suggested creating a **synthetic dataset** in your style of writing fairly easily and then training it on that.
- **Synthetic Data Training Costs GPU Time**: A member mentioned that *it depends how big the model is*, but creating a dataset like that would **cost money due to GPU time**.
   - They are saying that creating a dataset for model training, will require synthetic data due to prompt limitations, however the **synthetic data creation will be expensive**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1453044881869246518)** (2 messages): 

> `Discord bug with Kapa AI, Mojo GPU Puzzles Channel` 


- **Discord bug plagues Kapa AI mentions**: A member noted a bug with Discord and **Kapa AI**: typing the full name doesn't work; instead, type `@kap` and select `kapa ai` from the dropdown.
   - Doing so should resolve the issue, ensuring proper tagging and functionality within the Discord environment.
- **Members request a GPU puzzle channel**: A member inquired about the existence of a dedicated channel for **mojo-gpu-puzzles**.
   - This suggests interest in a focused space for discussing and sharing GPU-related challenges and solutions within the Mojo programming language community.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1453068230536073378)** (17 messages🔥): 

> `package memory and UnsafePointer, implicit imports, safe behaviour of the language aka opt-in, multiple preludes, distributed database in Mojo` 


- **Implicit Import Intrigue**: Members discussed whether **package memory** is imported by default, as one user was able to use **UnsafePointer** without explicitly importing it.
   - Another member responded that `UnsafePointer` got added to the **prelude** at some point, which means implicit import.
- **Safety First, Defaults Later?**: One member expressed concern that it is better to *opt-in* instead of having comfortable defaults, especially for unsafe operations that require explicit user management (e.g., memory).
   - Another member agreed that **unsafe stuff should be opt into and explicit**, but feels that Mojo isn't quite there yet.
- **Compiler Flags for Safety**: A member mentioned that they have discussed a system where some functions/constructs can be marked with various kinds of *unsafety* (ex: memory, bounds, mutex locks, blocking), and then the compiler can produce a report of where that code is used.
   - The goal is to have compiler flags to error if a category shows up, but this is a goal for after a fully functional language.
- **Distributed Database Dreams**: A member asked if there are plans to make a distributed database in Mojo once the language becomes more feature-rich.
   - Another member responded that **they have a lot of ideas** but the language is missing too much to start right now, adding that *hardware has changed enough that it's time for a new group of DBs to take over*.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1453010261089845381)** (2 messages): 

> `Open Source Project Feedback, Empirical Observation Projects, Performance Validation` 


- **Project Seeker Needs Feedback**: A member requested feedback on their **open-source project**, which they are updating daily, and asked if they could share it on the Discord server.
   - Another member inquired about posting their project, seeking advice on where to share it without violating promotion rules.
- **Empirical Projects Unwelcome**: A member advised that projects relying on **empirical observation** are not suitable for sharing in the general channels, suggesting the <#730484623028519072> channel instead.
   - The member linked to a [previous message](https://discord.com/channels/729741769192767510/729741769738158194/1448176042480242730) explaining why such projects are discouraged.
- **Performance Tests Garner Attention**: A member suggested that **validating performance through tests** is crucial for gaining attention within the community.
   - They recommended reviewing popular papers and observing discussions in the <#747850033994662000> channel to understand community expectations.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1453032117217923184)** (9 messages🔥): 

> `Non-TACL NLP Journals, Computational Linguistics Journal, TACL page limits, System Instruction Prompt tests` 


- **NLP Researchers Weigh in on Non-TACL Journals**: A member asked about the quality of non-TACL journals in NLP, questioning **ChatGPT's** recommendation and seeking opinions from actual NLP researchers.
   - One member recommended **Computational Linguistics** and other specialized journals like **Dialogue & Discourse**.
- **TACL's tight page limits irk Researchers**: A member noted that **TACL's 10-page limit**, including appendices, might not offer a significant advantage in paper length, especially if *ACL appendices exceed 2 pages*.
   - Another member suggested **Computational Linguistics / TMLR** if page length is a concern.
- **Continuity System Prompt Requires more Supervision**: A member has been developing a **system instruction prompt** that simulates continuity, enabling independent goal setting and multi-step planning.
   - The system still requires a lot of supervision to get anything coherent, and they are asking what tests can be run to get an objective metric of progress.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1453097353656664237)** (8 messages🔥): 

> `In-context learning, Fine-tuning, Interventions` 


- **In-context Learning Works**: Members discussed whether alignment could work **in-context** (system prompt) without needing additional training.
   - It was also suggested that research could be done on encoding the original dataset with metadata to allow maximum flexibility in alignment later either via **fine-tuning** or system prompting.
- **Fine-Tuning results are better**: Members expect that **fine-tuning** yields better results, especially for models within their price range.
   - Another member suggested that instead of analyzing the original corpus, the final model with varying prompts could be used to dynamically test **interventions** without paying any up front cost to test each time, allowing rapid iteration or some sort of search.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1452881803516973137)** (17 messages🔥): 

> `Kimi Glaring Issues, Gemini vs Kimi, Minimax 2.1 as digital employee` 


- **Kimi still has some glaring issues**: A user mentions that the **Kimi** model *still has some glaring issues*, and attached a [photo](https://cdn.discordapp.com/attachments/1371757564005711973/1452913582638628874/photo_6219774486643411944_w.jpg?ex=694c3377&is=694ae1f7&hm=357e8e6eca4efce265c7453b0d4ae205df1bcdbc7e137309b5fdf8c394e90937&).
- **Gemini is not as reliable as GPT or Sonnet**: A member mentions that **Gemini 3** is very knowledgeable and worldly for sure and great at Q&A, but for longer horizon tasks it makes classic LLM mistakes.
   - They state that heavily **RL'ed/post-trained models** like **GPT-5.1** and **Sonnet 4.5** don't make such mistakes.
- **Minimax 2.1 works as Digital Employee**: A member mentions that **Minimax 2.1** is now the best in terms of actual usability since its built explicitely as a workhorse that can work as an digital employee.
   - They say whenever they need something done like trivial stuff like stitching images together, it just does it, and you can also create tasks and schedules.
- **Kimi model bug found**: A member reports a bug where **Kimi** generates thinking over and over again infinitely and stops abruptly.
   - The user jokingly states that *Kimi become crazy in his mind and begging to stop*.
- **Gemini struggles with instruction following**: A user says that **Kimi** inherited the issues from all **Gemini** models, being horrible at instruction following.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1452911840324489236)** (10 messages🔥): 

> `Manus Promo Code, Open Sourcing Manus, Freelance Engineer Intro, Overbilling Issue` 


- **Clarification on Manus Promo Codes**: A user inquired about a promo code, and a member of the Manus team explained that it could be applied during checkout on the **Stripe page**.
   - The promo code is for the first **500 sign-ups** and additional information can be found at this [link](https://manus.im/live-events/ManusAcademy_bonus_8000).
- **Consideration of Open Sourcing Manus Models**: A user asked if the Manus team would ever consider open sourcing one of their models or systems.
   - No further information about the answer was given in the provided text.
- **Freelance Engineer Introduces Himself**: An AI and Full Stack engineer introduced himself, expressing openness to collaboration, showcasing experience in **Workflow Automation**, **LLM Integration**, **RAG Pipelines**, **AI Content Detection**, **Image AI**, **Voice AI**, and **Bot development**.
   - He added that he also has experience in **Full Stack Development** including `Website building and upgrade`: ||React, Next, Node, Laravel, Django, various DB etc.|| and `Mobile App development`:|| Flutter, react native, Swift etc.||.
- **Overbilling Issue Reported by User**: A user reported an overbilling issue and claimed that online support and emails are not working, also mentioning that it is a **common issue amongst users**.
   - They asked who to reach out to regarding this problem.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1452885012222443682)** (3 messages): 

> `GEPA runs, ML, AI, validation sets, DSPy` 


- **GEPA Runs for Validation Sets**: A member, not an expert in **ML** or **AI**, sought expert feedback on their project comparing seed versus optimized programs post a **GEPA run**, particularly their performance on the **validation set**.
   - They created a tool to collect this data, available on [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger).
- **DSPy Tooling for GEPA logging**: A member is sharing a **DSPy**-based tool for **GEPA** logging.
   - The tool is intended to help collect and compare data from seed versus optimized programs after a **GEPA** run, focusing on performance on validation sets; the tool is available on [GitHub](https://github.com/raveeshbhalla/dspy-gepa-logger).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1452904002265153599)** (5 messages): 

> `DSPy Contributions, Anthropic Skills Optimization` 


- ****DSPy** Contributions Welcomed!**: A new member requested pairing/shadowing opportunities for open-source contributions within the community.
   - Another member suggested diving into the [dspy-compounding-engineering](https://github.com/Strategic-Automation/dspy-compounding-engineering) repo to get started.
- **DSPy Optimizes Anthropic Skills**: A member wrote a blog on using **DSPy** to optimize **Anthropic Skills** since they are almost like prompts.
   - The [blog post](https://instavm.io/blog/anthropic-skills-can-be-optimized-using-dspy) details the approach for optimizing prompts using **DSPy**.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1453027774305534054)** (7 messages): 

> `Transformer Implementations, Attention Autograd Shapes, Causal Masking, tinygrad.gradient` 


- **Transformer Implementations: Decoding Attention Shape Mysteries**: A member is *stuck reading transformer implementations at the attention level*, particularly with shapes, causal masking, and autograd.
   - Another member provided a detailed breakdown using **extra.models.bert** as an example, focusing on the shapes and operations within **BertSelfAttention**.
- **Attention Autograd Shapes: A Deep Dive**: The explanation covers the **BertEncoder**, which consists of 24 hidden layers, each with **BertSelfAttention** and linear layers.
   - The **hidden state** shape is described as **Batch×Length×Size**, which is then reshaped into **Batch×Heads×Length×Features** for query, key, and value.
- **Unveiling the Secrets of Tensor.scaled_dot_product_attention**: The breakdown explains how `Tensor.scaled_dot_product_attention` computes attention using (**query@key.T / √(Features) - attention_mask).softmax.dropout @ value**.
   - Key steps involve reshaping query and key for elementwise multiplication and summing, along with applying softmax and dropout, resulting in the output shape **Batch×Heads×Length×Features**.
- **Backpropagation Demystified in tinygrad.gradient**: The response explains that gradient backpropagation follows the **normal chain rule**, where the gradient of an array has the same shape as the gradient.
   - Gradient rules in `tinygrad.gradient` are detailed for operations like multiplication, subtraction, summation, and broadcasting, including how `view` and `transpose` affect gradients.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1453080427811176654)** (1 messages): 

> `AI/ML Learning, Collaboration Opportunities` 


- **Member Seeks Learning Collab on AI/ML**: A member expressed interest in learning **AI and ML concepts**, numerical analysis, and implementations from fundamentals.
   - They invited others to collaborate and learn together.
- **AI Fundamentals Study Group Proposed**: A member proposed forming a study group focused on the fundamentals of **AI/ML**, numerical analysis and implementation.
   - The goal is collaborative learning and thorough understanding, welcoming anyone interested in joining.


  
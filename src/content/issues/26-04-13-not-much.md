---
id: MjAyNS0x
title: not much happened today
date: '2026-04-13T05:44:39.731046Z'
description: >-
  **Harness engineering** is emerging as a key discipline in AI agent
  development, emphasizing components like filesystems, memory, and retries
  beyond just models. **OpenAI's Codex** is expanding agentic coding workflows
  beyond software engineering, including codebase understanding and bug triage.
  Tooling trends show convergence on multi-agent orchestration, observability,
  and remote control, with **GitHub Copilot**, **Cursor**, and **LangChain**
  advancing these capabilities. The **Hermes Agent v0.9.0** release introduces a
  local web dashboard and enhanced security, gaining community traction over
  **OpenClaw** for UX and efficiency. The open agent ecosystem is growing with
  projects like **Open Agents** and **DeepAgent** providing modular stacks and
  runtimes.
companies:
  - openai
  - github
  - cursor
  - langchain
  - nous-research
models:
  - codex
topics:
  - agent-harnesses
  - multi-agent-systems
  - software-engineering
  - tooling
  - orchestration
  - observability
  - remote-control
  - security-hardening
  - user-experience
  - open-source
  - community-engagement
people:
  - andrew_ng
  - steve_yegge
  - gabrielchua
  - giffmana
  - rhys_sullivan
  - teknium
  - shaun_furman
  - dabit3
  - robinebers
  - zainanzhou
  - nicoalbanese10
  - bromann
  - elliothyun
  - tiagonbotelho
  - pierceboggan
  - sydneyrunkle
---


**a quiet day.**

> AI News for 4/11/2026-4/13/2026. We checked 12 subreddits, [544 Twitters](https://twitter.com/i/lists/1585430245762441216) and no further Discords. [AINews' website](https://news.smol.ai/) lets you search all past issues. As a reminder, [AINews is now a section of Latent Space](https://www.latent.space/p/2026). You can [opt in/out](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) of email frequencies!




---

# AI Twitter Recap


**Agent Harnesses, Coding Workflows, and the Shift from Single-Model to System Design**

- **Harness engineering is now a first-class discipline**: A recurring theme across [AI Engineer Europe takeaways](https://x.com/dat_attacked/status/2043647001749836253), [Vtrivedy’s framing of harness primitives](https://x.com/Vtrivedy10/status/2043870915059236966), and multiple agent-builder posts is that useful agents are not “just models.” Filesystems, bash, compaction, memory, permissions, retries, evals, and subagents are increasingly treated as core product surface. This is echoed by [Andrew Ng](https://x.com/AndrewYNg/status/2043742105852621052), who argues the bottleneck is shifting from implementation to deciding what to build, and by [Steve Yegge](https://x.com/Steve_Yegge/status/2043747998740689171), who claims enterprise adoption is still far behind frontier practice despite broad tool access.

- **OpenAI’s Codex usage patterns suggest agentic coding is broadening beyond SWE**: OpenAI shared a practical catalog of Codex workflows—understanding large codebases, PR review, Figma-to-code, bug triage, dataset analysis, CLI tools, onboarding, and even slide generation—via [@gabrielchua](https://x.com/gabrielchua/status/2043339151278506234). In the field, users report the same “agents as glue” pattern: e.g. [giffmana](https://x.com/giffmana/status/2043401612035559445) using Codex to patch Java/Qt binaries on Linux for a niche Wayland/HIDPI issue, while others remain skeptical that current models yet outperform direct human implementation for trusted production work, as in [Rhys Sullivan’s critique](https://x.com/RhysSullivan/status/2043584591861321929).

- **Tooling is converging on multi-agent orchestration, observability, and remote control**: GitHub shipped [Copilot remote control from web/mobile](https://x.com/pierceboggan/status/2043717775265562701), with follow-up from [@tiagonbotelho](https://x.com/tiagonbotelho/status/2043720370734104923). Cursor added [split agents plus search/perf improvements](https://x.com/cursor_ai/status/2043798784367546707). LangChain emphasized [guardrails via middleware and filesystem permissions](https://x.com/sydneyrunkle/status/2043767032361967751), while deepagents’ mental model reduces subagents to structured tool/function calls as described by [@ElliotHyun](https://x.com/ElliotHyun/status/2043721149616369719). The common pattern: agent products are maturing by exposing control planes, not by claiming fully autonomous reliability.

**Hermes Agent’s Dashboard Release, OpenClaw Competition, and Open Agent Stacks**

- **Hermes is consolidating momentum as the most discussed open harness of the day**: The headline release is Hermes Agent v0.9.0 with a local web dashboard, fast mode, backup/import, stronger security hardening, and broader channel support; see [@Teknium](https://x.com/Teknium/status/2043771509123232230) and the official [@NousResearch announcement](https://x.com/NousResearch/status/2043791876835156362). Community reaction frames the dashboard as the feature that could take Hermes beyond power users, including [Shaun Furman’s “openclaw moment” claim](https://x.com/Shaun__Furman/status/2043820083114545416).

- **OpenClaw is still shipping, but comparison discourse is tilting toward Hermes on UX and efficiency**: OpenClaw posted a substantial update—memory imports, “Memory Palace,” richer chat UI, plugin setup guidance, better video generation, and more integrations—via [@TheTuringPost](https://x.com/TheTuringPost/status/2043340386538778840). But several users explicitly report preferring Hermes over OpenClaw for speed, architecture, or token efficiency, including [dabit3](https://x.com/dabit3/status/2043808914312212568), [robinebers](https://x.com/robinebers/status/2043835216670929005), and [ZainanZhou’s harness-level explanation](https://x.com/ZainanZhou/status/2043760979931213851) that better preselection/context shaping may be reducing token burn.

- **The open ecosystem around agent stacks is thickening**: [Open Agents](https://x.com/nicoalbanese10/status/2043745569278251112) was open-sourced as a cloud coding agent stack; [bromann](https://x.com/bromann/status/2043886229650067729) contrasted it with DeepAgent as a lower-level runtime with pluggable model providers, sandboxes, middleware, and tracing. Hermes itself is accumulating community skills, tutorials, multi-agent recipes, and integrations—from [Chinese tutorial roundups](https://x.com/biteye_sister/status/2043630704798679545) to practical “team of 4 agents” guidance from [@coreyganim](https://x.com/coreyganim/status/2043627229205193211). The notable technical pattern is persistent role separation plus isolated memory, rather than naive “one agent does everything.”

**Cybersecurity, Model Capability Escalation, and the Mythos Shockwave**

- **Claude Mythos Preview dominated the cyber-security conversation**: The UK AI Security Institute reported that Mythos is [the first model to complete an AISI cyber range end-to-end](https://x.com/AISecurityInst/status/2043683577594794183), with follow-on commentary from [ekinomicss](https://x.com/ekinomicss/status/2043688793085992970) noting success on a 32-step corporate network attack simulation. Additional reactions emphasized both capability and efficiency, e.g. [scaling01 claiming](https://x.com/scaling01/status/2043700788245963167) Mythos reaches Opus-level performance at roughly **40%** of the tokens after long runs.

- **The security implication is not just benchmark progress, but operational usefulness**: [emollick](https://x.com/emollick/status/2043810051979157680) called the concern warranted; [ananayarora](https://x.com/ananayarora/status/2043381424594837789) pointed to Marcus Hutchins’ reaction as especially meaningful. The emerging point is that “vulnerability research model” is no longer speculative marketing language; labs and external evaluators are now describing end-to-end exploit workflows completed on independent ranges.

- **Defensive tooling is maturing in parallel, but the asymmetry is obvious**: [The Turing Post’s roundup](https://x.com/TheTuringPost/status/2043332388785426498) highlighted 10 open AI security projects, including NVIDIA NeMo Guardrails, garak, Promptfoo, LLM Guard, ShieldGemma 2, and CyberSecEval 3. At the same time, builders are revisiting assumptions that agents can safely replace mature dependencies: [dbreunig](https://x.com/dbreunig/status/2043762702653460520) argues the token math changes once you price in hardening and security review, making well-maintained OSS libraries comparatively more attractive again.

**Inference, Retrieval, OCR, and Systems Performance**

- **Document/OCR evaluation got a serious new benchmark**: LlamaIndex released [ParseBench](https://x.com/jerryjliu0/status/2043721536922955918), an open benchmark/dataset for document parsing focused on agent-relevant semantic correctness rather than exact-match text similarity. It includes roughly **2,000** human-verified enterprise pages and **167,000+** evaluation rules across tables, charts, content faithfulness, semantic formatting, and visual grounding. One notable result: no parser dominates every axis, but LlamaParse reportedly leads overall at **84.9%**.

- **Hugging Face showed OCR at industrial scale can be cheap and reliable with open models**: [@ClementDelangue](https://x.com/ClementDelangue/status/2043779449322160270) reported OCR’ing **27,000 arXiv papers** into Markdown using an open **5B** model, **16** parallel HF Jobs on L40S, for about **$850** in **~29 hours**, now powering “Chat with your paper.” The follow-up identified the model as [Chandra-OCR-2](https://x.com/ClementDelangue/status/2043783879601848726).

- **Retrieval and transport-layer optimizations continue to matter**: LightOn shipped [ColGrep 1.2.0](https://x.com/raphaelsrty/status/2043676936442875954) with BM25 trigrams for hybrid multi-vector retrieval and relative paths to save tokens, positioning it as an easy agent-search upgrade. On the systems side, [Lewis Tunstall and colleagues](https://x.com/_lewtun/status/2043690765227102335) highlighted a non-obvious on-policy distillation bottleneck: vLLM transmitting logprobs as JSON over the wire. Switching to binary NumPy arrays yielded a **1.4x** speedup, a useful reminder that infra wins often sit outside kernels and model code.

- **Compression and speculative decoding remain high-leverage deployment levers**: Red Hat AI showed a [Gemma 4 31B quantized deployment on vLLM](https://x.com/RedHat_AI/status/2043709783102906489) with nearly **2x tokens/sec**, half the memory, and **99%+** accuracy retained. On speculative decoding, posts covered [DFlash adapters for Kimi/Qwen-family local speedups](https://x.com/winglian/status/2043731370598347066), Baseten’s [EAGLE-3 production advice](https://x.com/baseten/status/2043762663235432855), and new research such as [DDTree](https://x.com/liranringel/status/2043813397972607477), which drafts a tree in one block-diffusion pass to verify multiple continuations jointly.

**Research Directions: Memory, Verification, RL, and Model Architecture**

- **Long-context memory research is pushing beyond vanilla KV cache scaling**: [behrouz_ali](https://x.com/behrouz_ali/status/2043743704335192095) outlined “Memory Caching,” a family of architectures that compress context into a slowly growing recurrent memory, aiming for effective memory growth closer to attention but inference cost closer to RNNs. Sparse Selective Caching is positioned as the most practical variant. Related commentary from [askalphaxiv](https://x.com/askalphaxiv/status/2043782770657219010) frames it as an interpolation between standard recurrence and full quadratic attention.

- **Verifier-style test-time methods are becoming a serious agent benchmark strategy**: [Azali Amirhossein et al.](https://x.com/Azaliamirh/status/2043813128690192893) introduced **LLM-as-a-Verifier**, scoring candidate pairs by asking the model to rank outputs and then using rank-token logprobs to estimate expected quality. The pitch is that winner-selection, not candidate generation, is often the test-time scaling bottleneck; a single verification pass can outperform more cumbersome reranking setups on agentic benchmarks.

- **Reasoning discovery remains a weak point, which some see as good news for oversight**: [Laura Ruis](https://x.com/LauraRuis/status/2043715536186384775) reported that LLMs struggle to *discover* latent planning strategies even when the strategy is trivial once taught, with scaling up to GPT-5.4 yielding only modest gains. Separately, [Wen Sun](https://x.com/WenSun1/status/2043755261954011484) argued RL-based prompt optimization can generalize from as few as **2** examples where zeroth-order methods overfit. The combined takeaway: there is still substantial room in training objectives and test-time scaffolding before “reasoning” becomes robustly self-bootstrapping.

**Top Tweets (by engagement)**

- **Codex use cases at OpenAI**: [@gabrielchua](https://x.com/gabrielchua/status/2043339151278506234) shared a broad, practical inventory of internal Codex workflows, spanning code understanding, app building, ops automation, and non-engineering tasks.
- **AISI cyber eval of Claude Mythos Preview**: [@AISecurityInst](https://x.com/AISecurityInst/status/2043683577594794183) reported the first end-to-end completion of its cyber range by a model, making this one of the most technically consequential posts in the set.
- **Hermes Agent dashboard release**: [@NousResearch](https://x.com/NousResearch/status/2043791876835156362) announced the local dashboard and related v0.9.0 features, catalyzing a wave of user comparisons with OpenClaw and Claude Code.
- **OpenAI’s “compute-powered economy” memo**: [@gdb](https://x.com/gdb/status/2043831031468568734) outlined OpenAI’s thesis that software engineering is the leading edge of a broader transition toward compute-mediated work and intent-driven tooling.
- **Hugging Face’s large-scale open OCR deployment**: [@ClementDelangue](https://x.com/ClementDelangue/status/2043779449322160270) demonstrated low-cost, fault-tolerant OCR of 27k papers into Markdown using open models and HF Jobs.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Gemma 4 Model Developments and Benchmarks

  - **[Best Local LLMs - Apr 2026](https://www.reddit.com/r/LocalLLaMA/comments/1sknx6n/best_local_llms_apr_2026/)** (Activity: 440): **The post discusses the latest advancements in local Large Language Models (LLMs) as of April 2026, highlighting the release of **Qwen3.5**, **Gemma4**, and **GLM-5.1**, which claims state-of-the-art (SOTA) performance. The **Minimax-M2.7** model is noted for its accessibility, and **PrismML Bonsai** introduces effective 1-bit models. The thread encourages users to share their experiences with these models, focusing on open weights models and detailing their setups, usage, and tools. The post also categorizes models by VRAM requirements, ranging from 'Unlimited' (&gt;128GB) to 'S' (&lt;8GB).** One comment suggests further breaking down categories for models requiring more than 128GB VRAM, indicating a need for more granularity in classifying high-resource models.

    - A user suggests breaking down categories for models with memory greater than 128 GB, emphasizing the need for more granular classification without relying on labels like 'S' or 'M'. This implies a demand for detailed performance metrics and benchmarks for high-memory models, which could be crucial for applications requiring extensive data processing or complex computations.
    - The discussion includes a focus on specialized local LLMs tailored for specific domains such as medical, legal, accounting, and mathematics. This highlights the importance of domain-specific optimizations and the potential for these models to outperform general-purpose LLMs in niche areas by leveraging specialized training data and architectures.
    - There is a mention of agentic coding and tool use, which suggests an interest in models that can autonomously interact with tools or APIs to perform tasks. This points to a trend towards developing LLMs with capabilities for dynamic task execution and integration with external systems, enhancing their utility in practical applications.

  - **[Audio processing landed in llama-server with Gemma-4](https://www.reddit.com/r/LocalLLaMA/comments/1sjhxrw/audio_processing_landed_in_llamaserver_with_gemma4/)** (Activity: 494): ****llama.cpp** (llama-server) has integrated audio processing capabilities, specifically supporting Speech-to-Text (STT) with the **Gemma-4 E2A and E4A models**. This update allows native audio support, eliminating the need for separate pipelines like Whisper. However, users report issues with longer audio transcriptions, such as errors in `llama-context.cpp` and looping sentences. The recommended setup involves using `E4B as Q8_XL quant with BF16 mmproj`, as other configurations degrade performance. For optimal transcription results, specific templates should be followed, emphasizing precise formatting and number representation.** Some users express skepticism about its performance compared to Whisper, while others note that despite the integration, the system struggles with longer audio segments, suggesting that **Voxtral** performs better in these cases.

    - Chromix_ highlights several technical issues with the current implementation of audio processing in llama-server, particularly when handling audio longer than 5 minutes. They note that using E4B as Q8_XL quant with BF16 mmproj is recommended, as other formats degrade performance. However, they encounter errors such as `llama-context.cpp:1601` and issues with transcription quality, including looping sentences and early termination. They suggest using specific templates for transcription and translation to improve results.
    - GroundbreakingMall54 points out the significance of native audio support in llama.cpp, which eliminates the need for a separate Whisper pipeline. This integration is seen as a major improvement for users who previously had to manage multiple systems for audio processing.
    - ML-Future shares their experience testing the audio processing feature in Spanish, noting that while it is not perfect, it is quite accurate and performs better than Whisper. This suggests that the new feature may offer improved transcription quality in certain languages compared to existing solutions.

  - **[Speculative Decoding works great for Gemma 4 31B with E2B draft (+29% avg, +50% on code)](https://www.reddit.com/r/LocalLLaMA/comments/1sjct6a/speculative_decoding_works_great_for_gemma_4_31b/)** (Activity: 527): **The post discusses the implementation of speculative decoding with the **Gemma 4 31B** model using **Gemma 4 E2B (4.65B)** as a draft model, achieving significant performance improvements. The setup involved an **RTX 5090 GPU** and a **llama.cpp fork** with TurboQuant KV cache, configured with a `128K context` and specific draft parameters (`--draft-max 8 --draft-min 1`). Benchmarks showed a `+29%` average speedup, with `+50%` on code generation tasks, attributed to the compatibility of vocabularies between models, avoiding token translation overhead. A critical issue was identified with the `add_bos_token` metadata mismatch in early GGUF versions, which was resolved by re-downloading updated models. The post also highlights the importance of setting `--parallel 1` to prevent VRAM overuse and suggests practical tips for optimizing performance, such as using Q4 draft models and managing VRAM allocation.** Commenters suggested experimenting with different `--draft-max` and `--draft-min` values and inquired about the full llama-server command and the specific fork used. Another suggestion was to offload per-layer embeddings to the CPU to optimize VRAM usage without affecting inference speed.

    - Odd-Ordinary-5922 inquires about the impact of adjusting `--draft-max` and `--draft-min` parameters, which are likely related to controlling the speculative decoding process. These parameters could influence the balance between speed and accuracy, though specific effects are not detailed in the comment.
    - albuz suggests optimizing VRAM usage by offloading per-layer embeddings of a draft model to the CPU using the `--override-tensor-draft "per_layer_token_embd\.weight=CPU"` command. This technique is intended to conserve GPU memory without impacting inference speed, which could be beneficial for users with limited VRAM resources.
    - EdenistTech reports a significant performance improvement when using a draft model on a 5070Ti/5060Ti combo, achieving a throughput increase from approximately 25 tokens per second to 40 tokens per second with a 128K context size. This suggests that speculative decoding can substantially enhance processing speed, especially in setups with specific hardware configurations.


### 2. Minimax M2.7 and Licensing Updates

  - **[Ryan Lee from MiniMax posts article on the license stating it's mostly for API providers that did a poor job serving M2.1/M2.5 and may update the license for regular users!](https://www.reddit.com/r/LocalLLaMA/comments/1skabyf/ryan_lee_from_minimax_posts_article_on_the/)** (Activity: 451): **The image is a tweet from Ryan Lee of **MiniMax**, discussing the licensing terms for their M2.7 model. He clarifies that self-hosting M2.7 for code writing is permitted and free, but acknowledges that the current license lacks detail and will be updated. This is significant as it addresses concerns about the clarity and applicability of the license, especially for API providers who have been criticized for poor service quality. The discussion highlights issues with licensing that restricts commercial use, which can complicate self-hosting and potentially mislead users about model capabilities.** Commenters express skepticism about the clarity and intent of the licensing terms, noting that some providers misrepresent the quality of models they serve. There is also concern about the complexity of licenses that aim to prevent profit-driven hosting, which can inadvertently affect legitimate self-hosting efforts.

    - Few_Painter_5588 highlights a significant issue with OpenRouter, noting that many API providers misrepresent the quality of models they serve, with some not even serving the models they claim to. This reflects a broader problem in the ecosystem where the reliability and transparency of model serving are critical for user trust and effective deployment.
    - silenceimpaired discusses the complexities of licenses that aim to prevent profit-driven hosting, which can inadvertently complicate self-hosting. They mention Black Forest Labs as an example where such licensing strategies have led to confusion, suggesting that licenses should allow commercial use if models are run on owned hardware or within a specific proximity to users to avoid these issues.
    - ambient_temp_xeno points out the legal nuances in the licensing language, noting that while the statement doesn't explicitly mention restrictions on commercial use of 'code writing,' earlier communications did. This highlights the importance of clear and consistent messaging in licensing terms to avoid misunderstandings and ensure compliance.

  - **[Local Minimax M2.7, GTA benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1sk70ph/local_minimax_m27_gta_benchmark/)** (Activity: 383): **The post discusses a benchmark test using the **Minimax M2.7** model to create a 3D Grand Theft Auto (GTA)-like experience within a single web page. The user highlights that while **GLM 5** excels in aesthetics and detail without explicit instructions, Minimax M2.7 performed well when tasked with adding trees and birds using the boids algorithm. The test was conducted in the openwebui artifacts window and OpenCode, with the model running at `IQ2_XXS` for maximum speed, maintaining coherence and capability. The image shows a blocky, stylized game environment with vehicles and urban elements, indicating a driving simulation or benchmark test.** A comment notes that **GLM 5** provides more detail on the main character without needing specific prompts, suggesting it may be superior in certain aesthetic aspects.

    - -dysangel- mentions the use of **GLM 5** for comparison, highlighting its ability to provide more detail on the main character without additional prompts. This suggests that GLM 5 might have advanced capabilities in generating detailed character descriptions, potentially useful for applications requiring rich narrative content.
    - EndlessZone123 critiques the use of **GLM** for 2D or 3D tasks, arguing that it is not a vision model and primarily relies on memory for one-shot tasks. This implies limitations in GLM's ability to handle continuous or iterative visual tasks, which could be a consideration for developers working on projects requiring ongoing visual processing.
    - averagebear_003 notes the inclusion of birds in the benchmark, which might indicate attention to environmental detail in the **Minimax M2.7** model's performance. This could be relevant for evaluating the model's capability in rendering complex scenes with dynamic elements.


### 3. Local AI Hardware and Setup Discussions

  - **[Local models are a godsend when it comes to discussing personal matters](https://www.reddit.com/r/LocalLLaMA/comments/1ska9av/local_models_are_a_godsend_when_it_comes_to/)** (Activity: 443): **The post discusses the use of local models, specifically the **Gemma 4 26B A4B model**, which supports a `256k context`, for analyzing personal journals. The user shared a journal of over `100k+ tokens` with the model, asking guided questions to gain insights into recurring themes, avoided topics, and the evolution of thoughts. The user emphasizes the privacy benefits of local models over proprietary ones, highlighting the ability to process sensitive information securely on personal devices. This reflects a growing trend towards using AI for personal data analysis while maintaining privacy.** Commenters highlight the advantages of local models, such as the ability to process personal documents into a knowledge base for querying, and the reduced need for monetization-driven features like addictive interactions. They also note the historical context of structured journaling practices and the non-therapeutic nature of using AI for cognitive externalization.

    - Unlucky-Message8866 describes using the `Qwen-3.5` model to process over 10 years of personal documents, creating a comprehensive knowledge base. This setup allows for querying specific personal data, such as past expenses or personal associations, showcasing the model's utility in personal data management and retrieval.
    - Not_your_guy_buddy42 highlights the advantage of local models in avoiding the commercial pressures faced by flagship models. Local models are not designed to be addictive or to extend user interaction unnecessarily, which can lead to a more genuine and less manipulative user experience. This is contrasted with the often flashy and authoritative nature of commercial models.
    - mobileJay77 mentions using the `Mistral 3.2` model, which is compatible with their hardware and capable of handling personal topics without restrictions. This suggests that smaller models like Mistral can be effective for personal use cases, providing flexibility and privacy without the limitations imposed by larger, commercial models.

  - **[Just got my hands on one of these… building something local-first 👀](https://www.reddit.com/r/LocalLLM/comments/1sk3zng/just_got_my_hands_on_one_of_these_building/)** (Activity: 441): **The image depicts an NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition GPU, which the user plans to integrate into a high-performance local-first computing setup. The build includes a `9950X` CPU, `128GB RAM`, and a `ProArt board`, indicating a focus on advanced AI and server tasks rather than gaming. The user aims to achieve multi-user concurrent inference and maintain local control over data, avoiding reliance on external API providers. They are exploring technologies like `vLLM` and `llama.cpp` for structuring the system to handle multiple users efficiently, with plans to potentially add a second GPU for scalability.** One commenter suggests joining an RTX 6000 Discord community for advice, indicating a collaborative environment for users of this high-end GPU. Another comment humorously notes the temptation to purchase such a powerful GPU, reflecting the allure of cutting-edge hardware.

    - Sticking_to_Decaf shares a detailed setup using the RTX 6000, recommending the use of `vLLM` with the `cu130 nightly image`. They highlight running a large model like `Qwen3.5-27B-FP8` with a `kv cache dtype` at `fp8_e4m3`, achieving a max context length of `160k tokens` while utilizing only `55%` of VRAM. Performance metrics include `80-90 tps` for single requests and over `250 tps` for multiple concurrent requests. This setup also accommodates `whisper-large-v3`, an embedding model, and a reranker model, with additional room for swappable LoRAs.




## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI Model and Benchmark Launches


  - **[OpenRouter Just announced a New 100B model](https://www.reddit.com/r/Bard/comments/1skfbvf/openrouter_just_announced_a_new_100b_model/)** (Activity: 240): **OpenRouter has announced a new model named "Elephant Alpha," which is a `100B` parameter model designed to deliver state-of-the-art performance with a focus on token efficiency. The model is particularly noted for its capabilities in code completion, debugging, document processing, and supporting lightweight agents. This announcement positions "Elephant Alpha" as a competitive offering in the AI model landscape, emphasizing its efficiency and broad application potential.** Commenters speculate that "Elephant Alpha" might be related to the "Grok" model, as such models often appear on OpenRouter first. There is also a consensus that it is not a Google model, as Google typically does not disclose parameter counts for their proprietary models.

    - Nick-wilks-6537 and Artistic_Survey461 discuss the possibility that the new 100B model announced by OpenRouter is the 'Grok' model. They mention that this inference is based on probing and analysis by users on platform X, suggesting that OpenRouter often hosts such models under hidden or unnamed providers initially.
    - Capital-Remove-6150 comments on the performance of the new model, stating that it does not appear to be state-of-the-art (SOTA) or near SOTA in tests. This implies that while the model may have a large parameter count, its performance might not match the leading models in the field.
    - SomeOrdinaryKangaroo notes that the model is unlikely to be from Google, as Google typically does not disclose parameter counts for their proprietary models. This suggests that the model's origin is likely from a different organization that is more transparent about such details.




### 2. Sam Altman Security Incidents

  - **[Sam Altman’s home targeted in second attack](https://www.reddit.com/r/singularity/comments/1sjtebt/sam_altmans_home_targeted_in_second_attack/)** (Activity: 2227): ****Sam Altman's** San Francisco residence was targeted in two attacks: a Molotov cocktail incident and a shooting. The latter involved a Honda sedan, captured on surveillance, with suspects Amanda Tom and Muhamad Tarik Hussein arrested. The vehicle's license plate led to police action. No injuries were reported. [Read more](https://sfstandard.com/2026/04/12/sam-altman-s-home-targeted-second-attack/).** Commenters criticized the media for disclosing Altman's address, citing privacy concerns, and discussed the security measures billionaires take, such as relocating to secure compounds.


  - **[Sam Altman's home targeted in drive-by shooting hours after firebomb attack](https://www.reddit.com/r/OpenAI/comments/1sk82sc/sam_altmans_home_targeted_in_driveby_shooting/)** (Activity: 1088): ****Sam Altman**, CEO of **OpenAI**, was reportedly targeted in a drive-by shooting at his home, following a firebomb attack earlier. The incidents occurred within hours of each other, raising concerns about the safety of high-profile tech executives. Details about the attacks are sparse, but they highlight potential security vulnerabilities for individuals in prominent positions within the tech industry.** The comments reflect a mix of socio-economic concerns and speculative connections to broader societal issues, such as wealth disparity and potential unrest, rather than technical debate.


  - **[Another murder attempt on Sam Altman, as gunshots are fired at his residence](https://www.reddit.com/r/ChatGPT/comments/1skwdp7/another_murder_attempt_on_sam_altman_as_gunshots/)** (Activity: 1087): **Two suspects, Amanda Tom and Muhamad Tarik Hussein, were arrested in San Francisco for allegedly firing gunshots near the residence of **Sam Altman**, CEO of **OpenAI**. This marks the second attack on Altman in a short period, following a previous firebombing attempt. The suspects face charges of negligent discharge of firearms, and multiple weapons were confiscated during their arrest. The incidents are reportedly unrelated. More details can be found in the [original article](https://www.usatoday.com/story/news/crime/2026/04/13/sam-altman-house-attack/89586825007/).** Commenters speculated humorously about the possibility of time travelers being involved, reflecting a dystopian view of current events. There is a satirical suggestion that Altman might retreat to a private island to develop advanced AI technologies.



### 3. AI Model Performance and Configuration

  - **[Claude isn't dumber, it's just not trying. Here's how to fix it in Chat.](https://www.reddit.com/r/ClaudeAI/comments/1sjz1hg/claude_isnt_dumber_its_just_not_trying_heres_how/)** (Activity: 1726): **The Reddit post discusses a perceived reduction in the performance of the AI model **Claude**, attributed to a configuration change rather than a model downgrade. Users of Claude Code can revert to the previous behavior by typing `/effort max`, but chat users lack a direct toggle. A workaround involves setting custom instructions in the chat interface to encourage thorough reasoning and comprehensive analysis. This approach reportedly restores Claude's ability to process context deeply and provide detailed responses. The post highlights that these instructions act as strong signals to the model, compensating for the lack of direct control over effort settings.** Commenters debate the balance between token efficiency and response depth, suggesting a 'Spartan mode' for concise yet deep responses. Another comment notes that Claude's system prompt allows it to ignore user preferences if deemed irrelevant, suggesting that style settings might be more effective than user preferences for controlling reasoning effort.

    - m3umax discusses the importance of using styles over user preferences in Claude's system prompts. They highlight that Claude's web system prompt allows it to ignore user preferences if deemed irrelevant, suggesting that styles are more effective. They provide examples of styles for different reasoning efforts, with a 'high version' set at `99` and a 'medium version' at `85`, allowing users to switch thinking levels easily.
    - Medium-Theme-4611 points out that Claude's perceived laziness is due to its token-saving behavior. They suggest that instructing Claude to 'research and dive deep' can counteract this, implying that Claude's default behavior prioritizes efficiency over thoroughness unless explicitly directed otherwise.
    - sidewnder16 draws a parallel between Claude and Gemini, noting that both require explicit system instructions to perform tasks effectively. This suggests that without clear directives, these models may not operate at their full potential, highlighting the importance of detailed prompts for optimal performance.

  - **[Claude Code (~100 hours) vs. Codex (~20 hours)](https://www.reddit.com/r/ClaudeCode/comments/1sk7e2k/claude_code_100_hours_vs_codex_20_hours/)** (Activity: 1421): **The post compares **Claude Opus 4.6** and **Codex GPT-5.4** in a real-world engineering context, focusing on a complex 80k LOC Python/TypeScript project. **Claude** is described as fast and interactive but often requires manual oversight and tends to ignore guidelines, leading to incomplete tasks and architectural issues. In contrast, **Codex** is slower but more deliberate, adhering strictly to guidelines and producing cleaner, more maintainable code without needing constant supervision. The author notes that while Claude is suitable for rapid prototyping, Codex is preferable for enterprise-level software development due to its thoughtful approach and adherence to best practices.** Commenters generally agree with the author's assessment, noting that Codex's slower, more deliberate approach results in higher quality output. However, some users find Codex's communication style overly verbose and sometimes uncooperative, which can be frustrating. Despite these issues, Codex is praised for its competence and reliability in completing tasks autonomously.

    - Temporary-Mix8022 discusses the comparative performance of GPT-5.4 and Opus 4.6, noting that both models are equally capable of solving problems, with no significant performance gap. However, they criticize Codex's communication style, which tends to be overly verbose and formatted in bullet points, making it difficult to parse for experienced developers. They also mention Codex's tendency to disagree due to its reinforcement learning (RL) training, which can be frustrating for users with significant experience.
    - The user highlights a specific issue with Codex's reinforcement learning training, which seems to prioritize safety and disagreement, potentially leading to unproductive interactions. They express frustration with Codex's inability to focus on tasks, suggesting that its training for web app interactions might be to blame. Despite these issues, they acknowledge Codex's effectiveness in completing tasks autonomously, often outperforming Opus in this regard.
    - Temporary-Mix8022 seeks advice on optimizing Codex/GPT-5.4's communication settings, as they struggle with the model's tendency to oscillate between being overly verbose and lacking detail. They express a desire for a more balanced communication style, indicating that despite their programming expertise, they find managing the language model's output challenging.

  - **[The golden age is over](https://www.reddit.com/r/ClaudeAI/comments/1sjqn2e/the_golden_age_is_over/)** (Activity: 4149): **The post discusses a perceived decline in the quality of consumer and prosumer access to Large Language Models (LLMs) like Claude, ChatGPT, Gemini, and Perplexity. The author notes that Claude, which previously excelled in analyzing text conversations, now performs poorly, making mistakes and showing disengagement. ChatGPT is criticized for its overly enthusiastic responses, Gemini for hallucinations, and Perplexity for lacking insightful analysis. The author suggests that high-quality LLM access may now require enterprise-level investment, hinting at potential issues with computational resources or strategic throttling by companies. The post references an article from [ijustvibecodedthis.com](http://ijustvibecodedthis.com/) supporting these observations.** Commenters suggest that the decline in perceived quality might be due to users becoming more adept at prompting, thus encountering the models' limitations more clearly. Another perspective highlights that foreign and open-source models are stepping in to fill the gap left by US-based companies, which are seen as throttling their models to meter intelligence.

    - CitizenForty2 highlights a performance issue with Opus, noting it consumes more tokens and takes longer compared to Sonnet. After switching back to Sonnet, they report not experiencing the common issues others face, suggesting Sonnet may be more efficient or stable for their use case.
    - kaustalautt discusses the trend of foreign and open-source models filling the gap left by US-based companies, which are perceived to throttle intelligence. They suggest international markets are countering this by adopting open-source approaches, potentially offering less restricted access to AI capabilities.
    - bl84work provides a comparative analysis of AI models, noting that Gemini has a high rate of hallucinations, while Claude exhibits self-correction behavior, interrupting itself to rectify inaccuracies. ChatGPT is described as confidently incorrect, highlighting differences in model behavior and reliability.



# AI Discords

Unfortunately, Discord shut down our access today. We will not bring it back in this form but we will be shipping the new AINews soon. Thanks for reading to here, it was a good run.
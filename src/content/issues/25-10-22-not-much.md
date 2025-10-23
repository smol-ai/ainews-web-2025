---
id: MjAyNS0x
title: not much happened today
date: '2025-10-22T05:44:39.731046Z'
description: >-
  **LangChain & LangGraph 1.0** released with major updates for reliable,
  controllable agents and unified docs, emphasizing "Agent Engineering."
  **Meta** introduced **PyTorch Monarch** and **TorchForge** for distributed
  programming and reinforcement learning, enabling large-scale agentic systems.
  **Microsoft Learn MCP** server now integrates with tools like **Claude Code**
  and **VS Code** for instant doc querying, accelerating grounded agent
  workflows. **vLLM** improved inference correctness with token ID returns and
  batch-invariant inference, collaborating with **Ray** for orchestration in
  PyTorch Foundation. **OpenAI** launched **ChatGPT Atlas**, a browser agent
  with contextual Q&A and advanced safety features, though early users note
  maturity challenges and caution around credential access.
companies:
  - langchain
  - meta
  - microsoft
  - openai
  - pytorch
  - ray
  - claude
models:
  - vllm
  - chatgpt-atlas
topics:
  - agent-frameworks
  - reinforcement-learning
  - distributed-computing
  - inference-correctness
  - serving-infrastructure
  - browser-agents
  - security
  - middleware
  - runtime-systems
  - documentation
people:
  - hwchase17
  - soumithchintala
  - masondrxy
  - robertnishihara
  - cryps1s
  - yuchenj_uw
---


a quiet day.

> AI News for 10/21/2025-10/22/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 7314 messages) for you. Estimated reading time saved (at 200wpm): 528 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

If you are deeply passionate/advocating AI coding at work, the AIE CODE speakers (ex Workshop) were [announced today](https://x.com/aiDotEngineer/status/1981062300254818356). The full list is on [the website](https://www.ai.engineer/code/2025) and last round of applications started today. Sponsorships have sold out and tickets will sell out soon too.

[List of speakers for the AI Engineer Code Summit 2025, featuring experts from various companies and research institutions.](https://resend-attachments.s3.amazonaws.com/gMAzQRqiIlzI2KY)

---

# AI Twitter Recap

**Agent frameworks, orchestration, and RL tooling (LangChain/LangGraph 1.0, PyTorch Monarch + Forge, MCP ecosystem)**

- **LangChain & LangGraph 1.0 (Python + TypeScript)**: Major rewrite focused on reliable, controllable agents. Highlights: a new `create_agent` template; provider-agnostic “standard content blocks”; middleware for controllability and context engineering; and durable, human-in-the-loop execution via LangGraph runtime. Unified docs across LangChain, LangGraph and LangSmith are live, and the team is explicitly leaning into “Agent Engineering.” Announcements and deep dives: [@hwchase17](https://twitter.com/hwchase17/status/1981030005229670438), [@LangChainAI](https://twitter.com/LangChainAI/status/1981030195873333269), [roundtable recap](https://twitter.com/bromann/status/1981076440780013666).
- **PyTorch’s new distributed & RL stack**: Meta introduced two building blocks for large-scale agentic systems: [Monarch](https://twitter.com/PyTorch/status/1981020264474231030) (a distributed programming framework for orchestrating clusters, debugging, and pretraining) and [TorchForge](https://twitter.com/PyTorch/status/1981035379126890748) (a PyTorch-native RL library with high-performance components and examples). The push underscores an end-to-end path from research to production for agent workloads. Teaser: [@soumithchintala](https://twitter.com/soumithchintala/status/1980812457301160196).
- **MCP goes mainstream**: The Microsoft Learn MCP server makes official docs instantly queryable inside tools like Claude Code and VS Code — no auth, OpenAI-compatible — accelerating grounded agent workflows: [@code](https://twitter.com/code/status/1981076900471562579). LangChain docs now ship with MCP built-in: [@masondrxy](https://twitter.com/masondrxy/status/1981003281603428670).

**Inference correctness and serving infra (vLLM + Ray)**

- **Eliminating retokenization drift in agent RL**: vLLM’s OpenAI-compatible endpoints can now return token IDs directly — add `"return_token_ids": true` — preventing subtle string→token mismatches that destabilize RL (e.g., JSON reformatting, template differences). Great collaboration with Agent Lightning/MSR, and a worthwhile read for anyone building self-improving agents: [@vllm_project](https://twitter.com/vllm_project/status/1981017184769061153).
- **Batch-invariant inference**: vLLM introduced a one-flag switch for bitwise-equivalent results across batch sizes (including prefill): set `VLLM_BATCH_INVARIANT=1`. This dramatically simplifies debugging and reproducibility of serving stacks: [@vllm_project](https://twitter.com/vllm_project/status/1981088861506982041).
- **vLLM x Ray, now in the PyTorch Foundation**: Coordination and placement matter as inference gets complex. Talks at PyTorchCon emphasized cross-node parallelism, prefill-decode disaggregation, prefix-aware routing, and wide expert parallelism — with Ray providing orchestration and vLLM the engine: [@robertnishihara](https://twitter.com/robertnishihara/status/1981112722361372924), [@vllm_project](https://twitter.com/vllm_project/status/1981045521671393441).

**Browser agents and safety (OpenAI Atlas launch + reactions)**

- **OpenAI’s ChatGPT Atlas**: The browser integrates an agent that can act on pages and introduces “Ask ChatGPT” (contextual page Q&A) plus defense-in-depth safeguards: logged-out mode for actions without credentials, a “Watch Mode” for sensitive sites, and rapid response to prompt injection campaigns. OpenAI details extensive red-teaming and new training to ignore malicious instructions — while noting attacks remain an unsolved frontier: [@cryps1s](https://twitter.com/cryps1s/status/1981037851279278414), [@OpenAI](https://twitter.com/OpenAI/status/1981098271901962439).
- **Reality check from practitioners**: Early users report “agent mode” frequently overthinks and stalls; caution is urged especially when granting access to credentials or email. Expect an extended maturity curve: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932), [follow-up](https://twitter.com/Yuchenj_UW/status/1980847565819302116), [password risk](https://twitter.com/Yuchenj_UW/status/1980855677397659869).

**Multimodal surge: OCR/VLMs and 3D/video**

- **OCR is hot (open, fast, cheap)**: AI2’s Apache-2.0-licensed olmOCR 2 lands with new datasets, unit-tested synthetic training, and claims SOTA — with costs around ~$178 per 1M pages; models + FP8 and a public demo are out: [@allen_ai](https://twitter.com/allen_ai/status/1981029159267659821), [overview](https://twitter.com/mervenoyann/status/1981040748133826918). DeepSeek-OCR is reportedly outpacing Qwen3-VL on community tests; deployment templates and endpoints are proliferating ([Baseten](https://twitter.com/basetenco/status/1980924381217104338), [HF Endpoints catalog](https://twitter.com/ErikKaum/status/1980965155145216336)). A short list of competitive OCR/VLMs was compiled here: [@HarveenChadha](https://twitter.com/HarveenChadha/status/1981055277408669934).
- **New VLMs and datasets**: Qwen3-VL arrives on HF with 1M context and stronger GUI/video reasoning: [@HuggingPapers](https://twitter.com/HuggingPapers/status/1980809413045940553). Liquid AI’s tiny VLM, **LFM2-VL-3B**, posts 51.8% MM-IFEval and 71.4% RealWorldQA with multilingual OCR strength and low hallucination rates: [@LiquidAI_](https://twitter.com/LiquidAI_/status/1980985540196393211). Hugging Face unveiled **FineVision** (24M curated multimodal samples across 185 subsets) to standardize VLM pretraining: [@HuggingPapers](https://twitter.com/HuggingPapers/status/1981093262912819418).
- **3D/video generation**: Tencent open-sourced **Hunyuan World 1.1 (WorldMirror)**, a single-pass, feed-forward video/multi-view to 3D recon model that outputs point clouds, depth, normals, camera params, and 3D Gaussians in seconds on a single GPU — with flexible geometric priors for consistency: [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1980930623536837013). On video gen, see new long-form attention approaches in UltraGen and MoGA: [@_akhaliq](https://twitter.com/_akhaliq/status/1980952631544799705), [MoGA](https://twitter.com/_akhaliq/status/1980952993563349127).

**Frontier models and methods (DeepSeek v3.2, memory layers, token efficiency, biomedical)**

- **DeepSeek v3.2 (685B MoE) focus on long-context cost/speed**: Attends to “most relevant tokens,” delivering 2–3× faster long-context inference and 6–7× lower processing cost than v3.1. MIT-licensed weights; API pricing listed at $0.28/$0.028/$0.42 per 1M input/cached/output tokens; optimized for Huawei/China chips. Performance is broadly similar to v3.1, with small gains on coding/agent tasks and slight dips on some math/science: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1980846573681520824).
- **Continual-learning “memory layers”**: A proposed layer of input-independent KV memories with fine-tuning only on high-TF-IDF slots is generating serious interest for scalable continual learning ([thread + paper summary](https://twitter.com/giffmana/status/1980869216149619009)). Follow-ups raised two practical points: include a “sink” slot to allow “no memory used” selections, and watch out for perf/throughput hits from random memory accesses in the inner loop: [@BlackHC](https://twitter.com/BlackHC/status/1981022197415068129), [@gallabytes](https://twitter.com/gallabytes/status/1981038852539371969).
- **Token efficiency via images + biomedical resolution**: Researchers continue exploring encoding text as images to nearly halve token counts for multimodal LLMs ([paper/code](https://twitter.com/iScienceLuvr/status/1980942325573648703)), and “native-resolution” training/inference measurably improves biomedical MLLMs ([paper](https://twitter.com/iScienceLuvr/status/1980944519001727281)). Also notable: a first pass at a transformer foundation model for MEG neuroimaging data, MEG-GPT ([abstract](https://twitter.com/iScienceLuvr/status/1980945270369399234)).

**Adjacent compute and datasets**

- **Verifiable quantum advantage (Google)**: Using the “Quantum Echoes” (OTOC) measurement on the Willow chip, Google reports the first verifiable quantum advantage — 13,000× faster than the best classical algorithm on a top supercomputer — with potential applications in NMR-based molecular modeling for materials/drug discovery. Peer-reviewed in Nature; verification via repetition on other quantum devices/experiments: [@sundarpichai](https://twitter.com/sundarpichai/status/1981013746698100811), [@GoogleQuantumAI](https://twitter.com/GoogleQuantumAI/status/1981016219340648778).
- **Agent training data at scale**: IBM + University of Washington released a 1.5M task scenario dataset on Hugging Face to push agent evaluation and “getting things done” workflows: [@IBMResearch](https://twitter.com/IBMResearch/status/1981066891062817274). Also, Stanford’s new CME295 (Transformers & LLMs) launched, and DeepMind + UCL published a free AI Research Foundations curriculum: [@omarsar0](https://twitter.com/omarsar0/status/1981030346037612847), [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1980962352775176637).

**Top tweets (by engagement)**

- Google’s “Quantum Echoes” claim of verifiable quantum advantage (13,000× speedup) on Willow: [@sundarpichai](https://twitter.com/sundarpichai/status/1981013746698100811).
- OpenAI’s Atlas adds “Ask ChatGPT,” reading the current page for instant answers: [@OpenAI](https://twitter.com/OpenAI/status/1981098271901962439).
- Early reactions to ChatGPT’s agent/browsing UX and safety concerns: [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932).
- Tencent’s Hunyuan World 1.1 open-sourced feed-forward video→3D world reconstruction: [@TencentHunyuan](https://twitter.com/TencentHunyuan/status/1980930623536837013).
- Higgsfield Popcorn launches AI storyboarding tool with consistent character editing: [@higgsfield_ai](https://twitter.com/higgsfield_ai/status/1981110992630341928).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen Team Contributions to llama.cpp

- [**Qwen team is helping llama.cpp again**](https://www.reddit.com/r/LocalLLaMA/comments/1oda8mk/qwen_team_is_helping_llamacpp_again/) (Activity: 1035): **The image is a screenshot from GitHub showing a post by a member of the Qwen team, detailing their contributions to the** `llama.cpp` **project. The post mentions specific technical updates, such as fixing Vision Transformer (ViT) positional embeddings and correcting the DeepStack implementation. This indicates ongoing collaboration and improvements in the** `llama.cpp` **project, which is a popular implementation for running large language models efficiently on consumer hardware. The image of cupcakes with bounding boxes likely serves as a visual demonstration of a feature related to object detection or image processing capabilities within the software.** The comments reflect a sentiment that non-Chinese AI labs have slowed in their output, while Chinese companies like Alibaba are rapidly advancing. There is also appreciation for the Qwen team's hands-on coding approach and a suggestion for them to assist with the Qwen3-Next architecture.
    - The comment by -p-e-w- highlights a perceived stagnation in AI model releases from major non-Chinese labs like Google, Meta, and Microsoft, contrasting it with the rapid development pace of Chinese companies such as DeepSeek and Alibaba. This suggests a shift in the AI landscape where Chinese firms are becoming more prominent in pushing the boundaries of AI technology.
    - YearZero discusses the potential for the Qwen team to assist with the Qwen3-Next architecture, referencing a specific pull request on GitHub ([link](https://github.com/ggml-org/llama.cpp/pull/16095)). The comment implies that the project is nearing completion and could benefit from peer review to finalize the architecture, indicating a collaborative approach to development.
    - GreenPastures2845 provides a link to a GitHub issue comment ([link](https://github.com/ggml-org/llama.cpp/issues/16207#issuecomment-3432273713)), which may contain further technical insights or discussions related to the llama.cpp project. This suggests ongoing community engagement and technical discourse around the project.
- [**hey](https://www.reddit.com/r/LocalLLaMA/comments/1od1hw4/hey_zai_two_weeks_was_yesterday/) [Z.ai](http://z.ai/)[, two weeks was yesterday](https://www.reddit.com/r/LocalLLaMA/comments/1od1hw4/hey_zai_two_weeks_was_yesterday/)** (Activity: 514): **The image is a meme highlighting a delay in the release of "GLM 4.6 Air" by [Z.ai](http://z.ai/), as mentioned in a Twitter exchange. Ivan Fioravanti humorously anticipates the release with a GIF, while [Z.ai](http://z.ai/) responds with a typical developer's promise of readiness in 'two weeks,' a common trope in software development for indefinite delays. The comments reflect a supportive community attitude, emphasizing the voluntary nature of open-source contributions and expressing eagerness to test the new model, particularly in comparison to existing versions like q4 (GGUF or AWQ) from REAP GLM 4.6.** Commenters generally express understanding and patience, acknowledging the voluntary and open-source nature of the work by [Z.ai](http://z.ai/), and show interest in comparing the upcoming release with existing models.
    - Leflakk expresses interest in testing the Q4 quantization formats, specifically GGUF or AWQ, for the REAP GLM 4.6 model. This suggests a focus on comparing performance and efficiency between these quantization methods, which are crucial for optimizing model deployment in resource-constrained environments.
    - nuclearbananana mentions that the 'two weeks' timeline is approximate, suggesting that software development timelines can be fluid and subject to change. This highlights the importance of flexibility in project management, especially in open-source projects where contributions are often voluntary.
    - inkberk emphasizes the significant contributions of [Z.ai](http://z.ai/) to the open-source community, implying that their work has had a substantial impact on the development and accessibility of AI technologies. This underscores the value of community-driven projects in advancing technological innovation.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Quantum Computing Breakthroughs by Google

- [**Google breakthrough in using Quantum computing for drug discovery and material science**](https://www.reddit.com/r/singularity/comments/1odbbbr/google_breakthrough_in_using_quantum_computing/) (Activity: 1125): **Google has announced a significant breakthrough in quantum computing with their Willow chip, which achieved a verifiable quantum advantage using an algorithm called Quantum Echoes. This algorithm is reported to be** `13,000 times faster` **than classical algorithms, enabling the explanation of molecular interactions through nuclear magnetic resonance. This advancement holds potential for substantial impacts in drug discovery and material science, marking a pivotal step in the real-world application of quantum computing. For more details, see the [Google blog post](https://blog.google/technology/research/quantum-echoes-willow-verifiable-quantum-advantage/).** Commenters are discussing the timeline of Google's quantum computing milestones, with some expressing curiosity about the future milestones and the challenges that lie ahead. The roadmap for these milestones can be found on [Google's Quantum AI roadmap](https://quantumai.google/roadmap).

### 2. Humanoid Robots and AI Interaction

- [**AheadForm unveils their new male humanoid robot face Origin M1**](https://www.reddit.com/r/singularity/comments/1od7n5c/aheadform_unveils_their_new_male_humanoid_robot/) (Activity: 667): **AheadForm has introduced the** `Origin M1`**, a new male humanoid robot face, as announced on [X (formerly Twitter)](https://x.com/XRoboHub/status/1980886176845517175). The design aims to enhance human-robot interaction by providing a more relatable and expressive interface. The unveiling highlights the ongoing trend in robotics to create more lifelike and emotionally engaging machines, though specific technical details such as the materials used, the range of expressions, or the underlying AI technology were not disclosed in the announcement.** The comments reflect skepticism and critique of the robot's design, with some users questioning the aesthetic choices and the necessity of a gendered appearance, indicating a broader debate on the anthropomorphism in robotics.
- [**thinking about Honda ASIMO rn 🥀**](https://www.reddit.com/r/singularity/comments/1odab1q/thinking_about_honda_asimo_rn/) (Activity: 448): **The image features Honda's ASIMO, a humanoid robot known for its advanced mobility and interaction capabilities, standing next to a person. ASIMO, developed by Honda, was a pioneering project in humanoid robotics, showcasing advanced walking, running, and interaction abilities. Despite its technological achievements, ASIMO is often seen as a project that was ahead of its time, similar to the Segway, and has not led to widespread adoption of humanoid robots. The comments reflect a sentiment that Japan, despite its prowess in industrial robotics, has not maintained a leading position in the development of modern humanoid robots, which are now being advanced by companies outside Japan.** Commenters express disappointment that Japan, despite its historical leadership in robotics, has not continued to lead in humanoid robotics, with modern advancements being driven by non-Japanese companies.
    - Distinct-Question-16 highlights the operational limitations of Honda's ASIMO, noting that it could only operate for '30 minutes - 1 hour' and required '3 hours for charging'. This reflects the significant advancements needed in battery technology and energy efficiency for humanoid robots to become more practical and autonomous in real-world applications.

### 3. Meta Policy Changes Impacting ChatGPT

- [**Lol openai cooked meta here**](https://www.reddit.com/r/OpenAI/comments/1od4xmy/lol_openai_cooked_meta_here/) (Activity: 1057): **The image is a meme highlighting a tweet from OpenAI about Meta's policy change that will affect the functionality of 1-800-ChatGPT on WhatsApp starting January 15, 2026. The tweet reassures users that ChatGPT will still be accessible through other platforms like an app, website, and browser. This reflects ongoing tensions and competitive dynamics between major tech companies, particularly in how they manage third-party integrations and platform policies.** The comments reflect a critical view of the post, with users expressing skepticism about idolizing one large tech company over another and dismissing the post as unimportant.
- [**Yea truly luckily**](https://www.reddit.com/r/ChatGPT/comments/1od4yqn/yea_truly_luckily/) (Activity: 502): **The image is a meme highlighting a tweet from OpenAI about Meta's policy change that will disable the 1-800-ChatGPT service on WhatsApp by January 15, 2026. OpenAI reassures users that ChatGPT will remain accessible through other platforms like apps, websites, and browsers. This change reflects ongoing adjustments in platform policies affecting AI service integrations.** Commenters are questioning the utility of accessing ChatGPT via WhatsApp, suggesting it might be for accessibility. There is also skepticism about the browser's functionality, with concerns about potential restrictions typical of OpenAI's approach.
    - *si1endeath* raises a concern about the browser's restrictions, noting that it blocks access to many sites, which aligns with **OpenAI's** approach to content moderation. This could impact users who rely on unrestricted browsing for research or other purposes.
    - Erik-AmaltheaFairy questions the operational model of a browser integrated with GPT, speculating on potential limitations for free users, such as search caps or prompts to upgrade to a paid subscription. This reflects broader concerns about monetization strategies in AI services.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**1. AI Models in the Wild: Performance, Costs, and Quirks**

- **Gemini 3 Release Rumors Fizzle Out**: Initial speculation across **LMArena** and **Perplexity** pointed to a late October release for **Gemini 3**, but updated reports now suggest a preview in December with an official release in January, according to [AI.google.dev](https://ai.google.dev/). Meanwhile, experiments on **Lithiumflow** (purported to be **Gemini 3 Pro**) showed impressive coding capabilities but failed on more specific prompts.
- **Claude Models Empty Developer Wallets**: Engineers in the **Cursor Community** and **MCP Contributors** Discord reported that using models like **Claude 4 Sonnet** is incredibly expensive, with costs reaching **$7 per request** on max mode. In multi-agent setups, costs ballooned to **$7-8 per action**, forcing some to abandon the platform for custom API solutions.
- **Sora 2 and Friends Show Off, With Limits**: Users in the **OpenAI** and **Nous Research** communities are generating videos with **Sora 2**, but report a daily limit of **30 videos**. Discussions also highlighted that **Veo 3** video generation lacks sound and model choice, while users noted **GPT-4o** still successfully performs accents that **GPT-5** fails to deliver on.

**2. The Developer Experience: Tools, IDEs, and APIs**

- **Cursor IDE Riddled With Security Holes and Bugs**: A [BleepingComputer article](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) circulated in the **Cursor Community** highlighted over **94 n-day security issues** in the IDE due to outdated **Chromium engines**. This news came amidst a major [Cursor outage](https://status.cursor.com/) and reports of a bug disabling the **"apply"** function, frustrating coders.
- **OpenRouter Sharpens Tool-Calling with :exacto Endpoints**: **OpenRouter** launched **:exacto** endpoints to improve tool-calling accuracy by routing requests to providers with superior structured-output performance, as detailed in their [announcement post](https://openrouter.ai/announcements/provider-variance-introducing-exacto). This aims to address performance variance, with initial benchmarks showing a **material lift in tool-call success** for models like `qwen/qwen3-coder:exacto`.
- **Engineers Wrestle with Massive Tool Contexts**: In the **MCP Contributors** Discord, developers managing servers with **60+ tools** are hitting context limits due to verbose tool descriptions. One engineer devised a workflow with just **3 tools** (list, describe, invoke) to manage over **50 CLI actions**, demonstrating the need for streamlined approaches to avoid overwhelming models and incurring high costs.

**3. Hardware and Systems Optimization: Pushing the Limits of Performance**

- **GPU Debates Pit Cloud Rentals Against Physical Rigs**: In the **Unsloth AI** discord, engineers debated the economics of renting a **DGX Spark** with **200GB VRAM** for **$4k** versus buying an **RTX 6000 Pro** for a similar price. Over in the **Yannick Kilcher** Discord, researchers without local GPUs recommended renting an **RTX 3090** on [vast.ai](https://vast.ai/) for as low as **$0.15 per hour** to slash experiment runtimes.
- **PyTorch's Helion Kernel Gets Benchmarked and Questioned**: Following the launch of **PyTorch Helion** ([blog post](https://pytorch.org/blog/helion/)), members of the **GPU MODE** community challenged its reported **14x speed-up**, calling it unrealistic. They argued the [int4_gemm implementation](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) should be benchmarked against fused kernels like **Marlin** or a simple **Triton gemm kernel** for a fairer comparison.
- **Mojo Language Segfaults on High-End Hardware**: Developers in the **Modular** community reported that the latest nightly build of **Mojo** causes a segfault when loading `Llama-3.1-8B-Instruct-GGUF` on an **H100** GPU. The issue appears to stem from the GPU attempting to run a **bf16** version of the model while the CPU correctly runs the **q4_k** dequantization kernels.

**4. The Shifting AI Landscape: New Releases and Big Tech Moves**

- **Google's Quantum Chip Claims 13,000x Speedup Over Supercomputers**: **GoogleAI** announced a major quantum computing milestone, using its **65-qubit Willow chip** and the **Quantum Echoes algorithm** to perform a task **13,000x faster** than top supercomputers, according to a [post on X](https://x.com/googleai/status/1981022228801307035). The news, shared in **Latent Space**, sparked discussions about the implications for cryptography and scientific modeling.
- **Unsloth and PyTorch Announce Quantization Collab**: **Unsloth AI** and **PyTorch** are collaborating on a new **Quantization Aware Training (QAT)** initiative, as per an [announcement on X](https://x.com/UnslothAI/status/1981021761782317368). The collaboration sparked technical questions among engineers about its implementation, particularly whether it would leave the vision encoder untouched.
- **Open-Source Tools for Agents and LLM Systems Gain Traction**: The **HuggingFace** community saw the launch of **Fenic**, a new tool that integrates directly with **Hugging Face Datasets** to create versioned context for agents, with its [repo on GitHub](https://github.com/typedef-ai/fenic). In **GPU MODE**, a new [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems) repository was introduced to curate papers and resources on LLM deployment and optimization.

**5. User Woes and Platform Problems: Bugs, Billing, and Bad Support**

- **Perplexity's Referral Program Under Fire for Fraud Flags**: Users in the **Perplexity AI** Discord are reporting major issues with the referral program, claiming legitimate leads are not being counted and some accounts received a **0 penny** payout after being flagged for fraud. This has led to widespread speculation about the criteria for a *quality lead* and frustration with the program's unreliability.
- **OpenAI Users Baffled by Mysterious Daily Charges**: An **OpenAI** user reported being billed a fixed **$15 USD daily** charge since October 9th, despite having deleted all API keys and projects. They shared [screenshots](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&) and are seeking answers from the community as support has not resolved the issue.
- **Manus.im Users Decry Bait-and-Switch on Credits and Suspend Accounts**: The **Manus.im** community is in an uproar over the platform's credit system, with users feeling they were *bait and switched* on a **Pro Plan** that no longer offers unlimited credits. Alongside credit frustrations, new users reported their accounts were being suspended immediately after entering payment details, with no clear reason or path to resolution.

---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Referral Referendum Results in Dubious Debacle**: Users are reporting issues with the Perplexity AI referral program, including leads not counting and the system potentially flagging legitimate referrals as fraudulent, resulting in a **0 penny** payout after review, with much speculation about tactics, and the value of the old referral program.
   - Many discussed the criteria for a *quality lead* and potential VPN and virtual machine tactics to get more referrals; some reported success while others vented frustration; additionally, some experienced issues with the **Comet browser** resetting accounts and Cloudflare checks.
- **Perplexity Pro's Palava: Unlimited or Limited?**: Users debated perceived limitations on **Perplexity Pro's** *unlimited access*, after a user shared an image showing usage limits, sparking discussions about potential updates to subscription terms and limits.
   - One user inquired whether the **Perplexity API** could access **ChatGPT5** and **Claude**, to which another member responded that the API is limited to **Sonar**.
- **Gemini Glimmers, China possibly creeping**: Speculation arose regarding the release date of **Gemini 3**, with mentions of a potential December release, while also discussing limitations Western AIs are placing on their products possibly setting the stage for China to take the lead.
   - Some expressed concerns that the east might take over software and AI development.
- **Android Antics Annoy App Users**: Users reported technical issues with Perplexity, including the **Android app UI reverting to an older version** and persistent *something went wrong* errors on the mobile app, Comet browser, and web versions.
   - The issues, in particular, pertained to generating files, and some users found the app crashing when writing code; one user suggested the web app as a workaround.
- **Mathematical Musings Merge with Artful Aesthetics**: A member shared [a link to their published article on the Math Art of Artist 0thernes](https://generativeai.pub/the-math-art-of-artist-0thernes-not-the-typical-96e009060bc1) and asked for feedback.
   - Additionally, they also shared links to several **Perplexity AI** pages, including pages about [Time-Based Researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0), and [Trump's Lies and Fact Denials](https://www.perplexity.ai/page/trump-s-lies-and-fact-denials-cgZjiGKUTvuj6sTWzTkm0A).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Release Hopes Dashed**: Initial speculation pointed to a **Gemini 3** release by late October, but later reports suggest a preview in December with an official release in January [according to AI.google.dev](https://ai.google.dev/).
   - Many had anticipated a 2024 release, but were ultimately disappointed by the updated timeline.
- **Veo 3 Generates Videos Without Model Choice or Sound**: Users are generating videos with **Veo 3**, noting that they lack model choice during generation and sound may be absent.
   - When present, sound in generated videos appears randomly selected, sparking curiosity about the potential of *Ling-1t*.
- **Lithiumflow's Gemini 3 Code Shows Promise and Faults**: Experiments with code generation on **Lithiumflow** (purported to be **Gemini 3 Pro**) reveal impressive capabilities in areas like voxel art and Rubik's cube solving, but also failures on more specific prompts.
   - Despite issues, it was generally superior to models like **Sonnet 4.5** and *OpenAI*, but it failed on the more specific prompts.
- **OpenAI Faces Fraud Claims Over Math Discovery**: **OpenAI** faces accusations of fraud, with critics alleging it regurgitated data from existing research.
   - Commentary dismissed the idea of any AI truly making a *'discovery'*, labeling **OpenAI** as *the biggest fraudsters in modern history*.
- **Lithiumflow and Orionmist: Same Model, Different Access?**: The hypothesis that **Lithiumflow** and **Orionmist** are the same model, with **Lithiumflow** having access to Google Search, emerged but remains unconfirmed.
   - A user suggested that **Gemini 2.5 Pro** could be superior to both, and also *Bytedance's* **Seed** has very good image understanding.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 4 Sonnet Bank Account Drainer**: Members complained that using **Claude 4 Sonnet** is quite expensive, reporting bills of around **$7 per request on max mode**.
   - The high costs occurred despite having more requests and using **Claude 4 Sonnet**.
- **Cursor "Apply" Tool Hits the Brakes**: Users are reporting that the new updates cause agents to mistakenly believe they are in *ask mode*, thus disabling the **"apply"** function.
   - Frustrated members are resorting to manually copying code by looking at the full output and using *show code*.
- **Cursor and Windsurf IDEs Teeming With Security Holes**: A [BleepingComputer article](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) highlights that **Cursor and Windsurf IDEs** are vulnerable due to outdated **Chromium and V8 engines**, resulting in over **94 n-day security issues**.
   - The vulnerabilities could lead to denial of service or even remote code execution.
- **The Great Cursor Outage of 2024 Stops Coders**: Users experienced widespread issues with **Cursor**, including connection failures and the inability to send messages; the [Cursor status page](https://status.cursor.com/) confirmed the **outage**.
   - One user bemoaned that their *threejs game is at a standstill*.
- **Cursor's Free Tier: From Claude Sonnet to Haiku**: Users confirmed that on the standard **$20 plan**, once **Claude Sonnet** credits are exhausted, the system automatically switches to the **free Haiku model**.
   - One user mentioned they would rather use a 3rd party API key when running on *extreme poverty*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth SLA Costs Not Available**: A user asked about the cost of an **Unsloth SLA**, but learned that it is currently **N/A**.
   - The user wasn't able to determine any specific pricing or service-level agreements.
- **Internet IPv4 Scans Result in Honeypot Bans**: A member mentioned scanning the entire **IPv4 internet**, and warned about being reported by **international honeypots** and needing to avoid certain networks.
   - Other members jokingly suggested using a botnet for scanning, but acknowledged that scanning the internet is harmful.
- **LLMs Fail to Crack Encrypted Data**: A user wanted to use a small LLM to classify encrypted messages, but another member noted that good encryption turns data into **pure noise**, making classification nearly impossible.
   - One member joked that *to analyze a modern crypto cipher to look for patterns using ML you'd probably have to harness the entire planet's GPU resources for the next million+ years*.
- **PyTorch and Unsloth Team up for QAT**: There is a new **Quantization Aware Training (QAT)** collab with **Pytorch**, and here is a [link](https://x.com/UnslothAI/status/1981021761782317368) to the announcement.
   - A member cited [this paper](https://www.arxiv.org/abs/2509.11986) and [this X post](https://x.com/SravanthiSinha/status/1980867770498838560?) to ask whether the collab leaves the vision encoder untouched.
- **DGX Spark vs RTX 6000**: Members debated the cost tradeoffs of getting a **DGX Spark** with **200 GB VRAM** for **$4k** vs buying an **RTX 6000 Pro** for **$3500-4k** on eBay.
   - The **DGX Spark**, equipped with **8x H100s**, was lauded as being able to train a **2-4B model** with full SFT in real-time.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Meta Kills ChatGPT WhatsApp Access**: Meta's policy change will shut down **1-800-ChatGPT** on **WhatsApp** after **January 15, 2026**, according to [openai.com](https://openai.com/index/chatgpt-whatsapp-transition/).
   - The team reminded users that **ChatGPT** is still available via the app, website, and browser.
- **Sora 2 Video Has Daily Limits**: Users report that **Sora 2** limits video generation to **30 videos a day**, with some bypassing restrictions by using VPNs.
   - One user lamented it takes only *1-2h and my Sora is Done xD*.
- **Baffling Bills Bedevil Bot Builders**: A user reported **fixed daily charges of $15 USD** on their OpenAI account since October 9th, even after deleting all API keys, with [screenshots](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&).
   - The user seeks answers on the OpenAI community.
- **AI OS: the Future?**: A member speculates that future **operative systems** might be driven by **LLM** assistance, using AI to handle system processes and user interactions.
   - This would be a departure from existing OS designs.
- **Circumventing Copyright Conundrums**: Members discussed how to skirt copyright issues, with one suggesting to describe a copyrighted character *(guy in a red and blue costume with black spider web symbols on it)*.
   - Another member warned that this might not be sufficient, stating that it is a **copyrighted IP** and they can't help with that.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Andromeda-alpha Sees Launch**: A new stealth model, **Andromeda-alpha**, specializing in **image and visual understanding**, has launched via [OpenRouter.ai](https://openrouter.ai/openrouter/andromeda-alpha) for community feedback.
   - All prompts and outputs are logged for model improvement, with the model intended for trial use only and not for production.
- **OpenRouter Pins Point Exacto**: OpenRouter launched **:exacto** endpoints for **higher tool-calling accuracy**, routing requests to providers with better structured-output performance, as documented in [a blog post](https://openrouter.ai/announcements/provider-variance-introducing-exacto).
   - Launch models include `moonshotai/kimi-k2-0905:exacto`, `deepseek/deepseek-v3.1-terminus:exacto`, `z-ai/glm-4.6:exacto`, `openai/gpt-oss-120b:exacto`, and `qwen/qwen3-coder:exacto`, with benchmarks showing **material lift in tool-call success**.
- **Objective-AI Exposes AI Confidence**: Ronald, CEO of [Objective-AI](https://objective-ai.io/), introduced a **Confidence Score** for OpenAI-compliant completion choices, derived from smarter mechanisms, not direct AI assessment.
   - Objective-AI enhances **cost-efficiency**, allowing users to maximize smaller models and employs **OpenRouter** to access a wide array of **Large Language Models (LLMs)**.
- **RooCode Relieves Resources**: **RooCode** offers free models such as **Grok Code Fast 1**, **Grok 4 Fast**, **Supernova 1 million**, and **Deepseek Chat 3.1**, providing an alternative to token-peddling platforms.
   - One user stated they've *nuked billions of tokens on roo cloud in the past couple months and it always went brrrrrrrrrrrr an was never backed off or rate limited.*
- **OpenRouter: Watch Out For Chutesflation**: Users reported experiencing unexpected high costs with OpenRouter, with one user reporting **$5** being consumed in just **23 days** despite minimal usage, leading to concerns about *chutesflation*.
   - One user stated, *You don’t have a choice, you’re most likely already using Chutes*, implying OpenRouter routes users to them by default.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Renting GPUs on Vast.ai Beats Local Runs**: For independent researchers without GPUs, renting an **RTX 3090** on [vast.ai](https://vast.ai) costs around **$0.15 per hour**, drastically cutting experiment runtimes.
   - Members noted experiment runtimes locally might be **5-8 hours**, and suggested using cloud GPUs instead of Google Colab due to resource limitations.
- **Navigating Function Calling Finetuning**: Members discussed finetuning small language models for function calling, and suggest using **lm-format-enforcer** or **Lark Grammar** to constrain output to follow a tool calling **JSON** format.
   - They added that the library **llguidance** offers syntax for structuring tool calls with minimal added latency and high accuracy, with the structure: `[Thinking] I will now call tool {tool_name}. [tool JSON]`.
- **Transformer Circuits Paper Gets Rave Reviews**: A member described the [transformer circuits post](https://transformer-circuits.pub/2025/linebreaks/index.html) as the *best paper to read* this week, noting its potential *big implications framework wise*.
   - The member noted its rigorous analysis of *his own idiocy and ignorance*, but the other member seemed to agree despite this.
- **DeepSeek Engineers Standardize Jsonnet Configuration**: DeepSeek uses the **jsonnet scheme** for their projects.
   - A member stated *if they're using it in DeepSeek I should try it myself and see how good it is*.
- **Vibe Code IDE Released from Amazon**: Amazon's **Vibe Code IDE** exited its invite-only beta, giving users **500 credits** to start, and it's [designed to be 'spec based'](https://kiro.dev/blog/waitlist-is-over/), working around specifications for features and implementations, rather than solely prompts.
   - Members also noted that **Kiro**, like many **AI IDEs**, is also a **VScode fork**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **New Repo Awesome LLM Systems Takes Off**: A member introduced a new repository, [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems), featuring curated papers, blogs, and resources focusing on the systems aspect of **large language models**.
   - The repository aims to provide a comprehensive guide for those interested in the engineering challenges and solutions surrounding **LLM deployment and optimization**.
- **CuPy GPU Pointer Dominates PyTorch GPU Pointer**: A member highlighted significant performance disparities between **CuPy GPU pointers** and **PyTorch GPU pointers** when used in a custom **MatMul kernel**, noting a performance delta, and a bottleneck during **DLPack** conversion of CuPy arrays to PyTorch tensors, detailed in [this screenshot](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68faade6&is=68f95c66&hm=d84234753a2510107fb4d7ecd73bbf01b7e07a92a430333bafa04d79be3e8bd3).
   - The performance lag persists despite the same numerical results, suggesting potential inefficiencies in the **DLPack conversion** process.
- **warpGroup.arrive Injection Sparks Concern**: Verbose compilation reveals the injection of `warpgroup.arrive` to enable register use in **GMMA functions**, prompting worries that all **wgmma** might be sharing the same registers.
   - The member suggested that the injection is *to allow the use of registers in GMMA*.
- **Helion Benchmarking Baselines Get Questioned**: After the **Helion blog post** went live, a member challenged the reported **14x speed-up**, deeming it unrealistic even for memory-bound operations compared to **fp16**, referencing the [int4_gemm implementation](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) as a non-fused kernel.
   - It was suggested to benchmark against specialized kernels like **Marlin**/**Machete** or a simple **Triton gemm kernel** with `tl.dot_scaled` for a more equitable comparison; the **Helion** team invited everyone to a "meet the **Helion devs**" event.
- **Leaderboard Crowns New sort_v2 King**: A user clinched **first place** on the `sort_v2` leaderboard with submission ID `66238`, clocking in at **8.68 ms** on B200 and **16.3 ms** on A100, plus successful runs on H100 at **6.60 ms** and L4 at **52.7 ms**.
   - Another user had multiple entries and reached **first place** on L4 at **52.6 ms** and A100 at **16.0 ms** with submission ID `66241`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI Called 'Devil Machine'**: Members are calling generative **AI** a *'devil machine'*, due to understandable animosity from the artist community around image generation, and because of a concern that **critical thinking skills** are eroding.
   - Countering this sentiment, one member posited that **AI's** tendency to hallucinate actually causes them to *think critically a lot more*.
- **Qwen3 Embedding Accuracy Leaps Forward**: The newer quants of **Qwen3 embedding 8b** are exhibiting improved accuracy with *roocode* code indexing.
   - Reportedly, confidence scores are now higher for relevant queries and substantially lower for irrelevant ones compared to **mxbai-embed-large:latest**.
- **LM Studio Plugin support for Third-Party LLMs**: Members discussed the possibility of using **LM Studio** to communicate with third-party **LLMs** using an **API key**.
   - It was confirmed that with plugins, currently in closed beta, this functionality will be supported via an [OpenAI-compatible endpoint](https://lmstudio.ai/fuutott/openai-compat-endpoint-v2).
- **Chatterbox emerges as Local AI TTS Solution**: In response to a search for **AI voice** solutions, **Chatterbox** was recommended as a *really good* **local AI TTS** option.
   - It was confirmed that **Chatterbox** supports multiple languages, making it a versatile choice.
- **Vietnam not a hotbed for 4090 Bargains**: Despite inquiries about purchasing a **cheap 3090/4090** in Vietnam, a member pointed out that Vietnam is a supplier of **4090s** to China.
   - Therefore, bargain prices are unlikely.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Architects Refactor with AI**: A member prompted the AI to refactor code *like a sane person* using the prompt: `as a senior architect focused on modular code and maintainability`, for changes they prompt: `do not make changes or write code, answer question: do you have enough info to make these updates?` and `please create the minimal changeset (no tests)`.
   - The AI can thus **refactor code** in a way a human can easily follow and extend, and it avoids a re-write and/or re-design.
- **Fenic Connects to Hugging Face Datasets**: The open source project **Fenic** now directly integrates with **Hugging Face Datasets**, allowing users to hydrate version context straight from the Hub and safely tool it for agents, as detailed in the [documentation](https://huggingface.co/docs/hub/datasets-fenic).
   - Fenic, akin to e2b for compute, enables **data snapshotting**, agent context creation, and exposure of **MCP tools** via a dataframe API similar to pandas, with the [Fenic repo available on GitHub](https://github.com/typedef-ai/fenic).
- **Multi-Model Collab Improves Request Quality**: Evaluations on **multi-model collaboration** are underway, focusing on reducing the single-user rate of hallucinations per request and enhancing overall request quality; more details are available on [this blog](https://facilitair.ai).
   - Results have been achieved with **sequential collaboration**, and two OS repos using collaboration are currently at least at v1.
- **Databomz Workspace Blasts Off**: A member introduced **Databomz**, a workspace and Chrome extension for saving, organizing, and sharing prompts with features like tags, versions, and folders, with more information available at [www.databomz.com](http://www.databomz.com/).
   - The **Forever Free plan** includes most of the core features, and the creator is seeking feedback from active prompt users, also available at [github.com/Lnrchaos/NeSy-CML](https://github.com/Lnrchaos/NeSy-CML).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Firebase Studio Gets Lovable Full-Stack Upgrade**: Logan Kilpatrick confirmed [Firebase Studio integration](https://ai.studio/build) coming to the new **AI Studio**, allowing vibe-coding apps and incorporating its speech-to-text feature, but currently using the **Gemini 2.5 Pro** model.
   - Users requested simpler integration of databases, auth, and storage, and Logan invited feedback on unsupported use-cases like Chrome extensions and SolidJS.
- **Lovable Launches AI Shopify Full-Store Integration**: Lovable announced a **Shopify integration** that lets users spin up complete online stores via simple prompts, demoing the flow by launching its own merch store ([lovable.dev/merch](https://lovable.dev/merch)).
   - The feature is available to everyone with a bundled **30-day Shopify trial**, though existing Shopify stores can’t yet be imported or modified.
- **Project Mercury Pays Investment Bankers To Train AI**: [Project Mercury](https://www.entrepreneur.com/business-news/openai-is-paying-ex-investment-bankers-to-train-its-ai/498585) is paying contractors **$150 per hour** to feed financial models into **OpenAI**, expanding the real-world, practical use of AI across business sectors like finance and technology.
   - The project shows the continued demand to train AI on novel domains.
- **GoogleQuantumAI Runs Crysis**: **GoogleAI** announced a new milestone, using the **65-qubit Willow chip** and the **Quantum Echoes (OTOC) algorithm** to run a verifiable task **13,000x faster** than top supercomputers ([post on X](https://x.com/googleai/status/1981022228801307035)).
   - The team discussed implications for cryptography (SHA-256 safety), verifiability of results, real-world timelines for drug discovery and climate modeling, and even running Crysis/Doom.
- **Next.js Exams the Frameworks**: Guillermo Rauch announced [Next.js Evals](https://xcancel.com/rauchg/status/1981037270624076092), open-source ‘exams’ that let any LLM/agent prove it can correctly build with **Next.js** and other supported frameworks.
   - Models like **GPT-5-codex** and **Claude Sonnet 4.5** are currently scoring in the mid-40% range; community asks for real-world tasks, public traces, and cost columns.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Scammer Alert Shakes Manus Community**: A user was accused of being a *fraudster scammer* after allegedly soliciting login access to a paid **Manus** account, leading to a discussion about account security.
   - The accused user claimed to have found *another way* to resolve their issue, while others cautioned against sharing account credentials due to potential theft of personal and bank information.
- **Manus Credit System Stress**: Users expressed confusion and frustration with the **Pro Plan** credit system, with some feeling *bait and switched* due to a now-missing help page promising unlimited credits.
   - Some users are willing to pay $200+/month for an unlimited plan, while others pointed out the need for constant credit management and participation in improvement initiatives to earn free credits.
- **Account Suspensions Plague New Users**: A user reported that their and their girlfriend's account was suspended shortly after entering card details, suspecting that inviting too many employees may have triggered the suspension on **Manus**.
   - Despite the suspension, they seek support on retaining projects already made on the account.
- **Users Suggest Manus Explores Local Compute Utilization**: A user suggested leveraging local compute resources to enhance **Manus** capabilities.
   - This could enable building large native apps, processing huge datasets locally, running resource-intensive AI models, and faster builds using local machine power.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Comet Invitation Sparks Debate**: Members debated whether a free **1-year pro invitation** to [Perplexity Comet](https://www.perplexity.ai/) is valuable.
   - The discussion questioned the current benefits and features of **Perplexity Comet's pro version**.
- **Pacific-Prime Boasts Eco-Friendly Architecture**: Boris from France introduced the [Pacific-Prime architecture](https://huggingface.co/Pacific-Prime/pacific-prime), claiming its **25 layers** with **5 iterations** each can converge without error.
   - The architecture can run on **CUDA** with **1B** parameters on **6GB VRAM**, supposedly *two times more eco than llamacasual architecture*.
- **"Regular" Role Still Awarded**: Members confirmed that the "Regular" role is still given out on a per-case basis.
   - The role is awarded to those who *have been contributing to the community*.
- **Seeking Datasets with System Messages**: A member requested recommendations for **chat datasets WITH system messages**, specifically seeking recent resources.
   - The user expressed difficulty in finding up-to-date datasets of this type.
- **Claude Exhibits Shapeshifting!**: A member shared a post on X about [Claude shapeshifting](https://fxtwitter.com/wesg52/status/1980680563582538099).
   - No further details were discussed.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Encoding Tool Call Properties with Flexible Schemas**: Engineers are looking to encode a property in `inputSchema` that can be either an **array** or an **object** in tool calls, given that `oneOf` is not accepted in the subset of **JSONSchema** used.
   - The challenge lies in defining a flexible schema that LLMs can reason about and clients can accept, despite the limitations of the **JSONSchema** subset, thus complicating the schema definition for tool properties.
- **Navigating JSONSchema Subset Constraints in Tool Calls**: A subset of **JSONSchema** limits the definition of `inputSchema` for tool calls, omitting features like `oneOf` that allow multiple valid schema types for a property.
   - This limitation poses challenges when properties can be either arrays or objects, impacting the ability to specify flexible schemas for tool calls.
- **MCP Server Context Limits Reached with Extensive Tool Descriptions**: Engineers report that MCP servers with **60 tools** reach context limits quickly, even with max plans, due to the size of tool descriptions.
   - Dividing tools into multiple servers wasn't favored by customers, leading to the exploration of efficient context management and cost optimization solutions, since costs were **$7-8 per action** in multi-agent implementations.
- **Tool Workflows Streamlined for Context and Accuracy**: One engineer managed **50+ CLI actions** through a streamlined workflow of **3 tools** involving action listing, description, and invocation, guided by server instructions.
   - Using descriptions and input/output schemas for numerous tools can overwhelm models and exceed context limits, necessitating a streamlined workflow to improve efficiency and accuracy.
- **Engineering Custom Chats To Cut Costs**: To reduce costs, one engineer created a custom chat using direct API connections and optimized model selection, after noticing that running on Claude desktop or ChatGPT quickly accumulated charges.
   - The multi-agent MCP client with model switching incurred costs of **$7-8 per action**, leading them to abandon the setup.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **NPC Voices Sprout via DSPy-ElevenLabs Synthesis**: A member developed a voice generation system for game **NPCs**, leveraging **DSPy** to parse wiki content, and integrating with **ElevenLabs** for voice synthesis, as detailed in [the project's GitHub repo](https://github.com/Gielinor-Speaks/voiceover-mage) and [this devlog video](https://youtu.be/z3DQm2PiKpo).
   - The developer aims to automate voice curation via an **automated judging loop**, using manual selections as training signals to improve the quality of the generated voices.
- **Microsoft's Trace Program Edges Out DSPy?**: Microsoft's [Trace](https://microsoft.github.io/Trace/) program claims an **8% accuracy increase** over equivalent **DSPy** programs, according to a member's shared screenshot.
   - Despite this claim, some members plan to test **Trace** for a fair comparison, expecting to retain more granular control using **DSPy**.
- **Async DSPy Tackles OCR Obstacles**: A member explored **async DSPy modules** for parallel execution, specifically for OCR tasks with high throughput involving **Google Cloud Vision**, **DSPy Attachments library**, and layout via bboxes.
   - Challenges included repetition loop bugs, leading to exploration of alternatives like **paddleocr-vl** or **nanonets** for improved performance.
- **AI & Blockchain Engineer Shows off Expertise**: A member introduced themselves as a **Senior AI & Blockchain Engineer**, specializing in autonomous systems at the intersection of AI and blockchain, boasting experience across **Base**, **Solana**, and **EVM chains**.
   - Their expertise lies in **on-chain AI agents**, **LLM pipelines with real-time data**, and **AI agent orchestration with LangChain/AutoGen**.
- **Trainable Decorator Tickles Fancy**: One member voiced their appreciation for the trainable decorator.
   - It's unclear which trainable decorator, but they seemed very excited by it, simply saying *Oh, I like the trainable decorator! Very good idea*



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Sora's Video Puts on a Show**: A member shared a **Sora**-generated video, showcasing the tool's capabilities and offering a glimpse into its potential, showcased [here](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68fa4f13&is=68f8fd93&hm=ac5d3464bd020d568c770a6fa89ef8ccb2db40420fd45bee25dc3be64c60807f&).
   - The video's release prompted excited discussion regarding the realism and artistic possibilities afforded by **Sora**.
- **Nous Research Brain Trust Requested**: A member requested assistance from **Nous Research** for a research paper aimed at helping kids interested in **AI**.
   - The request highlights the community's interest in educational initiatives within AI.
- **GPT-3.5 Sonnet Sounds its Swan Song**: A member shared a link from fixvx.com, [available here](https://fixvx.com/dbreunig/status/1980733694634770710), expressing a sense of closure with the end of **GPT-3.5 Sonnet**.
   - The posting led to a reflection on the model's strengths and weaknesses relative to others.
- **GPT-5 Gets an Attitude Adjustment**: According to newly shared benchmark results, **GPT-5** has become less friendly, potentially due to the ongoing discussions around sycophancy in AI models.
   - This shift has sparked debate among members on the trade-offs between helpfulness and objective performance.
- **GPT-4o Still Busts Out The Accents**: A user noted that **GPT-4o** still performs a Jamaican accent in voice mode, in contrast to **GPT-5** which claims it can but then fails.
   - The observation was given by a user, noting that this is *one of the few metrics* they care about.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Users Find No Support**: A user reported they received *zero* support for **Kimi** and got no response from the support team. They were told it is not a support server.
   - A community member told them to DM a specific user for help.
- **Moderato and Allegretto Payment Plan Questions Arise**: A user inquired about upgrading to **Moderato** or **Allegretto** and the number of times they could use **OK Computer** under the payment plan.
   - Another user clarified that **Moderato** allows **20** uses per month, linking to a [relevant tweet](https://x.com/togethercompute/status/1980337943169651055).
- **K2 Has Blazing Speed on Together**: A user remarked on the impressive speed of **K2** when used on the **Together** platform.
   - No additional details were provided in the discussion.
- **Partnership Opportunities Explored with Kimi**: A user asked about who to contact regarding partnership opportunities with **Kimi**.
   - Another user suggested DMing a specific individual to pursue the potential partnership.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Missing Key Features**: Members discussed the most important thing that **Mojo** is missing: a *finished type system*, in addition to a wishlist of other features.
   - Other desired features included rounding out the **standard library datatypes**, **proper IO**, a good **async runtime**, an **effect system**, **static reflection**, **compiler plugins**, the ability to deal with more **restrictive targets**, **cluster compute**, **device/cluster modeling**, and some clone of **Erlang's OTP**.
- **Llama-3.1 hits snag on H100**: A member reported a segfault on the nightly build when loading `modularai/Llama-3.1-8B-Instruct-GGUF` on an **H100** GPU, related to the `session.load` function in *llama3/model.py*.
   - The issue seems specific to the GPU, as the model works fine on CPU, particularly with the `modular/max-nvidia-full:nightly` base image, but segfaults when pushed to the GPU.
- **GGUF Models Spark Segfault Sleuthing**: A member speculated that the segfault might persist with **bfloat16** models, given that dequantization kernels for **GGUF** (*q4_k*, etc.) weights have been built only for CPU execution so far.
   - The member confirmed that while the CPU runs the **q4_k** version, the GPU attempts to run the **bf16** version, leading to the segfault.
- **tensor_internal becomes just tensor**: Members announced that the `tensor_internal` package has been renamed to `tensor` ([Discord link](https://discord.com/channels/1087530497313357884/1224434323193594059/1430580184305635339)).
   - This is a critical update in the latest nightly for the **Mojo API**.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Vulkan Sine is Sinful**: The **Vulkan sine function's inaccuracy** causes errors and a member reported *failing a couple of errors* due to this issue.
   - This necessitates a custom sine function which could slow down the process.
- **Renderer Refactoring Required**: **tinygrad/uop/decompositions.py** triggers automatically if your renderer doesn't implement **sin/log/exp**.
   - A member didn't realize this would happen and modified their renderer accordingly to ensure that all **transcendental functions** pass.
- **JITBEAM's Jitted Jewels**: Clarification: **JITBEAM** specifically refers to the **BEAM** of jitted kernels.
   - The JITBEAM represents the BEAM specifically tailored for jitted kernels within the tinygrad framework.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Misused as Benchmarking Tool**: A member admitted to *never using Aider for its original purpose, only for benchmarking*, preferring local models like **Cline** and **RooCode**.
   - This suggests that some users are finding alternative uses for **Aider** beyond its intended design.
- **LLM History Retrieval Proves Difficult**: A user sought *evaluation logs and LLM histories* (the *full chat*), but another member confirmed they also *haven't had luck lately* retrieving those logs.
   - The lack of accessible logs may hinder debugging and analysis efforts for **LLM** interactions.
- **Suspicious Uniswap Portal Link Shared**: A user shared a suspicious link ([uniswap-portal.web.app](https://uniswap-portal.web.app)) related to **Uniswap**.
   - An image attachment associated with the link was flagged as **spam** by image analysis tools, indicating a potential security risk.
- **Aider's Project Maintenance in Question**: A member questioned the project's maintenance status given the absence of version updates since **August 10th** and limited activity.
   - The inquiry raises concerns about the project's ongoing support and development, though no official answer was given.
- **Chutes AI Cheap LLMs Model List Goes Live**: A member shared a [link](https://wuu73.org/r/chutes-models/) to a list of **Chutes AI models** for cheap LLMs, suggesting potential utility for the community.
   - The increasing use cases with **Chutes AI** are expected as community contribution to the model list increases in the near future.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Windsurf Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1430269580735152391)** (1269 messages🔥🔥🔥): 

> `Referral Program Issues, Comet Browser, Perplexity Pro, Gemini 3 Release Date, Technical Issues with Perplexity` 


- ****Referral Referendum: Dubious Dubbing Debacle Dominates Discourse****: Users discuss issues with the Perplexity AI referral program, including leads not counting and the system potentially flagging legitimate referrals as fraudulent, leading to a **0 penny** payout after review; many speculate about the criteria for a "quality lead" and potential VPN and virtual machine tactics to get more referrals, with some users reporting success and others frustration.
   - Some also report issues with Perplexity Pro limits and some users complaining that they are not getting credited leads even when their friends are downloading Comet from their links and using it.
- ****Comet Craze: Users Debate Browser Booty or Booty Call?****: Users shared referral links for **Comet browser**, with some offering a month of Pro in exchange for downloads, and they are experiencing issues such as the Comet browser resetting accounts, encountering Cloudflare checks, and facing difficulties in getting referrals to count.
   - Some users speculated that the value of old referral programs had been reduced.
- ****Pro Palava: Unlimited Access or Limited Lies?****: Users discuss perceived limitations on Perplexity Pro's "unlimited access", with one user attaching an image showing usage limits, sparking conversation about potential updates to subscription terms and limits.
   - There was a bounty system in place where members would receive money for leads and for reporting bugs.
- ****Gemini Jitters: Google's Glimmer of Hope or Glitch in the Matrix?****: Speculation arose regarding the release date of **Gemini 3**, with mentions of a potential December release, and discussing limitations western AIs are placing on their products possibly setting the stage for China to take the lead on new products.
   - There was also discussions of whether the east will take over software and AI, or whether the west will continue to be more advanced.
- ****Android Antics: UI U-Turn Upsets Users****: Users are reporting technical issues with Perplexity, including the **Android app UI reverting to an older version** and persistent "something went wrong" errors on the mobile app, Comet browser, and web versions of Perplexity, in particular generating files.
   - One user reported the app was crashing when writing code, others suggested the web app as a workaround.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430299672295247944)** (6 messages): 

> `Shareable Threads, Time-Based Researcher, Trump's Lies, Information Warfare, Math Art` 


- ****Shareable Threads** Reminder**: A member was reminded to ensure their thread is set to **Shareable**.
   - There was an attachment included as a visual guide on how to set threads to shareable.
- **Published on Math Art**: A member shared a [link to their published article on the Math Art of Artist 0thernes](https://generativeai.pub/the-math-art-of-artist-0thernes-not-the-typical-96e009060bc1) and asked for feedback.
   - The article discusses the artist's unique approach to merging math and art.
- **Links to Perplexity Pages**: A member shared links to several Perplexity AI pages, including [Reconciling Atomic Curvature](https://www.perplexity.ai/page/reconciling-atomic-curvature-a-27py5fYSRiyMr.91kDpiDAPerplexity), [Time-Based Researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0) and [Trump's Lies and Fact Denials](https://www.perplexity.ai/page/trump-s-lies-and-fact-denials-cgZjiGKUTvuj6sTWzTkm0A).
   - Another Perplexity AI page was shared that described the [logical endpoint of rhetorical trajectory and sophisticated information warfare](https://www.perplexity.ai/page/-g5fwV1GzRlqm_q32kk3y9g).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1430316660950302812)** (4 messages): 

> `Perplexity API, ChatGPT5, Claude, Sonar` 


- **Sonar Only on Perplexity API**: A member inquired whether the **Perplexity API** could access **ChatGPT5** and **Claude**, or just **Sonar**.
   - Another member responded that the API is limited to **Sonar**.
- **User expresses excitement about API**: A user indicated excitement without specifying why.
   - The user wrote *idk why , hehehehe*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1430269427261116426)** (1034 messages🔥🔥🔥): 

> `Gemini 3 Release Date Speculation, Image to video generation with Veo 3, Gemini 3 performance benchmark, OpenAI LibGen fraud accusations, Lithiumflow and orionmist` 


- **Gemini 3 release expectations rise and then crash**: Members speculated on a **Gemini 3** release, with some suggesting a release on the 25th of October, but others stated this was unlikely, and that November was a more realistic timeframe, depending on a potential new versioning system for AI [this may not be accurate](https://ai.google.dev/)
   - Ultimately, it was reported that the actual release may be much later, suggesting a preview in December and an official release in January, dashing the dreams of many for a 2025 release.
- **Generating Image to Video with Veo 3**: Members discussed generating videos with **Veo 3**, but it's been said that one can't choose the model when generating videos and a user said their video generated with no sound.
   - It was also said that the sound generated in videos seems to be randomly selected. Also some people find *Ling-1t* is intriguing.
- **Gemini 3 shows improved performance with a side of failure**: Members experimented with **Gemini 3** code generation on several test cases including voxel art, a wind tunnel, a Rubik's cube solver and other coding problems and non web GUI libraries.
   - It's been suggested that while it can be impressive, **Lithiumflow's** code (purported to be **Gemini 3 Pro**) did not run with some reporting *Lithiumflow failed on the more specific prompt*, but it was generally superior to other options such as **Sonnet 4.5** and *OpenAI* models.
- **OpenAI Accused of Fraudulent Claims in Math Discovery**: Accusations of fraud were levied against **OpenAI**, saying that it just regurgitated data from already available research.
   - A user said that *So far, no AI have made any kind of 'discovery' despite tinfoil hat claims on YT, so OAI will go down as the biggest fraudsters in modern history*.
- **Lithiumflow and Orionmist as Different Flavors of Gemini 3**: A user proposed that **Lithiumflow** and **Orionmist** are the same model, but the former one has access to Google Search, but this claim is unconfirmed, with some in disagreement.
   - A user suggested that **Gemini 2.5 Pro** could be superior, also it was stated that models from *Bytedance* called **Seed** had very good image understanding.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1430270761175744582)** (540 messages🔥🔥🔥): 

> `Claude 4 Sonnet cost analysis, Cursor's new "apply" issue, Cursor IDE Security vulnerabilities, The Great Cursor Outage of 2024, Cursor free models after credit exhaustion` 


- **Claude 4 Sonnet Expensive Taste**: Members noted that using **Claude 4 Sonnet** can become costly, with one user mentioning they paid **$7 per request on max mode** and their costs were still higher overall despite having more requests.
- **Cursor new "apply" issue emerges**: The latest updates cause agents to tell users they are in ask mode, disabling the "apply" function when chatting; users can now apply code directly in the chat after the chat provides it.
   - One member joked that *the apply tool is failing most of the time* so they look at the full output and *show code* instead.
- **Cursor Windsurf IDEs Compromised by Old Chromium**: A [BleepingComputer article](https://www.bleepingcomputer.com/news/security/cursor-windsurf-ides-riddled-with-94-plus-n-day-chromium-vulnerabilities/) confirmed that **Cursor and Windsurf IDEs** have over **94 n-day security issues** due to outdated **Chromium and V8 engines**, posing risks such as denial of service or remote code execution.
- **The Great Cursor Outage of 2024**: Users reported widespread issues with **Cursor**, including connection failures and inability to send messages.
   - The [Cursor status page](https://status.cursor.com/) confirmed the **outage**, with one user comically lamenting that their *threejs game is at a standstill*.
- **Cursor free models usage after paid credit exhaustion**: Users discussed what happens on the standard **$20 plan** after **Claude Sonnet** credits are used up, confirming the system switches to the **free Haiku model**.
   - One user still preferred to use a 3rd party API key when in *extreme poverty*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430560153760960512)** (2 messages): 

> `Cursor + Linear Integration, UI Error Messages` 


- **Cursor-Linear Integration Works Intermittently**: A user reported that the **Cursor integration with Linear** sometimes works after retrying, even without any apparent changes.
   - The user tried pinging **Cursor** in **Linear** and the integration worked upon retry.
- **UI Error Message is Misleading**: A user suggested updating the **UI error message** when a local branch is not pushed to the remote repository.
   - The user suggested replacing the error message with: *"The branch {branch name} does not exist in the remote repository."* because they were confused that the issue was with their environment configuration.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1430270902234255541)** (175 messages🔥🔥): 

> `Unsloth SLA cost, IPv4 scanning, LLMs for analyzing encrypted messages, Quantization Aware Training (QAT) collab with Pytorch, Reward function for coherence` 


- **Unsloth SLA Costs N/A**: A user inquired about the cost of an **Unsloth SLA**, but the response indicated it is currently **N/A**.
   - There's no additional discussion about specific pricing or service-level agreements.
- **Members Talk IPv4 Internet Scanning Perils**: A member mentioned scanning the entire **IPv4 internet**, but warned about being reported by **international honeypots** and needing to avoid certain networks to minimize issues with abuse reports.
   - Other members jokingly suggested using a botnet for scanning, but acknowledged that scanning the internet makes it *worse for everyone*.
- **LLMs Analyze Encrypted Messages? Not So Fast!**: A user wanted to use a small LLM to classify encrypted messages and binaries/hexcode, but another member pointed out that good encryption turns data into **pure noise**, making classification difficult.
   - One member joked that *to analyze a modern crypto cipher to look for patterns using ML you'd probably have to harness the entire planet's GPU resources for the next million+ years*.
- **Quantization Aware Training (QAT) colabs with Pytorch**: There is a new **Quantization Aware Training (QAT)** collab with **Pytorch**, and here is a [link](https://x.com/UnslothAI/status/1981021761782317368) to the announcement.
   - A member inquired whether the collab leaves the vision encoder untouched, and cited [this paper](https://www.arxiv.org/abs/2509.11986) and [this X post](https://x.com/SravanthiSinha/status/1980867770498838560?).
- **Coherence Reward Function Conundrums**: A member asked about creating a reward function to judge text coherence during RL, suggesting using an LLM to rate coherence from 1 to 5.
   - Another member suggested using **Flesch-Kincaid readability tests** or fine-tuning **ModernBert/Colbert**, noting that using an LLM would be too costly. There's also a [link](https://github.com/jxmorris12/language_tool_python) to **language_tool_python**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1430272012256940153)** (4 messages): 

> `Introductions, Channel Guidelines` 


- **Introductions to New Channel**: A member named motithewizerd_16739 introduced themselves to the new channel saying *"Hi all new here. Great stuff"*.
   - Another member theyruinedelise welcomed motithewizerd_16739.
- **Channel Guidelines Posted**: Unsloth AI welcomed everyone to the new channel and reminded them to follow the guidelines.
   - Namely to refrain from promotions since this is an intro channel.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1430282757690495177)** (207 messages🔥🔥): 

> `DGX Spark vs RTX 6000, OSS VSCode Fork for Local LLM Usage, Training completion timing, Token Initialization Importance` 


- **Debating DGX Spark vs RTX 6000 Pro**: Members debated getting a **DGX Spark** with **200 GB VRAM** for **$4k** vs buying an **RTX 6000 Pro** for **$3500-4k** on eBay.
   - The **DGX Spark**, equipped with **8x H100s**, was touted as being able to train a **2-4B model** with full SFT in real-time, with someone joking it *has to be a scam*.
- **OSS VSCode for Local LLMs**: A member requested the creation of an open-source fork of **VSCode** or any IDE built around local LLM usage, with good FIM and chat capabilities.
   - They were looking for something *free and doesn’t send me shit out to sam altman* but despaired at needing to *waste 3 months building it myself.*
- **Pod Idling Cost Anxiety**: A member expressed frustration when training finishes at inconvenient times, causing the pod to sit idle and incur unnecessary running costs.
   - Another member said they *tune the num of steps to match my day* to avoid the running costs of the pod.
- **Token Initialization**: A member emphasized the importance of good **token initialization**, especially when not dedicating a long stage to teaching them.
   - They also linked to an *interesting lil discussion* about audio encoding [Karpathy Tweet](https://x.com/karpathy/status/1980397031542989305).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1430281447347982460)** (63 messages🔥🔥): 

> `Unsloth with custom TRL Trainer, Qwen3-30B-A3B-Instruct-2507 KeyError, Fine-tuning Vision Models w/ Unsloth, Llama 3.1 Data Augmentation, FastLanguageModel Crash` 


- **Gradient Glitches with Custom TRL Trainer?**: A PhD student is facing gradient propagation issues when using a custom variant of the TRL trainer with Unsloth, particularly with modified `generation_and_score` and `compute_loss` functions.
   - They're seeking advice from anyone who has experience with custom TRL trainer implementations in Unsloth.
- **Qwen3's Fast Downloading Falls Flat!**: A user encountered a `KeyError` during the downloading of the `unsloth/Qwen3-30B-A3B-Instruct-2507` model when using Unsloth's `FastModel.from_pretrained`.
   - The error occurred after attempting to fetch 16 files, as seen in the traceback.
- **Vision Model Finetuning without Chat Templates?**: A user inquired about the possibility of fine-tuning vision models with Unsloth without using a chat template or user-assistant pairs dataset, aiming for a more text-completion style finetuning approach.
   - The user didn't get a definitive answer but is looking for ways to perform vision model finetuning similarly to text completion tasks.
- **Multiple QA versions: Data Augmentation or Overfitting?**: A user training Llama 3.1 Instruct version is wondering whether having multiple versions of the same question in their Q/A dataset is a good idea for data augmentation.
   - A member said that  *it would help reducing overfitting compared to simply training multiple epochs of the same data*.
- **FastLanguageModel Flounders on Parallel Fronts!**: A user reported that `FastLanguageModel` crashes during parallel requests, while `FastVisionModel` functions correctly.
   - The user is seeking assistance to resolve this issue with the language model.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1430351853694226583)** (4 messages): 

> `External GPUs on ARM MacBooks, Thunderbolt 5` 


- **Nvidia GPUs now run on ARM MacBooks**: TinyCorp successfully runs an **Nvidia GPU** on an **ARM MacBook** through **USB4** using an external GPU docking station, according to [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station).
- **Thunderbolt 5 port comes to Pros**: Mac 'Pros' now have **Thunderbolt 5**.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430362843701706793)** (1 messages): 

> `Meta, WhatsApp, ChatGPT, Policies` 


- **Meta Changes Policies, Ruins the Party**: Meta changed its policies so **1-800-ChatGPT** won't work on **WhatsApp** after **Jan 15, 2026**.
- **OpenAI offers lifelines for ChatGPT addicts**: Luckily we have an app, website, and browser you can use instead to access **ChatGPT**.
   - More info can be found on the [chatgpt-whatsapp-transition page](https://openai.com/index/chatgpt-whatsapp-transition/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1430270607379140659)** (345 messages🔥🔥): 

> `Sora 2 video generation limits, Sora 2 availability in the UK, AI animating Peppa Pig, Daily charges on OpenAI account, Copyrighted material accepted vs not accepted in Sora 2` 


- **Sora 2 video generation hits daily cap**: Users report Sora 2 limits video generation to **30 videos a day**, leading to quick exhaustion of the service and discussions about using VPNs to bypass restrictions.
   - Some users joked that the limit is *too low* with one stating it takes only *1-2h and my Sora is Done xD*.
- **UK users crave Sora 2 access via VPN**: Users inquired about **Sora 2's availability in the UK**, with suggestions to use VPNs, though availability is reportedly limited to the USA.
   - One user shared a [tweet](https://x.com/hamdfakhr8/status/1980693381585285182?t=zxNAhQv-3PuBDxGIDQpeTQ&s=19) implying that Sora 2 is only available in the USA.
- **AI Peppa Pig animated episodes incoming?**: A user sought recommendations for **AI programs** to animate 10-minute episodes of **Peppa Pig** and generate AI voices for the characters, linking to [Opus Pro agent](https://www.opus.pro/agent?ref_id=5XHUEZ3WP).
   - No specific program recommendations were provided.
- **Mysterious Daily $15 OpenAI Charges Plague Users**: A user reported **fixed daily charges of $15 USD** on their OpenAI account since October 9th, even after deleting all API keys and provided [screenshots](https://cdn.discordapp.com/attachments/998381918976479273/1430552464108556308/Captura_de_Tela_2025-10-22_as_10.43.57.png?ex=68fa314d&is=68f8dfcd&hm=ac45c8620b631401aa0bc85ff59c1098a590e604ce27d207135a45798d95ad4c&).
   - The user was seeking assistance in identifying the cause of these persistent charges.
- **Sora 2's Copyright Craziness Catches VeggieTales**: Users discussed **Sora 2's handling of copyrighted material**, noting inconsistencies where some prompts involving IPs like *VeggieTales* are accepted while others are blocked.
   - A member stated that *in general you are not allowed to violate copyrights even if the check happens to be miss it*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1430287585472938024)** (27 messages🔥): 

> `LLM OS, Lagging ChatGPT, Gemini, Sora, Realtime API` 


- **LLMs Powering Future Operative Systems?**: A member speculated that if **AI** continues to advance, we might see entire **operative systems** driven by **LLM** assistance.
   - This could involve AI handling system processes and user interactions more dynamically than current OS designs.
- **Custom RPG GPT Chat Lag Solution**: A user reported that a long chat with a custom **RPG GPT** was lagging and freezing in the browser, but worked fine in the mobile app.
   - Another user suggested using **scrcpy** as a potential workaround, suggesting it might be a **hardware issue**.
- **Lazy Loading Implementation**: A member mentioned that **Gemini** fixed lag issues by implementing *lazy loading*, which loads messages as the user scrolls.
   - Another member shared a **ChatGPT Lightsession** extension which might improve the experience.
- **Sora 2 Story Mode Conundrum**: A user asked about the location of **story mode** for **Sora 2**.
   - Another user tried to explain where to find the **edit option** for **storyboard generation**, but admitted losing access to **Sora 2** in the meantime, describing to *go to your profile in Sora 2, generate your first video, click on it and the edit option should appear somewhere*.
- **Questioning Realtime API's Newness**: A member questioned what makes the **GPT-realtime model** new, given that the **realtime API** was launched over a year ago, linking to [Introducing the Realtime API](https://openai.com/index/introducing-the-realtime-api/).
   - The discussion did not provide a definitive answer, focusing instead on other topics.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1430271122355786001)** (15 messages🔥): 

> `Copyright issues with AI generated content, Prompt engineering for copyrighted material, Impact of typos on AI model output, Building context in AI prompts, Generating video with AI from text prompts` 


- **Avoiding Copyright Claims with AI-Generated Content**: Members discussed how to avoid copyright issues when generating content with AI, with one user suggesting to circumvent the issue by describing a copyrighted character *(guy in a red and blue costume with black spider web symbols on it)* instead of naming the character.
   - Another member warned that this might not be sufficient, stating that it is a **copyrighted IP** and they can't help with that.
- **Prompt Typos Affect AI Output Quality**: A user asked if typos in prompts, such as *"crete a hagman gam"* instead of *"create a hangman game"*, negatively affect the output.
   - One member said that while simple prompts are less affected, **typos can cause confusion in complex prompts** where ambiguity is a problem, especially if the typo changes the meaning.
- **Building Context and Memory in AI Prompts**: Members discussed strategies for **building context** in AI prompts, focusing on *thoughtful, hierarchical structure and relevant content*.
   - Another member noted that typos force the model to guess, potentially leading to vaguer outputs unless the request is strengthened with additional guidance.
- **Generating cinematic video trailer with AI - Gemini**: A member asked for help with prompt engineering to generate cinematic video trailer with AI, using Gemini.
   - The user provided the [prompt](https://chatgpt.com/share/68f92672-a0f8-8011-a8a7-1d86973ff476) and a [video](https://cdn.discordapp.com/attachments/1046317269069864970/1430677767673876551/gemini_generated_video_D4CFBE60.mov?ex=68faa600&is=68f95480&hm=8172c60da7184e89c71e1c664a3e08ddee0bdda44c3f728669e041aefeed96e5&) as example of the output.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1430271122355786001)** (15 messages🔥): 

> `Copyright infringement, Avoiding copyright, Typos in prompts, Building context, cinematic trailer for ninjas` 


- **Copyright police say 'no web-slinging!'**: A user inquired about generating "ultimate spiderman web swinging in New York City for Sora AI v2," but was told that [generating copyrighted IP is not allowed](https://community.openai.com/tos).
   - In response, a member suggested using descriptive terms like *"guy in a red and blue costume with black spider web symbols on it"* to avoid copyright issues.
- **Typos trigger AI tirades?**: A member questioned whether typos in prompts negatively affect the output of AI models, such as asking to *"crete a hagman gam"* instead of *"create a hangman game".
   - It was suggested that typos force the model to guess, potentially affecting the quality of the output and resulting in more vague responses, particularly if the typo introduces ambiguity.
- **Context is key?**: A user inquired about [best practices for building context in prompts](https://community.openai.com/tos) and significant use cases where memory and context make a difference.
   - Responses highlighted the importance of *thoughtful, hierarchical structure and relevant content*, as well as tailoring context to personal preferences and specific workflows.
- **Nifty ninja server needs video**: A user shared a [prompt and Gemini-generated video](https://cdn.discordapp.com/attachments/1046317269069864970/1430677767673876551/gemini_generated_video_D4CFBE60.mov?ex=68faa600&is=68f95480&hm=8172c60da7184e89c71e1c664a3e08ddee0bdda44c3f728669e041aefeed96e5) for a short cinematic trailer announcing the release of a role-playing server inspired by ninjas and legends.
   - The user sought advice on improving the prompt to achieve a better rendering of the concept of a vertical trailer (9:16 format, duration about 10 seconds), with a dark, mystical, and emotional Japanese-inspired aesthetic.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1430293802685562983)** (2 messages): 

> `Andromeda-alpha, :exacto` 


- **New Stealth Model "Andromeda-alpha" Launched**: A new smaller reasoning model, **Andromeda-alpha**, specialized in **image and visual understanding** has been launched for community feedback via [OpenRouter.ai](https://openrouter.ai/openrouter/andromeda-alpha).
   - All prompts and outputs will be logged to improve the model, and is intended for trial use only, without uploading any personal, confidential or sensitive information and not for production.
- **Introducing `:exacto`: tool-calling endpoints**: OpenRouter launched a new class of endpoints called `:exacto`, for **higher tool-calling accuracy** by routing requests to providers that demonstrate **measurably better structured-output performance** and are documented in a new [blog post](https://openrouter.ai/announcements/provider-variance-introducing-exacto).
   - Launch models include `moonshotai/kimi-k2-0905:exacto`, `deepseek/deepseek-v3.1-terminus:exacto`, `z-ai/glm-4.6:exacto`, `openai/gpt-oss-120b:exacto`, and `qwen/qwen3-coder:exacto`, with internal and external benchmarks showing **material lift in tool-call success**.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1430340923639988436)** (1 messages): 

> `AI Confidence Scores, Cost-Efficient AI, OpenRouter LLMs, AI Agents and Workflows` 


- **Objective-AI: AI Confidence Scores Explained**: Ronald, CEO of [Objective-AI](https://objective-ai.io/), introduced a **Confidence Score** for OpenAI-compliant completion choices, derived not from direct AI assessment, but from smarter mechanisms.
   - Objective-AI leverages the diversity of **AI models** to provide **transparent statistics**.
- **Cost-Effective AI with Objective-AI**: Objective-AI's Confidence Score system enhances **cost-efficiency**, allowing users to maximize the utility of smaller models when used effectively.
   - Ronald mentioned that you can get a lot of value out of those little tiny models if you use them right.
- **OpenRouter Integrated by Objective-AI**: Objective-AI employs **OpenRouter** to access a wide array of **Large Language Models (LLMs)**, enhancing the platform's versatility and capabilities.
   - Ronald stated that they use **OpenRouter** for the wide variety of LLMs.
- **Objective-AI Building Free AI Agents and Workflows**: Ronald announced the development of **reliable AI Agents, Workflows, and Automations** at no charge, excluding runtime costs, to provide practical demonstrations of its capabilities.
   - He stated that he's personally building reliable/robust AI Agents, Workflows, and Automations free of charge, sans runtime costs. He also mentioned *n8n* integration exists, with documentation and examples coming soon.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1430274752639140062)** (106 messages🔥🔥): 

> `Free image generation models, Stealth model, MCP setup, GPT-5 Codex quota, Model overcharging` 


- ****RooCode** Offers Free Models**: **RooCode** offers free models such as **Grok Code Fast 1**, **Grok 4 Fast**, **Supernova 1 million**, and **Deepseek Chat 3.1**, providing an alternative to token-peddling platforms.
   - One user stated they've *nuked billions of tokens on roo cloud in the past couple months and it always went brrrrrrrrrrrr an was never backed off or rate limited.*
- ****Chutesflation** Is Real**: Users are experiencing unexpected high costs with OpenRouter, with one user reporting **$5** being consumed in just **23 days** despite minimal usage, leading to concerns about *chutesflation*.
   - Another user stated, *You don’t have a choice, you’re most likely already using Chutes*, implying OpenRouter routes users to them by default.
- ****OpenRouter API** has regional restrictions**: Users are encountering issues due to regional restrictions, indicated by a **403 error** with the message *Country, region, or territory not supported* from the OpenAI provider.
   - It was explained that location data collected by Cloudflare is forwarded to providers, and [OpenRouter itself does not impose regional restrictions](https://support.cloudflare.com).
- ****Exacto** Endpoint uses most expensive providers**: The **exacto** endpoint for **Qwen3 coder** is enabled by default, but it consistently uses expensive providers like **Google Vertex** and **Alibaba Open Source**, causing expenses to double.
   - A community member suggested the increased expense isn't related to the launch, and they'd have to change the model to opt in to exacto.
- ****GPT-5** is a good mathematical model**: Users discussed the best mathematical model and it was stated **GPT-5** is a good model, with a link to [matharena.ai](https://matharena.ai/).
   - It was stated that **Grok 4** is also pretty decent, **DeepSeek** is meh, anything **Claude** is bad.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1430275337345826928)** (37 messages🔥): 

> `Qwen Model Sizes, Roleplay Slop Solutions, ExactoNow Endpoint, Nebius AI Studio` 


- ****Qwen's Quirky Quantization Quandaries!****: Members discussed the unusual **1.7B** size of the [Alibaba_Qwen model](https://x.com/Alibaba_Qwen/status/19806659326253838682) in relation to its vision encoder and the more standard **32B** version.
   - Enthusiasts expressed interest in acquiring these models, noting the decent performance of the **Qwen chat website** and the model's potential as a local vision model, while highlighting crazy scores for such a small model.
- ****Squelching Slop with Savvy Selection!****: The community explored solutions to the *severe slop problem* in **roleplays**, suggesting methods like cranking up the temperature for multiple generations and using a lower-temperature model as a judge to pick the best.
   - One user proposed a dual-pass pipeline to rewrite prose and avoid repetitive phrases, and mentioned previous work by Sam Peach in providing lists of phrases to ban LLMs from using, which are easily implemented on Kobold.ccp.
- ****ExactoNow Excites Endpoint Enthusiasts!****: The community reacted to the new **ExactoNow** endpoint, an initiative to curate providers and filter out worse-performing models.
   - Users suggested displaying stats for models on the non-Exacto page or adding a badge indicating Exacto *quality*.
- ****Nebius AI's Nimble Numbers with New Qwen!****: A member reported that the [OpenRouter.ai Qwen2.5-coder-7b-instruct](https://openrouter.ai/qwen/qwen2.5-coder-7b-instruct) model performs exceptionally well via **Nebius AI Studio (Fast)**.
   - They noted that it is currently the only provider for this model.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1430286254066503812)** (61 messages🔥🔥): 

> `Independent research without GPU, Publishing challenges, Function calling with small language models, Calibration of digital twin cities, Constrained sampling and logit biasing` 


- **Independent Research on a Shoestring**: A member is conducting independent research without a GPU, leading to **5-8 hour experiment runtimes** and disrupted sleep.
   - Another member recommended using [vast.ai](https://vast.ai) to rent a GPU (e.g., an **RTX 3090** for around **$0.15 per hour**), but the first member aims to keep experiments accessible via Google Colab.
- **The Uphill Battle of Publishing**: A member described publishing as an *uphill battle*, particularly without a GPU, limiting choices to math-heavy projects and writing papers to aid understanding of existing concepts, referencing [this paper](https://arxiv.org/abs/2208.11970).
   - They also noted the long preparation time for **NeurIPS** submissions, with deadlines potentially in March 2026, but suggested **ICML** as an alternative in January.
- **Navigating Function Calling Finetuning**: A member asked about the inner workings of finetuning a small language model for function calling using libraries like **litellm/openai**, specifically how **JSON schema** is handled under the hood.
   - Suggestions included using **lm-format-enforcer** or **Lark Grammar** to constrain output to follow a tool calling **JSON** format, as well as llguidance's syntax for structuring tool calls with minimal added latency and high accuracy, and structured output as `[Thinking] I will now call tool {tool_name}. [tool JSON]`.
- **Realtime Prompting Prowess**: Members discussed realtime capabilities from **OpenAI/Google**, highlighting video chat in the **Gemini** mobile app and **OpenAI's** realtime audio chat prompting guidance.
   - One member shared a link to [OpenAI's realtime prompting guide](https://cookbook.openai.com/examples/realtime_prompting_guide), expressing both excitement and caution about the technology's potential to anticipate user actions.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1430318141795336354)** (34 messages🔥): 

> `Transformer Circuits, VinePPO Experiments, Jsonnet Configuration` 


- **Transformer Circuits named 'Best Paper'**: A member shared a [link to transformer circuits post](https://transformer-circuits.pub/2025/linebreaks/index.html), calling it the *best paper to read* this week, with another concurring it has *big implications framework wise*.
   - They qualified his recommendation noting his rigorous analysis of *his own idiocy and ignorance*.
- **VinePPO Experiments Configuration Details Revealed**: A member shared the [configuration of the VinePPO experiments](https://github.com/McGill-NLP/VinePPO/tree/main/configs), noting its fairly complex setup using **jsonnet**.
   - If there's interest, he is open to presenting on that, although he admits it is *overengineered* and that he *would have done it differently*.
- **DeepSeek's Jsonnet Configuration Praised**: A member is pondering trying the **jsonnet scheme** for a project.
   - He justified his interest citing that *if they're using it in DeepSeek I should try it myself and see how good it is*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1430287286565863445)** (13 messages🔥): 

> `Amazon vibe code IDE, Kiro spec based IDE, LLM Brain Rot, Apple AI researchers` 


- **Amazon's Vibe Code IDE exits Beta**: Amazon's **Vibe Code IDE** is out of invite-only beta, giving users **500 credits** to start, and it's [designed to be "spec based"](https://kiro.dev/blog/waitlist-is-over/), working around specifications for features and implementations, rather than solely prompts.
- **Kiro, Spec IDE, Based**: A member notes that **Kiro**, like many **AI IDEs**, is also a **VScode fork** and designed to be *"spec based"* as in it works around specifications for features and implementations, rather than solely prompts.
   - Another member suggests that converting spec text to code is a pattern that any platform could pick up and any special vscode UI they make can easily be cloned.
- **"LLM Brain Rot" Paper Dismissed!**: Members shared a link to a paper titled [LLM Brain Rot](https://llm-brain-rot.github.io/), but found it to be of low quality.
   - One member remarked *"I thought oh this will be a fun and easy paper and was surprised at how poor it was"* while another agreed that *"[t]here is a good paper to be written with that title. This is just not the one."*
- **Apple Seeks AI Researchers for Reasoning**: Members linked to an article about [Apple seeking AI researchers for reasoning](https://the-decoder.com/apple-seeks-ai-researchers-for-reasoning-even-as-its-own-study-questions-current-models/).


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1430369914685755564)** (15 messages🔥): 

> `NCU wrapper scripts, Custom NCU metrics, PyTorch Conference AI Infra panel discussion, L2 cache hit rates, LLM Systems` 


- **NCU Wrapper Scripts Seekers Customize Metrics**: A member was seeking [GitHub repos for NCU wrapper scripts](https://github.com/NVIDIA/nsight-compute) to pass in a list of metrics to be profiled using `--metrics`.
   - Another member suggested making custom sections/sets with their own metrics using the [NVIDIA Nsight Compute Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#section-files).
- **GPU Kernel Gurus Gather 'Round AI Infra**: The [PyTorch Conference AI Infra panel discussion on GPU kernels](https://aiinfrasummit2025.sched.com/event/28FoW/panel-discussion-the-ai-kernel-revolution-rethinking-execution-from-the-ground-up-robert-lange-sakanaai-simran-arora-stanford-university-nathan-lambert-allen-institute-moderated-by-mark-saroufim-gpu-mode) consensus is that peak performance for a kernel can be achieved using **PTX/assembly**.
- **L2 Cache Conundrums: Hit Rates Exceed 100%**: A member asked if anyone has seen **L2 cache hit rates of >100% on NCU** and how this can be explained.
   - Another member shared a thread that might be useful: [NVIDIA Forum](https://forums.developer.nvidia.com/t/weird-number-for-l2-cache-hitrate/120341/2)
- **Awesome LLM Systems Launches into Orbit**: A member published a new repo, [Awesome LLM Systems](https://github.com/romitjain/awesome-llm-systems), a curated list of key papers, blogs, and resources on the systems side of large language models.
- **Numerical Stability Nuggets Now Navigable**: A member sought resources to learn more about **numerical stability** after reading about **FA4's correctness optimization** and the deterministic inference blogs, pointing to [stochastic rounding](https://arxiv.org/pdf/2207.10321).
   - Another member shared a good tldr, as per prof david bindel: [epubs.siam.org](https://epubs.siam.org/doi/book/10.1137/1.9781611971491)


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1430323860888223825)** (1 messages): 

> `CuPy GPU Pointer vs PyTorch GPU Pointer, DLPack Conversion Performance, MatMul Kernel Performance` 


- **CuPy GPU Pointer Performance Beats PyTorch GPU Pointer**: A member inquired about the performance differences between **CuPy GPU pointers** and **PyTorch GPU pointers** when used in a custom **MatMul kernel**, noting a significant performance delta.
- **DLPack Conversion Bottleneck**: The member observed that converting a **CuPy array** to a **PyTorch tensor** using **DLPack** and then back to CuPy results in a performance hit despite yielding the same numerical results.
   - They questioned whether there is an inherent reason for this performance difference, showing a [screenshot](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68faade6&is=68f95c66&hm=d84234753a2510107fb4d7ecd73bbf01b7e07a92a430333bafa04d79be3e8bd3) of the performance comparison.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430516998030364702)** (9 messages🔥): 

> `warpGroup.arrive injection, Nvidia Stream Benchmark, CUDA video upscale project` 


- **`warpGroup.arrive` injection causes concern**: Verbose compilation gives a message saying *`warpgroup.arrive` is injected in around line 2580 by compiler to allow use of registers in GMMA in function*, suggesting that **all wgmma use the same registers**.
   - It was further clarified that this occurs to *allow the use of registers in GMMA*.
- **Navigating Nvidia's Stream Benchmark Options**: A member inquired about **Nvidia's Stream benchmark** to measure achievable bandwidth, referencing a Decoupled Look-back lecture.
   - Another member shared several relevant links, including [cpplinks](https://github.com/MattPD/cpplinks/blob/master/performance.tools.md#memory-benchmarking), [jeffhammond/STREAM](https://github.com/jeffhammond/STREAM) and [UoB-HPC/BabelStream](https://github.com/UoB-HPC/BabelStream) which supports GPUs.
- **CUDA project struggles with video upscaling speed**: A member asked for help on a CUDA project which upscales video in real time that achieves only **0.5 FPS** despite GPU acceleration.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1430412919752884235)** (2 messages): 

> `CPU affinity, taskset command, nice command` 


- **Optimize CPU Usage with Taskset and Nice**: To optimize CPU usage, especially when the node isn't running other tasks, pin your threads using the `taskset` command.
   - Further, use the `nice` command to ensure these threads remain prioritized on the CPU, preventing them from being deprioritized.
- **Taskset Command Explained**: The `taskset` command allows you to bind a process or thread to a specific CPU core or set of cores.
   - This ensures that the process or thread runs only on the specified cores, potentially improving performance by reducing context switching and cache misses.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1430648973412139068)** (2 messages): 

> `Relevant Algorithms for Optimization, Real-time Video Upscaling Projects` 


- **Optimization-Hungry Algorithms Beckon**: A member inquired about current, relevant algorithms that could significantly benefit from optimizations.
   - They also asked about classic algorithms suitable for experimentation, aiming to stay updated with the latest standards, seeking project ideas, specifically for real-time video upscaling.
- **Real-time Upscaling Project Idea**: The member is looking for a project to develop real-time video upscaling.
   - They want to keep up with the newest standards and are looking for a good algorithm to experiment with.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430295615459299420)** (1 messages): 

> `SIG Hiring, PyTorch Conference` 


- ****SIG**nals Recruit Quant Talent!**: Jacob from **Susquehanna International Group (SIG)**, a quantitative trading firm, announced they are [hiring across many roles](https://sig.com/careers/quant/).
   - Interested candidates are encouraged to DM Jacob for further discussion or meet him in person at the [PyTorch conference](https://calendly.com/jacob-baumbach-sig/pytorch-2025).
- ****Quant** Opportunity Knocks**: **SIG** is actively seeking quantitative talent across various roles within the firm.
   - Interested individuals are encouraged to explore career opportunities and connect with Jacob for more details.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430646355851870279)** (1 messages): 

> `GPU Rental, Acer Nitro 5, New vs Used GPUs, Utilizing Newest GPU Features` 


- **Rent a GPU to Learn the Ropes?**: A member with limited experience wondered if **renting a cheap GPU** is a good way for beginners to get started and improve their skills.
   - They questioned whether it makes sense to buy the newest, most powerful GPUs without first learning to fully utilize their features; the member is currently using an **Acer Nitro 5 gaming laptop**.
- **New Features Unnecessary for Beginners?**: A participant expressed the opinion that there's little point in getting the newest GPU hardware if one cannot fully exploit the new features.
   - This query suggests a strategy of **learning the fundamentals on affordable hardware** before investing in high-end GPUs.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1430574442614099969)** (3 messages): 

> `C++ update, Python version` 


- **C++ Version Faces Release Delays**: The C++ version is under development and facing delays from publishers.
   - Due to publisher-related delays, the release is not expected by the end of the year.
- **Python Version Still Being Discussed**: A Python-friendly version is being discussed.
   - However, no concrete plans or timelines have been established.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1430346894260965582)** (3 messages): 

> `Hackathon Projects, Working Group Ideas` 


- **Working Group Ideas become Hackathon fodder**: A user inquired whether the working group ideas created by mobicham are for the hackathon.
   - Mobicham responded that those are just ideas, and teams can choose to work on other projects during the hackathon.
- **Alternative Hackathon Projects Encouraged**: Mobicham clarified that the provided topics are merely suggestions.
   - Teams have the flexibility to pursue other projects that align with their interests and skills during the hackathon.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

bobmarleybiceps: im in oc too
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/)** (1 messages): 

snektron: https://www.thingiverse.com/thing:7179241 i finally bothered to upload this
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1430341094968660099)** (3 messages): 

> `Profiling Logs, NCU Analysis` 


- **Actionable Insights from Profiling Logs**: A member suggested turning the overwhelming **profiler logs** and **metrics** into actionable insights.
   - Another member suggested that this should be a sub-component of a **kernel generating agent**.
- **NCU Unbottlenecks Profiling?**: A member inquired whether **NCU** and similar tools already pinpoint issues and **bottlenecks**.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1430561313859571742)** (3 messages): 

> `sort_v2 leaderboard, B200 performance, H100 performance, L4 performance, A100 performance` 


- **Sort_v2 Leaderboard Champion Declared**: A user achieved **first place** on the `sort_v2` leaderboard with submission ID `66238`, showcasing **8.68 ms** on B200 and **16.3 ms** on A100.
   - The submission was also successful on H100 at **6.60 ms** and L4 at **52.7 ms**.
- **Another sort_v2 entry lands**: A user submission with ID `66239` to the `sort_v2` leaderboard demonstrating **8.83 ms** on B200, **52.6 ms** on L4, **6.60 ms** on H100, and **16.5 ms** on A100.
   - This submission showcases consistent performance across different hardware configurations.
- **sort_v2 First Place Finisher Strikes Again!**: A user achieved **first place** on L4 at **52.6 ms** and A100 at **16.0 ms** with submission ID `66241` to the `sort_v2` leaderboard.
   - They also had successful runs on B200 at **8.69 ms** and H100 at **6.58 ms**.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1430427470817005701)** (2 messages): 

> `Jax translation, TPU, PyTorch, vLLM` 


- **PyTorch models get JAXed**: A new library allows running **PyTorch models** through a **Jax translation layer** via this [link](https://google.github.io/torchax/).
- **vLLM goes to TPU**: The new version of **vLLM** will apparently use **Jax translation** to run on the **TPU** according to [this blogpost](https://blog.vllm.ai/2025/10/16/vllm-tpu.html).


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1430294254919614544)** (3 messages): 

> `GymAgent, Reasoning Models, Action->Response agents, MCP integration` 


- **GymAgent Decoded**: The **GymAgent** is an **Action->Response agent** that observes the environment at every turn and can reason in natural language before writing code to run in the game.
   - Unlike pure reasoning models with **MCP integration** (like Claude Code), it cannot choose when to observe or otherwise interact with the state.
- **Action vs Reasoning**: **GymAgent** uses **Action->Response** which allows it to observe the environment every turn.
   - **Reasoning Models** with **MCP Integration** can choose when to observe or interact with the state.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1430312950366736544)** (8 messages🔥): 

> `CUTLASS Cmake, TiledCopy, Thread-Value layouts` 


- **CUTLASS gets CMake Example**: A member shared a [CUTLASS CMake example](https://github.com/leimao/CUTLASS-Examples) for building simple CUTLASS code.
   - This may be a *good starter example* on how to use CMake with CUTLASS.
- **TiledCopy Threads Copying Zero Value?**: A member asked if value **0** is copied by multiple threads in a `TiledCopy` example from the documentation.
   - The image depicts a code snippet using `make_tiled_copy` with `Copy_Atom`, `Layout<Shape<_32,_8>>`, and `Layout<Shape< _4,_1>>` for thread and value layouts, raising questions about data duplication.
- **Thread-Value Layouts Inverted**: A member noted that the image shows two *inverse* **Thread-Value layouts** which map data coordinates to **(Thread, Value)** coordinates.
   - They clarified that *T32V0 means the 0th data item from thread 32's POV*.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1430289283943501865)** (15 messages🔥): 

> `Mojo Language, Apple Silicon, GPU Algorithms, Metal Toolchain` 


- ****Modular's Mojo** Aiming High**: A member expressed that the goals of **Modular** and the **Mojo** language are ambitious and it would be exciting to see them succeed.
- **Mojo on **Apple Silicon**: A Mixed Bag**: One user reported completing the first 8 problems in **2-3 hours** on an **Apple Silicon** machine, noting they were simple but that some problems cannot be done on their computer.
   - Another user confirmed that **Mojo** does work on **Apple Silicon** in alpha, but requires installing the **Metal toolchain** and **Xcode**, which they found undesirable.
- ****GPU Algorithm** Problems in Focus**: A user expressed hope that upcoming problems on **GPU algorithms** would delve deeper into **GPU layouts, blocks, and threads**.
   - Another user suggested problems **25-34** would be cool but the first user cannot do them on their machine and joked about needing a **DGX**.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430363465423519775)** (5 messages): 

> `Porting Qwen 3 to Burn, Mega Kernel Compilation, Hackathon Team Formation, Hackathon Registration Status` 


- **Qwen 3 Sets Sights on Burn Port**: A member plans to port **Qwen 3** to [Burn](https://burn.dev/) during the IRL hackathon, aiming to compile the **0.6B variant** into a single [mega kernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17).
- **Mega Kernel Quest Kicks Off**: A member is exploring the viability of Burn for serious work and compiling **LLMs** into a **megakernel**, despite being new to GPU programming and Burn.
- **Hackathon Squad Seeks Kernel Cracker**: A member skilled in **Rust** is seeking a hackathon team, particularly someone adept at kernels, to collaborate on **I/O** or **communication-related projects** such as **KV/weight transfers** or **disk-based KV cache**.
- **Hackathon Hopefuls Await Approval**: Some members are still awaiting hackathon approval to finalize their travel plans.
   - An organizer responded that the hackathon is **sold out by approximately 6x**.


  

---


### **GPU MODE ▷ #[opencl-vulkan](https://discord.com/channels/1189498204333543425/1418990184367919267/1430575051484430358)** (1 messages): 

> `coopVecMatMulAddNV in GLSL, GLSL Cooperative Vector Extension, VkConvertCooperativeVectorMatrixInfoNV, Hand-prepacked buffers with coopVec, Row-major order in GLSL` 


- **Exploring GLSL's coopVec with Hand-Prepacked Buffers**: A member inquired about using `coopVecMatMulAddNV` in GLSL with hand-prepacked buffers, facing issues with Vulkan drivers not allowing layout changes via `VkConvertCooperativeVectorMatrixInfoNV`.
   - They have a simple MLP using `coopVecMatMulAddNV` but the output isn't equivalent to a non-coop-vec version, using float16_t arrays in RowMajor order.
- **GLSL Coop-Vec Output Discrepancies**: A member implemented an MLP using `coopVecMatMulAddNV` in GLSL but observed that the output differs from a non-coop-vec version.
   - They are using float16_t arrays in RowMajor order for weights and biases, asking if it's possible to manually pack data for `gl_CooperativeVectorMatrixLayoutRowMajorNV` to function correctly.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1430281207177936916)** (17 messages🔥): 

> `Helion blog post, int4_gemm reference, Helion vs Triton performance` 


- **Helion Blog Post Goes Live**: The **Helion blog post** is now live at [pytorch.org](https://pytorch.org/blog/helion/), featuring **performance numbers** and **comparisons** to **Gluon**, **CUTEst**, and **Cutlass**.
   - The team invites everyone to a "meet the **Helion devs**" event.
- **int4_gemm Reference Request**: A member inquired about the reference for **int4_gemm**.
   - Another member pointed to the [examples folder in the repo](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py), which uses **Liger kernels** and **torch.compile kernels** as a reference.
- **Helion's Speedup Claims Under Scrutiny**: A member questioned the reported **14x speed-up**, suggesting it's unrealistic even for memory-bound operations compared to **fp16**, and pointed out that the [int4_gemm implementation](https://github.com/pytorch/helion/blob/main/examples/int4_gemm.py#L166-L178) is not a fused kernel.
   - It was suggested that comparison should be made with fused kernels like **Marlin** or **Machete**, or at least a simple **Triton gemm kernel** with `tl.dot_scaled`.
- **Desire for Superior Benchmarking Baselines**: A member expressed that it's not really a bug, but rather a mismatch of expectations: The reader would expect to see **speed-up comparison** with respect to specialized **cutlass/cute kernels**.
   - A member agreed to update the reference implementation and encouraged creating an issue on GitHub.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1430298738399903755)** (51 messages🔥): 

> `AI 'devil machine', Critical thinking skills, Qwen3 embedding, LM Studio API key, Local AI TTS` 


- **AI is a 'Devil Machine'?**: Members discussed the sentiment that generative AI is a 'devil machine', noting understandable animosity from the artist community due to image generation and a concern that **people are losing critical thinking skills**.
   - One member countered that they are actually *thinking critically a lot more* due to AI's tendency to hallucinate.
- **Qwen3 Embedding Improved**: One member stated that *the newer quants (the fixed ones) of **Qwen3 embedding 8b*** are working with roocode code indexing and are *a lot more accurate* than what they used before.
   - They clarified that the confidence score is a lot higher for relevant queries, and a lot less for irrelevant ones than the **mxbai-embed-large:latest**.
- **LM Studio supports third-party LLMs**: A member asked whether they can use **LM Studio** to communicate with a third-party LLM if they have an API key for it.
   - Another member responded that *with plugins (which are closed beta right now), you will be able to* and shared a [link to an OpenAI-compatible endpoint](https://lmstudio.ai/fuutott/openai-compat-endpoint-v2).
- **Local AI TTS Explored**: Following a member looking for **AI voice** solutions, another member recommended *chatterbox* as a *really good* **local AI TTS** option.
   - It was confirmed that **Chatterbox** supports multiple languages.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1430280534151663716)** (34 messages🔥): 

> `ATX power supplies, Ground loop issues, GPU powering issues, Vietnam cheap 3090/4090, Mi50 blower egpu` 


- **ATX Power Supply Green Wire Gets Grounded**: Members discussed how sharing the **green wire** on most **ATX** power supplies allows it to turn on when grounded, but can cause a *ground loop issue*.
- **Multiple PSUs can Backflow Into Each Other**: It was mentioned that powering a **GPU** from a separate **PSU** than the motherboard's PCIE power can cause PSUs to backflow into each other, phantom motherboard powering, and ground loop issues, but that you can avoid issues using sync cables.
   - They advised against connecting **12V rails** in parallel, as this might cause issues when splitting a **GPU** over multiple **PSUs**.
- **Vietnam 4090s Bargains are Unlikely**: One member asked where to buy a **cheap 3090/4090** in Vietnam, but another member said that Vietnam is one of the regions supplying **4090s** to China, so they wouldn't expect any huge bargains.
- **Mi50 blower eGPU repasted and flashed**: A member repasted their **Mi50 blower eGPU** with Arctic paste, reporting **50°C** during inference at 3/4 fan speed, and planned to flash it with v420.rom to get full **32GB** on Vulkan and look into full temp readout.
   - Another member said to use the [274474 rom](https://gist.github.com/evilJazz/14a4c82a67f2c52a6bb5f9cea02f5e13) as the V240 has a **178W** power cap.
- **AMD Instinct MI50 Shroud printable**: One member mentioned that with fans at 100% their card was never above **90°C junction**, while another linked to a [Printables.com model](https://www.printables.com/model/1421067-amd-instinct-mi50-shroud) for an **AMD Instinct MI50** shroud.
   - They also considered repasting again with **PTM7950**.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1430272074861248592)** (53 messages🔥): 

> `RAG implementation, AI Fact Checking, HF download with Xet, AI Agent Course` 


- **RAG not so difficult?**: Members discussed how to build a RAG system: *Request -> RAG Service -> AI Service -> RAG Service -> Response*, suggesting the use of a **PostgreSQL** database with a vector plugin.
   - One member noted that while a basic RAG setup is straightforward, tuning it for relevant results across a wide range of topics is tricky, and someone else added that having a small model before and after every step as a **sanity check** is a helpful measure.
- **AI Agents for reliable Fact Checking?**: Members discussed implementing an **AI agent** for reliably fact-checking news, suggesting the agent needs web search capabilities.
   - One member noted that it could probably give a probable likelihood of fake vs real but not *100% certainty*.
- **`hf download` saves to `hub`?**: A member asked why `hf download repo/id` saves files to `hub` instead of `xet` and how to ensure the CLI downloads with `xet`.
   - Another member explained that **HF** has its own setup saving to the hub, and `xet` is an entirely different setup.
- **AI Agent Course Outcomes and Benefits**: A member inquired about the outcomes and benefits of taking an **AI Agent Course**, since they are a beginner and curious.
   - One member responded that the outcome and benefit are *learning, satisfying your curiosity in technology and learning where this knowledge can take you* and that it can allow one to make AI agents for themselves, or to offer the service to other people/companies.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1430284183174905896)** (2 messages): 

> `AI Refactoring, vibecoder.buzz` 


- **AI Pauses to Refactor Sane Code**: A member learned to pause and prompt the AI to refactor code *like a sane person* with the prompt: `as a senior architect focused on modular code and maintainability`.
   - For changes, they prompt: `do not make changes or write code, answer question: do you have enough info to make these updates?` and `please create the minimal changeset (no tests)`.
- **vibecoder.buzz goes live!**: A member reported that their project is live at [vibecoder.buzz](https://vibecoder.buzz).
   - The project cost them **$2** for a domain to enable email verification, deviating from their initial goal of spending **$0**.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1430286618723352666)** (2 messages): 

> `Databomz, Chrome extension for prompts, Prompt organization` 


- **Databomz: Prompt Workspace Blasts Off**: A member introduced **Databomz**, a workspace and Chrome extension for saving, organizing, and sharing prompts with features like tags, versions, and folders, with more information available at [www.databomz.com](http://www.databomz.com/).
- **Forever Free Tier Tempts Prompt Power-Users**: The **Forever Free plan** includes most of the core features, and the creator is seeking feedback from active prompt users.
   - The project's GitHub repository can be found at [github.com/Lnrchaos/NeSy-CML](https://github.com/Lnrchaos/NeSy-CML).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430318686987751496)** (4 messages): 

> `Fenic integrates with Hugging Face Datasets, Multi-model collaboration, Prompt optimization tool` 


- **Fenic Plugs into 🤗 Datasets**: The open source project **Fenic** now directly integrates with **Hugging Face Datasets**, allowing users to hydrate version context straight from the Hub and safely tool it for agents, as detailed in the [documentation](https://huggingface.co/docs/hub/datasets-fenic).
   - Fenic, akin to e2b for compute, enables data snapshotting, agent context creation, and exposure of MCP tools via a dataframe API similar to pandas, with the [Fenic repo available on GitHub](https://github.com/typedef-ai/fenic).
- **Multi-Model Collaboration Evaluated**: Evaluations on multi-model collaboration are underway, focusing on reducing the single-user rate of hallucinations per request and enhancing overall request quality; more details are available on [this blog](https://facilitair.ai).
   - Good results have been achieved with sequential collaboration, and two OS repos using collaboration are currently at least at v1.
- **Datafrosch Newsletter Experimentation**: There is ongoing experimentation with the [Datafrosch newsletter](https://datafrosch.fun/blog/rss-newsletter.html) and the community is encouraged to share their insights.
   - The community is encouraged to discuss what's working and what's not.
- **Genie Prompt Optimizer Chrome Extension**: A tool has been created as a Chrome browser extension, designed to improve prompts, available for trial at the [Chrome Web Store](https://chromewebstore.google.com/detail/genie-prompt-optimizer/eejkodpbdljgoiidoekiiogpehoghnip).
   - Feedback on the tool is highly encouraged.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1430411424315736145)** (1 messages): 

> `Diffusion Models, DDPM` 


- **Math Behind Diffusion Models Simplified**: A member shared their article which aims to simplify the math behind **Diffusion Models (DDPM)** for beginners: [The Math Behind Diffusion Models (DDPM)](https://joydeep31415.medium.com/the-math-behind-diffusion-models-ddpm-9fabe9c9f1d9).
- **Beginner Resources for Diffusion Models**: Another resource that was mentioned was a beginner friendly article on understanding the math behind **Denoising Diffusion Probabilistic Models (DDPM)**.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1430295971765424210)** (2 messages): 

> `MBZUAI K2Think, OpenAI text-embedding-3-large dataset` 


- **MBZUAI K2Think competition opens**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/mbzuai_mbzuai-mbzuai-k2think-activity-7383761114959876097-0R7f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD53GRUB60-DZ9YvQ9NaG-LySvMdcC2QJzI) for the **MBZUAI K2Think** competition and invited others to team up.
- **Seeking OpenAI's text-embedding-3-large training dataset**: A member inquired whether the dataset used in training of OpenAI's `text-embedding-3-large` embedding model is public.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1430467324099694635)** (2 messages): 

> `Cogito 14b, AI Agent Course, Benefits of the AI Agent Course` 


- **Cogito:14b Triumphs in Agent Course**: A member completed the agent-course using the **Cogito:14b** ollama model and shared [personal insights on LinkedIn](https://www.linkedin.com/posts/duhyeon-kim-6623082b1_aiagents-huggingface-cogito-activity-7386672913896067072-YDbx?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEr5baoBwnyzQRLN-2xbgBSLqVXBm-f_i_QHi).
   - They invited others to react if interested in their reflections on the **AI agent** and the course.
- **Beginner Seeks Course Outcome**: A new member expressed curiosity about the motivations and advantages of taking the **AI Agent Course**.
   - They asked for insights into the expected benefits and outcomes for participants.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1430295667871449169)** (50 messages🔥): 

> `AI Studio Upgraded, Lovable Shopify Integration, OpenAI Project Mercury, Langchain Community, GoogleQuantumAI speed-up` 


- **AI Studio Gets Firebase Full-Stack Upgrade**: Logan Kilpatrick confirms [Firebase Studio integration](https://ai.studio/build) is coming to the new **AI Studio**, which is praised for vibe-coding apps and its speech-to-text feature.
   - Users request easier integration of databases, auth, and storage, and Logan invites feedback on unsupported use-cases like Chrome extensions and SolidJS, clarifying that **Gemini 2.5 Pro** is the current model.
- **Lovable Launches AI Shopify Full-Store Integration**: Lovable announced a **Shopify integration** that lets users spin up complete online stores via simple prompts, demoing the flow by launching its own merch store ([lovable.dev/merch](https://lovable.dev/merch)).
   - The feature is available to everyone with a bundled **30-day Shopify trial**, though existing Shopify stores can’t yet be imported or modified.
- **Project Mercury Pays Investment Bankers to Train OpenAI**: [Project Mercury](https://www.entrepreneur.com/business-news/openai-is-paying-ex-investment-bankers-to-train-its-ai/498585) pays contractors **$150 per hour** to feed financial models into AI, expanding the real-world, practical use of AI across business sectors like finance and technology.
- **GoogleQuantumAI Echoes a Willow Speed-Up**: **GoogleAI** announced a new milestone, using the **65-qubit Willow chip** and the **Quantum Echoes (OTOC) algorithm** to run a verifiable task **13,000x faster** than top supercomputers ([post on X](https://x.com/googleai/status/1981022228801307035)).
   - The team discussed implications for cryptography (SHA-256 safety), verifiability of results, real-world timelines for drug discovery and climate modeling, and running Crysis/Doom.
- **Next.js Evals Examines Framework AI-llity**: Guillermo Rauch announced [Next.js Evals](https://xcancel.com/rauchg/status/1981037270624076092), open-source ‘exams’ that let any LLM/agent prove it can correctly build with **Next.js** and other supported frameworks.
   - Models like **GPT-5-codex** and **Claude Sonnet 4.5** are currently scoring in the mid-40% range; community asks for real-world tasks, public traces, and cost columns.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430331836764131479)** (8 messages🔥): 

> `Sesame AI Funding, Sora Roadmap` 


- **Sesame Secures $250M in Series B Funding**: [Sesame](https://x.com/AnjneyMidha/status/1980705692253331624) launches its beta with a **$250M Series B** funding round led by **Sequoia & Spark**.
- **Sora's Socials, Android Rollout, and Cameos Coming Soon**: **OpenAI's Bill Peebles** revealed [Sora's roadmap updates](https://x.com/billpeeb/status/1981118483607032050) including **character cameos** launching within days, clip-stitching editor, upcoming **social channels** for groups, feed improvements, reduced over-moderation, performance boosts, and an **Android release**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1430272461353521272)** (38 messages🔥): 

> `Scammer Alert, Manus Credit System Stress, Account Suspensions, Local Compute Utilization, Pro Plan Credit Limits` 


- **Accusations Fly: User Labeled as 'Fraudster'!**: A user was publicly accused of being a *fraudster scammer* for allegedly soliciting login access to a paid account for law school exam research, sparking a heated discussion.
   - The accusing member cautioned against sharing account credentials, citing potential risks like personal and bank info theft, but the accused party responded by claiming to have found *another way* to resolve the issue.
- **Users Share Website Building Experiences**: A user asked the community for examples of websites built using Manus and to share their experiences with the platform.
   - Another member responded to this question by DMing.
- **Credit Confusion: Pro Plan Promises Unlimited?**: Users expressed confusion and frustration regarding the credit system, particularly concerning the **Pro Plan**, with one user feeling *bait and switched* after purchasing the plan based on a now-missing help page that promised unlimited credits.
   - Some users are willing to pay $200+/month for an unlimited plan and some also pointed out that despite adjustments, the credit system requires constant management and participation in improvement initiatives for free credits.
- **Account Suspensions Plague New Users**: One user reported that their and their girlfriend's account was suspended shortly after inputting card details, and they suspected inviting too many employees may have triggered the suspension.
   - Despite this, they had projects already made on the account, and were looking for support on how to retain the good projects.
- **Brainstorming Local Compute**: A user suggested utilizing local compute resources to augment Manus' capabilities.
   - The user suggested to enable building large native apps, processing huge datasets locally, running resource-intensive AI models on local hardware, and faster builds using your machine's full power.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1430283392334958633)** (23 messages🔥): 

> `Perplexity Comet, Pacific-Prime model, Regular Role, Chat Datasets with System Messages, Pretraining Interview` 


- **Perplexity Comet Invitation Piques Interest**: A member asked if a free **1-year pro invitation** to [Perplexity Comet](https://www.perplexity.ai/) is worth it now.
   - The question sparked curiosity about the current value and features of Perplexity Comet's pro version.
- **Pacific-Prime Claims Eco-Friendliness**: Boris from France introduced the [Pacific-Prime architecture](https://huggingface.co/Pacific-Prime/pacific-prime), highlighting that each of its **25 layers** performs **5 iterations** and learns to converge without error.
   - The model can run on **CUDA** supporting **1B** parameters with **6GB VRAM**, and the author claims it's *two times more eco than llamacasual architecture*.
- **"Regular" Role Still Exists, Awarded Per-Case**: A member inquired whether the "Regular" role is still being given out, suspecting it was an old practice.
   - Another member confirmed they received it recently and a third member confirmed it is given on a per-case basis to those who *have been contributing to the community*.
- **Pretraining Interview Recommended**: A member shared and recommended [an interview about pretraining](https://spotify.link/O1llYVO8FXb).
   - No further details about the interview were given.
- **System Message Chat Datasets Sought**: A member asked for recommendations for **chat datasets WITH system messages**.
   - They stated they were having trouble finding anything recent.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1430554105486639224)** (2 messages): 

> `Claude shapeshifting, Geometric structures in AI tasks` 


- **Claude shapeshifts!**: A member shared a post on X about [Claude shapeshifting](https://fxtwitter.com/wesg52/status/1980680563582538099).
- **Geometric structures for AI tasks?**: A member is wondering in how many more tasks we can find a geometric structure of similar kind to the one presented in [Transformer Circuits Thread on Linebreaks](https://transformer-circuits.pub/2025/linebreaks/index.html).


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1430574403174793398)** (1 messages): 

> `Tool Call Property Encoding, JSONSchema Subsets, LLM Reasoning with Schemas` 


- **Encoding Array/Object Properties in Tool Calls**: A member inquired about how to encode a property in `inputSchema` that can be either an **array** or an **object** in tool calls, given that `oneOf` is not accepted in the subset of JSONSchema used.
   - The discussion revolves around finding a way to define a flexible schema that LLMs can reason about and clients can accept when the property can have multiple data types.
- **JSONSchema Subset Limitations**: The user is working within the constraints of a limited subset of **JSONSchema** when defining the `inputSchema` for tool calls.
   - Features like `oneOf`, which would allow specifying multiple valid schema types for a property, are not supported in this environment, posing a challenge for properties that can be either arrays or objects.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1430596327594786988)** (23 messages🔥): 

> `MCP Server Context Limits, Optimizing MCP Costs, Tool Call Context Management, Workflow-Based Tool Execution, Subagents for Task Isolation` 


- **MCP Server Reaches Context Limit Fast**: A user is facing context limit issues with their MCP server containing **60 tools**, even with a max plan, and suspects the tool descriptions bloat the context.
   - Dividing tools into multiple servers wasn't well-received by customers, which has led the user to explore more efficient solutions to managing context and costs.
- **Engineers build custom chats to tackle cost**: One engineer built a custom chat with direct API connections and optimized model selection to manage costs, noting that costs can quickly accumulate when running in Claude desktop or ChatGPT due to plan limitations.
   - They found that a multi-agent MCP client with model switching resulted in costs of **$7-8 per action**, leading them to abandon that approach.
- **Streamlining Tool Workflows for Accuracy and Context**: An engineer implemented a workflow of **3 tools** to manage **50+ CLI actions**, which includes: listing actions, describing actions, and invoking actions, using server instructions to guide the LLM.
   - They emphasized that using descriptions and input/output schemas for a large number of tools can overwhelm models and exceed context limits, thus requiring a streamlined workflow.
- **Tool Call Context Should Expire for Efficiency**: One engineer proposed a way to signal to the agent that a tool call is no longer needed in the context to prevent bloating, suggesting a `hint` or field in the `tools/list` call.
   - The idea is that once the LLM has used the information from a tool call (e.g., listing documentation sections), the call could be removed from the context, improving efficiency.
- **Subagents Isolate Tasks in MCP**: Discussion included using subagents to isolate tasks, though MCP doesn't currently have a formal notion of subagents or agent-to-agent communication.
   - It was suggested that server `instructions` could be used to guide clients to prefer using subagents for certain tasks when possible, though client conformance would still be an issue.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1430385505270497361)** (1 messages): 

> `Voice Generation, Automated Judging Loop, ElevenLabs Integration, DSPy Optimization` 


- **NPC Voices Emerge from DSPy System**: A member built a voice generation system for game **NPCs** using **DSPy** to parse wiki content and generate character voice prompts, integrating with **ElevenLabs** to produce three voice candidates per character, as described in the project's [GitHub repository](https://github.com/Gielinor-Speaks/voiceover-mage).
   - The member is manually curating the voices, but aims to automate this via an automated judging loop, and is seeking advice on leveraging DSPy's optimization and compilation features, showcased in [this devlog video](https://youtu.be/z3DQm2PiKpo).
- **Automated Voice Judging Loop**: The voice-generation system's developer plans to add an **automated judging loop** to reduce manual curation, collecting manual selections as training signals to create examples.
   - The goal is to have the system learn what makes a *"good"* voice match for different character archetypes without manual judgment on every candidate.
- **ElevenLabs Synthesis with Emotion Mapping**: The system leverages **ElevenLabs** for voice synthesis and includes an **emotion mapping layer**, tying in-game animation IDs to synthesis parameters.
   - This allows the same character voice to sound angry, happy, or scared based on in-game context.
- **DSPy Optimization Tips Requested**: The member is seeking advice on structuring the character analysis pipeline and leveraging **DSPy's optimization and compilation features** more effectively, hoping to improve the subjective quality judgments of generated voices.
   - The member seeks advice due to lack of community support from the game's subreddit, which is *anti-AI*, and instead views this as a learning opportunity to leverage DSPy.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1430274415668494556)** (2 messages): 

> `DSPy, ArXiv Papers` 


- **DSPy Featured in New ArXiv Paper**: A member shared an [ArXiv link](https://arxiv.org/abs/2510.13907v1) to a new paper that utilizes **DSPy** in its repository.
   - However, the actual code for the paper has not been published yet.
- **Code Not Yet Published**: The ArXiv paper mentioned utilizes **DSPy**, however the actual code for the paper has not been published yet.
   - Members are eagerly awaiting the code release to examine the implementation details.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1430278974042542276)** (19 messages🔥): 

> `Microsoft Trace vs DSPy, Async DSPy Modules for OCR, Enforcing Providers with OpenRouter, AI & Blockchain Engineer Specialization` 


- **Microsoft's Trace Program Claims Accuracy Boost Over DSPy**: Microsoft's [Trace](https://microsoft.github.io/Trace/) program claims an **8% increase in accuracy** over an equivalent DSPy program, according to a screenshot shared by a member.
   - Another member expressed interest, planning to test it for a fair comparison but expects to maintain more granular control with DSPy.
- **Asynchronous DSPy Modules tackle OCR Challenges**: A member inquired about **async versions of DSPy modules** for parallel execution, particularly for OCR tasks.
   - Another member confirmed async capabilities, sharing their challenges with high throughput involving **Google Cloud Vision**, **DSPy Attachments library**, and layout via bboxes, noting repetition loop bugs and exploring alternatives like **paddleocr-vl** or **nanonets**.
- **OpenRouter Provider Enforcement remains a mystery**: A member asked for help on **how to enforce a specific provider when using OpenRouter within DSPy**.
   - No solution was provided in the current conversation.
- **AI & Blockchain Engineer unveils Agent Architecture Prowess**: A member introduced themselves as a **Senior AI & Blockchain Engineer**, specializing in building intelligent, autonomous systems at the intersection of AI and blockchain.
   - Their expertise includes **on-chain AI agents**, **LLM pipelines with real-time data**, **AI agent orchestration with LangChain/AutoGen**, and experience across multiple blockchain platforms like **Base**, **Solana**, and **EVM chains**.
- **Trainable Decorator tickles Fancy**: One member voiced their appreciation for the trainable decorator.
   - It's unclear which trainable decorator, but they seemed very excited by it, simply saying *Oh, I like the trainable decorator! Very good idea*


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1430269558203224124)** (20 messages🔥): 

> `Sora video, Nous research help, GPT-3.5 Sonnet, GPT-5 personality change, GPT-4o voice mode` 


- ****Sora** Video Debuts!**: A member shared a video created with **Sora** showcasing its capabilities, [available here](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68fa4f13&is=68f8fd93&hm=ac5d3464bd020d568c770a6fa89ef8ccb2db40420fd45bee25dc3be64c60807f&).
- ****Nous** Research Assistance Sought!**: A member requested assistance from Nous Research for a research paper aimed at helping kids interested in AI.
- **Farewell, **3.5 Sonnet**!**: A member shared a link from fixvx.com, [available here](https://fixvx.com/dbreunig/status/1980733694634770710), expressing a sense of finality regarding the end of **GPT-3.5 Sonnet**.
- ****GPT-5**'s Cold Shoulder!**: A member shared benchmark results, noting that **GPT-5** has become less friendly, attributing it to the sycophancy debate.
- ****GPT-4o** still nails the Jamaican Accent!**: A member pointed out that **GPT-4o** still performs a Jamaican accent in voice mode, while **GPT-5** claims it will but fails to change the voice.
   - They said that this is *one of the few metrics* they care about.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1430338693037691073)** (2 messages): 

> `Microsoft Trace, Microsoft Winsock` 


- **Microsoft Trace Utility Spotted**: A member shared a link to [Microsoft Trace](https://microsoft.github.io/Trace/), a utility, noting *apparently it's not all that new*.
- **Winsock Kernel in Userspace?**: Another member shared a project about running [Winsock Kernel in Userspace](https://github.com/microsoft/Windows-driver-samples/tree/main/network/winsock/userspace), implemented by Microsoft.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430281050692648983)** (19 messages🔥): 

> `Kimi support, Moderato and Allegretto, K2 on together, partnership with Kimi` 


- **Kimi Gets Zero Support Allegations**: A member complained about receiving *zero* support for **Kimi** and getting no response from the support team.
   - Another member clarified that it *isn't* a support server and suggested DMing a specific user.
- **Moderato and Allegretto payment plan**: A member expressed interest in upgrading to **Moderato** or **Allegretto** but couldn't find information on how many times they could use **OK Computer** in the payment plan.
   - Another member shared that **Moderato** allows for **20** uses per month, with a link to a [relevant tweet](https://x.com/togethercompute/status/1980337943169651055).
- **K2 blazing fast on Together**: A member remarked on the speed of **K2** on **Together**.
   - No further information was given.
- **Partnership with Kimi discussion**: A member inquired about who to DM for a partnership with **Kimi**.
   - Another member suggested DMing a specific user.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1430377338834522247)** (2 messages): 

> `Mojo features, Mojo missing type system, Mojo standard library datatypes, Mojo async runtime, Mojo effect system` 


- **Mojo's Type System: Still in Development**: A member stated that the most important thing that **Mojo** is missing is a *finished type system*.
- **Mojo's wishlist items**: After the type system, a member mentioned these other features **Mojo** should have: rounding out the **standard library datatypes**, **proper IO**, a good **async runtime**, an **effect system**, **static reflection**, **compiler plugins**, the ability to deal with more **restrictive targets**, **cluster compute**, **device/cluster modeling**, and some clone of **Erlang's OTP**.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1430593668888264704)** (13 messages🔥): 

> `Segfault on H100 with Llama-3.1, GGUF vs bfloat16 models, tensor_internal package renamed to tensor` 


- ****Llama-3.1** Segfaults on **H100** with Nightly Build**: A member reported a segfault on the nightly build when loading `modularai/Llama-3.1-8B-Instruct-GGUF` on an **H100** GPU, narrowing it down to the `session.load` function in *llama3/model.py*.
   - The issue seems specific to the GPU, as the model works fine on CPU, particularly with the `modular/max-nvidia-full:nightly` base image, but segfaults when pushed to the GPU.
- ****GGUF** Models Spark Segfault Speculation**: A member inquired whether the segfault persists with **bfloat16** models, noting that dequantization kernels for **GGUF** (*q4_k*, etc.) weights have been built only for CPU execution so far.
   - The member confirmed that while the CPU runs the **q4_k** version, the GPU attempts to run the **bf16** version, leading to the segfault.
- ****tensor_internal** morphs into **tensor****: A member highlighted a critical update in the latest nightly for the **Mojo API**: the `tensor_internal` package has been renamed to `tensor` ([Discord link](https://discord.com/channels/1087530497313357884/1224434323193594059/1430580184305635339)).


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1430290731641340057)** (6 messages): 

> `Vulkan, sine function, renderer, transcendental functions` 


- **Vulkan's Inaccurate Sine Function Causes Errors**: The **Vulkan sine function's inaccuracy** causes errors, necessitating a custom sine function which could slow down the process.
   - A member reported *failing a couple of errors* due to this issue.
- **Tinygrad's Renderer Requirements**: **tinygrad/uop/decompositions.py** triggers automatically if your renderer doesn't implement **sin/log/exp**.
   - A member didn't realize this would happen and modified their renderer accordingly to ensure that all **transcendental functions** pass.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1430668420646572123)** (2 messages): 

> `BEAM, JITBEAM, jitted kernels` 


- **BEAM and JITBEAM Disambiguation**: A member inquired about the difference between **BEAM** and **JITBEAM**, questioning if both are functional.
   - Another member clarified that **JITBEAM** specifically refers to the **BEAM** of jitted kernels.
- **JITBEAM**: It is the BEAM of jitted kernels.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1430427902549295135)** (6 messages): 

> `Aider for benchmarking, Coding with Local models, Evaluation logs and LLM histories, Leaderboard ranking, Uniswap Portal Spam` 


- **Aider Abused for Benchmarking**: A member mentioned they have *never used Aider for it’s original purpose, only for benchmarking.*
   - They clarified they have only been *coding with Local models* such as **Cline** and **RooCode**.
- **LLM History Troubles**: One user was seeking *evaluation logs and LLM histories*, and clarified that they meant the *full chat*. 
   - Another member confirmed they *haven't had luck either lately* in retrieving those logs.
- **Uniswap Portal Spam**: A user shared a suspicious looking link ([uniswap-portal.web.app](https://uniswap-portal.web.app)).
   - An image attachment was flagged as **spam** by image analysis.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1430541841698127902)** (1 messages): 

> `Project maintenance, Version updates` 


- **Project Maintenance Status Queried**: A member inquired about the maintenance status of the project given the lack of version updates since **August 10th** and limited activity.
   - No answer to the status of the project was given.
- **No updates**: No updates were given after the question was asked.
   - -


  
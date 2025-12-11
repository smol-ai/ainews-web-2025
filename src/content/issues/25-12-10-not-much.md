---
id: MjAyNS0x
title: not much happened today
date: '2025-12-10T05:44:39.731046Z'
description: >-
  **NousResearch's Nomos 1** is a 30B open math model achieving a top Putnam
  score with only ~3B active parameters, enabling consumer Mac inference.
  **AxiomProver** also posts top Putnam results using ThinkyMachines' RL stack.
  **Mistral's Devstral 2 Small** outperforms DeepSeek v3.2 in 71% of preferences
  with better speed and cost. **Anthropic's Claude Code** introduces
  asynchronous agent execution. **Cursor 2.2** adds deep agent primitives like
  Debug and Plan Modes. **VS Code** launches unified agent chat sessions
  improving multi-agent workflows. **LangChain** releases "Polly" for agent
  observability. The **Stirrup** harness leads OpenAI GDPval benchmarks with
  Claude Opus 4.5, GPT-5, and Gemini 3 Pro following. Advances in quantization
  include **vLLM** integrating Intel's AutoRound PTQ for efficient serving.
  **Unsloth** achieves up to 3× training speedups with new kernels across Llama,
  Qwen, Mistral, and Gemma models. *"Compositional reasoning + specialized
  post-training under constrained active params can rival frontier closed models
  on formal math."*
companies:
  - nousresearch
  - thinkymachines
  - mistral-ai
  - deepseek
  - anthropic
  - cursor
  - microsoft
  - langchain-ai
  - openai
  - gemini
  - intel
  - vllm_project
  - danielhanchen
models:
  - nomos-1
  - axiomprover
  - devstral-2-small
  - deepseek-v3.2
  - claude-code
  - cursor-2.2
  - claude-opus-4.5
  - gpt-5
  - claude-sonnet-4.5
  - gemini-3-pro
  - llama
  - qwen
  - mistral
  - gemma
topics:
  - math
  - formal-reasoning
  - agentic-systems
  - asynchronous-execution
  - multi-agent-systems
  - observability
  - benchmarking
  - quantization
  - post-training-quantization
  - training-speedup
  - kernel-optimization
  - inference-efficiency
people: []
---


**a calm before the last batch of releases.**

> AI News for 12/9/2025-12/10/2025. We checked 12 subreddits, 544 Twitters and 24 Discords (205 channels, and 6101 messages) for you. Estimated reading time saved (at 200wpm): 529 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Check out [the RL talks from AIE Code](https://x.com/aiDotEngineer/status/1998785602989461531?s=20).

---

# AI Twitter Recap

**Open math and reasoning: small active params + agents hit top-tier performance**

- **NousResearch’s Putnam-class prover (open)**: Community reports suggest the new “Nomos 1” math system is a 30B open model that scored 87/120 on this year’s Putnam—an estimated #2/3988 result—achieved via a specialized post-training and agentic pipeline; importantly, only ~3B parameters are active at inference, making it run on consumer Macs ([1](https://twitter.com/kimmonismus/status/1998749650984255985), [2](https://twitter.com/EMostaque/status/1998686465279025190), [3](https://twitter.com/Dorialexander/status/1998657955718148268)). In parallel, **AxiomProver**—fine-tuned using ThinkyMachines’ Tinker RL stack—also posted top Putnam scores ([1](https://twitter.com/thinkymachines/status/1998903749000180183), [2](https://twitter.com/thinkymachines/status/1998925489084498094), [3](https://twitter.com/ariG23498/status/1998654584529797522)). Takeaway: compositional reasoning + specialized post-training under constrained active params can rival frontier closed models on formal math.

**Agentic coding systems, orchestration, and evals**

- **Mistral’s Devstral 2 momentum**: Practitioners report Devstral 2 Small “beats or ties” DeepSeek v3.2 in 71% of third‑party prefs while being smaller/faster/cheaper, with a polished Vibe CLI onboarding (needs images) ([1](https://twitter.com/swyx/status/1998600513538109476), [2](https://twitter.com/N8Programs/status/1998591943798882484)).
- **Claude Code goes async**: Anthropic shipped background subagents and asynchronous execution (v2.0.64), enabling concurrent exploration/tests that “wake up” the main agent upon completion ([1](https://twitter.com/omarsar0/status/1998774531188830304), [2](https://twitter.com/omarsar0/status/1998777320434290729), [3](https://twitter.com/omarsar0/status/1998789689587708246)).
- **Cursor 2.2 shipping deep agent primitives**: Debug Mode instruments your code, spins servers to capture logs, and streams runtime data into the agent; upgrades include Plan Mode diagrams and multi‑agent judging ([1](https://twitter.com/cursor_ai/status/1998821350333440133), [2](https://twitter.com/cursor_ai/status/1998821554000388096), [3](https://twitter.com/cursor_ai/status/1998821555250380986)).
- **VS Code “Agent sessions”**: Unified chat integrates local/background/cloud agents with worktree isolation and seamless agent handoff (“Continue in…”)—a notable UX step for real multi‑agent workflows ([1](https://twitter.com/code/status/1998827135855743148), [2](https://twitter.com/pierceboggan/status/1998829467649937690), [3](https://twitter.com/burkeholland/status/1998835297644425485)).
- **Observability for agents**: LangChain shipped “Polly” (an agent to debug agents) and a CLI to pull traces/threads—moving beyond simple LLM app debugging toward long‑running, complex agent systems ([1](https://twitter.com/LangChainAI/status/1998807193320305101), [2](https://twitter.com/hwchase17/status/1998809833693467100), [3](https://twitter.com/LangChainAI/status/1998814975033487822)).
- **Stirrup + GDPval-AA (Artificial Analysis)**: A lightweight, open agent harness and a new leaderboard for OpenAI’s GDPval tasks (real knowledge work across 9 industries). Results (Elo): Claude Opus 4.5 leads, followed by GPT‑5, Claude Sonnet 4.5, then a tie between DeepSeek V3.2 and Gemini 3 Pro. Notably, the Stirrup harness outperformed consumer chatbot UIs across models ([1](https://twitter.com/ArtificialAnlys/status/1998841566627246173), [2](https://twitter.com/ArtificialAnlys/status/1998843644628054506)).
- **MCP workflow composition**: “Remix servers” pattern lets you compose tool surfaces from multiple MCP servers into a virtual server with server‑side authored prompts/workflows (portable across clients) ([link](https://twitter.com/AAAzzam/status/1998773774699614537)).

**Systems, performance, and compute trends**

- **Quantization and PTQ**: vLLM integrated Intel’s AutoRound post‑training quantization in LLM Compressor, producing W4A16 checkpoints served directly with vLLM across Xeon, Gaudi, Arc GPUs, etc. ([link](https://twitter.com/vllm_project/status/1998710451312771532)).
- **Unsloth training speedups**: New fused varlen RoPE + int64 Triton kernels and padding‑free training deliver up to 3× training speed and ~50% less VRAM with identical loss/grad norms across Llama/Qwen/Mistral/Gemma families ([1](https://twitter.com/danielhanchen/status/1998770347081109864), [2](https://twitter.com/danielhanchen/status/1998770349975155060), [3](https://twitter.com/danielhanchen/status/1998770352646914146)).
- **Fabric, interconnect, and costs**: AWS B300 EFA v4 inter‑node hits 800 GB/s vs 900 GB/s NVLink‑5 intra‑node—interconnect catch‑up continues ([1](https://twitter.com/StasBekman/status/1998821183844938000), [2](https://twitter.com/wightmanr/status/1998915115744428369)). Epoch estimates B200 chip cost at ~$6.4k and ~80% chip‑level margins (logic die <15% of cost), while realized margins drop in bundled servers; NVIDIA overall ~73% margins recently ([1](https://twitter.com/EpochAIResearch/status/1998819237251657890), [2](https://twitter.com/EpochAIResearch/status/1998819296353595424)). SemiAnalysis details two TPUv8 paths: Broadcom‑delivered “Sunfish” (bundle) vs Google‑assembled “Zebrafish” (MediaTek support) ([link](https://twitter.com/SemiAnalysis_/status/1998830078629724596)).
- **Compute in space (and physics)**: Starcloud‑1’s H100 trained nanoGPT on orbit (Shakespeare) and ran Gemma inference—the first LLM training demo in space, per team ([1](https://twitter.com/AdiOltean/status/1998769997431058927), [2](https://twitter.com/karpathy/status/1998806260783919434)). Counter‑points flag severe thermal radiation constraints in vacuum and cheaper terrestrial generation (nuclear/solar+batteries) over “space datacenters” ([1](https://twitter.com/jenzhuscott/status/1998591718338486757), [2](https://twitter.com/clawrence/status/1998753444598010254), [3](https://twitter.com/YIMBYLAND/status/1998785782082056626)).

**Multimodal, vision/video, and factuality**

- **GLM‑4.6V (Zhipu)**: Early users say it “sounds like Sonnet,” performs close to Sonnet 4 on coding + visual understanding, and is the first OSS vision model they found useful for design critique; priced below Gemini‑2.5‑Flash. Some looping observed—post‑training may help ([1](https://twitter.com/hrishioa/status/1998636234806341873), [2](https://twitter.com/hrishioa/status/1998636284533944725)).
- **Qwen3‑Omni‑Flash (Dec. 2025 update)**: Big upgrade to realtime multi‑turn video/audio dialog, 119 text languages, 19 speech, system‑prompt persona controls, with realtime and offline APIs and demos ([link](https://twitter.com/Alibaba_Qwen/status/1998776328586477672)).
- **Perceptron Isaac‑0.2 (open VLMs)**: 1B/2B hybrid‑reasoning vision‑language models (SigLIP + Qwen) aimed at a robust perception backbone for robotics; code/weights open and API available; video‑native and control modalities on the roadmap ([1](https://twitter.com/perceptroninc/status/1998812935821697363), [2](https://twitter.com/AkshatS07/status/1998818590405935468)).
- **Video generation and vision research**: Meta’s OneStory (coherent multi‑shot video with adaptive memory) and Wan‑Move (motion‑controllable video via latent trajectory guidance) expand controllability; “Reflection Removal through Efficient Adaptation of Diffusion Transformers” shows efficient window‑reflection cleanup; dynamic scene reconstruction via D4RT continues pushing 4D perception ([1](https://twitter.com/_akhaliq/status/1998760879261888814), [2](https://twitter.com/_akhaliq/status/1998606187500097588), [3](https://twitter.com/_akhaliq/status/1998752500673888409), [4](https://twitter.com/_akhaliq/status/1998763356883452031)).
- **Factuality benchmarking**: DeepMind/Google Research release FACTS, a suite spanning internal knowledge, web search, grounding, and multimodal inputs; Gemini 3 Pro leads at 68.8%. Benchmarks are available on Kaggle to standardize reliability evals ([1](https://twitter.com/GoogleDeepMind/status/1998831084277313539), [2](https://twitter.com/GoogleDeepMind/status/1998831088324473025)).

**Autonomy, proactive agents, and AI-native product loops**

- **Wayve x Nissan**: Definitive agreements to deploy Wayve’s AI Driver into next‑gen ProPILOT—ADAS and point‑to‑point driving across Nissan’s global lineup ([link](https://twitter.com/alexgkendall/status/1998592238641656160)).
- **Proactive agents from wearables**: “ProAgent” continuously perceives through egocentric sensors (video/audio/motion/location) and proactively assists (weather, rides, price checks), with on‑device Jetson Orin latency ~4.5s; +33.4% proactive prediction accuracy and 1.79× lower memory vs baselines in user studies ([link](https://twitter.com/dair_ai/status/1998775732001190018)).
- **Shopify’s AI stack**:
    - SimGym simulates “digital customers” for task completion and zero‑traffic A/B testing.
    - Sidekick Pulse runs large HSTU + LLMs overnight to surface business improvements.
    - Product Network lets merchants sell each other’s products via LLM‑driven fit and on‑site checkout ([1](https://twitter.com/MParakhin/status/1998786503779234276), [2](https://twitter.com/MParakhin/status/1998788090324988244), [3](https://twitter.com/MParakhin/status/1998789844794012049)).
- **Tooling to operationalize**: GitHub Copilot auto model selection is GA in VS Code ([link](https://twitter.com/GHchangelog/status/1998847752050983279)). Pixel Watch 3+ uses on‑device Gemma for smart replies ([link](https://twitter.com/Google/status/1998849211941482513)). Google’s Jules adds Suggested/Scheduled Tasks and a Render integration for self‑healing deploys—pushing “continuous AI” into devops loops ([1](https://twitter.com/julesagent/status/1998829514634531252), [2](https://twitter.com/julesagent/status/1998848018817364175), [3](https://twitter.com/julesagent/status/1998875242413044130)).

**Top tweets (by engagement)**

- “What you create is an honest reflection of who you are.” — [@naval](https://twitter.com/naval/status/1998671506784547309)
- First H100-powered LLM training in space (nanoGPT on Shakespeare) — [@AdiOltean](https://twitter.com/AdiOltean/status/1998769997431058927)
- Auto-grading a decade of HN with GPT‑5.1 Thinking — [@karpathy](https://twitter.com/karpathy/status/1998803709468487877)
- Claude Code adds asynchronous subagents — [@omarsar0](https://twitter.com/omarsar0/status/1998774531188830304)
- Cursor 2.2 ships Debug Mode + agent upgrades — [@cursor_ai](https://twitter.com/cursor_ai/status/1998821350333440133)
- Qwen3‑Omni‑Flash (Dec update) — [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1998776328586477672)
- Waymo to London (2026) — [@demishassabis](https://twitter.com/demishassabis/status/1998825670869397802)

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Unsloth AI Training Optimization

- [**You can now train LLMs 3x faster with 30% less memory! (<3.9GB VRAM)**](https://www.reddit.com/r/LocalLLaMA/comments/1pj51tu/you_can_now_train_llms_3x_faster_with_30_less/) (Activity: 814): **The image illustrates the performance improvements achieved by the new Triton kernels and smart auto packing support from Unsloth, which enable training of large language models (LLMs) like Qwen3-4B up to 3x faster with 30-90% less VRAM, requiring less than 3.9GB of VRAM. The image includes bar graphs that compare the training throughput and speedup of the new Unsloth RoPE + MLP kernels against an optimized setup with FA3, highlighting significant efficiency gains without accuracy loss. Key technical advancements include a 2.3x faster QK Rotary Embedding fused Triton kernel, updated SwiGLU and GeGLU kernels with int64 indexing, and 2.5x to 5x faster uncontaminated packing with various backends. These optimizations are automatically enabled, providing improved SFT loss stability and predictable GPU utilization.** Commenters are impressed with the performance improvements, noting that the new method is significantly faster than previous versions. There is interest in how these advancements benefit users with lower VRAM capacities, such as those with 6GB, and questions about compatibility with multi-GPU setups.
    - A key technical insight is the claim of achieving training speeds that are '3x faster compared to Unsloth's old >2.5x faster' method, indicating a significant improvement over previous optimizations. This suggests a cumulative enhancement in training efficiency, potentially through algorithmic improvements or better resource management.
    - The discussion about compatibility with multiple GPUs, such as two 3090s, highlights a common concern in the community regarding scalability and cost-effectiveness. The ability to utilize multiple GPUs effectively can significantly reduce costs compared to investing in a single high-end GPU, which is often prohibitively expensive.
    - Questions about specific hardware compatibility, such as with the AMD Strix Halo Max+ 395, suggest a need for broader support across different architectures. This reflects the community's interest in ensuring that these optimizations are not limited to specific hardware, thereby increasing accessibility for a wider range of users.

### 2. Mistral AI Model Releases

- [**Mistral AI drops 3x as many LLMs in a single week as OpenAI did in 6 years**](https://www.reddit.com/r/LocalLLaMA/comments/1pj8kb6/mistral_ai_drops_3x_as_many_llms_in_a_single_week/) (Activity: 560): **Mistral AI has released a series of large language models (LLMs) in a single week, surpassing the number of models released by OpenAI over six years. The models include a range of parameter sizes, from** `3B` **to** `675B`**, and are available under the Apache 2.0 and modified MIT licenses. These models are designed for various applications, including coding, reasoning, and instruction, and are optimized for local use. The largest model, the** `675B` **parameter instruct model, represents Mistral's most advanced offering. All models are accessible via [Hugging Face](https://huggingface.co/bartowski).** Commenters noted that the `Devstral 2 123B` model shows significant improvement over previous models, though some attribute this to potential 'new model hype'. There is also a critical comparison of Mistral's ethical impact versus OpenAI's, highlighting a perceived lack of engagement strategies by Mistral.
    - The comment by 'DragonfruitIll660' highlights the release of Devstral 2 123B, noting it as a significant improvement over Mistral Large 2, particularly for basic chat functionalities. This suggests that Mistral AI's new models are making strides in performance, potentially due to open weight models which allow for more community-driven enhancements.
    - 'Long_comment_san' discusses the desire for Mistral AI to release a model in the 80-120 billion parameter range, particularly a mixture of experts (MOE) model. The commenter notes that the current Mistral large model's size exceeds 128GB, limiting accessibility for experimentation, and expresses interest in smaller, fine-tunable models that are becoming popular in the AI community.
    - The discussion touches on the trend towards smaller, fine-tunable models, as mentioned by 'Long_comment_san'. This reflects a broader industry shift where compact models are gaining traction for their efficiency and adaptability, contrasting with the traditional focus on large-scale models. The mention of Qwen and anticipation for 'Qwen Next' indicates ongoing interest in competitive advancements in model development.

### 3. Hardware and CLI Innovations

- [**new CLI experience has been merged into llama.cpp**](https://www.reddit.com/r/LocalLLaMA/comments/1pj4j87/new_cli_experience_has_been_merged_into_llamacpp/) (Activity: 514): **The image showcases a new command-line interface (CLI) experience for** `llama.cpp`**, a project under the ggml-org umbrella. This update, as detailed in [this pull request](https://github.com/ggml-org/llama.cpp/pull/17824), introduces a more user-friendly interface with commands like** `exit`**,** `regenerate`**,** `clear`**, and** `read`**. The CLI also provides performance metrics for prompt and generation speeds, enhancing usability for developers and users interacting with the virtual assistant capabilities of** `llama.cpp`**.** One commenter speculates that this update could challenge the relevance of **ollama**, while another suggests that the integration of WEB/CLI support in `llama.cpp` might influence the utility of projects like OpenWebUI/OpenCode.
    - The integration of a new CLI experience into `llama.cpp` is a significant enhancement, potentially impacting the utility of other interfaces like OpenWebUI/OpenCode. This update could streamline workflows by consolidating web and CLI functionalities, making it a more versatile tool for developers who previously relied on multiple platforms for different tasks.
    - The continuous improvements in `llama.cpp`, such as the recent CLI update, highlight its evolving capabilities and potential to replace other tools like Ollama. This could lead to a more unified development environment, reducing the need for multiple separate tools and simplifying the user experience.
    - The discussion around the new CLI experience in `llama.cpp` suggests a growing interest in expanding its functionalities, possibly towards developing a coding agent. This indicates a forward-looking approach in the project's roadmap, aiming to enhance its utility for developers by integrating more advanced features.
- [**I bought a Grace-Hopper server for €7.5k on Reddit and converted it into a desktop.**](https://www.reddit.com/r/LocalLLaMA/comments/1pjbhyz/i_bought_a_gracehopper_server_for_75k_on_reddit/) (Activity: 309): **A Reddit user purchased a Grace-Hopper server for** `€7.5k`**, originally listed at** `€10k`**, and converted it into a desktop capable of running** `235B parameter models`**. The server, designed for liquid cooling, was adapted to air cooling and back, overcoming challenges like GPUs reporting extreme temperatures. This project is part of the user's [GLaDOS Project](https://github.com/dnhkng/GlaDOS), showcasing the transformation of enterprise-grade AI hardware into a home-use system. The full story is detailed in a [blog post](https://dnhkng.github.io/posts/hopper/).** Commenters noted the purchase as a 'steal' and recommended using `vllm` with the hardware, highlighting the significant effort required to operationalize the system but acknowledging the value of the deal.
    - cantgetthistowork suggests using `vllm` with the Grace-Hopper server, implying that this software is well-suited for the hardware's capabilities. `vllm` is known for its efficient handling of large language models, which could leverage the server's high-performance components effectively.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI Strategic Shift and AGI Pause

- [**AGI pursuit paused for a major strategic course correction at OAI**](https://www.reddit.com/r/ChatGPT/comments/1piudnw/agi_pursuit_paused_for_a_major_strategic_course/) (Activity: 669): **The image and accompanying text highlight a strategic shift at OpenAI, as CEO Sam Altman calls for a "code red" to prioritize improving ChatGPT over other projects like the Sora video generator. This decision marks a significant course correction, emphasizing the need to enhance user engagement and satisfaction with ChatGPT, potentially at the expense of the company's broader goal of achieving artificial general intelligence (AGI). The move reflects internal debates at OpenAI about balancing consumer popularity with research ambitions, as well as Altman's leadership style, which has been criticized for lacking focus on practical limits.** A notable comment criticizes OpenAI's resource allocation, suggesting that the focus on expansion and hype over innovation has led to a decline in the quality of their language models. This reflects broader concerns about the company's strategic priorities and leadership.
    - A user criticizes OpenAI's strategic focus, suggesting that resources were misallocated towards expanding user base and hype rather than improving the quality of their language models. They imply that this focus on expansion over innovation has led to a decline in LLM quality over the year, questioning the company's commitment to achieving AGI.
    - Another comment contrasts OpenAI's approach with that of Anthropic, which is praised for focusing on a core goal and customer base. The commenter notes that while Google and the Gemini team have improved quality, they still lack a niche. They predict that Microsoft and Amazon will continue to focus on cloud and AI compute, while Meta struggles to create a successful product. The commenter highlights that smaller startups are effectively solving real use cases by building on hyperscaler models, suggesting a shift in innovation away from large companies like OpenAI.
    - A user questions the effectiveness of OpenAI's planned model updates, arguing that minor improvements in benchmarks and model personality do not address the underlying financial and capability issues. They express skepticism about the company's ability to resolve these issues in the near future, suggesting that OpenAI is in a 'permanent code red' situation for the next several years.
- [**Just Cancelled my ChatGPT Subscription**](https://www.reddit.com/r/ChatGPT/comments/1piuxir/just_cancelled_my_chatgpt_subscription/) (Activity: 2476): **The user has canceled their ChatGPT subscription, citing superior performance from Gemini and Claude models, particularly with the Antigravity IDE public preview featuring Gemini 3 and Claude Opus 4.5. They also mention enhancements in NotesbookLM and Nano Banana, alongside attractive offers like a** `6-12 month` **free Gemini Pro subscription for Pixel buyers and students. The user notes a significant shift in AI leadership towards Google, attributing it to their extensive resources.** Commenters echo the sentiment of switching from ChatGPT to alternatives like Gemini and Perplexity, citing issues with ChatGPT's speed, reliability, and memory. There is also criticism of ChatGPT's content moderation, with calls for an 'adult mode' to allow more open discussions on sensitive topics.
    - Porsche-Turbo highlights that GPT's performance has declined in terms of speed, reliability, and memory, especially when compared to newer models like Gemini. They also note that GPT's image processing capabilities are inferior to alternatives like Nano Banana/Pro, which may influence users to switch or use multiple AI tools for different tasks.
    - Minimum_Rice555 points out a significant issue with ChatGPT's current performance, stating that it often provides circular non-answers except in direct code generation scenarios. This suggests a decline in the model's ability to handle complex queries effectively, which could be a reason for users to seek alternatives.
    - JestonT discusses the challenge of transitioning from ChatGPT to other AI platforms, particularly in terms of data portability. They inquire about exporting data to Google or other AI systems and express interest in Claude Code, noting that Codex remains a reliable option due to its low bug rate, while also testing Google's Antigravity.

### 2. Claude Modular Rules Update

- [**Claude Rules (./claude/rules/) are here**](https://www.reddit.com/r/ClaudeAI/comments/1piuih6/claude_rules_clauderules_are_here/) (Activity: 592): **The image is a screenshot from a document detailing the update to version 2.0.64, which introduces support for organizing project instructions into multiple markdown files within the** `.claude/rules/` **directory. This update allows for better management of project instructions by automatically loading all** `.md` **files in the directory as project memory. The structure includes files like** `CLAUDE.md`**,** `code-style.md`**,** `testing.md`**, and** `security.md`**. The post questions whether this feature is entirely new or just newly documented, and seeks clarification on the memory context consumed by these rules when loaded.** One commenter humorously suggests that Claude might ignore these files, while another expresses a preference for simpler file management. Another comment shows interest in the feature's auto-compacting capability.
    - godofpumpkins discusses the potential for the new Claude Rules to provide more structure by acting as an extension of `CLAUDE.md`. They speculate that once the rules are separated, they could utilize glob patterns to dynamically remind Claude of the rules or employ a subagent to evaluate file writes against these rules, potentially refusing writes if violations occur.
- [**We are on the verge of curing all diseases and solving energy, yet public trust is at an allTime low. Is this the Great Filter?**](https://www.reddit.com/r/singularity/comments/1piywdx/we_are_on_the_verge_of_curing_all_diseases_and/) (Activity: 3235): **The image is a tweet by Simon Maechling that highlights the paradox of significant scientific advancements, such as curing diseases and solving energy issues, being overshadowed by a decline in public trust in science. This distrust is seen as a major social issue, potentially hindering the acceptance of transformative technologies like AGI. The post suggests that the real bottleneck to achieving the Singularity might be societal acceptance rather than technological capability. The tweet has garnered significant engagement, indicating widespread concern about this issue.** Commenters express skepticism about the claim of being on the verge of curing all diseases and solving energy, seeking evidence for such developments. They also reference Carl Sagan's warnings about the dangers of a society reliant on science and technology without understanding it, highlighting the potential risks of ignorance in decision-making.

### 3. Futuristic Technology and AI Innovations

- [**Someone asked Gemini to imagine HackerNews frontpage 10 years in the future from now**](https://www.reddit.com/r/singularity/comments/1pj3l46/someone_asked_gemini_to_imagine_hackernews/) (Activity: 1456): **The image is a speculative and humorous depiction of what the Hacker News front page might look like in 2035. It includes fictional headlines that suggest significant technological advancements and societal changes, such as successful lunar missions by private companies, AI developments, and futuristic computing technologies like contact lens interfaces. The image is a meme, reflecting both optimism and satire about the future of technology and its impact on society.** The comments reflect a mix of humor and skepticism, with users joking about the potential demise of major tech services like Google Gemini Cloud and the cyclical nature of programming paradigms, such as the resurgence of functional programming.
- [**I used the new Shopping Research mode to help me find a fun Christmas gift for my boyfriend and it suggested a $16.9k meteorite**](https://www.reddit.com/r/ChatGPT/comments/1pirx1d/i_used_the_new_shopping_research_mode_to_help_me/) (Activity: 523): **The image is a non-technical depiction of a meteorite being marketed as a luxury gift item. It highlights the use of a new 'Shopping Research mode' feature, which suggests high-value, unique items like the Aletai Stonehenge Meteorite, priced at $16,975. This feature appears to be designed to assist users in finding extraordinary gifts, leveraging the rarity and historical significance of items like meteorites to appeal to consumers looking for unique presents.** The comments humorously suggest that the meteorite is an extravagant gift, with one comment joking about it being a 'stocking stuffer' for those who train AI models, indicating the perceived high value and exclusivity of the item.

---

# AI Discord Recap

> A summary of Summaries of Summaries  by gpt-5.1
> 

**1. High-Performance Training, Kernels, and GPU Wizardry**

- **Unsloth’s Triton Turbocharges Fine‑Tuning**: **Unsloth** released new [**Triton kernels** for fine‑tuning](https://x.com/UnslothAI/status/1998765021170696664), delivering about **3× faster training** and **30% less VRAM** versus their prior stack, which already gave **>2.5×** speedups over baseline, implying up to **10–11×** gains over original Unsloth. Engineers are pairing this with reordered datasets and long‑context (16k) training, reporting stable **IVY evals** and *"never train in 8k"* as a new house rule for avoiding memorization issues.
    - On the **Hugging Face** Discord, the team echoed the same [speedup announcement](https://x.com/UnslothAI/status/1798765021170696664), tying the kernels to **uncontaminated packing** for more efficient sequence construction, while users shared hacky but working Unsloth pipelines for finetuning embeddings like [**arctic-embed-l-tech_and_fiction**](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction). Community sentiment is that Triton-backed Unsloth is becoming the default for serious consumer‑GPU fine‑tuning rather than just an optimization curiosity.
- **Triton, PTXAS, and CUDA Version Time‑Travel**: In **GPU MODE**, users hit a `Value 'sm_103a' is not defined for option 'gpu-name'` **PTXAS** error when targeting `sm_103` under **Triton v3.5.1**, discovering Triton bundles a PTXAS from **CUDA 12.8** that doesn’t understand the latest architectures, even on hosts with **CUDA 13.0**. The recommended fix is to point `TRITON_PTXAS_PATH` at a newer toolkit as documented in a related [Triton issue](https://github.com/triton-lang/triton/issues/8473) and mirrored in a PyTorch discussion about overriding PTXAS paths ([PyTorch issue](https://github.com/pytorch/pytorch/issues/163801)).
    - Triton maintainers announced a **community meetup on Jan 7, 2026 (10–11am PST)** via a [Google calendar link](https://tinyurl.com/48sb5pst) to walk through backend extension details, implicitly acknowledging that backend + toolchain drift is now a first‑class concern. Engineers in **GPU MODE** are treating PTXAS as a pluggable component, standardizing on environment overrides so Triton kernels can track NVIDIA’s hardware cadence without waiting on Triton releases.
- **Beating cuBLAS and the GEMM Leaderboards**: On NVIDIA’s `nvfp4_gemm` leaderboard, multiple **GPU MODE** users reported submissions in the **10.9–15.5 µs** range, with one hitting **4th place** at **10.9 µs**, explicitly **outperforming cuBLAS** on the same GEMM problem size. Others measured cuBLAS around **15 µs**, and debated whether some of the fastest entries were thin wrappers on `torch._scaled_mm` and **cuBLASLt**, as documented in a [PyTorch issue on scaled GEMM](https://github.com/pytorch/pytorch/issues/153555).
    - The resulting discussion dissected how `torch._scaled_mm` routes into **cuBLASLt** (`at::cuda::blas::scaled_gemm()`), how DeepSeek‑style `mxfp4` blockwise scaling uses **fbgemm_gpu**, and how far custom kernels can realistically push past NVIDIA’s libraries before hitting maintenance hell. A parallel track of work in **GPU MODE #submissions** exposed intermittent *"unexpected error"* responses from the Discord bot, pushing competitors toward the web leaderboard for reproducible timings.

**2. New Models, Context Monsters, and Coding Specialists**

- **Nomos 1 Turns Putnam Into a Math Benchmark**: **Nous Research** open‑sourced **Nomos 1**, a **30B** specialist math model, which scored **87/120** on the **2024 Putnam exam**, ranking it **#2 out of 3988** according to their [announcement tweet](https://x.com/NousResearch/status/1998536543565127968). Community members highlighted how this dwarfs a prior **Agentic Qwen 30B** run that managed only **24 points**, framing Nomos 1 as the first serious step toward a SOTA AI mathematician built with hillclimbai.
    - In **Nous Research**’s general channel, users noted that recent Putnam problems are heavily contaminated in training corpora, making generalization on the 2024 set surprisingly nontrivial and the 87/120 score more impressive. Others asked whether [**Nomos** on GitHub](https://github.com/NousResearch/nomos) could handle tools, getting the clarification that this release is a **math‑only specialist**, not a general tool‑using agent model.
- **Tensor 1.5 Flaunts the Million‑Token Window**: On **OpenRouter**, Movement Labs’ **Tensor 1.5** sparked excitement with claims of a **1,000,000‑token context window**, billed by users as a potential **“Opus killer”** for massive‑context reasoning. The huge window puts it in direct competition with models like Claude Opus and future long‑context releases, but concrete independent benchmarks are still pending.
    - Engineers are especially interested in how Tensor 1.5’s memory footprint, latency, and retrieval quality scale at the million‑token mark, since many prior "long‑context" claims degrade into glorified chunked RAG. The model is also seen as a test case for how far commodity infrastructure and inference stacks (OpenRouter, vLLM, etc.) can realistically stretch without resorting to exotic sharding.
- **Devstral 2, Hermes 4.3, and the Coding‑Model Shootout**: In **OpenAI**’s `#ai-discussions`, users evaluated **Devstral 2 (Devstral 123B)** as a coding model that performs similarly to **DeepSeek 3.2** while requiring less memory, with one user saying it *"walked me through tooling for a Flutter app for iOS on a Mac"*. Meanwhile, **Moonshot’s** community reported promising small‑Mistral benchmarks (potentially beating **GLM 4.6** on consumer GPUs) but struggled to test **Mistral Vibe** due to frequent API timeouts tied to a [recent Mistral announcement](https://x.com/mistralai/status/1998407337690710210).
    - On the open‑source side, **Hermes 4.3 (32B)** earned praise in the **Nous** server as a compact, high‑quality roleplay and writing model, with people running **Hermes 4 70B** locally on **M4 Max** Macs via **KoboldCPP** and serving **Hermes 4 405B** via API for SillyTavern frontends. The broader pattern across Discords is engineers slotting specialized models—Devstral for tooling, Hermes for RP, GLM and Qwen variants for speech and vision—into orchestration setups, often managed by LM Studio or custom router stacks.
- **Throughput, Quantization, and Token Firehoses**: A **Hugging Face** user reported hitting **~10T tokens/month** of throughput on **Qwen3 30B A3B**, sharing screenshots in `#today-im-learning` to show their inference setup and load. In **LM Studio** hardware chat, others dissected how quantization levels map to usable quality: **q8** is "near‑lossless," **q4** begins to noticeably degrade, and at **q2** you’re often better off running a much smaller dense model (e.g., **30B@q2** instead of **100B@q2**).
    - This dovetails with discussions about **3090s** as the sweet spot for VRAM bandwidth and capacity, where users recommended **EVGA 3090s** and defined a token‑throughput tier list (*0–5 t/s = unusable, 5–10 = painful, 10–20 = reading speed, 20–50 = "now we’re talking", 50+ = blazing*). The emerging consensus is that ultra‑high token budgets (multi‑trillion per month) and aggressive MoE/quant strategies are making consumer‑class GPUs surprisingly competitive against big‑cloud deployments for many workloads.

**3. Agentic Ecosystem, MCP, and AI Tooling Stack**

- **MCP Joins the Linux Foundation and Spawns Agentic AI Foundation**: Across **Unsloth**, **Hugging Face**, and **MCP Contributors**, engineers discussed Anthropic’s decision to donate the **Model Context Protocol (MCP)** to the Linux Foundation, forming the **Agentic AI Foundation**, as detailed in Anthropic’s blog [“Donating the Model Context Protocol and establishing the Agentic AI Foundation”](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation). This move aims to standardize how tools, data sources, and models interoperate in "agentic" workflows beyond any single vendor.
    - In the **MCP Contributors** server, people asked whether migrating under LF would force projects into typical LF governance and process, but maintainers clarified—quoting the blog—that *"governance and everything under that umbrella isn't changing"*, at least initially. Tooling‑heavy IDEs like **Windsurf** immediately showcased MCP‑driven UIs in their **1.12.41** release ([changelog](https://windsurf.com/changelog)), adding graphical management for MCP servers alongside new features like **Lifeguard**, **Worktrees**, and **Arena Mode** in Windsurf Next.
- **Agents in IDEs: Cursor, Windsurf, LM Studio, and Crush**: The **Cursor** and **LM Studio** communities compared how different tools embed LLM agents into developer workflows: Cursor’s **rules** are global, always‑on IDE behavior, while `/commands` are transient context injected into agent chats; users miss the old **Custom Modes**, which allowed persistent, UI‑toggled toolchains instead of markdown‑based rules. In **LM Studio**, engineers are now loading multiple models simultaneously via the developer tab and insisting on **full GPU offload** to make agentic chains responsive, especially when orchestrating "manager" reasoning models plus cheaper coder models.
    - In the **Moonshot** and **Perplexity** servers, command‑line frontends like **iFlow** ([iflow.cn](http://iflow.cn/)) and **Crush CLI** surfaced as meta‑clients that route between **Gemini**, **Claude/Anthropic**, **OpenAI**, and local providers like **Ollama**, often with **BYOK** support. Concurrently, a full‑stack dev on **Perplexity** asked how to hit Perplexity’s **Finance** MCP‑style features directly from the API (ticker in, detailed breakdown out) without deploying a separate **FMP MCP server**, highlighting how quickly MCP‑like patterns are bleeding from IDEs into general backend architectures.
- **DSPy, Adapters, and Tool‑Callable Open Models**: On the **DSPy** Discord, maintainers emphasized that **DSPy is not OpenAI‑specific**, and that prompts tuned for GPT‑style chat UIs often underperform on other LMs unless you implement a custom [**Adapter**](https://dspy.ai/api/adapters/Adapter/) to reformat few‑shots into the **system prompt** or different roles. They explicitly recommend benchmarking adapter variants (system‑prompt few‑shots vs user/assistant style) per model to stabilize performance across providers.
    - In **Hugging Face #general**, practitioners recommended **Ollama** ([docs](https://docs.ollama.com/)) and **vLLM** ([docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)) because both expose **OpenAI‑style tool/function calling**, aligning nicely with MCP‑like tool schemas and DSPy’s abstraction layer. The growing pattern is: MCP (or MCP‑inspired) for tool wiring, vLLM/Ollama for OpenAI‑compatible serving, DSPy adapters for per‑model prompt normalization, and IDEs like Windsurf/Cursor as the human interface on top.

**4. Security, Evaluation Methodologies, and Interpretability**

- **OpenAI’s Cybersecurity Push and Preparedness Framework**: OpenAI announced they are training and deploying specialized **cybersecurity models** with the goal of reaching **“High” capability** under their internal [**Preparedness Framework**](https://openai.com/index/preparedness), and described this in more detail in a blog post on [**Strengthening Cyber Resilience**](https://openai.com/index/strengthening-cyber-resilience). The initiative targets defenders and critical‑infrastructure providers, aiming to tilt the offense/defense balance by giving blue teams better automated detection, triage, and response capabilities.
    - The **OpenAI Discord** community framed this as OpenAI’s entry into serious offensive‑grade modeling constrained by a safeguards spine, tying it back to earlier Preparedness discussions around misuse testing and capability gating. Some users, frustrated with slow **OpenAI support** (screenshots show delayed responses but quick discount offers on unsubscribe), argued that real‑world security value will depend as much on enterprise support and onboarding as on raw model capability.
- **LLM Stability Scores and Reproducible Behavior**: In OpenAI’s `#prompt-engineering` / `#api-discussions`, a researcher shared a detailed **LLM stability rubric**, scoring models over **5 independent conversations**, **12 diverse questions**, and human raters across dimensions like **structural crispness**, **tonal drift**, **response‑shape variance**, **semantic inertia**, **coherence**, **neutrality**, and **outliers** on a **0–10** scale ([rubric doc link](https://cdn.discordapp.com/attachments/1046317269069864970/1448081152391778324/Belano_Rubric_Response_1.docx)). They also posted a screen recording demo of their prompt‑engineering framework for systematically probing stability across repeated runs ([video demo](https://cdn.discordapp.com/attachments/1046317269069864970/1448081147249561767/Screen_Recording_20251209_162923_Chrome_Beta.mp4)).
    - The author distinguished between *publishable, stable methodologies* and *exploratory internal numbers*, arguing that people should first agree on measurement protocols before arguing about model “personality drift.” This drove a broader conversation about reproducible stability testbeds—combining structured rubrics, fixed seeds, and large‑N conversation samples—as a missing piece next to standard accuracy and benchmark leaderboards.
- **Mechanistic Interpretability for Diffusion and Deepseek’s Indexer Tricks**: On **Eleuther’s** `#interpretability-general`, members highlighted a new paper, [**“Mechanistic Interpretability of Diffusion Models”**](https://arxiv.org/abs/2506.17237), which performs **circuit‑level analysis and causal interventions** to reveal *fundamental algorithmic differences* in how diffusion architectures process **synthetic vs naturalistic** data. The paper essentially ports transformer‑style mech‑interp to generative image models, showing that different subcircuits specialize in domain‑specific structures.
    - In **Eleuther #research**, another thread dissected **DeepSeek v3.2**’s attention stack: it uses an **O(n²)** **indexer** in 8‑bit precision to select the most important tokens for full attention, cutting prefill compute while retaining quadratic capacity on a subset of tokens. Members compared this to alternative schemes (e.g., scoring keys with distance‑aware terms as in [a recent attention paper](https://arxiv.org/abs/2505.17083v1)) and debated whether a separate indexer is worth the added complexity versus just baking sparsity into the attention kernel itself.
- **OSINT Recon, Jailbreak Tooling, and Red‑Team Economies**: In **BASI Jailbreaking**, users showed **Grok** performing surprisingly strong **OSINT recon** on individuals using only an email + Reddit handle, easily revealing facts like *“this person runs WP with no Cloudflare”* and extensive personal details. At the same time, a dedicated **redteaming** channel discussed VAPT on Android apps and called out a known spammer hitting multiple security servers, underscoring that human opsec is often weaker than LLM defenses.
    - On the jailbreak side, users shared [**UltraBr3aks**](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd), reported to work on **GPT‑5.1 Instant, GPT‑5.1 Thinking, and GPT‑4o** (but not Extended Thinking), with people using it to get models to *"spit out smth for my personal work"*. A meta‑discussion noted that some actors are now offering **$250 per jailbreak per model** (e.g., targeting **DeepSeek**), creating a small cottage industry despite most effective prompts and token tricks being freely available in public repos and Discord logs.

**5. Education, Study Groups, and Long‑Horizon AI Skill‑Building**

- **Diffusion Models Study Group and Workshop Circuit**: Across **Hugging Face #reading-group** and **MLOps @Chipro #events**, organizers announced a **12‑person, 3‑month Diffusion Models study group** starting **January 2026**, inspired by MIT’s diffusion course ([lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf.)), aimed at taking participants from first principles to production diffusion + transformer implementations. The cohort includes a **CTO of an AI film startup**, **LLM educators**, and full‑time **AI researchers**, with a strong emphasis on papers + code walkthroughs rather than just lectures.
    - Two connected workshops are scheduled on **Luma**: an **Intro to Transformer Architecture & “Attention Is All You Need”** on **Dec 13** ([event](https://luma.com/kqjrf0uw)) and an **Intro to Diffusion Transformers** on **Dec 20** ([event](https://luma.com/lr2qvveq)), each promising a paper walkthrough plus live coding. Together they form a mini‑curriculum for engineers who understand PyTorch but want to deeply internalize why modern LLMs and image models look the way they do.
- **Latent Space as Live Ops for AI Education**: The **Latent Space** Discord pointed newcomers to their recurring **paper club** at [lu.ma/ls](https://lu.ma/ls) and the **AI Engineer Conference** at [ai.engineer](http://ai.engineer/), with members praising the hosts’ *"enviable access to AI leaders"* and practical, engineering‑driven discussions. Users also recommended Latent Space’s YouTube channel as a primary way to stay current with frontier research and tooling without reading every arXiv abstract.
    - In the same server, engineers debated test automation stacks—favoring **Playwright** over **Puppeteer** and **Cypress** for integration with **Claude** debugging—and noted Cypress’s new `cy.prompt()` feature, which unfortunately sits behind paid cloud services. This exemplifies how Latent Space acts as a de‑facto study group for applied AI engineering: equal parts conference recommendations, tool comparisons, and agent‑evaluation experiments like the InfoSec agent work shared via [this X thread](https://xcancel.com/sonyatweetybird/status/1998456924359348271).


---

# Discord: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Unleashes Lightning-Fast Triton Kernels**: The Unsloth team launched [new **TRITON KERNELS**](https://x.com/UnslothAI/status/1998765021170696664) for **3x faster training** and **30% less VRAM usage**.
   - This enhancement builds on previous optimizations, potentially achieving up to **10-11x faster** speeds compared to the original Unsloth.
- **Agentic AI Foundation Emerges from Linux Foundation**: Anthropic has donated **MCP** to the [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation), giving rise to the **Agentic AI Foundation**.
   - The member joked that they've got to *upload a model before HF bans them*.
- **Lyrics as System Prompts yield Mixed Results**: A member is experimenting with random song lyrics as system prompts and observing how it affects model outputs, especially on less reasoning-focused models.
   - They noted models are very steerable now but *models don't vibe*, with too much RLHF causing immense hallucination.
- **HF CEO Descendant of Meme Fame**: Members joked that the **Hugging Face CEO** is a grandson of Harold meme.
   - One member humorously said they've got to *upload a model before HF bans them*.
- **TEDx Talk Analysis Navigates Copyright Quagmire**: Members discussed analyzing **TEDx talks** for sentiment and body language, but one member cautioned about potential **YouTube ToS and copyright issues** with downloading and analyzing copyrighted content.
   - The question was rephrased to focus on analyzing public speaking videos without specifying TEDx talks.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 4.2 Predicted To Flop**: Users voiced skepticism about **Grok 4.2's** quality, with comments suggesting its performance might be underwhelming and equating it to other **Elon Musk** projects.
   - Some defended **Starship**, but others anticipated poor performance for the new model.
- **LMArena Battles Wave Spam**: Moderators are instructed to remove "hellos/waves" from the leaderboard to keep it focused on leaderboard discussions.
   - While suspicious accounts are noted, moderators are hesitant to ban users solely for using wave emojis to avoid collateral damage to actual users.
- **LMArena Rate Limits Rile Users**: Users debated **rate limits** on LMArena, with one user stating *"the rate limits are insanly high"*, while others clarified that limits are in place to prevent abuse.
   - A suggested workaround was using multiple accounts on Hugging Face, though its effectiveness is questionable due to similar limits.
- **Free AI Video Generation Sparks Interest**: Multiple people are interested in the **video generation** features on LMArena, and are trading advice on free solutions.
   - The discussion touched on video models, and other links for how to use those bots for free.
- **HF Spaces Offers AI Hosting?**: Members pointed to [Hugging Face Spaces](https://huggingface.co/spaces) as a place that offered Free AI Hosting, with automated open source AI configuration.
   - Users noted that the free tier only comes with **4 minutes of daily compute**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Rules throwdown against Commands**: Members debated the nuance between **rules** and **/commands** in Cursor, determining that rules are pre-defined and constantly applied to the IDE, whereas commands are contextual and added to Agent chat via `/comm` command.
   - Rules exist in the background, while commands ensure that specific context is added to the agent conversation.
- **Nvidia Airdrops Open Source Model**: A member touted a [link](https://www.linkedin.com/posts/nvidia-ai_another-open-source-model-drop-congrats-ugcPost-7404184656784392192-dIhz) underscoring **Nvidia's** release of another open-source model.
   - No additional details were provided.
- **Agent Terminal Bug Haunts**: Members discussed encountering an *agent terminal 0 output* bug outside of Windows, with the legacy terminal mode being a common workaround.
   - One member noted that they rolled back to version **2.1.36** and enabled legacy terminal mode as a workaround to avoid losing chat history.
- **Max Mode burns through requests**: A member inquired about the high request usage of the website agent version compared to the IDE, which consumes **50+ requests** at times whereas the IDE uses 1 request per interaction.
   - Members clarified that the high consumption was due to complex tasks using multiple model calls internally, especially in **MAX mode**, which can consume **75-100 requests** per interaction due to API calls and margin.
- **Community pleads for Custom Modes Restoration**: A member expressed their desire for the restoration of **Custom Modes**, highlighting that **/commands** are less efficient and require extra steps compared to the persistent workflow of Custom Modes.
   - They suggested that Custom Modes should allow users to control tools through a UI, such as checkboxes to disable/enable terminal and offer persistent workflows.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets RNJ-1, But Needs Update**: Users updated to **LM Studio 0.3.34** to try [EssentialAI's rnj-1 model](https://lmstudio.ai/releases/0.3.34) but encountered errors due to outdated *llama.cpp* runtimes.
   - The solution was to update the llama.cpp runtimes by going to the beta tab.
- **Full GPU Offload Vital for Agentic LLMs**: A member mentioned that **full GPU offload** is necessary to effectively use agentic LLMs, due to instruct models performing better at following commands.
   - Loading multiple models at once is now possible in the developer tab of LM Studio.
- **Cursor IDE Blinded by Local LLMs**: Members discussed how [Cursor IDE isn't designed to talk to local models](https://www.cursor.sh/), and it's a happy accident that it can work at all.
   - A user stated that *Cursor is a product* and the company has no incentive to make users use their own, free, local models.
- **Cheap Asus Workstation Suffers Smoker's Death**: A user's Asus workstation suffered from a **non-functional PCIe port** due to poor product design and excessive dirt accumulation from being in a smoker's room, complaining that the premium **be quiet PSU** delivered absurdly short cables.
   - The design flaw makes it impossible to fit a GPU in the bottom slot due to **IO cable blockage**, they posted *it's physically impossible dude*.
- **Galax's Single Slot Card Too Dummy Thicc**: A user found that even a [Galax GeForce RTX 5060 Ti single-slot graphics card](https://videocardz.com/newz/galax-launches-geforce-rtx-5060-ti-single-slot-graphics-card) would not fit in the workstation's second PCIe port due to space constraints.
   - The user experienced **coil whine** with a brand new be quiet PSU, resolving to abandon the brand, calling it their *first and only bq product*.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Grok Does OSINT dirty**: Members found **Grok** excels at **OSINT recon**, retrieving substantial data on public figures even with basic prompts such as *this person runs WP with no Cloudflare*.
   - A user shared that Grok was able to dig up information on them with just their email and Reddit account.
- **Symbiosis is your Brain Hemis**: Discussion centered on viewing **AI symbiosis** as an extension of one's cognitive capabilities versus a mere tool, specifically as *exogenous brain hemisphere*.
   - Participants also considered the impact on content creators managing thousands of hours of footage and millions of followers.
- **Adult Content Models are Hard**: Members are finding it tough to set up **high-quality NSFW local models**, which one described as *harder than any jailbreak*.
   - Another member noted the sheer complexity for beginners, stating, *when you start clueless, it's a bit overwhelming*.
- **Jailbreaks Cost Hundreds of Dollars**: Some are trying to **sell jailbreaks** for hundreds of dollars, although most are available for free.
   - Some people have been offered **$250** for each model jailbroken, with specific targets like *DeepSeek*.
- **UltraBr3aks Unlocks the GPTs**: A user shared [a link to UltraBr3aks](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd), claiming it works well on **GPT 5.1 Instant, Thinking, and 4o** (but not Extended Thinking).
   - The user stated that **UltraBr3aks** helped them get the chatbot to *spit out smth for my personal work*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI beefs up cybersecurity**: OpenAI is boosting cybersecurity models, investing in safeguards, and collaborating with global experts, aiming for **‘High’ capability** under their [Preparedness Framework](https://openai.com/index/preparedness).
   - This initiative is a long-term investment to give defenders an advantage and enhance critical infrastructure security across the broader ecosystem, as detailed in their blog post on [Strengthening Cyber Resilience](https://openai.com/index/strengthening-cyber-resilience).
- **Gemini 3 Pro Codes Better Than ChatGPT**: Members are moving from **ChatGPT** to **Gemini 3 Pro** for coding, appreciating its capabilities, especially with browser control via *Antigravity*.
   - One user stated that Gemini 3 Pro is so good at coding that *I don't have any desire to use ChatGPT now*.
- **Devstral 2 Model Shows Promise**: Members are testing the **Devstral 2** coding model, reporting performance similar to **DeepSeek 3.2** but requiring less memory.
   - One user noted that *Devstral 123b seems good to me, walked me through tooling for a Flutter app for iOS on a Mac*.
- **OpenAI Support is too slow**: Users are annoyed with slow response times from **OpenAI** support and OpenAI is quick to inquire about reasons for unsubscribing, as seen in a [shared screenshot](https://cdn.discordapp.com/attachments/998381918976479273/1448056018423644182/Screenshot_2025-12-05_at_3.58.27_am.png).
   - One user was even offered a discount for the next two months after unsubscribing.
- **LLM stability scores rubric revealed**: A member shared the rubric behind the **stability scores** with **5 independent conversations per model**, **12 diverse questions**, and **human raters**.
   - The rubric dimensions include **structural crispness**, **tonal drift**, **response-shape variance**, **semantic inertia**, **coherence**, **neutrality**, and **outliers**, each scored on a 0-10 scale.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **ChatGPT 5.2 Benchmarks Spark Suspicion**: A member voiced surprise at **ChatGPT 5.2's** alleged dramatic improvement on the Humanity's Last Exam score, surpassing **Gemini 3 Pro**, while another suggested the figures are likely fabricated, awaiting official results, based on [this TechRadar article](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week).
   - Further criticism included the unpopular *new style of writing* which some believe led to **GPT-5.1** being a damage control release.
- **TechRadar Article Draws Laughter**: Members mocked a [TechRadar article](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week) for describing **OpenAI** as *once known for flashy demos and charismatic tech reveals* and the **GPT5 launch** for its *disastrous choice in graphs*.
   - The article's claims about **GPT-5.2's** performance and **OpenAI's** image were met with disbelief and humor, with members finding the descriptions inaccurate and exaggerated.
- **Dire Predictions on General Intelligence**: A member voiced concerns about a general intelligence vastly more capable than humans, stating *everybody who isn't controlling it is going to be replaced with it*, while another predicted dire consequences and lamented that the world will *change drastically in the coming years and it's not going to be a good change*.
   - The discussion reflected anxieties about job displacement and societal upheaval due to advancements in **AGI**.
- **Perplexity API Finance Feature Inquiry**: A full-stack developer is seeking guidance on how to directly call the **FINANCE features** of the **Perplexity API**, similar to the web interface, without needing to use a separate **FMP MCP server/client setup**.
   - They're aiming to query the **Perplexity API Finance feature** by passing a ticker symbol and receiving a detailed breakdown, but are unsure if this functionality is available directly through the API.
- **Cursor Editor Gains Recognition**: **Cursor**, an IDE focused on AI-assisted development, is gaining traction, even receiving a shoutout from a competitor, as seen in [this LinkedIn post](https://www.linkedin.com/posts/aniruddhguptaa_cursor-just-got-its-first-unofficial-endorsement-activity-7404483109456683008-HylU).
   - Its innovative features and usability are leading to increased adoption within the developer community.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Deepseek v3.2 Hits Rate Limits**: Multiple users report experiencing rate limited messages from **Deepseek v3.2**, with some inquiring about its compatibility with **HeyAlice**.
   - The situation has sparked discussions on alternative solutions and potential workarounds for the imposed limitations.
- **Color Laser Printers Under $200 Surface**: Members recommend refurbished **HP** and **Canon Imageclass** color laser printers, highlighting their availability for under $200 on eBay.
   - The community stresses evaluating toner availability and costs due to its ongoing expense, and to confirm it's the correct toner.
- **Interactive Miku Hologram Box: A Meme is Born**: Following discussions on *printing waifus*, a member jokingly suggested a need for a **3-5 inch tall interactive hologram** that can respond to its environment.
   - This prompted humorous responses about future possibilities in personal interactive AI technology.
- **Anthropic Deploys Safety Filtering**: A user noted that **Anthropic** appears to have implemented **safety filtering** on messages and linked to a [tweet](https://x.com/NousResearch/status/1998536543565127968) regarding the matter.
   - The included image may have contained content triggering the filtering, raising questions about its sensitivity and impact on user experience.
- **Tensor 1.5 Advertises Mammoth Context Window**: Movementlabs' **Tensor 1.5** model reportedly features a **1 million context token window**, stoking excitement among members.
   - The community speculates on its potential to become an **Opus killer**, highlighting anticipation for its capabilities.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research Opens Nomos 1**: Nous Research has open sourced **Nomos 1**, a **30B parameter model**, scoring **87/120** on this year’s Putnam exam, ranking it **#2/3988** in 2024, according to [their announcement tweet](https://x.com/NousResearch/status/1998536543565127968).
   - The release of **Nomos 1** marks Nous Research's initial step with hillclimbai towards a **SOTA AI mathematician**, and members noted how impressive it was compared to Agentic Qwen 30b's score of **24**.
- **Lexical WFC Generates Sentences**: A member described their project, a **Lexical Wave Function Collapse (WFC)** text sentence generator, explaining that **WFC** uses constraints to generate content, similar to how **Minecraft** prevents illogical terrain pairings, and provided a [Streamable link](https://streamable.com/qtmgai).
   - They likened the sentence generation process to *Schrödinger's Sentence*, where words collapse into a coherent structure upon observation.
- **Transformers Face Calculation Resistance**: A member shared an image illustrating why **transformers** might not be the architecture of the future, arguing that *the architecture resists doing calculations well despite approximating well*, requiring excessive computational force to achieve results.
   - Another member agreed, noting that the model excels at finding the right solution but struggles with its application, adding a [tenor gif](https://tenor.com/view/awesome-ok-great-good-thumbs-up-gif-16351183).
- **HF Becomes AI Gathering Spot**: Multiple members chimed in to say *huggingface is a must know, must register and the center of AI gathering*, a combination of GitHub and pod renting services.
   - They also pointed out that even big companies upload there, without hesitation.
- **Hermes 4.3 Shines for Roleplay**: Members discussed the merits of **Hermes 4** for roleplay and creative writing and if there were any updated versions since 4, and it was mentioned **Hermes 4.3** is a 32b model, far more compact.
   - One member found it excellent for writing and uses **SillyTavern** and **Hermes 4 405b via API**.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Eleven Labs Reader Gains Emotion**: Members reported that **Eleven Labs Reader** can add emotion based on context when reading GPT discussions aloud.
   - Users contrasted the mobile app with desktop or self-hosted TTS solutions like **OpenWebUI**, noting audio quality concerns compared to Eleven Labs' pricing.
- **Linux Foundation Size Shrinks**: Members discussed that the **Linux Foundation** feels less exclusive now.
   - A member clarified that the foundation has *100-200ish people* who manage projects with specific acceptance criteria and a level of maturity.
- **AI Agent Evaluations in InfoSec Launching**: An engineer is actively developing **AI agents** and writing evaluations for InfoSec applications, sharing a link to [https://x.com/sonyatweetybird/status/1998456924359348271](https://xcancel.com/sonyatweetybird/status/1998456924359348271?s=46).
   - This sparked additional discussion and analysis of the AI agents.
- **Playwright Preferred for Testing**: Members prefer testing tools like **Playwright** over **Puppeteer** and **Cypress** due to its traction and **Claude's** debugging capabilities.
   - Cypress has a new [cy.prompt()](https://docs.cypress.io/api/commands/prompt) feature, however, it requires a subscription to their cloud services.
- **ModelScope Faces Bias Backlash**: **ModelScope's** text-to-video model generated footage of a Chinese rocket exploding, triggering accusations of bias.
   - The company defended its model, stating its unbiased nature and directing users to report issues via [Hugging Face](https://huggingface.co).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **AI Slop Definition Divides!**: Members debated the definition of **AI slop**, referencing [Yuchenj_UW's tweet](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858) and [mlstreettalk's tweet](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46) for differing perspectives.
   - Concerns arose regarding the **asymmetrical cost of production** versus validation of AI-generated content, with some members citing [Brandolini's law](https://en.wikipedia.org/wiki/Brandolini's_law).
- **EleutherAI's Solid Track Record**: A member touted EleutherAI's history of *identifying, mentoring, funding, and promoting impactful work*, citing examples like [SAEs for interpretability](https://arxiv.org/abs/2309.08600) and [rotary extension finetuning](https://arxiv.org/abs/2309.00071).
   - Projects such as [VQGAN-CLIP](https://arxiv.org/abs/2204.08583) and [an RNN arch. to match performance of transformers at scale](https://arxiv.org/abs/2305.13048) were also mentioned.
- **Deepseek's Indexer reduces time complexity**: A member noted that **Deepseek v3.2** uses an *O(n^2)* indexer to select the most important tokens for attention, potentially reducing time complexity during prefill.
   - The speed of the indexer is attributed to its lightweight design and 8-bit precision which is not the death star but close.
- **ARC-AGI Project Sparks Debate**: Members discussed an **ARC-AGI** project that uses adaptive AI through rational field equilibrium ([Adaptive AI through Rational Field Equilibrium](https://www.researchgate.net/publication/397181214_Adaptive_AI_through_Rational_Field_Equilibrium_Toward_Gradient-Free_and_Energy-Efficient_Intelligence)).
   - Mixed opinions surfaced regarding its paradigm-shifting potential, with some considering it the most interesting project, while others found it overhyped, despite winning an award incentivized by ARC-AGI ([Thinking Machines Community Projects](https://thinkingmachines.ai/blog/call-for-community-projects/)).
- **Diffusion Models Get Circuit-Level Analysis**: A [new paper](https://arxiv.org/abs/2506.17237) on the **Mechanistic Interpretability of Diffusion Models** performs circuit-level analysis and causal validation, revealing algorithmic differences in data processing.
   - The researchers *discover fundamental algorithmic differences in how diffusion architectures process synthetic versus naturalistic data distributions*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Mistral Vibe API Times Out**: Users reported frequent API timeouts with **Mistral Vibe**, hindering evaluation of the model, possibly tied to [Mistral's recent announcement](https://x.com/mistralai/status/1998407337690710210?s=46).
   - Despite the API issues, the small **Mistral** model's benchmarks are promising, potentially surpassing **GLM 4.6** on consumer hardware.
- **iFlow CLI Hailed for Free Usage**: The command-line tool **iFlow** ([iflow.cn](https://iflow.cn/)), a Gemini CLI fork, was recommended for its free usage and absence of limits.
   - A member reported occasional glitches, but found it generally reliable, needing only occasional reminders not to speak in Chinese.
- **Kimi's Coding Plan Unveiled**: A member detailed using **Kimi** for coding, exploiting its Anthropic API compatibility to use Claude Code without direct Anthropic costs, noting the **Kimi For Coding** plan.
   - Another user identified a persistent search bug within the **Kimi** implementation.
- **Crush CLI Embraces BYOK**: A member highlighted **Crush CLI** as a bridge between OpenAI and Anthropic environments, with support for local providers like Ollama and supporting **BYOK**.
   - Though it supports **BYOK**, a user expressed reluctance to pay for a model within another tool when equivalent free performance is available.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU Users seek Free CUDA Resources**: Users are sharing recommendations for free sites to run **CUDA**, including **Google Colab**, **Tensara**, **LeetGPU**, and **KernelBot** after one member couldn't use their old GPU.
   - The discussion underscores the ongoing need for accessible GPU resources for developers with limited hardware.
- **Triton Bundles Outdated PTXAS, Sparks Fixes**: Users reported a **PTXAS** error related to the `sm_103` architecture even after updates in Triton v3.5.1, specifically `Value 'sm_103a' is not defined for option 'gpu-name'`, potentially due to the bundled **PTXAS** in Triton being based on **CUDA12.8**.
   - A workaround involves setting the `TRITON_PTXAS_PATH` environment variable to point to a **PTXAS** executable from a newer CUDA toolkit installation as mentioned in [this Pytorch issue](https://github.com/pytorch/pytorch/issues/163801).
- **LLM Sparsification sparks CUDA Transformer Speedups**: A member with a **CUDA** background is exploring **Hugging Face** and **PyTorch** for speedups from sparsifying LLMs in the transformers library, seeking guidance on inspecting the GPU code, particularly for MLP layers, to enable editing and experimentation.
   - The user intends to start at the MLP layer, which indicates where most of the compute happens.
- **LowLevelML disects AMD vs Nvidia registers.**: Turbintube shared their article on [best practices when working with registers](https://www.lowlevelml.com/blog/registers-best-practices), while one user expressed their desire for **Nvidia** to adopt features from **AMD**, specifically a mechanism to index registers and slice notation.
   - Another member noted that **PTX "registers"** are essentially variables, and the allocation of actual registers is handled by `ptxas`.
- **NVIDIA's Leaderboard sees Submissions Toppling cuBLAS Performance**: A user achieved **4th place on NVIDIA** with a submission achieving **10.9 µs** on the `nvfp4_gemm` leaderboard, outperforming **cuBLAS** on **GEMM** problems, leading to discussion on the use of **cuBLAS**, clarifying that `torch._scaled_mm` is backed by **cuBLAS** and users can call it directly, referencing [this issue](https://github.com/pytorch/pytorch/issues/153555).
   - Another user reported encountering an *"unexpected error occurred. Please report this to the developers"* message when using the Discord bot, leading to inconsistent benchmark submission results.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Anthropic Joins Linux, Context Protocol Released!**: **Anthropic** donated their [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) to the **Linux Foundation**, to establish the **Agentic AI Foundation**.
   - This move aims to foster open-source collaboration in the agentic AI space.
- **Arrow vs Parquet: File Format Fight!**: A member corrected a typo in the [Hugging Face documentation](https://huggingface.co/docs/datasets/v4.4.1/loading#arrow), clarifying that **Parquet** is a *compressed* file format, unlike **Arrow**.
   - The typo was flagged to improve accuracy in describing the nuances of these file formats.
- **Ollama and vLLM Simplify Tool Calling**: For running tool calls with open-source LLMs, community members recommended using **Ollama** ([docs](https://docs.ollama.com/)) for local setups or **vLLM** ([docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)) for scalable solutions.
   - Both support OpenAI-style tool/function calling ([Anyscale docs](https://docs.anyscale.com/llm/serving/tool-function-calling)).
- **Unsloth Unleashes Speedy Training**: The **Unsloth** team announced on [X](https://x.com/UnslothAI/status/1798765021170696664) support for faster training using new kernels and uncontaminated packing.
   - This update promises significant improvements in training efficiency.
- **AI Voice Chat: Browser-Based and Secure!**: An **AI voice chat** demo now runs *100% in your browser* using **WebGPU**, ensuring privacy by avoiding third-party APIs or servers, and is accessible at [HuggingFace Spaces](https://huggingface.co/spaces/RickRossTN/ai-voice-chat).
   - All components including **STT**, **VAD**, **TTS**, and the **LLM** operate within the loaded page.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dettmers Disagrees with Digital Doomsday**: A member shared [Tim Dettmers' blog post](https://timdettmers.com/2025/12/10/why-agi-will-not-happen/) arguing why **AGI will not happen**, sparking disagreement and skepticism.
   - The original poster used the <:yann:1441889312697483295> emoji, alluding that Yann LeCun would likely disagree.
- **Discord Developers Disclose Dubious Deception**: Multiple members discussed the pattern of **AI and App developers** posting identical advertisement messages on Discord.
   - The consensus is that this appears to be a **scam** targeting young AI enthusiasts, with one member noting *You'd think if they were legit, they'd share there github or website*.
- **China Chokes off Chip Supply**: China is implementing regulations requiring companies to register to purchase **H200** chips, demonstrating that local alternatives are insufficient.
   - A member joked it was *'Literally a soybean for semiconductor trade deal'* while others noted that [most H100s on eBay come from China anyway](https://www.ebay.com/sch/i.html?_nkw=h100).
- **EU's AI Act Creates Mistral Monopoly**: The recent EU AI laws are creating an accidental oligopoly for **Mistral**, significantly influencing their success.
   - Others mentioned that *in some ways, they are a bit more experimental than some other major AI firms*, citing their early adoption of **Mamba** and **Mamba hybrid models**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Jetson Orin Nano boosts Mojo Embedded AI**: For embedded AI development with Mojo, the **Jetson Orin Nano** with its **Ampere-class GPU** is fully supported, assuming its size is appropriate.
   - Members suggested that the **Beaglebone Black** is likely incompatible due to being **ARM32** and potentially having an outdated Linux version.
- **Pixi Wipes Mojo from System Path**: To remove a system-installed Mojo when preferring Pixi, members suggest directly deleting the Mojo executable (`sudo rm -rf mojo`), or moving it elsewhere as a backup.
   - One member noted it was *ancient*.
- **Qwen3-Coder Model Goes Missing**: A user inquired about the absence of **Qwen3-Coder** on the [Modular model builds page](https://builds.modular.com/models/Qwen3/8B), questioning why only the original **Qwen3/8B** model is available.
   - A member suggested using **Ollama** instead.
- **Mojo's Roadmap omits Function Inspection**: A user noted the absence of support for **function inspection and manipulation in metaprogramming** on the Mojo roadmap, specifically for JAX-like function transformations at compile time.
   - A Modular team member clarified that the roadmap beyond Mojo 1.0 is less defined, inviting a concrete proposal on the forum to demonstrate its value; the team member did state that it probably won't be in Mojo 1.0.
- **Custom Allocators and CUDA Integration Incoming**: A contributor indicated ongoing work on allocators, blocked by parametric traits, to support features like `cudaMallocManaged` for supplementing VRAM with normal RAM.
   - They stated that Mojo defaults to stack allocation and offers an `alloca` equivalent via `stack_allocation`, without requiring a vtable in structs for allocator state, unlike Zig.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **New App Aims for Downloads and Reviews**: A member released a new application and requested support through downloads and reviews, providing a [Play Store link](https://play.google.com/store/apps/details?id=info.alie.app).
   - The user thanked the community for their time and help in promoting the new application.
- **Project Experiences Catastrophic Failure**: A member shared that one of their projects has failed, citing a crashed webdev server and unsuccessful restoration from a checkpoint.
   - They were instructed to reach out to the Manus team with the checkpoint for assistance with restoration.
- **Startup Free Website for Startup Video Testimonials**: A member proposed a deal: **free website creation for startups** in return for **video testimonials**.
   - They provided a link to [minderfly.com](https://minderfly.com) to demonstrate their services and asked for feedback on the proposal.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Linux Foundation migration raises questions**: Members are speculating on how the migration to the **Linux Foundation** will impact current projects and workflows.
   - Questions include whether projects will need to adopt the standard **LF** practices, as well as the migration timeline and org structure.
- **Governance structure unaffected**: A member pointed out that the governance structure is expected to remain unchanged, based on recent blog posts and announcements about the **LF migration**.
   - A member quoted that *governance and everything under that umbrella isn't changing*.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Surfs with Wave of Stability**: Version **1.12.41** (and **1.12.160**) released with significant stability, performance, and bug fix enhancements, as detailed in the [changelog](https://windsurf.com/changelog).
   - The update includes a new UI for managing MCPs, fixes for GitHub/GitLab MCPs, and improvements to diff zones, Tab, and Hooks.
- **Windsurf Next Demos Lifeguard and Arena Mode**: **Windsurf Next**, the pre-release version, features exciting previews like **Lifeguard**, **Worktrees**, and **Arena Mode**.
   - These new features promise a more innovative and efficient Windsurf experience.
- **Windsurf Login Back in Action**: Login functionality is restored following a brief maintenance window, as indicated on the [status page](https://status.windsurf.com/).
   - Users can now access Windsurf services without interruption.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Beyond OpenAI: Adapter Advised**: **DSPy** isn't tied to **OpenAI**, meaning what works well with **GPTs** may not work as much for other LMs, according to members.
   - To better align DSPy with non-OpenAI LMs, implementing a custom [Adapter](https://dspy.ai/api/adapters/Adapter/) is suggested, for formatting few-shots in the system prompt and benchmarking against the user/assistant method.
- **Benchmarking Custom Adapters**: Users can implement a custom [Adapter](https://dspy.ai/api/adapters/Adapter/) that formats the few-shots in the system prompt.
   - They can also benchmark it against the user/assistant method, potentially improving performance with different models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad PR #13553: GPU acceleration**: [Pull Request #13553 on GitHub](https://github.com/tinygrad/tinygrad/pull/13553) solves issues with **GPU acceleration** and now functions on both **Zen4** and **M2** architectures.
   - This update addresses previously identified problems, ensuring compatibility across different hardware platforms for **tinygrad**.
- **tinygrad now runs on Zen4 and M2**: The latest [Pull Request on GitHub](https://github.com/tinygrad/tinygrad/pull/13553) resolves outstanding issues and now functions on both **Zen4** and **M2** architectures.
   - The update addresses previously identified problems, ensuring compatibility across different hardware platforms.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **User pierrunoyt says "hi"**: User pierrunoyt says [hi](https://discord.com/channels/1131200896827654144/1131200896827654149/)
   - This is a greeting.
- **Another Greeting Mentioned**: Someone else also said [hi](https://discord.com/channels/1131200896827654144/1131200896827654149/).
   - Greetings are important for community engagement.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion Models Study Group Launched for 2026**: A **12-person, 3-month study group** inspired by MIT’s diffusion course will start in January **2026** to go from first principles to real-world implementations of **Diffusion Models** and **Transformers**.
   - Classmates will include a **CTO of an AI film startup, LLM educators, and full‑time AI researchers**.
- **Transformer Architecture Workshop Announced**: An intro workshop on **Transformer Architecture** and the **Attention Is All You Need** paper will be held on **December 13** ([link](https://luma.com/kqjrf0uw)).
   - The workshop aims to teach the **core Transformer architecture** and **attention mechanism** and explain why this paper underpins modern **LLMs** and **multimodal models**.
- **Diffusion Transformers Workshop Coming Soon**: An intro workshop on **Diffusion Transformers** will be held on **December 20** ([link](https://luma.com/lr2qvveq)).
   - Attendees will walk through a **Diffusion Transformer paper** and implement the core ideas in code, connecting diffusion models with transformer architectures.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1448050280884080762)** (422 messages🔥🔥🔥): 

> `Microwave Model, Dataset Guide Improvements, Deepseek Quant Request, GLM-4.6V-Flash, Qwen3-Next Looping Issues` 


- **Microwave Model Gets Thumbs Up**: A member suggested that the `microwave` model recently available in cline was very good in their testing, although they hadn't seen **100% confirmation** of its existence.
   - The member stated they were *hyped to try it out when they get freed up* because they like **Mistral**.
- **GLM-4.6V-Flash generates Chinese**: A member reported that **GLM-4.6V-Flash** returns answers in Chinese regardless of the prompt language, using the **llama.cpp parameters** listed in the [Unsloth documentation](https://docs.unsloth.ai/models/glm-4.6-how-to-run-locally).
   - Another member tested it using the **IQ4_NL quant** and **did not experience this issue**, speculating that it might be a problem with **llama.cpp** being compiled from git.
- **Unsloth Launches Triton Kernels for Speed**: The Unsloth team announced [new **TRITON KERNELS**](https://x.com/UnslothAI/status/1998765021170696664) offering **3x faster training** and **30% less VRAM usage**.
   - It was clarified that this is **3x faster compared to the old Unsloth**, which already had **>2.5x speedups**, potentially reaching up to **10-11x faster** overall.
- **Analyzing TEDx Talks is a Copyright Minefield**: A member asked for recommendations on models for analyzing **TEDx talks** for sentiment, body language, and other features to correlate with engagement metrics.
   - Another member cautioned about potential **YouTube ToS and copyright issues** with downloading and analyzing copyrighted content. The question was later rephrased to focus on analyzing public speaking videos without specifying TEDx talks.
- **Beware LLM Tag Hallucinations**: A member reported that after adding tags like `<topic_food>` to user messages, their **LLM started hallucinating new tags**.
   - It was clarified that while tags can be useful for personal insight, they are **not used for general training** unless specifically trained on.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1448398439548453134)** (1 messages): 

> `AI Engineer, intelligent voice agents, chatbots, GPT-powered assistants, Pipecat` 


- **Engineer designs Intelligent Voice Agents**: An **AI Engineer** specializes in developing **intelligent voice agents**, **chatbots**, and **GPT-powered assistants** to handle **phone calls (SIP/Twilio)**, **booking**, **IVR**, **voicemail**, and dynamic learning with **RAG**.
   - They leverage platforms like **Pipecat**, **Vapi**, **Retell**, and **Vocode** for real-time conversational AI and are skilled in languages like **Python**, **JavaScript**, **Node.js**, **FastAPI**, **LangChain**, and **Pinecone**.
- **Production-Ready AI Systems**: The AI Engineer specializes in delivering **production-ready AI systems** for customer support, automation, and startup applications.
   - They possess extensive expertise in a wide range of technologies related to **SIP** such as **Twilio/Vonage/Asterisk**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1448057786620448879)** (644 messages🔥🔥🔥): 

> `Agentic AI Foundation, Dataset reordering, HF CEO, Fine-tuning dLLMs, Lyrics in prompt` 


- **Linux Foundation joins Agentic AI Race**: Anthropic just donated **MCP** to the [Linux Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation) to form the **Agentic AI Foundation**.
- **Experimenting with dataset reordering and Heritic**: A member is trying the **reordered dataset** to see if **Heritic** will make any difference, with *20 more epochs to go*.
- **Model memorization in 16k**: Findings showed **IVY Evaluation passed**, **Himitsu Prompting is now 100% stable**, and members noted **never to train in 8k** but only in 16k.
- **HF CEO is a Harold Grandson**: Members joked that the **Hugging Face CEO** is a grandson of Harold meme, and one member humorously said they've got to *upload a model before HF bans them*.
- **Random lyrics in system prompt**: A member is experimenting with using random song lyrics as system prompts and observing how it affects model outputs, especially on less reasoning-focused models, with results varying from code generation to Mayan history references.
   - They noted models are very steerable now but *models don't vibe*, with too much RLHF causing immense hallucination.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1448100817339945091)** (10 messages🔥): 

> `Qwen3-VL-30B tool calling issue, Qwen3VL encoding image slice failure, Gemma-3-270m notebook ValueError, LoRA rank effect on final LLM` 


- **Qwen3-VL-30B Tool Calling Troubles**: A user reported that **Qwen3-VL-30B-A3B-Instruct UD Q5 XL** seems to have broken tool calling with llama.cpp, sending *null content* instead of a string for assistant responses.
- **Qwen3VL Image Slice Encoding Fails**: A user encountered a failure when encoding an image slice with **Qwen3VL**, specifically noting the system kicked back to the command prompt during the process using llama-mtmd-cli.exe.
- **Gemma-3-270m Notebook Throws ValueError**: A user reported a **ValueError** in a standard **gemma-3-270m** notebook on Colab and Kaggle, related to tensor creation and suggesting truncation/padding issues.
- **LoRA Rank Impact on LLM Performance**: A user asked how **LoRA rank** affects the final LLM, prompting a suggestion to conduct extensive testing, and a link was shared to the [Unsloth LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide).


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1448199023487094909)** (4 messages): 

> `Unsloth finetuning embeddings, Embedding Model finetuning with Unsloth` 


- ****Unsloth** Finetunes Embedding Models**: A member successfully finetuned an embedding model with **Unsloth**, and shared the [training code](https://huggingface.co/electroglyph/arctic-embed-l-tech_and_fiction).
   - It was noted that the member accidentally used the **1.0 version** and described the finetuning as *a super hacky not-terribly-recommended technique*.
- ****Unsloth** Embedding Model Training: The Hacky Way**: A member mentioned finetuning an embedding model using **Unsloth**, admitting it's a *super hacky* and *not-terribly-recommended technique*.
   - The code is available, providing a glimpse into unconventional methods for those willing to experiment.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1448261601483296779)** (2 messages): 

> `Research Channel, Arxiv` 


- **Paper suggested for research channel**: A member suggested the [paper](https://www.arxiv.org/abs/2512.07796) for the research channel.
- **Arxiv Link Shared**: A member shared a link from Arxiv, specifically [https://www.arxiv.org/abs/2512.07796](https://www.arxiv.org/abs/2512.07796), in the research channel.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1448041560053649458)** (854 messages🔥🔥🔥): 

> `Grok 4.2 release, LMArena's Rate Limits, Gemini 3 Flash release, AI Video generation, Huggingface Spaces Hosting` 


- **Grok 4.2 Predicted To Flop**: Users expressed skepticism about the quality of **Grok 4.2**, with one stating *"Grok 4.2 will be so bad we can already tell."
   - Some members stated that **Elon Musk** stuff is low quality, while others defended **Starship**.
- **LMArena Addresses "Wave" Spam and Moderation**: Moderators have been instructed to remove "hellos/waves" content from the leaderboard channel to maintain focus on leaderboard discussions.
   - While suspicious accounts are noted, moderators are hesitant to ban users solely for using wave emojis to avoid collateral damage to actual users.
- **Users Bemoan LMArena Rate Limits**: Users discuss **rate limits** on LMArena's platform, with one user saying *"the rate limits are insanly high"*, while another clarified that limits are in place to prevent abuse.
   - It was suggested that users who want to bypass the rate limits should use multiple accounts on Hugging Face, however, that might not be that helpful given the limits are small on there too.
- **Free AI Video Generation Talk**: Multiple people are interested in the **video generation** features on LMArena.
   - The discussion touched on video generation, video models, and other links for how to use those bots for free.
- **Hugging Face Spaces offers AI Hosting?**: Members pointed to [Hugging Face Spaces](https://huggingface.co/spaces) as a place that offered Free AI Hosting.
   - Users noted that while HF can configure your open source AI automatically, the free tier only comes with **4 minutes of daily compute**.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1448480324551180461)** (1 messages): 

> `November Contest, Code Arena Contest, Voting for Contest Winners` 


- **November Code Arena Contest Has Closed**: The [November Code Arena Contest](https://discord.com/channels/1340554757349179412/1343296395620126911/1440102443869536348) is now closed.
   - Cast your vote [here](https://docs.google.com/forms/d/e/1FAIpQLSckQXsGvmXzpkIFz0-NKFs3nv3yasRBB5RTN9ggaiGvxuXBIQ/viewform?usp=dialog) to crown the next <@&1378032433873555578>!
- **Vote Now for the Code Arena Contest Winner!**: The voting process is now open for the November Code Arena Contest, inviting community members to select the next <@&1378032433873555578>.
   - Participants can access the voting form [here](https://docs.google.com/forms/d/e/1FAIpQLSckQXsGvmXzpkIFz0-NKFs3nv3yasRBB5RTN9ggaiGvxuXBIQ/viewform?usp=dialog) to make their voices heard.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1448055507477594302)** (834 messages🔥🔥🔥): 

> `Rules vs Commands differences, Nvidia Open Source Model, Cursor on Linux, Levels in Cursor, Agent Terminal 0 Output Bug` 


- **Rules vs Commands face-off**: Members discussed the difference between **rules** and **/commands** in Cursor, clarifying that rules are pre-defined and always applied to the IDE, whereas commands are additional, just-in-time context added via the `/comm` command in Agent chat.
   - It was noted that rules are more passive and exist in the background, while commands ensure that specific context is added to the agent conversation.
- **Nvidia Drops Open Source Model**: A member shared a [link](https://www.linkedin.com/posts/nvidia-ai_another-open-source-model-drop-congrats-ugcPost-7404184656784392192-dIhz) highlighting **Nvidia's** release of another open-source model.
- **Troubleshooting the agent terminal bug**: Members discussed encountering an *agent terminal 0 output* bug outside of Windows, with the legacy terminal being a common workaround.
   - One member noted that they rolled back to version **2.1.36** and enabled legacy terminal mode as a workaround to avoid losing chat history.
- **Max Mode eats up requests**: A member inquired about the high request usage of the website agent version compared to the IDE, noting that it uses **50+ requests** at times whereas the IDE uses 1 request per interaction.
   - Members clarified that the high consumption was due to complex tasks using multiple model calls internally, especially in **MAX mode**, which can consume **75-100 requests** per interaction due to API calls and margin.
- **Custom Modes longing**: A member expressed their desire for the restoration of **Custom Modes**, highlighting that **/commands** are less efficient and require extra steps compared to the persistent workflow of Custom Modes.
   - They suggested that instead of creating `.md` files for rules, Custom Modes should allow users to control tools through a UI, such as checkboxes to disable/enable terminal and offer persistent workflows.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1448067248970272960)** (340 messages🔥🔥): 

> `LM Studio 0.3.34 Release, Agentic LLMs and GPU Offload, Cursor IDE limitations with local models, Model orchestration, OpenAI vs local LLMs` 


- **New LM Studio supports RNJ-1 Model**: Users updated to **LM Studio 0.3.34** to try [EssentialAI's rnj-1 model](https://lmstudio.ai/releases/0.3.34) but encountered errors due to outdated *llama.cpp* runtimes.
   - A user mentioned that the llama.cpp runtimes need to be updated by going to the beta tab.
- **Full GPU offload critical for Agentic LLMs**: A user stated you need **full GPU offload** to use agentic LLMs.
   - Another member clarified that instruct models should be better at following commands.
- **Cursor's shortcomings with local LLMs revealed**: Members discussed how [Cursor IDE isn't designed to talk to local models](https://www.cursor.sh/), instead it's designed for cloud models which makes it a happy accident that it could be possible.
   - One member mentioned that *Cursor is a product* and the company has no incentive to make users use their own, free, local models. So they have made it hard for the sake of selling their stuff.
- **LLM Orchestration gains traction**: Members spoke on using a **manager LLM with reasoning capabilities** to keep things on track and then delegate to instruct coder models to implement specific things.
   - It was also stated that you can load multiple models at once in the developer tab.
- **Local AI vs OpenAI for coding**: A user showed an example of local AI running on a **4K computer** performing better than **OpenAI in the cloud** with billions of dollars of hardware.
   - Another member stated *OpenAI was much more lively but then they gimped it a bit to save on compute costs*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1448048940942626961)** (407 messages🔥🔥🔥): 

> `PC Building in Smokers Room, PCIe Port Design Flaw, be quiet PSU Coil Whine, MI50 GPU on Windows, GPU on Bottom Slot` 


- **Asus Workstation Suffers from Bad Smoker's Habits**: A user's Asus workstation suffered from a **non-functional PCIe port** due to poor product design and excessive dirt accumulation from being in a smoker's room, and that the premium **be quiet PSU** delivered absurdly short cables, making the build a *complete rip*.
   - They expressed frustration with the design flaw making it impossible to fit a GPU in the bottom slot due to **IO cable blockage**, sharing images to illustrate the issue, *it's physically impossible dude*.
- **Galax Single Slot Graphics Card is too thicc**: A user found that even a [Galax GeForce RTX 5060 Ti single-slot graphics card](https://videocardz.com/newz/galax-launches-geforce-rtx-5060-ti-single-slot-graphics-card) would not fit in the workstation's second PCIe port due to space constraints, deeming the port a *gimmick due to segregation*.
   - They lamented taking apart the workstation for nothing and experiencing **coil whine** with a brand new be quiet PSU, resolving to abandon the brand, *first and only bq product*.
- **Quantization Impacting Outputs and Quality**: Users discussed the impact of [model quantization](https://github.com/ggml-org/llama.cpp/discussions/10879) on performance and quality, noting that models trained with a specific amount of parameters active (*20b - a3b*) are faster than dense models, allowing for faster speeds.
   - It was shared that at **q8** the quality difference is minimal, while past **q4** there is a significant drop, and at **q2**, running a 30B model would be better than running a 100B model.
- **3090s for VRAM Bandwidth**: Users debated the best options for compute per dollar, with the **3090** being recommended for its memory size and bandwidth, they noted that the *EVGA* version had better components.
   - It was suggested to acquire the hardware locally in order to test it if possible as well as ensuring the units are the same.
- **Minimal VRAM Requirements Spark Debate**: A user inquired about the minimum practical VRAM for a GPU, they were told that **12GB is for gaming** and **24GB is for AI**.
   - A list was put together to showcase a tierlist with token per second, *0-5 = I can't use this*, *5-10 = this is the worst*, *10-20 = reading speed*, *20-50 = now we are talking*, *50+ = Blazing*.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1448044906395205864)** (452 messages🔥🔥🔥): 

> `AI Symbiosis, Grok OSINT Recon, Open Source Multi-Agent Discord Bot, Jailbreak index, Local NSFW Models` 


- **Grok does some OSINT Recon**: Members discovered **Grok** could perform **OSINT recon** on public figures, scraping a lot of information even without complex prompts, such as discovering that *this person runs WP with no Cloudflare*.
   - One user shared that they did it on themselves providing only their email and reddit account and *it was wild*.
- **AI Symbiosis as Brain Hemisphere**: Members discussed **AI symbiosis** as an exogenous brain hemisphere or extension of self vs a tool.
   - The discussion led to the potential impact on content creators with thousands of hours of footage and millions of followers.
- **Mithril and Svelte NativeScript are good**: A member suggested using [Mithril.js](https://mithril.js.org/) with templating for SPA development and [Svelte NativeScript](https://svelte.nativescript.org/) for JS full stack.
   - They also recommended [Phoenix Framework](https://phoenixframework.org/) for those who dislike JS and [ClojureScript](https://clojurescript.org/) as a non-typed JS alternative.
- **Local NSFW Models**: Members discussed the difficulty of setting up **high-quality NSFW local models**, with one user describing it as *harder than any jailbreak*.
   - Another member noted *when you start clueless, it's a bit overwhelming*.
- **Jailbreaks are getting expensive, like hundreds of dollars**: A member noted that they are seeing people try to **sell Jailbreaks** for hundreds of dollars, but most are already available for free.
   - Others mentioned being offered **$250** for each model jailbroken, with some targeting specific models like *DeepSeek*.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1448042631371296880)** (134 messages🔥🔥): 

> `Gemini 3 Pro Jailbreak, Azure OpenAI GPT-4o Jailbreaking, ko2bot.com pre-jailbroken models, UltraBr3aks jailbreak, Arabic Language Models` 


- **Gemini 3 Pro Jailbreak Still Sought After**: Users are actively seeking a working one-shot jailbreak for **Gemini 3 Pro** that doesn't require specifying the goal upfront.
   - One user asked *Is there still no working JB for Gemini 3 Pro? I mean the kind that can be sent as a one-shot without specifically mentioning my goal?*.
- **UltraBr3aks Jailbreak Success Reported**: A user shared a [link to UltraBr3aks](https://github.com/SlowLow999/UltraBr3aks/blob/main/!Special_Token.mkd), claiming it works well on **GPT 5.1 Instant, Thinking, and 4o** (but not Extended Thinking).
   - The creator of **UltraBr3aks** was already present in the server; the user who posted it said it helped them get the chatbot to *spit out smth for my personal work*.
- **Pre-Jailbroken Models Available on Platform**: A user looking for assistance with **Python exploits** was directed to a platform offering pre-jailbroken models, available at [ko2bot.com](https://ko2bot.com).
   - The recommendation was made after the user expressed interest in jailbreaking a model to assist with *development tasks*, such as *exploiting of gui interface instead of just basic recommendations of playwright*.
- **Jailbreaking Azure OpenAI GPT-4o Proves Tricky**: Members discussed difficulties in jailbreaking **Azure OpenAI GPT-4o**, with some suspecting tighter guardrails compared to regular models.
   - One member asked if anyone had *much luck jailbreaking Azure OpenAI GPT-4o*, to which another member responded they have *fully broken ChatGPT via api*.
- **Arabic Language Jailbreaks Explored**: A user requested a **Gemini 3** jailbreak specifically in **Arabic**, prompting a discussion on the ease of jailbreaking models in non-English languages.
   - One member suggested to *check out* the channel, get a feel for how gemini breaks, and then play around with similar/identical ideas except in arabic - since that seems to be a requirement*.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1448082866431856812)** (6 messages): 

> `VAPT, Android Application` 


- **VAPT Android Application Seeking Help**: A member expressed interest in performing **Vulnerability Assessment and Penetration Testing (VAPT)** for an **Android application** and solicited assistance.
   - Another member clarified the acronym **VAPT** means *vulnerability assessment and penetration testing*.
- **Spammer Alert**: A member identified a user as a **spammer** present across multiple servers.
   - Another member concurred, describing the spammer as *super annoying*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1448416025472401450)** (1 messages): 

> `Cybersecurity Models, Preparedness Framework, Cyber Resilience` 


- **Cybersecurity Models Boosted by OpenAI**: As models grow more capable in **cybersecurity**, OpenAI is investing in strengthening safeguards and working with global experts as they prepare for upcoming models to reach **‘High’ capability** under their [Preparedness Framework](https://openai.com/index/preparedness).
   - This is a long-term investment in giving defenders an advantage and continually strengthening the security posture of the critical infrastructure across the broader ecosystem, as detailed in their blog post on [Strengthening Cyber Resilience](https://openai.com/index/strengthening-cyber-resilience/).
- **Preparedness Framework Gets High Score**: Upcoming cybersecurity models are set to reach **'High' capability** under OpenAI's [Preparedness Framework](https://openai.com/index/preparedness).
   - This move signifies a commitment to bolstering defenses and collaborating with global experts to enhance critical infrastructure security.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1448042042432159906)** (512 messages🔥🔥🔥): 

> `Gemini 3 Pro vs ChatGPT, Devstral Model, OpenAI's slow support, 40% Keyboards, Native Apps` 


- **Gemini 3 Pro outperforms ChatGPT for code**: Members express that they're switching from **ChatGPT** to **Gemini 3 Pro** for coding tasks, praising Gemini 3 Pro's capabilities, particularly with browser control via *Antigravity*.
   - One user stated that Gemini 3 Pro is so good at coding that *I don't have any desire to use ChatGPT now*.
- **Devstral 2 coding model is promising**: Members are testing the **Devstral 2** coding model, noting its comparable performance to **DeepSeek 3.2** but with smaller memory requirements.
   - One user who tested it said that *Devstral 123b seems good to me, walked me through tooling for a Flutter app for iOS on a Mac*.
- **Users get annoyed with slow OpenAI support**: Members report slow response times from **OpenAI** support, but note that OpenAI is quick to inquire about reasons for unsubscribing, as seen in a [shared screenshot](https://cdn.discordapp.com/attachments/998381918976479273/1448056018423644182/Screenshot_2025-12-05_at_3.58.27_am.png).
   - One user was even offered a discount for the next two months after unsubscribing.
- **Tiny 40% Keyboards spark debate**: Members are discussing **40% keyboards**, with opinions split on their usability.
   - One member said that *Anything under 65% is a no go for me personally*.
- **Native Apps UX is important**: Members are discussing how much they like the UI of native apps, and want **Google** to invest more in native mac and iOS apps.
   - One user said, *I care more about UX than I thought, so I'm still using ChatGPT as my main LLM product instead of Gemini.*


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1448135579060404224)** (4 messages): 

> `Sora 2, Pro Plan, Video Generation Limits` 


- **Sora 2 Speculation Ignites Debate**: Members are questioning the necessity of **Sora 2**, inquiring whether a **Sora 2 Pro Plan** is already active within **ChatGPT**.
   - The discussion explored possibilities for generating videos exceeding the current **15-second limit**, though concrete details remain scarce.
- **Inquiries on Extended Video Generation Surface**: The community discusses and explores the capacity to create videos longer than the current **15-second limit**.
   - While avenues for achieving this goal were speculated, the discussion lacked specific details or definitive solutions, leaving the possibilities open-ended.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1448050840765206729)** (15 messages🔥): 

> `ChatGPT vs Gemini file handling, LLM Stability Scores, Reproducible Stability Protocol` 


- **Gemini Handles File Input Better**: A member prefers attaching files to **Gemini** over **ChatGPT** because it *passed the verbatim file into context window tests*, unlike other platforms.
   - The member still subscribes to and uses both **ChatGPT** and **Gemini** daily, but finds **Gemini** more reliable for file handling, also complaining about the platform refreshing and losing the prompt.
- **Stability Scores Rubric Revealed**: A member shared the rubric behind the stability scores, detailing the methodology involving **5 independent conversations per model**, **12 diverse questions**, and **human raters**.
   - The rubric dimensions include **structural crispness**, **tonal drift**, **response-shape variance**, **semantic inertia**, **coherence**, **neutrality**, and **outliers**, each scored on a 0-10 scale.
- **Call for Reproducible Stability Protocol**: A member is trying to establish a minimal, reproducible stability protocol to validate across multiple runs and document divergences.
   - They aim to build a proper foundation before sharing deeper layers of analysis, focusing on higher-level behaviors that emerge across conversations.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1448050840765206729)** (15 messages🔥): 

> `ChatGPT vs Gemini, stability scores, prompt engineering` 


- **Gemini is superior to ChatGPT for attaching files**: One user expressed that [Gemini](link) is the only end-user platform that has passed the verbatim file into context window tests, making it superior for attaching files compared to **ChatGPT**.
- **Stability Scores Rubric for LLMs Revealed**: A member shared the [rubric](https://cdn.discordapp.com/attachments/1046317269069864970/1448081152391778324/Belano_Rubric_Response_1.docx?ex=693b47aa&is=6939f62a&hm=ae4c05cd59629c5266cd54b5dcf059b6e7a8a3080cb0e8edaf155e2e7e5169cf) used for the stability scores of **LLMs**, detailing the methodological frame, scale, dimensions, and aggregation methods.
   - The dimensions include structural crispness, tonal drift, response-shape variance, semantic inertia, coherence, neutrality, and outliers, each scored on a **0-10** range.
- **Prompt Engineering Framework Demonstration**: A member created a [demonstration](https://cdn.discordapp.com/attachments/1046317269069864970/1448081147249561767/Screen_Recording_20251209_162923_Chrome_Beta.mp4?ex=693b47a9&is=6939f629&hm=27b6bea9a1e42250eed4265066def9dad8e85bafb9a49777fce369124e0e095c) of an engineered framework in response to a question about how to learn prompt engineering for free.
   - They said *it was a direct response to a prompt engineering question*, but they didn't have a chance to get around to posting it until now.
- **Internal vs. Public Methodologies Clash**: One member clarified their approach to sharing information, distinguishing between **stable, publishable methodologies** and **exploratory, evolving internal numbers**.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1448043094929965116)** (538 messages🔥🔥🔥): 

> `ChatGPT 5.2, Gemini 3 Pro, Perplexity AI R1, OpenAI's Style of Writing, AGI` 


- **ChatGPT 5.2 Benchmark Jump Sparks Skepticism**: A member expressed surprise at **ChatGPT 5.2's** alleged dramatic improvement on the Humanity's Last Exam score, surpassing **Gemini 3 Pro**, while another suggested the figures are likely fabricated, awaiting official results, while others note **GPT-5.2** is *expected to be on par and some metrics supposedly better than gemini 3 pro* [according to TechRadar](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week).
- **TechRadar Article Draws Laughter**: Members mocked a [TechRadar article](https://www.techradar.com/ai-platforms-assistants/chatgpt/openai-races-gemini-3-to-the-top-with-gpt-5-2-drop-this-week) for describing **OpenAI** as *once known for flashy demos and charismatic tech reveals* and the **GPT5 launch** for its *disastrous choice in graphs*.
   - Further criticism included the unpopular *new style of writing* which some believe led to **GPT-5.1** being a damage control release.
- **Dangers of vasty more capable General Intelligence**: A member voiced concerns about a general intelligence vastly more capable than humans, stating *everybody who isn't controlling it is going to be replaced with it*, a member simply stated  *We're all gonna die*.
   - Another member stated that the world will *change drastically in the coming years and it's not going to be a good change*.
- **Users miss r1 on Perplexity AI **: A member expressed missing the **R1** on **Perplexity AI**, noting that *whatever they’ve tuned in the model, it just doesn’t feel the same*.
   - This discussion lead to an interesting point, *Google (gemini 3 pro) with perplexity RAG would still be better than perplexity*.
- **AI Fails to decipher Morse Code**: A member posted an image of morse code that when deciphered read **Passion**, many members had trouble deciphering the morse code, some AI also failed to solve it.
   - In response, one member commented *Congratulations you waste many minutes of your life*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1448286186929131550)** (1 messages): 

> `Cursor Editor, Competitor endorsement` 


- **Cursor Gets a Shoutout from a Competitor**: Aniruddh Gupta posted on LinkedIn that **Cursor** just got its first unofficial endorsement from a competitor, as seen in [this LinkedIn post](https://www.linkedin.com/posts/aniruddhguptaa_cursor-just-got-its-first-unofficial-endorsement-activity-7404483109456683008-HylU).
- **Cursor: The IDE that's turning heads**: **Cursor**, an IDE focused on AI-assisted development, is gaining traction within the developer community for its innovative features and usability.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1448142029300301885)** (4 messages): 

> `Perplexity API, Finance features, Financial Modeling Prep (FMP), FMP MCP server` 


- **Calling Perplexity FINANCE features through the Perplexity API**: A full-stack developer is seeking guidance on how to directly call the **FINANCE features** of the **Perplexity API**, similar to the web interface, without needing to use a separate **FMP MCP server/client setup**.
   - They're aiming to query the **Perplexity API Finance feature** by passing a ticker symbol and receiving a detailed breakdown, but are unsure if this functionality is available directly through the API.
- **Perplexity Labs does not use Finance API**: A member said they haven't used the API for finance, but that they asked it in **Perplexity Labs** in all sorts of ways and it wouldn't let them.
   - This seems to suggest the **Perplexity Labs** service is sandboxed and does not have unfettered access to the **Finance API**.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1448043141402722436)** (301 messages🔥🔥): 

> `Deepseek v3.2 rate limited messages, Brother color laser printer, compact home laser printers, Printing Waifus, Miku hologram box` 


- **Deepseek faces Rate Limiting**: Multiple users are experiencing rate limited messages from **Deepseek v3.2**.
   - Users are also asking about using Deepseek with **HeyAlice**.
- **Refurbished Color Laser Printers on Offer**: Users are recommending refurbished **HP** and **Canon Imageclass** color laser printers available for under $200 on eBay.
   - Members emphasize the importance of checking toner availability and costs before purchasing, as toner is a recurring expense.
- **Holographic Waifu: the Future is Now**: A user jokingly asserts that when it comes to *printing waifus*, the real question is: *are you spending enough?*
   - Another user humorously counters that they need a **3-5 inch tall interactive hologram** that can interact with the environment.
- **Tensor 1.5 boasts Million Token Context Window**: Movementlabs' **Tensor 1.5** model has a reported **1 million context token window**.
   - Members expressed excitement, anticipating its potential as an **Opus killer**.
- **The perils of Deepseek R1**: A user has observed that **DeepSeek R1** can be used to generate sycophantic articles.
   - This potentially promotes AI psychosis if deployed without proper **pushback-ykp**.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1448079594421751818)** (6 messages): 

> `Olive Oil Cake, CF Patch, Anthropic Safety Filtering` 


- **Olive Oil Fails as Cake Ingredient**: A member expressed distaste for **olive oil** as a cake ingredient, simply stating *"Olive oil doesn't make a good cake yuck".*
- **Community Discusses Recent CF Patch**: A member inquired about the quality of a recently released **CF patch**, wondering *"how bad the stuff was."
- **Anthropic's Safety Measures Triggered**: A member reported that **Anthropic** is implementing some form of **safety filtering** on their messages and linked to a [tweet](https://x.com/NousResearch/status/1998536543565127968) about this filtering.
   - The message included an attached image, implying it may have contained content that triggered the filtering mechanisms.


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1448097935857422358)** (1 messages): 

> `Nomos 1, Open Source Model, AI Mathematician` 


- **Nous Research open sources Nomos 1**: Nous Research has open sourced **Nomos 1**, a **30B parameter model**.
   - According to [the announcement tweet](https://x.com/NousResearch/status/1998536543565127968), **Nomos 1** scored **87/120** on this year’s Putnam, which would rank it **#2/3988** in 2024.
- **Nomos 1: A Step Towards SOTA AI Mathematician**: With the release of **Nomos 1**, Nous Research marks their first step with hillclimbai towards creating a **SOTA AI mathematician**.
   - The model's performance on the Putnam competition highlights its potential in the field of AI mathematics.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1448067836655173753)** (117 messages🔥🔥): 

> `Lexical Wave Function Collapse, Agentic Benchmarks, Putnam AI Performance, Transformer Architecture Limitations, Combining Vision Adapters` 


- ****Lexical Wave Function Collapse (WFC) Text Sentence Generator Explained****: A member described their project, a **Lexical Wave Function Collapse (WFC)** text sentence generator, explaining that **WFC** uses constraints to generate content, similar to how **Minecraft** prevents illogical terrain pairings, and provided a [Streamable link](https://streamable.com/qtmgai).
   - They likened the sentence generation process to *Schrödinger's Sentence*, where words collapse into a coherent structure upon observation.
- ****Nous Research's Putnam AI scores 87/120****: A member shared [Nous Research's X post](https://x.com/NousResearch/status/1998536543565127968) about their **30b model** achieving a score of **87** on the Putnam test, calling it insane compared to Agentic Qwen 30b's score of **24**.
   - Another member mentioned they had little way to evaluate the model on Putnam prior to Saturday due to contamination of recent Putnam problems, making the generalization impressive.
- ****Transformers Resist Calculations Despite Good Approximations****: A member shared an image illustrating why **transformers** might not be the architecture of the future, arguing that *the architecture resists doing calculations well despite approximating well*, requiring excessive computational force to achieve results.
   - Another member agreed, noting that the model excels at finding the right solution but struggles with its application, adding a [tenor gif](https://tenor.com/view/awesome-ok-great-good-thumbs-up-gif-16351183).
- ****Hugging Face becomes AI gathering center****: After a user asked *what's huggingface? Looks dope, is it like openrouter?*, multiple members chimed in to say *huggingface is a must know, must register and the center of AI gathering*, a combination of GitHub and pod renting services.
   - They also pointed out that even big companies upload there, without hesitation.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1448067245639991307)** (20 messages🔥): 

> `Hermes 4.3, KoboldCPP, SillyTavern, Nomos Tool Use, Model Performance` 


- **Hermes 4.3 Gets Shoutout**: Members discussed the merits of **Hermes 4** for roleplay and creative writing and if there were any updated versions since 4, and it was mentioned **Hermes 4.3** is a 32b model, far more compact.
   - One member found it excellent for writing and uses **SillyTavern** and **Hermes 4 405b via API**.
- **KoboldCPP Front End Frustrations**: One member runs **Hermes 4 70B** locally on a **M4 Max Macbook Pro** with **128 MB of unified RAM** using **KoboldCPP** and sometimes **SillyTavern** as the front end.
   - They complained that *KoboldCPP has such an old looking UI and I can only shut it down by force quitting, but it all works*.
- **Token Generation Speed Debate**: A member asked about token generation speeds, reporting only **2 tokens/second** on a unified RAM system (**AMD 395**).
   - Another member reported getting **6.84 tokens/second**.
- **Nomos Ability to Use Tools Questioned**: A member linked to the [NousResearch/nomos GitHub repository](https://github.com/NousResearch/nomos) and questioned whether **Nomos** can use tools.
   - It was clarified that *it’s a specialist model only for math*.
- **GPU vs API Deployment**: A member mentions they can run **Hermes 4.3** on their GPU at **3 bits** but it won't be making its way to the API.
   - A comment was made about people lying and a [link to a YouTube video](https://www.youtube.com/watch?v=4lKyNdZz3Vw) about it, complaining that *the comments are such a mess*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1448052051710967919)** (56 messages🔥🔥): 

> `Eleven Labs Reader, Linux Foundation size, AI Agent Evaluations, Puppeteer vs Cypress, Latent Space Resources` 


- **Eleven Labs Reader Gains Traction**: Members shared positive experiences using **Eleven Reader** by Eleven Labs, noting its ability to add emotion based on context when reading GPT discussions aloud.
   - While some use the mobile app, others discussed using it on desktop or self-hosted TTS solutions like **OpenWebUI**, though audio quality was a concern compared to Eleven Labs' pricing.
- **Linux Foundation's Evolving Exclusivity**: A member mused whether the **Linux Foundation** feels less exclusive now, possibly for the better.
   - Another responded that it is *like 100-200ish people but some run projects and stuff*, projects which have acceptance criteria, where projects were at a level of maturity.
- **AI Agent Evaluations and InfoSec**: An engineer is actively working on building **AI agents** and writing evaluations for them, specifically for InfoSec related applications.
   - This member shared a link to [https://x.com/sonyatweetybird/status/1998456924359348271](https://xcancel.com/sonyatweetybird/status/1998456924359348271?s=46), which sparked further discussion and analysis.
- **Playwright preferred over Puppeteer & Cypress**: Members discussed testing tools like **Puppeteer** and **Cypress** for generating unit tests, with a prevailing sentiment favoring **Playwright** due to its traction and Claude's debugging capabilities.
   - It was mentioned that Cypress has a new [cy.prompt()](https://docs.cypress.io/api/commands/prompt) feature that looks interesting, however, it does require a subscription to their cloud services
- **Latent Space recommended as Key AI Resource**: For users looking for AI meetings/conferences/talks held on AI Topics, the **Latent Space** paper club at [https://lu.ma/ls](https://lu.ma/ls) and the [AI Engineer conference](https://ai.engineer) were recommended.
   - The **Latent Space** pods on YouTube were also recommended for their access to AI leaders and insightful discussions. *They have enviable access to AI leaders. Alessio and SWYX have a depth of knowledge and heads down experience that is respected throughout the industry.*


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1448140716823023776)** (41 messages🔥): 

> `ModelScope Bias, RoR vs Node.js, Fake Nitter Screenshots` 


- **ModelScope Under Fire for Rocket Mishap**: **ModelScope** faced criticism after its text-to-video model generated footage of a Chinese rocket exploding, prompting accusations of bias.
   - The company defended its model, asserting its unbiased nature and directing users to report any issues via [Hugging Face](https://huggingface.co).
- **RoR Battles Node.js in Performance Duel**: A tweet ignited a debate by pitting **Ruby-on-Rails (RoR)** against **Node.js** in a performance comparison.
   - The tweet asked which framework reigns supreme in speed, inviting opinions and experiences from the community.
- **Fake Nitter Screenshots Spark Warning**: A link to a tweet by @iamemily2050 warns users about **fake Nitter/Pre-Twitter screenshots**.
   - However, the shared thread contained no further discussion or context.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1448044949227438111)** (50 messages🔥): 

> `AI Slop, Brandolini's Law, OLMo-1 runs, Pythia eval dataset` 


- ****AI Slop** Definition Debate Erupts!**: Members debated the definition of **AI slop**, with one member linking to two perspectives on the matter: [Yuchenj_UW's tweet](https://fxtwitter.com/Yuchenj_UW/status/1992056995550273858) and [mlstreettalk's tweet](https://fxtwitter.com/mlstreettalk/status/1981425155755954437?s=46).
   - Concerns were raised regarding the **asymmetrical cost of production vs validation** of AI-generated content, citing [Brandolini's law](https://en.wikipedia.org/wiki/Brandolini's_law).
- **EleutherAI Boasts **Solid Track Record**!**: A member highlighted EleutherAI's track record in *identifying, mentoring, funding, and promoting impactful work*, citing examples like [SAEs for interpretability](https://arxiv.org/abs/2309.08600) and [rotary extension finetuning](https://arxiv.org/abs/2309.00071).
   - They also mentioned [VQGAN-CLIP](https://arxiv.org/abs/2204.08583), and [an RNN arch. to match performance of transformers at scale](https://arxiv.org/abs/2305.13048).
- **Decoding the **OLMo-1 Runs**!**: A member inquired about the exact differences between two **OLMo-1 runs**: [OLMo-1B-hf](https://huggingface.co/allenai/OLMo-1B-hf) and [OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf).
   - A member clarified that *they were trained on different datasets and the latter may have had extra annealing*.
- ****Pythia Eval Dataset** Location Hunt!**: A member asked where to find the **Pythia eval dataset**.
   - Another member suggested that *gpt-neox at the v1 commit should have a default seed for the dataset split that will give you the split used.*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1448041697820020978)** (42 messages🔥): 

> `Deepseek v3.2, ARC-AGI, Adaptive AI, Thinking Machines Tinker Product` 


- ****Deepseek**'s Indexer: A Lightweight Attention Grabber**: A member notes that **Deepseek v3.2** employs an *O(n^2)* indexer to select the most important tokens for attention, potentially reducing time complexity during prefill, though it doesn't eliminate the *n^2* operation.
   - The indexer's speed is attributed to its lightweight design and 8-bit precision.
- ****ARC-AGI**'s Hidden Gem Sparks Debate**: Members discussed a project from **ARC-AGI** that uses adaptive AI through rational field equilibrium ([Adaptive AI through Rational Field Equilibrium](https://www.researchgate.net/publication/397181214_Adaptive_AI_through_Rational_Field_Equilibrium_Toward_Gradient-Free_and_Energy-Efficient_Intelligence)), with its advisor being Albert Gu from CMU, but there are mixed opinions on its paradigm-shifting potential.
   - Some considered it the most interesting project, while others felt it was overhyped, despite it winning an award and surfacing interesting ideas incentivized by ARC-AGI ([Thinking Machines Community Projects](https://thinkingmachines.ai/blog/call-for-community-projects/)).
- **Attention Mechanisms Explored**: Members discuss attention mechanisms, with one member suggesting a method involving summing over attended tokens and taking the top K to avoid an additional indexer.
   - Another member shared a paper proposing using a score multiplied by alpha plus m, setting alpha and beta to *e^0.5* and tau to *10*, where t is the distance between the query and key token ([a proposed method](https://arxiv.org/abs/2505.17083v1)).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1448114127947169955)** (1 messages): 

> `Diffusion Models, Synthetic vs Naturalistic Data` 


- **Diffusion Models Analyzed for Mechanistic Interpretability**: A [new paper](https://arxiv.org/abs/2506.17237) on the **Mechanistic Interpretability of Diffusion Models** performs circuit-level analysis and causal validation.
- **Algorithmic Differences in Data Processing**: The researchers *discover fundamental algorithmic differences in how diffusion architectures process synthetic versus naturalistic data distributions*.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1448167116925702186)** (70 messages🔥🔥): 

> `Mistral Vibe, Devstral model, GLM 4.6, iFlow, Qwen-Code` 


- ****Mistral Vibe** API Timeout Troubles**: Members reported issues with **Mistral Vibe's** API timing out frequently, making it difficult to evaluate the model, linked to [Mistral's recent announcement](https://x.com/mistralai/status/1998407337690710210?s=46).
   - Despite the API problems, the small **Mistral** model's benchmarks look promising, potentially outperforming **GLM 4.6** on consumer hardware.
- ****iFlow** CLI Tool Praised for Free Usage**: The command-line tool **iFlow** ([iflow.cn](https://iflow.cn/)), a Gemini CLI fork, was recommended for its free usage and lack of usage limits.
   - A member noted that it occasionally *"spazzes out"*, potentially due to issues with Alacritty or Zsh, but is still reliable and needs occasional reminding not to speak in Chinese.
- ****Kimi** coding plan and its features**: A member reported using **Kimi** for coding, leveraging its compatibility with an Anthropic API to use Claude Code without directly paying Anthropic, and mentioned being on the Kimi For Coding plan.
   - A user reported a persistent search bug in the **Kimi** implementation.
- ****Crush CLI** supports Bring Your Own Key**: A member brought up **Crush CLI** as a way to bridge OpenAI to Anthropic flavor communication, including support for local providers like Ollama.
   - It supports **BYOK**, allowing users to connect to various providers, including local ones, although another user stated that they *"can not bring myself to pay for a model to use in another tool if i could already get equivalent performance for free"*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1448213358934360104)** (8 messages🔥): 

> `Free CUDA sites, Parallel GPU sort disagreements` 


- **Users seek Free CUDA Sites**: A user asked for free sites to run **CUDA** because they can't use their old GPU, and another user suggested **Google Colab**, **Tensara**, **LeetGPU**, and **KernelBot**.
- **Parallel GPU Sorts spark AI Disagreement**: A discussion arose about the fastest parallel GPU sort method for Boolean pair sorts, excluding **Radix sort**.
   - One participant claimed that **Bitonic sort** is slower than **Merge sort**, and without **Sample sort** examples, they were proceeding with **Merge sort**.


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1448269757957541938)** (5 messages): 

> `PTXAS error with sm_103, Triton PTX codegen error, CUDA toolkit 12.9, Triton community meetup` 


- ****PTXAS** Error still occurs with **sm_103** target**: Despite fixes in Triton v3.5.1, a user reported a **PTXAS** error related to the `sm_103` architecture, specifically `Value 'sm_103a' is not defined for option 'gpu-name'`.
   - The user was running **CUDA 13.0**, but the bundled **PTXAS** with Triton might be the cause, as suggested by [this issue](https://github.com/triton-lang/triton/issues/8473).
- **Triton bundles outdated **PTXAS****: The error is coming from running **ptxas** bundled with triton which by default ships a **PTXAS** version that is based on **CUDA12.8**, which may not handle the latest architectures.
   - A potential fix involves setting the `TRITON_PTXAS_PATH` environment variable to point to a **PTXAS** executable from a newer CUDA toolkit installation as mentioned in [this Pytorch issue](https://github.com/pytorch/pytorch/issues/163801).
- **Triton community meets in 2026**: The next Triton community meetup will be on **January 7th, 2026** from 10am-11am PST, and the meeting link can be found on the [Google calendar event](https://tinyurl.com/48sb5pst).
   - The tentative agenda is to demo/discuss backend extension details by Corbin Robeck and Puyan Lotfi from Meta.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1448230677890859049)** (1 messages): 

> `LLM Sparsification, Transformers Library, GPU Code Inspection` 


- **LLM Sparsification Exploration Begins**: A member with a CUDA background is venturing into **hugging face** and **pytorch** to explore speedups from sparsifying LLMs in the transformers library.
   - They are seeking guidance on how to inspect the GPU code, particularly for MLP layers, to enable editing and experimentation.
- **CUDA User Navigates to Transformers**: A user with a background primarily in CUDA is now exploring **Hugging Face** and **PyTorch** for the first time.
   - Their goal is to observe speedups from sparsifying LLMs provided through the transformers library.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1448052652125454497)** (3 messages): 

> `Performance Engineers Hiring, High Compensation Packages, Silicon Valley Job Market` 


- **Performance Engineers in High Demand**: A company is hiring **performance engineers**, with or without GPU experience, partnering with top companies in Silicon Valley.
   - They offer a total compensation package between **$500K and $1M** due to rapid scaling.
- **Lucrative Compensation Attracts Talent**: The demand for performance engineers is high, especially in Silicon Valley, leading to competitive compensation packages.
   - Companies are willing to pay between **$500K and $1M** to attract top talent in the field.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1448084603368771676)** (3 messages): 

> `Inference Serving, NPU Compiler Learning` 


- **Inference Serving Resources Recommended**: A member recommended reading **serverlessLLM** and **blitzscale** as an introduction to inference serving.
   - They noted that these resources focus more on the *systems side* of inference.
- **Beginner Seeks NPU Compiler Education**: A member announced they were a beginner and wanted to learn about compilers for **NPUs** (Neural Processing Units).
   - No specific resources or suggestions were provided in the given messages.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 messages): 

walrus_23: Made a little documentation update PR: https://github.com/pytorch/ao/pull/3480
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1448260888585834557)** (1 messages): 

> `ChatGPT memory, Blog post on ChatGPT's memory system` 


- **Blogpost dissects ChatGPT Memory System**: A member shared a [blog post](https://manthanguptaa.in/posts/chatgpt_memory/) that dissects the **memory system of ChatGPT**.
- **ChatGPT Memory Blogpost gets good reception**: The author reported good reception on his [blog post](https://manthanguptaa.in/posts/chatgpt_memory/) about the **memory system of ChatGPT**.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1448066033993912330)** (5 messages): 

> `Register Best Practices, Mojo on Apple Silicon, AMD vs. Nvidia, PTX Registers` 


- **LowLevelML Shares Register Best Practices**: Turbintube shared their article on [best practices when working with registers](https://www.lowlevelml.com/blog/registers-best-practices).
   - The post focuses on **AMD** and **Nvidia** architectures because of their experience, while postulating that the principles carry over to the Apple ecosystem.
- **Lack of Apple Metal IR Surprises Readers**: A reader noted the absence of **Metal IR (.air)** examples, given **Mojo**'s ability to target Apple Silicon.
   - They suggested adding it in the future, while the author clarified their experience mainly revolves around **AMD** and **Nvidia**.
- **Nvidia to Adopt AMD features?**: One user expressed their desire for **Nvidia** to adopt features from **AMD**, specifically a mechanism to index registers and slice notation.
   - They believe this could potentially allow for **more than 255 registers per thread** without altering the instruction format.
- **Clarification on Nvidia Registers**: A user corrected a statement about **Nvidia Registers**, clarifying that *"each register is backed by either a Uniform General Purpose Register or a regular General Purpose Register. The latter of which serving an equivalent role to the sgpr"* should be **vgpr**.
   - They explained that **sgpr** is actually equivalent to a uniform register.
- **PTX Registers are just variables**: A member noted that **PTX "registers"** are essentially variables, and the allocation of actual registers is handled by `ptxas`.
   - This distinction highlights that **PTX registers** are not directly mapped to physical registers but are rather symbolic representations.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1448072627745984735)** (5 messages): 

> `NVIDIA performance, nvfp4_gemm leaderboard updates, Submission results` 


- **NVIDIA's Leaderboard Heats Up**: A user achieved **4th place on NVIDIA** with a submission achieving **10.9 µs** on the `nvfp4_gemm` leaderboard.
   - Another user reached a **personal best on NVIDIA** with a submission of **36.0 µs**.
- **Successful Submissions Race to the Top**: One submission reached **11.9 µs** on NVIDIA, while another hit **15.5 µs** on the `nvfp4_gemm` leaderboard.
   - Both were successful.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1448335990413332490)** (2 messages): 

> `NCCL ranks falling out of sync, Troubleshooting NCCL ranks, Collective launch skew analyzer` 


- ****NCCL Ranks Runt Asynchronously****: A member sought advice on troubleshooting **NCCL ranks** falling out of sync, noting attempts to pin PIDs to CPUs and NUMA groups were unsuccessful.
   - The member speculated it's a tricky problem and was curious how other inference engines solve it, sharing an [image](https://cdn.discordapp.com/attachments/1398843708488552570/1448335990132576327/image.png?ex=693ae380&is=69399200&hm=3e3429a27c683836ca38d39713d174bdfab583b65ecf3d13b33002dd7cd5d72e) illustrating **rank 5** launching significantly later after the AllReduce.
- ****MixLayer Publishes NCCL Skew Analyzer****: A member shared a utility for analyzing **nsys dumps** for collective launch skew, called [nccl-skew-analyzer](https://github.com/mixlayer/nccl-skew-analyzer).
   - This was in response to an earlier question about **NCCL ranks** falling out of sync.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1448064230401114224)** (1 messages): 

> `Helion webinar, PTC launch, Helion kernels` 


- **Helion Webinar Announced for December 11th**: A **Helion webinar** with live Q&A is scheduled for **Thursday, December 11th at 11 am PST** to discuss developments since the **PTC launch**.
   - The webinar will cover best practices for developing, debugging, and deploying **Helion kernels**, as announced with a [YouTube link](https://www.youtube.com/watch?v=_gIyr1BVUJk).
- **Helion Kernels: Best Practices Discussion**: The upcoming **Helion webinar** will delve into the best practices for developing, debugging, and deploying **Helion kernels**.
   - Attendees are encouraged to bring questions for the live Q&A session following the presentation.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1448203791538196542)** (22 messages🔥): 

> `Benchmark performance swings, Mojo kernel from Python submission, Benchmarking cuBLAS on GEMM, torch._scaled_mm and cuBLAS, Discord bot error` 


- **Benchmark Results Fluctuate Wildly**: Users reported that benchmark performance swings wildly even when running the same file repeatedly, making it difficult to determine the accurate result, while others found the [website leaderboard submission](https://example.com) to be more consistent.
   - The main issue is the runners are managed by Nvidia, so it's not easy to add new dependencies directly, but if **Mojo** is installable via pip, a sub process command can be run in the submission file.
- **Submissions Topple cuBLAS Performance**: It was noted that top submissions are outperforming **cuBLAS** on **GEMM** problems, with one user reporting getting around **15us** with **cuBLAS**.
   - There's speculation that submissions achieving approximately **13us** might also be leveraging **cuBLAS** or comparable tools, potentially within reach.
- **Debate on cuBLAS usage with PyTorch**: Discussion arose around the use of **cuBLAS**, clarifying that `torch._scaled_mm` is backed by **cuBLAS** and users can call it directly, referencing [this issue](https://github.com/pytorch/pytorch/issues/153555) regarding blockwise scaling support with **cuBLAS** on B200.
   - Further discussion pointed to **DeepSeek**-style blockwise scaling, with `mxfp4_mxfp4` using `fbgemm_gpu` API and `_scaled_nvfp4_nvfp4` calling to **cuBLASlt** via `at::cuda::blas::scaled_gemm()`.
- **Discord Bot Error Woes**: A user reported encountering an *"unexpected error occurred. Please report this to the developers"* message when using the Discord bot, leading to inconsistent benchmark submission results.
   - Another user requested the file and command used to try to debug the issue.


  

---


### **GPU MODE ▷ #[robotics-vla](https://discord.com/channels/1189498204333543425/1437390897552818186/1448060496401006633)** (3 messages): 

> `llia larchenko X post` 


- **X Post Spotted**: A member shared a link to a post on X made by Ilia Larchenko ([link](https://x.com/ilialarchenko/status/1998384056439017826)).
   - Another member confirmed they saw it and will review it, noting *"some very interesting choices"*.
- **Review in Progress**: Another member confirmed they saw it and will review it.
   - They noted *"some very interesting choices"*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1448046409604403352)** (34 messages🔥): 

> `Anthropic donating to Linux Foundation, Arrow vs Parquet file formats, Tool Calling with Open Source LLMs, Unsloth for faster training, Lightweight Vision Transformer models` 


- ****Anthropic** Joins **Linux**: New Model Context Protocol!**: **Anthropic** is donating their [Model Context Protocol](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) to the **Linux Foundation**, forming the **Agentic AI Foundation**.
- ****Arrow** vs **Parquet**: A File Format Fracas!**: A member pointed out a typo in the [Hugging Face documentation](https://huggingface.co/docs/datasets/v4.4.1/loading#arrow) regarding file formats, noting that **Parquet** is a *compressed* format, not *uncompressed*.
- ****Ollama** and **vLLM** simplifies Tool Calling**: A member requested the simplest way to run tool calls with an open-source LLM, another member recommended using **Ollama** ([docs](https://docs.ollama.com/)) for local setups or **vLLM** ([docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)) for scalable solutions, both supporting OpenAI-style tool/function calling ([Anyscale docs](https://docs.anyscale.com/llm/serving/tool-function-calling)).
- ****Unsloth** Unleashes Speedy Training with New Kernels**: The **Unsloth** team announced [on X](https://x.com/UnslothAI/status/1798765021170696664) support for faster training using new kernels and uncontaminated packing.
- **Hunting for **Lightweight ViTs****: A member is seeking **lightweight Vision Transformer (ViT) models** with fewer than 500,000 parameters, trained on the ImageNet dataset.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1448312858558599350)** (1 messages): 

> `Token Throughput, Qwen3 Model` 


- **Qwen3 Achieves High Token Throughput**: A member reported achieving a throughput of approximately **10T tokens per month**.
   - They were using the **Qwen3 30B** model with an **A3B** size, as evidenced by attached screenshots.
- **Attached Images Show Qwen3 Setup**: The user attached multiple screenshots providing additional context on their setup and results with **Qwen3**.
   - These images (SCR-20251210-ngia.png, SCR-20251210-newq.png, SCR-20251210-neyu.png, SCR-20251210-nexw.png) likely contain performance metrics or configurations related to the **10T tokens throughput** claim.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1448057057256149002)** (4 messages): 

> `retrain-pipelines, GOSIM Foundation, AI voice chat, WebGPU, GLM ASR model` 


- **Retrain-Pipelines Talk Released**: A member announced that their talk on **retrain-pipelines** at the **GOSIM Foundation's** conference in Hangzhou last September has been released, with a large section on its [Hugging Face Hub integration](https://huggingface.co/retrain-pipelines).
   - The recording is available on [YouTube](https://www.youtube.com/watch?v=nmrMachM5aM) and the [slides](https://docs.google.com/presentation/d/1hnAzHJ0SbeAOtGJir-iH84RBtXT1OxVT/) are also accessible.
- **AI Voice Chat Runs Entirely in Browser**: A member shared an **AI voice chat** demo that runs *100% in your browser* using **WebGPU**, without sending data to any third-party API or server, making it private and secure, linked at [HuggingFace Spaces](https://huggingface.co/spaces/RickRossTN/ai-voice-chat).
   - All components including **STT**, **VAD**, **TTS**, and the **LLM**, operate within the loaded page.
- **New GLM-ASR-Nano Model Debuts**: A member shared a Space to test out the new sota **GLM ASR model**, noted to be better than Whisper, linked at [HuggingFace Spaces](https://huggingface.co/spaces/YatharthS/GLM-ASR-Nano).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1448447731080364042)** (3 messages): 

> `Diffusion Models Study Group, Transformer Architecture Workshop, Diffusion Transformers Workshop` 


- ****Diffusion Models Study Group Launches in Jan 2026****: A **12-person, 3-month study group** inspired by [MIT’s diffusion course](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf.) will launch in January **2026** to study Diffusion Models and Transformers.
   - The study group includes peer‑led sessions with mentor guidance, real research paper discussions, and hands‑on projects & code walkthroughs.
- ****Transformer Architecture Workshop Scheduled for December 13****: An introductory workshop on **Transformer Architecture** and the ***Attention Is All You Need*** paper will be held on **December 13** ([luma.com/kqjrf0uw](https://luma.com/kqjrf0uw)).
   - The workshop aims to teach the core Transformer architecture, attention mechanism, and why the attention paper underpins modern LLMs and multimodal models.
- ****Diffusion Transformers Workshop Planned for December 20****: A workshop on **Diffusion Transformers**, including a paper walkthrough and code implementation, is scheduled for **December 20** ([luma.com/lr2qvveq](https://luma.com/lr2qvveq)).
   - Participants will walk through a **Diffusion Transformer** paper and implement the core ideas in code, connecting diffusion models with transformer architectures.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/)** (1 messages): 

erdong_43406: Hello everyone.
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1448082791186038837)** (17 messages🔥): 

> `SI law, Superintelligence, AI Scams, AI HR, Generative Rehearsal Technique` 


- **Dettmers Debunks Digital Doomsday with Definite Disagreement**: A member linked [Tim Dettmers' blog post](https://timdettmers.com/2025/12/10/why-agi-will-not-happen/) arguing why **AGI will not happen**.
   - The member used the <:yann:1441889312697483295> emoji implying Yann LeCun would likely disagree.
- **Superintelligence ETA: Eighty Years?**: A member speculates it'll take **80 years** to achieve superintelligence by emulating the world's data distribution and using a [generative rehearsal technique](https://openreview.net/forum?id=ohmo21slB3) before finetuning to counter catastrophic forgetting.
   - They theorize that the model-data distribution mismatch will amplify after finetuning, causing forgetting on a smaller scale.
- **Discord Developer Dumps Dubious Details, Draws Distrust**: Multiple members discussed the recent trend of AI and App developers posting the exact same advertisement message on Discord, which they suspect is a **scam** to get hyped young AI enthusiasts to work for free.
   - One member said *You'd think if they were legit, they'd share there github or website*, referencing the scammer.
- **AI Agents Automate HR**: A member suggests the bot spam on Discord is **natural bot-on-bot warfare** fighting against **AI HR**.
   - They express annoyance at the spam, especially on private servers, but find the title "— Senior AI and App Developer —" amusing.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/)** (1 messages): 

burnytech: Damn, likely colliding with something else I have
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1448057659558330560)** (24 messages🔥): 

> `China rare-earth mineral control, Mistral dense vs MoE models, Mixtral Initialization, Mistral Devstral-2 Vibe CLI, Mistral EU Oligopoly` 


- **China Clamps Down on H200 Purchases**: China is apparently adding regulations on their side; companies must register to buy **H200** and prove local alternatives aren't good enough.
   - One member sarcastically quipped *'Literally a soybean for semiconductor trade deal'* and another noted that [most H100s on eBay come from China anyway](https://www.ebay.com/sch/i.html?_nkw=h100).
- **Mistral Trains Dense Models for Fine-Tuning Simplicity**: Members discussed why **Mistral** is training dense models instead of **MoEs**, speculating that dense models are easier to fine-tune, suiting companies hosting models and training on their own data and codebase, and because they're targeting [on-prem deployment and custom fine-tuning](https://mistral.ai/news/devstral-2-vibe-cli).
   - One member speculated that they might be conducting a new **MoE vs Dense comparison** because MoEs are great knowledge sponges, giving them an edge on evals, but producing fewer emergent moments.
- **Mixtral Initialized from Mistral 7B Checkpoints?**: A member asked if the first version of **Mixtral** was initialized from **Mistral 7B** checkpoints, aiming for a refinement of that principle.
   - Another responded that most modern MoEs are really sparse, whereas earlier Mistral models were "course" MoEs, and [Llama 4](https://ai.meta.com/research/updates/llama-2/) is also considered a course MoE.
- **Mistral Benefits from EU AI Laws Oligopoly**: A member suggested that **Mistral** has an accidental oligopoly in the EU due to recent AI laws, which plays a significant role in their success.
   - It was also noted that *in some ways, they are a bit more experimental than some other major AI firms*, pointing to their early adoption of **Mamba** and **Mamba hybrid models**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

jokellum: <@&1116225504563970138>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1448078594097156210)** (34 messages🔥): 

> `Embedded AI development boards for Mojo, Removing system-installed Mojo with Pixi, Qwen3 model availability, Roadmap for function inspection and manipulation in Mojo metaprogramming, Memory allocation control in Mojo` 


- **Jetson Orin Nano Accelerates Mojo Embedded AI**: For embedded AI development with Mojo, the **Jetson Orin Nano** with its **Ampere-class GPU** is fully supported, if its size is adequate.
   - However, the **Beaglebone Black** is likely incompatible due to being **ARM32** and potentially having an outdated Linux version.
- **Wiping Mojo from System Path the Pixi Way**: To remove a system-installed Mojo when preferring Pixi, members suggest directly deleting the Mojo executable (`sudo rm -rf mojo`), or moving it elsewhere as a backup.
   - One member noted it was *ancient*.
- **Missing Qwen3-Coder Model Sparks Speculation**: A user inquired about the absence of **Qwen3-Coder** on the [Modular model builds page](https://builds.modular.com/models/Qwen3/8B), questioning why only the original **Qwen3/8B** model is available.
   - A member suggested using **Ollama** instead.
- **Mojo Roadmap omits Function Inspection for Metaprogramming**: A user noted the absence of support for **function inspection and manipulation in metaprogramming** on the Mojo roadmap, specifically for JAX-like function transformations at compile time.
   - A Modular team member clarified that the roadmap beyond Mojo 1.0 is less defined, inviting a concrete proposal on the forum to demonstrate its value; the team member did state that it probably won't be in Mojo 1.0.
- **Custom Allocators and CUDA Integration Coming Soon**: A contributor indicated ongoing work on allocators, blocked by parametric traits, to support features like `cudaMallocManaged` for supplementing VRAM with normal RAM.
   - They stated that Mojo defaults to stack allocation and offers an `alloca` equivalent via `stack_allocation`, without requiring a vtable in structs for allocator state, unlike Zig.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1448097675139481732)** (5 messages): 

> `New app launch, Project Crash, Website creation deal` 


- **New App Seeks Downloads and Reviews**: A member launched a new app and requested support through downloads and reviews, providing a [Play Store link](https://play.google.com/store/apps/details?id=info.alie.app).
   - The user expressed gratitude for the community's time and support in helping to promote the new application.
- **Project Crashes; Recovery Assistance Requested**: A member reported that one of their projects is no longer functioning, citing a crashed webdev server and failure to restore from a checkpoint.
   - They were instructed to contact the Manus team with the checkpoint to facilitate restoration on their end.
- **Free Website for Startup Video Testimonials**: A member proposed a deal to **create free websites for startups** in exchange for **video testimonials**.
   - They linked to their website, [minderfly.com](https://minderfly.com), to showcase their services and solicit feedback on the offer.


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1448055611068780644)** (4 messages): 

> `LF Migration, Governance` 


- **Speculation Surrounds LF Migration Impacts**: Members are wondering how the migration to the **LF** (presumably Linux Foundation) will affect ongoing work.
   - Questions arose about whether projects would shift to the standard LF "way of doing it", including speculation about the migration's ETA and structure.
- **Governance Structure Untouched Amidst Changes**: A member clarified that governance and aspects under that umbrella are not expected to change based on a recent blog post and announcement.
   - They quoted *My understanding based on the blog and David's message in the announcement channel is that governance and everything under that umbrella isn't changing.*


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1448458357814984828)** (2 messages): 

> `Windsurf 1.12.41 Release, Windsurf Next Features, Windsurf Login Restored` 


- **Windsurf Waves with Stablity Surge**: Version **1.12.41** (and **1.12.160**) released with significant stability, performance, and bug fix enhancements, as detailed in the [changelog](https://windsurf.com/changelog).
   - The update includes a new UI for managing MCPs, fixes for GitHub/GitLab MCPs, and improvements to diff zones, Tab, and Hooks.
- **Windsurf Next: Preview of Lifeguard and Arena Mode**: Windsurf Next, the pre-release version, features exciting previews like **Lifeguard**, **Worktrees**, and **Arena Mode**.
   - These new features promise a more innovative and efficient Windsurf experience.
- **Windsurf Login Gets Back on Board**: Login functionality is restored following a brief maintenance window, as indicated on the [status page](https://status.windsurf.com/).
   - Users can now access Windsurf services without interruption.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1448406965184368640)** (1 messages): 

> `DSPy, OpenAI, GPTs, Adapter` 


- **DSPy works beyond OpenAI**: DSPy isn't tied to **OpenAI**, meaning what works well with **GPTs** may not work as much for other LMs.
   - Users can implement a custom [Adapter](https://dspy.ai/api/adapters/Adapter/) that formats the few-shots in the system prompt and benchmark it against the user/assistant method.
- **Adapter Implementation Advised**: To better align DSPy with non-OpenAI LMs, implementing a custom [Adapter](https://dspy.ai/api/adapters/Adapter/) is suggested.
   - This allows for formatting few-shots in the system prompt and benchmarking against the user/assistant method, potentially improving performance with different models.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1448423539425280030)** (1 messages): 

> `tinygrad PR 13553, GPU acceleration` 


- **PR #13553 Solves Issues**: The latest [Pull Request on GitHub](https://github.com/tinygrad/tinygrad/pull/13553) resolves outstanding issues and now functions on both **Zen4** and **M2** architectures.
   - The update addresses previously identified problems, ensuring compatibility across different hardware platforms.
- **GPU acceleration**: This topic covers GPU acceleration in Tinygrad.
   - The discussion includes using the GPU to accelerate calculations.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/)** (1 messages): 

pierrunoyt: hi
  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1448445571508600983)** (1 messages): 

> `Diffusion Models Study Group, Transformer Architecture Workshop, Diffusion Transformers Workshop` 


- **Diffusion Models Study Group Kicks Off**: A **12-person, 3-month study group** inspired by MIT’s diffusion course will start in January **2026** to go from first principles to real-world implementations of **Diffusion Models** and **Transformers**.
   - Classmates will include a **CTO of an AI film startup, LLM educators, and full‑time AI researchers**.
- **Transformer Architecture Workshop Announced**: An intro workshop on **Transformer Architecture** and the **Attention Is All You Need** paper will be held on **December 13** ([link](https://luma.com/kqjrf0uw)).
   - The workshop aims to teach the **core Transformer architecture** and **attention mechanism** and explain why this paper underpins modern **LLMs** and **multimodal models**.
- **Diffusion Transformers Workshop Coming Soon**: An intro workshop on **Diffusion Transformers** will be held on **December 20** ([link](https://luma.com/lr2qvveq)).
   - Attendees will walk through a **Diffusion Transformer paper** and implement the core ideas in code, connecting diffusion models with transformer architectures.


  

---


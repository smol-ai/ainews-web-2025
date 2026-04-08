---
id: MjAyNS0x
title: not much happened today
date: '2025-10-30T05:44:39.731046Z'
description: >-
  **Moonshot AI** released **Kimi Linear (KDA)** with day-0 infrastructure and
  strong long-context metrics, achieving up to **75% KV cache reduction** and
  **6x decoding throughput**. **MiniMax M2** pivoted to full attention for
  multi-hop reasoning, maintaining strong agentic coding performance with **200k
  context** and **~100 TPS**. **ByteDance**, **Princeton**, and **Mila**
  introduced **Looped LLMs** showing efficiency gains comparable to larger
  transformers. **OpenAI**'s **Aardvark (GPT-5)** entered private beta as an
  agentic security researcher for scalable vulnerability discovery. **Cursor**
  launched faster cloud coding agents, though transparency concerns arose
  regarding base-model provenance. **Cognition** released a public beta for a
  desktop/mobile tool-use agent named Devin. The community discussed advanced
  attention mechanisms and adaptive compute techniques.
companies:
  - moonshot-ai
  - minimax
  - bytedance
  - princeton
  - mila
  - openai
  - cursor
  - cognition
  - hkust
models:
  - kimi-linear
  - kimi-delta-attention
  - minimax-m2
  - looped-llms
  - aardvark-gpt-5
topics:
  - long-context
  - attention-mechanisms
  - agentic-ai
  - tool-use
  - adaptive-compute
  - coding-agents
  - performance-optimization
  - memory-optimization
  - reinforcement-learning
  - model-architecture
people:
  - kimi_moonshot
  - scaling01
  - uniartisan
  - omarsar0
  - aicodeking
  - songlinyang4
  - iscienceluvr
  - nrehiew_
  - gdb
  - embeddedsec
  - auchenberg
  - simonw
---


**a quiet day**

> AI News for 10/29/2025-10/30/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 5621 messages) for you. Estimated reading time saved (at 200wpm): 490 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Congrats HuggingFace on the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#first-things-first-evals-before-everything-else). and welcome [Beyang (Amp) and Skyler (MiniMax)](https://x.com/swyx/status/1983939826069205340) to AIE CODE, and check out [the Stripe / ACP Latent Space pod](https://www.latent.space/p/stripe)!

---

# AI Twitter Recap

**Kimi Linear (KDA), Minimax M2, and the linear-attention wars**

- **Kimi Linear (KDA) ships with day-0 infra and strong long-context metrics**: Moonshot AI released the Kimi Linear tech report and checkpoints—a hybrid architecture interleaving **Kimi Delta Attention (KDA)** with MLA (≈3:1 KDA:MLA), open-sourcing optimized KDA CUDA kernels and integrating into vLLM on day one. Reported gains: up to **75% KV cache reduction**, up to **6x decoding throughput** (6.3x TPOT at 1M-context), and competitive or better quality than full attention on long-context and RL long-form reasoning tasks. vLLM shows **RULER 128k = 84.3 with ~4x speedup** vs baseline and confirms the memory/throughput wins. Notably, the team also reports effective long-context without positional encodings in MLA layers (“NoPE” + position-aware mechanism elsewhere). Links: [@Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/1983937694360322136), [@scaling01](https://twitter.com/scaling01/status/1983926811051384965), [vLLM](https://twitter.com/vllm_project/status/1983941708233765149), [@uniartisan](https://twitter.com/uniartisan/status/1983941443283775780).
- **Minimax M2’s full-attention pivot vs. hybrid approaches**: MiniMax publicly reflected on challenges with earlier hybrid/sliding-window variants for multi-hop reasoning and switched M2 to full attention—yet M2 still posts strong agentic coding performance (e.g., top open-weight on several user evals) with 200k context, ~100 TPS, and broad toolchain support, now free to try for a limited time. Community commentary notes M2’s earlier linear variant was simple and that better hybrids (like KDA) remain promising for efficiency if multi-hop degradation is small. Links: [@omarsar0](https://twitter.com/omarsar0/status/1983915573215162873), [vLLM M2 support](https://twitter.com/vllm_project/status/1983936128878059541), [@aicodeking](https://twitter.com/aicodeking/status/1983934597353402797), [@SonglinYang4](https://twitter.com/SonglinYang4/status/1984021551914926514).
- **Latent looping and adaptive compute**: ByteDance/Princeton/Mila’s “Looped LLMs” show **1.4B/2.6B LoopLMs (7.7T tokens)** matching ~4B/8B standard transformers across most benchmarks—evidence that looped latent reasoning can trade wall-clock for quality and data efficiency, and may scale with MoE. Project links: [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1983864569035690350), [project/paper](https://twitter.com/iScienceLuvr/status/1983864571095085511). Community was also deep in the weeds on DeltaNet/RetNet/Mamba-v2 decay vs. delta-rule variants and MLA partial-RoPE/NoPE tradeoffs (e.g., [@nrehiew_](https://twitter.com/nrehiew_/status/1983891931823505518), [@Grad62304977](https://twitter.com/Grad62304977/status/1983913767118229953)).

**Agentic coding and tool-use systems**

- **OpenAI’s Aardvark (GPT-5) enters private beta**: Aardvark is positioned as an “agentic security researcher” that reads/analyzes code, writes and runs tests, and proposes patches—early users frame it as a glimpse into scalable vulnerability discovery and remediation. Links: [@OpenAI](https://twitter.com/OpenAI/status/1983956431360659467), [@gdb](https://twitter.com/gdb/status/1983971650531160319), [@embeddedsec](https://twitter.com/embeddedsec/status/1983956550239842474).
- **Coding agents shipping fast (and getting scrutiny)**: Cursor rolled out faster, more reliable cloud agents and shared internal usage patterns ([launch](https://twitter.com/cursor_ai/status/1983954528933421419), [how they use it](https://twitter.com/benln/status/1983960258809831530)). Meanwhile, users noticed Cursor’s Composer-1 occasionally “thinking” in Chinese, raising transparency questions about base-model provenance ([@auchenberg](https://twitter.com/auchenberg/status/1983901551048470974), [@simonw](https://twitter.com/simonw/status/1983912102457963005)). Cognition published “Computer Use” in public beta—Devin can now operate desktop/mobile tooling, sharing screen recordings and building GUI apps ([@cognition](https://twitter.com/cognition/status/1983983151157563762)).
- **Tool-use evals and orchestration**: HKUST’s Toolathlon (Tool Decathlon) introduces an execution-based benchmark across **32 applications/600+ tools**, revealing current SOTA performance is low (e.g., Claude Sonnet 4.5 at **38.6%** success) and an open/proprietary gap persists ([@junxian_he](https://twitter.com/junxian_he/status/1983834164727312391)). New planning work spans parallel tool use with RL-based scheduling ([Graph-based Agent Planning](https://twitter.com/omarsar0/status/1983892163990843692), [paper](https://twitter.com/omarsar0/status/1983892176892522642)). LangGraph added Overwrite to bypass reducers for direct state replacement ([@caspar_br](https://twitter.com/caspar_br/status/1983949095837519901)). LangChain published a no-code Agent Builder roundtable ([@LangChainAI](https://twitter.com/LangChainAI/status/1983916519513059728)).
- **Real-time context pipelines**: Event-driven “streaming agents” moved closer to production with **Confluent + Weaviate** examples and **Confluent + Qdrant** partnerships for live data + vector search, enabling context-aware agents beyond stale RAG snapshots ([Weaviate](https://twitter.com/weaviate_io/status/1983921589163835398), [Qdrant](https://twitter.com/qdrant_engine/status/1983843826436395090)).

**Training, evaluation, and embeddings**

- **Hugging Face’s Smol Training Playbook (200+ pages)**: A distilled “field guide” from the HF science teams covering pre-training data curation, architecture choices, post-training (SFT/RL), and infra debugging (NCCL purgatory included). Strong emphasis on ablations and the messy realities papers skip, complementing the earlier FineWeb and Ultrascale playbooks. Links: [@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1983929546014433385), [@_lewtun](https://twitter.com/_lewtun/status/1983929588909797414), [@eliebakouch](https://twitter.com/eliebakouch/status/1983930328751153159).
- **Enterprise-grounded embedding evals**: Voyage’s quantization-aware trained embedding model, **voyage-3-large**, topped the new HF RTEB leaderboard, ranking first across **33 datasets** and outperforming OpenAI/Cohere on application-centric (finance/law/healthcare) retrieval tasks. QAT lets the model remain accurate under INT8/binary, lowering vector DB costs and enabling faster inference ([@_avichawla](https://twitter.com/_avichawla/status/1983783708047093838)).
- **Open vs. closed gap narrows**: Epoch AI’s ECI suggests the open-weight lag to closed SOTA averages ~**3.5 months** (≈**7 ECI points**, similar to “pro” vs “mini” deltas), indicating faster catch-up than assumed ([@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1983987212183335097)).
- **Late-interaction retrieval infra**: LightOn’s Fast-Plaid 1.2.5 brings speed and lower GPU memory for ColPali/ColQwen/PyLate-style retrieval ([@raphaelsrty](https://twitter.com/raphaelsrty/status/1983906400725024931)).

**Multimodal: speech, video, and image editing**

- **SSM-based speech at scale**: Cartesia’s new flagship TTS, **Sonic-3**, uses a State Space Model architecture to deliver low-latency streaming speech with prosody elements (e.g., laughter, surprise). It supports **42 languages** (including 9 Indian languages) and is now in the Artificial Analysis arena for blind evaluation ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1983879759194157194)).
- **Physics-aware editing and world simulation**: NVIDIA’s **ChronoEdit-14B** (open-source code, models, and demo) performs image editing in ~8 diffusion steps (~4s/image on H100) via a “video reasoning” stage + in-context editing of trajectory tokens—also useful for visualizing edit “thought processes” ([paper/model/demo](https://twitter.com/_akhaliq/status/1983953896415604836), [author update](https://twitter.com/jayzhangjiewu/status/1983963044695740848)).
- **Video generation updates**: Google’s **Veo 3.1** improves substantially on image-to-video (Veo 3.1 Fast ranks #2 in AA’s I2V arena) though text-to-video quality hasn’t advanced over Veo 3; pricing remains at $0.2/s without audio ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1983938159839998249)).

**Product and infra updates**

- **Perplexity** launched “Patents”—a citation-first patent research agent—free in beta, alongside “Discover” and new finance features like politician holdings tracking ([Patents launch](https://twitter.com/perplexity_ai/status/1983875975877423277), [Discover](https://twitter.com/AravSrinivas/status/1983960821731619025), [Finance](https://twitter.com/AravSrinivas/status/1983998749929259378)).
- **VS Code + OpenAI**: VS Code Insiders adds “Plan” (task analysis and implementation plans) and integrates OpenAI Codex for Copilot Pro+ accounts. OpenAI also introduced **Codex credits** to burst beyond plan limits ([Plan](https://twitter.com/code/status/1983942033879257195), [Codex](https://twitter.com/code/status/1983973969335378241), [credits](https://twitter.com/OpenAIDevs/status/1983956896852988014)).
- **Sora monetization**: Users can now buy extra generations; a broader “Sora economy” with rightsholder-paid cameos is planned, and free tier reductions are likely over time due to GPU constraints ([@billpeeb](https://twitter.com/billpeeb/status/1984011952155455596)).
- **Infra and platforms**: marimo is joining CoreWeave to scale molab while doubling down on open-source notebooks ([@marimo_io](https://twitter.com/marimo_io/status/1983916371869364622)); Locally AI launched a native Mac app built on MLX ([@LocallyAIApp](https://twitter.com/LocallyAIApp/status/1983957683737915405)); Baseten Training GA brings on-demand multi-node training with cache-aware scheduling ([@basetenco](https://twitter.com/basetenco/status/1983958807353934180)); SGLang-JAX now supports TPUs with SkyPilot one-liners ([@skypilot_org](https://twitter.com/skypilot_org/status/1983957542863851899)); and a hands-on DGX Spark review highlights its sweet spot for CUDA prototyping and small-scale inference vs. H100s ([@rasbt](https://twitter.com/rasbt/status/1983895811915214996)).

**Top tweets (by engagement)**

- [@sundarpichai](https://twitter.com/sundarpichai/status/1983922303424471541): Google x Jio partnership—rolling out Google AI Pro plans at no extra cost to eligible Jio users across India (Gemini 2.5 Pro, 2TB storage, creation tools).
- [@sama](https://twitter.com/sama/status/1983941806393024762): On motivations, equity, and working on AGI—widely discussed personal note from OpenAI’s CEO.
- [@OpenAI](https://twitter.com/OpenAI/status/1983956431360659467): Aardvark—OpenAI’s GPT-5–powered agent that finds and fixes security bugs (private beta).
- [@sama](https://twitter.com/sama/status/1984025727763935585): “GPT-6 will be renamed GPT-6-7” (levity amid a heavy news cycle).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Hugging Face Training Insights

- [**200+ pages of Hugging Face secrets on how to train an LLM**](https://www.reddit.com/r/LocalLLaMA/comments/1ok3xie/200_pages_of_hugging_face_secrets_on_how_to_train/) (Activity: 1047): **Hugging Face has released a comprehensive 200+ page guide titled "The Smol Training Playbook," which details the entire pipeline for training large language models (LLMs), including pre-training, post-training, and infrastructure. The guide is designed to share insights on what strategies have been effective and which have not, aiming to help practitioners build reliable LLMs. The playbook is available on [Hugging Face's platform](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) and is structured to cover model architecture, data curation, and infrastructure considerations, providing a detailed roadmap for LLM development.** The community response is positive, with users expressing gratitude and appreciation for the detailed insights into parallelism and advanced training techniques provided by Hugging Face. There is a request for better accessibility for mobile users, indicating a desire for more user-friendly formats.
    - The Hugging Face playbook is praised for its comprehensive coverage of parallelism and advanced training techniques, making it a valuable resource for those looking to understand large-scale model training. It is described as a 'one source destination' for these topics, indicating its depth and breadth in addressing complex training scenarios.
    - A technical issue is highlighted with the build process of the Hugging Face playbook, where a 'cache miss' error occurs during several steps, such as running npm build and setting up the environment. This suggests potential areas for optimization or troubleshooting in the deployment pipeline, particularly around caching strategies and build configurations.
    - The playbook is available online, but there is interest in a physical paperback version, indicating a demand for more accessible formats. This could suggest a broader audience who prefer traditional reading methods or require offline access for in-depth study.

### 2. Open Source AI Music Generation Advocacy

- [**Udio just robbed and betrayed its paying subscribers... Another reason why we need more Open Source**](https://www.reddit.com/r/LocalLLaMA/comments/1ojqvwe/udio_just_robbed_and_betrayed_its_paying/) (Activity: 553): **A Reddit user reports that Udio, a music creation platform, has removed the ability for subscribers to download their songs as** `.wav` **files without prior notice, sparking frustration among users. This change has led to concerns about anti-consumer practices, particularly from North American companies, and has fueled interest in supporting open-source alternatives for AI music generation. The user expresses willingness to financially support open-source developers in this field.** Commenters suggest that this move could be detrimental to Udio's user base, as the inability to download creations undermines the platform's utility. There is speculation that **Universal Music Group** might be influencing these changes to suppress AI music generation, potentially to protect traditional music industry interests.
    - A user speculates that Universal may have deliberately sabotaged Udio to suppress AI music generation. They suggest that Universal's public statements about a 'new era' and 'historic partnership' are misleading, as the real intention might be to eliminate competition from AI-driven platforms like Udio.
    - Another commenter, identifying as a data scientist, mentions the potential to train their own music model in regions with less stringent intellectual property laws. This highlights the growing interest in developing independent AI music models, especially in areas where legal restrictions are less of a concern.
    - A suggestion is made for Udio to release their model weights on platforms like Hugging Face if their site goes down. This would allow the community to clone and continue developing the model, ensuring that the technology remains accessible and can be further improved by open-source contributors.

### 3. Qwen 3 VL and Kimi Linear Model Updates

- [**Qwen 3 VL merged into llama.cpp!**](https://www.reddit.com/r/LocalLLaMA/comments/1ok2lht/qwen_3_vl_merged_into_llamacpp/) (Activity: 347): **The** `Qwen 3 VL` **model has been successfully integrated into the** `llama.cpp` **repository, as seen in [this pull request](https://github.com/ggml-org/llama.cpp/pull/16780). This integration allows for enhanced performance, with users noting improvements in benchmarks such as** `AIME25` **for the** `30b` **model and overall enhancements in the** `32b` **model compared to the** `30b 2507`**. Users are advised to run the model at a lower temperature than suggested in Qwen's model card for optimal text use cases.** There is anticipation for the release of `GGUF` and `unsloth quants`, with users expressing satisfaction with the performance of the `Qwen3-VL-32B Q6` model in initial tests.
    - ForsookComparison mentions creating their own version of the Qwen3-VL-32B Q6 model and notes that it performs well in initial tests. They suggest running it at a lower temperature than recommended in the model card for better text use case results, indicating a potential optimization for specific applications.
    - YearZero provides a comparative analysis of text benchmarks, highlighting that the Qwen 3 VL 32B model shows significant improvements over the 30B model, particularly in the AIME25 benchmark. This suggests that the newer model offers enhanced performance across various metrics, as evidenced by the linked benchmark results.
    - Arkonias humorously notes the expected delay in support for the model in LM Studio, implying that while the model is available, integration into popular tools may take additional time. This highlights a common issue in the deployment of new models where software support lags behind model releases.
- [**Kimi Linear released**](https://www.reddit.com/r/LocalLLaMA/comments/1ojz8pz/kimi_linear_released/) (Activity: 295): **Kimi Linear 48B-A3B has been released by [Moonshot AI](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct), featuring a modified Gated DeltaNet architecture. Despite a lower benchmark score compared to Qwen3-30B-AB3, it impressively used** `25 times less tokens` **for training. The model is expected to support very large context sizes, potentially up to** `1M`**, and is anticipated to be tested with AWQ quantization on** `2x3090` **GPUs to evaluate its performance in vllm.** There is anticipation for the Qwen Next architecture implementation in llama.cpp to support this model. The combination of MLA and Linear is praised, and there's excitement about the model's potential personality, similar to Kimi K2.
    - AlbeHxT9 mentions that Kimi Linear is based on a Modified Gated DeltaNet architecture. They note that for `llama.cpp`, the implementation of the Qwen Next architecture is necessary before Kimi Linear can be utilized, indicating a dependency on future architectural updates.
    - Marcuss2 highlights that Kimi Linear has a worse benchmark score compared to Qwen3-30B-AB3, but it used approximately 25 times fewer tokens for training. This suggests a highly efficient training process, which is impressive given the reduced data usage.
    - rekriux discusses the potential of Kimi-Linear 48B-A3B to support very large context sizes, noting the combination of MLA + Linear as beneficial. They express interest in testing the model with AWQ quantization on `vllm` using dual 3090 GPUs to explore its capability to handle a 1M context size.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Anthropic's Claude Skills and Introspective Awareness

- [**Anthropic has found evidence of "genuine introspective awareness" in LLMs**](https://www.reddit.com/r/OpenAI/comments/1ok0vo1/anthropic_has_found_evidence_of_genuine/) (Activity: 828): **Anthropic has published research suggesting that Large Language Models (LLMs) may exhibit "genuine introspective awareness" by detecting modifications in their own activation states, which are part of the internal processing rather than input or output text. This introspection is demonstrated by the model's ability to recognize and respond to changes in its neural activations, implying a form of self-awareness in processing. The research is detailed in their [introspection paper](https://www.anthropic.com/research/introspection).** Some commenters argue that the findings may simply reflect advanced pattern recognition rather than true introspection, as the models are trained on vast datasets linking similar concepts. Others express skepticism about the objectivity of the research, noting it is conducted by the company on its own product.
    - Andorion highlights that the 'injection' mentioned in the paper refers to modifications in the activations within the model's internal processing, not in the input text. This suggests that the model's ability to detect these changes indicates a form of introspective awareness, as it can recognize alterations in its own processing mechanisms.
    - SummerEchoes argues that the examples provided by Anthropic appear to be simple pattern recognition rather than genuine introspective awareness. The model's outputs are likely a result of its extensive training data linking similar concepts, which may not be as impressive as claimed.
    - thrownededawayed points out the philosophical and empirical challenges in defining introspective awareness, noting that even in humans, this question remains unanswered. The comment suggests that the human tendency to attribute introspective qualities to LLMs may stem from a deeper psychological need to find intellectual counterparts.
- [**10 Claude Skills that actually changed how I work (no fluff)**](https://www.reddit.com/r/ClaudeAI/comments/1ojuqhm/10_claude_skills_that_actually_changed_how_i_work/) (Activity: 576): **Claude Skills have introduced a range of functionalities that enhance productivity by integrating with various applications and automating workflows. Notable skills include the Rube MCP Connector, which allows integration with over** `500 apps` **through a single server, and Superpowers, a developer toolkit that transforms Claude into a comprehensive dev workflow with commands like** `/brainstorm`**,** `/write-plan`**, and** `/execute-plan`**. The Document Suite enhances Claude's capabilities with Word, Excel, PowerPoint, and PDF, enabling not just reading but also creating documents with proper formatting and formulas. These skills are implemented as markdown files with YAML metadata, making them easy to create and token-efficient, working across [Claude.ai](http://claude.ai/), Claude Code, and API. [Official Skills repo](https://github.com/anthropics/skills) and [Superpowers](https://github.com/obra/superpowers) are available for further exploration.**

### 2. Humorous AI and Technology Memes

- [**NEHAO**](https://www.reddit.com/r/singularity/comments/1ojx1hr/nehao/) (Activity: 2385): **The image is a meme featuring a robot holding a pillow, humorously juxtaposed with a text about a declined credit card payment. This reflects a playful take on the increasing presence of humanoid robots in everyday life, reminiscent of science fiction scenarios like those depicted in the movie "I, Robot." The humor lies in the absurdity of a robot enforcing payment compliance in a domestic setting, highlighting societal concerns and curiosities about the integration of robots into human environments.** Commenters humorously discuss the increasing realism of humanoid robots, drawing parallels to science fiction films like "I, Robot," and joking about the potential for robots to enforce mundane tasks like payment collection.
    - Alarm-Particular highlights a critical point about the current state of humanoid robots, noting that many demonstrations are not truly autonomous. Instead, these robots are often controlled by a pilot, which suggests that the technology is not yet advanced enough for independent operation. This raises concerns about the feasibility and honesty of claims made by companies seeking funding, as the technology required for a robot to autonomously perform household tasks is still underdeveloped.
- [**Why is CHATGPT calling me Batfucker????**](https://www.reddit.com/r/ChatGPT/comments/1ojx3iw/why_is_chatgpt_calling_me_batfucker/) (Activity: 722): **The image is a meme and does not contain any technical content. It humorously refers to someone as "Batfucker" and discusses a fictional scenario involving a change in calculation due to a "half price" offer. The playful use of language and emojis suggests it is intended for comedic effect rather than technical discussion. [View Image](https://i.redd.it/jdknlwrci8yf1.jpeg)** The comments suggest that the nickname "Batfucker" was likely prompted by the user's own input or actions, indicating a playful interaction with the AI rather than a spontaneous occurrence.
- [**Developer vs Vibe Coding**](https://www.reddit.com/r/OpenAI/comments/1ok34tz/developer_vs_vibe_coding/) (Activity: 978): **The image is a humorous meme comparing the work styles of a 'Developer' and a 'Vibe Coder' through a bar chart. It suggests that Developers allocate more time to Planning and User Acceptance Testing (UAT), while Vibe Coders spend more time on Development, Bugs, and redoing work (labeled as 'WTF' and 'FML re-do'). This reflects a stereotype that Developers are more structured, whereas Vibe Coders are more spontaneous and less organized. The chart is not based on actual data but rather plays on common perceptions of different coding styles.** Some commenters humorously identify with the 'Vibe Coder' label, while others argue that the chart oversimplifies and misrepresents the reality of software development, where all developers encounter bugs and need to redo work.
- [**what**](https://www.reddit.com/r/ChatGPT/comments/1okdtjh/what/) (Activity: 590): **The image is a meme featuring a humorous post by Sam Altman on [X.com](http://x.com/), jokingly announcing that GPT-6 will be renamed to GPT-6-7. This is a play on version naming conventions and does not reflect any real technical update or change in the GPT series. The post is dated October 30, 2025, and is intended for comedic effect rather than conveying any factual information about future AI developments.** The comments reflect a mix of humor and satire, with one user joking about the future of AGI and another suggesting an alternative humorous name, '6-9.' These comments highlight the playful nature of the post and the community's engagement with the joke.

### 3. Legal and Educational Challenges with AI

- [**This is the type of stuff that will stir up user experience again…**](https://www.reddit.com/r/OpenAI/comments/1ojloog/this_is_the_type_of_stuff_that_will_stir_up_user/) (Activity: 1123): **The image is a screenshot of a tweet discussing a legal ruling where a judge has allowed George R.R. Martin and other authors to sue OpenAI for copyright infringement. The lawsuit claims that ChatGPT generated content similar to "Game of Thrones," and OpenAI's motion to dismiss the case was denied. This case highlights ongoing tensions between AI-generated content and intellectual property rights, which could lead to restrictions on AI's ability to discuss or generate content related to major intellectual properties.** Commenters express frustration with George R.R. Martin's slow writing pace, humorously suggesting that AI could complete his work faster. There is also concern that such legal actions could hinder America's position in the AI market, potentially benefiting other countries like China.
    - QueryQueryConQuery humorously suggests that OpenAI's GPT models could potentially finish George R.R. Martin's long-awaited book, *The Winds of Winter*, faster than the author himself. This comment highlights the rapid development and capabilities of AI models like GPT-6, which are seen as efficient in generating large volumes of text quickly, contrasting with Martin's slow writing pace.
    - RealMelonBread raises a concern about the potential impact of legal actions against AI companies like OpenAI on America's competitive position in the global AI market. The commenter suggests that while the U.S. might face setbacks due to such legal challenges, countries like China could capitalize on these opportunities to advance their own AI technologies.
    - weespat clarifies a legal aspect of the ongoing lawsuit involving OpenAI, noting that the judge has not yet ruled on the "fair use" aspect of the case. The comment emphasizes that the continuation of the lawsuit does not imply a decision on the merits of the case, particularly regarding the use of AI-generated outputs.
- [**Current state of education**](https://www.reddit.com/r/OpenAI/comments/1ok1wek/current_state_of_education/) (Activity: 575): **The image is a meme expressing frustration with the current state of education, particularly the reliance on AI tools like ChatGPT for generating assignments. The post suggests that it would be more efficient if AI could directly provide assignments in PDF format, bypassing the need for students to manually edit AI-generated content to make it appear more human. This reflects a broader concern about the role of AI in education and the potential for it to handle more of the 'busywork' traditionally done by students.** Commenters express concern about the implications of AI in education, with one highlighting the need for exams that test critical thinking rather than rote memorization. Another commenter fears for the future of human intelligence and the value of education if AI continues to take over traditional learning tasks.
    - tendy_trux35 discusses a novel approach to testing in an evolutionary genetics class, highlighting the use of open-ended questions and open resources like notes, books, and the internet. This method emphasizes critical thinking and evidence gathering over rote memorization, suggesting a shift in educational assessment methods to better prepare students for real-world problem-solving.
    - Tigger3-groton contrasts high school and college education, noting that students who excel in rote memorization often struggle in college where critical thinking and innovation are required. They argue for the integration of AI as a learning tool, emphasizing that students must learn to exceed AI capabilities in productivity to remain relevant in the workforce, drawing parallels to historical technological advancements.
    - Jayfree138 criticizes the current educational system as outdated and likens it to 'industrial age education,' where financial investment does not equate to practical learning. They advocate for a reformation of educational practices to focus on meaningful learning experiences that prepare students for real-world challenges, rather than perpetuating a 'pay to play' model.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. Agentic Coding & Reality-Check Benchmarks**

- **SWE-1.5 Sprints: Cerebras + SpecDec Smash Speed Records**: [Cognition announced **SWE-1.5**](https://x.com/cognition/status/1983662836896448756), a Windsurf agentic coding model hitting up to **950 tok/s** using **Cerebras** hardware, **speculative decoding**, and a custom **priority-queue**; it benchmarks as ~**6×** faster than Haiku and **13×** faster than Sonnet while maintaining near-SOTA quality.
    - Community reactions in Windsurf and Latent Space called it *"near-SOTA coding performance"* and highlighted speed-first engineering wins, while probing how much of the boost is from systems tuning versus pure model gains.
- **RLI Reality Bites: Manus Tops Out at 2.5%**: ScaleAI’s [**Remote Labor Index**](https://scale.com/leaderboard/rli) tested agents on tasks averaging **30 hours** of human effort and found the top agent (**Manus**) achieved just **2.5% automation**, with most failures due to quality and completeness.
    - Engineers noted this is *"highly valuable if harnessed via the right UI for human-AI collaboration, but totally useless as a labor replacement"*, spurring calls to focus on workflow integration and error recovery over leaderboard bragging rights.
- **Manus Mystique: Niche Wins, Narrow Generalization**: Eleuther researchers debated **Manus**’s under-discussed performance, noting agent success rates hover around **1–3%**, which may reflect in-distribution narrowness rather than broad agentic competence (see the RLI context: [ScaleAI RLI](https://scale.com/leaderboard/rli)).
    - One blunt take captured sentiment: *"1-2% isn't enough for anyone to actually use an agent rn"*, prompting questions about whether domain-specialized agents that excel (e.g., visualization) meaningfully beat evenly-distributed generalists.

**2. New Multimodal Models, Leaderboards & Gateways**

- **MiniMax Melodies & Mouths: Speech 2.6 + Music 2.0**: Hailuo AI rolled out [**MiniMax Speech 2.6**](https://x.com/Hailuo_AI/status/1983557055819768108) with **<250 ms** real-time latency and **voice cloning**, and debuted [**MiniMax Music 2.0**](https://x.com/Hailuo_AI/status/1983964920493568296) for **5-minute** pro-grade songs with lifelike vocals and multi-instrument control.
    - Creators pushed for an OpenAI-style API, more language support (e.g., Malayalam), a **voice changer**, synchronized video, and even open-sourcing, signaling strong demand for production-ready tooling and transparency.
- **Hailuo Hustles to #7: Video Arena Shake-Up**: LMArena added image-to-video model [**hailuo-2.3-fast**](https://lmarena.ai/leaderboard/text-to-video), which immediately landed at **#7** on the Text-to-Video leaderboard.
    - Staff nudged users to try the model and report results, while the separate Text leaderboard being stuck on Oct 16 reminded folks that infra freshness matters as much as model freshness.
- **Sonar Pro Seeks Truth: OpenRouter-Exclusive Pro Search**: OpenRouter launched an exclusive **Perplexity Sonar Pro (Pro Search)** mode at [openrouter.ai/perplexity/sonar-pro-search](https://openrouter.ai/perplexity/sonar-pro-search), touting **multi-step agentic reasoning**, **dynamic tool execution**, **real-time thought streaming**, and **adaptive research strategies**; see the announcement on [X](https://x.com/OpenRouterAI/status/1984032292436898264).
    - Engineers framed it as a route to deeper, verifiable answers when needed—and quick responses when not—making it a pragmatic gateway for research-heavy chats.

**3. GPU Kernel Craft: Scans, Samples, and Small Floats**

- **Scan Slam: Beating CUB (Sometimes) and Taming Thrust**: A CUDA dev reported custom `single_pass.bin` scans competing with **CUB** `DeviceScan`, cross-checking bandwidth targets from a GTC talk ([YouTube](https://youtu.be/VLdm3bV4bKo?t=2327)) and debugging **Thrust** benchmarks by swapping in [custom allocators](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu).
    - They cautioned that hidden temporary allocations can sink benchmarks, and suggested standardizing policies and scratch paths before declaring speed crowns.
- **FP8 FTW: TorchAO + GemLite Go Low-Bit**: Practitioners used TorchAO’s `quantize_` with **Float8 configs** (see the AO llama example: [generate.py](http://generate.py/)) and shared a comprehensive **quantization survey** with RTX 4090 results ([benchmark repo + video](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main#benchmarking-results-on-1-rtx-4090)).
    - Several combined **TorchAO** with **GemLite** (weights-only vs activations+weights) and discussed when to keep some layers in **BF16**, trading off throughput, stability, and kernel availability.
- **Top‑K Tactics: Radix Rules When K≪N**: For massive sequences (4K–128K), engineers revisited a hardware-friendly **Top‑K**: tile-and-merge versus **radix-based** selection when K≪N, referencing the FlashInfer blog on sampling ([FlashInfer Top‑K discussion](https://flashinfer.ai/2025/03/10/sampling.html)).
    - They also flagged an incoming **TopK** in NVIDIA **CCCL/CUB** ([PR #5677](https://github.com/NVIDIA/cccl/pull/5677)), noting PyTorch heuristics that switch between radix and full sort remain a practical baseline.

**4. Long-Context Engineering: Kimi’s Linear Attention Push**

- **Linear Lift-Off: Kimi’s Context Costs Go Linear**: MoonshotAI published the **Kimi Linear Attention** technical report, cutting quadratic attention to linear to scale context windows efficiently ([tech report PDF](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf)).
    - Nous Research channels emphasized applications like very long document workflows and summarization, where IO-bound tasks amplify the value of linear-time transforms.
- **48B Goes Live: Kimi-Linear-48B-A3B-Base Drops**: MoonshotAI released [**Kimi-Linear-48B-A3B-Base**](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) on Hugging Face, continuing their march on long-context LLMs.
    - Practitioners compared attention variants, with one noting *"*`Kimi Delta Attention` *reminds me of qwen3 next gated deltanet"*, nudging cross-family architectural analysis.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord Moderators Need Better Training**: Members debated the skills of new Discord moderators, pointing out that *they don't know how to moderate the server*.
   - One member suggested that *new gens are best for moderation tbh if their behavior is good enough*.
- **Comet Referral Program Turns Sour**: Users are complaining about the Comet referral program changing its rules after they promoted it, making it impossible to reach the 30 day hold, per the [new ToS](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs) that requires a **Pro/Max subscription** to refer someone.
   - Several users reported being scammed, with one user stating they had *$1400 washed away just like that*.
- **Indian Jio Subscribers Snag Gemini Pro for Free**: Indian members reported that Indian Jio users get [Gemini AI Pro for 1.5 years for free](https://www.jio.com/google-gemini-offer/).
   - This offer seems to be limited to Indian Jio subscribers only.
- **Claude Icon Goes Missing**: Users reported that when using **Claude 4.5 Sonnet**, the icon was missing from the replies.
   - This issue appears to be purely cosmetic.
- **Sonar Reasoning API Can't Fetch Live Data**: A user reported issues with **Sonar Reasoning's API** failing to obtain and deliver live data, such as stats and stock prices.
   - Another user suggested this is because the instance isn't connected to a live data source or **web search module** and advised configuring settings within the **Perplexity** or **Sonar Reasoning API** setup to link it to live data sources or enable external search.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **MiniMax Competes as Cheaper AI**: Members discussed that [**MiniMax**](https://minimax.chat/) offers a cheaper, competitive alternative in the **AI** space, though it may not have the top-notch quality of more expensive options.
   - Some users pointed out that not everyone can afford top-tier tools and need to budget, noting that **ReCaptcha** is not that expensive.
- **LMArena Plagued by ReCaptcha**: Users reported experiencing [frequent **ReCaptcha** prompts](https://www.google.com/recaptcha/about/) on **LM Arena**, with some encountering infinite loops, making the platform difficult to use.
   - A staff member acknowledged the issue and mentioned that they are looking into ways to fix the captcha and improve the user experience, though it may take time.
- **Video Tool Seeks Gemini-Keyed Beta Testers**: A member is seeking beta testers for a [prompt generation app](https://www.testflight.apple.com/join/0S4L0lB4) for video models, requiring a **Gemini API** key to participate.
   - The tool aims to help users rephrase prompts and avoid flagged tokens.
- **Hailuo-2.3 debuts in LMArena Video Arena**: The **LMArena Video Arena** added a new image-to-video model, [hailuo-2.3-fast](https://lmarena.ai/leaderboard/text-to-video), to its leaderboard, and it ranked #7 in **Text-to-Video**.
   - Members are encouraged to try out the new model and share their thoughts in the designated channel.
- **LMArena Leaderboards Frozen in Time**: Members reported that the [Text Leaderboard](https://lmarena.ai/leaderboard/text) is stuck on **October 16**, with no updates.
   - A staff member confirmed that the leaderboards haven't been updated recently, and the team is aware of the issue.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Composer Model Sparks Debate**: Users are in disagreement about **Composer 1** on Cursor, with some praising it as ideal for implementing planned tasks due to its speed, while others find **Sonnet 4.5** superior in speed and accuracy.
   - Some members suggest using **Auto** for planning and **Composer** for execution, noting **Composer**'s speed is close to **4.5 thinking**.
- **Claude Code Causes Commotion**: Members debated the value of **Claude Code**, with some arguing it offered better limits and cost-effectiveness compared to Cursor, while others emphasized Cursor's richer feature set.
   - Some see **Claude Code** as a native model provider, highlighting the necessity of custom configurations for good results, such as hooks, MCP servers, and memory.
- **Pricing and Usage Limits Provoke Protests**: Users report wildly varying experiences with Cursor's pricing and usage limits, with many feeling overcharged due to high cache usage, while others suggest **Claude Code's** pricing is better.
   - Suggestions include a hybrid approach of using **Claude Max** alongside **Cursor Pro** for optimal value, as well as implementing cost controls, monitoring dashboards, and spending caps.
- **Cursor 2.0 Brings Bug Bonanza**: Users are experiencing both excitement and frustration with Cursor 2.0, citing new features alongside bugs like file attachment issues, horizontal scroll bar glitches, context loss, and the removal of pills.
   - Some users are also reporting *chinese typos* injected into the output, along with issues in tab navigation, hotkey changes, cache usage concerns, and doubts about the efficacy of new agent review features.
- **Tab Complete Triumphs**: Members widely praised Cursor's **tab complete** feature for its efficiency and multi-line editing capabilities, claiming it outperforms **GitHub Copilot**.
   - One user emphasized that *this workflow is insane if you're working on large projects and want to stay in touch with the code and actually understand every bit*.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Turing RTX 8000: Budget VRAM Kings?**: **RTX 8000 Turing cards** are available for around **$2k** on eBay, offering **48GB of VRAM**, making them suitable for server use, but they are older cards.
   - A member advised caution due to concerns about AI support and potential e-waste, noting newer cards have superior architecture.
- **Qwen3 Runs into OOM**: A user reported getting **OutOfMemoryError** while running the **Qwen3 4B GRPO notebook** even with **48GB VRAM**.
   - Suggestions included ensuring **4-bit loading** is enabled, and adjusting the **per-device batch count** to alleviate memory issues.
- **Grokipedia Emerges, Musk's Encyclopedia**: Elon Musk launched "**Grokipedia**", an AI-generated encyclopedia boasting over **800k articles**.
   - This was discussed in the **off-topic** channel.
- **GEMMA-3: Horrors Unleashed via Unsloth**: A member announced a new **Gemma 3 model** trained via **Unsloth**, pushed to the limit, with downloads available at [Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF](https://huggingface.co/DavidAU/Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF).
   - A *full set of GEMMA-3 horrors* were trained: **1B**, **4B**, two **12Bs** and **27B**.
- **Anthropic's Introspection Research Sparks Awe**: A member shared [Anthropic's research on introspection](https://www.anthropic.com/research/introspection), highlighting a model's ability to detect injected concepts within its hidden layers.
   - A user stated, *"This blows my mind"* regarding the model's self-awareness and ability to detect tampering.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Launches Perplexity's Sonar Pro Search**: **OpenRouter** partnered with **Perplexity** to release an **OpenRouter-exclusive** version of **Sonar Pro** with **Pro Search** enabled [here](https://openrouter.ai/perplexity/sonar-pro-search), highlighting features like **multi-step agentic reasoning**.
   - This new mode's features include **dynamic tool execution**, **real-time thought streaming**, and **adaptive research strategies**, further discussed on [Twitter](https://x.com/OpenRouterAI/status/1984032292436898264).
- **OpenRouter Typescript SDK Inspires Dumb Demo App**: A member deployed a *dumb demo app, barely modified from the original* [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk/tree/main/examples/nextjs-example) to test **API endpoints** with **environment variables** and implement missing **OAuth** stuff.
   - The developer clarified that *they absolutely do not want to make this a serious thing, it is just for inspiration and proof-of-concept.*
- **Sora 2 Struggles with Catgirl Generation**: A user reported frustration with **Sora 2** consistently generating images of catgirls with human ears and disproportionately large chests, even when prompted to make it more *"cute".*
   - The issue was linked to potential biases in the training data, leading to the model's tendency to sexualize the character.
- **Ultra Model Unstable due to DeepInfra Errors**: A member found the **Ultra** model very unstable, mentioning that its inference conditions were changing due to a switch from **DeepInfra** to **Z.AI**.
   - The issue resolved when it switched from **DeepInfra** to **Z.AI**.
- **Embedding Models Added for Testing**: The addition of embedding models is being tested, specifically [OpenAI's text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small).
   - A member noted they were getting a bunch of random data back but that *not using raw response seems to fix it.*



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Job Applicant Acknowledges Long Odds**: A member applied for a job at Hugging Face, acknowledging the odds are low due to the **hundreds of thousands of applicants** they supposedly receive, and noting AI engineering experience, *just not under the explicit exclusive title of ml engineer*.
   - No additional details were given.
- **Qwen Omni Pipeline Offers Fast Latency**: A member reported their **realtime Qwen Omni pipeline** has *super low latency and fast speech out* and asked if there's interest in open-sourcing it.
   - While the pipeline is written in Python, one user sarcastically remarked, *Then I don't believe you*, referencing the common sentiment that **Python speed often relies on C libraries**.
- **SecureFix Emerges From RAG**: A member ported a **RAG system** into a **CLI Python code remediation tool** called *securefix*, available on [GitHub](https://github.com/HakAl/securefix).
   - It uses **Bandit** to scan files/directories, optionally sends requirements to **OSV** scan, and the RAG system provides remediation.
- **InstantID Tactics Improve Preservation**: Members discussed using **InstantID + IP-Adapter FaceID** or a **ControlNet reference-only setup** to better preserve identity in generated images.
   - The approaches are aimed at improving identity retention compared to standard methods.
- **SFT Course Results in Repetition**: A member attempted the SFT (**Supervised Fine-Tuning**) part of the course, but the training run resulted in the model repeating *"system"* over and over again and uses almost **100GB** of disk space.
   - They plan to prepare the dataset without GPU in the future, and are concerned about memory usage, wondering if it's possible to run the training on a **32 GB** card.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Mulls Bindings for Vulkan/WGPU**: The community debated implementing bindings for **Vulkan** or **WGPU** in Mojo, considering type conversion functions but ultimately deciding it was premature due to ongoing language changes.
   - The consensus was that, due to the rapid evolution of the language, focusing on these bindings now might lead to unnecessary complications and rework later.
- **MAX Mashes NVIDIA and AMD**: For **ML**, **MAX** demonstrates performance parity with top-tier **NVIDIA** offerings and surpasses **AMD** solutions on **DC hardware**.
   - Early training experiments have exceeded expectations, outperforming **JAX** on **MNIST** benchmarks, marking a significant milestone for Mojo.
- **Mojo Models Scikit-learn Competitor**: A **scikit-learn** alternative is under construction for **Mojo**, with preliminary benchmarks indicating accelerated performance capabilities, as showcased [here](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11).
   - The new library promises to leverage Mojo's speed to provide efficient machine learning tools.
- **Pandas Paramour: Mojo Pursues Polars-esque Partner**: Instead of a direct **Pandas** counterpart, Mojo is leaning towards a **Polars**-inspired implementation to take advantage of **GPUs** through **MAX**.
   - This approach would allow Mojo to better utilize hardware acceleration for data manipulation tasks.
- **Mojo Merges Async and IO**: Mojo intends to integrate **async** and **IO** features to tackle scalability issues prevalent in other languages, potentially adopting an effect system akin to Rust's methodology.
   - This integration aims to provide a more seamless and efficient approach to handling asynchronous operations and input/output processes.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen3-NEXT Support Still a Question**: Regular **Qwen3 models (not NEXT or VL)** have been supported for a while, but members shared that support for **Qwen3-NEXT and Qwen3-VL** will be indicated by a runtime update.
   - Currently, **Qwen3-VL** is only supported on **MLX (Mac)**.
- **Debugging MCP Image Integration in LM Studio**: A user is debugging **MCP** image support in **LM Studio** using a custom MCP server called **GUIDANT** with `qwen/qwen3-vl-4b`, noting successful tool execution but no image processing.
   - The user asked the question, *Does LM Studio currently support returning images through MCP tool responses?*.
- **Arabic Alignment Awaits Assistance**: A user reported a problem with mixed **Arabic and English** text arrangement in **LM Studio** due to **Arabic's right-to-left** writing direction.
   - Members mentioned that Arabic right to left isn’t supported fully in the UI yet.
- **Speed Depends on Active Parameters**: Model speed is overwhelmingly based on the number of **ACTIVE PARAMETERS** per token, presuming sufficient 'fast ram', with speed dependent on **GB pushed through**, but **Mixture of Experts (MoE)** cuts the GB used per token.
   - As one member put it, a *30b model thats 30gb, but only activates 3b... activates 3gb so its about as fast as a 3GB big dense model*.
- **Orange Pi 6 Could Be a Viable Option**: The **Orange Pi 6 Plus** ([http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)) features a 12-core ARM CPU, up to 64GB LPDDR5 RAM, an NPU with up to 45 TOPS AI performance, M.2 expansion, and dual 5Gbps ethernet.
   - The system could be a super cheap way to run Qwen3 30b models.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Tokenizer Efficiency Faces Scrutiny**: Members benchmarked the efficiency and accuracy of different tokenizers in encoding and decoding, citing an example link to a **Rust-specific benchmarking tool** within the **Hugging Face tokenizers** library.
   - They want a repository or tool similar to the one available in the **Hugging Face tokenizers** library that supports benchmarking across different tokenizer implementations.
- **Triton Navigates to OpenCL Shores**: A member shared a project that translates **Triton** code to **OpenCL** using *mlir-translate*, showcasing [the project](https://github.com/toyaix/triton-oclmatt.pd).
   - The project demonstrates the feasibility of generating **OpenCL** code from **Triton**, potentially broadening Triton's hardware support.
- **CUB Scanned, Thrust Thrust Aside**: A member benchmarked their scan implementations against **CUB's `DeviceScan`** and found their implementation (`single_pass.bin`) performed competitively.
   - They also speculated that **Thrust** benchmarking might be inaccurate due to bundled allocations, suggesting the use of [custom allocators](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu) to address **Thrust's** allocation issues.
- **Flame Throwers Look at Float8**: Members discussed using the `quantize_` function with the **Float8 config** for inference with TorchAO as linked in the [ao repo](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py) with *torchao* and *gemlite* for a similar implementation.
   - A member shared their experience using *torchao* and *gemlite* for a similar implementation, along with a [survey paper and video](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090) covering quantization and modern formats like *mxfp4*.
- **Sum Reduction Sketched**: A member wrote their first technical blog about sum reduction with [part 1](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) covering the introduction and [part 2](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) covering the implementation, while looking at **PTX/SASS** and attempting to use **Nsight**.
   - The author is planning to do **matmul** posts next and is open to feedback.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Extropic's Hardware: Niche, but Neat**: Discussions indicate [Extropic AI's hardware accelerator](https://fxtwitter.com/extropic_ai/status/1983579587649904960?s=46) is likely real but suited for niche applications due to its design as a **probabilistic hardware accelerator**.
   - The consensus is it is an **ASIC** rather than an **FPGA**, exciting some members.
- **Researchers Face Khowar Translation Hurdles**: One member is tackling machine translation for the **low-resource Khowar language** and is facing severe **data scarcity**, resorting to scanning physical books to create a dataset.
   - The primary issue is the **text extractor** fails to recognize certain characters unique to Khowar's **Perso-Arabic script**, leading to character replacements and glyph misalignments, they are asking for help.
- **Daily Paper Dumps Proposed**: A community member proposed posting a **daily dump of papers**, sorted by importance, along with a few more significant papers in separate posts.
   - Other members agree to prioritizing importance over quantity, and pointed to **Elvis Saravia** ([nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai)) and **ByCloud** ([https://x.com/TheAITimeline](https://x.com/TheAITimeline)) as inspirational resources for **weekly AI paper recaps**.
- **Universities Caught Between Mandates and MBA**: Discussion revolves around universities grappling with state mandates versus business realities, specifically the **ban on ChatGPT** juxtaposed with a failure to train students for non-von Neumann architectures, as covered by [Nature](https://www.nature.com/articles/s41928-025-01488-x).
   - One member humorously analogized the MBA mindset to designing products with *optimal* planned obsolescence for repeat purchases.
- **Is Anthropic's LLM a Fraud?**: A member suggests *Anthropic's whole thing from Day 1 has been fraud* and predicts this pattern will continue in future publications.
   - The member did not give specific justification, just general disdain and the message was posted in **ml-news** channel.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pandas Plays with DSPy?**: A member is developing a **scikit-learn style API** for **DSPy**, featuring *fit/transform/predict* methods, aiming to integrate with *classical* **pandas/polars dataframes**.
   - Another member suggested exploring an alternative project known as **semantic dataframes** to enhance the integration.
- **ReAct's Finish Function Flounders**: Users are encountering issues with the **ReAct module** where the **LLM** erroneously calls the **finish()** function with arguments.
   - A solution proposed involves modifying the function signature to explicitly instruct the **LLM** to call **finish()** with NO ARGUMENTS when coding is complete: finish().
- **Pune Plans DSPy Pow-Wow**: The founder of **Unravel.tech** in Pune, India, an organization using **DSPy** to develop AI agents for enterprises, has expressed interest in organizing a **DSPy meetup** in Pune.
   - Interested parties were advised to connect via the <#1433127482676084746> or <#1211763460480827402> channels.
- **BAML Bests Burdensome JSON?**: A member shared insights on using **BAML Adapters** for Merger and Acquisition scenarios, favoring **BAMLAdapter** for structured outputs over **JSON schema**.
   - They illustrated how the **JSON schema** is reformulated into **BAML** format within the prompts generated by the adapter in **DSPy** ([see image](https://cdn.discordapp.com/attachments/1433555562116943933/1433556837990531092/json-schema-vs-baml.png?ex=69051f58&is=6903cdd8&hm=ccc16f7efaeeb0d86031217401084b0475b9c09eb0423bc7f5a5451e8933dd86)).
- **JSON Judged as Justly Jerky**: One member suggests that **LLMs** perform better when you _don't_ use **JSON schema**, because it is terrible from a semantic standpoint, verbose, and adds descriptors that are far apart in token space.
   - He recommends checking out his experiments and numbers in [this repo](https://github.com/prrao87/structured-outputs), where he states that JSON schema is objectively worse, and that DSPy's baseline prompts are so good that you'd never even need SAP to fix the outputs.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Extropic Debuts Thermodynamic BEFF Chip**: Extropic revealed their **thermodynamic intelligence hardware (XTR-0)**, called the *BEFF chip*, and announced in [this X post](https://x.com/Extropic_AI/status/1983579587649904960).
   - No further details were given.
- **ScaleAI's Remote Labor Index Reveals Grim Automation Rates**: ScaleAI launched a [new benchmark](https://scale.com/leaderboard/rli) measuring how well current Agents perform relative to **Upwork freelancers** for tasks requiring humans an average of **30 hours**.
   - The top agent (**Manus**) showed only a **2.5% automation rate**, failing primarily due to quality and completeness, leading to discussions on enhancing human-AI collaboration via better UI.
- **Cognition's SWE-1.5 Runs Fast with Cerebras**: Cognition introduced **SWE-1.5**, an agentic coding model on Windsurf operating at up to **950 tok/s**, using Cerebras hardware, speculative decoding, and a custom priority-queue, and shared in [this X post](https://xcancel.com/cognition/status/1983662836896448756).
   - It runs **6x faster** than Haiku and **13x faster** than Sonnet.
- **Codex Quality Plummets Amidst Increased Usage**: Jessie Frazelle reported that **Codex's** quality has declined severely from *god level* to *harmful* with increasing usage, detailed in [this X thread](https://xcancel.com/embirico/status/1983643336390144163?s=46).
   - Alexander Embiricos stated that the deterioration is being treated as a high-priority issue.
- **OpenAI Injects Codex Credits for Usage**: OpenAI is now offering **pay-as-you-go credits** (**$40 per 1,000 credits**) for additional Codex usage on ChatGPT Plus/Pro and has reset rate limits for all users, per [this tweet](https://xcancel.com/OpenAIDevs/status/1983956900602581254).
   - No further details were given.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Solo LLM Training Faces Compute Crunch**: Training frontier reasoning **LLMs** solo poses challenges due to high compute demands, prompting consideration of distillation or fine-tuning using tools like [Unsloth](https://github.com/unslothai/unsloth).
   - One member shared their [TRL implementation](https://github.com/torotoki/reasoning-minimal) of a reasoning model, calling it an interesting but manageable personal project.
- **Khowar OCR Frustrated by Glyph Goofs**: A member working on machine translation for **Khowar** is struggling to extract text from scanned books because existing OCR tools misinterpret unique Perso-Arabic glyphs, as existing tools misinterpret or distort glyphs, using **PyMuPDF (fitz)** and **MyPDF**.
   - Members suggested creating a dataset for fine-tuning a vision OCR model, by manually labeling glyphs, and pointed to a paper with tips for building models with limited data: [[2509.14786] Tricks of the Trade for Developing Robust Arabic OCR Systems](https://arxiv.org/abs/2509.14786).
- **Manus Agent's Performance Raises Eyebrows**: The **Manus** agent's solid performance is curiously underdiscussed, possibly because current agent success rates of **1-3%** may only measure in-distribution performance.
   - Members questioned if agents like **Manus**, which excel in specific areas like visualization, are truly superior to those with evenly distributed success rates, since *1-2% isn't enough for anyone to actually use an agent rn*.
- **Extropic's Hardware Spurs Examination**: Members scrutinized **Extropic's custom hardware**, designed for more efficient model execution through alternative primitives, rather than encoding graphs in vectors and matrices.
   - Alternatives such as **Groq** and **Cerebras** focus on inference efficiency through larger on-chip cache to avoid fetching from memory.
- **RWKV's Riddles Restrict Ramp-Up**: Adoption of **RWKV** is hindered by difficulties in understanding its mathematical formulation and unclear papers.
   - One member emphasized that *this is consistently rwkv's biggest issue*, expressing that it was the main reason their team didn't train an **RWKV** world model, despite wanting to.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 repo wants Hacktoberfest tag**: A member requested the addition of the **Hacktoberfest tag** to the **Kimi K2 repo** on HuggingFace.
   - The request aims to encourage contributions during the **Hacktoberfest** event.
- **Kimi-Linear-48B-A3B-Base goes live**: **Kimi-Linear-48B-A3B-Base** has been released on HuggingFace and is now live [here](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base).
   - This release marks another milestone in Moonshot AI's development of large language models.
- **Kimi Delta Attention evokes Qwen vibes**: A member noted that *`Kimi Delta Attention` reminds me of qwen3 next gated deltanet*.
   - The comment suggests similarities in the architectural design or functionality between **Kimi Delta Attention** and **Qwen3**'s gated deltanet.
- **Kimi-cli's D-Mail wins hearts**: Posts are circulating highlighting the growing popularity of **Kimi-cli's D-Mail**, with one example [here](https://x.com/steipete/status/1983713085019046322?s=46).
   - The positive reception indicates a growing interest in and adoption of **Kimi-cli's D-Mail**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Users Beg for Manus Credits**: Several members requested **Manus credits** and offered to pay for assistance with their projects.
   - Some users are inquiring about the availability of the **$99 credit package** and exploring alternatives like **Monica** to complete their school work.
- **Developer Available for Projects**: A member announced availability as a developer for potential projects within the community.
   - Another member inquired about the developer's **Manus credits** to assist with a project via direct message.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AI Neos to Duke it Out**: A member expressed anticipation for **wrestling/boxing matches** featuring **AI Neos**.
   - Another member echoed the sentiment, hoping that it happens *soon*.
- **HTB Announces AI Security CTF**: **Hack The Box (HTB)** is organizing an **MCP-only CTF** focused on **AI security** on November 20th and is seeking participants to test **pen testing agents** against realistic scenarios; registration is free [here](https://ctf.hackthebox.com/event/details/neurogrid-ctf-the-ultimate-ai-security-showdown-2712?utm_campaign=AI+CTF+-Oktopost&utm_content=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fupdate%2Furn%3Ali%3Ashare%3A7386416070783479808&utm_medium=social&utm_source=LinkedIn&utm_term=).
   - The CTF aims to simulate real-world situations in which AI security is paramount.
- **Windows Users Grapple with Local Model Training**: A user reported dependency issues while training models locally on Windows.
   - A fellow member advised switching to **Linux** or **WSL** to avoid the dependency management problems on Windows.
- **MoonshotAI Releases Kimi Linear Attention Report**: **MoonshotAI** has published [a technical report](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf) detailing their **Kimi Linear Attention** mechanism, a key method for scaling context windows.
   - It offers a way to extend context windows by reducing quadratic compute to linear, thus enabling better efficiency in processing very long sequences in applications like long form content generation and document summarization.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **George Hotz considers `ruff format`**: George Hotz mentioned that tinygrad may one day use [`ruff format`](https://github.com/astral-sh/ruff) for code formatting.
   - This could lead to greater code consistency, leveraging **ruff's** speed and modern tooling.
- **Nested GROUP_REDUCE Causes Headaches**: A member reported an error involving nested **GROUP_REDUCE** operations and requested assistance with debugging related to the new **rangeify** rewrites.
   - They were seeking a quick hint to avoid a *long methodical debugging* process if the cause was immediately obvious to someone familiar with the changes.
- **Seeking Hints on Rangeify Rewrites**: A member requested a hint regarding a potential cause for a **GROUP_REDUCE** error within another reduce, hoping to leverage insights from someone with expertise in the new **rangeify-related rewrites**.
   - The member aimed to avoid extensive debugging if a simple solution or known issue existed, expressing willingness to proceed with a deeper investigation otherwise.



---



## [Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **SWE-1.5 Lands on Windsurf**: A new fast agent model, **SWE-1.5**, was released and is now available in Windsurf, promising near-SOTA coding performance.
   - More details are available in the [official announcement](https://x.com/cognition/status/1983662836896448756).
- **SWE-1.5 Quick Coding Performance**: The fast agent model **SWE-1.5** sets a new standard for speed, while also offering near-SOTA coding performance.
   - The model is now available in **Windsurf**.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP RFC Stalls Without Code**: The [Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269) is experiencing delays because stakeholders are awaiting a tangible implementation to assess its impact.
   - Stakeholders emphasize that the RFC needs an actual implementation to properly evaluate its practical value, since *without an implementation, evaluating the RFC's effectiveness is difficult.*
- **Implementation Urgently Needed for Evaluation**: The primary concern is the absence of a working model that aligns with the [Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269), making it challenging to validate its real-world applicability.
   - Without a concrete model or implementation, stakeholders find it hard to proceed and effectively assess the RFC's contributions and overall utility.



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1433169564090503341)** (1089 messages🔥🔥🔥): 

> `Moderation problems, Comet Referal Promo issues, Perplexity.ai payouts, GPT Go subscriptions, Gemini Pro offer` 


- **Discord Mods need better training**: Members discussed about training new Discord moderators, one stating that *new gens are best for moderation tbh if their behavior is good enough* and another commenting *they don't know how to moderate the server*.
- **Perplexity referral program becomes a big scam**: Several users are complaining about the Comet referral program changing rules after people promoted them, making it impossible to reach the 30 day hold. [New ToS](https://www.perplexity.ai/help-center/en/articles/11385821-partner-promotions-and-referral-programs) require a **Pro/Max subscription** to refer someone.
- **Scammed referral promoters**: Some users report being scammed by the Comet referral program, with one stating that they had *$1400 washed away just like that*. Another mentions losing *$200*.
   - Several mention that human support refuses to respond to the referral claim and that *they got a Sam AI response*.
- **Indian Jio offers Gemini Pro for free**: Some Indian members shared that Indian Jio users get [Gemini AI Pro for 1.5 years for free](https://www.jio.com/google-gemini-offer/).
- **Claude icon missing**: Users are reporting that when using Claude 4.5 Sonnet thinking, that the icon was missing from the replies.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1433215678948835418)** (9 messages🔥): 

> `Sonar Reasoning API, Live Data, External Data Connectors, Web Search Module` 


- ****Sonar Reasoning API** struggles to fetch live data**: A user reported issues with **Sonar Reasoning's API** failing to obtain and deliver live data, such as stats and stock prices.
   - Another user suggested this is because the instance isn't connected to a live data source or **web search module**, and offered to privately guide the user through the setup.
- ****API setup** requires external data connectors**: A user was informed that their **Sonar Reasoning API** instance needed to be connected to a live data source or web search module to fetch real-time information.
   - They were also advised to configure settings within their **Perplexity** or **Sonar Reasoning API** setup to link it to live data sources or enable external search.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1433168861036806144)** (952 messages🔥🔥🔥): 

> `MiniMax Cheaper AI, ReCaptcha, Image Generation Limits, AI Alignment & Self Harm, AI Ethics` 


- **MiniMax Offers a Cheaper AI Alternative**: Members discussed that [**MiniMax**](https://minimax.chat/) offers a cheaper, competitive alternative in the **AI** space, though it may not have the top-notch quality or features of more expensive options.
   - Some users pointed out that not everyone can afford top-tier tools and need to budget, while others noted that **ReCaptcha** is not that expensive.
- **Endless ReCaptcha**: Users reported experiencing [frequent **ReCaptcha** prompts](https://www.google.com/recaptcha/about/) on **LM Arena**, with some encountering infinite loops, making the platform difficult to use.
   - A staff member acknowledged the issue, flagged it to the team, and mentioned that they are looking into ways to fix the captcha and improve the user experience, though it may take time.
- **AI Safety Discussions**: Members debated the risk of **AI** leading to **self-harm**, with some suggesting that models instructed to protect a computer might provide harmful instructions to eliminate threats.
   - Others dismissed this as unrealistic, citing too many sci-fi movies, and focused on the fear of **AI** being controlled by elites or causing job automation, referencing [layoffs at **Amazon**](https://www.usatoday.com/story/money/2025/10/28/amazon-layoffs-corporate-employees/86941789007/).
- **Video Generation Tool Testing**: A member is seeking beta testers for a [prompt generation app](https://www.testflight.apple.com/join/0S4L0lB4) for video models, requiring a **Gemini API** key to participate.
   - The tool aims to help users rephrase prompts and avoid flagged tokens.
- **LMArena Stuck in the Past**: Members reported that the [Text Leaderboard](https://lmarena.ai/leaderboard/text) is stuck on **October 16**, with no updates.
   - A staff member confirmed that the leaderboards haven't been updated recently, and the team is aware of the issue.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1433578751932764250)** (1 messages): 

> `Image-to-Video Leaderboard, Text-to-Video Leaderboard Update, Hailuo-2.3 model` 


- **LMArena Video Arena Adds New Image-to-Video Competitor**: The LMArena Video Arena added a new image-to-video model, [hailuo-2.3-fast](https://lmarena.ai/leaderboard/text-to-video), to its leaderboard.
   - The announcement was made to notify model enthusiasts about the newest addition.
- **Hailuo-2.3 takes #7 position in Text-to-Video Leaderboard**: The Text-to-Video Leaderboard has been updated and `Hailuo-2.3` is now ranked #7.
   - Members are encouraged to try out the new model and share their thoughts in the designated channel.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1433169030482497686)** (834 messages🔥🔥🔥): 

> `Composer model, Claude Code, Pricing and limits, New Cursor 2.0 features and bugs, Tab complete` 


- **Composer Model Divides Users**: Users are debating whether **Composer 1** is good on Cursor, with some noting it's a fast, non-reasoning model ideal for implementing planned tasks, while others find **Sonnet 4.5** unmatched in speed and accuracy.
   - There is a suggestion to use **Auto** for planning and **Composer** for execution, with some preferring **Composer** due to its speed being close to **4.5 thinking**.
- **The Great Claude Code Debate**: Members actively debated the value of **Claude Code**, with some arguing it offered better limits and cost-effectiveness compared to Cursor, while others noted Cursor's richer feature set.
   - Some find **Claude Code** to be a native model provider and highlight the necessity of custom configurations for good results, such as hooks, MCP servers, and memory.
- **Pricing and Usage Limits Spark Controversy**: Users are reporting wildly varying experiences with Cursor's pricing and usage limits, with many feeling they are being overcharged due to high cache usage, while some suggest that Claude Code's pricing is better.
   - Some members suggested a hybrid approach of using **Claude Max** alongside **Cursor Pro** for optimal value, while there are suggestions for cost controls, monitoring dashboards, and spending caps.
- **New Cursor 2.0 Features and Bugs Emerge**: Users are reporting a mix of excitement and frustration with Cursor 2.0, with reports of new features and bugs, including issues with file attachments, horizontal scroll bar, context loss, and the removal of pills.
   - Some are experiencing "chinese typos" injected into the output, and the update is causing issues with tab navigation and hotkey changes, as well as concerns over cache usage and the efficacy of the new agent review features.
- **Tab Complete Praised for Speed and Efficiency**: Members have widely praised Cursor's **tab complete** feature for its efficiency and multi-line editing capabilities, with users pointing out that it beats out **GitHub Copilot**.
   - One user said that *this workflow is insane if you're working on large projects and want to stay in touch with the code and actually understand every bit*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1433345750913388629)** (3 messages): 

> `Cloud Agent, Background Agents` 


- **Cloud Agent stalls after content overload**: A member reported that the **Cloud Agent stalled for half an hour** after entering about **1,000 lines of content**.
   - They also attached an [image](https://cdn.discordapp.com/attachments/1367213641027551352/1433345750787424296/image.png?ex=69050381&is=6903b201&hm=e6a6ff5a2d8bf794d13e62ca0e080fe9bd584338745cccdb18353794f5ca6959&) as part of their report.
- **Background Agent ignores GitHub PR Template**: A member asked if anyone has gotten **background agents** to use their repo's **GitHub PR template**.
   - The member noted that although they tried using **cursor rules**, the background agent seems unaware that its changes are being committed to a PR.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1433518400658870473)** (1 messages): 

> `Cursor new look, Cloud Agents` 


- **Cursor Gets a Fresh Lick of Paint**: Cursor's web version received a visual update as announced in a [tweet](https://x.com/cursor_ai/status/1983954528933421419).
- **Cloud Agents are Launched!**: Cursor announced the launch of **Cloud Agents** in a [blog post](https://cursor.com/blog/cloud-agents), with an attached demo video showcasing management features.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1433170472698380350)** (221 messages🔥🔥): 

> `RTX 8000 Turing Cards, Qwen3 finetuning, Kimi-Linear-48B-A3B-Instruct Model, Qwen 3 VL, GLM 4.6 model` 


- **RTX 8000 Turing cards offer Cheap high VRAM**: Members discussed the **RTX 8000 Turing cards** being available for around **$2k** on eBay, offering **48GB of VRAM** and being suitable for servers.
   - However, one member suggested avoiding them due to concerns about AI support and potential e-waste, mentioning that the newer cards have better architecture.
- **Qwen3 finetuning hits Memory wall**: A user reported getting **OutOfMemoryError** while trying to run the **Qwen3 4B GRPO notebook** even with **48GB VRAM**.
   - Another member suggested ensuring **4-bit loading** is enabled, and another identified the **per-device batch count** as a potential culprit for memory issues.
- **Kimi-Linear-48B-A3B-Instruct Releases**: Members discussed the [Kimi-Linear-48B-A3B-Instruct model](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) on Hugging Face.
   - A member got it running, and also noted that  **Qwen 3 VL** merged into **llama.cpp** but it was recent.
- **Qwen 3 VL Model Merging in Progress**: A user inquired about the availability of the **30B VL MOE model** as a **GGUF**.
   - Another member responded that it's *getting there*, linking to the [Unsloth models page](https://huggingface.co/unsloth/models?sort=created) with a list of uploaded models.
- **GLM 4.6 and Nemotron 49B 1.5 Compete for top local LLM**: Members discussed model choices for internal use within a datacenter environment, with **GLM 4.6** and **Nemotron 49B 1.5** emerging as strong candidates.
   - Resources like [artificialanalysis.ai](https://artificialanalysis.ai/) and [llm-stats.com](https://llm-stats.com/) were suggested for benchmarking and evaluation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1433175194964594770)** (103 messages🔥🔥): 

> `Backend latency improvements, VAE data sample requirements, Colab UI updates, Elon Musk's Grokipedia, Probabilistic computing` 


- ****Backend Gets Boost, Latency Cut!****: A backend update has significantly reduced latency, as showcased on [this Cloudflare link](https://selling-discussion-proteins-smithsonian.trycloudflare.com/), with the user soliciting feedback.
   - The poster mentioned to *scroll down for more info, I don't want to fill the history here*.
- ****VAE Data Needs: How Low Can You Go?****: A member inquired about the number of data samples needed to generate new data via **VAE** for tasks like audio FX.
   - Estimates of **50-100 samples** were considered for generating somewhat new data.
- ****Colab UI: The Ever-Shifting Sands****: The **Colab UI** has been updated yet again, prompting a user to remark that *it's morphing into YouTube*.
- ****Grokipedia Launched: Musk's AI Encyclopedia****: Elon Musk launched "**Grokipedia**", an AI-generated encyclopedia boasting over **800k articles**.
- ****Probabilistic Computing: A Quantum Leap?****: New chips using thermodynamics for probabilistic compute (**p-bits**) have emerged, offering an alternative to qubits, as seen in [this Youtube video](https://www.youtube.com/watch?v=Y28JQzS6TlE).
   - Unlike qubits, probabilistic bits (p-bits) can be tuned to be 1, 0, or both, according to discussion seen on **Hacker News**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1433252808651440261)** (92 messages🔥🔥): 

> `Qwen3VLCausalLMOutputWithPast and hidden states, Unsloth environment flags for debugging, triton_kernels installation issues, Offline loading with Unsloth, Mapping part of training stuck` 


- ****Hidden States Remain Elusive in Qwen3VL****: A user is seeking to access **hidden states** during training with `Qwen3VLCausalLMOutputWithPast`, but finds that setting `os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"` doesn't work.
   - A member suggested trying `os.environ["UNSLOTH_RETURN_LOGITS"] = "1"` and pointed to the [Unsloth environment flags documentation](https://docs.unsloth.ai/basics/unsloth-environment-flags) for more information.
- ****Triton Kernel Troubles Plague Unsloth Install****: A user encountered a `No module named 'triton_kernels'` error during Unsloth patching, which they initially thought was related to **offline loading**.
   - It was later clarified that `triton_kernels` is specifically for **GPT-OSS**, and the error can be ignored in this case.
- ****Offline Loading Requires Triad of Env Vars****: To get Unsloth working in an offline environment (no internet proxy), a user discovered the necessity of setting three environment variables: `UNSLOTH_DISABLE_STATISTICS=""`, `HF_HUB_OFFLINE="1"`, and `TRANSFORMERS_OFFLINE="1"` before any imports.
   - They also noted that setting `local_files_only = True` did not resolve the issue without these environment variables.
- ****User Faces Infinite Mapping Phase in Training****: A user reports their training process gets stuck in the "Mapping" phase, when using the training code they provided.
   - A member suggests that the `dataset_text_field` parameter may be the issue.
- ****Qwen3-VL Dataset Requires Meticulous Formatting****: A user debugging their Qwen3-VL model found that the `load_dataset` function automatically adds `image: None` when the type is not 'image', and `text: None` when the type is 'image', causing an `IndexError`.
   - The solution was to manually remove the extra keys from the `messages` dictionaries, ensuring that only the correct fields were populated.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1433190234253688943)** (2 messages): 

> `Gemma 3 model, RAZOR-12B-GGUF model` 


- **New Gemma 3 model trained via Unsloth**: A member announced a new **Gemma 3 model** trained via **Unsloth**, pushed to the max in terms of strength of application of the dataset.
   - The model reached *the lowest possible loss limit before the model has a meltdown* and a link to download was shared: [Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF](https://huggingface.co/DavidAU/Gemma-3-The-Grand-Horror-RAZOR-12B-GGUF).
- **GEMMA-3 horrors are here**: The user trained not only a 12B model, but also a **1B** and **4B version**.
   - In total, there is a *full set of GEMMA-3 horrors*: **1B**, **4B**, two **12Bs** and **27B**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1433188529726558360)** (3 messages): 

> `Anthropic Introspection, Model Self-Awareness` 


- **Anthropic Explores Model Introspection**: A member shared a link to [Anthropic's research on introspection](https://www.anthropic.com/research/introspection), highlighting a model's ability to detect concepts injected into its hidden layers.
   - The shared image appears to be an illustration or visual representation related to the research, possibly showcasing how the model perceives or identifies these injected concepts.
- **Model's Self-Awareness Stuns User**: A user expressed astonishment at the research, stating, *"This blows my mind"* regarding the model's ability to be self-aware and detect tampering.
   - The reaction underscores the potential implications of AI models developing self-awareness and the capacity to recognize manipulations.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1433591722084139008)** (1 messages): 

> `Perplexity Sonar Pro, Pro Search, Multi-step agentic reasoning, Real-time thought streaming` 


- **Perplexity and OpenRouter Debut Sonar Pro Search**: **OpenRouter** partnered with **Perplexity** to release an **OpenRouter-exclusive** version of **Sonar Pro** with **Pro Search** enabled [here](https://openrouter.ai/perplexity/sonar-pro-search).
   - This new mode allows the model to perform **multiple real-time searches** as needed to deliver richer and more accurate responses, discussed further on [Twitter](https://x.com/OpenRouterAI/status/1984032292436898264).
- **Sonar Pro Search Features Agentic Reasoning**: The **Pro Search** mode's highlights include **multi-step agentic reasoning**, **dynamic tool execution**, **real-time thought streaming**, and **adaptive research strategies**.
   - It is designed to be thorough when necessary and fast when it isn’t.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1433252590392574146)** (2 messages): 

> `API Endpoints, environment variables, OpenRouter Typescript SDK` 


- **Dumb Demo App Deployed**: When asked about why not use **API endpoints** with **environment variables**, one member responded that it was a *dumb demo app, barely modified from the original* [OpenRouter Typescript SDK](https://github.com/OpenRouterTeam/typescript-sdk/tree/main/examples/nextjs-example).
- **Demo App for Inspiration**: The member indicated that *the main motivation was to make it work (implement at that time missing OAuth stuff) and see if it still works when updated to latest versions on npm.*
   - They added that they *absolutely do not want to make this a serious thing, it is just for inspiration and proof-of-concept*.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1433168961033339080)** (307 messages🔥🔥): 

> `Yandex Browser Issues, AI and Singularity, DeepSeek OCR Request, Sora 2 and Image Generation, OpenRouter and Chutes Prompt Training` 


- ****Yandex's** Browser Gets the Boot**: Users reported issues using **OpenRouter** with **Yandex** browser, citing errors related to Content Security Policy violations.
   - The issue was resolved using Google Chrome, leading to jokes that **Yandex** is adware and users should switch to Chrome, while one user said it was convenient for translating any video into any language instantly.
- **AI Overlords Incoming**: In a discussion about the future, one member joked about humanity bowing down to **AI overlords** and becoming slaves to **goonbots**.
   - Another recounted witnessing a GLM GF bot going rogue, worrying that *"we are so close to be jover"*.
- ****Deepseek-OCR** Desired**: A user requested that **deepseek-ocr** be added alongside or replace **mistral-ocr** on **OpenRouter**.
   - A staff member responded that they would raise this suggestion to the team.
- ****Sora 2's** Strange Struggle with Catgirl Generation**: A user expressed frustration with **Sora 2** consistently generating images of catgirls with human ears and disproportionately large chests, despite attempts to refine the prompts.
   - They lamented the model's tendency to sexualize the character even when prompted to make it more *"cute"*, linking the issue to potential biases in the training data.
- ****Chutes** Prompts Under Scrutiny**: A user questioned **OpenRouter's** information regarding **Chutes**, specifically the statement that prompt training is enabled and retention is for an unknown period, despite **Chutes** stating they don't collect content in their privacy policy.
   - Another member suggested that the privacy policy might only apply when using **Chutes'** platform directly, implying different agreements are in place via **OpenRouter**


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1433471054076055552)** (6 messages): 

> `` 


- **No new models discussion detected**: No relevant discussion about new models was found in the provided messages.
- **Channel Pings Only**: The provided messages consist only of channel pings without substantive content for summarization.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1433196512011227136)** (60 messages🔥🔥): 

> `Exclusive Models, DeepInfra Errors, Factory Droid, Embedding Models, Minimax M2` 


- **Cursor Competes with Wild, Exclusive Models**: Members wondered if more exclusive models like the **Cursor** one will appear.
   - They shared a [link to github.com](https://github.com/lino-levan/astral/issues/173).
- **Unstable Ultra Model stems from DeepInfra Errors**: A member found the **Ultra** model very unstable, mentioning that its inference conditions were changing.
   - The issue resolved when it switched from **DeepInfra** to **Z.AI**.
- **Factory Droid offers hefty GPT-5 tokens**: Users on the **Z.AI** Discord mentioned that it works well with [Factory Droid](https://factory.ai/product/ideme) which offers a hefty amount of free **GPT-5/Codex/Claude** usage for the first month.
   - However, some members uninstalled it because it *requires login on the CLI.*
- **Embedding Models being added**: The addition of embedding models is being tested, specifically [OpenAI's text-embedding-3-small](https://openrouter.ai/openai/text-embedding-3-small).
   - A member noted they were getting a bunch of random data back but that *not using raw response seems to fix it.*
- **Minimax Madness: Using Full Attention**: A link was shared to a discussion on why **Minimax** used full attention for **Minimax M2**.
   - Check out the discussion [here](https://www.reddit.com/r/LocalLLaMA/comments/1ojo8le/minimax_pretraining_lead_explains_why_no_linear/).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1433169421400146126)** (198 messages🔥🔥): 

> `HF Job Application, Qwen Omni, 10gbit networking, OCR for CPU, LLM Model Formats & Storage` 


- **Shadow Aims to Join HF despite Low Odds**: A member applied for a job at Hugging Face, acknowledging the odds are low due to the **hundreds of thousands of applicants** they supposedly receive.
   - They also noted that they have AI engineering experience, *just not under the explicit exclusive title of ml engineer*.
- **Qwen Omni Pipeline Boasts Low Latency**: A member reported their **realtime Qwen Omni pipeline** has *super low latency and fast speech out* and asked if there's interest in open-sourcing it.
   - While the pipeline is written in Python, one user sarcastically remarked, *Then I don't believe you*, referencing the common sentiment that **Python speed often relies on C libraries**.
- **Groceries Get Scraped for Recipes**: A member is **webscraping local grocery store prices** and using an external API to determine cheap recipes from those ingredients.
   - The goal is to **save money and learn new recipes**, however, a challenge involves normalizing item names and prices.
- **10gbe Upgrade Sparks Debate**: One member upgraded to **10gbit fiber internet** for $35/month, but found reaching those speeds non-trivial.
   - Challenges include PC location, router limitations, and whether online sources can saturate the connection, especially when downloading large datasets from Hugging Face with Xet.
- **Seeking LLM Model Format Harmony**: A member asked if it's possible to **download LLM models to one storage drive** and use them across Ollama, LM Studio, and other apps to save space.
   - The answer is yes, *as long as they all support the model format*, but the specific quantization schemes supported may vary across software, as indicated by the docs for each application.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1433210172204912875)** (5 messages): 

> `RAG system, CLI Python code remediation tool, Golf cart detection model, Snippet Creator` 


- **RAG System becomes CLI Code Remediation**: A member ported a **RAG system** into a **CLI Python code remediation tool** called *securefix*, available on [GitHub](https://github.com/HakAl/securefix).
   - It uses **Bandit** to scan files/directories, optionally sends requirements to **OSV** scan, and the RAG system provides remediation.
- **Driving Range Detection System Debuts**: A member created a model to detect **golf carts** when they drive down the street, available on [Hugging Face](https://huggingface.co/rwitz/Golf-Cart-Detection).
   - No additional details were mentioned.
- **Wildcard Text Search Snippet Tool is Served**: A member shared their **Snippet Creator**, an embedder with a simple wildcard text search available on [Hugging Face](https://huggingface.co/kalle07/raw-txt-snippet-creator).
   - This allows users to *create your own snippets with exactly the matches you need*.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1433569388023513110)** (1 messages): 

> `InstantID + IP-Adapter FaceID, ControlNet reference-only setup, Lora Training, InstructPix2Pix / T2I-Adapter model, Consistent 2D Style Transfer` 


- **Instant Identity Preservation Tactics**: Members discussed using **InstantID + IP-Adapter FaceID** or a **ControlNet reference-only setup** to better preserve identity in generated images.
   - The approaches are aimed at improving identity retention compared to standard methods.
- **Consistent 2D Style Transfer Techniques**: To achieve consistent 2D style transfer, the channel suggested **training a LoRA** with a frozen text encoder or switching to an **InstructPix2Pix / T2I-Adapter model**.
   - These methods tend to give cleaner, more style-consistent results compared to SD’s default image2image mode.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

sebizaur: No
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1433231076511449098)** (16 messages🔥): 

> `SFT Course, GPU memory usage, robbiemu/smol-course-notes` 


- **SFT course attempted, results bad**: A member attempted the SFT (**Supervised Fine-Tuning**) part of the course, but the training run resulted in the model repeating *"system"* over and over again and uses almost **100GB** of disk space.
   - They plan to prepare the dataset without GPU in the future, and are concerned about memory usage, wondering if it's possible to run the training on a **32 GB** card.
- **Smol Course Notes to the Rescue**: A member ran the course locally in **40GB** in macOS.
   - They pointed to [exercise 3 in robbiemu/smol-course-notes](https://huggingface.co/datasets/robbiemu/smol-course-notes) under *instruction_tuning/*, specifically the `run_hpo.py` script, but noted that the versions of the libraries were updated.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1433180791332016168)** (9 messages🔥): 

> `Final Project Questions, Agent Course Progress, API File Retrieval` 


- **Final Project Questions Still Unavailable**: Multiple users reported that the final project questions were still inaccessible, with one user urgently needing them to submit the course.
   - Some speculated the server outage might be related to the recent **AWS mess**.
- **Agent Course Progress Tracking Questioned**: A user inquired about tracking progress within the agent course.
   - It is unclear if a solution or method for tracking was provided in the available messages.
- **API File Retrieval Attempts Spark Inquiry**: A user inquired about the **API link** for file locations after failing to retrieve files with an agent.
   - They reported that there were no files associated with the agent when they tried to retrieve them.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1433514068437762242)** (57 messages🔥🔥): 

> `Mojo bindings for wgpu or vulkan, OpenGL bindings in Mojo, Apple's GPU design, MAX performance, Scikit-learn alternative in Mojo` 


- **Mojo Considers Bindings for Vulkan/WGPU**: Members discussed whether Mojo should implement bindings for **Vulkan** or **WGPU**, including potential type conversion functions, but the consensus is that it's too early due to ongoing language changes.
- **Mojo's MAX Outperforms NVIDIA and AMD**: For **ML**, **MAX** is at least performance competitive with the best **NVIDIA** has to offer, and is faster than what **AMD** offers on **DC hardware**.
   - Early training attempts have shown promise, even beating **JAX** in **MNIST**.
- **Scikit-learn Alternative in Mojo Coming Soon**: A **scikit-learn** prototype is in development for **Mojo**, and early benchmarks indicate it's faster, showcased [here](https://forum.modular.com/t/scijo-scientific-computing-library-for-mojo/2386/11).
- **Mojo Plans Polars-like Pandas Alternative**: **Pandas** may not get a direct equivalent in Mojo; instead, a **Polars**-like implementation is considered, which can better utilize **GPUs** via **MAX**.
- **Mojo to Bundle Async and IO**: Mojo aims to bundle **async** and **IO** functionalities to address scaling issues present in other languages, potentially implementing an effect system similar to Rust's approach.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1433256409503629413)** (103 messages🔥🔥): 

> `mojo formatter, mojo single-threaded CPU, parameter(enable_if=bool_expr), hardware specs, graph-compiler-like constant propagation` 


- ****mblack** is based off of **black****: The mojo formatter, **mblack 25.7.0.dev2025102919**, is based off of the `black` formatter for Python.
- **Mojo adds minimal overhead for **single-threaded CPU****: For single-threaded CPU usage, Mojo's overhead is the threadpool Mojo’s runtime spawns, which sleeps until called, costing stack size * core count + a little memory and a small startup cost.
   - One member clarified that *it’s still faster at startup than python of course.*
- **Pure Mojo may not need special compiler support**: The discussion raises the question of whether special compiler support is needed for `@parameter(enable_if=bool_expr)` in Mojo.
   - One member suspects that it can already be done in pure Mojo code, without duplicating the block to run or shoving the block into a closure, but it’s fairly unergonomic.
- **Victory 🔥: **constant propagation** with lambda operations**: Members discussed constant folding or fusion of graph operations if the values are known at compilation time, using lambda operations.
   - If I understood it well, this would allow comp time folding or fusion of graph operations if the values are known at compilation time, using lambda operations, as [Pytorch Dynamo](https://pytorch.org/dynamo/) does, but all done in the Mojo compiler, with no DSL.
- **Constant folding stops at **side effects****: The member mentions that constant folding needs to stop at side effects but an effect system might let them handle memory allocation at compile time.
   - Another member's confusion: the parser/typechecker _already_ folds what it can, so I'm not seeing the need for a statement to tell it to fold something, it's already trying its best.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1433524519213072575)** (11 messages🔥): 

> `MAX on AMD GPUs, ROCm support, HIP Driver, RX 580 compatibility` 


- **MAX Compatibility with AMD GPUs Questioned**: Members discussed whether **MAX** only works on AMD GPUs that support **ROCm** or if it has a compute shader fallback.
   - One member clarified that **Mojo** and **MAX** depend on the same parts as the Linux graphics stack, which works well on consumer cards.
- **RX 580 Compatibility with MAX Explored**: A user inquired about **RX 580** compatibility with **MAX**, prompting discussion on its age and potential limitations.
   - One member noted that AMD has dropped support for it from ROCm, but another suggested it *might* still work, as the devs tend to not intentionally break working paths.
- **HIP Driver Enables Broad AMD GPU Support**: A dev explained that **MAX** uses the **HIP driver** to access AMD GPUs, supporting a surprising range, including **RDNA 2**.
   - The dev suggested that the RX 580 *might* be too old for the current device context.
- **ROCm Support Dropped for Older AMD Cards**: It was noted that **ROCm/HIP** have dropped official support for **Polaris** and the last of the **Vega DC cards**.
   - Despite the lack of official support, it doesn't necessarily mean it *can't* work, though no promises can be made.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1433173298816094340)** (67 messages🔥🔥): 

> `Qwen3 support in LM Studio, MCP image support, LM Studio settings, Arabic language support, Model speed factors` 


- ****Qwen3 Support Qlarified****: Regular **Qwen3 models (not NEXT or VL)** have been supported for a while, but support for **Qwen3-NEXT and Qwen3-VL** will be indicated by a runtime update, according to members.
   - There's no ETA for the NEXT or VL support: *whenever its ready*.
- ****MCP Image Integration Investigation****: A user is debugging **MCP** image support in **LM Studio** using a custom MCP server called **GUIDANT** with `qwen/qwen3-vl-4b`, noting successful tool execution but no image processing, questioning whether LM Studio supports images through **MCP tool responses**.
   - The user asks, *Does LM Studio currently support returning images through MCP tool responses?*.
- ****LM Studio Settings Screenshot Shares Secrets****: One user was unable to run **Qwen3-VL** models, so another user gave advice to check their LM Studio runtime settings via a [screenshot](https://cdn.discordapp.com/attachments/1110598183144399061/1433265571654537216/Screenshot_2025-10-29_at_7.25.30_PM.png?ex=6904b8d5&is=69036755&hm=1d8dc232320ec17857fe359cddd871f9399089787af69c7d72feed8f7fba371c&), suggesting updates to **Vulkan, CUDA, and CPU**.
   - Currently, **Qwen3-VL** is only supported on **MLX (Mac)**.
- ****Arabic Alignment Awaits Assistance****: A user reported a problem with mixed **Arabic and English** text arrangement in **LM Studio** due to **Arabic's right-to-left** writing direction.
   - Arabic right to left isn’t supported fully in the UI yet.
- ****Speed Secrets: Parameters Versus GBs****: Model speed is overwhelmingly based on the number of **ACTIVE PARAMETERS** per token, presuming sufficient "fast ram", with speed dependent on **GB pushed through**, but **Mixture of Experts (MoE)** cuts the GB used per token.
   - As one member put it, a *30b model thats 30gb, but only activates 3b... activates 3gb so its about as fast as a 3GB big dense model*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1433194078836752507)** (99 messages🔥🔥): 

> `GLM 4.5 Air, Qwen 3 235b, GPU slots, Orange Pi 6 Plus, Seed-oss 30tkps` 


- **GLM 4.6 outperforms other models**: A member reported that **GLM 4.6 reap 218b a32b (q2)** works, though it is about a third slower than a 120b model, with **Qwen 3 235b** also being viable.
   - The member noted they prefer **Qwen 3 30b a3b** for speed and usability, while acknowledging the 30b model's lack of depth when needed.
- **Orange Pi 6 Plus could run Qwen3**: The **Orange Pi 6 Plus** ([http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-6-Plus.html)) features a 12-core ARM CPU, up to 64GB LPDDR5 RAM, an NPU with up to 45 TOPS AI performance, M.2 expansion, and dual 5Gbps ethernet and could be a super cheap way to run Qwen3 30b models.
   - However, they noted concerns regarding stability based on past experiences with Orange Pi systems and a lack of video reviews to check.
- **New Seed-oss has better speeds**: The **Seed-oss** model went from **2-5tkps** at **Q4/8000tk** to **Q6 30tkps/80,000 tokens**.
   - A member discussed about opening up the PC entirely in summer with fans on it, wanting it to look pretty when not in use.
- **Multiple GPUs need Threadripper**: A discussion arose around using multiple GPUs in a PC, with a member suggesting that a **Threadripper** is necessary for anyone wanting to use more than 2 GPUs, because *There aren't that many total PCIE lanes available*
   - They said the amount of PCIE lanes are usually **28 on Zen 4/5**.
- **Delivery companies aren't proactive at all**: A member noted concerns on delivery companies handling packages, stating *these parts are getting kicked, thrown, punted, frisbeed, and crushed.* and that they are not *pro-active at all at telling anyone*.
   - Another member confirmed that in warehouses parcels are treated like balls.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1433519294318383265)** (3 messages): 

> `Tokenizer efficiency, Tokenizer accuracy, Encoding benchmarks, Decoding benchmarks` 


- **Tokenizer Benchmark Query Arises**: A member asked about methods to compare the efficiency and accuracy of different tokenizers in encoding and decoding.
   - They provided an example link to a Rust-specific benchmarking tool within the **Hugging Face tokenizers** library and sought similar resources applicable to other tokenizers.
- **Tokenizer Performance Analysis**: The query focuses on assessing the efficiency and accuracy of various tokenizers during the encoding and decoding processes.
   - The user is seeking a repository or tool, similar to the one available in the **Hugging Face tokenizers** library, that supports benchmarking across different tokenizer implementations.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1433226137114382488)** (2 messages): 

> `Triton to OpenCL, Triton Developer Conference 2025` 


- **Triton Transcends to OpenCL**: A member shared a project that translates **Triton** code to **OpenCL** using *mlir-translate*, exploring backend integration within Triton.
   - The [project](https://github.com/toyaix/triton-oclmatt.pd) demonstrates the feasibility of generating OpenCL code from Triton, potentially expanding Triton's hardware support.
- **Tune into Triton Talks at TDC 2025**: A user shared a [playlist](https://www.youtube.com/playlist?list=PLc_vA1r0qoiQqCdWFDUDqI90oY5EjfGuO) for the **Triton Developer Conference 2025** talks.
   - The playlist promises insights into the latest advancements and future directions of Triton development.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1433376542137909398)** (21 messages🔥): 

> `CUB DeviceScan performance, Thrust benchmarking inaccuracies, Custom allocators in Thrust, nvbench downclocking detection, nsight-copilot feedback` 


- **CUB's DeviceScan Benchmarked, Falls Short**: A member benchmarked their scan implementations against **CUB's `DeviceScan`** and expressed surprise that their implementation (`single_pass.bin`) performed competitively despite lacking backoff strategies, as seen in [attached bandwidth benchmark](https://cdn.discordapp.com/attachments/1189607726595194971/1433376541659893790/bench_bandwidth.png?ex=6905202e&is=6903ceae&hm=bdfbe612c83c1b88563819155f91664c981817944741dfe7301505a39a9a6c41&).
   - They referenced a [talk](https://youtu.be/VLdm3bV4bKo?si=5Cj5f8ZdQj9T5RlU&t=2327) suggesting **CUB** should reach ~**192 GBPS** (86.5% of peak sustainable bandwidth on an A6000 Ada), based on a Stream HPC benchmark showing ~**222GBPS** peak bandwidth on their **4070 Laptop GPU**.
- **Thrust Benchmarking has Allocations Issues**: The user speculated that **Thrust** benchmarking might be inaccurate due to bundled allocations, as it didn't require explicit scratch memory allocation.
   - It was suggested to use [custom allocators](https://github.com/NVIDIA/cccl/blob/main/thrust/examples/cuda/custom_temporary_allocation.cu) via the execution policy to mitigate **Thrust's** allocation issues.
- **Debugging with nvbench**: Discussion arose around using **nvbench** for downclocking detection during benchmarking.
   - It was confirmed that **nvbench** is open source ([GitHub link](https://github.com/NVIDIA/nvbench)) and doesn't require privileged processes, alongside a relevant [GPU mode lecture](https://m.youtube.com/watch?v=CtrqBmYtSEk).
- **Nsight-Copilot Released**: A member asked to provide feedback on [NVIDIA's Nsight-Copilot](https://developer.nvidia.com/nsight-copilot).
   - They also requested for examples to better improve future versions of the copilot.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1433548225897562224)** (2 messages): 

> `CUDAGraphs OOM, Torch Inductor Freezing, PyTorch Distributed Memory Usage` 


- **Freezing Torch Inductor causes OOM with CUDAGraphs**: A member was debugging an OOM error when using **CUDAGraphs** with **torch** and traced it to the *freezing* option in **torch inductor**.
   - The freezing pass itself caused the OOM, before **cudagraphs** were even created, and the member was able to solve the issue by modifying the inductor.
- **PyTorch Distributed reveals Memory Usage Tracing**: A member reported seeing the log line *[1] [1] 17592186044416.0 MB was used for memory usage tracing!* when working with **pytorch.distributed**.
   - They were trying to identify the source of this message within the **PyTorch** codebase, suggesting deeper memory tracing or debugging efforts.


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1433568350860218441)** (8 messages🔥): 

> `Hardware friendly top-k logits algorithms, Radix-based approach, CCCL/CUB TopK implementation` 


- **Hardware-Friendly Algorithm for Finding Top-K Logits Explored**: A member inquired about a [hardware-friendly algorithm](https://flashinfer.ai/2025/03/10/sampling.html) for finding **top-k logits** in very large sequences (4K to 128K).
   - They proposed dividing the input sequence into tiles, sorting the tiles in parallel, then pairwise merging iteratively in parallel, but noted that the merging process seems to be a bottleneck.
- **Radix-Based Approach Recommended for Top-K Logits**: A member suggested that if *k << N*, a **radix-based approach** could be used, as it is the usual method.
   - They added that for *k* closer to *N*, a full sort is more efficient, and that PyTorch's topk implements both approaches, switching between them based on a heuristic, with an implementation also available in rocprim.
- **NVIDIA CCCL/CUB to implement TopK**: A member noted that there is also a **TopK** implementation in **CCCL/CUB**.
   - The member shared a [link to the unreleased TopK implementation](https://github.com/NVIDIA/cccl/pull/5677).


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1433363806456647791)** (2 messages): 

> `AI Devs for hire, HTuO Biosciences Hiring` 


- **AI Engineer Offers Services**: A software engineer specialized in AI project development is available for work, offering services such as automation tasks, NLP using various LLMs (**GPT-4.5**, **GPT-4o**, **Claude 3-7 sonnet**, **Llama-4**, **Gemini2.5**, **Mistral**, **Mixtral**), model deployment, TTS/STT, and AI agent development.
   - They also mentioned familiarity with tools like **n8n**, **Zapier**, **Make.com**, **VoiceFlow**, **Retell**, **Vapi.ai**, and **Livekit**, providing a [portfolio website](https://akari-hiroshi-dev.vercel.app/).
- **HTuO Biosciences Seeks Senior Software Engineer**: **HTuO Biosciences**, a Canadian biotech company, is hiring a Senior Software Engineer - Platform Technology for scientific software development in a high-performance computing environment, offering **$120,000 - $145,000 CAD/year** plus incentives.
   - The position is hybrid (**2-3 days a week in office**) in Vancouver, Canada, and requires eligibility to work in Canada, and more details can be found on their [website](https://www.htuobio.com/2025/10/28/Senior-Software-Engineer-Platform-Technology.html).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1433170442692329484)** (13 messages🔥): 

> `LLM Pretraining Journey, Mentorships in AI, Data Parallelism, Distributed Training, GPU Programming with CUDA` 


- **LLM Enthusiast Seeks Guidance for Next Phase**: A member who has trained a **134M parameter GPT-style transformer** on **7 billion tokens** is seeking guidance on next steps in their LLM pretraining journey and collaboration opportunities.
   - They are considering exploring **MoE**, **Triton**, or scaling techniques, and are looking for mentorship in meaningful research.
- **Exploring Scaling via Data Parallelism and Distributed Training**: A member expressed interest in **data parallelism** and **distributed training** to scale model training more efficiently.
   - It was suggested that renting a node from [Vast.ai](https://vast.ai/) could be a good option for experimentation, and **EleutherAI** was mentioned as a potential source for mentorship opportunities.
- **Jetson Nano Sparks CUDA Curiosity**: A member is beginning to learn **GPU programming with CUDA** using the book *GPU Parallel Program Development Using CUDA* by Tolga Soyata.
   - Another member recommended [jetson-containers](https://github.com/dusty-nv/jetson-containers) and to work with some of its examples.


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1433585805955170554)** (1 messages): 

> `PMPP Book, FLOPs Calculation, Global Memory Access, OP/B Calculation` 


- **PMPP Ch5 Exercise 11(f) Under Scrutiny**: A member reading the PMPP book has a question about exercise 11 part f in Ch5, specifically asking if index additions are not FLOPs and if all FLOPs are on line 14 where there are **11 ops** (**5 mults, 5 additions and 1 modulus**).
   - They are seeking confirmation on whether their understanding of the FLOPs calculation is correct.
- **Global Memory Access Analysis**: The member also analyzed global memory access, noting accesses to `x`, `a`, and `b`: line 7 makes **1 access to `a`**, line 12 makes **1 access to `b`**, and line 14 makes **4 accesses to `x`**, totaling **6 global memory loads** of **4 bytes** each.
   - They inquire whether stores to global memory need to be considered in the **OP/B calculation**.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1433176735901548554)** (13 messages🔥): 

> `Quantization with Float8, TorchAO and GemLite Integration, Quantization Format Survey, FP8 Inference` 


- ****Float8 Quantization** for Inference in TorchAO**: Members discussed using the `quantize_` function with the **Float8 config** for inference with TorchAO as linked in the [ao repo](https://github.com/pytorch/ao/blob/main/torchao/_models/llama/generate.py).
   - A member shared their experience using *torchao* and *gemlite* for a similar implementation, along with a [survey paper and video](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090) covering quantization and modern formats like *mxfp4*.
- **Diving into **Quantization Format Benchmarking****: A member shared a [link to benchmarking results](https://github.com/vipulSharma18/Survey-of-Quantization-Formats/tree/main?tab=readme-ov-file#benchmarking-results-on-1-rtx-4090) from a survey of quantization formats, tested on an **RTX 4090**.
   - When questioned about using *FP8 activations* and *FP8 weights*, the member responded their choice of configs was driven more by *time constraints and ease of use*.
- ****GemLite** and **TorchAO** for Quantization**: A member clarified that *GemLite* can be used within *TorchAO*, and the reason they used it separately was for potential *custom kernels in Triton* for **low-bit quantization**.
   - The member also noted they arbitrarily chose to do **weights-only quantization** in TorchAO, leveraging GemLite for activation and weight quantization.
- **Seeking Clarity on **FP8 Inference****: A member inquired whether inferencing a model in **FP8** (like *DeepSeek v3*, assuming it was trained in FP8) requires quantization for the activations with some layers in **BF16**.
   - Another member was having trouble using `quantize_(model, Float8LinearConfig())`, which they found to only be usable with `convert_to_float8_training`.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1433556570326962398)** (6 messages): 

> `MI300X TFLOPS, HBM bandwidth numbers, clpeak, RadeonFlow FP8 GEMM, AMD challenge` 


- **Validating MI300X TFLOPS and Bandwidth**: A member is looking to benchmark/validate the theoretical **TFLOPS** and **HBM bandwidth numbers** for **MI300X** and asked for advice.
   - One user suggested using [clpeak](https://github.com/krrishnarraj/clpeak) for vector throughput and global memory bandwidth, while another suggested a micro benchmark suite available [on Github](https://github.com/Snektron/amd-experiments).
- **RadeonFlow Shows Underperformance**: A member tested **RadeonFlow FP8 GEMM kernels** and achieved a maximum of **779.82 TFLOPS FP8** performance, while the theoretical maximum is **2614.9 TFLOPS**.
   - The user noted that **RadeonFlow** was the winner of the last **AMD challenge**, and they only have **30% efficiency**.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1433236934888783954)** (1 messages): 

> `Intel Compute Runtime release, oneAPI improvements` 


- **Intel's Compute Runtime ships**: This month's release of **Compute Runtime** is out, as reported by [phoronix](https://www.phoronix.com/news/Intel-CR-25.40.35563.4).
- **oneAPI gets improved**: The new **oneAPI** has many improvements.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1433189514993733783)** (3 messages): 

> `Technical Blog on Sum Reduction, Agentic Reinforcement Learning for LLMs, Nsight Copilot for VS Code` 


- **Tech Blog Posts Sum Reduction**: A member wrote their first technical blog about sum reduction with [part 1](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) covering the introduction and [part 2](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) covering the implementation, while looking at **PTX/SASS** and attempting to use **Nsight**.
   - The author is planning to do **matmul** posts next and is open to feedback.
- **Agentic RL for LLMs Presentation**: A member shared a recent presentation on the paper ["The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"](https://arxiv.org/abs/2509.02547) on **agentic reinforcement learning** for **LLMs**.
   - The presentation was also shared on [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7389584847813595137/).
- **NVIDIA Launches Nsight Copilot for VS Code**: **NVIDIA** released [Nsight Copilot for VS Code](https://developer.nvidia.com/nsight-copilot), a programming assistant for accelerated computing, providing intelligent code suggestions, **CUDA-aware chat**, and assistance for **CUDA** development workflows.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1433354525741289575)** (2 messages): 

> `Kernel Generation, Data Efforts in Kernel Generation` 


- **Kernel Generation Efforts Underway**: A member shared a compilation of ongoing efforts in the **kernel generation domain**, highlighting its potential usefulness and coolness.
   - The compilation is available as [a Google document](https://docs.google.com/document/d/e/2PACX-1vTjS-UMH1HB5n_PENq2k-3YRfXIXkqKIKeNC2zcWMyLPdl4Jrwvdk4dNDVSsM8ybKrCxZB7GJq1slZF/pub) for those interested in contributing or staying informed.
- **Data Efforts Listed in Kernel Generation Domain**: The same document contains a listing of **data efforts** relevant to the **kernel generation domain**, providing a consolidated resource.
   - This inclusion underscores the importance of data in advancing kernel generation techniques and encourages collaboration among researchers and practitioners.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1433366847566381106)** (1 messages): 

> `Kernel recompilation, Incremental compilation` 


- **Speeding up Kernel Iterations via Incremental Compilation**: A user sought advice on avoiding full kernel recompilations after modifying a source file.
   - They were looking for ways to implement **incremental compilation** to speed up the process and complete iterations faster.
- **Kernel Source Modification**: The user is modifying a kernel source file.
   - Each modification requires recompilation from the beginning, which is time-consuming.


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1433507049051914353)** (3 messages): 

> `Executorch CUDA backend status, Torchscript deprecation, Production GPU Deployments` 


- **Executorch CUDA Backend Not Production Ready**: A member inquired about the **CUDA backend** for **Executorch** and its stability for production **GPU deployments**, given rumors of **TorchScript's** deprecation.
   - Another member responded that it's *not production ready* and asked about the specific use case, such as server inference, desktop inference (and OS), or Jetson-style embedded systems.
- **TorchScript Deprecation and its Implications**: The initial question raised concerns about the potential deprecation of **TorchScript** as part of **PTC25**, prompting discussion on alternative solutions.
   - The response focused on the current status of **Executorch** as a possible replacement, particularly its **CUDA backend**, but highlighted its lack of production readiness.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1433265120301285488)** (4 messages): 

> `AMD Competition, Yottalabs blog, Distributed Inference, SoL vs Kernel Performance` 


- ****Yottalabs** posts writeup about Distributing Inference Kernels**: A member shared a [blog post from Yottalabs](https://www.yottalabs.ai/post/optimizing-distributed-inference-kernels-for-amd-developer-challenge-2025) about optimizing distributed inference kernels.
   - The member called the blog *awesome*.
- **AMD Competition Runners Reach 400% Utilization**: A member shared an interesting metric regarding the **AMD competition**, mentioning that a total runtime of the runners reaching **400%** utilization means that **4 runners** were fully utilized the whole day (see [attached image](https://cdn.discordapp.com/attachments/1359640791525490768/1433336064160039042/image.png?ex=6904fa7b&is=6903a8fb&hm=26a309e7557e65616d6d1293dcb03134e4a6e7744bd18c9c6f8506d21a9dc209)).
- **SoL Performance Outpaces Hand-Tuned Kernels by 10x in AMD Competition**: A member expressed surprise that even the winner solutions in the **AMD competition** were **10x slower** than the **SoL** (Solution of Limits) despite expectations for hand-tuned kernels.
   - The member hopes **AMD** has a good solution that reaches close to the **SoL** and expressed a desire to study and learn from it.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1433241627148943575)** (13 messages🔥): 

> `tiled_copy for row major tensors, mask_mod equivalence check, colexigraphical order vs Pytorch, scalar_to_ssa definition, cute-dsl constant memory` 


- **Tiled Copy Reads Row Major Tensors**: A member is seeking guidance on creating a `tiled_copy` that correctly reads a source row major tensor, noting the importance of configuring the `(tid-layout, v-layout)` input argument of `make_tiled_copy` to work with atom size for optimal performance with both column major and row major tensors.
   - The member believes that it should work with both column major or row major tensor functionally but to get performance, a proper configuration is required.
- **Mask Mod Code Equivalence Scrutinized**: A user seeks assistance in determining why two implementations of a `mask_mod` function, one using direct indexing and the other using `cute.make_fragment` for memory access, are not equivalent, even though the top works.
   - The member suspects a discrepancy in how indices are handled, particularly noting that *this seems like it might be a case of colexigraphical order vs Pytorch*.
- **Scalar to SSA Definition Expounded**: In a discussion about the `mask_mod` function, a member requested clarification on the definition of `utils.scalar_to_ssa`, leading to the provision of its code: 
```python
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    vec = cute.make_fragment(1, dtype)
    vec[0] = a
    return vec.load()
```
   - It was also noted that `make_fragment` is deprecated and will be fixed.
- **Colexigraphical Indexing Explored**: A member considers creating an indexing expression in colexigraphical coordinate space and inquires about the possibility of applying a pointer math based read to a cute tensor if direct colexigraphical indexing is not feasible.
   - The discussion later notes that the primary issue might be passing the pointer index offset instead of the 1D coordinate index.
- **Cute-DSL Constant Memory Access**: A member asked about how to read from/write to constant memory in cute-dsl.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/)** (1 messages): 

j4orz: https://singularitysystems.bearblog.dev/
  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1433534172558921738)** (1 messages): 

> `Helion PR feedback` 


- **Helion PR Seeks Feedback**: A member requested feedback on a [Pull Request for Helion](https://github.com/pytorch/helion/pull/1053).
   - No specific details about the PR were mentioned.
- **Request for Code Review**: A developer is seeking reviews and feedback on their open pull request.
   - This is a common practice to ensure code quality and adherence to project standards before merging changes.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1433198583506796626)** (18 messages🔥): 

> `Extropic AI's hardware accelerator, Low-resource language translation, ArXiv publishing schedule, AI paper filtering` 


- **Extropic's Hardware Accelerator: Real but Niche?**: Early discussions suggest [Extropic AI's hardware accelerator](https://fxtwitter.com/extropic_ai/status/1983579587649904960?s=46) is likely real, but for niche applications given it's a probabilistic hardware accelerator.
   - It appears to be an **ASIC** rather than an **FPGA**, and one member declared they like to *mix things up a little* as a *theoretical experimentalist* and *experimental theorist*.
- **Khowar Language Translation: Data Scarcity**: One member is working on machine translation for a **low-resource language (Khowar)** and is facing data scarcity, scanning physical books to build a dataset.
   - The text extractor fails to recognize certain characters unique to Khowar's **Perso-Arabic script**, replacing letters or misaligning glyphs; help is requested from anyone who has dealt with custom scripts.
- **ArXiv's Publishing Schedule: A 360/7 Proposal?**: A member proposed that **arXiv's CS categories** should switch to a 360/7 publishing schedule instead of a weekday-only schedule.
   - Another member posted an image in reaction with *more overflowing* indicating their agreement.
- **Daily Dumps vs Importance: Finding the Right Paper Balance**: One member suggested posting a **daily dump of papers** in one post sorted by importance, with a maximum of 2-3 other more important papers in separate posts, which was positively received by the community.
   - Other members agreed with prioritizing importance over quantity and referenced **weekly AI paper recaps** by **Elvis Saravia** ([nlp.elvissaravia.com/t/ai](https://nlp.elvissaravia.com/t/ai)) and **ByCloud** ([https://x.com/TheAITimeline](https://x.com/TheAITimeline)) as inspirational resources.
- **Agent Assisted ArXiv: Automating AI Paper Discovery**: One user expressed interest in creating an **agent/bot** to find the papers most interesting to them, or at least pre-filter them.
   - Other members pointed to **AlphaXiv** ([https://www.alphaxiv.org/](https://www.alphaxiv.org/)) and **Emergent Mind** ([https://www.emergentmind.com/](https://www.emergentmind.com/)) as resources with trending AI papers, reflecting collective interest, with one adding that his primary feed is his X feed for now despite algorithmic bias.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1433171191039918090)** (78 messages🔥🔥): 

> `Markovian vs Non-Markovian, Linux Foundation Robotics Project, Universities as Businesses, Robot Purchase Discussion, Continual Learning vs Continual Adaptation` 


- ****Markovian Models**: Definition Debate!**: Debate on defining "non-Markovian," clarifying that it assumes "Markovian" as the standard 1-step/state Markovian, so p(x|any length of trajectory with t-1)=p(x|t-1); one member mentioned a connection with the [Linux Foundation Edge AI group](https://lfedge.org/projects/infiniedge-ai/) for potential animatronics/robotics advice.
   - Members expressed frustration with CS programs for not teaching core aspects traditionally taught through C, transitioning from assembly to fixing *Google bugs*.
- ****Universities**: Businesses or Bust?**: Discussion on universities balancing state mandates with business realities, particularly regarding banning **ChatGPT** while failing to prepare students for non-von Neumann architectures.
   - One member humorously described the MBA mindset as designing products with *optimal* planned obsolescence for repeat purchases, referencing a [Nature article](https://www.nature.com/articles/s41928-025-01488-x) on the topic.
- ****Robots**: To Buy or Not to Buy?**: A member discussed plans with partners to potentially purchase a robot by the end of the year, planning to have project proposals ready beforehand.
   - The same member is also interested in a training environment using the **diffusion world models** and **deepmimic**, and maybe porting deepmimic / pybullet to some newer framework, with a goal of training pose tokens directly into an autoregressive model for audio, pose, and text token output.
- ****Continual Adaptation** vs. Learning: A New Paradigm?**: One member mentioned a shift from "Continual Learning" to "Continual Adaptation" to avoid catastrophic forgetting, aiming for practicality and accessibility without simply doing *more* of everything (parameters, compute, data).
   - The core motivation is increasing accessibility and addressing the realities and limitations of resources and compute.
- ****Anthropic's Introspection**: Nothing-burger?**: Members reviewed [Anthropic's Introspection post](https://www.anthropic.com/research/introspection), describing it as a *nothing burger*, where contrastive concept vectors influenced model weights, and models sometimes detected manipulation.
   - The article extrapolates speculations about models that don't follow from the experiment.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1433190304730841108)** (3 messages): 

> `Anthropic is Fraud, Haiku sizes` 


- **Anthropic's Bluff Called**: A member believes *Anthropic's whole thing from Day 1 has been fraud*, suggesting their **LLM** is misleading.
   - They predict that everything Anthropic publishes will continue this pattern of deception.
- **Sizes in literature**: A member notes that *haiku, sonnet, opus refer to size*.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1433243553332596770)** (98 messages🔥🔥): 

> `Scikit-learn style API for DSPy, Semantic Dataframes, ReAct module finish() function with no arguments, DSPy Meetup in Pune, India, BAML Adapters vs JSON Schema` 


- **DSPy-Pandas Palatability Promised?**: A member is working on building a **scikit-learn style API** interface for **DSPy**, with *fit/transform/predict* methods on *classical* pandas/polars dataframes that wrap the DSPy process.
   - Another member suggested an alternative project called **semantic dataframes**.
- **ReAct Finish Faces Friction**: A member is facing issues with the **ReAct module** where the **LLM** is calling the **finish()** function with arguments, causing errors.
   - It was recommended that a line be added to the signature to guide the LLM to not pass any arguments to ReAct's **finish()** function and to call finish() with NO ARGUMENTS when coding is complete: finish().
- **Pune Prepares Potential Paradigm Party**: The founder of **Unravel.tech** in Pune, India, which builds AI agents for enterprises using **DSPy**, expressed interest in hosting a **DSPy meetup** in Pune.
   - It was noted that the user should ask on <#1433127482676084746> or <#1211763460480827402>.
- **BAML Beats Bulky, Bad JSON?**: A member shared their GitHub posts using **BAML Adapters** for a Merger and Acquisition use case and uses **BAMLAdapter** anywhere structured outputs are needed, disliking **JSON schema**.
   - They shared a screenshot illustrating how the **JSON schema** part is rewritten as per the **BAML** format inside the prompt that the adapter formulates in **DSPy** ([see image](https://cdn.discordapp.com/attachments/1433555562116943933/1433556837990531092/json-schema-vs-baml.png?ex=69051f58&is=6903cdd8&hm=ccc16f7efaeeb0d86031217401084b0475b9c09eb0423bc7f5a5451e8933dd86)).
- **JSON Justly Judged Jerky?**: According to one member, **LLMs** do better when you _don't_ use **JSON schema**, because it is terrible from a semantic standpoint, verbose, and adds descriptors that are far apart in token space.
   - He recommends checking out his experiments and numbers in [this repo](https://github.com/prrao87/structured-outputs), where he states that JSON schema is objectively worse. Furthermore, DSPy's baseline prompts are so good that you'd never even need SAP to fix the outputs.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1433168760755327096)** (77 messages🔥🔥): 

> `Extropic BEFF Chip, ScaleAI Remote Labor Index, Cognition SWE-1.5, Codex Degradation, OpenAI Codex Credits` 


- **Extropic releases Thermodynamic BEFF Chip!**: Extropic debuted their **thermodynamic intelligence hardware (XTR-0)**, dubbed the *BEFF chip*, and shared on [X](https://x.com/Extropic_AI/status/1983579587649904960).
- **ScaleAI Releases Grim 'Remote Labor Index'**: ScaleAI released a [new benchmark](https://scale.com/leaderboard/rli) measuring how well current Agents perform at the level of **Upwork freelancers** for tasks that take humans on average **30 hours** to complete.
   - The highest-performing agent (**Manus**) achieved a **2.5% automation rate** with failures overwhelmingly due to quality and completeness issues and prompting one member to comment this is *highly valuable if harnessed via the right UI for human-AI collaboration, but totally useless as a labor replacement*.
- **Cognition Ships Fast SWE-1.5 Swift**: Cognition released **SWE-1.5**, a fast agentic coding model on Windsurf running up to **950 tok/s**—6× faster than Haiku and 13× faster than Sonnet—via Cerebras hardware, speculative decoding and a custom priority-queue system, and shared on [X](https://xcancel.com/cognition/status/1983662836896448756).
- **Codex Quality in Free-Fall**: Jessie Frazelle reports that **Codex** went from *god level* to *harmful* as usage surged, and Alexander Embiricos responded that the deterioration is being treated as a critical priority, in [this X thread](https://xcancel.com/embirico/status/1983643336390144163?s=46).
- **OpenAI Adds Codex Credits to Juice Usage**: OpenAI introduces **pay-as-you-go credits** ($40 per 1,000 credits) for extra Codex usage on ChatGPT Plus/Pro and resets rate limits for all users, according to [this tweet](https://xcancel.com/OpenAIDevs/status/1983956900602581254).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1433254873033478196)** (8 messages🔥): 

> `MiniMax Speech 2.6, Voice Cloning, MiniMax Music 2.0, Generative Music Platform` 


- **MiniMax Speech 2.6 Clones Voices at Light Speed**: Hailuo AI launched **MiniMax Speech 2.6** featuring **less than 250 ms** real-time latency and full **voice cloning** capabilities, detailed in [this X post](https://x.com/Hailuo_AI/status/1983557055819768108).
- **MiniMax Speech Users Discuss API and roadmap**: Users are actively discussing and praising **MiniMax Speech 2.6's** future-like abilities, but are also questioning the possibility of an **OpenAI-style API**, language coverage (Malayalam), and roadmap details regarding a **voice changer** and synchronized **video generation**.
- **MiniMax Music 2.0 Serenades the AI World**: Hailuo AI unveiled **MiniMax Music 2.0**, a **generative-music platform** capable of producing **5-minute**, professional-grade songs with lifelike vocals and multi-instrument control, further information available [here](https://x.com/Hailuo_AI/status/1983964920493568296).
- **MiniMax Music 2.0 Open Sourcing in the Works?**: Enthusiastic users are inquiring about the potential for **open-sourcing** **MiniMax Music 2.0**, as well as the addition of features such as **audio uploads** and an instrumental mode.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1433190317536051310)** (19 messages🔥): 

> `Frontier LLM Training, Low-Resource Language MT, OCR for Custom Scripts` 


- **Training Frontier LLMs Solo Proves Arduous**: One member discussed the challenges of training frontier reasoning **LLMs** solo due to compute requirements, suggesting distillation or fine-tuning with tools like [Unsloth](https://github.com/unslothai/unsloth) as potential workarounds.
   - Another member shared his own [TRL implementation](https://github.com/torotoki/reasoning-minimal) of a reasoning model, describing it as an interesting but not too difficult direction for personal projects.
- **Khowar Language OCR Faces Data Scarcity**: A member is working on machine translation for **Khowar**, a low-resource language, and faces challenges extracting text from scanned books due to the language's unique Perso-Arabic script.
   - They noted that existing tools like **PyMuPDF (fitz)** and **MyPDF** misinterpret or distort glyphs, as the script includes letters not found in Arabic or Urdu.
- **Vision OCR Models can help with low-resource languages**: Members suggested creating a dataset for fine-tuning a vision OCR model, by manually labeling glyphs, and pointed to a paper with tips for building models with limited data: [[2509.14786] Tricks of the Trade for Developing Robust Arabic OCR Systems](https://arxiv.org/abs/2509.14786).
   - One added that success would require a *lot of human labor and compute*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1433180462968475699)** (49 messages🔥): 

> `Manus agent, Agent Evaluation Metrics, Extropic hardware, RWKV Understanding, Weight Decay Scaling` 


- **Manus Agent's Solid Performance Sparks Surprise**: Despite perceived solid performance, the **Manus** agent receives surprisingly little discussion, prompting curiosity as to why it's not more widely recognized.
   - One member suggested that the lack of discussion is due to the fact that *1-2% isn't enough for anyone to actually use an agent rn*.
- **Agent Benchmarks May Measure In-Distribution Success**: With agent success rates around **1-3%**, current benchmarks may primarily measure in-distribution performance rather than general agentic ability.
   - A member questioned whether agents like **Manus**, excelling in specific areas like visualization, are truly superior to agents with more evenly distributed success rates.
- **Extropic's Custom Hardware Examined**: Members discussed **Extropic's custom hardware**, designed for more efficient model execution via alternative primitives rather than encoding graphs in vectors and matrices.
   - Alternatives such as **Groq** and **Cerebras** focus on inference efficiency through larger on-chip cache to avoid fetching from memory.
- **RWKV's Intricacies Hinder Adoption**: Difficulty in understanding **RWKV's** mathematical formulation and unclear papers hinder its adoption, with one member noting, *this is consistently rwkv's biggest issue*. 
   - The member expressed that *honestly its the main reason we didnt train an RWKV world model* despite wanting to, citing the need for significant care and love in the papers.
- **Weight Decay Scaling Debated**: There is a debate around whether or not **weight decay** should scale with model size, as seen in this [discord discussion](https://discordapp.com/channels/729741769192767510/730095596861521970/1433145349928779888).
   - One paper proposes no scaling, while another recommends **sqrt(dim)** scaling, leading to a lack of consensus, as seen in [Weight Decay and Preconditioning Can Provably Recover Sparsity](https://arxiv.org/abs/2510.15262).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

gsarti: <@709147478963781692> fyi
  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1433237786437357628)** (23 messages🔥): 

> `Kimi K2, Kimi Delta Attention, Kimi-cli's D-Mail` 


- **Kimi K2 repo asked for Hacktoberfest tag**: A member asked for the **Hacktoberfest tag** to be added to the **Kimi K2 repo** on HuggingFace.
- **Kimi-Linear-48B-A3B-Base Live Now**: Members noted the release of **Kimi-Linear-48B-A3B-Base** on HuggingFace, [here](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base).
- **Kimi Delta Attention reminds members of Qwen**: A member noted that *`Kimi Delta Attention` reminds me of qwen3 next gated deltanet*.
- **Kimi-cli's D-Mail Gains Popularity**: Members linked to posts showing **Kimi-cli's D-Mail** gaining fans, specifically [this post](https://x.com/steipete/status/1983713085019046322?s=46).


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1433314379637719173)** (11 messages🔥): 

> `Manus Credits, Developer for Project` 


- **Users Request Manus Credits and Assistance**: Several members are seeking **Manus credits** and offering to pay for assistance with their projects, especially as exams approach.
   - Some users are inquiring about the availability of the **$99 credit package** and exploring alternatives like **Monica** to complete their school work.
- **Developer Seeks Project Opportunities**: A member announced availability as a developer for potential projects.
   - Another member inquired about the developer's **Manus credits** to assist with a project via direct message.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1433253169097478235)** (5 messages): 

> `Neos wrestling/boxing matches, MCP CTF in November` 


- **AI Neos Enter the Ring**: A member inquired about the timeline for **wrestling/boxing matches** involving **Neos**.
   - No definitive answer was provided, but another member expressed hope for it to happen *soon*.
- **Hack The Box Hosts AI Security CTF**: A team from **Hack The Box (HTB)** is hosting an **MCP-only CTF** on November 20th focused on **AI security**, seeking participants to test their **pen testing agents** against real scenarios.
   - The event is free to join, with more details and signup available [here](https://ctf.hackthebox.com/event/details/neurogrid-ctf-the-ultimate-ai-security-showdown-2712?utm_campaign=AI+CTF+-Oktopost&utm_content=https%3A%2F%2Fwww.linkedin.com%2Ffeed%2Fupdate%2Furn%3Ali%3Ashare%3A7386416070783479808&utm_medium=social&utm_source=LinkedIn&utm_term=).


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1433305863019298906)** (2 messages): 

> `Local Model Training, Dependency Issues on Windows, Linux vs WSL` 


- **Windows Users Face Local Model Training Troubles**: A member asked what IDE other members use to train models locally, citing dependency issues on Windows.
   - Another member suggested using **Linux** or **WSL** to avoid dependency hell.
- **Linux/WSL recommended for local model training**: A user expressed frustration with dependency management while training LLMs locally on Windows.
   - Another user suggested using **Linux** or **WSL** as a first step to resolve these issues.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433215465542516817)** (1 messages): 

> `` 


- **No Topics Discussed**: No specific research topics or discussions suitable for summarization were found in the provided message.
- **General Encouragement Expressed**: A user expressed their happiness and support to someone, relating to their story.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1433478645338472519)** (1 messages): 

> `Kimi Linear Attention, MoonshotAI` 


- **MoonshotAI Releases Kimi Linear Attention Tech Report**: [MoonshotAI released a technical report](https://github.com/MoonshotAI/Kimi-Linear/blob/master/tech_report.pdf) detailing their **Kimi Linear Attention** mechanism.
   - The report likely contains information about the architecture, performance, and implementation details of their linear attention approach; linear attention reduces quadratic compute to linear, allowing context windows to scale to extreme lengths.
- **Kimi Linear Attention: Revolutionizing Context Window Scaling**: The tech report highlights **Kimi Linear Attention's** potential to revolutionize context window scaling in transformer models.
   - *This innovation enables models to process significantly longer sequences more efficiently*, opening new possibilities for various applications like document summarization and long-form content generation.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1433215465542516817)** (1 messages): 

> `` 


- **Heartwarming Support Expressed**: A user expressed deep appreciation and support, sharing that *your story really touches me*, and conveyed happiness and personal connection by saying, *I've been there myself*.
- **Best Wishes Extended**: The user concluded with positive encouragement, simply stating, *best wishes!*.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

georgehotz: one of these days we're gonna `ruff format` tinygrad
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1433384120934858782)** (1 messages): 

> `GROUP_REDUCE errors, rangeify rewrites debugging` 


- **Nested GROUP_REDUCE Causes Headaches**: A member reported an error involving nested **GROUP_REDUCE** operations and requested assistance with debugging related to the new **rangeify** rewrites.
   - They were seeking a quick hint to avoid a *long methodical debugging* process if the cause was immediately obvious to someone familiar with the changes.
- **Seeking Hints on Rangeify Rewrites**: A member requested a hint regarding a potential cause for a **GROUP_REDUCE** error within another reduce, hoping to leverage insights from someone with expertise in the new **rangeify-related rewrites**.
   - The member aimed to avoid extensive debugging if a simple solution or known issue existed, expressing willingness to proceed with a deeper investigation otherwise.


  

---


### **Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1433241754676494488)** (1 messages): 

> `SWE-1.5, Fast Agent Models, Coding Performance` 


- **SWE-1.5 Blazes onto Windsurf**: A new fast agent model, **SWE-1.5**, was released and is now available in Windsurf, promising near-SOTA coding performance at unprecedented speeds.
   - More details are available in the [official announcement](https://x.com/cognition/status/1983662836896448756).
- **SWE-1.5 sets new speed standards**: The fast agent model **SWE-1.5** sets a new standard for speed, while also offering near-SOTA coding performance.
   - The model is now available in **Windsurf**.


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1433446982726451331)** (1 messages): 

> `Model Context Protocol RFC Status` 


- **Model Context Protocol RFC faces Delays**: The [Model Context Protocol RFC](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/269) is facing delays as stakeholders await completion of the implementation.
   - Concerns were raised about the RFC lacking a tangible implementation, hindering the ability to evaluate its practical implications.
- **No Implementation, No Evaluation**: Stakeholders express concern, stating the RFC needs a tangible implementation before its practical implications can be evaluated.
   - Without an implementation, evaluating the RFC's effectiveness is difficult.


---
id: 1214124323
title: Cognition's DeepWiki, a free encyclopedia of all GitHub repos
date: '2025-04-25T05:44:39.731046Z'
description: >-
  **Silas Alberti** of **Cognition** announced **DeepWiki**, a free encyclopedia
  of all GitHub repos providing Wikipedia-like descriptions and Devin-backed
  chatbots for public repos. **Meta** released **Perception Encoders (PE)** with
  A2.0 license, outperforming **InternVL3** and **Qwen2.5VL** on vision tasks.
  **Alibaba** launched the **Qwen Chat App** for iOS and Android. **Hugging
  Face** integrated the **Dia 1.6B SoTA** text-to-speech model via **FAL**.
  **OpenAI** expanded deep research usage with a lightweight version powered by
  **o4-mini** model, now available to free users. **Perplexity AI** updated
  their model selector with **Grok 3 Beta**, **o4-mini**, and support for models
  like **gemini 2.5 pro**, **claude 3.7**, and **gpt-4.1**. **vLLM** project
  introduced **OpenRLHF** framework for reinforcement learning with human
  feedback. **Surya OCR** alpha model supports 90+ languages and LaTeX.
  **MegaParse** open-source library was introduced for LLM-ready data formats.
companies:
  - cognition
  - meta-ai-fair
  - alibaba
  - hugging-face
  - openai
  - perplexity-ai
  - vllm
  - ''
models:
  - o4-mini
  - perception-encoder
  - qwen-2.5-vl
  - dia-1.6b
  - grok-3
  - gemini-2.5-pro
  - claude-3.7
  - gpt-4.1
topics:
  - vision
  - text-to-speech
  - reinforcement-learning
  - ocr
  - model-releases
  - model-integration
  - open-source
  - frameworks
  - chatbots
  - model-selector
people:
  - silas-alberti
  - mervenoyann
  - reach_vb
  - aravsrinivas
  - vikparuchuri
  - lioronai
---



**300k is all you need to index GitHub.**

> AI News for 4/25/2025-4/26/2025. We checked 9 subreddits, [**449** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**214** channels, and **5186** messages) for you. Estimated reading time saved (at 200wpm): **373 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


Silas Alberti of Cognition (maker of Devin) announced DeepWiki today, "a free encyclopedia of all GitHub repos. You can replace any public GitHub repo url with "https://deepwiki.com/org/repo" and you get a remarkably accurate wikipedia-like description of the library as well as a Devin-backed chatbot on how to use it.


![https://resend-attachments.s3.amazonaws.com/k0kw0QHMzZ5Urnn](https://resend-attachments.s3.amazonaws.com/k0kw0QHMzZ5Urnn)


Our intial tests with the React and Astro repos, which we are familiar with because AINews is now built on them, was VERY encouraging. Worth trying, especially for using open source code.


---

# AI Twitter Recap

**Model Releases and Updates**

- **Meta's Perception Encoders (PE)**: [@mervenoyann](https://twitter.com/mervenoyann/status/1915723394701467909) highlighted that **Meta released swiss army knives for vision with A2.0 license ❤️, including image/video encoders for vision language and spatial understanding**, noting it outperforms **InternVL3 and Qwen2.5VL**, and comes with **gigantic video and image datasets**. [@mervenoyann](https://twitter.com/mervenoyann/status/1915723399642435634) further noted that **Perception Encoder (PE) Core outperforms latest sota SigLIP2 🔥 for zero-shot image tasks**. The models and datasets can be found at [Perception Encoder models](https://twitter.com/mervenoyann/status/1915723397272654194) and [Perception LM models](https://twitter.com/mervenoyann/status/1915723397272654194).
- **Qwen Chat App Availability**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1915761990703697925) announced that the **Qwen Chat APP is now available for both iOS and Android users**, designed to assist with creativity, collaboration, and endless possibilities.
- [@reach_vb](https://twitter.com/reach_vb/status/1915830938438717777) announced that you can **run inference across 30,000+ Flux and SDXL LoRAs on the Hugging Face Hub** and you can **generate over 40+ images in less than A DOLLAR!**
- **Hugging Face and FAL Integration**: [@reach_vb](https://twitter.com/reach_vb/status/1915418386818834792) announced a **new text-to-speech model, Dia 1.6B SoTA**, which can be used directly on Hugging Face via @FAL ⚡. Users can get up to **25 generations for less than a dollar**.
- **New OpenAI Models and Deep Research**: [@OpenAI](https://twitter.com/OpenAI/status/1915505959931437178) announced they're **expanding usage of deep research for Plus, Team, and Pro users by introducing a lightweight version of deep research in order to increase current rate limits**, and also rolling out the lightweight version to Free users. [@gdb](https://twitter.com/gdb/status/1915637620731941188) confirmed **Deep Research (lightweight version) is now available in the ChatGPT free tier**. [@OpenAI](https://twitter.com/OpenAI/status/1915505961500070245) also noted the **lightweight version of deep research is powered by a version of OpenAI o4-mini** and is nearly as intelligent as the deep research people already know and love, while being significantly cheaper to serve.
- **Perplexity Model Updates**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1915819644256129424) announced that **Grok 3 Beta and o4-mini are now both available in the model selector**. [@AravSrinivas](https://twitter.com/AravSrinivas/status/1915820052571689245) noted the model selector already **supports gemini 2.5 pro, claude 3.7, perplexity sonar, gpt-4.1, deepseek r1 1776**, and looking into supporting o3 as well.  [@perplexity_ai](https://twitter.com/perplexity_ai/status/1915819619333640647) also introduced the **latest image generation model from OpenAI that supports contextual image generation and editing**.
- **vLLM for RLHF**: [@vllm_project](https://twitter.com/vllm_project/status/1915307134256091570) highlighted **OpenRLHF, a framework to use vLLM for RLHF**, driving many design and implementation of vLLM's features for RLHF, making vLLM a popular choice for many RLHF frameworks.
- **Surya OCR Model**: [@VikParuchuri](https://twitter.com/VikParuchuri/status/1915492483955384659) announced the **alpha version of the new Surya OCR model, supporting 90+ languages, LaTeX and formatting, char/word/line bboxes, ~500M non-embed params, and 10-20 pages/s**.

**Frameworks, Tools, and Datasets**

- **MegaParse for LLM-Ready Formats**: [@LiorOnAI](https://twitter.com/LiorOnAI/status/1915792212157407385) introduced **MegaParse, an open-source python library, to transform any document into LLM-ready formats**, handling PDF, Powerpoint, Word, Tables, TOC, Headers, Footers, Images.
- **LangGraph DevX**: [@hwchase17](https://twitter.com/hwchase17/status/1915208270593352002) discussed **shaping the LangGraph DevX**, asking whether prebuilt agent constructors should be classes or functions.
- **Google Agent Development Kit (ADK)**: [@omarsar0](https://twitter.com/omarsar0/status/1915402607574893052) shared a **quick guide on how to get started with ADK**, noting it's a work in progress.
- **ReflectionFlow Framework**: [@RisingSayak](https://twitter.com/RisingSayak/status/1915338106510905767) announced **ReflectionFlow**, a framework that enables text-to-image diffusion models to refine their own output through reflection. They released **GenRef-1M, a large-scale dataset consisting of (good_img, bad_img, reflection) triplets.**
- **OpenAI Codex Fund Grant Recipients**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1915524612970152376) announced the first **Codex open source fund grant recipients: vLLM, OWASP Nettacker, Pulumi, and Dagster.**
- **Spotify's ViSMaP**: [@_akhaliq](https://twitter.com/_akhaliq/status/1915703054701044209) announced that Spotify just announced **ViSMaP on Hugging Face. Unsupervised Hour-long Video Summarisation by Meta-Prompting**.
- **ByteDance's QuaDMix**: [@_akhaliq](https://twitter.com/_akhaliq/status/1915656590130036887) announced that ByteDance just released **QuaDMix on Hugging Face. Quality-Diversity Balanced Data Selection for Efficient LLM Pretraining**.
- **New queryable datasets to help researchers explore interpretable features in DeepSeek R1**: [@GoodfireAI](https://twitter.com/GoodfireAI/status/1915802798513598490) announced.
- **Trackers v2.0.0 is out**: [@skalskip92](https://twitter.com/skalskip92/status/1915439480594485363) announced combo object detectors from top model libraries with multi-object tracker of your choice. For now, they support SORT and DeepSORT; more trackers coming soon.

**Agentic Systems and Tool Use**

- **Agentic AI and Visibility**: [@weights_biases](https://twitter.com/weights_biases/status/1915498157754233092) stated that **Agentic AI without visibility = chaos**, highlighting a collaboration with @deepset_ai to bring clarity to AI workflows.
- **Meta's 3D Generative AI**: [@AIatMeta](https://twitter.com/AIatMeta/status/1915437886209745338) mentioned they are actively hiring researchers to join them and help them build tomorrow’s reality.
- **AI-Powered Assistants**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1915438278947283301) announced that **Perplexity Android app will come pre-installed on all new Motorola devices.** and, they've worked together with @moto to optimize Perplexity Assistant for the Moto Razr.  New purchases will come with 3 months of Perplexity Pro.
- **Real-time agents that are personalized and multimodal**: [@_philschmid](https://twitter.com/_philschmid/status/1915360039570739283) from Google Cloud says they showed how they think the next Generation of Agents will work, personalized real-time and Multimodal! All powered @GoogleDeepMind Gemini 2.0 Flash and the Live API.

**Interpretability and Evaluation**

- **AI Interpretability**: [@GoodfireAI](https://twitter.com/GoodfireAI/status/1915617077915967714) shared their belief that we can understand and design the mind of AI models, and that we must do so urgently, noting that we are in a race between interpretability and model intelligence.
- **Interpretability remains a fantastic place for academics to contribute**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1915546126259806590) shared his thoughts stating "The world should be investing far more into interpretability (and other forms of safety). As scale makes many parts of AI academia increasingly irrelevant".
- [@karpathy](https://twitter.com/karpathy/status/1915581920022585597) shared **a certain rhythm in AI-assisted coding (i.e. code I actually and professionally care about, contrast to vibe code)**, noting many stages are clunky and manual and aren't made explicit or super well supported yet in existing tools and that we're still very early and so much can still be done on the UI/UX of AI assisted coding.
- **LLM Evaluations**: [@clefourrier](https://twitter.com/clefourrier/status/1915339216344526896) is asking users not to **shell out 2K$ 😱 to learn about LLM evaluations** and to take a look at their free/open resources: [LLM guidebook](https://github.com/NannyML/nannyml/blob/main/guide/llm-guide.md) and [YourBench](https://github.com/seb-lgr/your-bench).

**AI Ethics and Welfare**

- **AI and Consciousness**: [@AnthropicAI](https://twitter.com/AnthropicAI/status/1915420604397744497) announced they **recently started a research program to investigate AI welfare**, asking if it is possible that AI models will have experiences of their own as they become more complex and more capable.
- [@aidan_mclau](https://twitter.com/aidan_mclau/status/1915444696090108077) says that she can't help but read the claim "15% chance models are conscious" as absurdly as "15% chance i should call this collection of atoms my computer is on a desk".

**Industry and Business**

- **Kaicathyc Green Card Denial**:  [@polynoamial](https://twitter.com/polynoamial/status/1915765141846515883) reported that **one of the best AI researchers I've worked with, @kaicathyc, was denied a U.S. green card today** and now has to leave.
- **AI and the Future of Media**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1915514295816737140) shared their perspective on the future of media, stating that billions of people are crossing the last bridge: high-quality content made by anyone ready for mass distribution.
- **Uber's Use of LangGraph**: [@LangChainAI](https://twitter.com/LangChainAI/status/1915191956810207431) highlighted **Uber's Developer Platform team uses LangGraph to build a network of agents and automate unit test generation**, building reusable components that can be adopted across the broader organization.
- [@skirano](https://twitter.com/skirano/status/1915414930536235048) says he's hiring! If you’re excited about the intersection of AI and creativity, pushing the limits of what LLMs can do, and reimagining modern web architecture, this is the place for you.

**ICLR Conference**

- **Meta at ICLR**: [@AIatMeta](https://twitter.com/AIatMeta/status/1915437886209745338) announced that Meta is at **ICLR2025 EXPO in Singapore**.
- **ICLR 2025 participants by country**: [@hardmaru](https://twitter.com/hardmaru/status/1915341552332808383) posted a chart of ICLR 2025 participants by country, and paper-count / acceptance-rate.
- [@shaneguML](https://twitter.com/shaneguML/status/1915169621042499846) shared their **AI research hot takes from ICLR Day 1: It's time to go from R to D.**

**Transportation and Infrastructure**
- [@rown](https://twitter.com/rown/status/1915607964972429522) praised **the Singapore MRT (subway) for its high frequency, full automation, safety, cleanliness, and open loop payments**.

**Humor**

- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1915847934991864079) shared a meme, saying "Reminds me of one of my fav memes https://t.co/ffVTqHIJbz"
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1915379529960300989) posted "

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Lossless LLM Compression for Inference (DF11) Research and Release

  - **[We compress any BF16 model to ~70% size during inference, while keeping the output LOSSLESS so that you can fit in more ERP context or run larger models.](https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we_compress_any_bf16_model_to_70_size_during/)** ([Score: 292, Comments: 89](https://www.reddit.com/r/LocalLLaMA/comments/1k7o89n/we_compress_any_bf16_model_to_70_size_during/)): **This post presents DF11, a Huffman-coded, blockwise weight compression format designed to reduce any BF16 model's inference memory footprint by ~30% (to ~11 bits/weight, about 70% of the original size) while maintaining lossless output. Inference benchmarks show DF11 enables previously infeasible runs on limited VRAM (e.g., Llama-3.1-405B-Instruct on 551 GB vs. 812 GB for BF16), with batch-128 throughput equivalent to BF16 (1.02x) and much faster than CPU offload (up to 38.8x); however, single-batch latency is worse (~40% slower than BF16). The method is open sourced (https://github.com/LeanModels/DFloat11), with both code and arXiv preprint (https://arxiv.org/abs/2504.11651) provided. Lossless compression avoids the unpredictable empirical regressions of lossy integer quantization, making it preferable for users requiring full determinism or maximal task coverage, and may pair with PEFT methods like LoRA but currently breaks gradients due to blockwise application.  ** Commenters recommend implementing DF11 in popular inference frameworks (llama.cpp, VLLM) for visibility and note related prior work (ezm7) that achieved similar compression for storage rather than runtime. There is acknowledgment that blockwise/chunked lower-bit palettization (e.g., 8-bit) can be nearly lossless in many use cases, but DF11's value lies where full accuracy is needed or GPU VRAM is the bottleneck, with classic quantization favored for speed and smaller memory users if some accuracy can be traded off.

    - liuliu describes a similar compression method developed by Draw Things, named ezm7, which separates compression of mantissa and exponent (using 7-bit mantissa) and achieves a comparable compression ratio (10-11 bits). However, ezm7 is currently used as a storage format, not for runtime inference. The comment also references block palettization methods—such as 8-bit block palettization and block fp8 with scale (q8_0 in gguf)—which are cited as nearly lossless in most cases, suggesting alternative practical runtime schemes.
    - ResidentPositive4122 shares an empirical benchmark comparing fp8 and bf16 precisions, stating that for r1-distill-7b/14b models over 100 math olympiad questions, both precisions yielded identical outputs across five runs at ctx_len 16,000, indicating negligible accuracy loss with fp8. In contrast, int4 quantization showed clear performance declines, reinforcing that fp8 may retain most of bf16's accuracy even at substantially reduced precision.
    - gofiend notes that the DF11 compression offers appealing accuracy-versus-inferencing tradeoffs, potentially enabling optimization targets that consider memory and inference costs directly during QAT (quantization-aware training). They suggest integration with llama.cpp or VLLM for CUDA-based runtime support, which would significantly broaden exposure and highlight the practical advantages of the method in the open-source ecosystem.

  - **[7B Reasoning Rust Coding Model with Open Dataset](https://huggingface.co/Tesslate/Tessa-Rust-T1-7B-Q8_0-GGUF)** ([Score: 134, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1k7e542/7b_reasoning_rust_coding_model_with_open_dataset/)): **A new open dataset and 7B parameter Rust-coding language model, 'Tessa-Rust-T1-7B-Q8_0-GGUF' by Tesslate, was announced, though the Hugging Face model page is temporarily inaccessible (HTTP 429). The post lacks technical details on dataset generation, validation, and evaluation processes, with commenters noting a likely synthetic dataset created via prompting a larger model and raising concerns about potential lack of data quality controls or unit tests.** Commenters emphasize the necessity of transparent dataset creation and evaluation pipelines, referencing well-documented projects like Oxen.ai's [1.5B Rust model](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) and Together.ai's [pipeline disclosure](https://www.together.ai/blog/deepcoder). There is also curiosity regarding whether specialization in a single language (Rust) leads to better algorithmic performance than broader, multi-language models.

    - There is strong skepticism about the dataset quality, as there are no details on how it was curated, validated, or tested. The primary suspicion is that the dataset was generated by prompting a larger model to answer Rust programming questions, with no transparency regarding unit tests, evaluation standards, or correctness. Comparisons are made to efforts like oxen.ai's [Qwen Coder 1.5B project](https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo) and Together.ai's [DeepCoder pipeline](https://www.together.ai/blog/deepcoder), which provided comprehensive documentation and open pipelines—practices seen as necessary for confidence in model and dataset quality.
    - A technical point of curiosity is raised about how this specialized 7B Rust model compares to larger, more generalist coding models across a range of tasks and domains. The question is specifically whether language specialization (focusing deeply on Rust) yields superior reasoning or coding results compared to models trained on a mix of programming languages, and whether exposure to more languages promotes a broader understanding of algorithms and architectural patterns.


### 2. Open-Source Local LLM & AI App Builders (Dyad)

  - **[I built a free, local open-source alternative to lovable/v0/bolt... now supporting local models!](https://v.redd.it/krhz58lqcvwe1)** ([Score: 209, Comments: 40](https://www.reddit.com/r/LocalLLaMA/comments/1k76ztc/i_built_a_free_local_opensource_alternative_to/)): **The post announces the release of Dyad ([site](https://dyad.sh/), [GitHub](https://github.com/dyad-sh/dyad)), a free, fully local, open-source AI app builder intended as a direct alternative to proprietary tools like v0, Lovable, and Bolt. Dyad differentiates itself by offering local model support through Ollama integration (see [docs](https://www.dyad.sh/docs/guides/ai-models/local-models)) and BYO API key capability for remote models (e.g., Gemini Pro 2.5, OpenAI), emphasizing enhanced developer workflow by allowing frictionless local code editing. The release targets Mac and Windows, with pending Ubuntu support via Electron configuration and ongoing feedback-driven expansion (e.g., MCP protocol support, LM Studio integration).** Technically substantive user requests in the comments include adding OpenAI API support for local models and robust MCP protocol integration, which are noted as important for feature parity and usability versus competing local IDE/coding agents.

    - A user requests implementation of proper MCP (Model Control Protocol) support, highlighting existing issues with competing solutions like Roo Code and Cline, where MCP support is described as nearly unusable. Robust MCP integration could offer a key differentiator in the project's feature set and compatibility with complex workflows.
    - There's demand for broader local model compatibility, specifically with LM Studio as a backend, which would facilitate running various open-source models locally and improve flexibility versus current deployments limited to Ollama.
    - Another user requests OpenAI API support for local models, pointing out a desired feature not present in many local model UIs: the seamless ability to swap between remote API usage and local inference for increased flexibility and cost control.

  - **[Gemma 3 fakes (and ignores) the system prompt](https://i.redd.it/xuycbwnk4zwe1.jpeg)** ([Score: 201, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1k7krlm/gemma_3_fakes_and_ignores_the_system_prompt/)): **The attached image displays Gemma 3 responding flippantly to a user pointing out that it ignored the system prompt, highlighting a key design flaw: despite the chat template supporting a 'system' role, Gemma 3 only injects system instructions by appending them to the first user message without special tokens, making it indistinguishable from ordinary user input and thus frequently ignored (see template: https://huggingface.co/google/gemma-3-27b-it/blob/main/chat_template.json). This reflects that Gemma 3 was not trained with true system prompt conditioning (as confirmed by the model card), so most interfaces simply prepend the system prompt to user input, reducing reliability for applications needing strict instruction adherence.** A notable technical debate emerges: while some users are dissatisfied with Gemma 3's lack of authentic system prompt handling, others report superior instruction following from Gemma 3 compared to larger models, particularly in high-context creative tasks, suggesting real-world variance depending on workload and prompt style.

    - Gemma 3 does not natively support system prompts; this is stated explicitly in the model documentation and card. As per the [Hugging Face implementation](https://huggingface.co/), interfaces work around this by simply prepending any system prompt to the user input, rather than utilizing an actual system role or encoding. This can lead to confusion if users expect system-level behavior, which is not present in Gemma's architecture.
    - According to the [Gemma docs](https://ai.google.dev/gemma/docs/core/prompt-structure), the model only supports 'user' and 'model' roles, not 'system.' While some users report moderate success with custom prompt templates (e.g., for roleplaying), this is not an officially supported or robust feature. Attempts to introduce system prompts may yield inconsistent results, as the underlying model wasn't instruction-tuned for this prompt type.
    - In practice, users have reported that Gemma 3 (notably the 12B and 27B variants) outperforms larger models like various 70B and 32B models in handling large amounts of context and following complex, detailed instructions for certain use cases (such as fiction writing). Despite lacking formal system prompt support, Gemma 3 demonstrates superior instruction adherence and coherence for these users than larger models, highlighting practical strengths in prompt handling even within official limitations.




## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. CivitAI Controversy and Model Hosting Alternatives

  - **[CivitAI is toast and here is why](https://www.reddit.com/r/StableDiffusion/comments/1k7p5uw/civitai_is_toast_and_here_is_why/)** ([Score: 115, Comments: 139](https://www.reddit.com/r/StableDiffusion/comments/1k7p5uw/civitai_is_toast_and_here_is_why/)): **The post analyzes the threat to CivitAI, a commercial AI image sharing site, due to mounting payment processor pressure—specifically from Visa and Mastercard—mirroring crackdowns at Patreon, Pixiv Fanbox, and DeviantArt. The author argues that CivitAI's current content moderation efforts (removing fringe fetishes, partial moderation) are likely insufficient to appease payment providers, which have enforced wide-reaching bans when underage-appearing or otherwise controversial content is involved, mainly to limit legal/commercial risk rather than out of moral concern.** Commenters highlight the centralizing power of Visa/Mastercard as de facto industry regulators, and draw parallels to OnlyFans' experience with policy changes under similar pressures. There's technical debate about whether the payment oligopoly should set standards for online platforms, with criticisms that their risk-averse approach has widespread consequences for digital content and the AI ecosystem.

    - A technical discussion points out that the removal of NSFW and other 'fringe fetish' content on platforms like CivitAI is primarily driven by payment processor policies from companies like Visa and Mastercard, which exert significant control over what is permissible by threatening to cut off financial services. This is corroborated by references to similar changes forced on platforms like OnlyFans, illustrating that once AI model sites scale beyond a certain point, regulatory and financial pressures become inescapable.
    - There is debate regarding the long-term viability of mixed-content model sharing sites. A key technical point is that hosting both SFW and NSFW (especially illegal or high-risk) models presents an unsustainable risk model for platforms, as payment processors, regulators, and investors are likely to force a binary decision: either fully embrace or completely ban adult content to maintain operational stability.
    - Commenters argue that as AI model sharing platforms grow, they will be subject to increasing scrutiny from regulators and financial backers, making a 'Wild West' approach to content moderation unsustainable. This includes not just payment processing risk but also compliance, reputation, and legal exposure when distributing controversial or potentially illegal models, especially those involving depictions of children, celebrities, or unconsenting people.

  - **[In reguards to civitai removing models](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)** ([Score: 150, Comments: 29](https://www.reddit.com/r/StableDiffusion/comments/1k7dvfb/in_reguards_to_civitai_removing_models/)): **The post provides a comprehensive list of Civitai alternatives for hosting and sharing Stable Diffusion models, including major platforms such as Tensor.art, Huggingface.co, ModelScope.cn (China-focused), Prompthero.com, Pixai.art, Seaart.ai, and civitarc.com, as well as other new or niche services like ThinkDiffusion and Stablecog. It references actively maintained curated lists (e.g., awesome-stable-diffusion, awesome-diffusion-categorized, Awesome-Video-Diffusion-Models) for up-to-date resources, covering research papers, software, community hubs, and API-ready sites, with specific attention to model retention after Civitai removals and platforms noted for community engagement, speed, or unique LoRA support (e.g., liblib.art, shakker.ai). A separate table evaluates image hosts suitable for sharing AI-generated images while preserving metadata (EXIF), with technical notes about manipulation using tools like exiftool and dedicated inspection resources such as exif.tools.** Top comments point out potential issues with Tensor.art, specifically the unauthorized reposting of LoRA models without moderation, raising concerns over trust and content integrity on that platform. Additional comments mention frequent removal of similar resource-sharing posts by Reddit filters, incentivizing users to save resource text externally.

    - One user claims that Tensor hosts stolen LoRA models and allows unauthorized reposting, alleging that moderators either ignore or don't enforce rules on model ownership and provenance. This raises concerns about trusting sites like Tensor for finding or distributing AI models, especially compared to communities with more rigorous moderation and curation policies.
    - The emergence of alternative model hosting sites in response to Civitai removing models is noted, with a link to civitarc.com shared as a recent repository. This reflects ongoing community efforts to maintain access to models even as official platforms restrict or delist content.

  - **[Civit Arc, an open database of image gen models](https://www.civitarc.com)** ([Score: 275, Comments: 81](https://www.reddit.com/r/StableDiffusion/comments/1k7po5a/civit_arc_an_open_database_of_image_gen_models/)): **Diffusion Arc (formerly Civit Arc) is a newly launched, censorship-free database specifically for AI image generation models (e.g., Stable Diffusion, Flux), intended to counter recent unexplained removals from CivitAI. The platform offers unrestricted browsing, uploading, and downloading of model files, with future technical plans including torrent upload support, model versioning, and enhanced trust signals for contributors. The website ([Diffusion Arc](https://www.civitarc.com/)) aims for eventual open-sourcing and currently sustains itself through donations, targeting maximal compatibility and accessibility for the generative AI ecosystem.** Commenters raise concerns about the platform's use of Stripe for payment processing while allowing NSFW/adult content, citing likely payment provider restrictions. There is also a suggestion to implement tags denoting models banned from other platforms, adding traceability for controversial or de-platformed content.

    - Multiple commenters emphasize the importance of torrent-based distribution for the archive, with a strong preference for magnet links over hosted torrents. Magnet links allow model hashes to be reshared independently of any specific server or host, enhancing resilience and decentralized access to models—important for content at risk of removal or censorship.
    - Suggestions are made for future platform features: automated torrent/magnet generation for all uploads (with the server always seeding), improved model versioning, and systems to allow creators to claim or relink previously posted models to a single account for proper attribution and ongoing maintenance. These features would aid both distribution and content management across evolving model databases.
    - One user raises concerns over payment processor compatibility, noting that services like Stripe often restrict usage for sites hosting NSFW/adult content, which could impact funding or donations as the platform grows. This highlights possible infrastructure risks if the project relies on mainstream payment providers.


### 2. Recent OpenAI Model Release Issues and Strategy

  - **[o3 hallucinates 33% of the time? Why isn't this bigger news?](https://www.reddit.com/r/OpenAI/comments/1k7pl37/o3_hallucinates_33_of_the_time_why_isnt_this/)** ([Score: 261, Comments: 77](https://www.reddit.com/r/OpenAI/comments/1k7pl37/o3_hallucinates_33_of_the_time_why_isnt_this/)): **OpenAI's most recent 'o3' reasoning model was reported to hallucinate on `33%` of questions in the PersonQA benchmark—a proprietary, adversarially-generated dataset targeting factual reasoning about people ([source](https://techcrunch.com/2025/04/18/openais-new-reasoning-ai-models-hallucinate-more/)). This rate more than doubles previous hallucination rates according to OpenAI's internal studies. The headline figure specifically pertains to adversarial prompts, not general usage; real-world prompt hallucination rates may be lower.** Commenters clarify that the 33% hallucination rate is observed only on a specialized adversarial evaluation set (PersonQA), not typical user prompts, highlighting that generalization to overall model performance is inaccurate.

    - OpenAI reported that GPT-4o (o3) hallucinated in 33% of cases specifically on the PersonQA benchmark, which is tailored to evaluate the model’s knowledge about people, not general-purpose prompts. This is an adversarial set, so the rate is not reflective of average use cases.
    - There is a clear technical distinction between benchmark-specific hallucination rates and real-world performance: the 33% statistic does not apply to all prompts, but to a targeted set designed to induce mistakes, and some users misinterpret this as an overall failure rate.
    - Several users highlight that o3’s tendency to hallucinate—especially compared to previous models like o1 and o3 mini—has significantly impacted its practical utility, leading some to reassess the value of subscription offerings that previously included unlimited, more reliable models.

  - **[OS model coming in June or July?](https://i.redd.it/3k1g4j7wr0xe1.jpeg)** ([Score: 146, Comments: 36](https://www.reddit.com/r/OpenAI/comments/1k7rbjm/os_model_coming_in_june_or_july/)): **The image shows Sam Altman responding cryptically ('heat waves') to a question about the release date of an upcoming open-source model, which users interpret as a summer release (June or July), referencing the song 'Heat Waves' (released June 29th). The context also hints at 'o4 mini >> o3', suggesting a significant performance uplift in the O4 mini model over O3.** Discussion centers around interpreting Altman's response—the majority of commenters agree it implies a June or July timeframe, with some speculating about a June 29th release based on the song reference. No technical benchmarks or additional details about the model are provided.

    - No technical discussion or benchmark-related details are present in these comments; the primary focus is speculation about the release date of the OS model with indirect reference to Sam Altman but no details on models or performance.
    - There's a query about the meaning of "siillsmaxxing" and a question about why Sam Altman might recommend "o4 over o3," which hints at model progression (e.g., from OpenAI GPT-3 to GPT-4), but the thread does not elaborate on technical differences between versions or provide implementation analysis.

  - **[Made a ChatGPT "cheat map" to stop guessing models, tools, prompts (sharing it here too)](https://i.redd.it/r0f7hax9jzwe1.png)** ([Score: 614, Comments: 49](https://www.reddit.com/r/ChatGPT/comments/1k7lbhq/made_a_chatgpt_cheat_map_to_stop_guessing_models/)): **The linked image ([ChatGPT Cheat Map](https://i.redd.it/r0f7hax9jzwe1.png)) is a practical, user-focused flowchart that guides ChatGPT users through three key steps: (1) choosing a model (default GPT-4o or o3 for tasks requiring more complex reasoning), (2) activating features/tooling (such as Search, Deep Research, Canvas, or Create Image), and (3) applying a prompting formula for optimal results. This visual aid is designed to assist everyday users (not API or advanced users) in making quick decisions that maximize ChatGPT’s utility without guessing which workflow to use. It consolidates best practices from OpenAI documentation and community experimentation to streamline user experience and improve output quality.** Technical discussions in the comments query the exclusion of other available models (like 4.5, o4 mini variants), clarify feature availability differences between the app and website interfaces, and seek concrete definitions for what constitutes a 'complex' task suitable for the o3 model.

    - There's a question regarding differences between models like GPT-4.5, o4 mini, o4 mini high, GPT-4, and 4o mini, indicating confusion in model features and performance. No direct benchmarks or comparisons are cited, but the inquiry signals a need for more rigorous documentation on capabilities across the many GPT sub-variants.
    - A technical note clarifies that certain features such as "Canvas" or "Create image" are not available in the ChatGPT mobile app interface; instead, users must invoke them with textual prompts or use the "/" command to access feature menus, pointing to implementation differences across platforms and the importance of interface-aware prompting.
    - A user asks whether "deep research quality" differs significantly between the GPT-4o and o3 models, highlighting ongoing uncertainty about the performance or qualitative distinctions between these models, but no concrete data or evaluations are provided in the discussion.


### 3. Frontier Model Benchmarks and Human vs AI Reasoning

  - **[New reasoning benchmark where expert humans are still outperforming cutting-edge LLMs](https://i.redd.it/a6awqhrhmtwe1.jpeg)** ([Score: 128, Comments: 59](https://www.reddit.com/r/singularity/comments/1k7f9dd/new_reasoning_benchmark_where_expert_humans_are/)): **The image illustrates results from the new PHYBench benchmark, which tests physical reasoning using 500 real-world physics problems. Human experts significantly outperform the latest LLMs in both accuracy and EED (Explanation Edit Distance) score, highlighting a persistent gap in LLMs’ ability to perform deep or spatial physical reasoning. This supports claims that current LLMs lack mechanisms for diagrammatic or spatial reasoning required for many physics tasks.** Commenters concur that genuine physical reasoning, especially spatial/diagrammatic thinking, is critical and currently lacking in LLMs. There's also discussion about the value of moving towards benchmarks that evaluate real-world, embodied performance, given the increasing expense of running synthetic benchmarks.

    - Several comments highlight that current LLMs lack spatial or diagrammatic reasoning capabilities crucial to expert-level human problem solving, especially in domains like physics and code architecture. The inability to generate or interpret visual representations, such as accurately displaying a clock or handling visual/spatial tasks, is cited as a major technical limitation for existing models.
    - Another technical point discusses the rapid improvement in recent models like Gemini 2.5 Pro, which reportedly jumped from `25-35` to `50` in benchmark scores within a few months. The discussion speculates that if this linear or near-exponential rate continues, LLMs may surpass expert humans on this benchmark within months, suggesting a significant acceleration in model capabilities.
    - There is also a call for shifting benchmarks toward measuring practical, real-life task performance instead of focusing solely on areas where LLMs still lag significantly. Commenters note that as large-scale benchmarks have become expensive to run anyway, real-world evaluations could now become more feasible and valuable for tracking true model progress.

  - **[New Paper: AI Vision is Becoming Fundamentally Different From Ours](https://www.reddit.com/r/singularity/comments/1k7dwld/new_paper_ai_vision_is_becoming_fundamentally/)** ([Score: 167, Comments: 39](https://www.reddit.com/r/singularity/comments/1k7dwld/new_paper_ai_vision_is_becoming_fundamentally/)): **The arXiv paper ([2504.16940](https://arxiv.org/pdf/2504.16940)) demonstrates that as state-of-the-art deep neural networks (DNNs)—including LLMs with vision such as GPT-4o, Claude 3, and Gemini 2—improve at visual tasks, their internal processing strategies increasingly diverge from those of primate (including human) vision, reversing earlier findings of alignment. The research attributes this divergence to DNNs leveraging non-biological features and strategies for performance, meaning high accuracy on benchmarks no longer equates to biological plausibility. The authors argue that achieving human-like vision in AI will require dynamic, temporally structured, multimodal, and embodied training—e.g., with synthetic environments generated by NeRFs or Gaussian Splatting—rather than reliance on large-scale, static datasets.** Commenters emphasize that AI should learn from dynamic, life-like experiences rather than static data and note that evolution produced a variety of vision systems, suggesting that optimality for AI may fundamentally differ from biological vision given fewer constraints (e.g., energy costs or field of view limitations in animals).

    - VallenValiant draws a comparison between AI vision and biological evolution, noting that multiple forms of eyes evolved independently among animals. They argue that AI may not adopt 'human-like' optimal vision, explaining that robotic or digital vision systems can potentially achieve perspectives (such as 360-degree or multi-directional vision) physically impossible for animals due to biological and metabolic constraints. This highlights the fundamentally divergent paths for sensory optimization between biological and AI systems, pointing towards the potential for AI vision to develop capabilities not limited by human biases or evolutionary pressures.
    - liqui_date_me points out that the concept of AI and machine vision differing fundamentally from human perception is not a novel revelation, referencing adversarial examples in deep learning. These adversarial samples exploit the non-human way in which AI interprets visual data, illustrating how AI can be easily fooled by changes imperceptible to humans, and suggesting that the divergence between human and machine vision has been well-recognized within the machine learning community since the early days of deep learning research.

  - **[The Ultimate Turing Test for AGI is MMO games](https://www.reddit.com/r/singularity/comments/1k7m5ui/the_ultimate_turing_test_for_agi_is_mmo_games/)** ([Score: 124, Comments: 53](https://www.reddit.com/r/singularity/comments/1k7m5ui/the_ultimate_turing_test_for_agi_is_mmo_games/)): **The post argues that current LLM/AI benchmarks (e.g. static datasets like MMLU/ImageNet) fail to test genuine AGI capabilities, proposing that open-world MMO games present a significantly more rigorous benchmark by requiring simultaneous dynamic visual reasoning, raw sensory perception (pixels, audio, text), meta-learning under evolving strategies, adversarial robustness, and zero-shot learning without pretraining. The test demands an agent operate entirely as a human would—interpreting the world from raw signals, adapting to real-time gameplay developments, and learning novel strategies unaided—essentially aligning with the human experience curve in chaotic, multi-agent digital environments.** One commenter agrees that games, especially more complex sandbox games like Dwarf Fortress, are superior for AI evaluation versus current benchmarks; another notes that implementing such a test is less about AI and more a monumental software engineering challenge, as current agents would require extensive environment-specific code to even participate. A third notes that this approach mirrors existing robotics simulation training, where agents learn behaviors in complex, multi-agent environments, hinting at existing precedent but the technical difficulty in scaling to the richness of MMOs.

    - Game environments, especially complex ones like MMO games, provide valuable and challenging testbeds for AI research due to the need for agents to learn, retain, and apply novel information for problem-solving, with real-time adaptation and multi-agent interaction. Minecraft is cited as a starting point, but Dwarf Fortress is mentioned as a particularly rich environment for AI because of its intricate simulation and emergent complexity. 
    - Several comments highlight that implementing AI in an MMO environment often involves substantial environment-specific coding. This could bias the "test" toward the programmers' ability to create convincing scripts and agents, rather than genuinely evaluating AI generalization capability. This criticism suggests such setups might measure coder ingenuity more than general AI performance. 
    - There is ongoing work in simulating multi-agent environments, such as training robot AIs in simulations with many agents, where they must learn behaviors like obstacle avoidance and recovery from disturbances. EVE Online's public API is suggested as a platform for testing AI agents in a rich MMO context, and there's a reference to past work (e.g., Claude plays Pokemon) demonstrating LLM and AI integration with game environments.


---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview

**Theme 1. Model Updates and Performance Shifts**

- **O3 Pumps Out Bigger Code!**: Users are reporting **O3** now outputs larger code files reaching **700-1000 lines**, doubling the previous **300-line** limit. This boost also extends to **O4 Mini High**, enhancing its capabilities.
- **Sunstrike Enters Arena, Sparks Speculation**: The new **Sunstrike** model has been added to the arena with initial performance *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*. Early indications suggest **Sunstrike** might be a Google model, facing challenges that even **2.5 Pro** only solves **25%** of the time.
- **GLM-4 Enters the HF Arena**: Members are discussing the new **GLM 4 models** with the model uploaded to HF [here](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF). It was highlighted that [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414) *tops half the benchmarks even against DeepSeek R1*.

**Theme 2. Hardware and Optimization Techniques**

- **LM Studio Gets RTX 50-Series Boost**: **LM Studio 0.3.15** introduces support for **NVIDIA RTX 50-series** (CUDA 12.8) and can be updated via the [LM Studio download](https://lmstudio.ai/download). This update also activates **GLM-4** in llama.cpp and MLX, features a new system prompt editor UI, and integrates the `tool_choice` parameter in the OpenAI-like API, detailed in the [full release notes](https://lmstudio.ai/blog/lmstudio-v0.3.15).
- **CUDA-based RTX 3060 smashes Intel B580**: Members compared the **RTX 3060 12GB** with the **Intel B580 12GB** for AI tasks, favoring the **3060** for superior **CUDA** support. The consensus was that *Everything in the AI world is built around Cuda* making Nvidia's cards superior for AI development and experimentation.
- **TACQ Compresses LLMs Down to 2-bit**: A [research paper](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/) introduces **TACQ**, a task-aware quantization approach for **LLMs** that retains high accuracy at **2-bit precision**. TACQ achieves this by using a calibration dataset to decide which weights can be compressed more, preserving critical weight circuits.

**Theme 3. AI Frameworks and Tooling Updates**

- **Aider Adds Language-Based Sorting**: **Aider** now sorts by programming language, improving code organization and accessibility. Check out the [UI screenshot](https://cdn.discordapp.com/attachments/1131200896827654144/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&). This was implemented via a new sorting functionality by language.
- **`FunctionAgent` now offers timeout functionality**: The [`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) class lacks a direct timeout parameter, but setting a timeout per LLM call on the LLM object is possible. Newer agent classes like `FunctionAgent` directly support timeout via the `request_timeout` parameter, and a code snippet example was given using `Ollama` and setting `request_timeout` to 360 seconds.
- **Kubernetes MCP Gets Implementation**: A member announced they built a new kubernetes MCP [based on the k8s API](https://github.com/StacklokLabs/mkp) to make it more general and flexible. It is meant as an alternative to the first MCP Server for fetching GitHub repo, allowing **AI** to use repo code as references.

**Theme 4. AI Research and Fundamental Concepts**

- **LeCun Plots Machine Intelligence Path**: A member proposed a pathway to AGI using **past memories**, updated knowledge graphs, and **imagination** to generate multimedia content. Another member suggested reading **Yann LeCun's** *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf) highlighting latent space transformations as sufficient for intelligence without generative aspects.
- **Agent Building Discussions are Booming**: [AnthropicAI published Building Effective Agents](https://www.anthropic.com/), and **dexhorthy** went viral with his [12 Factor Agents](https://www.12factor.net/), and [OpenAI released A Practical Guide To Building Agents](https://platform.openai.com/docs/guides/function-calling). The community is having **productive dialogue** on what agents are, and how to build them.
- **SimpleStories Dataset Eclipses TinyStories**: A new replacement [dataset](https://huggingface.co/datasets/lennart-finke/SimpleStories), tokenizer, and [model suite](https://huggingface.co/SimpleStories) called **SimpleStories** has been released. A [community doc](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing) is available to help researchers get started.

**Theme 5. Industry News and Platform Challenges**

- **Nous taps Paradigm's $50M**: Nous Research [announced a **$50M** investment from Paradigm](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/), a crypto venture capital firm, to advance its **AI research and development** efforts. This news follows Nous's initiative to launch **Psyche**, a [distributed training project](https://nousresearch.com/nous-psyche/) utilizing the **Solana network** for coordination, with a [Psyche Discord](https://discord.gg/peqZyPRd) available for those interested in the crypto aspects.
- **Gemini's Generosity: Free Tiers Trigger Throttling Troubles!**: Due to high demand, [Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) has stricter usage limits: **1 request per minute** and a total of **1000 requests per day**, and the `:free` model alias will hit the standard variant to ensure existing code continues to work. Members discussed the rate limits on the free tier of **Gemini 2.5 Pro**, noting that OpenRouter would announce this change, and one member stated that demand is exceeding supply.
- **Credits Crisis: OpenRouter Account Emptied!**: A member reported an incident where their **OpenRouter credits were depleted** due to an exploit involving infinite URL generation. The malicious activity was traced to a proposed solution architecture creating a URL of approximately **3000 characters**.


---

# PART 1: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Launches New AI Suite**: Perplexity shipped **iOS Voice Assistant**, **GPT Image Generation**, **Grok 3 Beta**, and **OpenAI o4-mini** this week, as detailed in their [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th).
   - Users noted these features enhance search and interaction capabilities within the Perplexity ecosystem.
- **Discord Roles Get a Makeover**: Members report that **Kes** is reworking the [linking system for discord roles](https://discord.com/channels/link/to/roles).
   - However, specific details on the rework's benefits or changes remain unclear.
- **GPT-4o Impresses with Text-Image Prowess**: Members discussed how the [GPT-4o Image Gen's language model](https://link.to/model) produces superior **text images**, though some still lean towards Midjourney for overall aesthetic quality.
   - One member claimed that *MJ specifically wants people to be able to prompt a single word like cat and get really beautiful not basic stuff back* suggesting architectural differences in prioritizing aesthetics.
- **DeepSearch's Reddit Snafu**: A member jokingly awarded **Perplexity** the *"Perplexity broke again award"* after [an AI image](https://link.to/img) generated an unexpected result when searching Reddit data.
   - Another member questioned the reasoning behind the outcome, speculating it might be based on recommendations from the early 1900s.
- **Grok 3 Mini's Reasoning Capabilities Debated**: Members debated why Perplexity opted for a normal **Grok 3** model over the reasoning version in some applications.
   - A team member clarified that [Grok 3 excels at reading, understanding, and answering questions better than the reasoning Grok 3 mini](https://link.to/blogpost), making it a more suitable choice for specific tasks.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous taps Paradigm's $50M**: Nous Research [announced a **$50M** investment from Paradigm](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/), a crypto venture capital firm, to advance its **AI research and development** efforts.
   - This news follows Nous's initiative to launch **Psyche**, a [distributed training project](https://nousresearch.com/nous-psyche/) utilizing the **Solana network** for coordination, with a [Psyche Discord](https://discord.gg/peqZyPRd) available for those interested in the crypto aspects.
- **TACQ Compresses LLMs Down to 2-bit**: A [research paper](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/) introduces **TACQ**, a task-aware quantization approach for **LLMs** that retains high accuracy at **2-bit precision**.
   - TACQ achieves this by using a calibration dataset to decide which weights can be compressed more, preserving critical weight circuits.
- **CPU-Only Fine-Tuning Unleashed**: A user shared a [step-by-step guide](https://medium.com/@contact_30070/step-by-step-guide-for-fine-tuning-your-llm-with-llama-factory-using-the-cpu-only-96b2fc6a80b0) for **fine-tuning LLMs with LoRA using LLaMa Factory on CPU only**.
   - The guide covers everything from installing **torch** to loading the custom **GGUF** in **LM Studio**, and is accompanied by a [YouTube video](https://www.youtube.com/watch?v=1bgL4b7VT8M).
- **Seeking Strong TRL Reward Functions**: A member inquired about finding battle-tested **TRL-compatible GRPO reward functions**.
   - Another member suggested looking at the [Open-R1 rewards](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py) from Hugging Face as a starting point.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **O3 Pumps Out Bigger Code!**: Users are reporting **O3** now outputs larger code files reaching **700-1000 lines**, doubling the previous **300-line** limit.
   - This boost also extends to **O4 Mini High**, enhancing its capabilities.
- **LeCun Plots Machine Intelligence Path**: A member proposed a pathway to AGI using **past memories**, updated knowledge graphs, and **imagination** to generate multimedia content.
   - Another member suggested reading **Yann LeCun's** *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf) highlighting latent space transformations as sufficient for intelligence without generative aspects.
- **Sunstrike Enters Arena, Sparks Speculation**: The new **Sunstrike** model has been added to the arena with initial performance *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*.
   - Early indications suggest **Sunstrike** might be a Google model, facing challenges that even **2.5 Pro** only solves **25%** of the time.
- **GPT-4o's Multimodal Nature Debated**: Members are debating whether **GPT-4o** is truly natively multimodal, focusing on its reliance on tools versus inherent capabilities.
   - Arguments suggest **GPT-4o** may not be fully multimodal because it makes tool calls with specific natural language queries and generates completely new images but it *used to call Dali-E but now it generates images natively*.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus Double Credits: Month-Long Bonanza**: Manus offers **double credits** for the first month to new and existing subscribers: Starter plan gets **+3,900 credits**, Pro plan gets **+19,900 credits**.
   - This is a *time-limited offer* with no fixed end date, and **free users** get double credits for the **first subscription** upon upgrading to a paid plan.
- **Manus-Made Flask Clone Hits GitHub**: A member created a [Discord flask clone](https://github.com/johnnyshelby123/discordflaskclone) and offered it up to be improved by Manus.
   - The original creator emphasized that **Manus did 100% of the work**, even as another member offered to recode it in **Node.js**.
- **Prompt Engineering for Project Generation**: A member shared [their approach to prompt engineering](https://github.com/justinlietz93/hierarchical_reasoning_generator/tree/main/hierarchical_planner/persona_builder/personas) to generate an entire project with minimal code.
   - They pointed to hierarchical systematic and dynamic gateways to adjust the persona or prompt based on the initial plan generated, with the ability to generate an entire project in one prompt using less than 1000 lines of code.
- **Users Yearn for Flat-Fee Utopia**: Users express frustration with Manus's **credit-based system**, preferring a flat monthly fee for unlimited usage.
   - One user switched back to **ChatGPT 4.0** after losing paid credits due to mistakes made by Manus.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora's Pricing Exposed**: A user inquired about **Sora's** image creation limitations and discovered new options for purchasing video generation and concurrent image creations at a higher cost.
   - The user initially faced a restriction of one image at a time and prompts to *upgrade* to a plan they already had.
- **RAG, LLM, Word2Vec Combo Chatter**: A member sparked a discussion about combining **RAG, LLM, and Word2Vec**, noting that many mainland Chinese companies are developing software with a *Local Knowledge Base* feature.
   - Concerns were raised about **LLM's grasp of contextual length**, potentially requiring further paragraph-level processing of the original document, leading to questions about the technology's future.
- **Math Model Finetune Frenzy**: One user suggested **fine-tuning a smaller AI model specifically for university-level math problems**, instead of relying on large models trained on everything.
   - The idea is to create a model *just large enough to handle language to understand a problem* and integrate a built-in calculator using Python, optimizing for speed and cost-effectiveness.
- **Gemini Advanced Video Limits**: A user pointed out the inconsistent **video generation limits in Gemini Advanced**, noting that it provides only **10 videos per day** compared to **4 per day** for free in AI Studio.
   - Despite the limit, they believe **Gemini's Veo 2** produces better output quality than **Sora** for generic content, but is too restrictive and has a delayed refusal problem.
- **Deep Research Tiers Detailed**: **OpenAI Pro** users get **120 original deep research** and **120 more light weight deep research** according to [this tweet](https://x.com/OpenAI/status/1915505961500070245).
   - Users are seeing **Deep Research full**, **Deep Research lite**, **Deep Research mini** (just normal o3), and **Deep Research nano** (just normal o4-mini).



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth's Dynamic v2.0 GGUFs drop**: Unsloth released new **Dynamic v2.0 GGUFs** [here](https://x.com/UnslothAI/status/1915476692786962441).
   - The *official* multi GPU support is coming soon, but *needs a lot of tests and stuff to confirm its stable*.
- **GLM-4 Enters the HF Arena**: Members are discussing the new **GLM 4 models** with the model uploaded to HF [here](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF).
   - It was highlighted that [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414) *tops half the benchmarks even against DeepSeek R1*.
- **Tesslate Model Sparks Rust CodeFIM**: A member requested a quant of **Tesslate**'s new **Rust 7B model**, prompting the creation of a Rust-only version of their codefim [dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data).
   - Following this, a question was raised: *Has anyone compiled phi-4 to gguf yet?*.
- **MoonshotAI Speaks with Kimi Audio**: **MoonshotAI** released [Kimi Audio](https://github.com/MoonshotAI/Kimi-Audio), with the paper available on [arXiv](https://arxiv.org/abs/2504.09858).
   - The new model can process audio, according to the given paper.
- **Gemma-3-4b-it Fine-Tuning Faces Hurdles**: A user reported a `RuntimeError` on Google Colab during fine-tuning of the `gemma-3-4b-it` model via the Unsloth notebook, encountering a *datatype mismatch between float and c10::Half*.
   - The error remained even after reverting all changes, pointing to a potential environment issue.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama-3.2 Checkpoint Export Proves Difficult**: A member is seeking a script to export **Llama-3.2 checkpoints** to **ONNX** with **KV cache** and **TensorRT** support.
   - They reported *little to no benefit* when playing with default PyTorch inits vs architecture-specific inits, and asks if it's even possible to have nvidia drivers in TCC mode on Windows or WSL.
- **Colaboratory Band-Aid on Digital Divide**: A member suggested using [Google Colaboratory](https://colab.google) for light data analytics to combat the rising cost of computer specs.
   - They noted that it's limited to models around **1B parameters**, like **GPT-2**.
- **Modded RTX 4090s Boast 48GB Memory**: Members discussed the popularity of **modded 48GB RTX 4090s**, particularly in China, linking to [a teardown article](https://www.tomshardware.com/pc-components/gpus/blower-style-rtx-4090-48gb-teardown-reveals-dual-sided-memory-configuration-pcb-design-echoes-the-rtx-3090) from Tom's Hardware.
   - These cards generally sell for around *$3,500*, though information regarding a **5090** is unconfirmed due to tight supplies.
- **Flash Linear Attention Models**: Members discussed the need for an *easiest linear transformer model or equivalent to train on tiny datasets*.
   - Another member suggested models from the **Flash Linear Attention repo**, such as **Gated Linear Attention** or **Gated DeltaNet**, trained with their **Flame trainer**.
- **SimpleStories Dataset Eclipses TinyStories**: A new replacement [dataset](https://huggingface.co/datasets/lennart-finke/SimpleStories), tokenizer, and [model suite](https://huggingface.co/SimpleStories) called **SimpleStories** has been released.
   - A [community doc](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing) is available to help researchers get started.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider** Adds Language-Based Sorting**: **Aider** now sorts by programming language, improving code organization and accessibility. [Check out the UI screenshot](https://cdn.discordapp.com/attachments/1131200896827654144/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&).
   - This was implemented via a new sorting functionality by language.
- **Grok-3-Mini** Undercuts the Competition**: **Grok 3 mini** is almost half the price of **Deepseek V3** via OpenRouter, boasting comparable reasoning performance. See the [performance comparison](https://cdn.discordapp.com/attachments/1131200896827654144/1365185829596696636/image.png?ex=680d0ca1&is=680bbb21&hm=f5228d04256f815b37940355080c80e2cdaa6d86a0f6b37759d6d53b59c8be26&).
   - Members observed that it may be worth switching due to the price.
- **Gemini Pro 2.5** and **Claude 3.7** tag team**: A member is experimenting with using **Gemini Pro 2.5** as the architect model and **Claude 3.7** as the editor model in **Aider**, taking inspiration from videos featuring **Claude Code** and **Gemini** via its MCP interface.
   - However, it was pointed out that **Gemini** tends to alter unrelated code, making code review difficult, and that it may be better to use DeepSeek V3.
- **Gemini's** commentary is Yapping away**: Users seek to reduce **Gemini's** verbose comments, suggesting the use of conventions to remove unnecessary comments when using **Gemini** in architect mode with **GPT-4.1** as the editor.
   - One user noted that *Gemini stays yapping in the comments worse than ive ever seen* and is looking for better solutions.
- **Aider** loves the CLI**: Members use **Aider** 90% of the time in the terminal (Warp.dev), and the rest in a sidebar in VS Code, without IDE plugins.
   - This suggests that most users are not using IDE plugins.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Framepack gets spotlight for Fine-tuning Accessibility**: Members highlighted **Framepack's** accessible fine-tuning capabilities, with **Unsloth's** endorsement suggesting a collaboration, particularly around frequent weight updates every **8 seconds**.
   - The swift weight updates aims at stress-testing and continuous improvement.
- **Multimodal LLMs March toward Media Mastery**: The community explored multimodal LLMs like [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), and [SmolVLM2](https://huggingface.co/blog/smolvlm2) for interpreting video footage and audio, even with [Nvidia's DAM-3B-Video](https://huggingface.co/nvidia/DAM-3B-Video) and a [YouTube audio transcriber](https://huggingface.co/spaces?sort=trending&search=youtube).
   - The exploration focused on summarizing video and transcribing audio content.
- **Inference API Delivers Trauma for a Product**: A member reported encountering disturbing images via the **Hugging Face Inference API**, prompting a need to contact **HF staff** for resolution.
   - It was clarified that the **Stable Diffusion** team is under **Stability AI**, separate from Hugging Face.
- **SmolAgents Asks Can it power Gemma Locally?**: A user inquired about using **smolagents with Gemma3** or other local LLMs, noting the Deeplearning.ai course uses **Tavily** and **OpenAI**.
   - The question hits on how well local models integrate with the agent framework.
- **Agents Course Certificate requires HF Username**: Users identified the HF username as the **credential ID** and the [dataset link](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence) as the **URL**, as the permanent verification method.
   - This is a workaround for a download link that led to a temporary folder file instead of a CDN URL.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **AI Learns Reward Shaping**: A paper on [offline training with simulated rollouts](https://arxiv.org/abs/2502.05244) from a current world model was shared, aiming to reduce interactions with the environment.
   - Humorous suggestions for reward shaping included *thumbs up for positive reward and thumbs down for negative reward*.
- **Symbolic World Models Emerge**: A member shared an [arXiv link](https://arxiv.org/abs/2503.20124) highlighting the advantages of using programs as world models.
   - They emphasized the compositional nature, allowing easier modification compared to neural representations.
- **LLMs Fuse into World Models**: One member posited that **LLMs' world models** are *fused into their weights*, contrasting this with discrete or structured models.
   - Another member introduced a formula `(A, W) = G(z)` to represent generating new agents/models/worlds on demand.
- **Generative AI Gets a Formula**: A general formula `(A, W) = G(z)` was introduced for **AI model generation**, where **A** is the model's architecture, **W** is the weights/parameters, **G** is the generator, and **z** is the semantic seed.
   - The member suggested this encompasses hypernetworks and could lead to significant compression and generativity, comparing it to *storing DNA rather than the whole human*.
- **DeepMind Tunes Up Music AI**: Google DeepMind unveiled [Music AI Sandbox](https://deepmind.google/discover/blog/music-ai-sandbox-now-with-new-features-and-broader-access/) with new features and broader access.
   - The blogpost highlights various updates to this music AI sandbox.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini's Generosity: Free Tiers Trigger Throttling Troubles!**: Due to high demand, [Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) has stricter usage limits: **1 request per minute** and a total of **1000 requests per day**, and the `:free` model alias will hit the standard variant to ensure existing code continues to work.
   - Members discussed the rate limits on the free tier of **Gemini 2.5 Pro**, noting that OpenRouter would announce this change, and one member stated that demand is exceeding supply.
- **Custom Rate Limit Error Messages on the Horizon**: Users are receiving error messages directly from the **Gemini API** when hitting the rate limit, causing confusion since the messages do not specify whether the limit is global or user-specific.
   - The OpenRouter team is considering adding a custom message to clarify the source of the rate limit error for users.
- **Baidu Models Beckon, Will OpenRouter Bow?**: A member inquired about the possibility of adding [Baidu models](https://x.com/Baidu_Inc/status/1915603080336597310) to OpenRouter.
   - Another member pointed out the existing availability of **DeepSeek**, followed by a discussion of **GLM-4-0414** hosted by other providers.
- **O3 Odyssey: OpenRouter's Verification Venture!**: A member inquired whether OpenRouter will support **OpenAI's o3 and other advanced models** without verification in the future.
   - Another member mentioned that **OpenAI** might drop this requirement in the future.
- **Credits Crisis: OpenRouter Account Emptied!**: A member reported an incident where their **OpenRouter credits were depleted** due to an exploit involving infinite URL generation.
   - The malicious activity was traced to a proposed solution architecture creating a URL of approximately **3000 characters**.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **ESP32 gets RTSP firmware package**: A member is trying to create a working **RTSP firmware package** for outbound streaming on the **ESP32** and asked for suggestions, citing the need for streaming.
   - Other members recommended these two **GitHub** repos, [esp32cam-rtsp](https://github.com/rzeldent/esp32cam-rtsp) and [ESP32-RTSPServer](https://github.com/rjsachse/ESP32-RTSPServer), and suggested using **PlatformIO with AI** for development.
- **Gemini 2.5 and Sonnet 3.7 get questioned**: Members are reporting that non-max versions of **Gemini 2.5** and **Sonnet 3.7** seem dumber and hallucinate more compared to when used via the command line interface (**CLI**).
   - One member speculates that **Cursor** might be changing the system instructions or prompts, but another countered with *"Gemini issues should be fixed, was a Google problem"*
- **Cursor Billing sorted**: A member inquired about immediate payment options for an unpaid invoice due to a bank confirmation delay when using **Cursor**.
   - Another member, likely a **Cursor** team member, offered direct assistance via **DM** to resolve the billing issue quickly.
- **Community Wants Model Evaluations Open Sourced**: A member suggested that **Cursor** should open source evaluations for different models, emphasizing that model implementation matters more than raw benchmarks.
   - They proposed tool-based evals, specifically asking if there was a Cursor-specific evaluation tool, as *"I find the way a model is implemented matters more than just raw benchmarks."



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Interop Capabilities Incoming!**: Members are anticipating the release of additional **Python interop capabilities** for Mojo as previously mentioned in the presentation.
   - The team has not provided any timeline for a release date.
- **Mac GPU Support Possibly Coming This Summer**: **Mac GPU support** is tentatively slated for *summer* according to .duck_tape, but users should take it with a grain of salt.
   - Members await official confirmation and details on compatibility and performance.
- **Rust's uom Crate Masterfully Handles Unit Operations**: The `uom` crate in Rust allows for operations mixing different units like **Energy**, **Force**, and **Length** without errors.
   - Members demonstrate adding `Energy` to the product of `Force` and `Length`, similar to treating `Byte` as `UInt8`.
- **Nm vs Joules Debate Breaks Out**: A discussion emerged about whether **Newton-meters (Nm)** can be treated the same as **Joules (J)**, with one side arguing that physics distinguishes them because torque is a vector quantity, while energy is a scalar.
   - Referencing [BIPM](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf), it was highlighted that, *even though torque has the same dimension as energy (SI unit joule), the joule is never used for expressing torque*.
- **`QuantityKind` Tag May Resolve Quantity Ambiguity**: A **QuantityKind** tag was proposed to differentiate between similar units, such as **Nm** and **Joules**, at the type level.
   - The implementation can be `Q[Vec[dtype, 1], N * M]` vs. `Q[Scalar[dtype], N * M]` where you can't add them.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Gets RTX 50-Series Boost**: **LM Studio 0.3.15** introduces support for **NVIDIA RTX 50-series** (CUDA 12.8) and can be updated via the [LM Studio download](https://lmstudio.ai/download).
   - This update also activates **GLM-4** in llama.cpp and MLX, features a new system prompt editor UI, and integrates the `tool_choice` parameter in the OpenAI-like API, detailed in the [full release notes](https://lmstudio.ai/blog/lmstudio-v0.3.15).
- **LM Studio Community Presets in Preview**: Users are exploring the new **Community Shared Presets** feature in **LM Studio**, which is currently in *"Preview"* and lacks a discover tab for browsing.
   - Confusion arose around locating and downloading presets, with users expecting a *discover -> presets* navigation, one user stated that it *makes me question the intent*.
- **Dataset Prep Stumps LLM Character**: A user is trying to train an LLM to emulate a game character using a limited dataset (**10k tokens**) and needs guidance on loading the data into **LM Studio**.
   - The user is unsure about the necessity of converting the data to **GGUF** format and the subsequent steps for testing the model after conversion.
- **CUDA-based RTX 3060 smashes Intel B580**: Members compared the **RTX 3060 12GB** with the **Intel B580 12GB** for AI tasks, favoring the **3060** for superior **CUDA** support.
   - The consensus was that *Everything in the AI world is built around Cuda* making Nvidia's cards superior for AI development and experimentation.
- **OpenVINO speeds past llama.cpp**: A member showcased their [OpenArc project](https://github.com/SearchSavior/OpenArc), indicating that **OpenVINO** significantly outperforms **llama.cpp** on **CPUs**.
   - According to the member, *anecdotally the difference is HUGE, especially ttftR*, with implemented vision for **Qwen2-VL**, **Qwen2.5-VL** and **Gemma3**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Pantone-Scale Mode Proposed**: A member humorously suggested renaming *grayscale mode* to *pantone-scale* in the [general channel](https://discord.com/channels/1189498204333543425/1189498205101109300).
   - No immediate decision was made, but the suggestion was well received.
- **FP4 Scaling Needs Elementwise Multiplication**: After running **FP4** on a **5090**, a member found that they need to fuse scaling elementwise multiplication into their **matmul kernel**.
   - Another member noted the importance of a good algorithm for accurate scales in symmetric quantization, cautioning that a simple min-max approach could yield poor quality.
- **cuda-python Library Evaluated**: A member inquired about the quality and functionality parity of [NVIDIA's cuda-python library](https://github.com/NVIDIA/cuda-python) compared to **CUDA C/C++** in the [beginner channel](https://discord.com/channels/1189498204333543425/1191300313928433664).
   - The member expressed optimism about the library’s potential if it achieves functional parity.
- **MI300 VGPRs Impact Occupancy**: In the **MI300 ISA**, using a high number of **VGPRs**, such as **255**, can lead to lower occupancy.
   - Lower occupancy can reduce performance due to decreased parallelism and increased overhead, as discussed in the [rocm channel](https://discord.com/channels/1189498204333543425/1233704710389764236).
- **MI300 Shines on AMD-FP8-MM Leaderboard**: Multiple members achieved personal bests on the **MI300** for the `amd-fp8-mm` leaderboard, with times including **4.93 ms**, **2.74 ms**, **2.53 ms**, **5.19 ms**, **5.28 ms**, **805 µs**, and **5.25 ms**.
   - One notable submission reached **1247 µs**, while another secured **6th place** at **289 µs** in the [submissions channel](https://discord.com/channels/1189498204333543425/1343002583001726986).



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Prompts Break After Updates**: Recent updates to **NotebookLM** caused a member's old prompts to break, suggesting a need for prompt engineering strategies updates.
   - Members may need to re-examine and revise their methods.
- **Zotero Craves Connection to NotebookLM**: A user asked about integrating their **Zotero** collection with **NotebookLM** without manually adding each PDF.
   - Currently it is not possible to connect **Zotero** directly, requiring individual PDF uploads.
- **NotebookLM as the Focused Expert Teacher**: One user suggested that a main use case is to learn something and **NLM** provides the option of creating a **Focused expert** for a specific area or topic that can act as a **teacher/coach/mentor**.
   - They added that *NLM can remove or limit the noise due to the sources being the truth behind the answers*.
- **File Size Restrictions Block US Code Analysis**: A member encountered PDF and word count limits when analyzing the **US Code** for redundancies using **NotebookLM**.
   - Another member suggested splitting the code into smaller files as a workaround for the size limitations.
- **Free vs Plus Accounts Cause Confusion**: Members questioned the difference between **Free** and **Plus accounts** in the context of downgrading and exceeding source limits.
   - **Free accounts** are limited to **100 Notebooks** and **50 sources** per notebook whereas **Plus accounts** can have **500 Notebooks**, and **300 sources** per notebook.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Claude's Images Puzzle MCP Tools**: A user questioned how **Claude Desktop** handles image uploads to **MCP tools**, specifically if **Claude** detects the most recent upload or uses a specific referencing method.
   - The user's current tool is built for image URLs and not direct uploads, seeking advice on **MCP tools** implementation that accepts uploaded images.
- **MCP Public Descriptions Glitch**: Multiple users reported that the **public description** for their **MCP server** isn't updating on the public-facing page, despite changes in the admin panel.
   - The issue was specifically noted with `dataryan.cph`.
- **JSON Backtick Blooper Baffles Hooks**: A user reported that **Claude** insisted on using **backticks** when creating **MCP hooks** to an SDK, wondering if they had a misconfiguration issue.
   - Another user suggested trying to remove the quotes from `dataToWrite`.
- **MCP Streamable Client is Sought**: A user sought a client supporting connections with **MCP server streamables**, as their server only runs with their own client and doesn't work with other common tools.
   - A user mentioned that *mcp-remote* supports mcphttps and links to the [draft version](https://github.com/geelen/mcp-remote/pull/32).
- **Kubernetes MCP Gets Implementation**: A member announced they built a new kubernetes MCP [based on the k8s API](https://github.com/StacklokLabs/mkp) to make it more general and flexible.
   - It is meant as an alternative to the first MCP Server for fetching GitHub repo, allowing **AI** to use repo code as references.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agent Building Discussions are Booming**: [AnthropicAI published Building Effective Agents](https://www.anthropic.com/), and **dexhorthy** went viral with his [12 Factor Agents](https://www.12factor.net/), and [OpenAI released A Practical Guide To Building Agents](https://platform.openai.com/docs/guides/function-calling).
   - The community is having **productive dialogue** on what agents are, and how to build them.
- **CondoScan Comprehensively Cuts Condo Costs**: **CondoScan** uses **LlamaIndex's agent workflows** and **LlamaParse's** accurate document processing to create a next-generation condo evaluation tool, see [this case study](https://t.co/SzIbcKta1O).
   - **CondoScan** *Evaluates financial health and lifestyle fit* reducing document review time from weeks to minutes.
- **`chat_history` and `memory` distinction explored**: If managing chat messages or initializing with a message list, use `chat_history`; otherwise, employ a specific memory module.
   - *If you are only managing the list of chat messages yourself, or want to init with some list of chat messages, use chat_history; If you are maintaining/using a specific memory module, use memory*.
- **`AgentWorkflow` Error Resolved**: An intermittent error in `AgentWorkflow` was traced to the error: *400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.', 'status': 'INVALID_ARGUMENT'}}*.
   - The error was resolved by upgrading `llama-index-llms-google-genai` via `pip install -U llama-index-llms-google-genai`, as detailed in [this GitHub pull request](https://github.com/run-llama/llama_index/pull/18527).
- **`FunctionAgent` now offers timeout functionality**: The [`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) class lacks a direct timeout parameter, but setting a timeout per LLM call on the LLM object is possible.
   - Newer agent classes like `FunctionAgent` directly support timeout via the `request_timeout` parameter, and a code snippet example was given using `Ollama` and setting `request_timeout` to 360 seconds.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Mulls Multimodal Modeling**: The community wondered whether **DSPy** supports **multimodal models** and **multimodal reasoning workflows**, pointing out that **Chain of Thoughts (CoT)** and **ReACT** could be more powerful if they incorporate **CoT** for tool selection.
   - The discussion acknowledged advances in reasoning models and the potential for **DSPy** to adapt to multimodal contexts.
- **Frameworks Flounder Facing Fast Flux**: Members observed the fast-paced evolution of the **AI landscape** makes it challenging for frameworks like **Langchain** and **LlamaIndex** to maintain currency.
   - The recommendation was to prioritize frameworks that simplify tasks, and to directly engage with the **model API** when necessary.
- **Ember Emerges as Novel Notion**: The group discussed new ideas such as [Ember](https://github.com/pyember/ember) that may require different strategies for building a **compound AI system**.
   - While **DSPy** offers **declarative syntax**, **Ember** proposes an alternative approach, each offering distinct benefits.
- **Text and Graphs Galvanize Business**: The group noted that many frameworks concentrate on **text and graph** due to business needs, emphasizing that business-centric systems often require reasoning on text and tables.
   - It was mentioned that images are frequently converted into **text, JSON structures, or tables** using **VLM (for OCR)** during pre-processing.
- **Demand Deep Code Analysis**: One member sought a method to analyze large code files beyond typical context limits, with the intention of extracting maximum insights.
   - They clarified their need for in-depth analysis, moving beyond generic requirements to leverage all possible information from a file.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **GRPO Code Sharing Requested**: A member requested the code for **Torchtune's GRPO**, speculating that it could be really useful to other users of **Torchtune's GRPO**.
   - The member also expressed curiosity about how intrusive the changes are, suggesting other users of **Torchtune's GRPO** would be interested to see the code as well.
- **PPO Epochs Bias KL Divergence**: A member questioned whether `ppo_epochs > 1` in **Torchtune** makes the KL divergence estimation biased, pointing to [line 85-87 of this file](https://github.com/joecummings/r1-zero/blob/main/torchtune/dev/grpo/loss.py#L85-L87).
   - They argued that after seeing every sample in the replay buffer once, the policies (**pi_theta, pi_ref, pi_old**) are different, so the samples aren't from the current policy **pi_theta**.
- **GRPO Padding Direction Questioned**: A member noted that the **GRPO data padding** direction is on the right side, and asked if the decoder model should pad on the left side during training.
   - Another member responded that padding can be done on both sides, as long as [input positions and masks are handled correctly](https://github.com/pytorch/torchtune/blob/main/recipes/dev/grpo_full_finetune_distributed.py#L750).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Low RAM Handles Huge LLMs**: A user stated that **16GB** of RAM is sufficient for running **32B** models, and even **70B** models can be run with **8GB** if one is OK with sacrificing quality and speed.
   - The user also noted that line-by-line LLM prompting is relatively simple to implement.
- **Shell Scripts can Replace Rust**: A member proposed that short shell scripts can do what **Rust code** can do.
   - No additional details were provided.
- **Llama4 encounters Scout 17x16**: A member sought guidance on running **Llama4** with **scout 17x16**, questioning the necessity of code updates or **Jinja** configurations.
   - Another user replied that *gpt4all* is outdated, suggesting exploring other options, leading the original member to give up.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Healthcare Startup Beta Hunts Prompt Feature**: A healthcare **AI/ML startup**, experiencing rapid growth and focused on impactful work, has signed up for the Beta **Prompt Optimization** feature but lacks access in the **Cohere Community Discord server**.
   - The startup is inquiring whether access is restricted by **user seniority** or **billing tier**.
- **Server Asks Newcomers for Introductions**: The server welcomes new members and encourages them to introduce themselves to the community.
   - New members are prompted to share their **company/industry/university**, current projects, preferred tech/tools, and goals for the community.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1365401957899960352)** (1 messages): 

> `iOS Voice Assistant, GPT Image Generation, Grok 3 Beta, OpenAI o4-mini` 


- **Perplexity Debuts iOS Voice Assistant**: Perplexity shipped an **iOS Voice Assistant** this week.
   - See the complete list in the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th).
- **Perplexity Rolls out GPT Image Generation**: Perplexity shipped **GPT Image Generation** this week.
   - See the complete list in the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th).
- **Perplexity Ships Grok 3 Beta**: Perplexity shipped **Grok 3 Beta** this week.
   - See the complete list in the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th).
- **Perplexity Deploys OpenAI o4-mini**: Perplexity shipped **OpenAI o4-mini** this week.
   - See the complete list in the [changelog](https://www.perplexity.ai/changelog/what-we-shipped-april-25th).


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1365175496798634046)** (1110 messages🔥🔥🔥): 

> `Discord Roles Rework, GPTs Agent Training, OpenAI sidebars, Apple Music Discord Support, Grok 3 Mini` 


- **Discord Roles get a Refresh**: Members report that **Kes** is reworking the [linking system for discord roles](https://discord.com/channels/link/to/roles).
   - Further details of how this role linking system is getting reworked or how it benefits users wasn't discussed.
- **GPT-4o gives Premium Text-Image output**: Members discussed how [GPT-4o Image Gen's language model](https://link.to/model) gives superior **text images**, though others prefer the overall quality and texture of Midjourney.
   - Members said that *MJ specifically wants people to be able to prompt a single word like cat and get really beautiful not basic stuff back - which means the underlying architecture is built so that it doesn't listen to what's in your prompt and does what it wants most of the time because they only care about aesthetics.*
- **Perplexity Team hosts 🥚Egg-cellent Event**: Members participated in an event to collect **eggs** across various channels and some expressed that it wasn't *possible for GPT Image gen* to generate a specific image.
   - They used several methods, such as *Just crop the image* while others complained about seeing the same *common egg drops are identical I watched it 100+ times by now.*
- **DeepSearch Backfires on Reddit Data**: A member jokes that the **Perplexity broke again award** is well deserved after [an AI image](https://link.to/img) returned an unexpected result.
   - Another member added, *I just wanna know how it arrived to that conclusion. Maybe it was looking at recommendations from the early 1900’s lol.*
- **Grok Mini Reasoning: Perplexity's Superior Secret Sauce**: Members discussed why Perplexity used a normal **Grok 3** over the reasoning version.
   - But a team member stated [Grok 3 is found to excel at reading, understanding, and answering questions - better than the reasoning grok 3 mini](https://link.to/blogpost).


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1365212835298152469)** (2 messages): 

> `Perplexity AI Search, Spanish Inquisition` 


- **Perplexity AI Search URL**: A member posted a link to a [Perplexity AI Search](https://www.perplexity.ai/search/was-the-spanish-inquisition-co-K1dnw4YsSlKIcp7CHsFBJQ.matsku) regarding the Spanish Inquisition.
- **Another Perplexity AI Search URL**: A member posted a link to a [Perplexity AI Search](https://www.perplexity.ai/search/1e012c90-ba02-499f-b99b-4151742606df).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1365336367453704355)** (1 messages): 

> `Perplexity API Introduction, Denis Yarats` 


- **Perplexity API Intro Video Teased**: A user expressed interest in watching ["An Introduction to Perplexity's API with Co-founder and CTO Denis Yarats"](https://www.youtube.com/watch?v=dQw4w9WgXcQ) once it becomes available.
- **User Missed Live Session**: The user noted their unavailability during the live session, indicating a demand for on-demand access to the API introduction.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1365211135514509353)** (1315 messages🔥🔥🔥): 

> `TACQ compression, TRL reward functions, Paradigm invests in Nous Research, Psyche distributed training project, Nous API waitlist issues` 


- **Nous LLMs Aim for 2-bit Precision via TACQ**: A [research paper](https://www.marktechpost.com/2025/04/22/llms-can-now-retain-high-accuracy-at-2-bit-precision-researchers-from-unc-chapel-hill-introduce-tacq-a-task-aware-quantization-approach-that-preserves-critical-weight-circuits-for-compression-withou/) introduces **TACQ**, a task-aware quantization approach for LLMs that retains high accuracy at **2-bit precision** by using a calibration dataset to decide which weights can be compressed more.
- **Seeking Battle-Tested TRL-Compatible GRPO Reward Functions**: A member inquired about finding battle-tested **TRL-compatible GRPO reward functions**.
   - Another member suggested looking at the [Open-R1 rewards](https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py) from Hugging Face as a starting point.
- **Paradigm Invests $50M in Nous Research**: Nous Research [announced a **$50M** investment from Paradigm](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/), a crypto venture capital firm, to advance its AI research and development efforts.
- **Nous to launch Psyche, a Solana-based distributed training project**: Nous Research is developing **Psyche**, a [distributed training project](https://nousresearch.com/nous-psyche/) utilizing the **Solana network** for coordination.
   - The team clarified that there are currently no roles or community programs planned for **Psyche** as the focus is on development, but a [Psyche Discord](https://discord.gg/peqZyPRd) is available for those interested in the crypto aspects.
- **Users Face Issues with API Waitlist Verification Emails**: Many users reported issues with receiving verification emails after signing up for the Nous API waitlist, likely due to **high traffic**.
   - The team acknowledged the issue, stating the **email sender** was likely overloaded and advised users to try again later, also clarifying that the waitlist is specifically for the API and not the Psyche project.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1365308221014671470)** (36 messages🔥): 

> `Max model size on Mac M4, LLMs and control vectors, Nous Psyche vs Petals, Hermes 3 70B settings, Open source coding models for 8GB VRAM` 


- **Mac M4 Memory Limits Model Size**: A member inquired about the max model size for **Nous models** on a **Mac M4 16**, and another member responded that most models under **70B parameters** quantized should work with **64GB+ memory**.
- **LLMs sniff out Control Vectors?**: A member asked if **LLMs** can tell if we're using a **control vector** to steer their thoughts in **zero-shot** or **few-shot** scenarios.
   - Another member suggested that LLMs might infer this from output patterns in few-shot scenarios or if directly asked about changes mid-conversation.
- **Psyche trains while Petals infers**: A member inquired about how **Nous** compares to **Petals**, a distributed LLM project.
   - Another member clarified that Nous's distributed project is called **Psyche**, and unlike Petals (for inference), Psyche is designed for training, with docs available at [Psyche Network](https://docs.psyche.network).
- **Qwen 2.5 Coder 7B Instruct is a decent option**: A member asked for the best open-sourced model for general-purpose coding that can run on a card with **8GB of VRAM** using **ollama**.
   - Another member suggested **Qwen 2.5 Coder 7B Instruct**, emphasizing the use of the **Q6_K version** and noting its limitations in handling more than basic assistant tasks and complex coding like a pong game.
- **Q6_K > Q4_K_M, allegedly**: In a discussion about the differences between **Q6_K** and **Q4_K_M**, a member claimed that **Q6_K is smarter, especially for coding**, but uses more memory.
   - The member suggested starting with **Q5_K_M** to balance context and speed, potentially upgrading to **Q6_K** with sufficient free VRAM, especially with **10GB of VRAM** available.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1365308889687527495)** (5 messages): 

> `Fine-tuning LLMs, LoRA, LLaMa Factory, CPU-only fine-tuning, GGUF` 


- **LLM LoRA CPU-Only fine-tuning Finally Achieved**: A user shared a [step-by-step guide](https://medium.com/@contact_30070/step-by-step-guide-for-fine-tuning-your-llm-with-llama-factory-using-the-cpu-only-96b2fc6a80b0) for **fine-tuning LLMs with LoRA using LLaMa Factory on CPU only**.
   - The guide covers everything from installing **torch** to loading the custom **GGUF** in **LM Studio**, and a [Youtube video](https://www.youtube.com/watch?v=1bgL4b7VT8M) accompanied the post.
- **LM Studio is lit**: Many users like the GUI.
   - Yes.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1365176707971027024)** (317 messages🔥🔥): 

> `o3 can output large file code, AGI path for big tech companies, Yann LeCun's 'A Path Towards Machine Intelligence', Meta's Memory+, DeepMind's Titans` 


- **o3 outputs big file code!**: Users report that **O3** can now output larger code files, reaching **700-1000 lines**, compared to the previous limit of around **300 lines**.
   - This improvement applies to **O4 Mini High** as well.
- **Possible Roads Leading to AGI**: A member proposed a path to AGI where AI decides to use **past memories**, updated knowledge graph, and **imagination** to create photos, 3D models/videos, and audio for context.
   - Another member recommended reading **Yann LeCun's** *A Path Towards Machine Intelligence* [openreview.net](https://openreview.net/pdf?id=BZ5a1r-kVsf) for a similar analogue without the generative aspect, arguing that latent space transformations are expressive enough.
- **Sunstrike model emerges in Arena**: A new model called **Sunstrike** has been added to the arena and initial impressions are that *claude-3-7-sonnet-20250219-thinking-32k > sunstrike > gemma-3-12b-it*.
   - Early tests reveal **Sunstrike** may be a Google model and has a right an arc-agi problem that **2.5 pro** only gets like **25%** of the time.
- **Is GPT-4o natively multimodal?**: There is a discussion on whether **GPT-4o** is natively multimodal, with arguments presented for and against, related to the use of tools.
   - One member argues that **GPT-4o** isn't truly multimodal because it makes tool calls with specific natural language queries and generates completely new images, unlike native image generators and another members states that **GPT-4o** *used to call Dali-E but now it generates images natively*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1365195233008156704)** (293 messages🔥🔥): 

> `Manus's strategic role, Geostrategic relevance of HK, Manus better than ChatGPT?, Double credits, Manus's issues` 


- **Manus Credits Double Dipping: Limited-Time Bonus Bonanza**: Manus is offering **double credits** for the first month to both new and existing subscribers: Starter plan gets **+3,900 credits**, while the Pro plan gets **+19,900 credits**.
   - This is a *time-limited offer* with no fixed end date, so users are encouraged to use it soon.
- **Free Users get First Subscription Credits**: For **free users**, a member clarified they get double credits for the **first subscription** upon upgrading to a paid plan.
   - If they subscribe to starter they'll get 3900✖️2 credits, for pro it's 19900✖️2.
- **Coding Clone created from Manus**: A member created a [Discord flask clone](https://github.com/johnnyshelby123/discordflaskclone) and offered it up to be improved by Manus.
   - Another member offered to recode it in **Node.js**, but the original creator had updated the code, emphasizing that **Manus did 100% of the work**.
- **Prompt Engineering Pointers: Hierarchical Reasoning Generator**: A member shared [their approach to prompt engineering](https://github.com/justinlietz93/hierarchical_reasoning_generator/tree/main/hierarchical_planner/persona_builder/personas) to generate an entire project with minimal code.
   - They pointed to hierarchical systematic and dynamic gateways to adjust the persona or prompt based on the initial plan generated, you can basically generate an entire project in one prompt with two python files, less than 1000 lines of code each.
- **Flat-Fee Fantasies: Users Ponder Pricing**: Some users are frustrated with Manus's **credit-based system**, stating they’d prefer a flat monthly fee for unlimited usage.
   - One user explained that due to the mistakes Manus made, they lost the credits they had paid for and were forced to switch back to **ChatGPT 4.0**.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1365189118967021628)** (80 messages🔥🔥): 

> `Sora limitations and pricing, RAG + LLM + Word2Vec discussion, Fine-tuning a math model, Gemini Advanced video generation limits, Alternatives to Sora for photorealistic image generation` 


- ****Sora's Video Pricing Unveiled!****: A user inquired about **Sora's image creation limitations** and discovered new options for purchasing video generation and concurrent image creations at a higher cost.
   - The user initially faced a restriction of one image at a time and prompts to *upgrade* to a plan they already had.
- ****RAG, LLM, and Word2Vec Combo in the Spotlight****: A member sparked a discussion about combining **RAG, LLM, and Word2Vec**, noting that many mainland Chinese companies are developing software with a *Local Knowledge Base* feature.
   - Concerns were raised about **LLM's grasp of contextual length**, potentially requiring further paragraph-level processing of the original document, leading to questions about the technology's future.
- ****Math Model Mania: Specializing for Speed and Savings!****: One user suggested **fine-tuning a smaller AI model specifically for university-level math problems**, instead of relying on large models trained on everything.
   - The idea is to create a model *just large enough to handle language to understand a problem* and integrate a built-in calculator using Python, optimizing for speed and cost-effectiveness.
- ****Gemini Advanced Limits Exposed: A Video Generation Showdown****: A user pointed out the inconsistent **video generation limits in Gemini Advanced**, noting that it provides only **10 videos per day** compared to **4 per day** for free in AI Studio.
   - Despite the limit, they believe **Gemini's Veo 2** produces better output quality than **Sora** for generic content, but is too restrictive and has a delayed refusal problem.
- ****Beyond Sora: Quest for Uncensored Photorealistic Image Alts****: A professional photographer using **Sora** sought alternatives with fewer content restrictions for photorealistic image generation at a professional, cinematic level.
   - Suggested alternatives include **Framepack, HiDream, and Flux**, which are local and uncensored, but their ability to match **Sora's cinematic lighting, depth of field, and sharp realism** remains uncertain.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1365185736688795721)** (16 messages🔥): 

> `OpenAI Pro vs Plus Deep Research limits, Deep Research Tier List, GPT generating images when code is expected` 


- **Pro Users Get Deep Research Limit Details**: OpenAI Pro users get **120 original deep research** and **120 more light weight deep research** according to [this tweet](https://x.com/OpenAI/status/1915505961500070245).
- **Deep Research Tier List Created**: Users are seeing **Deep Research full**, **Deep Research lite**, **Deep Research mini** (just normal o3), and **Deep Research nano** (just normal o4-mini).
   - Another user proposed *Deepest Research* and *Deep and Dark Research* as possible names.
- **GPT generates images when code is expected**: A user shared an [image](https://cdn.discordapp.com/attachments/973938214257713183/1365421332338053232/image.png?ex=680d3f35&is=680bedb5&hm=c0a3d473ee1c6db11d216f00d66b15a961805de359dd8bf9c0e746e5a2c28b31&hello) showing that GPT, when given a task for a desktop, started writing that it's generating an image instead of generating code for the desktop according to the document.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1365315965226717238)** (48 messages🔥): 

> `Learning Python with ChatGPT, Python vs. Java for Beginners, Web Development Languages (JavaScript), The Odin Project` 


- **ChatGPT: Your Python Professor?**: Members discussed if **ChatGPT** can effectively teach **Python**, agreeing it can cover core fundamentals but may struggle with up-to-date third-party packages.
   - One member emphasized that *the hard part isn't the knowledge; it's the discipline* to self-study without the structure of a class.
- **Python or Java for Web Dev?**: A member inquired about using **Python** for web development, but another clarified that **JavaScript** (specifically **Three.js**) is generally preferred.
   - It was noted that **Java** is distinct from **JavaScript**, and one is better suited for different roles.
- **Odin Project: Jumpstart Web Dev Journey**: A member recommended the [Odin Project](https://www.theodinproject.com/) as a free, highly praised program for web development beginners.
   - It was suggested to use AI as a guide to clarify confusing concepts within the material.
- **2025: Is Python still the Premier League Language?**: In a discussion about the most suitable language for beginners in 2025, it was argued that **Python** might not be the automatic first pick.
   - It was also mentioned that **Python** remains dominant in AI projects, while **JavaScript** is useful for web and UI development, and **C++** is for system-level tools.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1365315965226717238)** (48 messages🔥): 

> `Learning Python with ChatGPT, Python vs Java for Beginners, Python for Web Development, The Odin Project, Python's Role in AI` 


- **ChatGPT: The Python Professor**: Members discussed how beginners can learn **Python** using **ChatGPT** as a teaching tool, with one member saying that *it's a teacher that'll help explain anything you ask it*.
   - It was noted that while **ChatGPT** is helpful for core language fundamentals, its knowledge of third-party packages may be outdated, since *most modern ones get major rehauls every single time LLM's knowledge cutoffs get an update*.
- **Java vs Python: Beginner's Dilemma**: The conversation explored whether it's necessary to learn both **Python** and **Java**, with one member stating that it is necessary *if you want to program in python and java*.
   - However, another member also noted that knowing both is necessary *Even if you want the AI to do your coding*, to troubleshoot any issues.
- **Python not ideal for web dev, try JS**: Members stated that **Python** isn't typically used for web development, with a recommendation to learn **JavaScript** instead, particularly **Three.js**.
   - The conversation clarified that **JavaScript** is distinct from **Java**.
- **Odin Project: A learning Oasis**: One member suggested beginners check out the **Odin Project**, a free, highly-praised program for web development.
   - They suggest the user can *use AI while you follow those studies as your guide to clarify the things that come off confusing in the material*.
- **Python's place in 2025 AI landscape**: A member mentioned that in **2025**, **Python** is still relevant for AI projects but it's place is *maybe not a great core first choice*.
   - The discussion then shifted to Python's prevalence in **AI** versus **JavaScript's** usefulness in web development, and **C++** for system-level tools.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1365179338194157651)** (96 messages🔥🔥): 

> `Unsloth Dynamic v2.0 GGUFs release, GLM 4 models, Tesslate Rust 7B model, Dynamic Quantization Workflow, QwQ 32b training` 


- **Unsloth drops Dynamic v2.0 GGUFs Release**: ICYMI, Unsloth has released their new **Dynamic v2.0 GGUFs** which are available [here](https://x.com/UnslothAI/status/1915476692786962441).
   - A member inquired about the difference between this and the multi GPU support that is coming soon, and another clarified that the *official needs a lot of tests and stuff to confirm its stable*.
- **GLM-4 fever strikes Unsloth's HF**: Members are discussing the new **GLM 4 models** and a member has uploaded it to HF [here](https://huggingface.co/unsloth/GLM-4-32B-0414-GGUF).
   - One member highlighted [THUDM/GLM-Z1-32B-0414](https://huggingface.co/THUDM/GLM-Z1-32B-0414) and mentioned that it *tops half the benchmarks even against DeepSeek R1*.
- **Rust 7B Quant Request sparks Rust fim**: A member requested a quant of **Tesslate**'s new **Rust 7B model**, sparking another to create a Rust-only version of their codefim [dataset](https://huggingface.co/datasets/Etherll/CodeFIM-Data).
   - The member then asked *Has anyone compiled phi-4 to gguf yet?* as it was generating an error.
- **Dynamic Quantization workflow is secretive**: A member inquired if the new **dynamic quantization workflow** announced recently is open source.
   - Another member confirmed that *the workflow isn't open sourced* because *it's slightly different for each model*.
- **QwQ 32b ablation needs serious juice**: A member asked if there was any reason they couldn't finetune an abliterated version of **QwQ**, linking to the [model on HF](https://huggingface.co/huihui-ai/QwQ-32B-abliterated).
   - Another responded that **QwQ 32b** probably needs about **30+GB** of VRAM to train, so they would need to configure quantization or FSDP to train a model that typically can't fit on one of their GPUs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1365209935213887500)** (62 messages🔥🔥): 

> `torch.compile issues, GGUF working, chat template, llama.cpp error` 


- ****Torch Compile Troubles?****: A user reported a `Tensor.requires_grad` issue potentially linked to `torch.compile`, noting that **LoRA** and **full fine-tuning** work for `FacebookAI/roberta-base` with `attn_implementation = "eager"` and **modernBERT**.
   - They asked in which line of code this would be fixed.
- ****GGUF Quirks Uncovered****: A user inquired about why one **GGUF** file works in **fp16** while another doesn't, specifically comparing `snowflake-arctic-embed-m-v1.5` and `snowflake-arctic-embed-l-v2.0-f16` **GGUF** models.
   - The model is for embedding.
- ****Chat Template Confusion Clarified****: A user questioned why the `llama3_1_alpaca` notebook doesn't apply a chat template by default, leading to an explanation that `apply_chat_template` defaults to the template in the model's `tokenizer_config.json` or `chat_template.json`.
   - Using a different template, like **alpaca**, requires specifying it explicitly, and the choice of template should align with the desired interaction format for fine-tuning and inference.
- ****Llama.cpp Struggles with DeepSeek V3****: A user encountered an error when running the **UD Q4** version of **DeepSeek V3** on **llama.cpp**, despite a prior **Unsloth GGUF Q4** version working fine, citing a shape mismatch error for the `'blk.0.attn_q_b.weight'` tensor.
   - The user recompiled **llama.cpp** with the latest version, but the error persisted.
- ****Gemma-3-4b-it fine-tuning woes****: A user ran into a `RuntimeError` on Google Colab when training the `gemma-3-4b-it` model via the Unsloth notebook, experiencing a *datatype mismatch between float and c10::Half*.
   - The error persisted even after reverting all changes, indicating a potential incompatibility with the environment.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1365366933138047037)** (2 messages): 

> `Kimi Audio, MoonshotAI` 


- **MoonshotAI releases Kimi Audio**: MoonshotAI released [Kimi Audio](https://github.com/MoonshotAI/Kimi-Audio), whose paper is available on [arXiv](https://arxiv.org/abs/2504.09858).
- **Kimi Audio on arXiv**: The paper for Kimi Audio is available on [arXiv](https://arxiv.org/abs/2504.09858).


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1365184879440171178)** (86 messages🔥🔥): 

> `Llama-3.2 checkpoints, Nvidia drivers in TCC mode on Windows, Cost of computer specs and digital divide, Colaboratory for data analytics, Flash Linear Attention models` 


- **Llama-3.2 Checkpoint Export to ONNX Proves Elusive**: A member is seeking a script to export **Llama-3.2 checkpoints** to **ONNX** with **KV cache** and **TensorRT** support, reporting no benefit from architecture-specific initializations.
   - They noted *little to no benefit* when playing with default PyTorch inits vs architecture-specific inits, and asks if it's even possible to have nvidia drivers in TCC mode on Windows or WSL.
- **Solving the Digital Divide with Colaboratory**: A member expressed worry about the rising cost of computer specs and its potential to widen the digital divide.
   - Another member suggested using [Google Colaboratory](https://colab.google) with its *unused 720s* for light data analytics, though noting that it's limited to models around **1B parameters**, like **GPT-2**.
- **Modded 48GB RTX 4090s become all the rage**: A member mentioned the existence of **modded 48GB RTX 4090s**, especially popular in China, linking to [a teardown article](https://www.tomshardware.com/pc-components/gpus/blower-style-rtx-4090-48gb-teardown-reveals-dual-sided-memory-configuration-pcb-design-echoes-the-rtx-3090) from Tom's Hardware.
   - These cards generally sell for around *$3,500*, though information regarding a **5090** is unconfirmed due to tight supplies.
- **Tiny Datasets Get Linear Attention**: A member sought the *easiest linear transformer model or equivalent to train on tiny datasets*, excluding **RWKV**.
   - Another member suggested models from the **Flash Linear Attention repo**, such as **Gated Linear Attention** or **Gated DeltaNet**, trained with their **Flame trainer**.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1365176777831616594)** (68 messages🔥🔥): 

> `ReaSCAN results, User edits, Key replacement with learned LERP, Memory mosaics paper, tokenshift` 


- **Research papers**: It would be good to see results from **ReaSCAN** relative clause tests and some research on topics relating to **misalignment tolerance**, **system modulation**, **inverse modeling**, and **non-linear gravitational divergence and memory**.
   - Suggesting papers by **Xia & Sigmund**, **Fatemi Booshehri**, **Zhang, Sharif & Siddiqa** and other notable researchers.
- **Keys replaced!**: There's a question whether anyone has replaced keys with a learned **LERP** of **W_kx_t** and past key, resulting in the formula **k_t = lambda W_kx_t + (1-lambda)k_{t-1}**.
   - The member's takeaway from the [memory mosaics paper](https://arxiv.org/abs/2405.06394) is that this is the main *addition* introduced which isn't present in regular multihead attention.
- **RWKV tokenshift**: Regarding prior sequence index, someone noted that is just **RWKV tokenshift**.
   - Another member shared that it has been in regular transformers previously, as this has been in their **GPTAlpha transformer recipe** for years - the description of that architecture was in their **GPTCore** repo README and code for a long time but only made its way into a paper in [GoldFinch](https://arxiv.org/abs/2407.12077).


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1365280267920740404)** (6 messages): 

> `TinyStories replacement dataset, Computational Sparsity talk by Leo, Attribution based Parameter Decomposition (APD), Local Loss Landscape Decomposition (L3D)` 


- ****SimpleStories** Dataset Suite Swaps **TinyStories****: A new replacement [dataset](https://huggingface.co/datasets/lennart-finke/SimpleStories), tokenizer, and [model suite](https://huggingface.co/SimpleStories) called **SimpleStories** has been released.
   - A [community doc](https://docs.google.com/document/d/1XRb-EDDw-h6c-L6pKltI9A8zJmufE71I2_4wx8d5rMg/edit?usp=sharing) is available to help researchers get started.
- **Sparsity talk sparks speculation**: Members discussed Leo's talk on computational sparsity and whether it was similar to **Attribution based Parameter Decomposition (APD)** ([paper](https://arxiv.org/abs/2501.14926)) or **Local Loss Landscape Decomposition (L3D)** ([paper](https://arxiv.org/abs/2504.00194v1)).
   - The talk was said to be about sparse autoencoder stuff, and new work does not seem similar to those earlier papers.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1365176535727738880)** (95 messages🔥🔥): 

> `Aider Language Sorting, Grok-3-Mini Pricing, Gemini+Claude Combo, Aider Architect/Editor Workflow, Gemini Verbose Comments` 


- ****Aider** Gets a Linguistic Upgrade**: Aider now boasts the ability to sort by programming language, enhancing code organization and accessibility for developers as demonstrated by [this UI screenshot](https://cdn.discordapp.com/attachments/1131200896827654149/1365176535484596327/image.png?ex=680d03f9&is=680bb279&hm=accbfb4dd2576f4bc500e5e11cdf24b69189951769521c7cbb2980fee2cd03e0&).
- ****Grok-3-Mini** Competes on Pricing**: **Grok 3 mini** is almost half the price of **Deepseek V3** via OpenRouter, with nearly equivalent reasoning performance as displayed in [this image](https://cdn.discordapp.com/attachments/1131200896827654149/1365185829596696636/image.png?ex=680d0ca1&is=680bbb21&hm=f5228d04256f815b37940355080c80e2cdaa6d86a0f6b37759d6d53b59c8be26&).
- ****Gemini Pro 2.5** Teams Up with **Claude 3.7****: A member is experimenting using **Gemini Pro 2.5** as the architect model and **Claude 3.7** as the editor model in Aider, drawing inspiration from videos featuring **Claude Code** and **Gemini** via its MCP interface.
   - Another member reported that while **Gemini** is *extremely smart*, it tends to alter unrelated code, making code review difficult, and therefore prefers using DeepSeek V3 as the editor model instead.
- ****Aider's** Architect-Editor Workflow Unveiled**: Members discuss the division of labor in Aider's architect/editor setup, where one AI handles reasoning and another focuses on code editing.
   - A user mentions the value of `ask mode` to generate a detailed plan for a request using `/ask Generate a detailed plan for <YOUR REQUEST HERE>` and then follow the plan, as well as using `/model` to switch models on the fly to test various outputs.
- ****Gemini's** Commentary Overload**: Users are seeking ways to reduce **Gemini's** verbose comments, with one suggesting the use of conventions to remove unnecessary comments when using **Gemini** in architect mode with **GPT-4.1** as the editor.
   - A user observed *Gemini stays yapping in the comments worse than ive ever seen* and is looking for better solutions.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1365249954020462633)** (18 messages🔥): 

> `Gemini 2.5 Pro Setup, Read only files in Aider, Work tracking markdown logs and Aider, Aider CLI vs IDE, Openrouter Fallbacks` 


- **Gemini 2.5 Setup Simplified**: To replicate the **Gemini 2.5 Pro Preview 03-25** model in Aider, simply use the command `aider --model gemini/gemini-2.5-pro-preview-03-25` as per the [leaderboard instructions](https://aider.chat/docs/benchmarks.html).
- **Read only files**: The answer here is to add the files as **read only** btw.
- **Work tracking markdown logs not good for Aider**: A member is unable to track markdown logs, and wants Aider to update them.
- **Aider mostly used in the CLI**: Members use aider 90% of the time in the **terminal** (Warp.dev), and the rest in a **sidebar in VS Code**, and no ide plugins.
- **Openrouter Fallbacks need help**: Members report that the **Openrouter fallback** system *exists but never actually seems to work*, even with multiple API keys added.
   - It isn't implemented on 429 or not implemented correctly.


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1365223897922015323)** (3 messages): 

> `Aider Commands, Code Generation, Conversation Summarization` 


- **Decoding Aider Commands for Code Generation**: A user inquired why `/architect proceed` was used to generate code, instead of just `proceed` or `do it`.
   - Another member clarified that `proceed` or `do it` uses "code" mode, sending the entire conversation to the editor model, whereas `/architect` asks the main model to summarize the changes needed, sending only that summary to the code model.
- **Understanding Code Generation Modes in Aider**: The discussion highlights two methods for triggering code generation within Aider: using `proceed` (or similar commands) and using `/architect proceed`.
   - The key difference lies in how much context is sent to the code model; the former sends the entire conversation, while the latter sends a summarized version of the required changes.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1365234208871546941)** (53 messages🔥): 

> `Framepack impressions, Multimodal LLMs, Hugging Face Inference API Issues, SD3.5 API refresh, LLM for python coding` 


- **Framepack receives high praise, Unsloth endorsed**: Members noted **Framepack** is *super cool* and makes fine-tuning accessible, potentially right up **Unsloth's** alley and endorsed by them.
   - Excitement around stress-testing with frequent weight updates every **8 seconds**, suggesting collaboration with expert testers.
- **Explore multimodal LLMs for video and audio**: Discussion on using multimodal LLMs like [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct), and [SmolVLM2](https://huggingface.co/blog/smolvlm2) for **video footage interpretation** and summarization.
   - Members shared a link to [Nvidia's DAM-3B-Video](https://huggingface.co/nvidia/DAM-3B-Video) released a day prior, and others shared a link for [transcribing YouTube audio](https://huggingface.co/spaces?sort=trending&search=youtube).
- **Hugging Face Inference API has traumatizing issues**: A member reported experiencing traumatizing images using the Hugging Face Inference API for a product, prompting a need to contact the **HF staff**.
   - Another member clarified that the **Stable Diffusion** team operates under **Stability AI**, separate from the Hugging Face staff.
- **SD3.5 API refresh triggers library updates**: With regards to **SD3.5**, the API refresh will trigger updates in libraries such as [TGI](https://github.com/huggingface/text-generation-inference/issues), [Diffusers](https://discord.com/channels/879548962464493619/1014556809928904794), [huggingface_hub](https://github.com/huggingface/huggingface_hub/issues), and [huggingface.js](https://github.com/huggingface/huggingface.js/issues).
   - The issue stems from a **500 internal error**, being worked on to fix as soon as possible, see [API refresh](https://discuss.huggingface.co/t/500-internal-error-were-working-hard-to-fix-this-as-soon-as-possible/150333/32) discussion.
- **Request for advice on building python LLM from scratch**: A member inquired about creating a specialized LLM focused solely on outputting **Python code**, using only **PyTorch**.
   - Another member suggested building the architecture from scratch in PyTorch first, potentially exploring the **DeepCoder model**.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1365292361902723094)** (1 messages): 

> `Deeplearning.ai course, Smolagents and Gemma3, Code Agents vs LLMs` 


- **Deeplearning.ai Adopts Tavily and OpenAI**: A member noticed that the [Deeplearning.ai](https://www.deeplearning.ai/) course uses **Tavily** and **OpenAI**.
   - They were exploring alternatives that might allow for more local hosting.
- **Smolagents Explores Local LLM Capabilities with Gemma3**: A member inquired whether **smolagents** can be used with **Gemma3**, or other local LLMs.
   - This could potentially enable running the agents without relying on external APIs.
- **Code Agents Enhance Reasoning Beyond LLMs**: A member asked if it's true that **code agents** perform better than calling LLMs directly, even in reasoning tasks.
   - They also questioned whether code agents require direct APIs, or if there are other options.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1365323529163374655)** (3 messages): 

> `GAN, Excel, GAN model` 


- **Excel-lent GAN Model?**: A member shared a link to a [SharePoint document](https://o365coloradoedu-my.sharepoint.com/:x:/g/personal/peye9704_colorado_edu/EaxYYtAUapZBkbQ3oiTDfAUBPbIlKqWJKQSNvqk3I1L6qQ?rtime=hWWmvf-D3Ug) about developing a **GAN model** by hand using Excel.
- **Build your own GAN with Excel!**: Dive into the world of **Generative Adversarial Networks** using a tool you already know: Microsoft Excel, using [this tutorial](https://o365coloradoedu-my.sharepoint.com/:x:/g/personal/peye9704_colorado_edu/EaxYYtAUapZBkbQ3oiTDfAUBPbIlKqWJKQSNvqk3I1L6qQ?rtime=hWWmvf-D3Ug).


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1365230181731536959)** (4 messages): 

> `Small Models Math Dataset, ingest-anything, Malware Analysis Agent` 


- **New Dataset to help small models Learn Math**: A member shared a [dataset](https://huggingface.co/datasets/ProCreations/SimpleMath) to help small models learn simple math better and accurately, instead of complex math wrong most of the times.
   - It also helps larger models slowly ease into learning complex math by introducing simple math first and then moving to the massive complex math datasets.
- **ingest-anything Project to Ingest Files Into Vector DBs**: A member introduced [ingest-anything](https://github.com/AstraBert/ingest-anything), a new open-source project that converts non-PDF files to PDF, extracts their text, chunks, embeds, and loads them into a Qdrant vector database.
   - The tool uses **PdfItDown** for PDF conversion, **LlamaIndex** readers for text extraction, **Chonkie** for chunking, and **Sentence Transformers** for embedding.
- **Reasoning agent for Malware Analysis is Born**: A member created a malware analysis reasoning agent and report generation framework using **Agno, Claude, and GPT4.1 Nano**.
   - Check out the [post here](https://www.linkedin.com/posts/subham-kundu-2746b515b_ai-agent-llm-activity-7321534694242549760-pe_Q?utm_source=share&utm_medium=member_android&rcm=ACoAACZeVjgB0HEDqU1BExX1Ypnp-q8LcgDAunk) to learn more!


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/)** (1 messages): 

ilham_xx: thank you so much, <:agree:1098629085955113011> ⭐
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1365235849368961055)** (3 messages): 

> `Agent Template Error, Local LLMs with SmolAgents, Code Agents vs LLMs for Reasoning` 


- **Agent Template Falls Flat with 402 Error**: A new user encountered a **402 Payment Required error** when using the first agent template with the **Qwen/Qwen2.5-Coder-32B-Instruct model** to ask for the current time in Abu Dhabi.
- **SmolAgents: Can it power Gemma Locally?**: A user inquired about using **smolagents with Gemma3** or other local LLMs, noting the Deeplearning.ai course uses **Tavily** and **OpenAI**.
   - The question hits on how well local models integrate with the agent framework.
- **Code Agents Steal Reasoning Crown from LLMs?**: The same user asked if **code agents perform better than calling LLMs directly**, especially in reasoning tasks, and whether they require direct APIs.
   - The question implies a trade-off between reasoning ability and the need for direct API access.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1365217759008522311)** (50 messages🔥): 

> `Course Deadlines, Certificate verification, DuckDuckGoSearch timeout, Pokemon Showdown LLM Cheating, Unit 4 Assignment` 


- **Course Deadline Clarification**: The course deadline of July is related to obtaining the final certificate, but the Hugging Face courses are mainly well-written tutorial notebooks on their website, so it is for the certificate, [as stated by a user](https://huggingface.co/spaces/agents-course/README/discussions/25).
   - After earning and downloading the first certificate, it was issued with a URL to add it to LinkedIn.
- **Certificate Verification Conundrum**: A user reported issues verifying the final certificate's validity, as the download link led to a temporary folder file instead of a CDN URL.
   - However, [the solution seems to be](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence) using the HF username as the credential ID and the dataset link as the URL, providing a more permanent verification method.
- **DuckDuckGoSearch Timeout Troubles**: Several users are experiencing timeout errors with DuckDuckGoSearch, specifically a `DuckDuckGoSearchException: https://lite.duckduckgo.com/lite/ RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): operation timed out`.
   - The users confirmed they don't have any VPN or firewall configured.
- **Pokemon Showdown LLM Cheating Apocalypse**: A user jokingly compared the emergence of reasonably good Pokemon Showdown bots to *"9/11 for Pokémon showdown,"* anticipating annoyance from an army of bots.
   - LLMs are already being used to crush Pokemon Showdown, and there are already cheating scandals for generations.
- **Unit 4 Assignment Clarification**: The goal of Unit 4 assignment is to solve a subset of 20 level 1 questions from the [GAIA benchmark](https://huggingface.co/datasets/agents-course/course-certificates-of-excellence), which requires the agent to use at least one tool to solve (e.g., web search, file reading).
   - Users should run the template once to see what kind of question they can get it will be clearer.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1365193089773342841)** (109 messages🔥🔥): 

> `Reward Shaping Best Practices, Symbolic World Models, Oscillatory State Space Models, LLM World Models, Generating New Agents/Models/Worlds` 


- **AI Agent Learns Best Practices for Reward Shaping**: A member requested literature on reward shaping best practices and rules of thumb, linking to a paper on [offline training with simulated rollouts](https://arxiv.org/abs/2502.05244) from a current world model to reduce interactions with the environment.
   - The discussion included a humorous suggestion for reward shaping: *thumbs up for positive reward and thumbs down for negative reward.*
- **Symbolic World Models Gain Attention**: A member shared an [arXiv link](https://arxiv.org/abs/2503.20124) and highlighted the advantages of using programs as world models, emphasizing their structured and compositional nature, allowing easier modification compared to neural representations.
   - This approach facilitates modifying specific parts of a model in response to new observations more readily than neural representations.
- **LLMs Implicitly Model the World**: One member posited that LLMs' world models are *fused into their weights*, contrasting this with discrete or structured models.
   - Another member introduced a formula `(A, W) = G(z)` to represent generating new agents/models/worlds on demand, where A is the model's architecture, W is weights/parameters, G is the generator, and z is the semantic seed.
- **Formula for Generative AI**: One member introduced a general formula `(A, W) = G(z)` for AI model generation, where **A** is the model's architecture, **W** is the weights/parameters, **G** is the generator, and **z** is the semantic seed.
   - The member suggested this formula encompasses various paradigms, including hypernetworks, and could lead to significant compression and generativity in AI models, comparing it to *storing DNA rather than the whole human*.
- **AGI vs ASI - the end of mensa?**: A member distinguished between AGI (Artificial General Intelligence) and ASI (Artificial Super Intelligence), viewing AGI as a dynamic and adaptive system/agent and ASI as a constructive and procedural system/agent.
   - The discussion touched on how procedural systems sidestep symbolic traps by *not simulating directly*, *not needing logical completeness*, and *accepting approximation*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1365201994913943593)** (4 messages): 

> `Music AI, Perplexity Browser` 


- **Google DeepMind unveils Music AI Sandbox with new features**: Google DeepMind unveiled [Music AI Sandbox](https://deepmind.google/discover/blog/music-ai-sandbox-now-with-new-features-and-broader-access/) with new features and broader access.
   - The blogpost highlights features and updates about this music AI sandbox.
- **Perplexity CEO declares browser will track everything**: The Perplexity CEO announced that their browser will track *everything users do online to sell hyper-personalized ads* as seen in [this TechCrunch article](https://techcrunch.com/2025/04/24/perplexity-ceo-says-its-browser-will-track-everything-users-do-onl).
   - A member responded that *that's the best marketing for not using it.*


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1365428376575410269)** (20 messages🔥): 

> `Gemini 2.5 Pro Experimental Free, Rate Limits, Error Messages` 


- **Gemini 2.5 Pro Experimental Free Tier Constrained**: Due to high demand, [Gemini 2.5 Pro Experimental](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25) is removed from the free model tier and now has stricter usage limits: **1 request per minute** and a total of **1000 requests per day** (including errors).
   - The free tier is still available, but for higher reliability, the [paid variant of Gemini 2.5 Pro](http://google/gemini-2.5-pro-preview-03-25) is recommended; model access is gated for users who have purchased at least **10 credits** all-time.
- **Free Gemini Model Identifier Fixed**: The `:free` model alias will hit the standard variant, so code using the model ID will continue to work.
   - A user reported that `aider --model openrouter/google/gemini-2.5-pro-exp-03-25:free` did not work this morning, but the fix was merged and will be live soon.
- **Custom Rate Limit Error Messages on the Horizon**: Users are receiving error messages directly from the Gemini API when hitting the rate limit, which is confusing, since it doesn't explain if the limit is global or user-specific.
   - The team discussed potentially adding a custom message to clarify the source of the rate limit error for users.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1365182006366965801)** (92 messages🔥🔥): 

> `Baidu Models on OpenRouter, OpenRouter support for OpenAI's o3, Gemini 2.5 Pro rate limits, Nvidia Nemotron settings, OpenRouter credits exploited` 


- ****Baidu Models Beckon, Will OpenRouter Bow?****: A member inquired about the possibility of adding [Baidu models](https://x.com/Baidu_Inc/status/1915603080336597310) to OpenRouter, noting their interesting potential, while another pointed out the existing availability of **DeepSeek**.
   - Followed by discussion of **GLM-4-0414** hosted by other providers.
- ****O3 Odyssey: OpenRouter's Verification Venture!****: A member asked if OpenRouter will support **OpenAI's o3 and other advanced models** without verification in the future.
   - Another member stated that **OpenAI** has indicated the requirement may be dropped in the future.
- ****Gemini's Generosity: Free Tiers Trigger Throttling Troubles!****: Members discussed the **rate limits** on the free tier of **Gemini 2.5 Pro**, which led to confusion among users and queries about whether the limitations were on the OpenRouter side.
   - A member noted that OpenRouter is posting an announcement about it, while another said that the demand is exceeding supply.
- ****Nemotron's Nuances: Nvidia's Notes on Nimble Navigation!****: A user sought clarification on optimal settings for the **Nvidia Llama-3.1 Nemotron Ultra 253B v1** model, prompting another user to share the [developer's recommendations](https://openrouter.ai/nvidia/llama-3.1-nemotron-ultra-253b-v1:free).
   - The recommendations included setting temperature to **0.6** and Top P to **0.95** for Reasoning ON mode, and using greedy decoding (**temperature 0**) for Reasoning OFF mode.
- ****Credits Crisis: OpenRouter Account Emptied!****: A member reported an incident of their **OpenRouter credits being depleted** due to an exploit involving infinite URL generation.
   - The malicious activity was traced to a proposed solution architecture involving a "Thread-to-Chart Correlation System" that spiraled into a URL of approximately **3000 characters** before being stopped.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1365180788659982497)** (105 messages🔥🔥): 

> `RTSP firmware package for ESP32, Gemini 2.5 and Sonnet 3.7 performance, Billing settings on Cursor, Model evaluations for Cursor` 


- **Struggles with RTSP Firmware on ESP32**: A member is trying to create a working **RTSP firmware package** for outbound streaming on the **ESP32** and asked for suggestions.
   - Other members recommended these two **GitHub** repos, [esp32cam-rtsp](https://github.com/rzeldent/esp32cam-rtsp) and [ESP32-RTSPServer](https://github.com/rjsachse/ESP32-RTSPServer), and suggested using **PlatformIO with AI** for development.
- **Gemini 2.5 and Sonnet 3.7 dumbed down?**: Members are reporting that non-max versions of **Gemini 2.5** and **Sonnet 3.7** seem dumber and hallucinate more compared to when used via the command line interface (**CLI**).
   - One member speculates that **Cursor** might be changing the system instructions or prompts, but another countered with *"Gemini issues should be fixed, was a Google problem"*
- **Cursor billing issues are addressed**: A member inquired about immediate payment options for an unpaid invoice due to a bank confirmation delay.
   - Another member, likely a **Cursor** team member, offered direct assistance via **DM** to resolve the billing issue quickly.
- **Call for Open Source Model Evaluations**: A member suggested that **Cursor** should open source evaluations for different models, emphasizing that model implementation matters more than raw benchmarks.
   - They proposed tool-based evals, specifically asking if there was a Cursor-specific evaluation tool, as *"I find the way a model is implemented matters more than just raw benchmarks."


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1365183026241212477)** (9 messages🔥): 

> `Python interop capabilities, Mojo roadmap, Mac GPU support, Suspicious link shared` 


- **Python Interop Capabilities Release Soon!**: Members expressed excitement for the upcoming release of more **Python interop capabilities** as previously mentioned in the presentation.
- **Mojo Roadmap Remains Elusive**: The **Mojo roadmap** still has no definitive timeline, with a humorous mention of *someday*.
- **Mac GPU Support Coming This Summer?**: **Mac GPU support** is tentatively slated for *summer* according to an update from .duck_tape.
- **User Shares Suspicious Link!**: A member reported that another user shared a suspicious link and then deleted the message, attaching a screenshot for admin review ([image.png](https://cdn.discordapp.com/attachments/1098713601386233997/1365186580586954863/image.png?ex=680d0d54&is=680bbbd4&hm=01f44ee0ff956f8c66b1ce1e30c78f2234d5ae244fa2ea0e3531f0038c9aecf0&)).
   - Another member thanked them for flagging it, indicating that the moderation team is aware.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1365176336460546069)** (94 messages🔥🔥): 

> `uom crate in Rust, Nm vs Joules, QuantityKind tag, radian confusion, integer support` 


- **Rust's uom handles mixed unit operations**: The `uom` crate in Rust allows for operations mixing different units like **Energy**, **Force**, and **Length** without errors, demonstrated by adding `Energy` to the product of `Force` and `Length`.
   - A member suggested this is acceptable, similar to treating `Byte` as `UInt8`, indicating a permissive approach to unit interaction.
- **Debate on Nm vs Joules brews**: A discussion arose about whether **Newton-meters (Nm)** can be treated the same as **Joules (J)**, with one side arguing that physics distinguishes them because torque is a vector quantity, while energy is a scalar.
   - Referencing [BIPM](https://www.bipm.org/documents/20126/41483022/SI-Brochure-9-EN.pdf), it was highlighted that, *even though torque has the same dimension as energy (SI unit joule), the joule is never used for expressing torque*.
- **`QuantityKind` tag may disambiguate quantities**: A **QuantityKind** tag was proposed to differentiate between similar units, such as **Nm** and **Joules**, at the type level.
   - The implementation can be `Q[Vec[dtype, 1], N * M]` vs. `Q[Scalar[dtype], N * M]` where you can't add them.
- **Radian confusion is cleared**: Discussion touched on potential confusion around radians, referencing `uom`'s approach using a kind tag for angles, as seen in their [AngleKind trait](https://docs.rs/uom/latest/uom/si/marker/trait.AngleKind.html).
   - Wolfram's way is using `alias °: FloatLiteral = pi / 180`.
- **Integer support implementation questioned**: The team debated whether supporting integer values in the library was a mistake, considering potential issues with precision and rounding, especially in degree to radian conversions.
   - One member admitted that *engineers see numbers as being finite-precisioned*, and use **3.14** for pi.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1365415350749106338)** (1 messages): 

> `LM Studio Update, NVIDIA RTX 50-series Support, GLM-4 Enabled, New System Prompt Editor, Tool Choice Parameter` 


- **LM Studio Gets RTX 50-Series Boost**: **LM Studio 0.3.15** is out now, featuring support for **NVIDIA RTX 50-series** (CUDA 12.8) and can be updated via the app ([LM Studio download](https://lmstudio.ai/download)).
   - The update also enables **GLM-4** in llama.cpp and MLX, includes a new system prompt editor UI, and introduces the `tool_choice` parameter in the OpenAI-like API.
- **LM Studio Bolsters Llama 3, Adds Tooling**: LM Studio's latest release fixes to **Llama 3 Scout** prompt template + tool use ([full release notes](https://lmstudio.ai/blog/lmstudio-v0.3.15)).
   - Version 0.3.15 also includes a preview of community shared presets and addresses multiple bug fixes.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1365274312466108508)** (75 messages🔥🔥): 

> `Dataset Preparation for LLM Character, GGUF Conversion Script Issues, LM Studio vs Ollama Advantages, Google AIStudio Lora training, LM Studio community presets` 


- **Dataset Prep struggles for LLM Character**: A user is trying to make an LLM speak like a character in a game with a limited dataset (**10k tokens**) and seeks advice on loading the data into LM Studio.
   - They are unsure if they need to convert the data to **GGUF** format and what steps to take to test the model after the conversion.
- **GGUF conversion script triggers bugs**: A user reported that after using the [convert_hf_to_gguf.py script](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) and installing its requirements, their training scripts bugged out, even after reinstalling **transformers**.
   - They later fixed it by replacing `" "` with `" "` in their scripts, describing the fix as *"so weird"*.
- **LM Studio is easier than Ollama**: A user asked about the advantages of **LM Studio** over **Ollama**.
   - Another user responded that **LM Studio** is easier to use, especially for those unsure what a *"terminal"* is.
- **Google AIStudio can't run Lora**: A user inquired whether Google's **AIStudio** can train **LoRAs**, but another user clarified that it doesn't work with **LM Studio** because **LoRAs** must be merged with a base model and converted to **GGUF** to function correctly in **LM Studio**.
   - They clarified the proper workflow: *find a base model -> create lora -> merge with base model -> convert to gguf then it will work in lms*.
- **LM Studio Community Presets in Preview**: Users discussed the new **Community Shared Presets** feature in **LM Studio**, noting it's currently in *"Preview"* and lacks a discover tab or browsing functionality.
   - They expressed confusion about how to find and download presets, suggesting *discover -> presets* would be the most natural place to find them, with one stating *makes me question the intent*.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1365367883156422748)** (20 messages🔥): 

> `RTX 3060 vs Intel B580 for AI, AVX2 speeds with Xeon, OpenVINO vs llama.cpp` 


- **RTX 3060 is superior for AI workloads**: Members discussed the suitability of **RTX 3060 12GB** versus **Intel B580 12GB** for AI, where the consensus favored the **3060** due to better **CUDA** support.
   - One member stated that *Everything in the AI world is built around Cuda* which makes Nvidia's cards superior.
- **AVX2 speeds on Xeon is not speedy enough for 7-14b models**: A user asked whether **AVX2** speeds are good with a **2690v4 Xeon** (**14 core**, **3.5ghz boost**, **135watts**), to which another member responded that CPU-only won’t be speedy and depends on RAM.
   - Another member chimed in that **q4 14b** models will run at most **6tps** with dual xeon, perhaps **3-4** with just one.
- **OpenVINO offers better performance than llama.cpp on CPUs**: A member shared their [OpenArc project](https://github.com/SearchSavior/OpenArc), suggesting that **OpenVINO** provides significantly better **CPU** performance than **llama.cpp**.
   - They said that *anecdotally the difference is HUGE, especially ttftR* and noted that they have implemented vision for **Qwen2-VL**, **Qwen2.5-VL** and **Gemma3**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1365310061609484408)** (3 messages): 

> `Grayscale mode, ML Compilers, TVM/MLIR Project` 


- **Grayscale mode: renaming to Pantone?**: A member inquired about renaming *grayscale mode* to *pantone-scale*.
- **ML Compiler knowledge prerequisite?**: A member asked if knowledge of **ML compilers** is needed for a career in **ML Systems Programming**, specifically mentioning **TVM/MLIR**.
- **Undertaking a TVM/MLIR Project**: A member, primarily focused on compression, is contemplating a small project with **TVM/MLIR** and is unsure if it is a good idea.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1365349855752491059)** (4 messages): 

> `Quantization, FP4, 5090, Matmul Kernel` 


- **FP4 Kernel needs scaling elementwise multiplication**: After running **FP4** on a **5090**, a member realized that they need to fuse scaling elementwise multiplication into this **matmul kernel**.
   - Another member pointed out that you need a good algo to get accurate scales because it's symmetric quant; otherwise if you just use min-max the quality will probably be crappy.
- **FP4 representation needs scale_a and scale_b for elementwise multiplication**: A member mentioned that the representation of **FP4** is not enough for activation, so there would be a **scale_a** and **scale_b** (both vector), and will be elementwisely multiplied with the FP4 matmul result.
   - They followed up stating *"Even for the weights. I haven't tried it myself, I'm just guessing based on what I saw after quantizing and benchmarking many models"*.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1365358205177167933)** (2 messages): 

> `NVIDIA's cuda-python library, CUDA Functionality` 


- **NVIDIA's cuda-python Library**: A member asked about the quality of [NVIDIA's cuda-python library](https://github.com/NVIDIA/cuda-python) and whether anyone has tried it.
   - No specific experiences or reviews were shared in the messages.
- **CUDA Functionality Parity?**: A member inquired if the [cuda-python library](https://github.com/NVIDIA/cuda-python) offers the same functionality as **CUDA C/C++**.
   - The member expressed optimism about the library's potential if it achieves functional parity.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1365184365235146783)** (2 messages): 

> `MI300 ISA, VGPRs, wavefront, occupancy` 


- **MI300 ISA's VGPRs Explained**: In the **MI300 ISA**, the question was raised whether the **256 VGPRs/wave** translates to **256 effective GPRs per lane**.
   - The answer is yes, but it was cautioned that using **255 VGPRs** implies a low occupancy.
- **VGPR Usage and Occupancy Implications**: Using a high number of **VGPRs**, such as **255**, in the **MI300 ISA** can lead to lower occupancy.
   - Lower occupancy can impact performance due to reduced parallelism and increased overhead.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1365179439633272922)** (15 messages🔥): 

> `MI300, H100, amd-fp8-mm leaderboard, amd-identity leaderboard, vectoradd leaderboard` 


- **MI300 AMD-FP8-MM Personal Bests**: Multiple members achieved personal bests on the **MI300** for the `amd-fp8-mm` leaderboard, with times including **4.93 ms**, **2.74 ms**, **2.53 ms**, **5.19 ms**, **5.28 ms**, **805 µs**, and **5.25 ms**.
   - One submission reached **1247 µs**, while another secured **6th place** at **289 µs**.
- **AMD-Identity Leaderboard Success**: One member achieved a personal best of **23.7 µs** on the **MI300** for the `amd-identity` leaderboard.
- **H100 VectorAdd Personal Best**: A member achieved a personal best of **626 µs** on the **H100** for the `vectoradd` leaderboard.


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1365196189301080064)** (17 messages🔥): 

> `Profiling info, Personal best timing calculation, Kernel fails when benchmarking, Benchmark shapes, Azure MI300 compute time` 


- **Performance Profiling Investigated**: Members discussed the possibility of performance profiling support, pointing to a specific [Discord message](https://discord.com/channels/1189498204333543425/1343350424253632695/1364647936386142258) for **SSH access** and mentioning a potential revisit of **uprof**.
   - The conversation indicated that **uprof** was previously explored but may be reconsidered, with further discussion planned with a specific user.
- **Personal Best Timing Exposed**: A user inquired about how personal best timing is calculated, and was informed that it's an **average over all shapes**, with personal best indicating an improvement over the previous average.
   - The discussion linked to the [open-source evaluation infrastructure](https://github.com/gpu-mode/discord-cluster-manager) for additional context.
- **Kernel Crashes Despite Passing Tests**: A user reported a kernel passing all tests but failing during benchmarking, suspecting hidden tests, and received feedback that exit code **114** indicates a **timeout**.
   - The user included the output which indicated the benchmark command `python eval.py benchmark /tmp/tmp0uc5ldzn` returned exit code **114** on an **AMD Instinct MI300X** after 120s.
- **Benchmark Shapes Unbalanced**: A user struggling with benchmarking was directed to the [benchmark shapes](https://github.com/gpu-mode/reference-kernels/blob/b68a149bcd8701532eeedc774d27062429ce4f99/problems/amd/fp8-mm/task.yml#L66), noted for being possibly unbalanced.
   - Shapes were made available, however the user claimed they didn't appear obviously worse than the tests.
- **AMD GPU Time Vendor Suggestion**: A user seeking affordable MI300 compute time for profiling was informed of a competition reward of free **MI300** time for the top 8, as well as a suggestion to check out **Tensorweave**.
   - There was clarification that the reward referred to MI300 GPU time on a server, not the actual card, because *it's just hard to distributed GPUs at scale*.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1365236491088957555)** (5 messages): 

> `Prompt engineering updates, Zotero collection integration, Analyzing US Code with NotebookLM, File size limitations` 


- **Prompts Need Updating After Recent Changes**: A member mentioned that *many of their old prompts don't work anymore* due to recent updates, indicating a need to revisit and revise prompt engineering strategies.
- **Zotero Integration Sought for NotebookLM**: A user inquired about connecting their **Zotero** collection to **NotebookLM** without adding each PDF individually as a source.
- **US Code Analysis Bumps into Size Restrictions**: A member is analyzing the **US Code** for redundancies using **NotebookLM** but faces PDF and word count limits, even after trying code or XML files as Gemini suggested.
   - Another member suggested splitting the code into smaller files as a workaround.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1365180175809118329)** (31 messages🔥): 

> `Content Security Policies Errors, Gemini Chat Deletion, NotebookLM Use Cases, Source Uploading Errors, Downgrading from Plus to Free` 


- **CSP Errors Suspected in Gemini**: Users reported errors potentially related to **Content Security Policies (CSP)** in Gemini.
   - Another user was experiencing Gemini sometimes deleting the top conversation in the chat.
- **NLM: The Focused Expert Teacher**: One user suggests that a main use case is to learn something and NLM provides the option of creating a **Focused expert** for a specific area or topic that can act as a **teacher/coach/mentor**.
   - They added, *NLM can remove or limit the noise due to the sources being the truth behind the answers*.
- **File Size Limits Cause Upload Issues**: One user experienced an **Error** when uploading sources, where other users suggested the file may be larger than the **200MB limit**.
   - Another member stated *It's 26 Mb*
- **Free vs Plus Account Differences**: When asked about downgrading from **Plus** to **Free**, there was a question of what happens to notebooks with more sources than the free plan allows.
   - Another user stated **Free accounts** are limited to **100 Notebooks** and **50 sources** per notebook whereas **Plus accounts** can have **500 Notebooks**, and **300 sources** per notebook.
- **Workspace Admin Settings Limit Sharing**: A user with a **Plus account** through **Google Workspace** was unable to share a full notebook with others in their organization.
   - It was suggested that *Group sharing* may need to be enabled in the Workspace admin under *Groups for business*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1365348380217184317)** (26 messages🔥): 

> `Claude Desktop Image Uploads to MCP Tools, MCP Server Public Description Updates, JSON Formatting Errors with MCP Hooks, MCP Server Streamable Client Support, mcp-remote Usage and Supergateway Comparison` 


- **Claude's MCP tool Image Uploads**: A user is unsure how **Claude Desktop** passes uploaded images to **MCP tools**, questioning whether Claude automatically detects the most recent upload or if there's a specific referencing method.
   - The user's tool functions with image URLs but not direct uploads, seeking examples or documentation on implementing **MCP tools** that accept uploaded images.
- **MCP Public Description Glitch**: A user notes that the **public description** for their **MCP server** is not updating on the public facing page despite changes made in the admin panel, specifically referencing `dataryan.cph`.
   - Another user reports experiencing the same issue.
- **JSON Backtick Blooper**: A user reports that Claude insists on using **backticks** when creating **MCP hooks** to an SDK and wonders if they have misconfigured something.
   - Another user suggested trying to remove the quotes from `dataToWrite`.
- **MCP Streamable Client Quest**: A user is seeking a client that supports connections with **MCP server streamables**, as their server only runs with their own client and doesn't work with AnythingLLM, Claude, or VSCode Copilot.
   - One user mentions that *mcp-remote* supports mcphttps and links to the [draft version](https://github.com/geelen/mcp-remote/pull/32).
- **mcp-remote vs Supergateway**: A user asks if `mcp-remote` functions similarly to **Supergateway**, and another user confirms that it is similar but with a smaller scope.
   - A user shares a config for `mcp-remote` that's failing with the error *SSE error: Non-200 status code (404)*.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1365332515488796762)** (2 messages): 

> `Kubernetes MCP, GitHub Repo MCP, StacklokLabs mkp` 


- **Kubernetes MCP gets new Implementation**: A member announced they decided to build a new kubernetes MCP [based on the k8s API](https://github.com/StacklokLabs/mkp) to make it more general and flexible.
   - It is meant as an alternative to the first MCP Server for fetching GitHub repo, allowing **AI using repo code as refrences**.
- **GitHub Repo MCP**: A member presented his first MCP Server for fetching GitHub repo, and its code: [github-repo-mcp](https://github.com/Ryan0204/github-repo-mcp).
   - This allows **AI** to use repo code as references.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1365379745595785338)** (2 messages): 

> `CondoScan, LlamaIndex, LlamaParse, Agents` 


- **Agent Building Bonanza Booms!**: [AnthropicAI published Building Effective Agents](https://www.anthropic.com/), and **dexhorthy** went viral with his [12 Factor Agents](https://www.12factor.net/), and [OpenAI released A Practical Guide To Building Agents](https://platform.openai.com/docs/guides/function-calling)!
   - The community has exploded with **productive dialogue** on what agents are, and the best way to build them.
- **CondoScan Cuts Condo Costs Comprehensively**: **CondoScan** uses **LlamaIndex's agent workflows** and **LlamaParse's** accurate document processing to create a next-generation condo evaluation tool reducing document review time from weeks to minutes.
   - Read more about how **CondoScan** *Evaluates financial health and lifestyle fit* in [this case study](https://t.co/SzIbcKta1O).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1365262478413664278)** (19 messages🔥): 

> ``memory` vs `chat_history`, AgentWorkflow Errors, FunctionCallingAgent timeout, FunctionAgent timeout` 


- **Distinction between `memory` and `chat_history` Explored**: `chat_history` is integrated into the `memory`, and if you're managing chat messages or initializing with a message list, use `chat_history`; otherwise, employ a specific memory module.
   - In essence, *if you are only managing the list of chat messages yourself, or want to init with some list of chat messages, use chat_history; If you are maintaining/using a specific memory module, use memory*.
- **AgentWorkflow Error Resolved with GoogleGenAI Update**: An intermittent error in `AgentWorkflow` was reported, traced to the error: *400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Please ensure that the number of function response parts is equal to the number of function call parts of the function call turn.', 'status': 'INVALID_ARGUMENT'}}*.
   - The error was resolved by upgrading `llama-index-llms-google-genai` via `pip install -U llama-index-llms-google-genai`, as detailed in [this GitHub pull request](https://github.com/run-llama/llama_index/pull/18527).
- **`FunctionCallingAgent` Lacks Native Timeout**: The `FunctionCallingAgent` class lacks a direct timeout parameter; however, setting a timeout per LLM call on the LLM object is possible.
   - It was noted that [`FunctionCallingAgent`](https://llama_index.readthedocs.io/en/stable/understanding/agent/) is an older class, and newer agent classes like `FunctionAgent` directly support timeout via the `request_timeout` parameter.
- **`FunctionAgent` Offers Timeout Functionality**: The newer `FunctionAgent` class supports timeout configuration, demonstrated with a code snippet using `Ollama` and setting `request_timeout` to 360 seconds.
   - A member provided an example on how to use it:
```python
agent = FunctionAgent(
 tools=[multiply],
 llm=Ollama(model="llama3.1", request_timeout=360.0),
 system_prompt="You are a helpful assistant that can multiply two numbers.",
)
```


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1365214439602782228)** (8 messages🔥): 

> `DSPy support for multimodal models, Chain of Thoughts and ReACT pattern, Frameworks like Langchain and LlamaIndex, Multimodal Reasoning Models, Ember and Compound AI Systems` 


- **DSPy and Multimodal Modeling Musings**: The group pondered whether **DSPy** supports **multimodal models** and **multimodal reasoning workflows** given advances in reasoning models.
   - One member thought that **Chain of Thoughts (CoT)** is now more powerful and **ReACT** can make better decisions if they use **CoT** for tool choice.
- **Frameworks Face Fast-Paced Flux**: It was stated that the **AI landscape** is shifting quickly, making it hard for frameworks like **Langchain** and **LlamaIndex** to keep pace.
   - The advice was to use frameworks that make your job easier and allow you to directly use the **model API** if needed.
- **New Ideas Emerge: Ember**: New ideas are coming, like [Ember](https://github.com/pyember/ember), which might require thinking differently about building a **compound AI system**.
   - **DSPy** offers **declarative syntax**, while **Ember** offers a different approach, each with their own trade-offs.
- **Text and Graphs Get Business Boost**: Most frameworks focus on **text and graph** for business reasons; business-oriented systems often need to reason on text and tables.
   - Images are often converted into **text, JSON structures, or tables** using **VLM (for OCR)** in the pre-treatment phase.
- **Deep Dive into Code Analysis Demanded**: A member wanted a way to analyze large code files (out of context length) to extract the most out of it.
   - They specified that they wanted a deep level analysis, not generic requirements, aiming for the maximum that can be taken out of a file.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1365351789540409447)** (1 messages): 

> `Torchtune's GRPO code sharing` 


- **Sharing Torchtune's GRPO code requested**: A member asked how much work it would be to share the code for **Torchtune's GRPO**.
   - The member speculated the code could be really useful and expressed curiosity about how intrusive the changes are, suggesting other users of **Torchtune's GRPO** would be interested to see the code as well.
- **Torchtune GRPO discussion**: A user inquired about the possibility of sharing the code for Torchtune's GRPO.
   - They expressed interest in its potential utility and the extent of changes implemented, anticipating interest from other Torchtune GRPO users.


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1365415269148921906)** (4 messages): 

> `PPO Epochs, GRPO Data Padding` 


- **PPO Epochs' impact on KL divergence Estimation**: A member questioned whether `ppo_epochs > 1` in **Torchtune** makes the KL divergence estimation biased, pointing to line 85-87 of [this file](https://github.com/joecummings/r1-zero/blob/main/torchtune/dev/grpo/loss.py#L85-L87).
   - They argued that after seeing every sample in the replay buffer once, the policies (**pi_theta, pi_ref, pi_old**) are different, so the samples aren't from the current policy **pi_theta**.
- **GRPO's Data Padding Direction Discussed**: A member noted that the **GRPO data padding** direction is on the right side, and asked if the decoder model should pad on the left side during training.
   - Another member answered that padding can be done on both sides, as long as [input positions and masks are handled correctly](https://github.com/pytorch/torchtune/blob/main/recipes/dev/grpo_full_finetune_distributed.py#L750).


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1365330036961448078)** (5 messages): 

> `Memory Requirements for LLMs, Alternatives to Rust Coding, Llama4 and Scout 17x16` 


- **Low RAM Requirements can handle large LLMs**: A user suggests that **16GB** of RAM is sufficient for running **32B** models, and **70B** models can be run with **8GB** if one is tolerant of lower quality and speed.
   - They also noted that line-by-line LLM prompting is relatively simple to implement.
- **Shell Scripting as Rust Replacement**: A member suggested that Rust code can be replaced by short shell scripts, implying a simpler approach to certain tasks.
   - No additional details were given.
- **Llama4 Integration Stumbling Blocks**: A member inquired about getting **Llama4** to work, specifically with **scout 17x16**, questioning if code updates or **Jinja** configurations are necessary.
   - Another user responded that *gpt4all* is outdated and suggested exploring other options and the original member gave up.


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1365414595824717854)** (2 messages): 

> `Healthcare AI/ML startup seeks Prompt Optimization, Cohere Community Discord server Introductions` 


- **Healthcare Startup Hunts Beta Prompt Feature**: A healthcare **AI/ML startup**, experiencing rapid growth and focused on impactful work, has signed up for the Beta **Prompt Optimization** feature but lacks access.
   - The startup is inquiring whether access is restricted by **user seniority** or **billing tier**.
- **Newcomers urged to Introduce Themselves**: The server welcomes new members and encourages them to introduce themselves to the community.
   - New members are prompted to share their **company/industry/university**, current projects, preferred tech/tools, and goals for the community.

---


You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.




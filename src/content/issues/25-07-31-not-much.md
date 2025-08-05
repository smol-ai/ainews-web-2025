---
id: MjAyNS0w
title: Figma's $50+b IPO
date: '2025-07-31T05:44:39.731046Z'
description: >-
  **OpenAI**'s stealth model **horizon-alpha** on **OpenRouter** sparks
  speculation as a precursor to **GPT-5**, showing strong reasoning and SVG
  generation capabilities, comparable to **Gemini 2.5 Pro**. **Alibaba**
  released the **Qwen3-Coder** family, including a fast **Qwen3-Coder-Flash
  (30B-A3B)** variant with agentic features and 1M context length support via
  **UnslothAI**. **Cohere** launched **Command A Vision**, a 111B parameter
  open-weights vision-language model outperforming **GPT-4.1** and **Llama 4
  Maverick** on enterprise benchmarks. **Black Forest Labs** introduced **FLUX.1
  Krea [dev]**, an open-weights photorealism model compatible with fine-tuning
  tools like **diffusers** and **ostrisai**. **Zhipu AI** unveiled **GLM-4.5**,
  a hybrid reasoning open model with agentic capabilities available on
  **Together AI**. Discussions highlight the rising importance of
  **inference-time training** and **reasoning model generalization**. **Mistral
  AI** released the technical report for **Voxtral** continuing its open science
  efforts.
companies:
  - openai
  - openrouter
  - alibaba
  - unslothai
  - cohere
  - huggingface
  - black-forest-labs
  - diffusers
  - ostrisai
  - zhipu-ai
  - together-ai
  - mistral-ai
models:
  - horizon-alpha
  - gpt-5
  - gemini-2.5-pro
  - qwen3-coder
  - qwen3-coder-flash-30b-a3b
  - command-a-vision
  - gpt-4.1
  - llama-4-maverick
  - flux-1-krea-dev
  - glm-4.5
  - voxtral
topics:
  - reasoning
  - svg-generation
  - agentic-ai
  - context-windows
  - vision
  - fine-tuning
  - inference-time-training
  - model-generalization
  - open-models
  - technical-reports
people:
  - scaling01
  - teortaxestex
  - huybery
  - nickfrosst
  - aidangomez
  - reach_vb
  - zai_org
  - corbtt
  - jxmnop
  - teknuim1
---



**A happy outcome for a generational web platform.**

> AI News for 7/30/2025-7/31/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 5332 messages) for you. Estimated reading time saved (at 200wpm): 471 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As you know we try to keep things technical, but significant tech business stories do break through. The occasion of a new publicly listed software decacorn is both likely to solidify Figma's position as a web design platform (with [some AI work](https://www.youtube.com/watch?v=5q8YAUTYAyk&list=PLXDU_eVOJTx6rKQR1JEIktXodeHUawC_T&index=1)) and likely to mint lots of millionaires who will fund the next wave and cycle of tech.

![](https://resend-attachments.s3.amazonaws.com/HQsQzWhhkYI2wO3)

---

# AI Twitter Recap

**Model Releases, Updates, and Performance**

- **OpenAI's "Horizon-alpha" Sparks Speculation**: A new stealth model named **horizon-alpha**, available on **OpenRouter**, is generating significant buzz and is widely [speculated to be a new **OpenAI** model](https://twitter.com/scaling01/status/1950730582104604964), possibly a precursor to **GPT-5** or a "nano" version. [Initial testing by](https://twitter.com/scaling01/status/1950730792251891948) `@scaling01` suggested it was weak on benchmarks like LisanBench and not a reasoning model. However, subsequent tests with a reasoning mode enabled showed it is [casually capable of 20-digit multiplication](https://twitter.com/scaling01/status/1950949288521281820), processes thoughts for a [ridiculously long time](https://twitter.com/scaling01/status/1951048818897613069), and performs on par with or better than **Gemini 2.5 Pro** on [LisanBench](https://twitter.com/scaling01/status/1951068773869305999). The model also shows strong, if distinct, [SVG generation abilities](https://twitter.com/scaling01/status/1950847124146704780). `@teortaxesTex` notes it seems to excel at tasks involving "magic" and "ineffable soul," [smelling like a Sonnet killer](https://twitter.com/teortaxesTex/status/1951058712220549340).
- **Qwen3-Coder Family Released**: `@huybery` announced the release of **Qwen3-Coder**, a repo-scale coding model from **Alibaba**, which has seen significant community adoption and usage on platforms like **OpenRouter**. A smaller, faster version, **Qwen3-Coder-Flash (30B-A3B)**, was also [released for local users](https://twitter.com/huybery/status/1950925963979796877), offering basic agentic capabilities. The model is now [available in LM Studio](https://twitter.com/lmstudio/status/1950942293726503174) and can be run with **1M context length** via **UnslothAI**.
- **Cohere Releases "Command A Vision" VLM**: **Cohere** has entered the vision space with **Command A Vision**, a new [state-of-the-art 111B parameter open-weights Vision Language Model (VLM)](https://twitter.com/JayAlammar/status/1950931480349143259). As announced by `@nickfrosst`, the model weights are available on **Hugging Face** and it [outperforms models like GPT-4.1 and Llama 4 Maverick](https://twitter.com/aidangomez/status/1950927454383616343) on enterprise benchmarks.
- **FLUX.1 Krea [dev] for Photorealism**: **Black Forest Labs** has released **FLUX.1 Krea [dev]**, a new [state-of-the-art open-weights FLUX model](https://twitter.com/multimodalart/status/1950923544998658557) specifically built for photorealism. `@reach_vb` highlighted that it can be [run for free on ZeroGPU](https://twitter.com/reach_vb/status/1950948423986708525), and the creators noted that existing fine-tuning tools like those from **diffusers** and **ostrisai** [should work out of the box](https://twitter.com/multimodalart/status/1950932021817020867).
- **Zhipu AI Launches GLM-4.5**: `@Zai_org` announced **GLM-4.5**, a new open model that [unifies agentic capabilities](https://twitter.com/Zai_org/status/1950899064398364951). It is described as a hybrid reasoning model that can switch between "thinking" and "instant" modes and is [now available on Together AI](https://twitter.com/Zai_org/status/1950750962483675536).
- **Inference-Time Training and Reasoning Generalization**: `@corbtt` senses that **inference-time training** is [going to be a big deal soon](https://twitter.com/corbtt/status/1950705924684873988). Separately, `@jxmnop` inquired about examples of **reasoning model generalization**, such as a model trained on math problems getting better at creative writing. `@Teknium1` posits that a model will learn to do whatever improves its accuracy during its thinking process, [including hallucinating](https://twitter.com/Teknium1/status/1950865106725744913), referencing the null shot learning paper.
- **Mistral Releases Voxtral Technical Report**: In a continued commitment to open science, **Mistral AI** has [released the technical report for Voxtral](https://twitter.com/GuillaumeLample/status/1950855212677075122).
- **Step3 VLM Now Supported in vLLM**: `@vllm_project` announced that **Step3**, a fast and cost-effective VLM with **MFA & AFD**, is [now fully supported](https://twitter.com/vllm_project/status/1950954138541711802). `@teortaxesTex` notes that this model is [strongly multimodal and features a different in-house attention mechanism](https://twitter.com/teortaxesTex/status/1951008169989382218) than DeepSeek-V3.

**AI Tooling, Frameworks, and Infrastructure**

- **LangChain Introduces Deep Agents and Align Evals**: `@hwchase17` from **LangChain** explained the concept of **Deep Agents**, which combine a planning tool, file system, sub-agents, and a detailed system prompt, and [provided a video overview](https://twitter.com/hwchase17/status/1950989844936794511). The team also released **Align Evals**, inspired by work from `@eugeneyan`, to make it easier to [build and align LLM-evaluators](https://twitter.com/Hacubu/status/1950741838396027168).
- **Infrastructure and Deployment Advances**: **Microsoft and OpenAI** announced **Stargate Norway**, a new datacenter initiative. `@modal_labs` introduced **GPU snapshotting**, allowing for a [5-second cold-start of vLLM](https://twitter.com/akshat_b/status/1950967605121962164), a feature `@sarahcat21` called out as a feat of engineering. The **vLLM** project also highlighted that it will be featured in [5 talks at the PyTorch Conference 2025](https://twitter.com/vllm_project/status/1950821700679192654).
- **Funding for Developer Tools**: **Cline**, an open-source code agent, announced it has [raised $32M in Seed and Series A funding](https://twitter.com/cline/status/1950973599185248304), a story also covered by Forbes. `@sama` praised the founders' partnership, calling their story [remarkable](https://twitter.com/sama/status/1950936581810041143).
- **RAG, Context Engineering, and Data Quality**: The term **Context Rot** was highlighted as an [excellent and useful term by](https://twitter.com/jxmnop/status/1950678527550054848) `@jxmnop`. **DeepLearningAI** provided a technical breakdown of how transformers [process augmented prompts in RAG systems](https://twitter.com/DeepLearningAI/status/1950979807623139539). `@Teknium1` pointed out that a [large portion of a dataset was missing user turns](https://twitter.com/Teknium1/status/1950756952125972558), emphasizing the need to check data quality.
- **Hugging Face Launches "Tracks"**: `@_akhaliq` shared the launch of **Tracks**, a [100% open-source library from Hugging Face for experiment tracking](https://twitter.com/_akhaliq/status/1950617338136383605), positioned as an alternative to paid services.

**AI-Generated Media and Content**

- **Runway Aleph is Fully Released**: `@c_valenzuelab` announced the full rollout of **Runway Aleph** to all paid plans, describing it as a [completely new way of creating with AI](https://twitter.com/c_valenzuelab/status/1950920825185402986). A demo showed its capability for complex environment changes while [maintaining character consistency](https://twitter.com/c_valenzuelab/status/1951002926555734337). The release is part of a rapid series of updates from Runway in 2025.
- **Google Launches Veo 3 Fast and New Capabilities**: **Google DeepMind** announced that **Veo 3 Fast**, a quicker and more cost-effective text-to-video model, along with new [image-to-video capabilities for Veo 3](https://twitter.com/GoogleDeepMind/status/1950960418286940312), are now available in the Gemini API.
- **Midjourney's "Midjourney TV" Experiment**: `@DavidSHolz` described the new **Midjourney TV** experiment as [weirdly hypnotic](https://twitter.com/DavidSHolz/status/1950692691005657415). The feature provides a live stream of trending videos generated by the community.
- **Amazon Backs "Showrunner," the Netflix of AI**: It was reported that **Amazon** is investing in **Showrunner**, an AI-generated streaming service that [lets users generate scenes from prompts](https://twitter.com/TomLikesRobots/status/1950647978118488072). The platform is being developed by **Fable Simulation**, which originated the South Park AI experiment.

**Industry, Funding, and Geopolitics**

- **The US vs. China AI Race**: `@AndrewYNg` penned a detailed thread arguing that there is now a [path for **China** to surpass the U.S. in AI](https://twitter.com/AndrewYNg/status/1950941108000964654), citing its vibrant open-weights model ecosystem and aggressive moves in semiconductors. He notes that while top proprietary models are from the US, top open models often come from China. `@carlothinks` echoed this, quoting an ex-Alibaba CTO who claimed, ["China is building the future of AI, not Silicon Valley."](https://twitter.com/glennko/status/1950642750916792580)
- **Figma Goes Public**: **Figma** officially went public, with co-founder `@zoink` expressing [immense gratitude](https://twitter.com/saranormous/status/1950952597369577967). The event was marked by the **NYSE** tweeting ["Shipped: $FIG"](https://twitter.com/saranormous/status/1950952198340325798). `@saranormous` and `@sama` shared congratulatory messages.
- **Meta's Vision and M&A Activity**: **Mark Zuckerberg** shared **Meta's** vision for the future of ["personal superintelligence for everyone"](https://twitter.com/ylecun/status/1950660512967979245). Separately, `@steph_palazzolo` reported that Meta is on an M&A spree, having held talks with [video AI startups like **Pika**, **Higgsfield**, and **Runway**](https://twitter.com/steph_palazzolo/status/1951001998272372790).
- **Perplexity AI Launches Comet Shortcuts**: `@AravSrinivas` announced **Perplexity Comet Shortcuts**, which allow users to [automate repetitive web workflows with natural language prompts](https://twitter.com/AravSrinivas/status/1950981234554970382). One powerful example is the `/fact-check` shortcut.
- **AI Policy and Regulation**: It was reported that **Google**, **Anthropic**, **OpenAI**, and others will sign the **EU AI code of practice**. `@DanHendrycks` clarified that **xAI** is only signing the [safety portion, not the copyright portion](https://twitter.com/DanHendrycks/status/1950831617972519057). Meanwhile, `@qtnx_` noted a widespread global push for [ID age verification to access the internet](https://twitter.com/qtnx_/status/1950805548900966777).

**Broader Discourse and Developer Culture**

- **Developer Experience and Craftsmanship**: `@ID_AA_Carmack` posted a highly-trafficked reflection on the value of [rewriting an RL agent from scratch without looking at prior code](https://twitter.com/ID_AA_Carmack/status/1950621870463873448), noting it's a blessing when scale allows for it. `@ClementDelangue` shared a heartfelt message of thanks to [researchers who fight for open science and releasing open models](https://twitter.com/ClementDelangue/status/1950927952641749194), acknowledging the internal battles they often face in big tech.
- **Critiques of "Enshittification" and Past Tech Failures**: `@jxmnop` offered a counter-narrative to software "enshittification," arguing that, on the whole, [things seem to get slowly and consistently better](https://twitter.com/jxmnop/status/1950689279908450665), citing improvements in phone performance, internet speed, and transit apps. In a separate discussion, `@jeremyphoward` and `@random_walker` amplified critiques of the **DOGE** (Decentralized Organization for the Greater Good) project, with one commenter calling it a [failure at every single possible level](https://twitter.com/zacharynado/status/1950741189612720310) that crippled medical research while also failing to deliver its stated goals.
- **The Stanford NLP Legacy**: The founders of **Stanford NLP** won both **2025 ACL Test of Time awards**: the 25-year award to Gildea & `@jurafsky` for "Automatic Labeling of Semantic Roles," and the 10-year award to `@lmthang`, `@hyhieu226` & `@chrmanning` for ["Effective Approaches to Attention-based NMT"](https://twitter.com/stanfordnlp/status/1950644405821489532).

**Humor and Memes**

- **Tech Absurdity**: `@RonFilipkowski` joked that every DUI defense lawyer [hit the jackpot](https://twitter.com/code_star/status/1950640352517599615). `@lauriewired` pointed out that a bank **ACH** transaction is literally just an **SFTP** upload of a [940-byte ASCII text file](https://twitter.com/jeremyphoward/status/1950629141084271053). `@zacharynado` shared a comment explaining a failed Australian rocket launch by noting the engineers probably [forgot to account for Australia being upside down](https://twitter.com/zacharynado/status/1950964777934610750).
- **AI Life**: `@mlpowered` retweeted `@claudeai`'s simple reply, ["You're absolutely right."](https://twitter.com/mlpowered/status/1950685743061647391). `@typedfemale` compared a conversation to [replying to someone on Tinder](https://twitter.com/typedfemale/status/1950774437881745919). `@aidan_mclau` posted a video of a [chaotic Waymo trip](https://twitter.com/aidan_mclau/status/1950759916945482183).
- **Industry Commentary**: `@nearcyan` is having flashbacks to ['23](https://twitter.com/nearcyan/status/1950681931416556055). `@code_star` remarked it's [time to move 10^98 parquet files](https://twitter.com/code_star/status/1950639928242770330).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-Coder-30B-A3B and Flash Model Announcements and Benchmarks

- [**🚀 Qwen3-Coder-Flash released!**](https://i.redd.it/p7fpia2bz7gf1.jpeg) ([Score: 1197, Comments: 256](https://www.reddit.com/r/LocalLLaMA/comments/1me31d8/qwen3coderflash_released/)): **The image promotes Qwen3-Coder-Flash, specifically the** `Qwen3-Coder-30B-A3B-Instruct` **model, designed for lightning-fast and accurate code generation. It boasts a native** `256K context window` **(extendable up to** `1M tokens` **with YaRN), and is optimized for integration with platforms like Qwen Code, Cline, Roo Code, and Kilo Code. The post highlights function-calling, agent workflow support, and provides links to deployment resources on HuggingFace and ModelScope. Top comments discuss availability of GGUF-format models (including 1M context versions and Unsloth optimizations), fixes to model sharding and tool-calling, plus active community development and API access details.** Commenters praise the rapid evolution and open-source nature of the ecosystem, drawing attention to continuous fixes and strong community support. There is also enthusiasm for enhanced accessibility and technical improvements in recent model releases.
    - The release includes Dynamic Unsloth GGUFs of Qwen3-Coder-30B-A3B-Instruct, with standard and 1 million token context-length versions available on Hugging Face. There were fixes for tool-calling in both the 480B and 30B models (notably '30B thinking') and users are advised to redownload the first shard due to these updates. Comprehensive setup guides for local deployment are also provided by Unsloth, facilitating wider user experimentation and custom deployments.
    - Qwen-Code continues to improve post-launch, with several recent issue fixes and an active maintenance roadmap. For users in China, accessibility is enhanced via ModelScope APIs that provide 2,000 free API calls daily, and a free Qwen3-Coder API is also available through OpenRouter, broadening access and experimentation with the model. The main Qwen-Code repo remains at https://github.com/QwenLM/qwen-code, with active community engagement and patching.
- [**Qwen3-Coder-30B-A3B released!**](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) ([Score: 433, Comments: 83](https://www.reddit.com/r/LocalLLaMA/comments/1me2zc6/qwen3coder30ba3b_released/)): **Qwen3-Coder-30B-A3B, a large language model optimized for agentic coding applications (e.g., Qwen Code and Cline), has been released. Notably, the model omits "thinking tokens," suggesting a design focused primarily on direct code generation rather than step-by-step reasoning, which may impact tracing or interpretability for certain agentic tasks.** Commenters note the absence of thinking tokens, with speculation that the model will integrate well for agentic use-cases like Roo Code. There is also interest regarding the availability of a GGUF (quantized) version for easier deployment.
    - A discussion highlights that despite the model's lack of explicit Fill-In-the-Middle (FIM) support, users report that FIM capabilities are present, albeit not as robust as in Qwen2.5-Coder-7B/14B. This suggests partial FIM compatibility, which may impact workflows depending on heavy code infill or agentic coding tasks.
    - It's noted that the model is designed for agentic coding use cases (like Qwen Code, Cline), implying targeted optimization for multi-step reasoning or tool-use scenarios, which may differentiate its real-world coding utility from generalist models.
- [**I made a comparison chart for Qwen3-Coder-30B-A3B vs. Qwen3-Coder-480B-A35B**](https://i.redd.it/l6547uel88gf1.png) ([Score: 207, Comments: 16](https://www.reddit.com/r/LocalLLaMA/comments/1me4i2h/i_made_a_comparison_chart_for_qwen3coder30ba3b_vs/)): **The image is a radar chart comparing several technical benchmarks between Qwen3-Coder-30B-A3B (Flash) and Qwen3-Coder-480B-A35B. It shows that on agent capability tests ("mind2web" and "BFCL-v3"), both models perform similarly, suggesting parity in those tasks. However, there are notable performance gaps on programming-focused evaluations (Aider-Polyglot and SWE Multilingual), where the 480B variant outperforms the 30B. These insights suggest that while agent/decision tasks are competitive across sizes, pure coding ability significantly improves with larger model size. [View image](https://i.redd.it/l6547uel88gf1.png)** Commenters discuss that a dense Qwen3 32B model might close the gap seen in coding benchmarks, and express interest in comparisons with GPT-4.1 or o4-mini to contextualize these results.
    - Multiple users request the inclusion of the dense Qwen3 32B model in the comparison, noting that while it is not strictly a coding-specialized model, it performs very well at coding tasks. This suggests interest in understanding how dense architectures compare to mixture-of-experts (MoE) approaches in the Qwen3 family.
    - A user provides practical performance metrics for Qwen3-Coder-30B-A3B, observing that it achieves approximately `90 tokens/second` on Apple M4 Max hardware. They argue that the speed and lower hardware requirements make the 30B model more appealing than the much larger 480B version, given the dramatically increased parameter count (16x) for relatively modest performance gains.
    - There is a request for comparative benchmarks against proprietary models, specifically OpenAI's GPT-4.1 and o4-mini, indicating a desire for cross-family benchmarking using similar datasets or tasks for a better understanding of where open-source models stand relative to industry leaders.

### 2. Chinese Open-Source AI Model Momentum and Global Rankings

- [**Unbelievable: China Dominates Top 10 Open-Source Models on HuggingFace**](https://www.reddit.com/r/LocalLLaMA/comments/1mdsjn2/unbelievable_china_dominates_top_10_opensource/) ([Score: 756, Comments: 135](https://www.reddit.com/r/LocalLLaMA/comments/1mdsjn2/unbelievable_china_dominates_top_10_opensource/)): **July saw a surge in Chinese open-source AI model releases on HuggingFace, with models such as Kimi-K2, Qwen3, GLM-4.5, Tencent's HunyuanWorld, and Alibaba's Wan 2.2 dominating the platform's trending list. The post contrasts this with Meta's recent announcement to move toward more closed-source strategies, highlighting a reversal in openness between Chinese and Western AI ecosystems, with Chinese models now leading in open-source momentum on HuggingFace (see Hugging Face trending models).** Top comments debate the recent contributions from the West, notably citing only Mistral as a significant model, and suggest a paradox where China is currently more open in AI development than the West, attributed to shifting competitive dynamics and strategic openness.
    - Several commenters highlight that major recent open-source model contributions from the West are perceived as limited, with only Mistral mentioned by name and not consistently ranked at the very top of HuggingFace's leaderboards. This underscores a view that Western open-source progress is stagnating compared to China's current momentum.
    - A discussion develops around Meta's (Facebook's) and other tech giants' strategies, with criticism that planned top-tier models—such as those from Meta—may be restricted for internal use only rather than released openly, drawing negative comparisons to the approach taken historically by Amazon with proprietary innovations. This trend is seen as moving away from open-source principles in favor of company-internal deployment, further reducing public access to cutting-edge AI technology.
- [**Chinese models pulling away**](https://i.redd.it/727keqreo3gf1.png) ([Score: 1121, Comments: 133](https://www.reddit.com/r/LocalLLaMA/comments/1mdmsu9/chinese_models_pulling_away/)): **The post discusses the rapid progress and performance improvements of Chinese language models compared to Western offerings, particularly highlighting models like Qwen3-30B-A3B. The image (not provided in description, but inferred from comments and context) likely features benchmark or comparison charts showing how Chinese models are outperforming widely used Western models like LLaMA and Mistral in local Large Language Model (LLM) deployments. The discussion also references the migration of users from LLaMA-based models to newer Chinese-developed alternatives, due to better performance or reduced censorship.** Commenters debate whether the shift to Chinese models means abandoning communities like r/LocalLLaMA, with one emphasizing that Mistral models still receive significant attention, highlighting ongoing diversity in LLM preference based on use-case and community engagement.
    - A user outlines their progression through various local large language models: starting with LLaMA 3.1-3.2, moving to Mistral 3 Small and its variations (notably the less-censored Dolphin via R1 distillation), and ultimately adopting the Qwen3-30B-A3B model. This sequence highlights rapid switching as Chinese models like Qwen3-30B-A3B gain traction due to capability and tuning options.
    - Discussion notes Mistral's ongoing popularity in r/LocalLLaMA, disputing the narrative that users are abandoning non-Chinese models entirely. Mistral's active community engagement and model updates keep it relevant for localized language tasks.
    - A technical comment references Mistral's multiple small model releases within the month and anticipates the impact of an upcoming Mistral Large update, indicating continued development and competitive positioning against emerging Chinese models.
- [**Everyone from r/LocalLLama refreshing Hugging Face every 5 minutes today looking for GLM-4.5 GGUFs**](https://i.redd.it/f5iqhqp7z6gf1.jpeg) ([Score: 343, Comments: 71](https://www.reddit.com/r/LocalLLaMA/comments/1mdykfn/everyone_from_rlocalllama_refreshing_hugging_face/)): **The image is a meme satirizing the r/LocalLLaMA community's anticipation for the release of GLM-4.5 GGUF files on Hugging Face, as technical users await their availability for local inference. Commenters clarify that GGUF conversion for GLM-4.5 is still being debugged in llama.cpp (see draft PR [#14939](https://github.com/ggml-org/llama.cpp/pull/14939)), and current uploads are not reliable. Users interested in experimenting with GLM-4.5 are advised to try the [mlx-community/GLM-4.5-Air-4bit](https://huggingface.co/mlx-community/GLM-4.5-Air-4bit) version for MLX-based workflows while GGUF support is finalized.** Discussion highlights the lack of stable GGUF conversion for GLM-4.5 and the interim use of alternative backends like MLX, with some users prioritizing other models (e.g., Qwen3-Coder-30B-A3B-Instruct).
    - Support for GLM-4.5 GGUF in llama.cpp is still under development, with the main pull request ([github.com/ggml-org/llama.cpp/pull/14939](https://github.com/ggml-org/llama.cpp/pull/14939)) still in draft status. Current GLM-4.5 GGUF models may have conversion issues and are not considered stable; they should not be used until the implementation is finalized.
    - For users who can run models with MLX (e.g., via LMStudio), there's a working version of GLM-4.5 Air in 4-bit quantization already available from the MLX community ([huggingface.co/mlx-community/GLM-4.5-Air-4bit](https://huggingface.co/mlx-community/GLM-4.5-Air-4bit)), which has shown good performance in agentic coding tasks during community testing.
    - Unsloth GGUFs are best supported when using Unsloth's fork of llama.cpp, as it contains tailored code matching their quantization and GGUF implementation, improving compatibility and likely reducing conversion issues.

### 3. Upcoming and Potential Benchmark Innovations: Deepseek ACL 2025

- [**Deepseek just won the best paper award at ACL 2025 with a breakthrough innovation in long context, a model using this might come soon**](https://arxiv.org/abs/2502.11089) ([Score: 506, Comments: 35](https://www.reddit.com/r/LocalLLaMA/comments/1mdn6dp/deepseek_just_won_the_best_paper_award_at_acl/)): **Deepseek recently received the Best Paper award at ACL 2025 for a novel approach to long-context handling, likely centered around sparse attention mechanisms which improve scalability and efficiency in transformer architectures. This innovation could enable smaller language models to maintain longer and more effective context windows, addressing known limitations in existing models' ability to track long dependencies (see [Deepseek's sparse attention research](https://arxiv.org/abs/2402.04038) for background).** Commenters highlight that this work demonstrates genuine innovation from Deepseek beyond accusations of cloning, with several emphasizing that sparse attention is a major optimization likely to influence future LLM scalability and context retention, especially for smaller models.
    - Sparse attention has been highlighted as a major optimization strategy for long-context models, with commenters noting its potential to vastly improve efficiency and scale compared to standard dense attention approaches. This is seen as a key driver behind recent advances such as Deepseek's innovation.
    - The breakthrough is expected to help smaller models retain context much better as input length increases, directly addressing a weakness in current architectures where context retention typically degrades with length expansion. This has technical implications for memory usage and performance scaling.
    - There is speculation on whether Deepseek's advances could bring its models up to the performance tier of leading-edge systems like Gemini, particularly on specialized evaluation benches such as fiction.livebench. Such performance comparisons are regarded as a critical technical benchmark in the current LLM landscape.
- [**AMD Is Reportedly Looking to Introduce a Dedicated Discrete NPU, Similar to Gaming GPUs But Targeted Towards AI Performance On PCs; Taking Edge AI to New Levels**](https://wccftech.com/amd-is-looking-toward-introducing-a-dedicated-discrete-npu-similar-to-gaming-gpus/) ([Score: 273, Comments: 48](https://www.reddit.com/r/LocalLLaMA/comments/1mdx65u/amd_is_reportedly_looking_to_introduce_a/)): **AMD is reportedly exploring a dedicated discrete NPU (Neural Processing Unit) for PCs, aiming to deliver high AI performance as a standalone PCIe card, distinct from gaming GPUs. This approach could enable higher memory capacities (potentially 64-1024GB VRAM) for AI workloads and offload inference/LLM tasks from traditional GPUs, following directions similar to products like Qualcomm's Cloud AI 100 Ultra. AMD's current consumer AI stack, such as Strix Point APUs and XDNA engines, already supports large models (e.g., 128B parameter LLMs) for edge AI, but this would mark a shift for broader consumer/professional NPU deployment. [Full details](https://wccftech.com/amd-is-looking-toward-introducing-a-dedicated-discrete-npu-similar-to-gaming-gpus/).** Comments highlight the potential of dedicated AI NPUs in alleviating GPU bottlenecks for gaming and AI, as well as skepticism around AMD's software maturity (e.g., concerns about ROCm support catching up to hardware capabilities).
    - Dedicated NPUs could offload AI tasks from GPUs, enabling higher gaming performance (e.g., high-FPS 4K with AI-enhanced NPCs) by separating resources for gaming and AI workloads. Scalability in VRAM (up to 1TB) would benefit users needing large models or datasets locally.
    - There is consensus that strong driver and ML framework support is crucial; without robust ROCm (or equivalent) software, discrete NPUs would be hamstrung regardless of hardware performance. ROCm 7.0 is mentioned as a potential improvement, but maturity is still a concern.
    - Discussion highlights market segmentation: AMD could capture a new professional or prosumer segment left underserved by NVIDIA’s current focus, especially if AMD offers consumer NPUs with large memory and competitive performance per watt, bypassing the artificial segmentation seen in gaming GPUs (as with NVIDIA’s datacenter vs consumer product strategies).
- [**I built a local alternative to Grammarly that runs 100% offline**](https://v.redd.it/pxb4pfgaw8gf1) ([Score: 229, Comments: 61](https://www.reddit.com/r/LocalLLaMA/comments/1me7yia/i_built_a_local_alternative_to_grammarly_that/)): **The OP introduces '[refine.sh](http://refine.sh/)', a local Grammarly alternative leveraging the Gemma 3n E4B model for offline grammar checking, with a peak memory footprint under 500MB and 300MB when idle. The tool is in early development and operates fully offline, addressing privacy and local resource constraints.** Commenters provide alternative suggestions such as the FOSS [WritingTools](https://github.com/theJayTea/WritingTools), and raise concerns about the tool not being open source (FOSS).
    - One commenter notes that using LLMs for grammar correction is generally ineffective due to their difficulty in fine-tuning for grammar-specific tasks. They mention that Grammarly's recent switch to an LLM backend is causing problems, implying that rule-based systems or targeted NLP models may outperform general-purpose LLMs in this use case.
    - There are references to alternative open-source (FOSS) tools aimed at grammar correction. Notably, [WritingTools](https://github.com/theJayTea/WritingTools) and [Write with Harper](https://writewithharper.com/) are mentioned as free and open-source projects, with the latter emphasizing strict adherence to documented grammatical rules from style guides rather than relying on unconstrained LLM output.
- [**Junyang Lin is drinking tea**](https://i.redd.it/s3pv80fee7gf1.png) ([Score: 212, Comments: 31](https://www.reddit.com/r/LocalLLaMA/comments/1me095p/junyang_lin_is_drinking_tea/)): **The post, titled 'Junyang Lin is drinking tea,' includes an image that, in the absence of image analysis and direct technical content in the description, relies on contextual hints. The comments reference fast token generation speeds ("getting 120tok/s out of 30ba3b"), suggesting this could be a meme or informal nod to model developer Junyang Lin and the efficiency of one of their models, like 30B A3B (possibly a Llama or variant). No direct benchmarks, code, or technical implementation are present in the post itself.** Commenters express enthusiasm and highlight performance—specifically 120 tokens per second output—implying satisfaction with recent advancements or releases from Junyang Lin, and underlining the community's demand for efficient, powerful models.
    - One user highlights generating `120 tokens/second` using the 30B a3b model, which is a notably high inference speed for a 30B parameter model. This points to either highly optimized inference code or robust hardware support.
    - The poem references GLM 4.5 Air and Qwen3 Coder 30B a3b models, suggesting users are benchmarking multiple recent large language models. The explicit mention of 30 billion weights and silicon/GPU resources alludes to the significant computational demands and scale of these models.Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. OpenAI GPT-5 and Stealth Model Developments

- [**Google has now indexed an OpenAI docs page for GPT-5 in yet another sign of preparations for its release - the page 404s for now**](https://i.redd.it/2mfiosbhf7gf1.png) ([Score: 525, Comments: 103](https://www.reddit.com/r/singularity/comments/1me0e39/google_has_now_indexed_an_openai_docs_page_for/)): **The image shows a Google Search result indexing an official OpenAI documentation page titled 'OpenAI API Docs — GPT-5' (URL: https://platform.openai.com/docs/guides/gpt/gpt-5), but the actual page currently returns a 404 error. This event suggests that OpenAI is preparing to publicly update or release new documentation for GPT-5, indicating imminent progress towards its formal announcement or rollout. The documentation's appearance in Google's index is interpreted as an early sign of backend preparations for GPT-5, further fueling speculation about the timing of its release.** Comments speculate on the timing of the GPT-5 release, with some users expressing skepticism about an imminent launch and others urging patience. There is no deep technical debate, mostly anticipation and discussion about OpenAI's release cadence.
    - 
- [**'the codenames OpenAI is supposedly using for GPT-5 models: "o3-alpha > nectarine (GPT-5) > lobster (mini) > starfish (nano)."' | '"...Zenith, Summit, Lobster, Nectarine, Starfish, and o3-alpha—that are supposedly outperforming nearly every other known model," have been spotted on LMArena.'**](https://tech.yahoo.com/ai/articles/gpt-5-launch-might-imminent-151237832.html) ([Score: 163, Comments: 25](https://www.reddit.com/r/singularity/comments/1me4vyf/the_codenames_openai_is_supposedly_using_for_gpt5/)): **Leaked internal codenames for hypothetical future OpenAI models ('o3-alpha', 'nectarine', 'lobster', 'starfish') and their speculated model sizes (e.g., lobster=mini, starfish=nano), were allegedly observed on the LMArena benchmark, performing above most other models. There is speculation these represent stages of GPT-5 or next-gen offerings, with claims the models were visible but are no longer present for comparison on public leaderboards.** Commenters question article credibility and technical accuracy; some seek clarification on the specifics of 'O3 alpha'. There is skepticism regarding the presence and performance of these models, as well as the reliability of the coverage.
    - A user notes that the models referenced (Zenith, Summit, Lobster, Nectarine, Starfish, and o3-alpha) are no longer present on the LMArena leaderboard, hinting at issues with model benchmarking continuity or removal of test entries. This may affect the reliability of current public model performance comparisons.
    - A commenter queries the identity and capabilities of "O3 alpha," indicating ongoing ambiguity about internal codenames, lineage, and architecture for OpenAI's unreleased or experimental models, highlighting opacity in the progression from o3-alpha to finalized GPT-5 variants.
- [**OpenAI's new stealth model on Open Router**](https://www.reddit.com/gallery/1mdmxpe) ([Score: 185, Comments: 58](https://www.reddit.com/r/singularity/comments/1mdmxpe/openais_new_stealth_model_on_open_router/)): **A new, unannounced OpenAI model has appeared on OpenRouter (screenshot: [preview.redd.it/pgmajpmcs3gf1.png](https://preview.redd.it/pgmajpmcs3gf1.png)), prompting speculation about a possible AGI-related release. Benchmarks suggest the model underperforms at math, failing relatively easy problems, but outperforms others on coding tests—particularly on edge case handling, though its overall code quality is mediocre. Comparative references note that Claude 4 Sonnet did worse on benchmark test problems than Claude 3.7, but surpassed it in real-world tasks, highlighting the limitations of evaluating models solely on narrow benchmarks.** Commenters debate the disconnect between benchmark performance (e.g., math/coding tests) versus practical usability and robustness in real-world coding, with several noting that catching edge cases may be more valuable than raw benchmark scores.
    - One user noted the new OpenAI stealth model performs poorly on mathematical tasks, failing even fairly easy problems, suggesting that despite the hype around new AI models achieving high scores on standard math benchmarks, this particular model falls short in those areas.
    - Another commenter observed that while this model delivers the best results on their typical coding problem set—particularly in managing edge cases—the general code quality remains mediocre. Furthermore, they highlighted that performance on small benchmark-style problems does not necessarily reflect the model's value in broader, real-world applications, referencing their experience where Claude 4 sonnet underperformed versus Claude 3.7 in limited tests but excelled in practical work scenarios.
    - Some discussion speculates that this new model could be an early form of "GPT-5 Nano," based on its mixed performance—described as both impressive and lacking in different contexts—and similarities between generated games and outputs from anonymous LM Arena models suspected to be GPT-5 family members. This feeds into the theory that OpenAI is quietly testing next-generation, smaller-sized models in production settings.
- [**New (likely) OpenAI stealth model on openrouter, Horizon Alpha, first try made this**](https://i.redd.it/jci2n61015gf1.png) ([Score: 198, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1mdsala/new_likely_openai_stealth_model_on_openrouter/)): **A user tested 'Horizon Alpha', a newly surfaced model on OpenRouter purportedly built by OpenAI, by prompting it to generate a detailed Mario Bros game replica with pixel art. The resulting image ([view here](https://i.redd.it/jci2n61015gf1.png)) showcases intricate, classic pixel art game elements, including a score/coins/world/time UI bar, reflecting notable fidelity to the original game's design elements. Comments focus on the technical decision of replicating a status bar UI and speculate whether model outputs are consistent across runs (reusing color/font/UI motifs) or if the design varies significantly between outputs.** Commenters critically question the model's ability to produce a fully-playable, multi-level game, discuss whether 'Horizon Alpha' could be an open source model, and analyze the consistency and style decisions of its UI recreations, highlighting potential variance in repeated generations.
    - One commenter compares this unknown "Horizon Alpha" model to GPT-4.1, stating that in their testing, it does not perform as well, suggesting it's behind in either output quality or capabilities relative to GPT-4.1.
    - There is a technical observation about the model's handling of UI elements: specifically, the score/coins/world/time bar was generated in a way that stands out from the rest of the pixel art. The commenter suspects the model may not fully integrate UI elements with the background and questions whether re-running the prompt would result in consistent UI styles, such as similar colors or fonts.
    - A user clarifies that the model is designed for text generation, not for coding tasks like building fully playable levels or games, setting expectations about the kind of output the model can generate.
- [**OpenAI's new stealth model (horizon-alpha) coded this entire app in one go!**](https://www.reddit.com/r/OpenAI/comments/1mdrmdm/openais_new_stealth_model_horizonalpha_coded_this/) ([Score: 122, Comments: 44](https://www.reddit.com/r/OpenAI/comments/1mdrmdm/openais_new_stealth_model_horizonalpha_coded_this/)): **The post discusses OpenAI's unreleased model, 'horizon-alpha', which allegedly generated a full application from a single prompt (demo image linked) using [OpenRouter's API](https://openrouter.ai/openrouter/horizon-alpha). The prompt used is lengthy and can be reviewed [here](https://gist.github.com/alsamitech/7b7b7b2faf4f5005c91fdba5430a6de1). The OP notes the model performs well, with some quirks, and is particularly fast at reading and processing large files compared to other models, exhibiting strong error detection capabilities.** Top comments question the prompt's appropriateness and necessity, expressing that simple instructions may work as well, and suggesting that complex prompt engineering should not be necessary. Another commenter highlights the model's exceptional speed and ability to identify subtle errors quickly, calling it a 'game changer' compared to current models.
    - One commenter notes that 'horizon-alpha' demonstrates extremely fast file reading capabilities, claiming it can handle entire files "in a blink," a speed which they state is unmatched by other models. The model also exhibits improved error detection, quickly spotting issues in projects that would be difficult to identify otherwise, indicating potential advancements in both speed and code analysis accuracy.

### 2. Wan2.2 and Flux: New Model Releases and Benchmarks

- [**wan 2.2 fluid dynamics is impressive**](https://v.redd.it/vzff5xwhu4gf1) ([Score: 292, Comments: 31](https://www.reddit.com/r/StableDiffusion/comments/1mdrld2/wan_22_fluid_dynamics_is_impressive/)): **The OP demonstrates fluid/particle simulation using WAN 2.2 (image-to-video, version 14b), with source images generated in Flux Dev and audio added via mmaudio. The focus is on evaluating WAN 2.2's handling of complex physical phenomena like fluids and particles, noting impressive results but highlighting ongoing challenges in controlling camera angles and motion via prompting.** One top comment raises a technical limitation: WAN 2.2 tends to generate persistent fluid flow from any initial liquid, e.g., a stationary teardrop results in continuous artificial flow, which is reported as a common unresolved issue.
    - A user describes a notable limitation of Wan 2.2's fluid simulation: the model tends to continuously generate fluid from the same spot if there's a trace present (e.g., a teardrop on an eye results in an unrealistic waterfall-like effect). This indicates a challenge in modeling fluid persistence versus initiation, highlighting potential issues with the model's temporal consistency or thresholding in identifying when fluid generation should cease.
- [**Wan 2.2 Reel**](https://v.redd.it/d2h632ni35gf1) ([Score: 173, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mdsl2s/wan_22_reel/)): **The post showcases a demo reel using the Wan 2.2 GGUFQ5 i2v model, with all images generated via SDXL, Chroma, Flux, or movie screencaps. The total time for generation and editing was approximately 12 hours, and outputs demonstrate the capabilities of the involved generative pipelines.** A key technical critique from commenters addresses the lack of *consistency* and *narrative coherence* in current AI-generated video, arguing the next technical challenge is producing "watchable story" rather than just visually impressive short clips. Some technical interest in the specifics of video generation is also raised (e.g., resolution and inference steps used).
    - Discussion highlights challenges in AI-generated video: specifically, the lack of consistency and narrative structure, with current technology producing disjointed 3-5 second clips rather than coherent, longer stories. This points to story and temporal coherence as active frontiers for research and implementation.
    - Technical comments touch on performance of different quantization and precision modes: an example is given using FP8 on an RTX 5080, where generating a 5-second 720p video takes around 40 minutes. The commenter plans to benchmark Q4 or use Unsloth's Dynamic Quant for potentially faster inference, highlighting the tradeoff between quality and generation speed.
    - A query is raised about the resolution and inference steps (likely diffusion steps) used for the demo videos, relevant for reproducibility and to compare speed versus quality across hardware and quantization approaches.
- [**Another "WOW - Wan2.2 T2I is great" post with examples**](https://www.reddit.com/gallery/1me5t5u) ([Score: 144, Comments: 34](https://www.reddit.com/r/StableDiffusion/comments/1me5t5u/another_wow_wan22_t2i_is_great_post_with_examples/)): **The post discusses image generation using the Wan2.2 T2I model, emphasizing that a 4K image took approximately 1 hour to generate. The user notes that the workflow leverages CivitAI's native T2I setup with LightX2V (0.4), FastWAN (0.4), and Smartphone LoRA (1.0), and observes that sampler and scheduler selection (e.g., euler) critically impacts color saturation and image realism. The workflow reportedly does not support resolution scaling with 'bong' (res2ly), highlighting limitations in scaling functionality.** A comment claims Wan2.2 surpasses the Flux model in realism (e.g., fewer anatomical artifacts), but highlights a lack of features comparable to ControlNet or Pulix for ensuring image consistency across generations. Another notes disappointment in the lack of reproducibility due to the incomplete workflow description.
    - A user reports that Wan 2.2 produces more realistic images with fewer anatomical errors (such as missing limbs or deformed fingers) compared to Flux, highlighting an improvement in image fidelity and coherence. However, they note the absence of features akin to ControlNet or Pulix, which would enable more consistent image generation and control over output, suggesting a gap in guided or reference-based generation capabilities.
    - A technical question is raised about model requirements: one commenter asks if the impressive results require the full Wan 2.2 model or if the lighter, fp8-scaled versions (~14GB) are sufficient, noting they observed 'super weird results' with the fp8 variant, implying potential limitations or compatibility issues with quantized/optimized versions.
- [**PSA: WAN 2.2 does First Frame Last Frame out of the box**](https://www.reddit.com/r/StableDiffusion/comments/1me4306/psa_wan_22_does_first_frame_last_frame_out_of_the/) ([Score: 117, Comments: 19](https://www.reddit.com/r/StableDiffusion/comments/1me4306/psa_wan_22_does_first_frame_last_frame_out_of_the/)): **The post announces that the WAN 2.2 model enables First Frame Last Frame (FLF) video output "out of the box" within ComfyUI by updating the existing WAN 2.1 FLF2V workflow with the new 2.2 models and samplers. The provided Pastebin link contains the modified workflow definition, highlighting ease of upgrade for those already using FLF2V (see: [Pastebin workflow](https://pastebin.com/kiG56kGa)).** Top comments question whether the model supports true video looping (first=last frame) or degenerates to a static image, and seek clarification on whether the order of intermediate nodes (e.g., `LoraLoaderModelOnly`, `TorchCompilerModel`, `Patch Sage Attention`, `ModelSamplingSD3`) affects output fidelity, as users report mixed results with different sequencing. There's also inquiry into the utility of this workflow for low frame rate (e.g., 4fps) video interpolation.
    - A user inquires about whether WAN 2.2 can properly generate looped videos by setting the first and last frame to the same image, asking if the model avoids the common issue with other video models that produce a static image rather than a seamless loop.
    - There's a technical discussion about the impact of workflow node order between the model loader and the KSampler in WAN 2.2 pipelines. Two specific node orderings are compared: one where LoraLoaderModelOnly comes first, and another where TorchCompilerModel is first. The commenter asks whether these variations affect sample quality or consistency, noting mixed results in their own tests.
    - A user questions the suitability of WAN 2.2 for interpolation tasks, such as generating intermediary frames for a low frame rate (e.g., 4fps) video, aiming to clarify the model's effectiveness for this specific use case.
- [**Text-to-image comparison. FLUX.1 Krea [dev] Vs. Wan2.2-T2V-14B (Best of 5)**](https://www.reddit.com/gallery/1mec2dw) ([Score: 123, Comments: 58](https://www.reddit.com/r/StableDiffusion/comments/1mec2dw/texttoimage_comparison_flux1_krea_dev_vs/)): **A user conducted an informal side-by-side comparison of the FLUX.1 Krea [dev] and Wan2.2-T2V-14B text-to-image generative models across 35 samples each, using long-form prompts (~150 words). FLUX.1 Krea was run at 25 steps with CFG lowered from 3.5 to 2, while Wan2.2-T2V-14B used the Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32 lora at 0.6 strength to accelerate inference, impacting output visual quality. Major findings: Wan2.2-T2V-14B produced significantly more usable (4/5) and naturalistic outputs than FLUX, which exhibited frequent anatomical errors and less natural stylization. Lighting accuracy was slightly better in FLUX, but its contrast was unnaturally high and it consistently failed to accurately render freckles.** Top comments strongly preferred Wan2.2-T2V-14B, succinctly condensing the consensus to 'wan won' and suggesting prompt tweaks (e.g., '(freckles:6)') for controlling features. The discussion lacks deeper technical debate but indicates perceptible quality differences in practical use.
    - Several users observe that the FLUX.1 Krea model may have been trained incorporating a significant number of MidJourney-produced images, particularly those with distinct features like freckles, raising questions about the novelty and originality of the training data compared to WAN2.2-T2V-14B.
    - Technical comparisons note that WAN2.2-T2V-14B produces images visually comparable to TV show captures, suggesting higher photorealism and possibly a superior dataset or diffusion architecture versus Flux.1 Krea. Some users express a switch in preference toward WAN, citing product licensing differences (e.g., frustration with Flux and "bfl non commercial license").
- [**New Flux model from Black Forest Labs: FLUX.1-Krea-dev**](https://bfl.ai/announcements/flux-1-krea-dev) ([Score: 381, Comments: 250](https://www.reddit.com/r/StableDiffusion/comments/1me2l80/new_flux_model_from_black_forest_labs_flux1kreadev/)): **Black Forest Labs has released the FLUX.1-Krea-dev model, available on [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev). It is advertised as a drop-in replacement for the original flux-dev, aiming to produce AI images that are less distinguishable as synthetic, though early user tests report existing flux-dev LoRAs are *not* compatible. Notably, the model has trouble rendering human hands correctly, frequently producing images with 4 or 6 fingers (see [sample output](https://preview.redd.it/ap6yq3fxx7gf1.png?width=642&format=png&auto=webp&s=d27724498982d17cb5fc2d5795b3758efe826825)).** Commenters suspect heavy content filtering/censorship in the model, and some express disappointment that compatibility with older LoRAs was not as advertised.
    - Discussion highlights that while FLUX.1-Krea-dev was advertised as a drop-in replacement for previous FLUX dev models (including compatibility with existing LoRAs), actual user testing revealed that these older LoRAs do not function as expected with the new release.
    - One technical issue identified with FLUX.1-Krea-dev is its continued difficulty rendering human hands accurately, with outputs sometimes producing 4 or 6 fingers—an artifact common in less refined image generation models.
- [**Flux Krea is quite good for photographic gens relative to regular Flux Dev**](https://www.reddit.com/gallery/1meann0) ([Score: 145, Comments: 57](https://www.reddit.com/r/StableDiffusion/comments/1meann0/flux_krea_is_quite_good_for_photographic_gens/)): **The post provides visual results from Flux Krea, a photographic generation model by Flux ([Krea.ai](http://krea.ai/)), and highlights its improved realism for photographic outputs compared to the standard Flux Dev model. There are no explicit benchmarks or technical details about model architecture, but the post focuses on qualitative output differences under different generative models.** Top commenters critique the presence of a pervasive yellow filter resulting in 'lifeless and cold' images, suggesting a need for more neutral color defaults to give users greater post-generation control. Another point raised is the lack of direct side-by-side comparisons with identical prompts, making technical evaluation of improvements difficult.
    - Several users point out a distinct yellow or cold filter applied by Flux Krea, with one stating the model's attempt at a photographic look results in images that appear 'lifeless' and suggest they should have kept color tones neutral for more user control.
    - A request is made for rigorous benchmarking, such as direct side-by-side comparisons between Flux Krea and regular Flux Dev using identical prompts and settings to accurately assess differences in photographic quality.
    - There is speculation about the model's possible improvements due to 'better datasets & captaining,' indicating community interest in technical details on training data or strategies that led to observable differences in output.
- [**FLUX Krea DEV is really realistic improvement compared to FLUX Dev - Local model released and I tested 7 prompts locally in SwarmUI with regular FLUX Dev preset**](https://www.reddit.com/gallery/1me4u0a) ([Score: 132, Comments: 53](https://www.reddit.com/r/StableDiffusion/comments/1me4u0a/flux_krea_dev_is_really_realistic_improvement/)): **The post compares the new FLUX Krea DEV model with the previous FLUX Dev, emphasizing improved photorealism, especially in tasks like dinosaur image generation using SwarmUI. Seven prompts were tested locally with the regular FLUX Dev preset to benchmark output quality. Key technical queries in comments focus on model realism (notably with 'realistic dinosaur' generation), inference speed improvements over previous versions, and model size/VRAM requirements, particularly concerning compatibility with an RTX 4080 GPU (16GB VRAM).** Technical discussion centers on whether the new Krea DEV model materially accelerates inference and generates realism surpassing precedent, especially in complex tasks like dinosaurs, with some skepticism expressed about current AI capabilities in this specific domain.
    - A commenter inquires about the generation speed compared to the previous FLUX Dev, specifically asking if FLUX Krea DEV is faster, which implies a community interest in performance improvements and inference time benchmarks between these two local models.
    - A technical question is raised regarding the model's VRAM requirements and hardware compatibility—particularly if FLUX Krea DEV can run on an RTX 4080 (16GB VRAM). This reflects concerns about local deployment feasibility and model size, which are critical for users running models locally on consumer GPUs.

### 3. Steampunk Video Game Concepts and Prompt Techniques

- [**Steampunk Video Games In European Cities (Prompts Included)**](https://v.redd.it/u0s6z1avl8gf1) ([Score: 410, Comments: 31](https://www.reddit.com/r/aivideo/comments/1me6d7s/steampunk_video_games_in_european_cities_prompts/)): **The OP shares detailed text-to-image prompts designed for generating high-fidelity steampunk video game concept art set in iconic European cities (Paris, London, Venice) using Prompt Catalyst. These prompts specify camera perspective (third/first person), resolution (**`2560x1440`**, ultrawide aspect), in-world UI (HUD with pressure dials, minibars, cooldown clockfaces, minimap, steam meters), and stylistic elements (sepia lighting, particle effects, mechanical themes, etc.), emphasizing dynamic environment features (smoke, fog, steam) and photorealistic asset styling (--ar 6:5 --stylize 400 prompt tokens). The generation pipeline and complete workflow are supported by an external tutorial on [Prompt Catalyst's site](https://promptcatalyst.ai/tutorials/creating-video-game-concepts-and-assets).** Commenters note the high quality of the visual output and UI design, suggesting that the generated concepts surpass expectations for the steampunk genre and evoke comparisons to 'The Order: 1886' (implying missed potential in previous commercial implementations). There is consensus that these tools and prompts could meaningfully influence actual game development workflows if adopted by industry professionals.
    - There is a mention of how the steampunk aesthetic showcased in these images serves as inspiration and a critique for existing game franchises like The Order: 1886, suggesting that more effective or imaginative implementation in this genre is possible, especially for games set in European cities.
    - A commenter references 'Bioshock Infinite' repeatedly, invoking it as a benchmark or exemplar of steampunk/alternate-history video game design, implying that it still represents a high standard for aesthetic and narrative integration in the genre.
- [**Steampunk Video Games In European Cities (Prompts Included)**](https://v.redd.it/u0s6z1avl8gf1) ([Score: 410, Comments: 32](https://www.reddit.com/r/aivideo/comments/1me6d7s/steampunk_video_games_in_european_cities_prompts/)): **The post details highly structured prompts for generating steampunk-themed video game visuals using Prompt Catalyst (tutorial: https://promptcatalyst.ai/tutorials/creating-video-game-concepts-and-assets). Prompts specify technical parameters such as: third-/first-person perspectives, 2560x1440 resolution, 21:9 aspect ratio, in-game UIs with custom pressure-dial health bars, ability icons as clock faces, and environment effects including volumetric fog, real-time particle effects, and sepia lighting to accentuate brass and mechanical textures in historic European settings. Notably, animation and asset prompts are designed for high fidelity and stylization (--ar 6:5 --stylize 400).** Top comments suggest missed opportunities in existing games (e.g., 'the order 18xy'), general approval of the quality, and a desire for the steampunk genre to gain popularity, but no deep technical debate is present.
    - Commenters discuss the underutilization of steampunk aesthetics in AAA titles, referencing games like "The Order 1886" as a missed opportunity for better execution. There's a focus on how current graphics and world-building capabilities could more effectively realize the atmosphere and gameplay depth this genre demands, especially set in richly detailed European cities.
    - One commenter highlights how titles such as "Bioshock Infinite" set high expectations for the integration of steampunk themes and immersive environments, suggesting that future games could surpass these benchmarks if genre popularity and investment increased.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: Models Muscle Up with New Releases**

- **Qwen3 Drops 30B Bombshell**: Alibaba's **Qwen3-30B** model rivals **GPT-4o** in benchmarks, running locally on 33GB RAM in full precision via [Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), with quantized variants needing just 17GB. Community excitement centers on its multilingual prowess, though tool use falters in some setups like vllm.
- **Gemma 3 Fine-Tunes Watermark Woes**: Fine-tuning **Gemma 3 4B** on 16k context removes watermarks and boosts stability, as shown in a [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&), sparking a new competition ending in 7 days via [Unsloth's X post](https://x.com/UnslothAI/status/1950553466591723640). Users report enhanced translation across popular languages, positioning it as a compact alternative to larger models.
- **Arcee Unleashes AFM-4.5B Powerhouse**: Arcee.ai released **AFM-4.5B** with grouped query attention and ReLU² activations for high flexibility, available on [Hugging Face](https://xcancel.com/LucasAtkins7/status/1950278100874645621). Future variants target reasoning and tool use, backed by a DatologyAI data partnership.

**Theme 2: Hardware Hustles for AI Speed**

- **Quantization Zaps Bandwidth Bottlenecks**: Quantization slashes memory bandwidth and boosts compute speed beyond just model fitting, though keeping vision ops in **FP32** creates bottlenecks in conv layers. Users debate optimizing for consumer hardware, with dynamic 4-bit methods highlighted in [Unsloth's blog](https://unsloth.ai/blog/dynamic-4bit).
- **AMD's Strix Halo APUs Price Out Competitors**: **Strix Halo APUs** hit $1.6k for 64GB and $2k for 128GB, but EPYC systems win on value with upgradeable RAM, as discussed in a [Corsair AI Workstation post](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/). Soldered memory draws ire for scam-like limitations versus DIMM flexibility.
- **P104-100 GPU Bargain Sparks Scam Fears**: The **P104-100** GPU sells for 15 GBP on [Taobao](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616), touted as a 1080 equivalent for LLM inference despite PCIe 1.0 x4 constraints. Sharding across cards aids price-performance, but users warn of potential 4GB VRAM access issues.

**Theme 3: Censorship Clashes in Model Mayhem**

- **Qwen3 Shuts Down Sensitive Queries**: Asking **Qwen3-30B** about China's internet censorship triggered immediate chatbot shutdowns, amid excitement for its release on [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF). This highlights overzealous safety features limiting practical use.
- **OpenAI's Censorship Lectures Irk Users**: Heavy censorship in **OpenAI models** yields canned responses and moral lectures, with [Unsloth's Llama 4 guide](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune) advising against authoritative phrasing. Community frustration grows over reduced utility for coding and non-client tasks.
- **GLM 4.5 Air Mimics Gemini's Guardrails**: **GLM 4.5 Air** feels like **Gemini** with broken tool use in vllm, but excels in chatting and analysis per a [Z.ai blog](https://z.ai/blog/glm-4.5). Debates focus on balancing safety without crippling functionality.

**Theme 4: Agents Arm Up with Security Shields**

- **DeepSecure Locks Down AI Agents**: DeepTrail's open-source **DeepSecure** enables auth, delegation, and policy enforcement for agents via split-key architecture and macaroons, with Langchain examples like [secure workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py). It's designed for cross-model proxying, detailed in [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
- **MCP Servers Battle Context Leaks**: Single-instance **MCP servers** need user-context separation to prevent session data sharing, as users report EC2 deployment issues with Claude Desktop despite SSL setup, per [MCP docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers). Cursor connects fine, but state tools fail on Windows.
- **Cursor Agents Hijack Ports in Parallel Panic**: **Cursor Background Agents** unexpectedly hijack ports, disrupting dev environments; fixes include disabling auto-forward in VSCode or emptying ports array. Parallel task coordination uses API orchestration or Docker, with a [task queue script](https://github.com/example) proposed for dependencies.

**Theme 5: Education Explodes with Study Groups**

- **Diffusion Models Study Group Ignites**: A 12-person, 5-month group studies diffusion models via [MIT curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf), with free intros on Aug 2 (Flow Matching) and Aug 9 (PDEs/ODEs) at [Luma](https://lu.ma/kv8zf6va). It features peer-led sessions and projects for AI pros.
- **LLM Safety Research Resources Rally**: PhD students seek LLM safety resources, recommending [Alignment Forum posts](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter) on reasoning steps and beliefs in chain-of-thought. Focus includes bias mitigation and ethical domain adaptation.
- **Video Arena Launches Bot Battles**: LMArena's experimental **Video Arena** lets users generate and vote on AI videos via bot, with a staff AMA by [Thijs Simonian](https://www.linkedin.com/in/thijsdev/) via [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog). It supports free comparisons of top models for images and videos.




---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **R1-der R.I.P: Model Removed From LLM Selector**: The **R1 1776 model** has been removed from the LLM selector but is still accessible via the [OpenRouter API](https://openrouter.ai/perplexity/r1-1776).
   - Users are contemplating a switch to **O3** or **Sonnet 4** for reasoning tasks after the removal.
- **Android Comet App to Launch Soon**: The **Comet for Android** app is under development and is scheduled for release towards the end of the year.
   - While one user questioned the browser's potential capabilities, others lauded its performance on Windows.
- **Gemini 2.5 Pro Gets a Speed Boost**: Users reported a significant speed increase in **Gemini 2.5 Pro**, speculating it might be using **GPT-4.1**.
   - This improved performance may come with limitations like daily message caps for reasoning models.
- **Spaces Craze: Custom Instructions Heat Up**: Members discussed optimizing the use of the **Spaces** feature by adding custom instructions.
   - One user clarified that the **instruction field offers more options**, such as adding specific websites for data retrieval.
- **Deep Research API has Structured Outputs**: A member building a product stated that they're familiar with the **deep research and structured outputs API**.
   - They also asked to chat with somebody about **Enterprise API pricing**, early access, rate limits, and support, and requested some credits to test and integrate the API appropriately.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3-30B Shuts Down Chatbot**: **Qwen released Qwen3-30B** and one user reported that when they asked **Qwen3** why China censors the internet on their chatbot, the system immediately shut down their requests.
   - The release of **Qwen3-30B** excited the community, with a link to [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) provided for further exploration.
- **GLM 4.5 Air Mimics Gemini**: Users discussed **GLM 4.5 Air**, with one mentioning it feels like **Gemini** and highlighted a [blog post comparing it to other models](https://z.ai/blog/glm-4.5).
   - Members noted that the **tool use is broken** in vllm but it worked well for chatting, poetry analysis, and document search.
- **Quantization Speeds Up Compute**: Quantization isn’t just about fitting models into memory, it also **reduces memory bandwidth** and can **significantly improve compute speed**.
   - A member noted that keeping the vision head ops like conv layers in **FP32** seems suboptimal, as they tend to be quite slow and is a bottleneck.
- **Gemma 3's watermarks Removed After 16k Fine-Tuning**: After fine-tuning **Gemma 3 4B** with **16k** context, experiments found that **watermarks** were completely removed and models were more stable, according to an attached [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&).
   - A new **Gemma 3 competition** was announced with the notebooks being available and the competition ending in **7 days** with more info available on [Xitter](https://x.com/UnslothAI/status/1950553466591723640).
- **Unsloth Cracks Down on OpenAI's Censorship**: Members expressed disappointment with **OpenAI's heavy censorship**, sharing experiences of canned answers and lectures from ChatGPT.
   - One user pointed out [Unsloth's Llama 4 page](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune) that directs users to never use phrases that imply moral superiority or a sense of authority, and to generally avoid lecturing.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor's MCP Browser Automations Coming Soon**: Members are actively developing browser-reliant automations via **Cursor's MCP**, with early access slated for release in the coming weeks.
   - A member highlighted the ease of setup, noting it features a *one-click MCP setup* to facilitate building browser automations directly.
- **Parallel Agent Coordination Conundrums**: Members are grappling with managing parallel tasks with dependencies, given agents' lack of shared workspaces complicates simultaneous triggering.
   - Proposed solutions involve **external orchestration via API**, **file-based coordination**, and **Docker-based parallel execution**, including a sample [task queue script](https://github.com/example).
- **Cursor Background Agents Hijack Ports**: Engineers reported unexpected port hijacking by **Cursor Background Agents**, leading to debugging efforts to restore their development environments.
   - Mitigation suggestions include setting the `ports` property to an empty array or disabling *auto forward ports* in VSCode settings.
- **Background Agents Considered for Research**: A member explored utilizing background agents for research-oriented tasks, such as **refactor research** or **feature implementation research**.
   - They inquired about optimal workflows, considering options like having the agent draft markdown in the PR or directly implementing changes.
- **Fishy Terminal Defaults**: A member encountered an issue with **Cursor's integrated terminal** defaulting to the **fish** shell and sought solutions to change it.
   - Attempts to modify the shell via settings and wrappers eventually succeeded after temporarily renaming the fish binary, but the root cause remains unknown.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Dot-lol Data Collection Under Fire**: Concerns arose about [dot.lol](https://dot.lol)'s potential to **sell user data** and profiling information, cautioning users against assuming their data won't be used for targeted influence or profit.
   - While some worry about the impact of extensive data collection, others argue **data collection is inevitable** and users should focus on not linking data to their personal identity.
- **GPT-5 Buzz: August Release?**: Rumors hint at a possible **GPT-5** release in early **August**, with potential evidence of router architecture preparations found in the ChatGPT Mac app.
   - Community members are speculating on its impact and whether it will surpass other models, with some hoping for a free tier.
- **GDPR: Teeth or Toothless?**: Members debated the effectiveness of the EU's **GDPR** in preventing data collection by AI companies, with differing opinions on its impact.
   - Some believe that **GDPR** primarily affects the *use* of data rather than the *collection*, while others countered that *data collection is turned off for EU consumers*.
- **Zenith's Arena Encore**: Enthusiasm brewed for the return of the **Zenith** model to the **LMArena**, with anticipation for its potential **ELO** rating and overall performance.
   - While some lamented missing the chance to try it out, others held strong opinions on its value to the platform.
- **Lights, Camera, AI: Video Arena Launches**: The **LMArena** team launched an experimental **Video Arena** on Discord where users can generate and compare videos from top AI models for free, using the LMArena bot to **generate videos, images, and image-to-videos**.
   - A **Staff AMA** with the bot's developer, [Thijs Simonian](https://www.linkedin.com/in/thijsdev/), was announced, inviting users to submit questions via [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) for the AMA.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces go Boom**: Members discussed that **HF Spaces** might unexpectedly restart, suggesting pinning dependency versions to avoid issues, as described in [the docs](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts).
   - One user reported that both *Ilaria RVC* and *UVR5 UI* stopped working and recommended factory rebuilds, while others continued to work fine.
- **P104-100 GPU on Sale for $15!**: Users debated the merits of the **P104-100** GPU for AI tasks, with one claiming it's *practically a 1080* for **15 GBP** (though others called it a scam), available from [Taobao](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616).
   - Some cited limitations like **PCIE 1.0 x4**, while others emphasized its price-to-performance for LLM inference, even when sharding models across multiple cards.
- **Qwen 30B Challenges GPT-4o**: Users highlighted the release of the **Qwen 30B** model, claiming it rivals **GPT-4o** and can run locally in full precision with just 33GB RAM using the [Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF).
   - Users noted that a quantized version can run with 17GB RAM.
- **Diffusion Model MIT Curriculum Study Group**: A new **study group** will focus on learning **diffusion models** from scratch using **MIT’s curriculum** with early sign-up for **$50/month**, and two free intro sessions are available for non-members: **Aug 2** and **Aug 9**, details on [Luma](https://lu.ma/kv8zf6va).
   - The study group will be based on [MIT’s lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) and [previous recordings](https://aischolars.notion.site/).
- **MoviePy Makes Video Editing Server**: A member built a **MCP server** using **MoviePy** to handle basic video/audio editing tasks, integrating with clients like **Claude Desktop** and **Cursor AI**, available on [GitHub](https://github.com/Aditya2755/video-edit-mcp).
   - The author is seeking collaboration on features like object detection-based editing and TTS/SST-driven cuts.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-5 Hype Restsless Fans**: Users are actively debating the release of **GPT-5**, with some claiming access via **Microsoft Copilot/Azure**, but skeptics await an announcement from **OpenAI**.
   - One user humorously criticized **Sam Altman** for fueling hype and making fans *restless*.
- **Study and Learn: Distraction or Innovation?**: **OpenAI** launched a new **Study and Learn** feature, viewed by some as a simple system prompt and perhaps a distraction from **GPT-5** expectations.
   - One user even dumped the [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) into the O3 model for analysis.
- **Copilot and ChatGPT Face Off**: Discussions clarify that **Microsoft Copilot** uses either **GPT-4o** or **O4-mini-high**, with potential future integration of **GPT-5** based on source code hints.
   - Copilot's unlimited daily message cap prompts questions about why users still prefer **ChatGPT**, though some users still believe Google's [Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) is the best image generator.
- **Chat History Vanishes into Thin Air**: Multiple users reported disappearing **ChatGPT chat histories**, despite attempts to troubleshoot by logging in and out and clearing cache.
   - OpenAI support suggests this *could be an isolated bug*, emphasizing that *chat history recovery isn’t possible once it’s lost*.
- **Engineering new AI Memory Format**: A member introduced a new memory format proposal with [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709), aimed at optimized AI VM, systems interfacing with vector databases, or for protected symbolic transfers, emphasizing speed and efficiency over human readability
   - Another member provided a detailed line-by-line reading of the format, highlighting its core principles like **token embedding**, **semantic grouping**, and **binary encoding**.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSecure Unveils Open-Source AI Agent Auth**: **DeepTrail** introduced **DeepSecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure)), an open-source auth and delegation layer for AI agents, enabling authorization, agent-to-agent delegation, policy enforcement, and secure proxying across models, platforms, and frameworks.
   - The technology features a split-key architecture, gateway/proxy, separate control/data plane, policy engine, and macaroons, exemplified in integrations for Langchain/LangGraph like [secure multi-agent workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py) with fine-grained access controls and [delegation workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py).
- **OR Gives Free Messages for $10 Top-Up**: A one-time **$10** top-up on OpenRouter unlocks **1000 daily free messages**, even after the initial credits are depleted.
   - Users confirmed that after spending the initial $10, the **1000 requests/day** limit remains active.
- **API Blocks Unwanted Quantization**: OpenRouter's API now lets users specify acceptable quantization levels to avoid lower-precision models like **FP4** using the [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels).
   - The API allows excluding specific quantization levels, such as allowing everything *except* FP4 models.
- **Deepinfra's Secret Gemini Pro Deal**: **DeepInfra** negotiated a lower rate with **Google** for **Gemini 2.5 Pro** and passed the savings to customers, indicated by a 'partner' tag on DeepInfra's listing.
   - Unlike the **Kimi K2** model, DeepInfra's **Gemini 2.5 Pro** has a partner tag, signaling a direct partnership with Google.
- **Ori Bot Underperforms**: Users report that the **OpenRouter Ori bot** may be a *net negative* due to inaccurate responses, particularly in payment processing issues.
   - One user suggested that **Ori** often puts the fault on the user and asks questions that *lead nowhere*, and a developer is now working to update **Ori's** knowledge.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Metislist Rank Provokes François Frenzy**: A user shared [Metislist](https://www.metislist.com/), sparking debate over **François Chollet's** rank at #80, with many feeling the creator of **Keras** deserved a higher placement.
   - Some felt that Chollet should be in the top 50, with one user quipping *Damn you got beef with my boy François?*.
- **Arcee Aces AFM-4.5B Model Drop**: Lucas Atkins announced the release of **AFM-4.5B** and **AFM-4.5B-Base** on Hugging Face from [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621), touting the flexibility, high performance, and quality due to a data partnership with DatologyAI.
   - The models incorporate architectural improvements such as **grouped query attention** and **ReLU² activations**, with future releases planned for reasoning and tool use.
- **NotebookLM Now Summarizes Videos**: **NotebookLM** introduced a new feature for video overviews of articles and blog posts ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234)), enabling users to grasp content without reading full texts.
   - Users lauded the innovation and suggested further development for interactive modes.
- **GPT-5 Spotted on MacOS**: References to **gpt-5-auto** and **gpt-5-reasoning** were discovered in MacOS app cache files ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg)), hinting at the imminent arrival of **GPT-5**.
   - Further corroboration mentioned **gpt-5-reasoning-alpha** in a biology benchmarks repository, leading to speculation about a potential announcement or release.
- **Anthropic Aiming High for $170B Valuation**: Anthropic is reportedly seeking to raise **$5 billion**, potentially valuing the AI startup at **$170 billion**, with a projected revenue of **$9 billion** by the end of the year ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810)).
   - The news was met with comparisons to OpenAI and xAI.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Tableau Server Flexes LLM Orchestration Muscle**: A member reported successfully integrating the latest **Tableau Server edition** to incorporate an **LLM (server/on prem)** for **Vizql NLP**.
   - This setup aims to enable more sophisticated natural language processing capabilities within Tableau visualizations.
- **Gemini Agentic Framework Prototype Emerges**: A member shared a [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) prototype, describing it as a **one-shot prototype**.
   - The prototype leverages **AI Studio** for building agentic apps, emphasizing clear intention-setting for the builder agent to facilitate phased testing and focused model development.
- **NotebookLM Bypasses Bot Restrictions for Podcast Dreams**: In response to inquiries about podcast creation limitations due to bot restrictions on **NotebookLM**, a member clarified that the tools are accessible via the **API**.
   - They suggested rebuilding the workflow and manually loading reports into **NotebookLM**, as an alternative solution.
- **Obsidian and NotebookLM cozy up together**: An article detailing the integration of **NotebookLM**, **Obsidian**, and **Google Drive** was shared [here](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/).
   - A member volunteered to offer more detailed guidance on **Obsidian** usage, tailored to individual user needs.
- **NotebookLM Audio output varies between 8-15 minutes**: Users reported audio file generation with **NotebookLM** averaging *8-10 minutes*, though some have achieved up to *15 minutes*.
   - This discussion underscores the variability in output length, potentially influenced by content complexity and processing efficiency.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Tsunami Alert Issued After Russian Earthquake**: An **8.7 magnitude earthquake** off the coast of Russia triggered tsunami warnings for Hawaii and watches for the west coast of the United States.
   - Residents in affected areas were advised to monitor updates closely due to potential tsunami arrival hours later.
- **LM Studio Users Request Enhanced Conversation Handling**: Users are requesting a feature in LM Studio to copy and paste entire conversations, which are stored in **JSON format**.
   - One user pointed others to the [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) noting that many would find this feature useful.
- **LM Studio Model Relives February 18th**: A user reported that their LM Studio model was repeatedly referencing **February 18, 2024**, even when asked about current events.
   - Another user suggested checking the **system prompt** or **Jinja template** for the date.
- **Strix Halo APUs Pricey, Spark EPYC Debate**: The price of **Strix Halo APUs** is around **$1.6k** for 64GB and **$2k** for 128GB, but some members suggest that **EPYC** systems offer better value.
   - One member lamented *soldered memory* on such devices, drawing a comparison to a recent DIMM failure on a server and pointing to a [Corsair AI Workstation 300 with Strix Halo APU](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/).
- **9070 XT Performance Disappoints**: The **9070 XT** is significantly slower than a **4070 Ti Super**, with one user reporting a model running at **7 t/s** on their **4070 Ti Super** only achieved **3 t/s** on the **9070 XT**.
   - Another member suggested that RAM bandwidth limitations might be the cause.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaCloud Struggles to Read PDFs**: A member reported that **LlamaCloud** could not detect a **PDF file** and process it via API, using **n8n** for workflow simplification and linked to a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&).
   - This issue occurred when trying to process the **PDF file** via API.
- **Character AI Sparking Building Discussions**: Members discussed building a **character AI** with deep understanding of a large story, using a classic **RAG** pipeline with chunked text, embeddings, and a vector database.
   - This includes leveraging a **RAG** pipeline to create an AI with deep understanding.
- **Neo4j's Knowledge Graph Runs into Snags**: A member reported their simple graph storage implementing **Neo4j** is taking *ridiculously long to load* and their server is not compatible with **Neo4j 5.x**, and **LlamaIndex** doesn't seem to like **4.x**.
   - **Aura** is also blocked by the server proxy, presenting further roadblocks to implementation.
- **Flowmaker Gemini 2.5 Pro Bug meets Rapid Fix**: A member reported an error when using **Flowmaker** with **Gemini API** due to an invalid model name, requiring a number like *gemini-2.5-pro*.
   - A fix was [committed](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19) and deployed swiftly, resolving the issue.
- **Community Offers RAG Debugging Assistance**: A member offered help with a **MIT-licensed repo** designed to debug tricky **RAG issues**, including sparse retrieval, semantic drift, chunking collapse, and memory breakdowns.
   - Following the initial offer, a community member inquired about specific complex issues addressed by the repo, focusing on concrete examples of *sparse retrieval* and *semantic drift*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Expert Parallelism Exhibits Excellence?**: A member sought examples of **Expert Parallelism (EP)** outperforming **Tensor Parallelism (TP)**, but found that with **Qwen32B** and **Qwen 235B**, the all-reduce communication overhead made **EP** less performant.
   - They observed **EP** being beneficial only for models using **MLA** and needing **DP attention**.
- **Triton Treasures in Torch Compile**: To extract **PTX code**, members suggested using `TORCH_LOGS="output_code" python your_code.py` or accessing the `compiled_kernel.asm.keys()` dictionary, as detailed in [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir).
   - The `keys()` dictionary contains keys for intermediate representations including **llir, ttgir, ttir, ptx, and cubin**.
- **Inductor's Intriguing Influence on Triton**: To force **Triton code generation** for matmuls, members suggested configuring settings in [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461), by modifying settings such as **use_aten_gemm_kernels**, **autotune_fallback_to_aten**, **max_autotune_conv_backends**, and **max_autotune_gemm_backends**.
   - However, it was noted that built-in kernels are often faster and not every op is converted to Triton by default.
- **CuTeDSL Compiler Cuts Prefetching Code**: A member shared a [blogpost](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) and [code](https://github.com/simveit/software_pipelining_cute_dsl) on using **CuTeDSL** for **GEMM on H100**, explaining how to let the compiler handle prefetching.
   - The blogpost details an experimental argument to the `cutlass.range` operator to hint for prefetching, matching the performance of manual prefetching with simpler code.
- **Gmem Guardian: Synchthreads Saves the Day**: After copying from global memory (**gmem**) to shared memory, manually inserting a `synchthreads` (or equivalent) is necessary to sync all threads before moving on.
   - This guarantees that all shared memory elements are available for collective calculations such as **gemm**, reduction, or scan.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **M3 Ultra Crowned Local Inference King**: The **M3 Ultra** is gaining traction as a top choice for local inference, due to its **80-core GPU** and **512GB memory**, according to [this Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/).
   - One member instead bought a used **M1 16g** because nobody responded to their post.
- **Solving Model Latency Offshore**: A member is seeking solutions for running **LLMs offshore** with low latency, despite poor internet conditions.
   - Other members simply stated that they're fine with spending money on whatever they want.
- **Microsoft's Audio Instruction Research**: Members expressed interest in researching open-source solutions for improving **audio instruction-following** in **Speech-LLM models** to create better speech UIs, pointing to **Microsoft's Alignformer**.
   - Alignformer is not open source, so collaboration may be necessary.
- **ICL Demolishes Diagnostics?**: Members speculate that **in-context learning (ICL)** could break **interpretability tools** like **Sparse Autoencoders (SAEs)**, as **ICL** pushes activations out of their original training distribution, referencing the **Lucas Critique** and [this paper](https://arxiv.org/abs/2501.00070v1).
   - This issue isn't exclusive to **ICL**, but arises whenever **SAEs** encounter activation distributions different from those they were trained on, according to members.
- **Grouped GEMM Gaining Ground**: A member highlighted a PR supporting **torch._grouped_mm** in **GPT-NeoX**, now in PyTorch core, suggesting performance gains for **MoE** and linking to [this MoE implementation](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221).
   - They noted that interested users can use a one liner from TorchAO for **low precision MoE training**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Paper on CUDA Generalization Surfaces**: A member shared a [paper](https://huggingface.co/papers/2507.14111) about generalizing beyond **CUDA**, prompting a reminder to post such content in the appropriate channel.
   - No further discussion or alternative perspectives were provided.
- **Mojo Newbie Braves TLS Handshake Gremlins**: A new Mojo user reported a **TLS handshake EOF error** when attempting to run a Mojo project with **pixi** and **magic shell**, using a **Dockerfile** from Microsoft Copilot.
   - A suggested fix involved using the latest nightly `mojo` package with **pixi** and specific commands (`pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge` and `pixi add mojo`), but this fix failed even with a **VPN**.
- **Mojo external_call Anatomy Examined**: Users questioned why Mojo's `external_call` uses specific functions like `KGEN_CompilerRT_IO_FileOpen` instead of standard `fopen` from **libc**, with a concern if this choice is about safety.
   - A member clarified that these are artifacts from older **Mojo** versions and are not a high priority to fix, and that the **KGEN** namespace belongs to Modular and will be opened up eventually.
- **Python call from Mojo suffers Overhead**: A user discovered significant overhead when calling a Python no-op function from Mojo (4.5 seconds for 10 million calls) compared to direct Python execution (0.5 seconds).
   - Members explain that Mojo needs to start a **CPython** process, and CPython is embedded via `dlopen libpython`, so you shouldn't call it in a **hot loop**.
- **Spraying Glue In a Race Car Engine**: Discussion covered the performance impact of calling Python from Mojo, especially in hot loops for tasks like **OpenCV** or **Mujoco** robot simulation.
   - Members note that a lot of fast python libs are actually C libs with a wrapper, and that *interacting with the context dicts alone can easily eat several hundred cycles*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Deepseek-chat Suffers on OpenRouter**: Members observed that **Deepseek-chat** performed worse on **OpenRouter** compared to the official **DeepSeek API**, specifically when used as the editor model in architect mode.
   - A recommended fix involves using `aider --model openrouter/deepseek/deepseek-r1` to ensure the default config with `edit-format: diff` from [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) is applied.
- **Aider as Coding Model Training Catalyst?**: There was a suggestion that **Aider** could aid in coding model training by logging linting and undo actions during development workflows.
   - This approach would use **Aider** in *careful* development to generate valuable training data, although the commenter did not explicitly request developers to implement this feature.
- **Qwen3 Coder 30B-A3B Bursts Onto Scene**: An announcement was made of the new **Qwen3 Coder 30B-A3B**, shared via an image to validate its legitimacy.
   - Details are still emerging for this new model.
- **Litellm API Flounders With Connection Errors**: A user reported encountering numerous `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` errors.
   - Despite the errors, functionality remained unaffected for the user.
- **Open Model R1 v Qwen Coder Duel Requested**: A member requested advice on the best open model for **aider**, considering unlimited hardware, and expressed interest in testing **R1** and **Qwen Coder** models.
   - The member mentioned having **Runpod credits** to use, indicating plans for practical testing with these models.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **LLM Safety Research Resources Sought**: A PhD student requested resources on current **LLM safety/alignment research**, with suggestions including blogs from the **AI alignment forum**.
   - Specifically, the blog posts [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter) and [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1) were mentioned as helpful starting points.
- **Coding CUDA with Claude Confounds Coders**: A member discovered that writing in **CUDA** with **Claude** is difficult, requiring extensive preparation, comprehension, and arrangement.
   - They posited that the ultimate evaluation would be whether a Python programmer with some GPU and CUDA familiarity could utilize **Claude** to manage writing **kernels** and enhance performance, including [an image](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4).
- **Z.AI's 54 Repos Spark Interest**: A member inquired about the new **Z.AI 54 open source repos** and whether anyone has explored them, prompting curiosity within the community.
   - However, specific details regarding the contents or functionalities of these repositories were not elaborated upon.
- **Qwen3 allegedly equals GPT-4o**: A member shared a post noting that [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) rivals **OpenAI's GPT-4o** in both English and Chinese.
   - The community is enthusiastic about **Qwen3** as a potential strong contender in the language model space with benchmarks indicating performance may be on par with **GPT-4o**, particularly in multilingual applications.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Chatbot speaks Mandarin**: A user reported that the **Kimi chatbot** replied in **Chinese** despite receiving **English** prompts, possibly due to being logged out.
   - Screenshots revealed that while the reply was in English, the suggested sources and questions were in Chinese.
- **Kimi Learns from Social Media**: One member joked that **Kimi's training dataset** includes **Instagram** and **TikTok** comments, and linked to [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) to support this claim.
   - They suggest that this focus on social media data is what makes **Kimi** so good.
- **Moonshot AI's Got the Best Vibe**: A member stated that *moonshot got the best vibe no cap*, and linked to [X post](https://x.com/crystalsssup/status/1944287779896328668) about the AI community vibe check.
   - Another agreed that the community needs some competition.
- **Scale AI Provides Data**: A member stated that **Alexandr Wang** is the **founder and CEO of Scale AI**, which provides training data, annotation, and evaluation services.
   - They pointed out that **Scale AI** is essential for developing machine learning models.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Lume Edges Out Suna in Head-to-Head Showdown**: Members debated the merits of the **Lume** and **Suna** agentic systems; one member found that **Lume** coded out specific things with fewer mistakes.
   - The member noted they couldn't compare to **Manus** due to cost, and conceded they may not have prompted **Suna** correctly.
- **Manus' Comic Creation: A Diamond in the Rough?**: One member suggested that **Manus**' comic creation feature is nice but still can be improved, especially for free users.
   - Another member stated the service was declining in quality, with restrictive limits for free users, and questioned *whether Manus is dead*.
- **AI's Rosy View of Manus vs. Skeptical Human**: One member asked an AI what it thinks about the future of **Manus**, and it replied *I think the future of Manus is bright*.
   - Another member expressed skepticism, citing the release of agent modes from **OAI** and **Google** as competitive pressures.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP Server Faces Context Crisis**: A user questioned the need for additional **user-context separation** layers in a single cloud-deployed **MCP server instance** to prevent data sharing between unique sessions, referencing [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers).
   - Another user reported challenges connecting to their **MCP server** via **Claude Desktop**, despite successful deployment to **EC2** with proper **SSL** configuration, but could still connect via **Cursor**.
- **Cucumber and LLMs Cooking up BDD**: A user shared a production-ready **Behavior-Driven Development (BDD)** side project, which includes a [diagram of the solution](https://cdn.discordapp.com/attachments/1312302100125843476/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&).
   - Another user reported the **state tool** not working with **Windows-MCP** by **CursorTouch** in **Claude desktop**.
- **DeepTrail Unveils Deepsecure for AI Agent Auth**: **DeepTrail**, supported by Berkeley SkyDeck, is developing **Deepsecure**, an open-source auth and delegation layer for AI agents, documented on [GitHub](https://github.com/DeepTrail/deepsecure).
   - **Deepsecure's** architecture features a split-key design, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in its [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
- **FastMCP Asked About Tooling Dynamics**: A user asked about **FastMCP**'s dynamic tool selection capabilities when multiple tools are defined on the server.
   - Specifically, they want to know if **FastMCP** has logic to automatically select tools (e.g., math, web search, RAG, data interpreter) on the client side.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Gets Parameterized: Learnable Parameters Proposal Sparks Excitement**: A proposal to add learnable parameters (`dspy.variable` or `dspy.parameter`) to DSPy sparked enthusiasm, leading to [an issue on GitHub](https://github.com/stanfordnlp/dspy/issues/8593) to collect ideas and use cases.
   - The goal is to enable optimizers to generate optimal prompts by allowing *templates to be parameters/variables* and optimizing template variable placement.
- **F-Strings Flounder: Signature Implementation Stymied**: A member ran into issues implementing a signature using an f-string to verify code against a description and asked for help.
   - Another member advised against this approach, recommending that the parameter description be placed within `dspy.InputField()`.
- **DSPy Enters Genetic Prompting Arena**: A member highlighted a YouTube video comparing **DSPy** to **GEPA** and linked the [YouTube video](https://www.youtube.com/watch?v=o6RbVPFOslg) where it's said *DSPy optimizes the prompt you gave it; GEPA evolves a prompt you never imagined*.
   - The same member suggested evolving **MIPRO** into a reflective, genetic-style frontier engine for DSPy to generate prompts, challenging the YouTuber's claim.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **AMD Blazes Ahead in Gaming and AI**: A member suggested pairing the **7900XT** and **7800X3D** for gaming, noting AMD's consumer AI usability and long-term community benefits.
   - They linked to [a tweet](https://x.com/Teknium1/status/1950596567968477382) arguing in favor of AMD over Nvidia's **9070** and **9900X**.
- **Qwen Launches Coding Model with Thinking**: A member announced the release of the **Qwen3-30B-A3B-Thinking-2507** coding model on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).
   - The linked Hugging Face model offers a new tool for code generation.
- **RLVR: Algorithm or Marketing?**: A member questioned the classification of **RLVR** (Reinforcement Learning, Virtual Reality) as a reinforcement learning algorithm, inciting discussion.
   - Another member, teknium, stated *"RLVR is just not a RL Algo its just a target of rl"* in response to [an NVIDIA tweet](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670).



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Researcher Requests AI Safety Summer School Channel**: A new member requested a dedicated channel for an AI Safety **summer school** that happened weeks ago.
   - This reflects ongoing interest in academic and learning-focused discussions within the community.
- **Bias Buffs Band Together to Battle Bias**: A PhD student at JKU Linz is focusing on **mitigating social bias in LLMs**, with interests in **attribution for generative models**, **AI generated text detection**, and **domain adaptation**.
   - The student is keen to connect with others working on practical ethical concerns in domain-specific LLMs, seeking collaboration.
- **Kernel Koder Kodes Kernels in CUDA**: A member named Ali is deeply involved in **optimizing GPU kernels in Triton/CUDA** specifically for **autoregressive models**.
   - Ali is open to discussing low-level GPU programming, offering expertise in accelerating model performance.
- **Citation Configuration Conundrums Confront Cohere**: A member reported difficulty changing the **citation mode** using `citation_options` on `langchain_cohere.ChatCohere` and asked about implicit methods to pass citation options.
   - The member also inquired about the status of the [langchain-cohere repo](https://github.com/langchain-ai/langchain-cohere), noting its lack of recent updates and asking *if pull requests are welcomed*.
- **Senior Software Sage Seeks Sanctuary in South**: A remote **Senior Software Engineer** role is advertised, paying **$2K/month**, for a long-term contract, located in **Africa** or the **Americas**.
   - The role requires experience in **Ruby on Rails**, **Node.js**, **C#/.NET**, **Python**, **Java**, and strong English communication.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune User Requests LoRA-Style Adapters**: A Torchtune user expressed interest in **LoRA-style adapter support** that freezes original model weights and applies updates through additional trainable layers.
   - The user specified a desire for adapters that maintain the same forward compute path without increasing computational cost.
- **Torchtune Merges Weights After Training**: A user pointed out that Torchtune merges weights back in after training with an adapter, referencing the [Torchtune end-to-end workflow documentation](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html).
   - The user's comment sparked questions about the implications of merging weights in Torchtune.
- **ACL Paper Receives Accolades**: A member shared their **ACL paper** that won an award, with a link to the paper [here](https://aclanthology.org/2025.acl-long.266/).
   - No further discussion followed this announcement.
- **Glianorex finetunes trigger discussion**: A user asked if the **Glianorex finetunes** are public.
   - This comment could be interpreted as a complaint: *Glianorex is killing me and my doctor has been no help*.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Certificate Declaration Still Needs Completion**: A staff member has reminded another member to complete the **certificate declaration form** for the **LLM Agents (Berkeley MOOC)**.
   - The staff reiterated that the **form was never received**, despite a previous notification to the member.
- **Second Reminder for Certificate Submission**: The staff emphasized the importance of submitting the **Certificate Declaration Form** to finalize the **MOOC** requirements.
   - Failure to submit the form prevents the issuance of the certificate, impacting course completion verification.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI Scholars Launches Diffusion Models Study Group**: A new study group is kicking off a **12-person**, **5-month** program (**2-4 hrs/week**) by **AI Scholars** using [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) to study diffusion models, a key architecture in generative AI.
   - Confirmed members include the CTO of an AI film tool, AI art instructor, 2 LLM instructors, and 2 full-time AI researchers.
- **First Two Diffusion Models Sessions are Free**: The first two introductory sessions are free and open to non-members: August 2nd on *Flow Matching & Diffusion Models* and August 9th on *PDEs, ODEs, SDEs + A Brief History of Diffusion Models* ([links here](https://lu.ma/kv8zf6va)).
   - The program features peer-led sessions, mentor Q&A, hands-on projects, real research papers, and a tight, trusted cohort with a weekly format of 2 hours live class + 2 hours self-study, with students rotating teaching.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **User Seeks Cloud Deployment Strategy**: A user is seeking advice on deploying a language model, trained with a custom folder of PDFs, to the cloud for public use, and specifically wants a simple **GUI** for user queries.
   - Nomic suggested that the **enterprise plan** wasn't a good fit, and the user wondered about **Hugging Face deployment** as an alternative.
- **Enterprise Plan Inadequate for User Needs**: Nomic indicated that the **enterprise plan** isn't a good fit for the user's needs regarding deploying a custom language model.
   - The user is exploring alternative deployment strategies, such as **Hugging Face**, to make their language model accessible.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Plan to Eval Kimi 2?**: A member inquired about plans to evaluate **Kimi 2**.
   - They expressed curiosity about its **tool-use capabilities** after post-training.
- **Kimi 2 Tool-Use Post Training Interest**: Interest was expressed in evaluating **Kimi 2's tool-use performance** after its post-training phase.
   - The inquiry highlights the importance of assessing how well models adapt and utilize tools following their initial training.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399830622255710320)** (1152 messages🔥🔥🔥): 

> `R1 1776 Removal, Comet for Android, Gemini 2.5 Pro Speed, OpenRouter API for R1 1776, Daily Message Cap Limits` 


- ****R1-der R.I.P:** Model Removed From LLM Selector**: Members noticed that the **R1 1776 model** was removed from the LLM selector, but it is still accessible via the [OpenRouter API](https://openrouter.ai/perplexity/r1-1776).
   - Users speculate about switching to **O3** or **Sonnet 4** for reasoning tasks following the removal.
- ****Android Comet:** Mobile App to Launch Soon**: The **Comet for Android** app is currently in development and is expected to be released towards the end of the year.
   - A user expressed concerns about the browser's potential capabilities, while others praised its performance on Windows.
- ****Pro Gemini 2.5** Gets a Speed Boost**: Users have observed a significant increase in the speed of **Gemini 2.5 Pro**, suggesting it might be using GPT-4.1 instead of Gemini.
   - Members noted that this performance boost might come with limitations, such as daily message cap limits for reasoning models.
- ****Spaces Craze:** Custom Instructions Heat Up**: Members discussed how to best utilize the **Spaces** feature by adding custom instructions.
   - A user asked whether the space description field or the instruction field is better for setting the context for the space - with a member responding that the **instruction field offers more options** like adding certain websites from which to pull specific data.
- ****API API!:** Perplexity API Guidance**: Users shared tips on using the **Perplexity API**, with one noting that pro subscribers get a monthly allocation of **$5 USD** in credits.
   - A user who was facing a 401 error was advised to ensure their code and model selection (**sonar-pro**) are correct.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399842279937806439)** (4 messages): 

> `Enterprise API pricing, Deep Research API` 


- **Deep Research API has structured outputs**: A member building a product that needs a lot of deep research capability and raising money soon for launch stated that they're relaxed on dev questions because they are *aquainted with the **deep research and structured outputs api**.*
   - They also asked to chat with somebody about **Enterprise API pricing**, early access, rate limits, and support, and requested some credits just to test and integrate the API appropriately.
- **Team ready to answer questions**: A member confirmed that the team is taking a look and asked what questions another member might have.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399837823863095307)** (670 messages🔥🔥🔥): 

> `Hater Spams, GLM 4.5 Air, Qwen3-30B, OpenAI Censorship, Unsloth and Llama 3.1` 


- ****Discord Deals with Hater Spam****: A user reported a **hater spamming hate** and direct messaging everyone, leading to calls for banning certain words.
   - Members confirmed that it was *nothing important* and mentioned GPU-shaming as a common issue.
- ****Users Rave About GLM 4.5 Air****: Users discussed **GLM 4.5 Air**, with one mentioning it feels like **Gemini** and highlighted a [blog post comparing it to other models](https://z.ai/blog/glm-4.5).
   - Members noted that the **tool use is broken** in vllm but it worked well for chatting, poetry analysis, and document search.
- ****Qwen3-30B Release Excites Community****: **Qwen released Qwen3-30B** and [provided a link](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), leading to discussions about its performance.
   - One user reported that when they asked Qwen3 why China censors the internet on their chatbot, the system immediately shut down their requests.
- ****OpenAI's heavy Censorship Disappoints Users****: Members expressed disappointment with **OpenAI's heavy censorship**, sharing experiences of canned answers and lectures from ChatGPT.
   - One user pointed out [Unsloth's Llama 4 page](https://docs.unsloth.ai/basics/llama-4-how-to-run-and-fine-tune) that directs users to never use phrases that imply moral superiority or a sense of authority, and to generally avoid lecturing.
- ****Unsloth Enhances Llama 3.1****: Users inquired about the difference between **unsloth/Llama 3.1** and just **Llama 3.1**, with community members clarifying that Unsloth provides fixes, template adjustments, and tokenizer improvements.
   - Unsloth's team also makes it easier to fine-tune models on consumer hardware, offering faster fine-tuning speeds with less memory usage.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400100378560958464)** (6 messages): 

> `Unsloth introduction, Low end-to-end latency in TTS voice cloning` 


- **Indians and Students flock to Unsloth**: A member from India joined Unsloth after hearing about it on the **Hugging Face** official Discord, hoping to learn about finetuning and model deployment.
   - Another member mentioned he's planning to *get cracked at llms and join a dope company like Unsloth*.
- **Guidance Sought for Low-Latency TTS Voice Cloning**: A new member is seeking *concrete guidance* on achieving low end-to-end latency in **TTS voice cloning**.
   - The member requested advice on frameworks, model optimizations, or hardware strategies, and another member suggested our TTS fine-tuning notebooks.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399849804346949803)** (9 messages🔥): 

> `Gemma 3 4B Fine-Tuning, Custom Tokens, Watermark Removal, RoPE FTW, Language Translation` 


- **Gemma 3 4B Finishes 16k Training**: After fine-tuning **Gemma 3 4B** with **16k** context, experiments found that **watermarks** were completely removed and models were more stable.
   - The research results were posted with an attached [screenshot](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688bd0b9&is=688a7f39&hm=82cad388163625496b8cb3e6dd62035b51ae1d16ce0015df29ea910e04c4471f&).
- **Custom Tokens create Chaos**: It was noted that unless training from scratch or with very large datasets, it's better to avoid **custom tokens** because models understand token splits.
   - For example, if a model sees *Yuki* split into *<*, *yuki*, *>*, it can better understand that Yuki = yuki, and that it is my token.
- **RoPE Gets a Novel Prize**: The poster praised the genius who invented **RoPE (Rotary Positional Embedding)**, as it works very well especially on smaller models.
   - However, to support huge contexts on inference, the poster thinks the field needs to invert some better optimizations, and quantization isn't enough.
- **Gemma speaks all Languages**: After testing translation abilities, the poster joked *OMG, it knows every (popular, at least) single languageOpenAI is doomed*
   - They also mentioned they keep getting this [song](https://www.youtube.com/watch?v=NgQoLPnuEDM) stuck in their head.
- **New Gemma 3 Notebooks**: A new **Gemma 3 competition** was announced with the notebooks being available and the competition ending in **7 days**.
   - More info available on [Xitter](https://x.com/UnslothAI/status/1950553466591723640).


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399831912234881115)** (64 messages🔥🔥): 

> `Phi-4 Generate Flags Error, GGUF Conversion and Quantization of Fine-Tuned Models, Llama-CLI Performance Issues, RuntimeError in Google Colab, Unsloth BNB 4-bit Conversion` 


- **Phi-4 needs `do_sample=True`**: When generating with **Unsloth's Phi-4** model, users encountered an error related to invalid generation flags (temperature, min_p) and discovered adding `do_sample=True` resolves the issue, although it is not documented in the [official notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb#scrollTo=kR3gIAX-SM2qHi).
- **Qwen2.5 GGUF conversion woes**: Users are facing issues when trying to merge, convert to **GGUF**, and quantize a fine-tuned model based on **Qwen2.5-14B-Instruct-unsloth-bnb-4bit**, encountering a `ValueError: Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'` error during export to **FP16 GGUF**.
- **Slow Llama-CLI with UD-Q2_K_XL model**: A user reported extremely slow performance (0.5 tokens/s) using **Llama-CLI** with the **Q2_K_XL** model, despite using a high-end system with a **5090**, **178GB RAM**, and **EPYC 9334** processor, and settings that should provide much better performance.
- **LLama3 fine-tuning faces RuntimeError**: A user reported a `RuntimeError: PassManager::run failed` error when fine-tuning **llama-3-8b-Instruct-bnb-4bit** on Google Colab, using a custom dataset formatted with **ShareGPT** templates and the Unsloth library.
- **Whisper input_ids error**: A user found that setting `task_type = None` in the `FastModel.get_peft_model` function resolves the `input_ids` error encountered when training the **Whisper** notebook, referring to [this issue](https://github.com/huggingface/peft/issues/1988#issuecomment-2751367819) for context.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400049478131777556)** (8 messages🔥): 

> `Quantization optimization, Dynamic 4bit quantization, Hi-Fi Gan replacement, Autoregressive models, Mels dislike` 


- **Quantization Speeds Up Compute**: Quantization isn’t just about fitting models into memory, it also **reduces memory bandwidth** and can **significantly improve compute speed**.
   - A member noted that keeping the vision head ops like conv layers in **FP32** seems suboptimal, as they tend to be quite slow and is a bottleneck.
- **Dynamic 4bit quantization blogpost**: A member shared the [Dynamic 4bit quantization blogpost](https://unsloth.ai/blog/dynamic-4bit), relating to quantization optimization.
   - This blogpost is directly related to **quantization isn’t just about fitting models into memory**.
- **Hi-Fi Gan faces Autoregressive competition**: A member asked about the possibility to replace **Hi-Fi Gan** with [this](https://arxiv.org/abs/1609.03499) in **VITS**.
   - Another member asked if it was for autoregressive reasons, since the first member dislikes Mels; however the first member later decided against it due to long training times.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399847900598636554)** (102 messages🔥🔥): 

> `GRPO trainer batch size, SFTrainer validation error, Model fine-tuning parameters, Llama 3.2 data preparation, Gemma 3 fine-tuning` 


- **GRPO Trainer's Batch Size Deconstructed**: In the **GRPO trainer**, *per_device_train_size* represents the number of batches, which is then multiplied by the number of generations to determine the effective batch size.
   - For instance, with *per_device* set to **1** and *num_generation* to **6**, the configuration yields **3** unique prompts each with **6** generations under a single GPU, potentially leading to CUDA out-of-memory issues when expanding to **15k** tokens, especially considering the impact of GPU memory utilization on activation weights.
- **Seeking SFTrainer Validation Error Savior**: A user encountered an *evaluation_strategy* unexpected keyword error while trying to save the validation error with **SFTrainer**.
- **Llama 3.2 data format**: A user requested an example of the data preparation format to fine-tune **Llama 3.2 8B**.
- **Gemma 3 Text-Only Tuning Tactics**: Unsloth offers a [text-only notebook](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/gemma-3-how-to-fine-tune-and-run-llms/gemma-3-how-to-run-and-fine-tune#unsloth-fine-tuning-fixes-for-gemma-3) for fine-tuning **Gemma 3**, providing fixes for Unsloth fine-tuning.
- **Unlocking Positional Parameters for Adapter Loading**: When using *model.load_adapter*, the *adapter_name* is a required positional argument.
   - A user encountered a *ValueError* related to unsupported target modules (**ModuleDict**) and sought guidance on fixing this issue, aiming to merge fine-tuned LoRA adapters to the base model (**unsloth/gemma-3-4b-it-unsloth-bnb-4bit**) for GGUF conversion using Llama.cpp.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399830240024465498)** (475 messages🔥🔥🔥): 

> `MCP Browser, Parallel Agent Tasks, VSCode Marketplace with Cursor, Automatic Scrolling, Sonnet Model` 


- **Cursor's MCP Browser setup**: Members are building browser-reliant automations directly via **Cursor's MCP** and early access will be coming out in the next few weeks ideally.
   - One member stated, *It has a one-click MCP setup so you can build browser-reliant automations directly via MCP*.
- **Parallel Agent Coordination Conundrums**: Members are discussing how to handle parallel tasks with dependencies, since agents don’t share workspaces, making it difficult to trigger them all at once.
   - Suggested solutions include an **external orchestration with API**, **file-based coordination**, or a **Docker-based parallel execution**, complete with a detailed example [task queue script](https://github.com/example).
- **VSCode Marketplace Integration Inquiry**: A member inquired about the possibility of using the **VSCode marketplace with Cursor**.
   - There was no clear answer in the discussion.
- **Automatic Scrolling Woes**: A new Cursor user asked if it is possible to disable **auto-scrolling** when using the **Agent chat window** to better read the thought process of Claude and the generated code.
   - There was no clear answer in the discussion, but the [changelog 1.3](https://cursor.com/changelog) was posted.
- **The Curious Case of the Terminating Terminal**: A member was having trouble with Cursor deciding which shell to start in its integrated terminal in the agent, and it defaults to **fish**.
   - Attempts to change the shell via settings and wrappers led to the temporary renaming of the fish binary and subsequent success, though the underlying cause remains a *fishy* mystery.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399860099563917344)** (8 messages🔥): 

> `Background Agent Commands, Docker Build Cache, Port Hijacking, Background Agents for Research` 


- **Running Commands at Background Agent's Run's End**: A member inquired about executing a command, specifically a formatter, at the **end of a background agent's run**.
   - The member noted that `terminals` can be run during setup, but that's only at the beginning.
- **Busting Docker Build Cache**: A member sought advice on **busting the build cache** when using a custom Dockerfile with edited layers.
   - Another member suggested using `docker builder prune -f` or `docker system prune -a` to remove unused containers, networks, and images.
- **Cursor Background Agents Hijack Ports**: Engineers wasted time debugging why their dev environment was suddenly broken only to figure out that **Cursor Background Agents hijacked the ports**.
   - A member asked if setting the `ports` property to an empty array stops Cursor Background Agent from forwarding any ports, and another user suggested disabling “auto forward ports” in the VSCode settings instead.
- **Background Agents for Research**: A member inquired about using background agents for research, such as **refactor research or feature implementation research**.
   - The member asked about workflows, suggesting having the agent write markdown in the PR or letting it make changes directly.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399828973818478592)** (413 messages🔥🔥🔥): 

> `dot.lol data, GPT-5 Release, EU GDPR impact, Zenith Model Relaunch, Video Arena Channels` 


- **Dot-lol faces data collection scrutiny**: Members discussed the potential for [dot.lol](https://dot.lol) to **sell user data** and profiling information, emphasizing the naivety of assuming data will never be used for targeted influence or profit.
   - Concerns were raised about data collection outweighing the usefulness of online services, while others argued that **data collection is inevitable** and users should prioritize not linking data to their identity.
- **GPT-5 release slated for August?**: Rumors suggest that **GPT-5** may be released in early **August**, with possible mentions of preparations in the ChatGPT Mac app confirming a router architecture.
   - Some community members speculate about its potential impact, debating whether it will surpass other models and even expressing hope for a free tier.
- **EU's GDPR Effectiveness Debated**: The effectiveness of the EU's **GDPR** in preventing data collection by AI companies was discussed, with opinions split on whether it sufficiently affects data collection practices.
   - Some argue GDPR primarily affects the *use* of data rather than the *collection* of it, but that *data collection is turned off for EU consumers.*
- **Zenith: Still on User's Minds**: Members expressed interest in the return of the **Zenith** model to the LMArena, speculating on its potential **ELO** rating and performance.
   - Some members lamented not having the chance to try the model, while others voiced strong opinions on its value to the platform.
- **New Arena Channels for Videos Pop Up**: Community members discussed the existence of **multiple video arena channels** and whether this was intentional.
   - A moderator explained that the video arena channels are meant *to spread the generations out a bit* and that too much activity in one channel *would be a bit much*.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400151134299095092)** (1 messages): 

> `Video Arena, LMArena bot, Staff AMA` 


- **Video Arena Launches on Discord!**: The LMArena team launched an experimental **Video Arena** on Discord, allowing users to generate and compare videos from top AI models for free.
   - Users can learn how to use the bot in the specified channels and start generating videos, images, and image-to-videos, then vote on their preferred generation.
- **Vote on community generated videos!**: The LMArena bot allows users to **generate videos, images, and image-to-videos**, then lets anyone vote on which generations they prefer.
   - After a certain number of votes, the bot reveals the models used to generate each video.
- **Staff AMA with Bot Developer!**: To celebrate the launch of the **Video Arena**, a **Staff AMA** with the bot's developer, [Thijs Simonian](https://www.linkedin.com/in/thijsdev/), was announced.
   - Users were invited to submit questions via [Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSfS_zh67vgEeftUXzX6ujlNiiFoTQNQd7P7-Nx6aD5zpFVOBg/viewform?usp=dialog) for the AMA.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399829791682723901)** (290 messages🔥🔥): 

> `HF Space restarts, P104-100 GPU, LLM Deployment, Qwen 30B, SmolLM3` 


- **HF Spaces unexpectedly restart**: Members discussed why HF Spaces might unexpectedly restart and suggested pinning dependency versions to avoid issues caused by automatic reloads, as described in [the docs](https://huggingface.co/docs/hub/spaces-sdks-gradio-hardware#restarts).
   - One user reported that both *Ilaria RVC* and *UVR5 UI* stopped working, while others worked fine, and a factory rebuild was suggested.
- **P104-100: The $15 GPU**: Users debated the merits of the **P104-100** GPU for AI tasks, with one claiming it's *practically a 1080* for **15 GBP** (though others called it a scam), available from [Taobao](https://item.taobao.com/item.htm?id=897611642586&sku=Video%20memory%20capacity:8gb;Color%20classification:Galaxy%20p104-100%20(card%20length%2028cm);&sku_id=5919375243616).
   - Some noted its limitations (PCIE 1.0 x4, potentially only 4GB VRAM accessible), while others emphasized its price-to-performance ratio for LLM inference, even when sharding models across multiple cards.
- **Tiny LLMs for Edge Deployment**: Members sought advice on **low-latency LLM deployment** in remote, bandwidth-limited marine environments, with suggestions including edge/cloud hybrid approaches and **aggressive quantization**.
   - One user recommended checking out the [latest pytorch quant of smollm3 on HF](https://huggingface.co/pytorch/SmolLM3-3B-8da4w) for mobile deployment, and another suggested deploying apps when docked.
- **Qwen 30B Model: GPT-4o Challenger?**: Users highlighted the release of the **Qwen 30B** model, claiming it rivals **GPT-4o** and can run locally in full precision with just 33GB RAM ([Unsloth GGUF version](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)).
   - It was noted that a quantized version can run with 17GB RAM.
- **SmolLM3 Quantization Problems**: A member mentioned they were having quantization issues with **SmolLM3** using *torchao*, and others suggested trying *hqq* or the [official SmolLM3-3B-8da4w](https://huggingface.co/pytorch/SmolLM3-3B-8da4w).
   - One member noted that *--jinjai* should be used if using *llama.cpp*.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400252994091487382)** (4 messages): 

> `Muon Optimizer, Smithery` 


- **Muon Optimizer Hailed!**: Members shared a link to the **Muon Optimizer** ([https://kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)).
   - They exclaimed *smithery! smithery so killer!*
- **Smithery is killer**: Another member replied that **Smithery** is so killer
   - It seems that **Smithery** is well received by this channel.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399853091410415798)** (5 messages): 

> `Petite Elle Model, Gradio MBTI App, Video Editing MCP Server, Github Python Dataset` 


- **Petite Elle Gets the Mrad Treatment**: A member's model, [Petite Elle-L'aime-3-sft](https://huggingface.co/Tonic/petite-elle-L-aime-3-sft), received the **mradermacher treatment** and is expected to be among the best French-speaking models at that size.
   - The quantized versions are available at [mradermacher's Hugging Face page](https://huggingface.co/mradermacher/petite-elle-L-aime-3-sft-GGUF).
- **Gradio App Flows with Gemini for MBTI**: A new Gradio app for **MBTI (Myers-Briggs)** uses **PocketFlow** and **Gemini**.
   - Check out the [app](https://huggingface.co/spaces/Fancellu/mbti-pocketflow) and the underlying [PocketFlow library](https://github.com/The-Pocket/PocketFlow).
- **MoviePy Makes Video Editing Server**: A member built a **MCP server** using **MoviePy** to handle basic video/audio editing tasks, integrating with clients like **Claude Desktop** and **Cursor AI**.
   - The code is available on [GitHub](https://github.com/Aditya2755/video-edit-mcp), with the author seeking collaboration on features like object detection-based editing and TTS/SST-driven cuts.
- **Github Python Dataset Released**: A new dataset, [Github Python](https://huggingface.co/datasets/jblitzar/github-python), contains all Python files of a reasonable size after 2015 in repositories with over 10 stars on Github.
   - The dataset is filtered for deduplication and permissive licenses.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (2 messages): 

> `Diffusion Models, Flow Matching, MIT curriculum` 


- **Study Group Kicks off for Diffusion Models**: A new **study group** will focus on learning **diffusion models** from scratch, a core architecture in generative AI.
   - The group is based on **MIT’s curriculum** and consists of **12 people** over **5 months** (**2–4 hrs/week**).
- **Free Intro Sessions open to non-members**: The first two free intro sessions are available for non-members: **Aug 2** - What is **Flow Matching & Diffusion Models**?; **Aug 9** - **PDEs, ODEs, SDEs** + A Brief History of Diffusion Models.
   - The sessions are at **12 PM EST**, further details on [Luma](https://lu.ma/kv8zf6va).
- **MIT lecture notes will be used**: The study group will be based on [MIT’s lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) and [previous recordings](https://aischolars.notion.site/).
   - Early sign-up is **$50/month** (goes up to **$100/month**); funds go towards paying the **teaching assistant**.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 messages): 

hedi1421: Thanks 😅
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1400040059947712543)** (1 messages): 

> `Fixing transformers issue, DeepSpeed Integration` 


- **Transformers issue fix sought**: A member is requesting assistance with fixing [this transformers issue](https://github.com/huggingface/transformers/issues/39753).
   - No additional details about the issue were provided.
- **DeepSpeed Integration**: Discussion around DeepSpeed integration within the Hugging Face ecosystem.
   - Members are exploring the best practices and potential performance gains.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1399919753157673102)** (1 messages): 

> `DuckDuckGo deprecation, Smolagents merge` 


- **DuckDuckGo Search Package Faces Deprecation**: The `duckduckgo-search` package is still deprecated, as discussed in [this pull request](https://github.com/huggingface/smolagents/pull/1548).
   - A member inquired about the timeline for its merge into `smolagents`.
- **Smolagents Merge on the Horizon**: The proposed merge aims to integrate updates and fixes into the `smolagents` library.
   - Community members are eagerly awaiting the completion of the merge to leverage the latest improvements.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399995784853848126)** (3 messages): 

> `RAG System Construction, Tool Definition Problems in Unit 1` 


- **RAG System scans conversation history**: A member plans to build a **RAG system** and use an **LLM** to scan conversation history, extract user-specific cases, and embed them in vector space using a for loop.
   - They intend to test the feasibility of this approach.
- **Unit 1 Tool Definition Troubleshoot**: A member reports that their tool definition in **app.py** isn't reflected at runtime in **Unit 1**.
   - They have already tried restarting the space without success and is asking for suggestions.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399829176923717713)** (261 messages🔥🔥): 

> `Study and Learn feature, GPT-5, Copilot, Gemini vs ChatGPT, AI Ecosystems` 


- **OpenAI Releases **Study and Learn** Feature**: OpenAI launched a new **Study and Learn** feature, which some users see as a simple system prompt rather than a significant update and some users believe this feature is **intended to distract users from GPT-5 hype**.
   - One user dumped the [system prompt](https://discord.com/channels/974519864045756446/998381918976479273/1399983224880496711) into O3 model.
- **GPT-5 Speculation Fuels Discussion**: Members are actively debating the release of **GPT-5**, with some claiming to have access via Microsoft Copilot/Azure; however, many users are skeptical and are expecting a formal announcement from OpenAI.
   - One user commented on the CEO by saying *frock you **Sam Altman** for making your chatgpt fans restless waiting for the hypes you create*.
- **Comparing Copilot to ChatGPT**: Users discussed whether **Microsoft Copilot** uses **GPT-5**, however it was clarified that it uses either **GPT-4o** or **O4-mini-high**, and some leakers have found hints in the source code that **GPT-5** may be integrated in the future.
   - The daily message cap for **Copilot** is unlimited, leading some to question why people prefer **ChatGPT** for its tools.
- **ChatGPT and Gemini Comparison Underway**: Users debated preferences between **ChatGPT** and **Google Gemini**, with one user listing six key reasons for preferring **ChatGPT**, including connectors, RAG capabilities, style matching, memory, and deep research, but others were quick to counterpoint them.
   - One user noted, that [Google's Imagen4-Ultra](https://discord.com/channels/974519864045756446/998381918976479273/1400170254902235246) generates the best images after one user posted different AI-generated pictures of an *ultra rich man house like iron man house*.
- **Navigating the AI Ecosystem**: Members discussed choosing the right **AI ecosystem**, weighing options like **Apple One + ChatGPT Plus** against **Microsoft 365 + Microsoft Copilot** or **Google One AI Premium + Google Gemini**.
   - It was suggested to try out both to determine which best suits individual needs, with some mentioning Gemini's integration with Google Docs and Slides.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399902317595328642)** (24 messages🔥): 

> `GPT-5 versions, O4 mini vs 4o, Missing Chat History, ChatGPT memory issues` 


- **GPT-5 Versions Speculations**: Members speculate that mid and high tier versions of **GPT-5** will be superior, with one member noting that **Zenith** might be the top coding model until a new one is released.
   - No links provided.
- **O4 Mini Debated over 4o Model**: A member inquired whether **O4 mini** should be used over **4o (the free model)** for more intelligent responses, referencing **O4 mini** and **O3's advanced reasoning** capabilities.
   - No links provided.
- **ChatGPT Chat Histories Vanish, Stressing Users**: Several users reported their **ChatGPT chat histories disappearing**, with one user logging in and out, clearing cache, and checking multiple devices, to no avail.
   - An OpenAI support member suggested it *could be an isolated bug* and that *chat history recovery isn’t possible once it’s lost*, advising to periodically save copies of important information outside of ChatGPT.
- **Memory Issues Plague ChatGPT Users**: A user mentions they are working on a local workaround for **memory issues** experienced while using **ChatGPT**.
   - This was after another user mentioned that their recent chats after October 2024 were not loading, but they could still access new chats and custom instructions.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `Personalized GPT setup, AI Memory Format, Optimized AI VM` 


- **Personalized GPT Setup Resource Quest Begins**: A new user seeks resources for setting up **GPT projects** to track **food/diet**, **exercise**, and create a **planner** with time expectations, requesting instruction resources and prompts to enhance their account's capabilities.
   - Another member suggests a **personalized approach**, advising direct interaction with the model to discuss desired features and consider additional options, and mentioned that not everyone agrees what 'more powerful' means.
- **New AI Memory Format: Speed vs Readability Debated**: A member introduced a new memory format proposal, aimed at optimized AI VM, systems interfacing with vector databases, or for protected symbolic transfers, emphasizing speed and efficiency over human readability, using a [AI_ONLY_FORMAT_SPEC](https://discord.com/channels/1046317268651794482/1046317269069864970?event=1200057848907847709) to keep memories from being massively inefficient in terms of storage and readability.
   - Another member provided a detailed line-by-line reading of the format, highlighting its core principles like **token embedding**, **semantic grouping**, and **binary encoding**, cautioning against running it due to potential compression or encoding responses.
- **Prompt Engineering Effectiveness Examined**: The discussion involves the effectiveness of prompt engineering, with one member finding it *exhausting to explicitly spell out every aspect of the ask*, preferring to provide context and rely on the AI's reasoning skills.
   - Another member suggests that such conversational users are precisely who prompt engineers design system prompts for.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399879683956277360)** (10 messages🔥): 

> `GPT project guidance, Personalized AI models, AI memory format` 


- **Guidance Sought for GPT Project Setup**: A new user is seeking resources for setting up projects on their **GPT account**, specifically for tracking food/diet and exercise, and creating a planner with time expectations.
   - They are looking for guidance and tools to enhance the instructions for these common projects, potentially making them *more powerful*.
- **Personalized AI is Key**: A user suggests personalizing AI models by discussing desired features and considerations with the model itself.
   - They emphasize that what constitutes *more powerful* varies, with personalization being crucial for tailoring the AI to specific interests and goals.
- **Memory Format for AI**: A member introduced a new memory format suggestion for AI, designed for efficient storage and retrieval of conversational memories.
   - This format aims to improve persistent memory by using a compact, binary-encoded structure with semantic compression and implicit context, optimized for AI VM and vector database interfacing.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399839193567465624)** (2 messages): 

> `DeepTrail, DeepSecure, AI agent authorization, Agent delegation, Policy enforcement` 


- ****DeepTrail** builds open-source **DeepSecure****: A member is building **DeepTrail**, an open source auth and delegation layer for AI agents backed by Berkeley SkyDeck.
   - With **Deepsecure** ([https://github.com/DeepTrail/deepsecure](https://github.com/DeepTrail/deepsecure)), developers can integrate authorization, agent-to-agent delegation, policy enforcement, and secure proxying across any model, platform, or framework with just a few lines of code.
- ****DeepSecure's** technicals under the hood**: The technology involves a split-key architecture, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in the [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
   - There are also several simple examples and integrations for Langchain/LangGraph.
- ****DeepSecure** examples using Langchain/LangGraph**: The member has built some examples of **DeepSecure** integrations for Langchain/LangGraph, including [secure multi-agent workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py) with fine-grained access controls.
   - The repo also features [delegation workflows](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py), [advanced delegation patterns](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py), and [platform agent bootstrapping](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py).


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399849676123013373)** (152 messages🔥🔥): 

> `NotebookLLM, OpenRouter Pricing, Blocking quants via API, Becoming a provider, API Key Issues` 


- ****Unlock Daily Free Messages with a One-Time OR Top-Up****: Adding **$10** to OpenRouter credits unlocks **1000 daily free messages** as a one-time purchase, even if the credits are used up.
   - Users confirmed that even after the initial $10 of credits are exhausted, the **1000 requests/day** limit remains unlocked.
- ****API Enables Quantization Control****: Users can now specify acceptable quantization levels via the API to avoid lower-precision models like **FP4**, using the [provider routing documentation](https://openrouter.ai/docs/features/provider-routing#quantization-levels).
   - The API allows specifying exclusions, like allowing everything *except* FP4 models.
- ****Pydantic-AI and Kimi-K2 Join Forces to Tackle Bugs****: A user highlighted the benefits of **pydantic-ai**, including its fully pydantic approach, MCP server support, model/provider adapters, and automated graph crafting and noted they had fixed a bug using **Kimi-K2**.
   - The user emphasized pydantic-ai's ability to focus on business logic rather than *fiddling with gluing together an agentic framework from disparate bloatparty repos*.
- ****OR Faces Kimi-K2 Tool Calling Issue****: A user believes they've identified an issue with **Kimi K2's** tool calling support on OpenRouter, potentially fixable via model template adjustments.
   - The user provided [research](https://discord.com/channels/1091220969173028894/1400028050007265340) with examples from frameworks like vllm and expressed that fixing this issue could lead to **80% savings** for their business, and suggested that they will move to moonshot.
- ****Gemini Flash 1.5 Faces Overload Issues****: **Google Gemini Flash 1.5** is reportedly showing *error 503: The model is overloaded*, with a user sharing the [pricing structure](https://discord.com/channels/1091220969173028894/1195014798837043240/1400220368765194350).
   - The model has a fluctuating pricing, with input ranging from **$0.075 to $0.15** and output from **$0.30 to $0.60**.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1400206735389491232)** (2 messages): 

> `` 


- **No new models updates in OpenRouter**: There were no significant discussions or updates regarding new models in the OpenRouter channel.
   - The channel remained inactive, lacking any substantive information for summarization.
- **Readybot.io logs no new activity**: The Readybot.io logs indicate a period of silence in the OpenRouter - New Models channel.
   - Consequently, there are no specific topics or discussions to report from this time.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399913686432219186)** (63 messages🔥🔥): 

> `Quantized Providers, Groq's Quantization, Deepinfra Pricing, Vertex for Claude, OpenRouter's Ori bot` 


- **Quantized Providers' Default Status Debated**: A user suggested that quantized providers should be disabled by default, which could affect **Groq** due to its unique quantization approach.
   - Another user cautioned about the risks of publicly shaming providers before reaching a *critical mass* of users, potentially causing providers to exit **OpenRouter**.
- **Deepinfra Offers Gemini 2.5 Pro via Google**: **DeepInfra** reportedly negotiated a lower rate with **Google** for **Gemini 2.5 Pro** and passed the savings onto customers, confirmed by a user who cited a message from DeathMax and the presence of a 'partner' tag on DeepInfra's listing.
   - DeepInfra's **Gemini 2.5 Pro** has a "partner" tag unlike the **Kimi K2** model, indicating a direct partnership with Google.
- **Vertex proves Victorious for Claude 4**: One user reported better quality, throughput, and uptime by using **Vertex** for **Claude 4 Sonnet**.
   - The user also noted that AWS/GCP/Azure mirrors for closed models could provide qualitative differences.
- **OpenRouter Ori Bot's Accuracy Under Scrutiny**: A user suggested that **OpenRouter's Ori bot** might be a *net negative* due to inaccurate responses and should be limited or disabled.
   - The user pointed out that **Ori** often puts the fault on the user and asks questions that *lead nowhere*, especially in payment processing issues.
- **Adding Knowledge Update Feature for Ori bot**: One of the developers is working to add ways to update **Ori's** knowledge when it gets things wrong.
   - Others pointed out that **Ori** is missing a lot of knowledge and is hallucinating incorrect knowledge, and suggested limiting the bot's answers to a maximum of 2-3.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399843438395916339)** (188 messages🔥🔥): 

> `Metislist ranking, Arcee.ai Releases AFM-4.5B, NotebookLM video overviews, Claude abuse, ryOS release` 


- **Metislist Sparks Chollet Placement Debate**: A user shared a link to [Metislist](https://www.metislist.com/), a ranking of individuals in the AI field, which spurred discussion about François Chollet's placement at number 80.
   - Many agreed that Chollet, known for Keras, should be in the top 50, or even removed from the list entirely, one user joking *Damn you got beef with my boy François?*.
- **Arcee AI ships AFM-4.5B models**: Lucas Atkins announced the release of **AFM-4.5B** and **AFM-4.5B-Base** models on Hugging Face from [Arcee.ai](https://xcancel.com/LucasAtkins7/status/1950278100874645621), highlighting their design for flexibility, high performance, and quality due to a data partnership with DatologyAI.
   - The models feature architectural tweaks like **grouped query attention** and **ReLU² activations**, and the team plans to release future models for reasoning and tool use.
- **NotebookLM now does Video Overviews**: **NotebookLM** announced a new feature for video overviews of articles and blog posts ([xcancel.com](https://xcancel.com/NotebookLM/status/1950298236914139234)), enabling users to quickly grasp content without reading everything.
   - Users praised the innovation and suggested further development for learning tools and interactive modes.
- **GPT-5 spotted prowling MacOS**: References to **gpt-5-auto** and **gpt-5-reasoning** have been found in the MacOS app cache files ([xcancel.com](https://xcancel.com/apples_jimmy/status/1950514936444305534?s=46&t=fRVjULzONZQAlwHruKTgQg)), suggesting the imminent release of **GPT-5**.
   - Other users have corroborated this, mentioning **gpt-5-reasoning-alpha** in a biology benchmarks repository, while some speculated on an imminent announcement or release.
- **Anthropic Seeks Sky-High Valuation**: Anthropic is reportedly in talks to raise **$5 billion**, potentially valuing the AI startup at **$170 billion**, with projected revenue of **$9 billion** by year-end ([xcancel.com](https://xcancel.com/EdLudlow/status/1950561790695448810)).
   - This news sparked comparisons to other AI companies like OpenAI and xAI, although one user commented that he has *some second hand info that this is not the case*.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400171262210474005)** (17 messages🔥): 

> `Anthropic Fellows Papers, LLM Paper Club, Social Media Engagement` 


- **Anthropic Fellows Papers Featured**: The Latent Space Discord announced it would cover recent **Anthropic Fellows papers** in the <#1107320650961518663> channel.
- **LLM Paper Club Seeks Volunteers**: The **LLM Paper Club** is seeking volunteers to cover papers in future clubs; interested individuals are encouraged to sign up via a [Luma link](https://lu.ma/6uti3zzy).
- **Call for Retweets Misses the Mark**: A member posted a [link on X](https://x.com/latentspacepod/status/1950613048303231121) advertising the club, but lamented his low engagement.
   - He jokingly claimed to be *“not professional yapper”* and bad at tweeting.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399847059045093637)** (32 messages🔥): 

> `Tableau Vizql NLP Orchestration, Gemini Agentic Framework Prototype, NotebookLM Podcast Creation, Obsidian and NotebookLM Integration, NotebookLM Usage Analytics` 


- **Tableau Server used for LLM Orchestration**: A member mentioned using the latest **Tableau Server edition** to fit in the **LLM (server/on prem)** for **Vizql NLP**.
- **Gemini Agentic Framework Prototype Shared**: A member shared a [**Gemini agentic framework**](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688bd3f6&is=688a8276&hm=101f03e62cae13a72e1f4fdc681064aef0e5a3713de20aebac608c958f845b8b) prototype, noting it was a **one-shot prototype**.
   - They suggested using **AI Studio** to build agentic apps by detailing intentions clearly for the builder agent to create a working prototype, allowing for phased testing and model focus.
- **Bypassing Bot Restrictions for Podcast Creation**: A member inquired about creating podcasts with **NotebookLM**, given bot restrictions on logins.
   - Another member clarified that the tools driving **NotebookLM** are available via the **API**, suggesting a rebuild in another workflow and manual report loading into **NotebookLM**.
- **Obsidian and NotebookLM Integration Discussed**: A member shared an article on integrating **NotebookLM**, **Obsidian**, and **Google Drive** [here](https://www.xda-developers.com/using-notebooklm-obsidian-google-drive-together/).
   - Another member offered to provide more specifics on using **Obsidian**, depending on the other member's usage.
- **NotebookLM Audio output averages 8-15 minutes**: Some members asked about long audio file generation with **NotebookLM**.
   - Another member stated that their average output is *8-10 minutes*, although other members had generated up to *15 minutes*.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399828961021788412)** (156 messages🔥🔥): 

> `NotebookLM Video Overview Limits, Studio UI Changes, Video Generation Length, NotebookLM Rollout, Missing the new Notebook LM Feature` 


- **Pro Users Get 20 Video Overviews Daily**: Pro users of NotebookLM receive **20 video overviews per day**, as confirmed by a member, and found in [Google Support documentation](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.).
   - However, some users are experiencing delays in accessing the Video Overview feature and the updated Studio UI despite being Pro users.
- **Studio Interface Needs Sort/Filter Outputs**: A user suggested that the **Studio interface** needs the ability to *sort/filter outputs* and a *delete all option*, as well as the ability to **stop a video generation in progress** that won’t complete.
   - Another user highlighted that the *save all notes to source is gone*, potentially creating an issue with the source limit of **50** in the free version.
- **Video Generation Times Vary Wildly**: Users reported varying video generation times, with one user experiencing a **30-minute** generation from an Economist article, leading to discussion on whether **Veo 3** is being used.
   - A user described the output as *more of a presentation than an animated video*, noting its tendency towards typography, suitable for text-heavy content.
- **Rollout Reaches Germany, Slow for Others**: The **Video Overview feature** became available in **Germany** on Pro accounts, while many users, including those with Google Ultra, are still waiting for the rollout.
   - Google confirmed that the update is rolling out gradually to all users over the next week.
- **Video Overview Glitches Exposed**: Users are reporting that video overviews are limited to **6-7 minutes** and are experiencing abrupt transitions between sections.
   - There's a glitch where the video loads indefinitely until the page is refreshed.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399838809134334052)** (117 messages🔥🔥): 

> `LM Studio model renaming, Earthquake off the coast of Russia, LM Studio copy/paste conversation feature request, LM Studio Model stuck in time, Qwen 30B garbage output` 


- **Tsunami warning off the coast of Russia**: An **8.7 magnitude earthquake** occurred off the coast of Russia triggering tsunami warnings for Hawaii and watches for the west coast of the United States.
   - Residents in affected areas were advised to monitor updates closely due to potential tsunami arrival hours later.
- **Feature Request for LM Studio: Copy-Pasting Whole Conversations**: A user inquired about a feature in LM Studio to copy and paste entire conversations, which are stored in **JSON format** and relatively easy to manipulate.
   - Another user mentioned they started a Python app for extracting conversations but got distracted, suggesting others add a feature request in the [feature request channel](https://discord.com/channels/1110598183144399058/1128339362015346749) as many would find it useful.
- **LM Studio Model Stuck in Time Loop**: A user reported that their LM Studio model was repeatedly referencing **February 18, 2024**, even when asked about current events, providing a screenshot as evidence.
   - Another user suggested checking the **system prompt** or **Jinja template** for the date, as it might be causing the model to think it's in that specific date.
- **Qwen 30B's Sampler Settings**: A user noted that **Qwen 30B** frequently produces garbage output unless the prompt is reprocessed.
   - Another user suggested trying the official samplers or the ones provided to see if the output improves, with one noting a similar issue on Linux that was resolved by updating to experimental drivers.
- **lmstudio MCP client needs resources**: Users discussed the potential use of resources in the **LM Studio MCP client**, highlighting use cases such as lower-cost read-only references for syntax guides or dynamic codes.
   - One user mentioned they use resources for discovery, documentation, and navigation helpers, preferring resources over tool calls for quick reference information and hoping for client updates to support features from the **2025-06-18** update of the MCP specification.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399876515449147545)** (64 messages🔥🔥): 

> `GPU Usage, Strix Halo, Threadripper vs Epyc, Soldered RAM, 9070 XT Performance` 


- **Splitting Models Across GPUs Affects Usage**: When splitting a model between two GPUs, each GPU operates at **50% usage**, potentially reducing heat and noise, though this is dependent on how the model layers are split and processed sequentially.
   - A 4090 paired with a slower 3070 might result in the 4090 being idle while waiting for the 3070 to complete its tasks, but improvements in performance are still seen with an increase from **8 to 32 tok/sec**.
- **Strix Halo APUs Get Mixed Reviews**: The price of **Strix Halo APUs** seems to be set at around **$1.6k** for 64GB and **$2k** for 128GB, but some members suggest that EPYC systems offer better value for the price due to greater memory bandwidth and upgradeability.
   - One member lamented *soldered memory* on such devices, drawing a comparison to a recent DIMM failure on a server and pointing to a [Corsair AI Workstation 300 with Strix Halo APU](https://www.guru3d.com/story/compact-ai-pc-corsair-ai-workstation-300-with-strix-halo-apu/).
- **Threadripper and Epyc Face Off!**: While **Threadripper** is often considered best for consumers, **EPYC** can be a cheaper option due to the availability of refurbished parts, unlike **Threadripper**, which tends to be more expensive and harder to find.
   - A member noted that Epyc is cheaper because *there's quite a market of refurb/sh parts*, and pointed to [this reddit thread](https://old.reddit.com/r/LocalLLaMA/comments/1mcrx23/psa_the_new_threadripper_pros_9000_wx_are_still/) to further discussion.
- **Is Soldered RAM a SCAM?**: Members expressed confusion over why people buy PCs with *soldered storage*, especially given the high prices and limited memory bandwidth, such as **2500€** for a **128GB** soldered RAM device with **256GB/s** memory bandwidth.
   - One user stated *it's like asking to be scammed*, while another likened the concept to a console where everything comes in one package, albeit with the ability to build a better PC for the same price.
- **9070 XT Gets Humble Pie in Performance Tests**: The **9070 XT** is significantly slower than a **4070 Ti Super**, with one user reporting that a model running at **7 t/s** on their **4070 Ti Super** only achieved **3 t/s** on the **9070 XT**; however, another member suggested that RAM bandwidth limitations might be the cause.
   - It was stated that CUDA is good, but maybe Vulkan is not bad either.  One member found a **5070 Ti** for **749 eur**, but the price jumped to **1100 eur** the next day.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399844353538523288)** (4 messages): 

> `LlamaCloud document agents, LlamaCloud Managed Embeddings, Automated Asset Manager Fund Analysis, LexiconTrail agentic AI systems` 


- ****AI Agents Parse Financial Docs****: Transform complex financial documents into actionable data with **AI-powered document agents** that handle real-world formats like **10-Ks**, earnings reports, and regulatory filings; more information through the [LlamaIndex Webinar](https://twitter.com/llama_index/status/1950285220663742516).
- ****LlamaCloud Manages Embeddings****: **LlamaCloud Indexes** now have managed embeddings, meaning you no longer need to bring your own API key to embed your content; vectors will be embedded for you in addition to hosting them, according to [this Tweet](https://twitter.com/llama_index/status/1950345618779754644).
- ****Automated Asset Manager Fund Analysis Available****: Build an automated asset manager fund analysis with this comprehensive notebook, that shows how to process complex financial documents and extract actionable insights for investment analysis using **LlamaParse** to convert PDF to structured markdown, outlined in [this Tweet](https://twitter.com/llama_index/status/1950590734685671931).
- ****LexiconTrail 10x Agentic AI Systems****: **LexiconTrail** demonstrates how to build **10x faster agentic AI systems** using **NVIDIA Small Language Models** with advanced indexing capabilities, according to [this blog post](https://twitter.com/llama_index/status/1950662723785850911).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399848962025984000)** (126 messages🔥🔥): 

> `LlamaCloud PDF detection issues, Character AI architecture, Neo4j Knowledge Graph issues, Flowmaker Gemini 2.5 Pro bug` 


- **LlamaCloud can't detect PDFs, member needs direction**: A member reported that **LlamaCloud** could not detect a **PDF file** and process it via API, using **n8n** for workflow simplification and linked to a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688bcff0&is=688a7e70&hm=b8f51e99fbeae087df203303f7665c4eab8447bb0890b55823fd36074c5ad539&).
- **Character AI Building Discussion sparks**: Members discussed building a **character AI** with deep understanding of a large story, using a classic **RAG** pipeline with chunked text, embeddings, and a vector database.
- **Neo4j Woes and Graph Storage Overload**: A member is trying to implement **Neo4j** as their simple graph storage takes *ridiculously long to load*, but their server is not compatible with **Neo4j 5.x**, and **LlamaIndex** doesn't seem to like **4.x**, and **Aura** is blocked by the server proxy.
- **Flowmaker Fix Flashes Fast for Gemini 2.5 Pro Bug**: A member reported an error when using **Flowmaker** with **Gemini API**, due to an invalid model name and another quickly pointed out that [the model name](https://ai.google.dev/gemini-api/docs/models) required a number, e.g. *gemini-2.5-pro*.
   - A fix was [committed](https://github.com/run-llama/flow-maker/blob/aad0f47a81cacba662a07c4f2d70bd3425606e29/src/lib/llm-utils.ts#L19) and deployed swiftly, resolving the issue, where the member thanked them for their swift assistance.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1400095143251677184)** (3 messages): 

> `RAG debugging, Sparse retrieval, Semantic drift, Chunking collapse, Memory breakdowns` 


- **User offers RAG debugging assistance with MIT-licensed Repo**: A member offered help with a **MIT-licensed repo** designed to debug tricky **RAG issues**, including sparse retrieval, semantic drift, chunking collapse, and memory breakdowns.
   - Another member asked to share complex problems solved with the repo, specifically asking for more details on *sparse retrieval* and *semantic drift*.
- **Inquiries Regarding Specific RAG Debugging Issues**: Following the initial offer, a community member inquired about specific complex issues addressed by the MIT-licensed repo, focusing on concrete examples.
   - The inquiry specifically requested detailed instances of how the repo tackles **sparse retrieval** and **semantic drift**, seeking a more granular understanding beyond the general descriptions provided.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400002443323904084)** (5 messages): 

> `Expert Parallelism (EP) vs Tensor Parallelism (TP), Merge Sort troubles on GitHub` 


- **Expert Parallelism Experiences Explored**: A member is seeking examples where **Expert Parallelism (EP)** outperforms **Tensor Parallelism (TP)**, noting that in their experience with **Qwen32B** and **Qwen 235B**, the added communication from all-reduce operations after attention makes **EP** less performant.
   - They are finding **EP** only useful for models employing **MLA** and requiring **DP attention**.
- **Merge Sort Remainder Rescue Requested**: A member needs help with merge sort remainder issues in their [RinomXE GitHub project](https://github.com/maybeJosiah/RinomXE).
   - They are struggling with the draw order remainder logic, where doubling steps until over shape number fails to sort correctly and posted a javascript snippet for simulation to help.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399868515422441522)** (17 messages🔥): 

> `Torch Compile, Triton Code Generation, PTX Code Extraction, Inductor Configuration, GEMM Autotuning` 


- **Unlock PTX and Triton Codes from Torch Compile**: To get the **PTX code**, use `TORCH_LOGS="output_code" python your_code.py` or access the `compiled_kernel.asm.keys()` dictionary as detailed in [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir).
   - The dictionary contains keys for different intermediate representations including **llir, ttgir, ttir, ptx, and cubin**.
- **Bypass Non-Triton Code Generation in Torch Inductor**: To force **Triton code generation** for matmuls, configure settings in [torch._inductor.config.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py#L459-L461), but be aware that not every op is converted to Triton by default.
   - Options like **max_autotune_gemm_backends="TRITON"** and **max_autotune_conv_backends** can influence the autotuning process, although built-in kernels are often faster.
- **Achieve Pure Triton Code by Tinkering Inductor Configs**: To get inductor to *only* use triton code, members recommend modifying `config.py` and `utils.py`, specifically settings such as **use_aten_gemm_kernels**, **use_triton_template**, **autotune_fallback_to_aten**, **max_autotune_conv_backends**, and **max_autotune_gemm_backends**.
   - This involves preventing autotuning and fallback to prewritten kernels, potentially requiring exploration of the **'/tmp/torchinductor_{username}'** directory.
- **TMA Support Arrives with Triton 3.4.0**: **TMA (Tensor Memory Accelerator) support** is not yet available in the official Triton release; users must await version **3.4.0**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399872492797038774)** (9 messages🔥): 

> `livestream review, request accepted` 


- **Streamer plans Review Livestream**: A streamer was asked to do a [livestream review](https://www.twitch.tv/) for the community.
   - The streamer responded *I'm not sure if this is the confirmation mail. I'll confirm once I'm free! i doubt it's a welcome to our email list*.
- **Requests Accepted!**: A member stated the team has accepted all requests.
   - A different member confirmed their request was accepted: *Although I'm new to this but something cool to explore*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400083758618775636)** (2 messages): 

> `CUPTI metrics in kineto, torch.profiler metrics` 


- **Struggle enabling CUPTI metrics in kineto**: A member inquired about enabling **CUPTI metrics** in **kineto**, possibly through a custom build, within the **torch.profiler**.
   - They referenced a [relevant pull request](https://github.com/pytorch/pytorch/pull/125685), but indicated that it did not resolve their issue.
- **torch.profiler Configuration**: The member attempted to use **torch.profiler** with specific configurations to measure kernel performance.
   - They tried to configure **experimental_config** with **profiler_metrics** such as *kineto__tensor_core_insts*, *dram__bytes_read.sum*, and *dram__bytes_write.sum*, setting **profiler_measure_per_kernel** to True.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399912770396487731)** (6 messages): 

> `CUDA streams, Megatron-LM, Group GEMM, NYC Hackathon, Beginner Hackathon Tips` 


- **CUDA Streams and GEMM Performance**: A member inquired about the advantages of using **multiple CUDA streams** when running **GEMM kernels**, particularly in the context of **Megatron-LM** and **cuBLAS multi-stream Group GEMM**.
   - The user questioned the benefits compared to a single stream, noting concerns about overhead and the limited number of thread blocks.
- **Hackathon in NYC**: A member asked about a hackathon, and another member pointed them to a specific channel for more info.
   - The hackathon appears to be located in NYC.
- **General Hackathon tips for beginners**: A member shared a [link to general hackathon tips](https://x.com/ayushgun/status/1950444463899512960) on X, noting it's useful for beginners.
   - The tips are very general and not specific to GPUs.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 messages): 

ali_8366: Anyone here from Montreal? Would love to have a coffee chat
  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/)** (1 messages): 

vishomaru: Hello, anybody here was successful in profiling compute shaders with AMD GPU Profiler?
  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1399941036360601765)** (3 messages): 

> `AI Hackathon, CuTeDSL Blogpost, Software Pipelining` 


- **AI Hackathon Post Plugged**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/nadiveedishravanreddy_ai-hackathon-qwen-ugcPost-7355265897877434369-7RI5?utm_source=share&utm_medium=member_android) about an **AI Hackathon**.
   - The event now features **15 speakers**, including representatives from **Prime Intellect, Snowflake, and Jane Street**.
- **Course Plug with Star-Studded Speakers Lineup**: A member re-plugged a course, mentioning that it now includes **15 speakers** such as **Prime Intellect**, **Snowflake**, **Jane Street (Sylvain Gugger)**, and **Daniel Han** ([course link](https://maven.com/walk-with-code/scratch-to-scale?promoCode=gpumode40)).
   - They encouraged those with cost concerns to reach out for a discussion about potential assistance.
- **Compiler Automates CuTeDSL Optimization**: A member shared a [blogpost](https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/) and [code](https://github.com/simveit/software_pipelining_cute_dsl) on using **CuTeDSL** for **GEMM on H100**, detailing how to let the compiler handle prefetching.
   - The blogpost explains an experimental argument to the `cutlass.range` operator to hint for prefetching, achieving performance on par with manual prefetching with simpler code.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1400119615429804166)** (3 messages): 

> `Popcorn-cli DeserializationError, BadCredentialsException on MI300, B200 Timeout Issues, Discord Run Errors` 


- **Popcorn-cli hits DeserializationError on H100 and A100**: A user reported a *"DeserializationError | Raw Error: Deserialization failed because the 'libkernelbot' module is not available in the local environment"* when using the latest **popcorn-cli** version built from source on **H100** and **A100** GPUs.
   - The error is impacting **H100** Discord runs as well.
- **MI300 faces BadCredentialsException**: The user also encountered a *"BadCredentialsException | Raw Error: 401 {"message": "Bad credentials", "documentation_url": "https://docs.github.com/rest", "status": "401"}"* error on the **MI300**.
- **B200 Timed Out**: The user experienced a **300s timeout** on the **B200** for a run that previously completed successfully two weeks prior.
- **Popcorn devs are on the case!**: A member stated that the team is aware of the issues and actively working on a fix.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400147055963541671)** (5 messages): 

> `Benchmarking Explanation` 


- **Benchmarking Explanation Meeting Delayed**: Members coordinated a meeting to explain the benchmarking process, but the original meeting ended early due to low attendance.
   - A member apologized for oversleeping but confirmed availability for a follow-up meeting to explain benchmarking.
- **Overslept Member Still Available**: Despite oversleeping, a member confirmed they are still available to explain the benchmarking process.
   - The member aims to reschedule and provide the benchmarking explanation as planned.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399949797603020840)** (20 messages🔥): 

> `gmem synchthreads, cp.async.cg vs cp.async.ca, cutedsl ptx wrapper, nvvm wrapper, cutedsl older drivers` 


- **SyncThreads Savior after Gmem Transfer**: After copying from global memory (**gmem**) to shared memory, manually inserting a `synchthreads` (or equivalent) is necessary.
   - This ensures that all elements in shared memory have arrived before participating in collective calculations like **gemm**, reduction, or scan.
- **Controlling Cp.Async Choice in Cutedsl**: A member asked about controlling whether it’s `cp.async.cg` vs `cp.async.ca` in **cutedsl**.
   - The suggestion was to write custom assembler code and provide it as a copy operation, although this was not tested.
- **PTX Wrapper Revelation in Cutedsl**: There is no API for **ptx** wrapper in cutedsl, according to one member.
   - However, another member shared a link to example code on how to do it and said *In the official CuTeDSL code there is also something.* ([quack/utils.py](https://github.com/Dao-AILab/quack/blob/main/quack/utils.py#L67)).
- **Nvvm Wrapper Navigation Notes**: A member shared a link on how to write **nvvm** wrappers.
   - They shared a link to the [cutlass repo](https://github.com/NVIDIA/cutlass/blob/main/python/CuTeDSL/cutlass/cute/arch/nvvm_wrappers.py) as an example and a link to the [cutedsl docs](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html#module-cutlass.cute.nvgpu.cpasync).
- **Cutedsl Compatibility Concerns Clarified**: A member asked if it's okay to use **cutedsl** with older drivers.
   - They hadn't encountered any issues but wondered if internal testing had revealed any problems.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400002892735189022)** (1 messages): 

> `Distributed Training, LLMs, Distributed memory tricks` 


- **Ultrascale Playbook a Fantastic Resource**: The [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) on Hugging Face Spaces is a fantastic resource for distributed training of **LLMs**.
- **Memory optimization for Distributed Training**: The playbook offers a lot of distributed memory tricks for training LLMs.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399836795881132233)** (38 messages🔥): 

> `GPU Inference vs M3 Ultra, LLMs Offshore with low latency and bad internet, Topological data analysis experts, Speech-LLM models and audio instruction-following capabilities, Manipulating vector embeddings for machine translation` 


- **M3 Ultra as the local inference King**: A member suggested buying an **M3 Ultra** for local inference, linking to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/) discussing its **80-core GPU** and **512GB memory**.
   - Another member shared that they bought a used **M1 16g** citing the lack of response from other members.
- **Running LLMs Offshore with low latency**: A member is trying to get **LLMs running offshore** with low latency and bad internet, seeking suggestions on where to look for solutions.
   - Another member responded that spending hundreds/thousands is an acceptable use case if it's what they want to do.
- **Speech-LLM Audio Instruction Research**: A member expressed interest in conducting open-source research on improving **audio instruction-following capabilities** in **Speech-LLM** models, crucial for creating reliable user interfaces with speech integration.
   - They noted that the latest research is **Alignformer** from Microsoft, but its code is not open source and is gauging interest in collaboration.
- **Vector Embedding Manipulation for Machine Translation**: A member is planning to manipulate a vector embedding in a vector space off based off of a multilingual model.
   - The member wants to take the embedding and add the differences of the mean vectors of both languages, then rotate them till the loss is minimal around the new language mean, which someone else pointed out is a solved problem in [this paper](https://arxiv.org/abs/1309.4168).


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400101077415886860)** (2 messages): 

> `REST models, Compute cost` 


- **Community Model Payment Protocol Idea Floated**: A member wondered if a community model served via REST could use a **402** response to quote compute cost and enable client auto-payment.
   - They pondered how *single-rail vs. h402 multi-rail affects openness* in such a payment system.
- **Openness Implications**: Discussion revolves around the implications on openness when implementing a **402**-based payment system.
   - Concerns raised about single-rail versus multi-rail approaches.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399832395791728763)** (17 messages🔥): 

> `In-Context Learning (ICL), Interpretability Tools, Sparse Autoencoders (SAEs), Lucas Critique, Activation Distributions` 


- ****ICL** Might Break **Interpretability Tools**, Claims Spark Concerns**: A member speculated that **in-context learning (ICL)** could potentially break **interpretability tools** like **Sparse Autoencoders (SAEs)** by pushing activations out of the distribution they were trained on.
   - The member referenced the **Lucas Critique**, arguing that interventions (like prompting an LLM) require predictions based on microfoundations invariant to those interventions, and [shared a paper](https://arxiv.org/abs/2501.00070v1) to support their view.
- ****SAEs** Face Generalization Challenges with **ICL****: A member agreed that applying **SAEs** to contexts with significant **ICL** would likely fail because sparse representations don't generalize well to activation distributions they weren't trained on.
   - They clarified that this issue isn't specific to **ICL** but arises whenever **SAEs** are applied to activation distributions different from those they were trained on.
- ****ICL**'s Impact on Activation Distributions: **OOD**?**: A member posited that conditioning a model's behavior to a tiny slice of a large distribution via **ICL** might break diagnostics built for unconstrained circumstances, potentially leading to novel internal behaviors.
   - Countering, another member suggested **ICL** might push activations *in-distribution*, citing examples where **SAE** features activate on specific instances within a context, pointing to papers on *function vectors/task vectors*.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399902669526663381)** (1 messages): 

> `Model Evaluation Metrics` 


- **Debugging Model Evaluation Metrics**: A member offered to assist in debugging a function that takes a single input document and model predictions to return evaluation metrics.
- **Understanding Function Processes**: The suggestion involved understanding how the function processes data to identify potential issues.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 messages): 

> `Diffusion Models Study Group, Flow Matching, MIT Curriculum` 


- **New Diffusion Models Study Group Launches**: A new **12-person, 5-month study group** is starting to learn diffusion models from scratch, based on **MIT’s curriculum** ([lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf)).
   - The group is designed for those working in AI and includes CTOs, AI instructors, and AI researchers.
- **Attend Free Intro Sessions on Flow Matching and PDEs**: The study group is hosting **two free intro sessions** on **August 2nd** ([Flow Matching & Diffusion Models](https://lu.ma/kv8zf6va)) and **August 9th** ([PDEs, ODEs, SDEs + A Brief History of Diffusion Models](https://lu.ma/uk6ecrqo)), both at 12 PM EST.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399899213365645353)** (4 messages): 

> `MoE Implementation, grouped_mm, Low Precision Training, Float8 Training` 


- **Grouped GEMM Implementation Talk**: A member inquired about a PR supporting **torch._grouped_mm** in **GPT-NeoX**, now available in PyTorch core, for potential performance benefits, specifically mentioning [this MoE implementation](https://github.com/EleutherAI/gpt-neox/blob/d12c771198388980ee054617e537665f044e0584/megatron/model/moe_mlp.py#L221).
   - They said that users interested in **low precision MoE training** could use a one liner from TorchAO.
- **Digging Into PyTorch's Grouped GEMM Implementation**: A member asked about the underlying implementation of PyTorch's **_grouped_mm** and requested performance comparisons with megablocks grouped GEMMs.
   - Another member pointed out that it uses a **CUTLASS kernel** under the hood, linking to the [relevant source code](https://github.com/pytorch/pytorch/blob/62f98dbb44fb338ba849f93c491ea170af4c187c/aten/src/ATen/native/cuda/GroupMM.cu#L418).
- **Float8 Blockwise Pretraining Renaissance**: One member questioned the perceived lack of interest in **low-precision training** due to convergence issues, claiming it a *"hard sell unless perf is attractive.*"
   - Another member countered, citing **DeepseekV3's float8 blockwise pretraining** and their own stable convergence results with **FP8 rowwise**, achieving ~30-40% throughput improvement as detailed in [this PyTorch blog post](https://pytorch.org/blog/accelerating-large-scale-training-and-convergence-with-pytorch-float8-rowwise-on-crusoe-2k-h200s/).


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400078020387147887)** (7 messages): 

> `CUDA generalization paper, TLS handshake EOF error, Mojo package installation, Region-specific access issues` 


- **CUDA Generalization Paper Shared**: A member shared a [paper on generalizing beyond CUDA](https://huggingface.co/papers/2507.14111).
   - Another member thanked them for sharing and suggested posting such content to the appropriate channel in the future.
- **User Faces TLS Handshake EOF Error**: A new Mojo user reported encountering a **TLS handshake EOF error** while trying to run a Mojo project through both **pixi** and **magic shell**.
   - They noted issues with installing a **dockerfile** obtained from Microsoft Copilot.
- **Suggested Solution for TLS Handshake Issues**: A member suggested that the **TLS handshake issues** could be related to region-specific access to package repositories and provided a solution to try the new `mojo` package in the latest nightly using **pixi**.
   - The suggested commands were: `pixi init my-project -c https://conda.modular.com/max-nightly/ -c conda-forge`, followed by `cd my-project` and `pixi add mojo`.
- **VPN Fails to Fix Installation Issue**: The user who encountered the **TLS handshake EOF error** reported that the suggested solution did not work, even with a VPN.
   - Another member mentioned that region issues should be resolved since a migration to a new host a few months prior.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399916351182868503)** (41 messages🔥): 

> `Mojo external calls vs libc, Mojo to Python overhead, Embedding CPython in Mojo binaries, Python performance, Mojo and hot loops` 


- **Mojo's external calls raise questions**: Users questioned why Mojo's `external_call` uses specific functions like `KGEN_CompilerRT_IO_FileOpen` instead of standard `fopen` from libc, and whether this is for safety.
   - A member clarified that many of these are artifacts from when Mojo was less capable and are not a high priority to fix right now, and that the KGEN namespace belongs to Modular and will be opened up eventually.
- **Mojo to Python overhead seems steep**: A user found that calling a Python no-op function from Mojo is significantly slower (4.5 seconds for 10 million calls) compared to calling it directly from Python (0.5 seconds), noting that this difference is more significant than Python to Rust interop with Pyo3.
   - Others chimed in pointing out that Mojo needs to start a CPython process to execute Python functions, incurring overhead, and comparing it to Rust to Python interop is not an equal comparison.
- **Embedded CPython is in the binary**: Discussion arose whether CPython is embedded in the Mojo binary or if the Python code is compiled, affecting the performance overhead of Python calls from Mojo.
   - It was clarified that CPython is embedded via `dlopen libpython` and a pointer to the interpreter is maintained, and every call reuses the same interpreter, so you shouldn't call it in a hot loop for performance reasons.
- **Low latency trumps hot loops in mojo**: Discussion on performance impact of calling Python from Mojo, especially in hot loops for tasks like OpenCV or Mujoco robot simulation, emphasizing that doing so is like *spraying glue in a race car engine*.
   - Members note that a lot of fast python libs are actually C libs with a wrapper, and that *interacting with the context dicts alone can easily eat several hundred cycles*.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1399878169456152708)** (38 messages🔥): 

> `Aider site framework, Deepseek-chat OpenRouter Issues, SWE-bench Leaderboard, Aider's Role in Model Training, Qwen3 Coder 30B-A3B announcement` 


- **Deepseek-chat Dumbs Down on OpenRouter**: Members noticed **Deepseek-chat** performing worse on **OpenRouter** compared to the official **DeepSeek API**, returning entire functions instead of selective diffs when used as the editor model in architect mode.
   - Using `aider --model openrouter/deepseek/deepseek-r1` was suggested as a fix, because that ensures usage of the default config at [aider/resources/model-settings.yml](https://github.com/Aider-AI/aider/blob/main/aider/resources/model-settings.yml#L548) which has the `edit-format: diff` setting.
- **Aider Could Train Coding Models**: It was suggested that **Aider** could assist with coding model training by logging where linting or undo actions were needed during development workflows.
   - This would leverage Aider's use in *careful* development to provide valuable training data, though the commenter did not advocate for the developers to implement this feature.
- **Qwen3 Coder 30B-A3B Announced**: An image was shared of the new **Qwen3 Coder 30B-A3B**.
   - The image in question was a screenshot of the announcement, validating its legitimacy.
- **User Encounters Litellm API Connection Errors**: A user reported encountering numerous `litellm.APIConnectionError: Error parsing chunk: Expecting property name enclosed in double quotes: line 1 column 2` errors.
   - The errors didn't seem to impact functionality.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400180575096012831)** (3 messages): 

> `Open Model Selection, Hardware Considerations for Aider, Runpod Credits, R1 Model, Qwen Coder Model` 


- **Open Model Showdown: R1 vs Qwen Coder**: A member sought advice on the best open model to use with **aider**, given unlimited hardware and pondered testing **R1** and **Qwen Coder** models.
   - The member mentioned having **Runpod credits** to burn, suggesting an intention to conduct practical tests with these models.
- **Llama3 faces Integration Discussions With Aider**: Members had integration discussions regarding **Llama3** and compatibility issues with Aider.
   - Some members suggested some helpful integration improvements to existing model options.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399838202122207312)** (29 messages🔥): 

> `LLM Safety Alignment Research, AI alignment blogs, CUDA with Claude, Z.AI 54 open source repos, Math in paper discussion` 


- **LLM Safety Alignment Research Resources Sought**: A PhD student sought suggestions on how to catch up on current **LLM safety/alignment research**, particularly good survey papers.
   - Suggestions include four blogs from **AI alignment forum**, including [Thought Anchors: Which LLM Reasoning Steps Matter](https://www.alignmentforum.org/posts/iLHe3vLur3NgrFPFy/thought-anchors-which-llm-reasoning-steps-matter), [Measuring Beliefs of Language Models During Chain-of-Thought](https://www.alignmentforum.org/posts/a86uAnPykqNtmEbDH/measuring-beliefs-of-language-models-during-chain-of-thought-1), and more.
- **CUDA Coding with Claude causes Complexities**: A member found that writing in **CUDA** with the help of **Claude** is complex, requiring planning, deep understanding, and organization.
   - They suggested that the real test of intelligence would be if a Python developer with some GPU and CUDA knowledge could use **Claude** to navigate writing **kernels** and optimize performance, and included an [image](https://cdn.discordapp.com/attachments/986699377257119794/1400233738369110026/image.png?ex=688be4ca&is=688a934a&hm=1bcb11346477e61edf05cde9751d5e62ee8992a2f64216c07e4a1a8f8fb14cc4).
- **Z.AI 54 Open Source Repos Spark Interest**: A member asked if others have seen the new **Z.AI 54 open source repos** and tried looking into them.
   - No further details were provided about the repos or their specific content.
- **Love expressed for Voyager**: A member shared a link to [Voyager](https://www.youtube.com/watch?v=H0XYANRosVo) exclaiming their love for it, another immediately said "me too!".
   - Another member shared a link to a [Simons Foundation](https://www.youtube.com/playlist?list=PLWAzLum_3a18wO6C7TP8_4XGw4pDxy6G5) playlist that had dropped today.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399983269071818853)** (1 messages): 

> `Qwen3, GPT-4o` 


- **Qwen3 30B Matches GPT-4o Performance**: A member shared a post that [Qwen3 30B A3B 2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) is on par with **OpenAI GPT-4o** in English and Chinese.
- **Qwen3 Gains Traction**: Community members are excited about the potential of **Qwen3** as a strong competitor in the language model arena.
   - Early benchmarks suggest it may offer comparable performance to **GPT-4o** in certain tasks, particularly in multilingual contexts.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399833974972350474)** (30 messages🔥): 

> `Kimi chatbot, Moonshot AI vibe, OpenHands, Training dataset of Kimi, Scale AI` 


- **Kimi Chatbot Replies in Chinese**: A user reported that the **Kimi chatbot** replied in **Chinese** despite asking questions in **English**, suggesting that it may be related to being logged out.
   - Screenshots revealed that while the reply was in English, the suggested sources and questions were in Chinese.
- **Kimi's Training Data Leans on Social Media**: One member joked that **Kimi's training dataset** seems to include **Instagram** and **TikTok** comments, suggesting it's why it's good.
   - They linked to [Kimi on OpenHands v0.50.0k2](https://github.com/All-Hands-AI/OpenHands/releases/tag/0.50.0k2) to support this claim.
- **Moonshot AI Vibe**: One member said that *moonshot got the best vibe no cap*, while another agreed that the community needs some competition.
   - They linked to [X post](https://x.com/crystalsssup/status/1944287779896328668) about the AI community vibe check.
- **Alexandr Wang Founder of Scale AI**: A member mentioned **Alexandr Wang** is the **founder and CEO of Scale AI**, which is a data infrastructure company.
   - They pointed out that **Scale AI** provides training data, annotation, and evaluation services that are essential for developing machine learning models.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399839990069854348)** (25 messages🔥): 

> `Lume vs Suna, Manus' Comic Creation, The Future of Manus` 


- **Members Debate: Lume edges out Suna**: Members debated the merits of the **Lume** and **Suna** agentic systems; one member stated that *Lume did a much better job* to code out specific things and with less mistakes, but conceded they may not have prompted Suna correctly.
   - The member noted they couldn't compare to **Manus** due to prohibitive costs for certain tasks.
- **Manus Comic Creation: A Diamond in the Rough?**: One member suggested that **Manus**' comic creation feature is nice but still can be improved.
   - Another member stated the service was declining in quality, with restrictive limits for free users, and questions whether **Manus is dead**.
- **Optimistic AI vs. Skeptical Human: The Future of Manus**: One member asked an AI what it thinks about the future of **Manus**, and it replied *I think the future of Manus is bright*.
   - Another member expressed skepticism, citing the release of agent modes from **OAI** and **Google**.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399834491513475083)** (22 messages🔥): 

> `MCP Server Security, BDD Testing with LLMs and MCP, Windows-MCP issues with CursorTouch and Claude, FastMCP tool selection, Hosted MCP server` 


- **MCP Server Needs User-Context Separation**: A user is seeking clarification on whether a single cloud-deployed **MCP server instance** requires additional layers for **user-context separation** to prevent data sharing between unique sessions when accessed by multiple clients simultaneously, referencing [issue #1087](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [MCP Docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers).
- **Cursor Connects, Claude Fails**: A user reported successfully deploying an **MCP server to EC2** with properly configured **SSL certification and domain settings**, but they can only connect via **Cursor** and not **Claude Desktop**.
- **Cucumber, BDD, and LLMs Team Up!**: One member shared a side project based on **Behavior-Driven Development (BDD)** that is production-ready; they also included a [diagram of the solution](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688bd568&is=688a83e8&hm=2e86139e9f117265cd7cbef2afcc1a23a34a091e79402df9a0e051261231c695&).
- **Windows-MCP State Tool Tantrums**: One user is struggling with Windows-MCP by CursorTouch in Claude desktop because the **state tool doesn't work at all** and states: *Error calling tool 'State-Tool': 'Taskbar'*.
- **FastMCP Dynamic Tool Selection**: A user inquired whether **FastMCP** includes logic for **dynamically and automatically selecting tools** (e.g., math, web search, RAG, data interpreter) on the client side when multiple tools are defined on the server.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399914906672959628)** (2 messages): 

> `DeepTrail, Deepsecure, Open Source Auth, Delegation Layer for AI agents, Secure Multi-agent workflows` 


- ****DeepTrail** Introduces **Deepsecure** for AI Agent Authorization**: **DeepTrail**, backed by Berkeley SkyDeck, is developing **Deepsecure**, an open-source auth and delegation layer for AI agents, enabling integration of authorization, agent-to-agent delegation, policy enforcement, and secure proxying with minimal code via [GitHub](https://github.com/DeepTrail/deepsecure).
- **Exploring **Deepsecure's** Architecture and Secure Multi-Agent Workflows**: **Deepsecure's** architecture features a split-key design, gateway/proxy, separate control/data plane, policy engine, and macaroons for agent-agent delegation, detailed in its [technical overview](https://github.com/DeepTrail/deepsecure/blob/dev/docs/design/deepsecure-technical-overview.md).
- **Examples of **Deepsecure** Integration with Langchain/LangGraph**: Examples of **Deepsecure** integration with Langchain/LangGraph: *Secure Multi-agent workflows* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/05_langchain_secure_tools.py)), *Delegation Workflow* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/09_langchain_delegation_workflow.py)), *Advanced Delegation Patterns* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/11_advanced_delegation_patterns.py)), and *Platform Agent Bootstrapping* ([code link](https://github.com/DeepTrail/deepsecure/blob/dev/examples/12_platform_expansion_bootstrap.py)).
- **Premium Directory with Community Features and Marketplace**: A member started working on what is intended to be *a premium directory, with community features and evolve into a 1-click-install and even marketplace*, at [protocoldepot.dev](https://protocoldepot.dev/).


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399896190509912094)** (18 messages🔥): 

> `DSPy learnable parameters proposal, Signature implementation using f-strings, DSPy vs GEPA` 


- **DSPy Learnable Parameters Proposal Sparking Interest**: Members discussed a proposal for adding learnable parameters (`dspy.variable` or `dspy.parameter`) to DSPy, and [an issue was created](https://github.com/stanfordnlp/dspy/issues/8593) to gather ideas and use cases.
   - One member described it as a *really shiny proposal*, hoping to allow *templates to be parameters/variables* so the optimizer can spit out optimal prompts, as well as template variable placement.
- **F-Strings Cause Signature implementation Problems**: A member asked for help implementing a signature using an f-string, wanting to verify a code against a description.
   - Another user recommended against this approach and suggested *putting the parameter description within `dspy.InputField()`*.
- **DSPy Faces Off Against GEPA in Prompt Optimization Throwdown**: A member noted a YouTube video where **DSPy** was compared to **GEPA**, with the hot take *DSPy optimizes the prompt you gave it; GEPA evolves a prompt you never imagined* and linked the [YouTube video](https://www.youtube.com/watch?v=o6RbVPFOslg).
   - The same member proposed *turning MIPRO into a reflective, genetic-style frontier engine* for DSPy to spawn and maintain a Pareto-frontier of prompts, aiming to disprove the YouTuber's claim.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400072601677856810)** (15 messages🔥): 

> `AMD vs Nvidia for gaming, Qwen coding model release, RLVR discussion` 


- **AMD: The Way to Game and Blaze New AI Trails**: One member suggested getting the **7900XT** instead of the **9070** for gaming, pairing it with a **7800X3D** rather than a **9900X**, also noting AMD's consumer AI usability and potential long-term community benefits.
   - They linked to a [tweet](https://x.com/Teknium1/status/1950596567968477382) to bolster their argument.
- **Qwen Thinks Aloud with Coding Model Launch**: A member announced the impending release of the **Qwen3-30B-A3B-Thinking-2507** coding model on [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507).
   - The link to this Hugging Face model indicates a new tool for code generation.
- **Nvidia's RLVR: Is it Really an RL Algorithm?**: A member questioned the classification of **RLVR** (Reinforcement Learning, Virtual Reality) as a reinforcement learning algorithm, linking to an [NVIDIA tweet](https://fxtwitter.com/NVIDIAAIDev/status/1950279130450444670) prompting the discussion.
   - Another member, teknium, stated *"RLVR is just not a RL Algo its just a target of rl"*.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400127686264881253)** (3 messages): 

> `MRC model comparison, Summer school channel request, Senior Software Engineer remote job` 


- **Member Enquires about Summer School Channel**: A new member is asking about a dedicated channel for a **summer school** that took place some weeks ago.
- **MRC Model Comparison Tactics**: A member is asking whether to compare a custom **MRC model** against a large pretrained model's **zero-shot performance**, or to **fine-tune** the large model on the same dataset for a fairer comparison.
- **Long-Term Remote Senior Software Engineer Role Advertised**: A senior software engineer role is advertised, paying **$2K/month**, for a long-term remote contract, located in **Africa** or the **Americas**.
   - The role requires experience in **Ruby on Rails**, **Node.js**, **C#/.NET**, **Python**, **Java**, or similar, as well as strong communication skills in native or near-native English.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1400077240896590014)** (1 messages): 

> `Langchain-Cohere citation mode, langchain_cohere.ChatCohere` 


- **Citation options do not work on Langchain-Cohere**: A member was having problems to change the citation mode using `citation_options` on `langchain_cohere.ChatCohere`.
   - The member asked if there is any implicit way to pass the citation options since `langchain_cohere.ChatCohere` does not accept it explicitly.
- **Langchain-Cohere repo status: unmaintained?**: A member asked if the [langchain-cohere repo](https://github.com/langchain-ai/langchain-cohere) is an official repo.
   - They noted that the repository have not been updated in the past few months and also inquired *if pull requests are welcomed there*.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400056898665058364)** (6 messages): 

> `AI Safety, LLM Bias Mitigation, GPU Kernel Optimization` 


- **Stats Student Seeks Safe Spaces in AI**: A statistics master's student expressed interest in **ML research**, particularly in **technical AI safety**, and is open to research collaborations.
- **PhD Focuses on Ethical LLMs**: A PhD student at JKU Linz Austria is working on **mitigating social bias in LLMs**.
   - Their other interests include **attribution for generative models, AI generated text detection, and domain adaptation**, and they're interested in connecting with people working on practical ethical concerns with domain-specific LLMs.
- **RAGs and Graphs Grab Grad's Growth**: A recent Master's graduate from the Technical University of Munich is working on personal projects to gain more experience with **RAGs**, **knowledge graphs**, and new programming languages.
   - They hope to gain research experience, collaborate on projects, and meet like-minded people to stay on top of new tech.
- **Ali Aces Autoregressive Acceleration**: A member named Ali is working on **optimizing GPU kernels in Triton/CUDA for autoregressive models**.
   - They are always happy to chat about low-level GPU programming.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (2 messages): 

> `LoRA-style adapter in Torchtune, Merged weights in Torchtune` 


- **LoRA-style adapter requested for Torchtune**: A user inquired about **LoRA-style adapter support** in Torchtune, specifically one that retains the exact forward compute path without altering computational cost but freezes original model weights and applies updates through additional trainable layers.
   - They are looking for **additional trainable layers**.
- **Torchtune merges weights after training with an adapter**: A user shared a [link](https://docs.pytorch.org/torchtune/0.6/tutorials/e2e_flow.html) to the Torchtune documentation on **end-to-end workflow**, highlighting that Torchtune supports training with an adapter but merges the weights back in.
   - They are asking questions about merged weights.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1400146555914293331)** (2 messages): 

> `ACL Paper Award, Glianorex finetunes` 


- **ACL Paper wins Award**: A member shared their **ACL paper** that just won an award, linked [here](https://aclanthology.org/2025.acl-long.266/).
- **Glianorex Finetunes released**: A member asked if the **finetunes** are public, complaining that their *Glianorex is killing me and my doctor has been no help*.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400165808369438871)** (2 messages): 

> `Certificate Declaration Form` 


- **Certificate Declaration Form needs completing**: A member was reminded that the certificate declaration form had not been completed.
   - The staff confirmed that *we never got a certificate declaration form from you unfortunately*.
- **Certificate Still Required**: The staff reiterated that the certificate declaration form hadn't been received.
   - The member was previously informed that their form was missing.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 messages): 

> `Diffusion Models Study Group, MIT Diffusion Models Curriculum, Flow Matching, Generative AI, AI Education` 


- **Learn Diffusion Models from Scratch in New Study Group**: A new study group is starting a **12-person**, **5-month** program (**2-4 hrs/week**) based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) to learn diffusion models, which are now a core architecture in generative AI.
   - The first two introductory sessions are free and open to non-members: August 2nd on *Flow Matching & Diffusion Models* and August 9th on *PDEs, ODEs, SDEs + A Brief History of Diffusion Models* ([links here](https://lu.ma/kv8zf6va)).
- **AI Scholars Announces New Diffusion Models Study Group**: AI Scholars is launching a study group on Diffusion Models, with confirmed members including the CTO of an AI film tool, AI art instructor, 2 LLM instructors, and 2 full-time AI researchers.
   - The program features peer-led sessions, mentor Q&A, hands-on projects, real research papers, and a tight, trusted cohort with a weekly format of 2 hours live class + 2 hours self-study, with students rotating teaching.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399993952987512853)** (1 messages): 

> `Deploying custom language models, Hugging Face deployment, GUI for user queries` 


- **Cloud Deployment Strategy Sought**: A user inquired about how to deploy a language model, trained with a custom folder of PDFs, to the cloud for public use, specifically seeking a simple GUI for user queries.
   - Nomic suggested that the enterprise plan wasn't a good fit, and the user wondered about **Hugging Face deployment** as an alternative.
- **Enterprise Plan Not Suited**: Nomic indicated that the enterprise plan isn't a good fit for the user's needs.
   - The user is exploring alternative deployment strategies, such as Hugging Face, to make their language model accessible.


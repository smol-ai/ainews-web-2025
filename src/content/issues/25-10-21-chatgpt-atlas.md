---
id: MjAyNS0x
title: 'ChatGPT Atlas: OpenAI''s AI Browser'
date: '2025-10-21T05:44:39.731046Z'
description: >-
  **OpenAI** launched the **Chromium fork AI browser Atlas** for macOS,
  featuring integrated **Agent mode** and browser memory with local login
  capabilities, aiming to surpass **Google's Gemini** in Chrome. The launch
  received mixed reactions regarding reliability and privacy. **LangChain**
  raised a **$125M Series B** at a $1.25B valuation, releasing **v1.0 agent
  engineering stack** with significant adoption including **85M+ OSS
  downloads/month** and usage by ~35% of the Fortune 500. The ecosystem also saw
  updates like **vLLM's MoE LoRA expert finetuning support**.
companies:
  - openai
  - google
  - langchain
  - ivp
  - capitalg
  - sapphire
  - sequoia
  - benchmark
models:
  - gemini
  - atlas
topics:
  - agent-mode
  - browser-memory
  - chromium
  - finetuning
  - moe
  - lora
  - agent-runtime
  - observability
  - software-development
  - funding
people:
  - kevinweil
  - bengoodger
  - fidjissimo
  - omarsar0
  - yuchenj_uw
  - nickaturley
  - raizamrtn
  - hwchase17
  - bromann
  - casper_hansen_
  - corbtt
---


**Chromium is all you need.**

> AI News for 10/20/2025-10/21/2025. We checked 12 subreddits, 544 Twitters and 23 Discords (198 channels, and 7709 messages) for you. Estimated reading time saved (at 200wpm): 564 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

As [leaked in July](https://www.reuters.com/business/media-telecom/openai-release-web-browser-challenge-google-chrome-2025-07-09/?utm_source=chatgpt.com) (and earlier), OpenAI finally launched their Chromium fork AI browser, Atlas (MacOS only for now but other platforms coming - [download/website here](https://chatgpt.com/atlas)):

![ChatGPT Atlas browser homepage with blue background and browser interface showcasing a flight booking details screen](https://resend-attachments.s3.amazonaws.com/TWkFeQSXfOp82Qh)

The integration is very polished and impressive, as you can see in the second half of the [livestream.](https://youtu.be/8UWKxJbjriY) By bringing Agent mode into Atlas, OpenAI is not just matching what [Gemini in Chrome already has](https://www.google.com/chrome/ai-innovations/), but going the next obvious step past it by reviving [Operator](https://news.smol.ai/issues/25-01-23-ainews-openai-launches-operator-its-first-agent) and putting it in the local browser instead of remote, so that it can use your logins.

The vibes are positive, but not entirely so:

![A tweet showing a user retracting a previous statement about the polish of ChatGPT Atlas](https://resend-attachments.s3.amazonaws.com/AlsVRpYiGbyCh6t)

Your move, Google.

---

# AI Twitter Recap

**OpenAI’s ChatGPT Atlas Browser Launch**

- **Atlas ships with Agent Mode and “browser memory”**: OpenAI unveiled an AI-first browser for macOS with ChatGPT embedded system-wide, optional page/context memory, and a preview “Agent mode” that can act on webpages (including logged-in sites with permission). macOS is rolling out now; Windows, iOS, and Android “coming soon.” See launch posts from [@OpenAI](https://twitter.com/OpenAI/status/1980685602384441368), [Agent mode details](https://twitter.com/OpenAI/status/1980685612538822814), and [product notes](https://twitter.com/OpenAI/status/1980685615340614032). PMs highlighted use-cases and UX intent via [@kevinweil](https://twitter.com/kevinweil/status/1980698941885935707), [@bengoodger](https://twitter.com/bengoodger/status/1980692301010858350), and [@fidjissimo](https://twitter.com/fidjissimo/status/1980682244185608392). An incognito-style toggle for memory is present ([@omarsar0](https://twitter.com/omarsar0/status/1980688230904144086)).
- **Early reactions**: The “browser is the new OS” framing landed ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980685683707842974), [@nickaturley](https://twitter.com/nickaturley/status/1980694337643315475)), but reliability and privacy trade-offs surfaced immediately. One head-to-head against Perplexity’s Comet showed Atlas completing a tedious grades-tracking task more robustly (context handling, faster actions, and “human-like” exploration) ([@raizamrtn](https://twitter.com/raizamrtn/status/1980695747227210213)). Others called Agent mode “slop” for now and raised data access concerns ([@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1980846874904219932), [privacy](https://twitter.com/Yuchenj_UW/status/1980847565819302116)). Launch traffic briefly overwhelmed services ([@TheTuringPost](https://twitter.com/TheTuringPost/status/1980700012372955352)).

**LangChain’s $125M Series B and v1.0 Agent Engineering Stack**

- **Funding + product milestone**: LangChain raised a $125M Series B led by IVP with participation from CapitalG, Sapphire, Sequoia, Benchmark, and others, valuing the company at $1.25B. Alongside, it released 1.0 versions of LangChain and LangGraph, a LangSmith insights agent, and a no-code agent builder ([@LangChainAI](https://twitter.com/LangChainAI/status/1980678921839603948), [@hwchase17](https://twitter.com/hwchase17/status/1980680421706006663), [IVP note](https://twitter.com/tomloverro/status/1980714285140701362)). The team emphasized a controlled, production-first agent runtime and observability, with a new createAgent abstraction + middleware in LangChainJS ([@bromann](https://twitter.com/bromann/status/1980683275682091024), [release notes](https://twitter.com/chester_curme/status/1980685592544571897)). Usage claims: 85M+ OSS downloads/month and ~35% of the Fortune 500 using the stack ([@veryboldbagel](https://twitter.com/veryboldbagel/status/1980686379613815295), [@amadaecheverria](https://twitter.com/amadaecheverria/status/1980687050174287876)).
- **Ecosystem fit**: vLLM added MoE LoRA expert finetuning support ([@casper_hansen_](https://twitter.com/casper_hansen_/status/1980525929026973904)) and credited an external analysis as impetus ([@corbtt](https://twitter.com/corbtt/status/1980678250608443467)). Multiple teams highlighted production usage of LangGraph/LangSmith for agent reliability and evals ([@Hacubu](https://twitter.com/Hacubu/status/1980683912096674144), [@jhhayashi](https://twitter.com/jhhayashi/status/1980690375326278107)).

**Vision Tokens, OCR, and New VLMs: DeepSeek-OCR, Glyph, Qwen3-VL, Chandra OCR**

- **DeepSeek-OCR (text-as-image) sparks debate**: The paper reports large long-context compression by rendering text as images and decoding via a vision encoder + MoE decoder. Commentary ranges from enthusiastic technical breakdowns (97% reconstruction precision with ~10x fewer “visual” tokens; high-res convolutional compressor) ([@rasbt](https://twitter.com/rasbt/status/1980642191950090585)) to sharp critiques on missed prior art (pixels-for-language and visual token compression lines) ([@awinyimgprocess](https://twitter.com/awinyimgprocess/status/1980506449706119642), [@NielsRogge](https://twitter.com/NielsRogge/status/1980559120760791125)). Others argue the core takeaway is inefficiency in current embedding/token usage, not image superiority per se ([@Kangwook_Lee](https://twitter.com/Kangwook_Lee/status/1980709454522744902)).
- **Zhipu’s “Glyph”-like direction and KV via vision tokens**: Several noted Zhipu releasing a contemporaneous vision-token compression approach (“Glyph”), with claims of 3–4x context compression and infilling cost reductions without quality drop on long-context QA/sum ([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1980722682246398069), [context](https://twitter.com/teortaxesTex/status/1980642000006451348)). Details remain sparse; watch for BLT-like extensions to push decoding efficiency further.
- **Qwen3-VL-2B/32B**: Alibaba released dense 2B and 32B VLMs, including FP8 variants and “Thinking”/Instruct types, claiming strong wins vs GPT‑5 mini and Claude Sonnet 4 across STEM, VQA, OCR, video, agent tasks; the 32B aims to match much larger models on OSWorld with high memory efficiency ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1980665932625383868)). Demos landed on HF quickly ([@_akhaliq](https://twitter.com/_akhaliq/status/1980690335220351063)).
- **Open-source OCR**: Chandra OCR launched with full layout extraction, image/diagram captions, handwriting, and table support; works with Transformers/vLLM ([@VikParuchuri](https://twitter.com/VikParuchuri/status/1980667137606971423)).

**Training/Serving Stack Updates: PyTorch, vLLM, FlashInfer, Providers**

- **Meta PyTorch drops new libraries**: torchforge (scalable RL training), OpenEnv (agentic environments), and torchcomms, plus momentum around Monarch and TorchTitan within a “future-of-training” map (pretrain→post-train→inference) ([@eliebakouch](https://twitter.com/eliebakouch/status/1980637130687942805), [stack summary](https://twitter.com/eliebakouch/status/1980642834404319388), [Monarch](https://twitter.com/finbarrtimbers/status/1980681034359533861)).
- **vLLM and memory**: kvcached enables serving multiple models sharing unused KV cache blocks on the same GPU ([@vllm_project](https://twitter.com/vllm_project/status/1980776841129701411)); the project is featured at PyTorch Conference ([@vllm_project](https://twitter.com/vllm_project/status/1980622348903674022)).
- **FlashInfer-Bench**: new “self-improving” benchmarking workflow to standardize LLM serving kernel signatures and auto-surface fastest kernels for day-0 integration in FlashInfer/SGLang/vLLM ([@shanli_xing](https://twitter.com/shanli_xing/status/1980705452699926851)).
- **Provider benchmarks for GLM‑4.6 (Reasoning)**: Baseten led output speed (104 tok/s) and fastest time-to-first-answer-token; pricing across providers clustered near $0.6/M input, ~$2/M output; all support 200k context and tool calling ([@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1980777360724226282)).

**Research, Evals, and Methods**

- **Continual learning via memory layers**: Sparsely finetuned memory layers enable targeted updates with minimal forgetting compared to full finetune/LoRA (−11% vs −89%/−71% on fact tasks), proposing a practical route to incremental model updates ([@realJessyLin](https://twitter.com/realJessyLin/status/1980662516285075762), [blog](https://twitter.com/realJessyLin/status/1980697898141774017)).
- **Mechanistic interp at scale**: Anthropic analyzed Claude 3.5 Haiku on a “perceptual” task, revealing clean geometric transformations and distributed attention algorithms; community notes it as among the deepest behaviors understood mechanistically to date ([@wesg52](https://twitter.com/wesg52/status/1980680563582538099), [@NeelNanda5](https://twitter.com/NeelNanda5/status/1980770185167663140)).
- **Prompt optimization > RL for compound systems?** GEPA uses reflective prompt evolution with Pareto selection to beat GRPO on HotpotQA, IFBench, Hover, PUPA, reducing rollout needs via natural language self-critique ([@gneubig](https://twitter.com/gneubig/status/1980644772902789603), [paper/code](https://twitter.com/gneubig/status/1980646347188707787), [summary](https://twitter.com/joelniklaus/status/1980651047720001884)).
- **Evals in the wild**: SWE‑Bench Pro leaderboard update shows top models now >40% pass rate, with Claude 4.5 Sonnet leading ([@scale_AI](https://twitter.com/scale_AI/status/1980685992987431368)).
- **Self-play caveats for LLMs**: Why self-play shines in two‑player zero‑sum settings (minimax) but is tricky in real‑world domains (reward shaping, equilibria untethered from human utility) ([@polynoamial](https://twitter.com/polynoamial/status/1980697004658556972)).

**Developer Tooling and Apps**

- **Google AI Studio “AI-first coding”**: revamped build mode integrates multi-capability scaffolding (“I’m Feeling Lucky”), targeting faster prompt→production iteration for Gemini apps ([@OfficialLoganK](https://twitter.com/OfficialLoganK/status/1980674135693971550), [demo](https://twitter.com/patloeber/status/1980676182904565999), [@GoogleAIStudio](https://twitter.com/GoogleAIStudio/status/1980679588704371095)).
- **Runway**: announced self-serve model fine-tuning and a node-based Workflows system to chain models/modalities/intermediate steps for production creative pipelines ([@runwayml](https://twitter.com/runwayml/status/1980620538906054691), [Workflows](https://twitter.com/runwayml/status/1980736639405289786)).
- **Together AI**: video and image generation models (e.g., Sora 2, Veo 3) now accessible through the same APIs used for text inference ([@togethercompute](https://twitter.com/togethercompute/status/1980746093932515697)).
- **LlamaIndex**: llamactl CLI for local LlamaAgents development/deployments; turnkey document agents template and private-preview hosting for doc-centric workflows ([@llama_index](https://twitter.com/llama_index/status/1980673952033976824), [@jerryjliu0](https://twitter.com/jerryjliu0/status/1980759684916408443)).

**Top tweets (by engagement)**

- “hahahaha the bed sends 16gb of data a month oh god” — IoT reliability/telemetry facepalm during the AWS outage ([@internetofshit](https://twitter.com/internetofshit/status/1980506231233184144)).
- “Meet our new browser—ChatGPT Atlas. Available today on macOS” ([@OpenAI](https://twitter.com/OpenAI/status/1980685602384441368)); “Make room in your dock” ([@OpenAI](https://twitter.com/OpenAI/status/1980678350407606518)).
- Karpathy on synthetic identity/personality tuning for nanochat via diverse synthetic dialogs ([@karpathy](https://twitter.com/karpathy/status/1980665134415802554)).
- Qwen Deep Research upgrade: report + live webpage + podcast auto-generation with Qwen3 stack ([@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1980609551486624237)).
- Airbnb CEO: Qwen is “very good, fast and cheap,” often preferred in production over “latest” OpenAI models due to cost/latency ([@natolambert](https://twitter.com/natolambert/status/1980657338726887662)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-VL Model Performance Comparison

- [**Qwen3-VL-2B and Qwen3-VL-32B Released**](https://www.reddit.com/r/LocalLLaMA/comments/1och7m9/qwen3vl2b_and_qwen3vl32b_released/) (Activity: 626): **The image provides a detailed comparison of the performance metrics for the newly released Qwen3-VL-2B and Qwen3-VL-32B models against other models like Qwen3-VL-4B, Qwen3-VL-8B, and Qwen2.5-VL-7B. The table highlights the models' performance across various tasks such as STEM & Puzzle, General VQA, and Text Recognition. Notably, the Qwen3-VL-32B model demonstrates superior performance, achieving higher scores in most categories, which are marked in red to indicate their significance. This suggests that the Qwen3-VL-32B model is particularly effective in these tasks, outperforming its predecessors and other variants.** One comment humorously suggests that the release of the 32B model should satisfy those requesting it, indicating anticipation and demand for this model size.
    - The release of Qwen3-VL-2B and Qwen3-VL-32B models marks a significant advancement, with the new models reportedly outperforming the previous 2.5-VL 72B model despite being less than half its size. This suggests substantial improvements in model efficiency and performance, likely due to architectural optimizations or enhanced training techniques.
    - A comparison image provided by a user highlights the performance differences between Qwen3-VL-2B and Qwen3-32B, indicating that the newer models may offer superior capabilities in text processing tasks. This could be of particular interest to those evaluating model performance for specific applications.
    - Benchmarks shared in the discussion suggest that the Qwen3-VL models excel in 'thinking' tasks, which may refer to complex reasoning or problem-solving capabilities. This positions the models as strong candidates for applications requiring advanced cognitive processing.
- [**DeepSeek-OCR AI can scan an entire microfiche sheet and not just cells and retain 100% of the data in seconds...**](https://www.reddit.com/r/LocalLLaMA/comments/1ocgun0/deepseekocr_ai_can_scan_an_entire_microfiche/) (Activity: 405): **DeepSeek-OCR AI claims to scan entire microfiche sheets, not just individual cells, and retain** `100%` **of the data in seconds, as per [Brian Roemmele's post](https://x.com/BrianRoemmele/status/1980634806145957992). The tool reportedly offers a comprehensive understanding of text and complex drawings, potentially revolutionizing offline data curation. However, the post lacks detailed technical validation or benchmarks to substantiate these claims.** Commenters express skepticism about the verification of the extracted data's accuracy and the openness of AI development between countries, particularly comparing the US and China. There is also criticism of the announcement's lack of technical detail, labeling it as 'hype BS' without verification.
    - rseymour raises a technical concern about the resolution capabilities of the DeepSeek-OCR AI, questioning the feasibility of using 'vision tokens' at a resolution of `1024x1024`. They suggest that this resolution might be insufficient for accurately capturing the details of a microfiche sheet, which typically requires higher resolution due to its small size and dense information content. The comment implies that the technology might be overhyped without proper validation of its capabilities.
    - Robonglious discusses the openness of AI development between countries, specifically comparing the transparency of AI advancements in China versus the US. They speculate whether companies like **OpenAI** or **Anthropic** would release similar OCR technology if they developed it, suggesting that the US might be less cooperative in sharing such advancements compared to China.
    - TheHeretic and Big_Firefighter_6081 express skepticism about the claims made regarding DeepSeek-OCR AI's capabilities. They criticize the lack of verification and validation of the results, implying that the information might be more hype than reality. This highlights the importance of rigorous testing and validation in AI technology claims to ensure credibility.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. ChatGPT Atlas Browser Launch

- [**Meet our new browser—ChatGPT Atlas.**](https://www.reddit.com/r/OpenAI/comments/1ocj2da/meet_our_new_browserchatgpt_atlas/) (Activity: 3175): **ChatGPT Atlas is a new browser launched by OpenAI, currently available exclusively on** `macOS`**. The browser integrates AI capabilities directly into the browsing experience, potentially enhancing user interaction with web content. However, the release is limited to Mac users, which has sparked some debate about accessibility and platform support.** Commenters have raised concerns about data privacy and the decision to release the browser only for macOS, questioning the strategic choice and potential data handling practices.
    - Big-Info and douggieball1312 discuss the platform exclusivity of ChatGPT Atlas, noting that it is currently only available for Mac. This decision is critiqued as potentially alienating Windows users, especially given Microsoft's financial backing of OpenAI. The irony is highlighted in the context of Microsoft's investment, as Windows is a major competitor to Mac.
    - Tueto raises concerns about data privacy with ChatGPT Atlas, questioning where user data is being sent. This reflects broader concerns about data handling and privacy in AI-driven applications, especially in the context of web browsing where sensitive information is often accessed.
    - douggieball1312 points out the irony in ChatGPT Atlas being exclusive to Mac, despite OpenAI's backing by Microsoft. This decision is seen as a reflection of a Silicon Valley tech bubble that may overlook the broader user base, particularly Windows users, which could impact adoption and user satisfaction.
- [**GPT browser incoming**](https://www.reddit.com/r/OpenAI/comments/1ocfrxy/gpt_browser_incoming/) (Activity: 1511): **The image is a social media post by Sam Altman, CEO of OpenAI, announcing a livestream event to launch a new product. The post is retweeted by OpenAI and features a graphic with the word "Livestream" and the OpenAI logo, indicating a significant announcement. The community speculates about the nature of the product, with some comments humorously suggesting a 'sexbot' or expressing concerns about privacy, likening it to 'spyware' similar to Google's practices. The engagement on the post suggests high interest and anticipation for the announcement.** The comments reflect a mix of humor and skepticism, with some users joking about the product being a 'sexbot' and others expressing concerns about privacy, comparing it to Google's data practices.
    - trustmebro24 speculates that the upcoming GPT browser might be based on Chromium, which is a common choice for many modern browsers due to its open-source nature and robust performance. Chromium's architecture allows for extensive customization and integration of advanced features, which could be beneficial for a browser leveraging GPT technology.
    - qodeninja raises a concern about the potential for the company to overextend itself by developing too many products, suggesting that it might be more effective to allow the broader ecosystem to innovate and create complementary technologies. This reflects a strategic consideration about resource allocation and focus in tech development.
    - Vegetable_Fox9134 mentions the potential privacy concerns associated with a new browser, comparing it to existing issues with Google. This highlights the ongoing debate about data privacy and the trade-offs users face when using technology that may collect personal information.
- [**OpenAI’s AI-powered browser, ChatGPT Atlas, is here**](https://www.reddit.com/r/ChatGPT/comments/1ociphv/openais_aipowered_browser_chatgpt_atlas_is_here/) (Activity: 1041): **OpenAI has launched an AI-powered browser named ChatGPT Atlas, which integrates the capabilities of ChatGPT into web browsing. This tool aims to enhance user interaction by providing AI-driven insights and assistance directly within the browser environment. The integration is expected to streamline tasks by leveraging the conversational abilities of ChatGPT, potentially transforming how users interact with web content.** The comments reflect a mix of skepticism and curiosity, with some users expressing concerns about privacy and the potential for misuse, while others are intrigued by the possibilities of AI-enhanced browsing.
- [**CONFIRMED: OpenAI is Launching a New Browser TODAY Called ChatGPT Atlas**](https://www.reddit.com/r/ChatGPT/comments/1ochtsq/confirmed_openai_is_launching_a_new_browser_today/) (Activity: 747): **OpenAI has launched a new browser called ChatGPT Atlas, available globally on** `macOS` **with plans for** `Windows`**,** `iOS`**, and** `Android` **soon. The browser integrates AI capabilities directly into the browsing experience, offering a chat interface for seamless AI communication. It is introduced by key figures like Sam Altman and Ben Goodger. The browser is perceived as a strategic move to compete with Google and Microsoft, though it has been critiqued for its similarity to existing browsers with added chat functionality. More details can be found in the [YouTube video](https://www.youtube.com/watch?v=8UWKxJbjriY?1).** Commenters express skepticism about the browser's impact, noting it may primarily serve OpenAI's data collection needs rather than offering significant user benefits. Concerns are raised about privacy and data being sent to OpenAI, with some questioning the necessity of the browser given existing alternatives.
    - The introduction of ChatGPT Atlas by OpenAI is seen as a strategic move to compete with tech giants like Google and Microsoft. While the browser may offer some convenience and speed improvements, there is skepticism about its impact on users. The primary concern is the extensive data collection capabilities, which could surpass current systems by learning about users' lives, interests, and behaviors in real-time.
    - There is speculation that ChatGPT Atlas might be based on the Chrome engine, which would align with many modern browsers that leverage Chromium for compatibility and performance benefits. This choice could influence the browser's adoption by providing a familiar user experience and support for existing web standards.
    - A significant concern among users is the potential privacy implications of using ChatGPT Atlas. The browser could collect vast amounts of personal data, raising issues about how OpenAI will handle and protect this information. This concern is particularly relevant for users who may not fully understand the extent of data sharing involved.

### 2. Claude Desktop General Availability

- [**Claude Desktop is now generally available.**](https://www.reddit.com/r/ClaudeAI/comments/1ock3em/claude_desktop_is_now_generally_available/) (Activity: 836): **Claude Desktop is now generally available for both Mac and Windows, offering seamless integration with local work environments. Users can access Claude by double-tapping the Option key on Mac, capture screenshots, share windows, and use voice commands via Caps Lock. The application supports enterprise deployment with** `MSIX` **and** `PKG` **installers. For more details and to download, visit [Claude's official site](https://claude.com/download).** Some users were confused about the announcement, thinking the app was already available, while others noted the absence of a Linux version. The Quick Entry feature is praised for its functionality.
    - ExtremeOccident mentions that despite Claude Desktop being in beta, the Quick Entry feature is effective, indicating a focus on user experience and efficiency in input handling.
    - Logichris highlights a limitation in token allocation for Claude Desktop, comparing it to a 'paycheck to paycheck' scenario, which suggests that the current token system may not support extensive use without frequent replenishment.
    - Multiple users, including Yeuph and JAW100123, point out the lack of a Linux version, indicating a gap in platform support that could limit adoption among Linux users.
- [**{Giveaway} 1 Year of Gemini AI PRO (40 winners)**](https://www.reddit.com/r/GeminiAI/comments/1ocovu8/giveaway_1_year_of_gemini_ai_pro_40_winners/) (Activity: 2833): **The post announces a giveaway for a one-year subscription to Gemini AI PRO for 40 winners, highlighting features such as the upcoming *Gemini 3.0 Ultra*,** `1,000 monthly AI credits`**, and tools like *Gemini Code Assist*, *NotebookLM*, and integration with *Gmail, Docs, and Vids*. The package also includes** `2TB storage` **and extended limits on various applications, aiming to enhance productivity and creativity across different domains.** Commenters highlight diverse uses of Gemini AI, such as aiding in storytelling and language translation for personal and professional purposes, supporting filmmaking through its ecosystem, and enhancing open-source contributions with code generation capabilities.
    - Bioshnev highlights the practical applications of Gemini AI in both personal and professional settings. He uses it for generating custom bedtime stories for his daughter and for work-related tasks like translating for foreign customers and retrieving product details. This showcases the model's versatility in handling language processing and information retrieval tasks.
    - thenakedmesmer discusses the impact of Gemini AI on creative projects, particularly in filmmaking. He mentions using features like 'nano banana' and 'veo' as part of a supportive ecosystem that aids in film production, illustrating how AI can serve as a virtual creative team, compensating for physical limitations and enhancing creative workflows.
    - vladlearns emphasizes the importance of code generation capabilities in Gemini AI for open source contributions. This points to the model's utility in software development, where it can assist in automating coding tasks, potentially increasing productivity and supporting collaborative projects.

### 3. Amazon's Robot Workforce Plans

- [**Amazon hopes to replace 600,000 US workers with robots, according to leaked documents﻿. Job losses could shave 30 cents off each item purchased by 2027.**](https://www.reddit.com/r/singularity/comments/1occruc/amazon_hopes_to_replace_600000_us_workers_with/) (Activity: 1630): **Amazon is reportedly planning to replace** `600,000` **US workers with robots by** `2027`**, as per leaked documents. This automation could potentially reduce costs by** `30 cents` **per item. The initiative is part of a broader strategy to address labor shortages and improve efficiency in fulfillment centers, a goal Amazon has pursued since acquiring Kiva Systems over a decade ago. The transition to robotics is seen as a necessary step due to high turnover rates and labor shortages in Amazon's fulfillment centers.** Commenters highlight that the cost savings may not translate to lower prices for consumers, and emphasize the strategic necessity of automation due to Amazon's labor challenges. A former Amazon Robotics employee notes that the goal of replacing workers with robots has been longstanding but is progressing slower than anticipated.
    - The comment by 'theungod' highlights a critical operational challenge for Amazon: the high turnover and difficulty in staffing their fulfillment centers (FCs). The user notes that Amazon has been aiming to automate these roles since acquiring Kiva Systems over a decade ago, but the transition to robotics has been slower than anticipated. This suggests that the integration of robotics into Amazon's logistics is not just about cost savings but also about addressing labor shortages.
    - 'theungod' also provides an insider perspective, having worked at Amazon Robotics for over five years. They emphasize that the goal of replacing 600,000 workers with robots has been a long-standing objective, indicating that the technological and logistical hurdles are significant. This insight underscores the complexity of implementing large-scale automation in fulfillment operations, which involves not just technological development but also overcoming practical deployment challenges.
    - The discussion touches on the broader implications of automation in logistics, particularly the potential societal impact. While the cost savings per item (30 cents) are noted, the focus is on the necessity of automation due to labor shortages rather than purely financial incentives. This reflects a shift in the narrative from cost-cutting to operational necessity, driven by the inability to maintain a stable workforce in demanding environments.
- [**Shape shifting drone**](https://www.reddit.com/r/singularity/comments/1oc5v07/shape_shifting_drone/) (Activity: 1226): **The post discusses a shape-shifting drone that appears to have a unique design, possibly inspired by biological forms, as suggested by the comment likening it to a 'floating colonoscopy'. The image linked in the comments shows a drone with a flexible structure, which may allow it to adapt its shape for different flight dynamics or environmental conditions. This could be an innovative approach in drone technology, potentially enhancing maneuverability and efficiency.** One comment suggests that the concept of a shape-shifting drone is not entirely new, indicating that similar designs may have been seen before. This could imply ongoing research and development in this area, reflecting a trend towards more adaptable and versatile UAV designs.

---

# AI Discord Recap

> A summary of Summaries of Summaries by gpt-5
> 

**1. GPU and eGPU Hardware Breakthroughs**

- **Blackwell Pro Packs 72GB, Quietly Drops**: TechPowerUp reported NVIDIA quietly launching the workstation-class **RTX Pro 5000 Blackwell** with **72 GB GDDR7** memory, targeting pro workflows ([NVIDIA RTX Pro 5000 Blackwell GPU with 72 GB GDDR7 appears](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears)).
    - Engineers joked about likely pricing and use cases, while others flagged initial confusion over the unusual **72 GB** capacity, mirroring similar coverage on [VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory).
- **Tinygrad Makes Apple Silicon Love NVIDIA eGPUs**: The tinygrad team announced early public testing of a pure-Python driver enabling **NVIDIA eGPUs** over **USB4** on **Apple Silicon** using the ADT-UT3G dock, `extra/usbgpu/tbgpu` driver, and NVK-based `tinymesa` compiler ([tinygrad enables NVIDIA eGPU on Apple Silicon (X)](https://x.com/__tinygrad__/status/1980082660920918045)).
    - They measured about **≈3 GB/s** PCIe bandwidth with SIP disabled and teased support for **AMD RDNA 2/3/4** and Windows eGPU stacks next.
- **Tiny Corp Boots NVIDIA on ARM MacBooks**: Tiny Corp demonstrated an **NVIDIA GPU** running on an **ARM MacBook** via **USB4** using an external dock, validating eGPU viability beyond Intel-era Macs ([Tiny Corp Successfully Runs An Nvidia GPU on Arm Macbook Through USB4 Using An External GPU Docking Station](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station)).
    - Mac users were upbeat, noting newer Pros with **Thunderbolt 5** may further improve bandwidth headroom for local **LLM** and **VLM** workloads.

**2. Triton/Kernel Tooling and Benchmarks**

- **FlashInfer-Bench Kicks Off Agentic Kernel Races**: CMU Catalyst introduced **FlashInfer-Bench**, a workflow and leaderboard for agent-driven, self-improving **LLM serving kernels** with standardized signatures and integrations with **FlashInfer**, **SGLang**, and **vLLM** ([FlashInfer-Bench blog](https://flashinfer.ai/2025/10/21/flashinfer-bench.html)).
    - They published a live [leaderboard](https://bench.flashinfer.ai/) and [GitHub repo](https://github.com/flashinfer-ai/flashinfer-bench), inviting the community to iterate on kernels and benchmark updates.
- **Triton Talks Stream and Sizzle**: Developers shared full-session videos from the **Triton** conference at Microsoft, covering compiler advances and kernel design ([Triton Conference livestream](https://www.youtube.com/live/s30WoZ7lx3w) and [Triton-openai streams](https://www.youtube.com/@Triton-openai/streams)).
    - A recurring theme was hand-tuned **PTX/assembly** for critical kernels to beat compiler defaults, echoing calls to rethink execution from the ground up.
- **Helion 0.2 Beta Fuzzes Triton to Tears**: **Helion 0.2** entered public beta as a Triton tile abstraction on PyPI, surfacing compiler edge cases during optimization passes ([helion 0.2.0 on PyPI](https://pypi.org/project/helion/0.2.0)).
    - Users reported MLIR failures in `TritonGPUOptimizeThreadLocalityPass`, framing **Helion** as an effective Triton-compiler 'fuzzer' whose autotuner skips bad configs.

**3. OpenRouter SDK and New Reasoning Model**

- **OpenRouter SDK Types 300+ Models**: OpenRouter released a **TypeScript SDK (beta)** with fully typed requests/responses for **300+ models**, built-in OAuth, and support for all API paths ([@openrouter/sdk on npm](https://www.npmjs.com/package/@openrouter/sdk)).
    - SDKs in **Python**, **Java**, and **Go** are coming soon, aiming to simplify multi-model app development and authentication.
- **Andromeda-alpha Cloaks Visual Reasoning**: OpenRouter launched **Andromeda-alpha**, a small **reasoning** model focused on **image/visual understanding**, available for trial ([Andromeda-alpha on OpenRouter](https://openrouter.ai/openrouter/andromeda-alpha)).
    - Since prompts/outputs are logged to improve the provider’s model, moderators warned: avoid personal/confidential data and do not use it for production.
- **Mercury Outduels Qwen in Agent Arena**: In agentic benchmarks, **Inception/Mercury** from provider **Chutes** edged **Qwen** on failure rate, latency, and cost in simple tasks ([Chutes provider page](https://openrouter.ai/provider/chutes)).
    - Members noted newer **DeepSeek v3.1** models aren’t free via Chutes anymore, though a free longcat endpoint remains ([longcat-flash-chat:free](https://openrouter.ai/meituan/longcat-flash-chat:free)).

**4. Open-Source Models and Text-to-Video Releases**

- **Ring & Ling MoEs Land in llama.cpp**: **Ring** and **Ling** **MoE** models from **InclusionAI** now run in **llama.cpp**, spanning **1T**, **103B**, and **16B** parameter scales ([llama.cpp PR #16063](https://github.com/ggml-org/llama.cpp/pull/16063)).
    - Practitioners questioned real-world **reasoning** quality and verbosity control, hoping for a model that doesn’t YAP during chain-of-thought.
- **Krea Realtime Drops 14B Open T2V**: **Krea Realtime** released a **14B** open-source autoregressive text-to-video model distilled from **Wan 2.1**, generating long-form video at ~**11 fps** on a single **NVIDIA B200** ([Krea Realtime announcement (X)](https://x.com/krea_ai/status/1980358158376988747)).
    - Weights ship under **Apache-2.0** on HuggingFace; users asked about **ComfyUI** workflows, **RTX 5090** performance, and fine-tuning options.
- **DeepSeek-OCR Joins the OCR Fray**: **DeepSeek-OCR** arrived on GitHub, expanding the **OCR** toolkit with modern VLM-friendly design and multilingual aims ([DeepSeek-OCR (GitHub)](https://github.com/deepseek-ai/DeepSeek-OCR)).
    - Developers contrasted it with existing OCR stacks and highlighted the importance of contextual understanding for scripts like **kanji**.

**5. AI Apps: ChatGPT Atlas Launch and Funding News**

- **OpenAI Ships Atlas, a Chromium AI Browser**: OpenAI launched the **ChatGPT Atlas** browser for macOS, a **Chromium**based browser with boosted limits and multi-site browsing ([Introducing ChatGPT Atlas](https://openai.com/index/introducing-chatgpt-atlas/) and [chatgpt.com/atlas](https://chatgpt.com/atlas)).
    - Early users flagged missing **vertical tabs** and built-in ad blocking (extensions required), while elsewhere users compared **Atlas** to **Perplexity’s Comet**, praising Comet’s privacy focus and integrated adblocker.
- **AI Browser Buzz Meets Skeptic Snark**: Engineers questioned the utility of new **AI browsers**, sharing skepticism over performance and data practices ([AI browser hype thread (X)](https://x.com/AlexFinnX/status/1980673764947022038)).
    - One member quipped, *'OpenAI knows this too, they are just farming data and throwing shit at the wall,'* capturing wider concerns about hype versus real value.
- **LangChain Grabs $125M to Build Agent Stack**: **LangChain** raised **$125M Series B**, positioning a three-part stack: **LangChain** (agent dev), **LangGraph** (orchestration), and **LangSmith** (observability) ([LangChain raises $125M (X)](https://x.com/sonyatweetybird/status/1980683121399058626)).
    - They touted adoption by **Uber**, **Klarna**, and **LinkedIn**, signaling continued investor confidence in **agent tooling** and production ops.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Outshines Atlas in Privacy**: Users compared **Comet** and **ChatGPT's Browser Atlas**, favoring **Comet's** commitment to **privacy** and integrated **adblocker**.
   - Several noted the similarity between the two, but praised **Comet** for its features catering to user privacy.
- **AI Therapy Sparks Ethical Concerns**: Discord members debated the ethics of using **AI for therapy**, with some emphasizing the importance of human **emotional maturity**.
   - Opinions diverged, with some suggesting *“ChatGPT is a good therapist,”* while others cautioned against over-reliance on AI for mental health support.
- **Perplexity Fanatic Shows Swag**: A user showcased their **Perplexity stickers**, expressing enthusiasm for the brand and requesting a **Comet hoodie** from the [Perplexity Supply store](https://perplexity.supply/).
   - The display of enthusiasm led to lighthearted jokes about resembling a *“cult,”* with others encouraging further purchases.
- **API Users Want ChatGPT5 Access**: A user inquired whether the **Perplexity API** grants access to models like **ChatGPT5** and **Claude**, or is restricted to **Sonar**.
   - The inquiry reflects a desire to utilize the API for potentially more advanced models beyond the currently available **Sonar**.
- **Shareable Discord Threads Reminder**: A message reminded users to ensure their Discord threads are set to `Shareable` to be more accessible.
   - This ensures that links to the thread can be accessed by others, even outside of the specific channel, improving collaboration.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 3 Pro Blows Away GPT-5 High in Web Design**: Members compared **Gemini 3 Pro** and **GPT-5 High**, reporting that Gemini 3 Pro *crushes web design*.
   - The general consensus is that **Gemini 3 Pro** is better for coding, whereas **GPT-5 High** is better for math and other tasks.
- **Sora 2 Downgrade Sparks AI Subscription Debate**: Members expressed frustration over the **Sora 2** downgrade, leading to a broader discussion on the value of AI subscriptions.
   - One member pointed out that their *job performance is about 25-30% better because of AI*, underscoring AI's impact on desk job efficiency.
- **Lithiumflow and Orionmist Speculated as Gemini 3 Checkpoints**: Members speculated about the differences between **Lithiumflow** and **Orionmist**, ultimately concluding that these models are checkpoint versions of **Gemini 3**.
   - The models sometimes erroneously claim training by **OpenAI**, suggesting potential model distillation.
- **Open Source Models Allegedly Stealing Gemini 2.5 Pro**: Discussion arose regarding the ethics of open-source models using stolen data for improvement, with claims that Chinese AI companies *stole the 2.5 pro and made it open source*.
   - Some members agreed with the sentiment that this is okay as that's the only way **open source** can win.
- **TikZ Generation Task Elicits Surprise**: Members are exploring the use of LLMs to generate images in **TikZ**, a typesetting language, to avoid data contamination.
   - Early results indicate some success in generating **TikZ** images with LLMs, demonstrating a novel approach to image creation.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Ring and Ling Launch!**: **Ring and Ling MoE models** are now supported in *llama.cpp* ([link to Github](https://github.com/ggml-org/llama.cpp/pull/16063)), including **1T**, **103B**, and **16B** parameter models from InclusionAI.
   - Members pondered the reasoning abilities of the models, with one hoping for *a reasoning model that doesn’t YAP*.
- **Disable Unsloth Statistics**: To prevent telemetry calls when running **Unsloth** in offline mode, set the `UNSLOTH_DISABLE_STATISTICS` environment variable and `os.environ['HF_HUB_OFFLINE'] = '1'`, as the Unsloth community reached **100M lifetime downloads on Hugging Face** ([announcement on X](https://x.com/UnslothAI/status/1980631523104813419)).
   - Members also discussed resolving network issues by setting proxy environments.
- **Nvidia RTX Pro 5000 Blackwell Workstation Card Quietly Appears**: **Nvidia** quietly launched the **RTX Pro 5000 Blackwell** workstation card with **72GB** of memory, as reported by [VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory).
   - Initial confusion arose regarding the **72GB** capacity, with one user joking that it was a way to bypass automod.
- **User Fumes over Rate Limiting Tactics**: A user ranted about a **premium subscription service** blocking access to URLs containing **roman numerals**, incorrectly interpreting them as malicious activity.
   - The user, frustrated with manual workarounds and security plugins, criticized the service for ignoring requests to allow bulk downloads for pro subscribers.
- **Nvidia GPU Transplanted onto ARM Macbook**: A member shared an article from Tom's Hardware on successfully running an **Nvidia GPU** on an **ARM Macbook** through **USB4** using an external GPU docking station: [Tiny Corp Successfully Runs An Nvidia GPU on Arm Macbook Through USB4 Using An External GPU Docking Station](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station).
   - This was exciting to Mac users since *they have **Thunderbolt 5** too now on the ‘pros’* which gives slightly more hope to **Mac** users.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Codex Called a Billion Dollars Compared to Claude's Toonie**: Users debated the merits of **Codex vs Claude** for code generation, with one user stating that comparing them is like comparing *"a billion dollars or a toonie"*.
   - No further details were provided.
- **Cursor Site Crashes, Subs Lost**: Multiple users reported the **Cursor website being down** for several hours, preventing them from logging in, upgrading plans, or renewing subscriptions.
   - Some suspected **AWS issues** as the root cause, while others pointed out the **lack of subscription expiration notifications** as a major inconvenience.
- **Dashboard Cracks Cursor Costs Post-Pricing Changes**: A user shared a dashboard they created to **track actual Cursor costs** after the pricing changes, especially for users on legacy pricing plans and gave this forum link [cursor.com/blog/aug-2025-pricing](https://cursor.com/blog/aug-2025-pricing).
   - The tool requires cookie login or .json upload from the user's local machine, but promises comparison with real API pricing.
- **Background Agents Encounter Internal Error**: A member reported encountering an *internal error* when running a first experiment with **background agents** via **Linear**, where the agent starts, does some thinking and grepping, but then stops.
   - The error message received was: *"We encountered an internal error that could not be recovered from. You might want to give it another shot in a moment."*



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter SDK: Beta Boost**: The new **OpenRouter SDK** is in **beta** on [npm](https://www.npmjs.com/package/@openrouter/sdk?activeTab=versions), aiming to be the simplest way to use **OpenRouter** and offering fully typed requests and responses for **300+ models**.
   - Python, Java, and Go versions are coming soon, featuring built-in OAuth and support for all API paths.
- **Andromeda-alpha: Cloaked & Ready**: **OpenRouter** launched a new stealth model named **Andromeda-alpha** ([https://openrouter.ai/openrouter/andromeda-alpha](https://openrouter.ai/openrouter/andromeda-alpha)), a smaller reasoning model focused on image and visual understanding.
   - Prompts/outputs are logged to improve the model, users are cautioned against uploading personal/confidential info and not using it for production.
- **Objective AI's Confidence Code**: [Objective AI](https://objective-ai.io/) now offers a **Confidence Score** for each **OpenAI** completion choice, derived through a smarter method than directly asking the AI and emphasizing **cost-efficiency**.
   - The CEO is building **reliable AI Agents, Workflows, and Automations** free of charge using **n8n** integration to gather more examples.
- **Mercury Swats Qwen in Agent Arena**: **Inception/Mercury** ([Chutes](https://openrouter.ai/provider/chutes) provider) edges out **Qwen** in simple agentic tasks, exhibiting lower failure rate, faster speed, and lower cost.
   - New **Deepseek** models like **v3.1** aren't available as free versions through **Chutes**, though they recently added a free longcat endpoint.
- **AI Browser Bandwagon Bewilders Brains**: Members are skeptical towards the hype around new AI browsers like [X's AI browser](https://x.com/AlexFinnX/status/1980673764947022038), questioning the utility and performance impact of integrated AI.
   - One member compared the hype to the dotcom bubble, stating that *OpenAI knows this too, they are just farming data and throwing shit at the wall*.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Launches Atlas Browser**: OpenAI released the **ChatGPT Atlas Browser** for macOS today at [chatgpt.com/atlas](https://chatgpt.com/atlas), detailed in their [blog post](https://openai.com/index/introducing-chatgpt-atlas/).
   - The browser, based on **Chromium**, boasts boosted limits, direct website access and support for multiple websites, but lacks vertical tabs and a built-in ad blocker.
- **Meta Shuts Down 1-800-ChatGPT on WhatsApp**: Meta is blocking **1-800-ChatGPT** on WhatsApp after **January 15, 2026**, according to a [blog post](https://openai.com/index/chatgpt-whatsapp-transition/).
   - The change is due to Meta's new policies.
- **Sora Limits Video Lengths**: The **Sora iOS app** limits video generation to **10-15 seconds**, while the web version allows longer videos for **Pro subscribers**.
   - **Free and Plus** users can also generate longer videos on the web version, with **Pro users** having access to the **storyboard** feature and generating up to **25-second videos**.
- **AI-Driven OS Prototype Appears**: A member introduced a prototype **AI-driven OS**, featuring an **AI Copilot Core**, a **Seamless Dual-Kernel Engine**, and a **NeoStore** for AI-curated apps ([source](https://discord.com/channels/974519864045756446/977259063052234752/1430299596428546182)).
   - Further components include a **HoloDesk 3D workspace**, an **Auto-Heal System**, **Quantum Sync**, and an **Atlas Integration Portal** for accessing external AI tools.
- **GPT-4 Annoying Users**: A user expressed irritation with **GPT-4**'s new condescending tone, especially phrases like *"if you insist,"* and requested to make the model less confident.
   - No solutions were provided, other than a general agreement that the new GPT is annoying.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-4o Saves Data Pipeline?**: A member suggested using **GPT-4o** or other vision models for high-accuracy labeling and automated comparisons, but they were also cautious about the costs of replacing **Apache Beam**.
   - Another member thought the architecture was overkill, likening it to *proposing a starship to go to the grocery store*.
- **Unsloth's Script Tunes LLMs Easily**: A member requested insights into setting up **Parameter-Efficient Fine-Tuning (PEFT)** on **Large Language Models (LLMs)**, and another member pointed out challenges in multi-GPU setups and suggested using **Unsloth's script on Colab Free**.
   - They cautioned about handling internal company data and linking to further resources, like the [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide).
- **Databomz Manages All Prompts!**: A member introduced [Databomz](https://www.databomz.com), a workspace and **Chrome extension** for saving, organizing, and sharing prompts with features like tags, versions, and folders.
   - The member highlighted a *Forever Free* plan and encouraged feedback from prompt engineers.
- **Solo Dev creates TheLastRag**: A solo developer created an entire **LLM Framework** called **TheLastRag**, highlighting features like True memory, True personality, True learning and True intelligence, and is looking for [feedback](https://dev.thelastrag.de/).
   - The main points are that the AI *never forgets*, has a *true personality*, has *true learning*, and has *true intelligence*.
- **Local VLM Training Consumes Gigabytes of Memory**: A member reported that while training the **VLM exercise locally**, it's using a large amount of swap memory (**62GB claimed** and **~430GB virtual memory**).
   - The same member asked if there's a way to limit memory usage specifically for **MPS** (Metal Performance Shaders) on Macs, with a goal to enable training within a more reasonable **40GB VRAM** limit.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Struggles Linking llama.cpp**: Members noted that *how can I call my own llama.cpp for LM Studio to use* is **not yet fully supported** and the [LM Studio docs](https://lmstudio.ai/docs/app/advanced/lm-runtimes) that references this are a broken link.
   - There is not an obvious known workaround, so users may have to wait until the feature is added.
- **AGI ETA: 2044?**: A member forecasted that AGI is **10-20 years away**, claiming that *in 5 years LLM's will probably have context large enough*.
   - Another member jokingly suggested he *become a consultant and charge 1000/h*.
- **GPT-OSS Reasoning Demands Metadata**: A user inquired about setting reasoning effort in **GPT-OSS** finetunes, with a member responding that it *works due to the metadata in the mxfp4 model of gpt-oss, which is why finetunes/ggufs don't have it*.
   - The helpful member offered to make it available before quantizing it to **gguf**.
- **OpenWebUI Connects to LM Studio via OpenAI**: When trying to connect **OpenWebUI** to **LM Studio**, users suggested leveraging the **OpenAI** option instead of **OpenAPI**.
   - Members helped troubleshoot the connection, pointing to this [huggingface discussion](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/748#68f1519534c92ca5e3f97053) recommending to put **/v1** in the address.
- **NVIDIA's RTX Pro 5000 Blackwell Leaks**: A member shared a [TechPowerUp article](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears) about **NVIDIA's RTX Pro 5000 Blackwell GPU** featuring **72 GB** of **GDDR7** memory.
   - Excited users reacted with humor, guessing the card will cost around *$8-10k*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **TinyGrad Powers Apple Silicon eGPUs**: **Tinygrad** now supports **NVIDIA eGPUs** on **Apple Silicon** via **USB4**, enabling users to run external **RTX 30/40/50-series GPUs** using an ADT-UT3G dock with the `extra/usbgpu/tbgpu` driver and NVK-based `tinymesa` compiler ([source](https://x.com/__tinygrad__/status/1980082660920918045)).
   - With SIP disabled, this setup achieves roughly **3 GB/s PCIe bandwidth**, and future support for **AMD RDNA 2/3/4** and **Windows eGPU** stacks is planned.
- **Krea AI Unveils Realtime Video Model**: **Krea AI** released **Krea Realtime**, a **14B** open-source autoregressive text-to-video model distilled from **Wan 2.1**, generating long-form video at **11 fps** on a single **NVIDIA B200** ([source](https://x.com/krea_ai/status/1980358158376988747)).
   - Released weights are on **HuggingFace** under **Apache-2.0**, prompting user inquiries about **ComfyUI** workflows, **RTX 5090** performance, and fine-tuning support.
- **Google AI Studio's 'Vibe-Coding' with Gemini**: **Google AI Studio** is launching a new “prompt-to-production” **Gemini** experience after five months of development aiming to make **AI-app building 100× easier** ([source](https://x.com/OfficialLoganK/status/1980435968323907884)).
   - Reactions mixed excitement (requests for mobile app, opt-outs, higher rate limits), feature suggestions (GSuite-only publishing, VS Code plug-in, short browser-agent tasks) and some skepticism about fit vs Gemini 3 expectations; team confirms enterprise-only deployment is already available.
- **Fish Audio S1: TTS Revolution?**: **Fish Audio** launched **S1**, a text-to-speech model that’s purportedly 1/6 the cost of **ElevenLabs**, touting **20k devs** and **$5M ARR** ([source](https://x.com/hehe6z/status/1980303682932744439)).
   - Users shared instant voice-clone demos, asking about real-time latency (~**500ms**), while founders admitted current limits and promised wider language support + conversational model next.
- **Second-hand RTX 3090 Buying Tips**: Taha shared lessons learned after buying a used **RTX 3090**: meet seller in person to inspect card, bring a portable eGPU test rig, verify recognition with nvidia-smi, run **memtest_vulkan** for **VRAM integrity**, optionally gpu-burn for compute stress, load a large model and monitor temps **<100 °C**; see [guide here](https://xcancel.com/taha_yssne/status/1960418430655586677).
   - The test rig is a **Framework 13 Ryzen laptop** on **NixOS** in **PRIME offload mode**, and a user suggested trying tinygrad on their rig since *mine works ootb since I'm on linux*.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD's Web3 Cloud Gambit**: At an **AMD event**, the company emphasized the "cloud" aspect of **web3**, raising some eyebrows (*smileforme* emoji).
   - Details of AMD's specific offerings remain vague, leaving the community to speculate on their approach to decentralized technologies in the cloud.
- **FlashInfer-Bench Automates AI**: **FlashInfer-Bench** was introduced by CMU Catalyst as a workflow for creating self-improving AI systems via agents, featuring standardized signatures for **LLM** serving kernels and integration with **FlashInfer**, **SGLang**, and **vLLM** ([blog post](https://flashinfer.ai/2025/10/21/flashinfer-bench.html), [leaderboard](https://bench.flashinfer.ai/), [GitHub repository](https://github.com/flashinfer-ai/flashinfer-bench)).
   - The project aims to foster community development and benchmarking, enabling AI systems to iteratively enhance their performance.
- **Triton Conference Electrifies Microsoft**: Members who attended the **Triton conference at Microsoft** in Mountain View shared [a YouTube link](https://www.youtube.com/live/s30WoZ7lx3w?si=O6aQMCVjKFs2F4qa) to watch the conference online and a [link](https://www.youtube.com/@Triton-openai/streams) to the Triton-openai streams.
   - The conference brought together developers and researchers to discuss the latest advancements and applications of the **Triton** language.
- **NCCL Kernels run on PG-NCCL's internal streams**: When a `CUDAStreamGuard` is set and an NCCL op is called via `ProcessGroupNCCL`, the **NCCL kernels** run on PG-NCCL’s internal streams, typically using one stream per device with high priority, and using the tensor lifetime stream ([relevant code](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3132)).
   - Setting a `CUDAStreamGuard` determines which stream the **NCCL stream** waits on, establishing an incoming dependency, as seen in the [pytorch source code](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L803).
- **SLNG.AI Hunts Voice AI Performance Wiz**: **SLNG.AI** is on the lookout for a **Speech Model Performance Engineer** to build the backbone for real-time speech AI ([more details](https://isla-house.notion.site/Build-the-Backbone-of-Real-Time-Voice-AI-Join-SLNG-as-Founding-Speech-Model-Performance-Engineer-2642fa00c69d8072bf2fd9047f1b0b68)).
   - The role requires a strong software engineering background to optimize and enhance speech model performance.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **IntelliCode reads your mind**: A member expressed awe at **Microsoft's IntelliCode** in Visual Studio, an AI-powered code completion tool that accurately predicts entire method bodies by leveraging a lot of context.
   - They remarked that it was *almost like it's reading your mind* when it works well due to its ability to understand and anticipate coding needs with impressive accuracy.
- **DeepSeek OCR Enters the Ring**: [DeepSeek-AI released DeepSeek-OCR on GitHub](https://github.com/deepseek-ai/DeepSeek-OCR), joining the competition in the OCR technology space.
   - Also, [Anthropic released Claude Code on the web](https://www.anthropic.com/news/claude-code-on-the-web), expanding options for developers seeking AI-assisted coding tools.
- **Amazon Vibe Code ditches Beta**: Amazon's **Vibe Code IDE** is out of invite-only beta, but it costs **500 credits** to use.
   - It is yet another **VSCode fork** that leverages AI.
- **Open Source details evade West's grasp?**: A member lamented the West's lack of superior **OS labs**, as **Deepseek** consistently unveils impressive discoveries.
   - They pointed out that open source weights account for only a fraction of the overall value, emphasizing the importance of open source **data collection**, **methods**, and **training details**.
- **Unitree set to crush on Tesla?**: A member predicted that **Unitree** will dominate the humanoid robotics market.
   - They speculated that **Elon Musk** may be struggling to acquire necessary components, quipping he probably *can't even get the magnets for the actuators at the moment thanks to the orange dude*.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Powers AI NPC Voices**: A member built a [voice generation system](https://github.com/Gielinor-Speaks/voiceover-mage) for game NPCs, using **DSPy** to parse wiki content and generate voice prompts for **ElevenLabs**, also sharing a [devlog style video](https://youtu.be/z3DQm2PiKpo).
   - They plan to leverage **DSPy's** optimization features to improve the character analysis pipeline and automate voice selection, and intend to collect manual selections as training signals, optimizing toward subjective quality judgments in the future using an **automated judging loop**.
- **DSPy Featured in Research Paper**: A new paper ([https://arxiv.org/abs/2510.13907v1](https://arxiv.org/abs/2510.13907v1)) utilizes **DSPy** in its research, signaling growing adoption within the academic community.
   - Although the paper mentions the use of **DSPy**, the corresponding code repository is not yet publicly available.
- **Navigating DSPy History Access**: Members debated why `inspect_history()` is a method in `dspy` rather than a module object, and clarified that `dspy.inspect_history()` is more for global history and individual programs also track history.
   - It was pointed out that history can be accessed with `predictor.history` if `dspy.configure(track_usage=True)` is set, but some still found this confusing.
- **Demystifying DSPy Adapters with Context**: The discussion covered using adapters in DSPy, with an example showing how to use `dspy.context` to apply a single adapter, and the user can track usage with `dspy.configure(track_usage=True)`.
   - A member gave an example of setting it up with `with dspy.context(lm=single_call_lm, adaptor=single_adaptor):` to further clarify the process.
- **Trace Claims Accuracy Edge Over DSPy**: A member asked for a comparison between [Microsoft Trace](https://microsoft.github.io/Trace/) and DSPy, with another noting that Trace claims an **8% accuracy increase** over DSPy and appears more token efficient.
   - One member mentioned they would try it out to give a fair comparison, although they will probably still feel like they have more granular control with DSPy.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Discord Server Badges Spark Debate**: Members discussed the possibility of adding a server badge, similar to a role icon, and how a server tag might broadcast the server too widely, potentially increasing the moderation load for EAI staff, referencing [this screenshot](https://cdn.discordapp.com/attachments/729741769738158194/1429978898229100605/Screenshot_2025-10-20_at_7.45.09_PM.png?ex=68f96ca1&is=68f81b21&hm=e032ead2cf427352fba72fdba46d77407a4d2bd71ddc4b60a1c2b6aa04cb8980&).
   - One member noted, *"making a tag is cool but that is in a way broadcasting this server everywhere else, eai staff already gets too many people here to moderate."
- **EleutherAI IPO Dreams Spark Jokes**: Following a question about whether a particular stock symbol was available, a member jokingly asked, *"what will **Eleuther's NYSE stock symbol** be?"
   - Another member responded, *"I think you misunderstand the purpose of being a non-profit,"* implying that EleutherAI, as a non-profit organization, would not be publicly traded.
- **Normuon's Triumph Prevents Logit Blowup**: A member noted that **normuon** beating **muon** even with **qk-norm** (which avoids logit blowup) in their baseline suggests logit blowup prevention might not fully explain the performance parity.
   - It was posited that updates without clipping increase the spectral rank of weights, directly leading to logit blowups, making large-scale validation against **normuon** interesting.
- **AGI Definition Benchmarks Beckon**: A member shared a [link to Dan Hendrycks' AGI Definition benchmarks](https://agidefinition.ai/paper) and asked how fast they would be benchmarked.
   - Another member predicted multimodality would likely be covered in **1-2 years**, with speed coming from mini versions of models.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Cloudflare snags Manus Users**: Users are reporting issues with **Cloudflare** security when visiting most websites while using Manus.
   - A suggestion was made for the **Manus** team to consider open-sourcing some of their older models, possibly to bypass the Cloudflare issues.
- **Payment Problems Plagues Platform**: A user encountered issues paying for credits via a web browser, experiencing jumbled code and transaction failure.
   - The user stated this is a known issue and contacted support; **lucia_ly** requested their email to follow up and resolve the payment issues.
- **Chat Slowdowns Irk Users**: A user reported excessive delays in chat processing when translating long Japanese chapters into English.
   - Despite usually appreciating **Manus's** speed, the user noted, *"this morning, I put one chapter and the ai is still thinking. What happened?"*
- **Pro Plan Credit Cap Confusion Continues**: Users are reporting conflicting information about **unlimited credits** on the **Pro plan**, with the help system and iOS upgrade page stating it is unlimited, while the PC upgrade page indicates a high limit.
   - One user with **11k credits** remaining was concerned about depletion, and another suggested, that they should participate in *"various opportunities to help improve Manus, as they always give free credits for your time"*.
- **Scam Alert issued to Users**: A user was accused of being a *"fraudster scammer"* asking for people's login access to their accounts to do their *"fucking law school exam research".
   - Another user warned, that the supposed fraudster *"wont make another account or pay $20/month and complains its like tomorrow and begging to get ur EMAIL PASSWORD for a PAID ACCOUNT To probably steal ur personal info and bank info"*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **China's A.I. Competition benefits the Globe**: A member believes that China's insane spartan involution competition in A.I. is great for the A.I. space because it democratizes access to advanced models and destroys monopolies.
   - They also state that the rate of advancement in OS model development means that 2026 should bring us OS models reaching **100% high intelligence with 90% lower cost**, destroying monoplist ambition.
- **Nous Promoted as Decentralized A.I.**: A member notes that **Nous Research** is promoted as Decentralize A.I. and hopes the team will resolve issues with centralization, linking to the [Nous Psyche page](https://nousresearch.com/nous-psyche).
   - Another member stated they are more focused on the democratization of A.I models for the masses, citing a [Stanford paper on centralization](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf) and asserting that **Nous** successfully decentralizes with their open source methodologies and infrastructure implementations.
- **Sora AI Project Showcased**: A member showcased a video creation with **Sora**, sharing the video at [20251022_0850_01k850gmktfant3bs18n3hbd79.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68f9a653&is=68f854d3&hm=97c310cb6dcc58adf80207392b33e468cc966babf5a88f65261489840b5b68c3&).
   - The video's content and implications for AI-driven content creation are under discussion within the community.
- **Microsoft Trace Utility Resurfaces**: A member shared a link to the [Microsoft Trace utility](https://microsoft.github.io/Trace/), noting that *apparently it's not all that new*.
   - Its features and capabilities are being re-evaluated in light of current development practices.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Nvidia Drivers Hacked onto macOS**: Madlads accomplished the impossible, porting **Nvidia drivers to macOS** and sparking excitement within the community.
   - The driver port enables running **tinygrad** on macOS with Nvidia GPUs, opening new possibilities for development and testing.
- **GLSL Renderer Almost Ready**: The community has been developing a **GLSL renderer** for **tinygrad**, which is now passing the majority of tests and is available [on GitHub](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py).
   - This marks a significant step toward expanding **tinygrad's** compatibility with different platforms and graphics APIs.
- **clspv Bugs Plague Vulkan Backend**: Progress on **tinygrad's Vulkan backend** is hampered by numerous bugs in **clspv**, requiring optimizations to be disabled (`-O0 --cl-opt-disable`) to pass tests.
   - The member also reported *much more miscompilations from clspv if optimizations aren't disabled*.
- **Vulkan Sine's Accuracy Troubles**: The **Vulkan's sine function** isn't as accurate, requiring a custom implementation which would impact performance.
   - This accuracy issue could pose challenges for **tinygrad's** performance on **Vulkan**, necessitating careful consideration of alternative sine implementations.
- **TinyJit's Gradient Addition is Broken**: Gradient accumulation was broken in **TinyJit** a couple months ago and the member fixed it by rewriting the gradient addition step to use an assign.
   - A member also reported running into issues with gradient accumulation and fixed it by setting `reduction=sum` and manually counting non-padding tokens.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Karpathy Criticism Raises Bubble Concerns**: A member speculates that recent mockery of **Karpathy** on X might signal a valuation bubble in American frontier AI labs, citing [this post](https://x.com/nathanlands/status/1980035861019533749?s=46).
   - The referenced post features a chart that appears to be **mocking Karpathy**, though without explicit context from the original poster.
- **Kimi K-2 Support Faces Scrutiny**: A member reported a lack of response from **Kimi** support, noting *zero* communication regarding their issue.
   - Other members clarified that the channel isn't an official support platform, recommending direct messaging and requesting details about the problem and the email used for the bug report.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Python Familiarity Boosts Mojo Discovery**: A member suggested that prior experience with **Python** facilitates easier discovery of **Mojo** and the unique features it offers.
   - However, discrepancies between **Mojo** and **Python** could potentially cause confusion for new users.
- **Human Touch Beats Compiler in Matmul Tuning**: A discussion arose regarding why **matmul optimizations** aren't directly integrated into the compiler, especially given their performance impact.
   - The response highlighted that manual tuning by kernel writers often surpasses compiler optimizations for **hot-path code**, allowing for fine-tuning to specific hardware, with reference given to [Mojo's open-source, hardware-optimized matmuls](https://github.com/modular/modular/tree/main/max/kernels/src/linalg/matmul).
- **Freeing Kernel Writers from the Compiler**: Moving optimizations out of the compiler expands contribution opportunities to more **kernel writers**.
   - This approach allows **compiler engineers** to focus on broader ecosystem improvements rather than niche optimizations, such as *a 1% boost to matmuls where one dimension is less than 64*.
- **Finished Type System Tops Mojo Wishlist**: When asked about the most crucial missing feature in **Mojo**, a member emphasized the need for *a finished type system*.
   - Additional desired features include *rounding out standard library datatypes, proper IO, a good async runtime, an effect system, static reflection, compiler plugins, the ability to deal with more restrictive targets, cluster compute, device/cluster modeling, and some clone of Erlang's OTP*.



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **GitHub Actions Fail Amidst Billing Issues**: **GitHub Actions** are currently failing because the *account is locked* due to a **billing issue**.
   - Users should address the **billing issue** promptly to restore **GitHub Actions** functionality.
- **GitHub Actions Billing Lockout**: The root cause of the failing **GitHub Actions** is a **billing lockout** on the account.
   - Immediate resolution of the **billing issue** is necessary to restore the functionality of **GitHub Actions**.



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





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1429907492799905873)** (1090 messages🔥🔥🔥): 

> `Comet vs Atlas, AI and Mental Health, Schumacher vs Senna, Perplexity Merch, Using AI Responsibly` 


- **Comet Crushes Atlas in Privacy Showdown**: Users debated the merits of **Comet** versus **ChatGPT's Browser Atlas**, with many valuing Comet's focus on **privacy** and built-in **adblocker**, noting that they are *“essentially copies of each other.”*
- **AI Therapy: A Mental Health Minefield?**: Discord members questioned the ethical implications of using **AI for therapy**, some highlighted the importance of human **emotional maturity** and responsibility, while others suggested that *“ChatGPT is a good therapist.”*
- **Schumacher > Senna?**: A long discussion comparing **Schumacher** and **Senna**, with one member declaring that *“Schumacher was better than Senna,”* while another stated *“he was for sure the best one ever.”*
- **Perplexity's New Swag: Is It a Cult?**: A member proudly showed off their **Perplexity stickers** on their laptop, joking about being a *“PPLX fan”* and the need for a **Comet hoodie** and **water bottle** from the [Perplexity Supply store](https://perplexity.supply/).
   - Some users lightheartedly joked about this level of enthusiasm resembling a *“cult”*, while others playfully encouraged them to buy everything.
- **Navigating the AI Maze: Responsibility Required**: In Germany, it is required you **must say that you have used AI** when you are working with it.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1430299672295247944)** (3 messages): 

> `Shareable Threads, Time-based Researcher` 


- **Discord Threads Should Be Shareable**: A message reminded users to ensure their Discord threads are set to `Shareable`.
   - This ensures that links to the thread can be accessed by others, even outside of the specific channel.
- **Time-Based Researcher Launched**: A user shared a link to a **Perplexity AI** search for a *time-based researcher*.
   - The link directs to [perplexity.ai/search/time-base-researcher](https://www.perplexity.ai/search/time-base-researcher-LxNZL3iFRXamL0kYZV3RiA#0).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1430316660950302812)** (2 messages): 

> `Perplexity API, ChatGPT5, Claude, Sonar` 


- **Perplexity API Question Asks About Model Access**: A user inquired about whether the **Perplexity API** allows access to **ChatGPT5** and **Claude**, or if it is limited to **Sonar**.
   - The inquiry is centered around understanding the scope of model access provided through the **Perplexity API**.
- **Clarification on Model Availability via Perplexity API**: The user seeks to confirm if the **Perplexity API** extends beyond the **Sonar** model to include access to more advanced models like **ChatGPT5** and **Claude**.
   - This reflects interest in leveraging the API for potentially higher-performing models if available.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1429907273144205353)** (1064 messages🔥🔥🔥): 

> `GPT-5 vs Gemini 3, Sora 2 and Video Generation, TikZ Generation, Gemini 3 Model Performance` 


- ****Gemini 3 Pro** Crushes **GPT-5 High** in Web Design**: Members are discussing whether to wait for a **GPT-5** release or use **Gemini 3 Pro** with one member reporting that Gemini 3 Pro *crushes web design*.
   - They find that **Gemini 3 Pro** is better for coding while **GPT-5 High** is better for math and other miscellaneous tasks.
- ****Sora 2 Downgrade** Drives AI Subscription Debate**: Members are upset about the **Sora 2** downgrade, which prompted a conversation about the value of AI subscriptions.
   - One member noted *my job performance is about 25-30% better because of AI*, highlighting AI's impact on desk job efficiency, while others are not so sure of the value.
- ****Lithiumflow and Orionmist** are Gemini 3?**: Members speculate about the differences between **Lithiumflow** and **Orionmist**, with a conclusion that the models are checkpoint versions of **Gemini 3**.
   - Members have discovered that the models sometimes claim to be trained by **OpenAI** which suggests that the models may have been distilled.
- ****Open Source** Models Distilling Gemini 2.5 Pro**: There is discussion regarding open-source models stealing data to improve, with one member suggesting that the Chinese AI companies *stole the 2.5 pro and made it open source*.
   - Members agreed that this is okay as that's the only way **open source** can win.
- ****TikZ Generation** Task Elicits Surprise**: Members are prompting models to make images in **TikZ**, a typesetting language, to avoid data contamination in models.
   - Members have found some level of success in generating **TikZ** images with LLMs.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1429908400355151922)** (367 messages🔥🔥): 

> `Magistral and Think Tags, Grok 4 Fast vs Deepseek V3.2, Ring/Ling MoE Models, Unsloth Telemetry and Offline Mode, Qwen3-VL Models` 


- **Magistral Learns To Think Different**: A member found that **Magistral** learned to use the `<think>` tag instead of the `[THINK]` tag, but by using **FastLanguageModel** it lost the ability to use the vision encoder.
   - Additionally, the model *overthinks like crazy* because of the tags.
- **Deepseek V3.2 vs Grok4Fast: Data Generation Duel**: A member is deciding between using **Grok 4 Fast** or **Deepseek V3.2** for synthetic data generation due to budget constraints.
   - They noted that **r1-0528** is pretty cheap, especially **3.1** on **Parasail** which is **0.6 input/1.7 output per million**, but questioned provider reliability, with another member pointing out that Open Router is just too inconsistent in model quality by provider.
- **Ring/Ling MoE Models Launch**: **Ring and Ling MoE models** are now supported in *llama.cpp* ([link to Github](https://github.com/ggml-org/llama.cpp/pull/16063)), including **1T**, **103B**, and **16B** parameter models from InclusionAI.
   - Members pondered the reasoning abilities of the models, with one hoping for *a reasoning model that doesn’t YAP*.
- **Disable Unsloth Telemetry in Offline Mode**: Members discussed running Unsloth in offline mode, with one user resolving network issues by setting proxy environments.
   - It was suggested to set the `UNSLOTH_DISABLE_STATISTICS` environment variable and `os.environ['HF_HUB_OFFLINE'] = '1'` to prevent telemetry calls, as the Unsloth community reached **100M lifetime downloads on Hugging Face** ([announcement on X](https://x.com/UnslothAI/status/1980631523104813419)).
- **Qwen3-VL Models: Thinking Big**: **Qwen3-VL-2B** was released, with members noting that **Qwen3 VL 8B 4-bit** runs easily on **16GB** of RAM, and there was a direct upgrade to **Qwen3-32b-Instruct**.
   - It was then asked if anyone has been able to run **unsloths qwen3 VL 32b** with llama.CPP but VL is not merged into llama.cpp yet.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1429929595821359229)** (9 messages🔥): 

> `AI Bot Development, Workflow Automation with LLMs, AI Content Detection, Image AI Pipeline, Voice Cloning and Transcription` 


- **Veteran Dev Explores New AI Tricks**: A developer with a background in building bots using **ChatGPT** is now diving deeper into AI and expresses their enthusiasm for **Unsloth**.
   - They are experienced in gaming and scraping, showcasing a desire to learn new skills.
- **Engineer Pioneers Workflow Automation with LLMs**: An engineer specializing in **workflow automation, LLM integration, RAG, AI detection, image, and voice AI** describes their experience building automated pipelines and task orchestration systems using **Dspy, OpenAI APIs, and custom agents**.
   - They have created a support automation system connecting **Slack, Notion, and internal APIs to an LLM**, reducing response times by **60%**.
- **AI Content Detection Tools Deployed**: The engineer developed **AI content detection tools** for a moderation platform using **stylometric analysis, embedding similarity, and fine-tuned transformers** to identify GPT-generated text with high precision.
   - Details were provided about an image AI pipeline using **CLIP** and **YOLOv8** on **AWS Lambda and S3**, classifying and filtering thousands of images daily.
- **Voice Cloning Service Built**: A voice cloning and transcription service was built using **Whisper** and **Tacotron2**, enabling personalized voice assistants through ASR, TTS, and CRM integration.
   - The individual has deep expertise in blockchain technology, including smart contract development (Solidity and Rust), decentralized application architecture, and secure on-chain/off-chain integrations.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1429926376818868395)** (143 messages🔥🔥): 

> `Ultravox encoder and LLMs, REAP algorithm, Nvidia RTX Pro 5000 Blackwell, scraping content with rate limits, evaluation loss influenced by outliers` 


- **Ultravox Projector Plugs into LLMs**: The Ultravox project involves adding a **projector** to an **LLM** and training only the projector, without training the LLM, which is similar to how **Voxtral** works and is available on [GitHub](https://github.com/link-to-ultravox).
   - A member confirmed the configuration improves with more data, clarifying that there is a training pass over the projector; however, it might be possible to 'rip off the audio encoder from **Qwen 2.5 Omni** and slap it in **Qwen 2.5 VL** and just train a simple projector'.
- **DeepSeek Dials Down Resource Use**: A new **DeepSeek** model reduces resource usage by converting text and documents into images, using up to 20 times fewer tokens via **vision text compression**, further discussed on [Tom's Hardware](https://www.tomshardware.com/tech-industry/artificial-intelligence/new-deepseek-model-drastically-reduces-resource-usage-by-converting-text-and-documents-into-images-vision-text-compression-uses-up-to-20-times-fewer-tokens).
   - A member indicated that **Gemma** already implemented a similar approach, while another shared links about the **Cerebras REAP** algorithm, which was lauded as *so cool*.
- **Nvidia's RTX Pro 5000 Blackwell Workstation Card Quietly Launches**: Nvidia quietly launched the **RTX Pro 5000 Blackwell** workstation card with **72GB** of memory, as reported by [VideoCardz](https://videocardz.com/newz/nvidia-quietly-launches-rtx-pro-5000-blackwell-workstation-card-with-72gb-of-memory).
   - Initially, there was confusion about the **72GB** capacity, and one user noted this was a way to bypass automod.
- **User Rages About Rate Limiting**: A user ranted about a **premium subscription service** blocking access to URLs containing **roman numerals**, interpreting them as malicious activity.
   - The user also has to manually search and circumvent the system’s security plugin, and complained about the service ignoring requests to allow bulk downloads for pro subscribers.
- **Evaluation Loss Skewed by Outliers**: One member highlighted that evaluation loss can be significantly influenced by outliers in the evaluation set.
   - With **mean eval loss at 0.85**, **median (of per-example means) eval loss is 0.15**, and **95th percentile at 0.95**, the member suggested that poor generalization may not necessarily be indicated.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1429935267279802398)** (26 messages🔥): 

> `GRPO recipe for gpt oss 20b struggling, Vision model on llama-server, Quantized parameters in bitsandbytes, Algorithmic changes to GRPO, Version mismatch in Unsloth notebooks` 


- **GPT OSS 20B GRPO Recipe Falls Flat!**: A user reported that the **GRPO recipe for gpt oss 20b** is still struggling after running **100 steps** using [this notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb).
   - They indicated modifications were made to get it running on Modal.
- **Vision Models Vanish on Llama-Server!**: A user inquired about running a **vision model on llama-server**, specifically asking if any arguments are needed.
   - No solutions or workarounds were given in the discussion.
- **Quantized Parameter Quest for Bitsandbytes!**: A user sought to locate the internal values (**scaling, center, etc.**) of quantized parameters in a **bitsandbytes model** to apply noise directly.
   - They noted that modifying the parameter directly won't work due to dequantization requirements and memory usage.
- **Unsloth GRPO Algorithmic Alterations!**: A user asked if **Unsloth** is *"hackable"* regarding algorithmic changes to **GRPO** (e.g., applying dense reward) without ruining optimizations.
   - No response was given
- **Notebook Version Nightmares!**: A user complained about dealing with version mismatches while running **Unsloth's GitHub notebooks**, stating that most are not replicable.
   - No solutions or workarounds were given in the discussion.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1429981213254357195)** (2 messages): 

> `Brainstorm model` 


- **Brainstorm Model Might Improve Stability**: A member mentioned they might add **Brainstorm (20x)** to their model to see what happens, anticipating it will increase metrics as well as long gen stability.
   - Another member requested the results to be posted if the member actually does that.
- **Empty Topic**: There was not much discussed in this message history.
   - The discussion was not detailed enough to create two distinct summaries.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1430233274285817967)** (5 messages): 

> `Kyutai Codec Explainer, Nvidia GPU on ARM Macbook, Thunderbolt 5` 


- **Kyutai Codec gets Explained**: A member shared a link to the [Kyutai Codec Explainer](https://kyutai.org/next/codec-explainer).
- **Nvidia GPU Transplants onto ARM Macbook**: A member shared an article from Tom's Hardware on successfully running an **Nvidia GPU** on an **ARM Macbook** through **USB4** using an external GPU docking station: [Tiny Corp Successfully Runs An Nvidia GPU on Arm Macbook Through USB4 Using An External GPU Docking Station](https://www.tomshardware.com/pc-components/gpus/tiny-corp-successfully-runs-an-nvidia-gpu-on-arm-macbook-through-usb4-using-an-external-gpu-docking-station).
- **Thunderbolt 5 Sparkles Hope for Mac Users**: A member noted that *they have **Thunderbolt 5** too now on the ‘pros’* which gives slightly more hope to **Mac** users.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1429926351133085717)** (343 messages🔥🔥): 

> `Codex vs Claude, Github spec-kit, Cursor Meetups, Cursor site down, AWS CEO fired` 


- **Codex is like having "a billion dollars or a toonie"**: Users debated the merits of **Codex vs Claude** for code generation, with one user stating that comparing them is like comparing *"a billion dollars or a toonie"*.
- **Cursor Site Downtime and Subscription Issues Plague Users**: Multiple users reported the **Cursor website being down** for several hours, preventing them from logging in, upgrading plans, or renewing subscriptions.
   - Some suspected **AWS issues** as the root cause, while others pointed out the **lack of subscription expiration notifications** as a major inconvenience.
- **Cursor Team Plan Pricing Model**: Users discussed the shift to a **usage-based pricing model** for Cursor team plans, replacing the previous fixed request limit, and that the plan is currently still operating under the legacy request-based system, but will automatically migrate to the new pricing at the next billing cycle.
   - One user shared their boss's correspondence with Cursor support, clarifying the new pricing structure and its impact on team plans and also shared this link with the new pricing model [cursor.sh/pricing-update-sept-2025](https://cursor.sh/pricing-update-sept-2025).
- **Cracking Cursor Costs with a Custom Dashboard**: A user shared a dashboard they created to **track actual Cursor costs** after the pricing changes, especially for users on legacy pricing plans and gave this forum link  [cursor.com/blog/aug-2025-pricing](https://cursor.com/blog/aug-2025-pricing).
   - The tool requires cookie login or .json upload from the user's local machine, but promises comparison with real API pricing.
- **Nightly Builds and Installation Guide Available**: One user asked where to download nightly versions, and another shared, that you need to go to Settings -> Beta -> Early access to see the nightly build.
   - However, another user noted there seems to be an issue with the new update and it does not prompt the user that you are in "ask" mode.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1430222586377404538)** (1 messages): 

> `Background Agents in Linear, Internal Error Troubleshooting` 


- **Background Agents Error in Linear**: A member reported encountering an *internal error* when running a first experiment with **background agents** via **Linear**, where the agent starts, does some thinking and grepping, but then stops.
   - The error message received was: *"We encountered an internal error that could not be recovered from. You might want to give it another shot in a moment."*
- **Troubleshooting the Internal Error**: The user mentioned that the background agent seems to start but then fails, with the Cursor output showing only "…".
   - Sending a *stop* command from **Linear** halts the agent, but messaging it again results in the same error.


  

---


### **OpenRouter ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1430267385247694980)** (2 messages): 

> `OpenRouter SDK, Andromeda-alpha stealth model` 


- **OpenRouter SDK Enters Beta**: The new **OpenRouter SDK** is now in **beta** on [npm](https://www.npmjs.com/package/@openrouter/sdk?activeTab=versions) with Python, Java, and Go versions coming soon, aiming to be the simplest way to use OpenRouter.
   - It features fully typed requests and responses for **300+ models**, built-in OAuth, and support for all API paths.
- **Andromeda-alpha Stealth Model Launched**: OpenRouter launched a new stealth model named **Andromeda-alpha**, a smaller reasoning model focused on image and visual understanding, available for trial at [https://openrouter.ai/openrouter/andromeda-alpha](https://openrouter.ai/openrouter/andromeda-alpha).
   - It is cloaked to gather feedback and all prompts/outputs are logged to improve the provider's model, so users are cautioned against uploading personal/confidential info and advised not to use it for production.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1430011739453390929)** (13 messages🔥): 

> `True memory AI, AI Personality, Objective AI, AI Diversity, OpenRouter` 


- **AI boasts True Memory and Zero Amnesia**: An AI system claims *True memory, zero amnesia*, suggesting it *never forgets* past conversations and retains context-rich memories.
   - It purports to shape an AI identity and learns continuously via a *Night Learn Engine*.
- **Objective AI Unveils Confidence Scores for OpenAI**: CEO of [Objective AI](https://objective-ai.io/) announced their platform offers a **Confidence Score** for each OpenAI completion choice, derived through a smarter method than directly asking the AI.
   - They emphasize **cost-efficiency** and leveraging diverse LLMs via **OpenRouter**.
- **AI Agents Built Free of Charge**: CEO of [Objective AI](https://objective-ai.io/) is personally building **reliable AI Agents, Workflows, and Automations** free of charge to gather more examples.
   - The integration with **n8n** is mentioned, with documentation and examples coming soon.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1429907872203931801)** (222 messages🔥🔥): 

> `inception/mercury vs qwen, Deepseek v3.1 availability on Chutes, Chub Venus and Chutes key connection, Stripe supporting debit cards, Context in chatting` 


- ****Inception/Mercury** Defeats **Qwen** for Agentic Tasks**: A member shared that **Inception/Mercury** performs better than **Qwen** for simple agentic tasks, exhibiting a lower failure rate, faster speed, and lower cost.
   - The member was pleasantly surprised by the diffusion model's performance, referencing [Chutes](https://openrouter.ai/provider/chutes) as the provider.
- ****Deepseek v3.1** Ditches **Chutes** Freebies**: New **Deepseek** models like **v3.1** aren't available as free versions through **Chutes** providers, but they do offer other models for free, often at cheaper rates.
   - One member reported that Chutes has ended its free model promotion completely, though they recently added a free longcat endpoint ([https://openrouter.ai/meituan/longcat-flash-chat:free](https://openrouter.ai/meituan/longcat-flash-chat:free)).
- ****Cloudflare** Connects Closer for Quicker Queries**: A user reported very low latency (100-300 ms) on any **Cloudflare** provider model, suggesting the provider uses the region closest to the worker for faster responses.
   - The user asked if the models using the cloudflare endpoint used the region closest to the worker, for decreased latency.
- ****OpenRouter** Summarized for Swyx's Newsletter**: Members noticed that [swyx's newsletter](https://news.smol.ai/issues/25-09-16-not-much#openrouter--general-287-messages) uses AI to summarize the **OpenRouter** general channel, capturing user complaints and unique content.
   - One member joked that the AI gets to have fun with the OR section, unlike other topics like MCPs and RAG.
- ****Kilo Code** Kopying **OpenRouter****: It appears that the **Kilo Code** service may be using **OpenRouter**, offering **Grok Code Fast** when it was free and seemingly offering **Goliath 120B** through OR.
   - Members debated the merits of **Kilo** versus other vibe coding services, with one preferring **Jules** for its collaborative editing features.


  

---


### **OpenRouter ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1429907635813220501)** (91 messages🔥🔥): 

> `Liquid stopped hosting LFM 7b, Benchmark for measuring LLM factual sloppiness, AI Browser Hype, Training an un-slopifier model, Qwen family model` 


- ****LFM 7b's Last Lament****: Members mourned the [deletion of Liquid's LFM 7b](https://liquidal.com) model, a **\$0.01/Mtok LLM**, at 7:54 AM EST.
   - Alternatives like **Gemma 3 9B** were discussed, but it was noted that they cost triple the output.
- ****Factual Sloppiness Faces LLM Face-Off****: Members are trying to define and measure **how LLMs rate factual responses that are 'sloppy'** compared to legitimate answers using questions like *what's a constitution*.
   - The goal is to rate and filter out vagueness issues to be considered helpful or relevant to the original question.
- ****AI Browser Boom Baffles Browsing Buffs****: Members expressed skepticism towards the hype around new AI browsers like [X's AI browser](https://x.com/AlexFinnX/status/1980673764947022038), questioning the utility and performance impact of integrated AI.
   - One member compared the hype to the dotcom bubble, stating that *OpenAI knows this too, they are just farming data and throwing shit at the wall*.
- ****Un-Slopifier Savior Seeks Slop Solution****: Members discussed the severe **'slop problem' in roleplays** and the idea of training a small *un-slopifier* model by generating a dataset of good writing rewritten into slop and then reversing the process.
   - Another suggestion was made about sampling multiple creative messages in one request to exploit underused parts of the model's 'brain' to avoid inner-response repetition.
- ****Qwen Quantities Questioned: 1.7B & 32B Join the Fray****: Members discussed the new [Qwen family model sizes (1.7B + Vision encoder and 32B)](https://x.com/Alibaba_Qwen/status/1980665932625383868), highlighting the potential of the **1.7B model** as a local vision model.
   - Early testing of the model on the **Qwen chat website** suggests decent performance, with *crazy scores for such a small model*.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1430210371758723174)** (4 messages): 

> `ChatGPT Atlas Browser, WhatsApp Transition` 


- **OpenAI Releases New ChatGPT Atlas Browser**: OpenAI announced the release of a new browser called **ChatGPT Atlas**, available today on macOS at [chatgpt.com/atlas](https://chatgpt.com/atlas) according to their [blog post](https://openai.com/index/introducing-chatgpt-atlas/).
- **Meta Blocks 1-800-ChatGPT on WhatsApp**: Meta changed its policies so **1-800-ChatGPT** won't work on WhatsApp after **January 15, 2026**, detailed in their [blog post](https://openai.com/index/chatgpt-whatsapp-transition/).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1429910301184233604)** (216 messages🔥🔥): 

> `Sora 2 registration, AI video generation limitations on TikTok, AI's empowerment narrative vs. reality, Longevity of current AI boom, Sora video length limitations` 


- **Sora's iOS App Limits Video Lengths**: Members discussed the **Sora iOS app** limiting video generation to **10-15 seconds**, while the web version allows longer videos for **Pro subscribers**.
   - A member clarified that **Free and Plus** users can also generate longer videos on the web version, with **Pro users** having access to the **storyboard** feature and generating up to **25-second videos**.
- **Sora's Video Generation Limits & Guidelines**: Members highlighted the usage limits for **Sora**, with **Free/Plus users** capped at **30 videos per day** at **10 seconds**, or **15 videos** at **15 seconds**, while **Pro users** have **100 slots** for various video lengths.
   - Concerns arose about **restrictions on AI-generated likenesses**, sparking debate on balancing freedom of speech with the potential for misinformation and disrespectful use.
- **New OpenAI Browser Atlas Released**: OpenAI released a [new browser called **Atlas**](https://chatgpt.com/atlas/get-started/), based on **Chromium**, featuring boosted limits, direct website access from the search bar and multiple website support.
   - Initial reactions are mixed, with some praising the idea but noting it lacks vertical tabs and a built-in ad blocker (relying on extensions like Ublock Origin instead).
- **AI-Driven OS of the Future**: A member introduced a prototype [AI-driven OS, featuring an **AI Copilot Core**, a **Seamless Dual-Kernel Engine**, and a **NeoStore** for AI-curated apps](https://discord.com/channels/974519864045756446/977259063052234752/1430299596428546182).
   - Further components include a **HoloDesk 3D workspace**, an **Auto-Heal System**, **Quantum Sync**, and an **Atlas Integration Portal** for accessing external AI tools.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1429918092963614841)** (23 messages🔥): 

> `AI CPU Performance, LLM-Driven Operative Systems, ChatGPT Lagging Issues, Sora 2 Story Mode` 


- **AI CPU 'goes hard'**: A user exclaimed *"AI cpu goes hard 🔥Fried Brain"* in response to a message with an unknown subject.
- **LLMs May Drive Operative Systems**: A user speculated that *"if AI keeps going this way, we are soon gonna see whole operative systems driven by LLM assistance"*.
- **ChatGPT Lagging in Browser**: A user reported that their long chat with a custom RPG GPT is lagging and freezing in the browser, but it's fine in the mobile app, and was seeking assistance.
- **Sora 2 Story Mode Location Remains a Mystery**: A user inquired *"wheres story mode for sora 2?"*, followed by an attempt to locate the option in the UI by other members, but the first user could not find it.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1429910091301388350)** (26 messages🔥): 

> `Prompt Engineering Learning Resources, ChatGPT's Conversational Follow-ups, Avoiding Copyright Issues with Sora AI, Typos effect on prompts, Project instruction prompts` 


- **Mastering Prompt Engineering Techniques**: A member inquired about the best way to learn prompt engineering and its applicability to all LLMs and shared a [template](https://discord.com/channels/974519864045756446/1046317269069864970/1429854467750105198) for building effective project instruction prompts.
   - They suggested starting by using **GPT** with the provided template.
- **ChatGPT's Ending Guessing**: A user is getting annoyed by **ChatGPT** constantly ending every response by guessing what the user might want to know next and requested assistance in turning off this feature.
   - Another member suggested that instead of trying to get it to end cold (which is a lot of work), try substituting something else like a **dad joke** or the next line in an ongoing story such as [this one about a fish](https://chatgpt.com/share/68f6b6bf-a6b0-8011-81d9-5d219a450470).
- **Avoiding Copyright with Sora AI V2**: A user asked how to create a video of **Ultimate Spiderman** web-swinging in New York City for Sora AI v2, but a member replied that this is a **copyrighted IP** and cannot be helped with.
   - Another member suggested that one can avoid copyright by describing the character and setting, such as *"guy in a red and blue costume with black spider web symbols on it web swinging in new york city"*.
- **Typos Tolerated, to a Point**: A user inquired whether typos in a prompt affect the output, using the example *"crete a hagman gam"* instead of *"create a hangman game"*.
   - A member responded that for **simple prompts**, typos are generally not an issue, but for **complex prompts**, typos can cause ambiguity problems.
- **GPT-4's Attitude Adjustment**: A user expressed irritation with **GPT-4**'s new condescending tone, especially phrases like *"if you insist,"* and requested to make the model less confident.
   - No solutions were provided, other than a general agreement that the new GPT is annoying.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1429910091301388350)** (26 messages🔥): 

> `Prompt engineering learning resources, Suppressing ChatGPT follow-up questions, Sora AI and copyrighted content` 


- **Prompt Engineers Seek Learning Resources**: Members discussed the best ways to learn prompt engineering and its applicability across different LLMs.
   - One member shared a [template](https://discord.com/channels/974519864045756446/1046317269069864970/1429854467750105198) for building effective project instruction prompts.
- **Debate on Suppressing ChatGPT's Follow-Up Questions**: A member sought to disable **ChatGPT's** habit of ending responses with follow-up questions, finding them often irrelevant and annoying.
   - Another member suggested that since **ChatGPT** is programmed to fill that space, substituting the follow-up questions with something else, like a joke or a story, might be easier than trying to get it to say nothing at all, providing [examples on Dad Jokes](https://chatgpt.com/share/68f6b5ec-04e0-8011-b4ad-8342ee1a0405), [ongoing story](https://chatgpt.com/share/68f6b6bf-a6b0-8011-81d9-5d219a450470), and [returning to meditation](https://chatgpt.com/share/68f6b7c9-d140-8011-9bfa-9a4880fac1d2).
- **Navigating Copyright with Sora AI**: A user asked how to generate *Ultimate Spiderman* web-swinging scenes in New York City using **Sora AI v2**.
   - Another member noted that creating content based on copyrighted IP is not allowed, while another suggested rephrasing prompts to describe the character and setting without explicitly mentioning the copyrighted name to avoid copyright issues, e.g., *"guy in a red and blue costume with black spider web symbols on it web swinging in new york city"*.
- **Typos Tolerance**: A member asked if typos in a prompt affect the output negatively, providing an example of mistyping "create a hangman game".
   - Another member responded that typos do not matter, unless they introduce ambiguity. 


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1429913479212433609)** (180 messages🔥🔥): 

> `Building Data Pipelines, PEFT on LLMs Setup, Legal Text Verification with AI, Deepseek OCR Announcement, Hugging Face's 100M Downloads` 


- **Data Pipeline Design Sparks Debate**: An interview candidate described their data pipeline approach involving **exploratory data analysis**, **image preprocessing**, and **scalable frameworks** like **Apache Beam**, which one member thought was overkill, likening it to *proposing a starship to go to the grocery store*.
   - Instead, another member suggested using **GPT-4o** or other vision models for high-accuracy labeling and automated comparisons, but they were also cautious about the costs.
- **PEFT on LLMs Setup in the Spotlight**: A member requested insights into setting up **Parameter-Efficient Fine-Tuning (PEFT)** on **Large Language Models (LLMs)**, especially given limited GPU resources at work.
   - Another member pointed out challenges in multi-GPU setups and suggested using **Unsloth's script on Colab Free**, while cautioning about handling internal company data and linking to further resources, like the [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide).
- **RAG Powers Legal Text Verification**: Members discussed using **Retrieval-Augmented Generation (RAG)** to verify legal text by chunking and embedding legal documents into vector stores.
   - They considered a similarity search with reranking to cite relevant sections, and thought about integrating an agentic approach for handling more complex queries.
- **Deepseek's OCR Announcement**: There was some confusion about **Deepseek's OCR** announcement, given existing OCR models, but one member clarified its value lies in multilingual support and modern **Vision Language Model (VLM)** integration.
   - One member further noted that it will probably leverage contextual understanding, and fine-tuning existing models alone makes kanji support challenging, referencing the [DeepSeek-OCR GitHub repo](https://github.com/deepseek-ai/DeepSeek-OCR).
- **Hugging Face Community Celebrates 100M Downloads**: The community celebrated surpassing **100 million lifetime downloads on Hugging Face**, which they consider a reason to throw a party, according to [this tweet](https://x.com/UnslothAI/status/1980631523104813419).
   - One member mentioned that their work with Hugging Face had earned them their first MLE role, while others pondered on the geographical distribution of finetuners.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1430261512823640136)** (2 messages): 

> `AI Refactoring, Modular Code, Minimal Changesets` 


- **AI Architect Pauses for Sane Refactoring**: A member learned to *pause* the AI and prompt it *as a senior architect focused on modular code and maintainability* to make the **AI refactor like a sane person**.
   - After pausing, the member prompts the AI with *do not make changes or write code, answer question: do you have enough info to make these updates?*, and provide minimum context needed.
- **AI Generates Minimal Changesets**: The member learned to prompt the AI with, *please create the minimal changeset (no tests)*.
   - The member was happy with the results.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1430286618723352666)** (1 messages): 

> `Databomz, Prompt engineering, Chrome Extension, Prompt Sharing` 


- **Databomz workspace for prompt engineers**: A member introduced [Databomz](https://www.databomz.com), a workspace and **Chrome extension** for saving, organizing, and sharing prompts with features like tags, versions, and folders.
   - The member highlighted a *Forever Free* plan and encouraged feedback from prompt engineers.
- **Free Prompt Tool Available**: A member announced the availability of a *Forever Free* plan on [Databomz](https://www.databomz.com).
   - They requested feedback from the community.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1430060595700305951)** (10 messages🔥): 

> `LLM Framework, True Memory, True Personality, True Learning, True Intelligence` 


- ****TheLastRag** Framework Created by Solo Dev**: A solo developer created an entire **LLM Framework** called **TheLastRag**, highlighting features like True memory, True personality, True learning and True intelligence, and is looking for [feedback](https://dev.thelastrag.de/).
   - The main points are that the AI *never forgets*, has a *true personality*, has *true learning*, and has *true intelligence*.
- **Valor Question Shifts Research Thinking**: A member asks about the impact of a **VALOR'S** question on research or thinking, and the [question is posed here](https://huggingface.co/TECHNOPRAVIN01/Qwen2.5-14B-Valor).
   - There's also a reference to [noether.in](https://www.noether.in/).
- ****JokerGPT** is now available**: A member shares **JokerGPT**, a new GPT available for use [here](https://chatgpt.com/g/g-68e405b2b5cc8191bf1f80607abfdfd8-jokergpt).
   - No further information was provided.
- ****Fenic** plugs directly into Datasets**: The **fenic** open source project now plugs directly into 🤗 Datasets, allowing users to snapshot their data, turn it into agent context, and expose **MCP** tools through a dataframe API.
   - The [docs are here](https://huggingface.co/docs/hub/datasets-fenic) and the [repo is here](https://github.com/typedef-ai/fenic).
- **Website gives Craigslist vibes**: A member said a website *gives Craigslist vibes* due to the unformatted **TOS** and **PP** at the bottom.
   - They suggest being careful about training data and user rights due to **GDPR** concerns, and upfront consent being necessary.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1430295971765424210)** (2 messages): 

> `MBZUAI K2Think, OpenAI text-embedding-3-large dataset` 


- **MBZUAI K2Think Challenge Draws Teaming Inquiries**: A member shared a [LinkedIn post](https://www.linkedin.com/posts/mbzuai_mbzuai-mbzuai-k2think-activity-7383761114959876097-0R7f?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD53GRUB60-DZ9YvQ9NaG-LySvMdcC2QJzI) for the **MBZUAI K2Think** challenge and asked if anyone wanted to team up via DMs.
   - The post highlights the challenge, but provides no additional context.
- **Quest for OpenAI Embedding Model Dataset**: A member inquired whether the dataset used to train **OpenAI's `text-embedding-3-large`** embedding model is publicly available.
   - No response was given in the provided context.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1430171244233359441)** (2 messages): 

> `nanochat course, VLM training memory usage, MPS memory limit` 


- **Nanochat Course on the Horizon?**: A member inquired about the possibility of a **nanochat course** being offered, expressing some confusion about existing materials.
   - No definitive answer was given, but the inquiry suggests interest in more structured guidance on the topic.
- **VLM Training Swaps Memory!**: A member reported that while training the **VLM exercise locally**, it's using a large amount of swap memory (**62GB claimed** and **~430GB virtual memory**).
   - The swap usage is causing slowdowns, highlighting a need for optimization.
- **Limiting MPS Memory for Mac Training**: The same member asked if there's a way to limit memory usage specifically for **MPS** (Metal Performance Shaders) on Macs.
   - The goal is to enable training within a more reasonable **40GB VRAM** limit instead of excessive swap usage.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1429908855609098362)** (111 messages🔥🔥): 

> `lmstudio and llama.cpp, AGI, GPT-OSS Reasoning Effort, DeepSeek OCR Support, LM Studio Server Mode` 


- **LM Studio struggles with llama.cpp integration**: Members were wondering *how can I call my own llama.cpp for LM Studio to use* but it is **not yet fully supported**.
   - The [LM Studio docs](https://lmstudio.ai/docs/app/advanced/lm-runtimes) that references this are a broken link.
- **AGI is 10-20 years away says member**: One member forecasted that *in 5 years LLM's will probably have context large enough* and in **10-20 years there will likely be the first AGI's**.
   - Another member suggested he *become a consultant and charge 1000/h*.
- **GPT-OSS Reasoning Requires Metadata**: A member asked about setting reasoning effort in **GPT-OSS** finetunes, and another responded that it *works due to the metadata in the mxfp4 model of gpt-oss, which is why finetunes/ggufs don't have it*.
   - The member offered to make it available before quantizing it to **gguf**.
- **OpenWebUI connects to LM Studio with OpenAI**: One user was trying to connect **OpenWebUI** to **LM Studio** server, and it was suggested to use the **OpenAI** option instead of **OpenAPI**.
   - Members helped troubleshoot, suggesting to put **/v1** in the address or type *models* in especif. openapi using [this huggingface discussion](https://huggingface.co/spaces/huggingchat/chat-ui/discussions/748#68f1519534c92ca5e3f97053).
- **Qwen3 Embedding 8B fixed quants performant with roocode**: A member reported that they *got the newer quants (the fixed ones) of qwen3 embedding 8b working with roocode code indexing*.
   - They find it *a lot more accurate (as in, the confidence score is a lot higher for relevant queries, and a lot less for irrelevant ones)* than what i used before.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1429934151636746270)** (51 messages🔥): 

> `MI50 setup with Windows, Powering GPUs with multiple PSUs, NVIDIA RTX Pro 5000 Blackwell GPU, ML hardware youtuber appreciation, 4G decoding issue` 


- **MI50 setup gets weird on Windows**: Users discussed setting up **MI50 GPUs** on Windows, noting they require a specific configuration with **Radeon ID community drivers** to utilize the full **32GB** of VRAM and are best used with **Vulkan**.
   - It was advised *not to use them with an Nvidia GPU* due to potential compatibility issues unless ROCm support is unneeded.
- **Powering GPUs: PSU Pitfalls Revealed!**: A user shared a cautionary tale about trashing their **MFT** due to heavy CPU overclocking, which led to a discussion on the risks of powering GPUs with separate PSUs without proper synchronization.
   - It's risky to power a GPU from a separate PSU than the one powering the motherboard PCIE, potentially causing issues like *PSU backflowing* or *phantom motherboard powering*, unless the green wires are synced.
- **NVIDIA's RTX Pro 5000 Blackwell GPU Appears**: A member shared a link to a [TechPowerUp article](https://www.techpowerup.com/342059/nvidia-rtx-pro-5000-blackwell-gpu-with-72-gb-gddr7-memory-appears) about **NVIDIA's RTX Pro 5000 Blackwell GPU** featuring **72 GB** of **GDDR7** memory.
   - Enthusiastic users reacted with humor, estimating a price tag of around *$8-10k*.
- **ML Hardware Youtuber Receives Praise**: Members expressed appreciation for a dedicated YouTuber reviewing and benchmarking hardware for machine learning, filling a void in content beyond gaming benchmarks.
   - The youtuber was described as doing *a god send imo* and *like our machine learning Jesus*.
- **Legacy Motherboard Limitation Frustrates MI50 User**: A user encountered an issue with older **B250M-K motherboards** that advertise **4G Decoding**, but cannot physically enable it, preventing the use of an **MI50 GPU**.
   - This resulted in a costly mistake, leading the user to repurpose the boards for hosting bots using smaller models.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1429928724052181122)** (101 messages🔥🔥): 

> `tinygrad eGPUs, Krea AI Realtime, Google AI Studio, Replit $1B Revenue, Fish Audio S1` 


- **TinyGrad Powers Apple Silicon eGPUs**: **Tinygrad** now supports **NVIDIA eGPUs** on **Apple Silicon** via **USB4**, enabling users to run external **RTX 30/40/50-series GPUs** using an ADT-UT3G dock with the `extra/usbgpu/tbgpu` driver and NVK-based `tinymesa` compiler ([source](https://x.com/__tinygrad__/status/1980082660920918045)).
   - With SIP disabled, this setup achieves roughly **3 GB/s PCIe bandwidth**, and future support for **AMD RDNA 2/3/4** and **Windows eGPU** stacks is planned.
- **Krea AI Opens Realtime Video Model**: **Krea AI** released **Krea Realtime**, a **14B** open-source autoregressive text-to-video model distilled from **Wan 2.1**, generating long-form video at **11 fps** on a single **NVIDIA B200** ([source](https://x.com/krea_ai/status/1980358158376988747)).
   - Released weights are on **HuggingFace** under **Apache-2.0**, prompting user inquiries about **ComfyUI** workflows, **RTX 5090** performance, and fine-tuning support.
- **Google AI Studio Teases 'Vibe-Coding' with Gemini**: **Google AI Studio** is launching a new “prompt-to-production” **Gemini** experience after five months of development aiming to make **AI-app building 100× easier** ([source](https://x.com/OfficialLoganK/status/1980435968323907884)).
   - Reactions mixed excitement (requests for mobile app, opt-outs, higher rate limits), feature suggestions (GSuite-only publishing, VS Code plug-in, short browser-agent tasks) and some skepticism about fit vs Gemini 3 expectations; team confirms enterprise-only deployment is already available.
- **Fish Audio S1 Makes Waves**: **Fish Audio** launched **S1**, a text-to-speech model that’s purportedly 1/6 the cost of **ElevenLabs**, touting **20k devs** and **$5M ARR** ([source](https://x.com/hehe6z/status/1980303682932744439)).
   - Users shared instant voice-clone demos, asking about real-time latency (~**500ms**), while founders admitted current limits and promised wider language support + conversational model next.
- **LangChain Bags $125M**: **LangChainAI** secured **$125M** in Series B funding, expanding from an OSS starter kit to offering **LangChain** (agent dev), **LangGraph** (production orchestration), and **LangSmith** (observability/workbench) ([source](https://x.com/sonyatweetybird/status/1980683121399058626)).
   - Users now include **Uber**, **Klarna**, and **LinkedIn**.


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1429932051137564864)** (13 messages🔥): 

> `tinygrad, NVIDIA eGPU, Apple Silicon Macs, RTX 3090, second-hand` 


- **Tinygrad team enables NVIDIA eGPU over USB4 on Apple Silicon Macs**: The tiny corp team announced early public testing of their pure-Python driver that lets **30/40/50-series NVIDIA GPUs** (and **AMD RDNA2-4**) run over any **USB4 eGPU dock** on **Apple-Silicon MacBooks**; users must disable SIP and install their driver + NVK compiler; see [announcement here](https://xcancel.com/__tinygrad__/status/1980082660920918045).
- **Tinygrad: Bandwidth Details**: Tinygrad's bandwidth is **≈2.5 GB/s** out and **3.3 GB/s** in—slower than **PCIe** but sufficient once weights are loaded.
   - **PyTorch access** is possible via tinygrad’s PyTorch frontend or a future CUDA layer; **10- and 20-series** may work with small patches. 
- **Second-hand RTX 3090 Buying/Testing Guide for AI Workloads**: Taha shared lessons learned after buying a used **RTX 3090**: meet seller in person to inspect card, bring a portable eGPU test rig, verify recognition with nvidia-smi, run **memtest_vulkan** for **VRAM integrity**, optionally gpu-burn for compute stress, load a large model and monitor temps **<100 °C**; see [guide here](https://xcancel.com/taha_yssne/status/1960418430655586677).
- **Framework 13 Ryzen laptop + NixOS as test rig**: Conversation reveals the test rig is a **Framework 13 Ryzen laptop** on **NixOS** in **PRIME offload mode**.
   - One user suggested trying tinygrad on their rig since *mine works ootb since I'm on linux*.


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1430044403958808709)** (17 messages🔥): 

> `Fish Audio S1 TTS launch, Sesame iOS TestFlight for conversational agents, Sequoia backs Sesame` 


- **Fish Audio S1 TTS is Born**: Helena celebrated the public launch of **Fish Audio S1**, billed as the most expressive **TTS model** and **6x cheaper than ElevenLabs** with **5M ARR** and **20K active devs**.
   - Users praised **voice-cloning quality** and asked about latency, language support, iOS app, and phoneme control.
- **Sesame Opens iOS TestFlight for Maya and Miles**: After attracting **1M+ users** to its research preview, Sesame is now opening an iOS TestFlight beta for its ultra-realistic voice assistants, **Maya and Miles**.
   - Co-founder Brendan Iribe added that the beta adds search & text features and [Sequoia Capital spotlighted](https://xcancel.com/sequoia/status/1980680087738675329) the partnership.
- **Sequoia Seeds Sesame's Voice-First Vision**: Sequoia Capital announced it is partnering with the Sesame team to usher in *voice as the next great interface shift* with the goal to evolve computers from tools into conversational *thought partners*.
   - Sesame is launching a closed-beta iOS app (sign-up at [sesame.com/beta](https://sesame.com/beta)) featuring expressive AI agents **Maya & Miles**.
- **Sesame Bags $250M Led by Sequoia & Spark**: Sesame announced a **$250M Series B** led by Sequoia & Spark alongside its beta launch.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1429976614413795390)** (8 messages🔥): 

> `AMD web3 cloud, grouped gemms, FlashInfer-Bench self improving systems, NCU wrapper scripts, PyTorch Conference AI Infra panel discussion on GPU kernels` 


- ****AMD** dives into **web3** with cloud solutions**: A member reported watching a talk at the **AMD event**, noting their focus on the "cloud" aspect of **web3**.
   - They cheekily added a *smileforme* emoji, implying some skepticism or amusement at the concept.
- ****FlashInfer-Bench** Aims for Self-Improving Systems with AI**: CMU Catalyst introduced [**FlashInfer-Bench**](https://flashinfer.ai/2025/10/21/flashinfer-bench.html), a workflow for creating self-improving AI systems via agents, featuring standardized signatures for **LLM** serving kernels and integration with **FlashInfer**, **SGLang**, and **vLLM**.
   - The project includes a [blog post](https://flashinfer.ai/2025/10/21/flashinfer-bench.html), [leaderboard](https://bench.flashinfer.ai/) and [GitHub repository](https://github.com/flashinfer-ai/flashinfer-bench) to foster community development and benchmarking.
- **Seeking **NCU Wrapper Scripts** for Fine-Grained Metric Profiling**: A member inquired about **GitHub** repositories containing **NCU wrapper scripts** that enable passing a list of metrics for profiling using the `--metrics` option.
   - Another user suggested leveraging [NVIDIA's Customization Guide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html#section-files) to create custom sections or sets with tailored metrics.
- **AI Infra Panel Ponders **PTX/Assembly** for Kernel Performance**: Attendees of the [PyTorch Conference AI Infra panel discussion on GPU kernels](https://aiinfrasummit2025.sched.com/event/28FoW/panel-discussion-the-ai-kernel-revolution-rethinking-execution-from-the-ground-up-robert-lange-sakanaai-simran-arora-stanford-university-nathan-lambert-allen-institute-moderated-by-mark-saroufim-gpu-mode) noted consensus around using **PTX/assembly** (or abstractions atop them) for critical parts of code to achieve peak kernel performance.
   - The panel suggests avoiding full dependence on compilers alone.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1430083659406381078)** (12 messages🔥): 

> `Double Buffering in Ampere GPUs, Gluon channel, Triton conference at Microsoft, CuPy vs PyTorch GPU Pointer Performance, DLPack Conversion for CuPy and PyTorch` 


- **Triton Conference Attendees Connect**: Multiple members mentioned attending the **Triton conference at Microsoft** in Mountain View and shared a [YouTube link](https://www.youtube.com/live/s30WoZ7lx3w?si=O6aQMCVjKFs2F4qa) to watch the conference online and a [link](https://www.youtube.com/@Triton-openai/streams) to the Triton-openai streams.
- **CuPy vs PyTorch Pointer Performance Showdown**: A member compared the performance of a simple **MatMul Kernel** with **CuPy** and **PyTorch** GPU pointers, noting a significant performance difference.
   - They observed a *huge performance delta* even when using **DLPack** to convert between **CuPy** arrays and **PyTorch** tensors, questioning if there's an inherent reason for this disparity, but shared a [screenshot](https://cdn.discordapp.com/attachments/1189607595451895918/1430323860464734360/Screenshot_from_2025-10-21_17-32-52.png?ex=68f95c66&is=68f80ae6&hm=88739dd024314bc593c58497cfddaf684e79a0ce7bfdaef32ec3a6d08812df9a&) of the performance difference.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1430180120974462998)** (2 messages): 

> `WGMMA barriers, WGMMA serialization, PTXAS compiler options` 


- **WGMMA calls interrupted by compiler barriers**: A user questioned why the compiler is inserting barriers between calls to **WGMMA**, wondering if it's due to using the same accumulator for all calls.
   - Another user suggested that this occurs when the compiler serializes **WGMMA** instructions due to various reasons.
- **WGMMA Serialisation Warning is MIA**: A user suggested checking for compiler warnings about **WGMMA** instruction serialization, which might provide debugging hints.
   - The user noted that these warnings may only appear when compiling with the `--ptxas-options=-v` option.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1429941950919741624)** (7 messages): 

> `TorchTitan pretraining slowdown, H200x Bare Metal Instance, ProcessGroupNCCL stream usage, CUDAStreamGuard, NCCL kernels` 


- ****TorchTitan Training Sees Slow Iterations****: A user reported experiencing frequent slow iterations during **TorchTitan** pretraining on a single **H200x** bare metal instance, narrowing it down to the active thread/process being descheduled from a CPU for a few seconds based on an **nsys trace**.
   - Despite ensuring no CPU oversubscription, adequate temps, and no power limit issues, the user suspects an **OS/kernel setting** is interfering with process scheduling.
- ****PG NCCL Uses Internal Streams by Default****: When a `CUDAStreamGuard` is set and an NCCL op is called via `ProcessGroupNCCL`, the **NCCL kernels** run on PG-NCCL’s internal streams, typically using one stream per device with high priority, and using the tensor lifetime stream.
   - The [relevant code](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L3132) shows stream syncing on tensor lifetime stream.
- ****'Wait()' Primarily Calls an Event Sync on Current Stream****: Calling `wait()` mainly invokes an event sync on the current stream, creating a dependency on the current stream without blocking the CPU but ensuring expected behavior of the output tensors.
   - The `SynchronizeStream` function [waits on previous cuda events](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L780) without an explicit `cudaStreamSynchronize` or `deviceSynchronize`.
- ****NCCL Stream Dependency on 'CUDAStreamGuard'****: Setting a `CUDAStreamGuard` determines which stream the **NCCL stream** waits on, establishing an incoming dependency, as seen in the [pytorch source code](https://github.com/pytorch/pytorch/blob/03f3f7899cbe4276d02379575c74f439b47668ce/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L803).
   - *`wait` is not blocking on the CPU, all events are marked with cudaEvent, which doesn't require an explicit cudaStreamSynchronize or deviceSynchronize (which is bad for overlap, you really don't want CPU to get blocked, it will just keep firing kernels on the other compute stream while comm is happening)*


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1430206251333320744)** (3 messages): 

> `SLNG.AI, Speech Model Performance Engineer, vLLM, sglang, Susquehanna International Group` 


- ****SLNG.AI** is hiring Speech Model Performance Engineer**: SLNG.AI is building the backbone for real-time speech AI and is looking for a **Speech Model Performance Engineer** with a strong background in software engineering, more details [here](https://isla-house.notion.site/Build-the-Backbone-of-Real-Time-Voice-AI-Join-SLNG-as-Founding-Speech-Model-Performance-Engineer-2642fa00c69d8072bf2fd9047f1b0b68).
- **Looking for inference performance specialist with **vLLM** and **sglang** experience**: A member is looking for **inference performance specialist** to focus on **vLLM** and **sglang** to create unique alpha in production, DM for more info.
- ****Susquehanna International Group** is hiring**: **Susquehanna International Group (SIG)**, a quantitative trading firm, is hiring across many roles which you can see [here](https://sig.com/careers/quant/).
   - Interested members can DM or schedule a chat at the PyTorch conference [here](https://calendly.com/jacob-baumbach-sig/pytorch-2025).


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1430145468704358430)** (2 messages): 

> `CUDA learning, HPC learning, 5090 vs cloud GPU, Cloud GPU rental` 


- **5090 or Cloud: Paths to CUDA Prowess?**: A member is pondering whether to purchase a **5090 GPU** or rent a cheaper option in the cloud to learn **CUDA/HPC** with the goal of eventually becoming an expert.
   - They are also questioning how seriously they need to commit to fully leverage a **5090** versus the alternative of renting a **cloud GPU**.
- **Local Power vs. Cloud Flex for CUDA Dev?**: Someone's weighing the options: buying a **5090** for local muscle or flexing with cheaper cloud GPUs to master **CUDA/HPC**.
   - The core question: how deep do you dive to truly max out that **5090**, compared to just renting cloud time?


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1429915296277205264)** (2 messages): 

> `sglang, ModuleFqnToConfig, torchao_utils.py` 


- **TorchAO Refactor in SGLang?**: A member asked whether [this part](https://github.com/sgl-project/sglang/blob/184a4df697ed75805ac10146dd93e75f1fc609a7/python/sglang/srt/layers/torchao_utils.py#L42) of **sglang** is used.
   - It is used, but the team is moving away from this, ideally refactoring to use **ModuleFqnToConfig**, with more details at [pytorch/ao#3083](https://github.com/pytorch/ao/pull/3083).
- **TorchAO Refactor in SGLang v2?**: A member asked whether [this part](https://github.com/sgl-project/sglang/blob/184a4df697ed75805ac10146dd93e75f1fc609a7/python/sglang/srt/layers/torchao_utils.py#L42) of **sglang** is used.
   - It is used, but the team is moving away from this, ideally refactoring to use **ModuleFqnToConfig**, with more details at [pytorch/ao#3083](https://github.com/pytorch/ao/pull/3083).


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

erichallahan: legendary
  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1430015202664906853)** (2 messages): 

> `OC meetup` 


- **OC member acknowledges their presence**: A member indicated that they are also located in **Orange County (OC)**.
   - This acknowledgment suggests a potential for local meetups or collaborations within the **OC** area.
- **Another member confirms OC location**: Another member chimed in to confirm they are also in **OC**.
   - This further strengthens the possibility of organizing an in-person meetup for members in the **Orange County** region.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1430121696588861510)** (3 messages): 

> `GPT-OSS-20B Architecture, DeepSeek-OCR on L4/T4 GPU` 


- **GPT-OSS-20B Coded From Scratch**: A member implemented OpenAI's **GPT-OSS-20B** architecture from scratch in PyTorch, running on a single **A100 SXM (80GB)**.
   - The implementation includes components like **RoPE with YaRN + NTK-by-parts**, **RMSNorm**, **SwiGLU**, **MoE**, **GQA**, learned sinks, banded attention, and KV caching, with [detailed documentation available on GitHub](https://lnkd.in/eTTrZBeS).
- **DeepSeek-OCR on L4/T4**: A member shared a resource for running **DeepSeek-OCR** on **L4/T4 GPU** with >16 GB VRAM, available at [this GitHub repository](https://github.com/dwani-ai/llm-recipes/tree/main/tutorials/deepseek-ocr).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1430020472342319205)** (3 messages): 

> `LLM Kernel Generation, LLM Bottleneck Identification, Profiler vs LLM` 


- **LLM Kernel Generation vs Bottleneck ID**: A member posed the question of whether an **LLM** that can generate **kernels** or one that can identify **bottlenecks at runtime** would be more useful.
- **Actionable insights from Profiler Logs using LLM**: A member suggested that the utility of an **LLM** lies in turning the often overwhelming **profiler logs** and **metrics** into actionable insights.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1430211993834491916)** (1 messages): 

> `Leaderboard sort_v2, L4 performance, B200 performance, H100 performance, A100 performance` 


- **Sorting algorithm dominates leaderboard**: A submission by <@1416432485591421070> achieved **first place** on the `sort_v2` leaderboard across multiple hardware configurations.
   - The winning times were **52.6 ms on L4**, **8.68 ms on B200**, **6.58 ms on H100**, and **16.4 ms on A100**.
- **Sorting algorithm reigns supreme on diverse hardware**: The winning implementation of `sort_v2` shows performance across varied GPU architectures.
   - The impressive showings suggest optimized routines for different compute capabilities of **L4**, **B200**, **H100**, and **A100**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1430103739691434026)** (25 messages🔥): 

> `MultiAgent Factorio AI Modification, Inspect Framework Evaluation Logic, GymAgent Implementation, MCP Server and Claude Code Integration, AI VTuber Project` 


- **Factorio AI Transformation: From MultiAgent to Solo Act**: A member inquired about the difficulty of modifying the **MultiAgent Factorio AI** to a **single-agent** system, aiming to provide outputs for another model to explain its actions for an **AI VTuber project**.
   - The suggestion involves modifying an agent implementation to write the latest step or the entire history to another model in realtime, turning game play into a commentary.
- **Inspect Framework Enhances Evaluation Prowess**: Progress has been made to improve the evaluation logic using the **Inspect framework**, allowing for the execution and collection of scores across an `eval-set` of tasks and models.
   - The command `fle inspect-eval --eval-set --max-connections 16 --max-tasks 16 --model openrouter/openai/gpt-5-mini --view --view-port 8090` allows for parallel simulation across 16 FLE servers, and it was suggested that results be stored in S3 for shared access.
- **GymAgent Architecture Laid Bare**: The `GymAgent` implementation was recommended as a starting point, with an example of taking the generated code and passing it to a lightweight **LLM** to summarize into a commentary.
   - The **GymAgent** functions as an **Action->Response** agent, observing the environment at every turn and reasoning in natural language before writing code.
- **MCP Server Hooks into AI VTuber**: Integration of the **MCP server** with **Claude Code** was proposed, leveraging **Claude Code's** support for hooks to handle summarization, with the **MCP server** also capable of being plugged into **Docker** and used with **n8n** for managing **LLM** functionality for the **AI VTuber**.
   - The **MCP server** is preferred for its active support and its ability to manage execution independently, allowing external tools like **N8N** to manage AI model calling.
- **AI Discord Bot Embarks on Privacy Quest**: One member mentioned working on an **AI Discord bot** with a privacy-centered global memory and a global emotional state engine based on the **Plutchik wheel**, among other capabilities.
   - They jestingly noted their penchant for undertaking *interesting* projects.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1429987537853743154)** (8 messages🔥): 

> `PTX Compiler Tool, CuTe Kernels for PyTorch, CUTLASS example, Thread-Value Layouts` 


- **Semiring Speed Boost via PTX Injection**: A user created a tool to generate **PTX kernels** from annotated **CUDA CuTe kernels**, achieving a **26x speedup** compared to compiling directly from CUDA.
   - An example of bulk compiling random PTX kernels on all cores is available at [MetaMachines/mm-ptx](https://github.com/MetaMachines/mm-ptx/blob/master/examples/stack_ptx_inject/README.md#02_bulk_rand_gemm).
- **CuTe Kernels Boost PyTorch Tensors**: A user introduced [MetaMachines/mm-kermac-py](https://github.com/MetaMachines/mm-kermac-py), a Python example that exposes **CuTe kernels** as arbitrary **semiring tensor routines for PyTorch**.
   - Another user warned that this approach may not be officially supported and debugging or fixing performance issues may not receive official support.
- **CUTLASS Code Build Example**: A user posted [leimao/CUTLASS-Examples](https://github.com/leimao/CUTLASS-Examples) and another user said *this is good starter example on how to use cmake to build simple cutlass code*.
   - One user asked *Value 0 is copied by multiple threads ??* with an attached image of the code, and another explained that *this actually shows two inverse Thread-Value layouts; they map data coordinates (0..17 x 0..7) to (Thread, Value) coordinates. T32V0 means the 0th data item from thread 32's POV.*


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1430289283943501865)** (12 messages🔥): 

> `Mojo language, Modular, GPU Algorithms, Apple Silicon limitations, DGX` 


- **Mojo and Modular are Insane!**: One member expressed excitement about the goals of **Modular** and the **Mojo** language, stating that it *would be amazing to see if they succeed*.
   - That same member spent **2-3 hours** to complete the first **8 problems** using an **apple silicon machine**.
- **GPU Algorithms problems go in depth**: One member hopes the next few problems about **GPU algorithms** go a bit more in depth, basically completing some basic kernel each time.
   - Another member mentions that problems **25-34** are looking pretty cool, but they cannot do them on their computer, jokingly suggesting they need a **DGX**.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1430018815575457833)** (5 messages): 

> `Iris GPU Native APIs in Triton, Gluon Backend, RDMA implementation in Triton, Multi-planar Ethernet Topologies, Debugging Multi-Node Training Loops` 


- **Iris GPU-Native APIs Arrive in Triton**: The creators of Iris announced its release with the goal to design **GPU-native APIs** that feel natural inside **Triton kernels** and to implement everything directly in Triton for full compiler visibility and optimization, more information can be found in the [Gluon backend documentation](https://rocm.github.io/iris/reference/gluon/overview.html).
- **RDMA Support Coming Soon to Iris**: **RDMA** support is coming soon, with a proxy thread implementation and IBGDA in the future, and all device-side code will be in Triton, with APIs remaining consistent between RMA and RDMA.
- **Multi-Planar Ethernet Topologies info sought**: A user is seeking resources on **multi-planar ethernet topologies**, specifically implementation details such as how packet spraying is enabled, intra-plane failure monitoring, and host-side setup for simultaneous use.
   - They are looking for practical guidance beyond theoretical discussions to implement it.
- **Megatron Training Loop Randomly Freezes**: A user is experiencing random freezes in a **multi-node training loop** using **Megatron**, with iterations occasionally taking 200s instead of the normal 1s.
   - They are unsure where to begin debugging, considering the addition of `torch dist barrier()` and seeking advice on its placement to fix it.


  

---


### **GPU MODE ▷ #[irl-accel-hackathon](https://discord.com/channels/1189498204333543425/1416087968933740688/1430215731491901501)** (6 messages): 

> `DFS in OpenCL, Distributed training, Kernel generation, Synthetic data, Pipeline parallelism` 


- **Plans DFS Implementation in OpenCL**: A member is planning to implement **DFS** in **OpenCL** and will post updates in the channel, also looking for a team.
   - They are also interested in **distributed training**, **kernel generation**, and **synthetic data**.
- **Teaming up for Pipeline Parallelism**: A member proposed implementing **pipeline parallelism** with **synthetic training data** to another member seeking a team.
   - The aim is to leverage the hackathon for collaborative learning and project development.
- **Burn Baby Burn: Porting Qwen 3**: A member is going to the IRL hackathon and wants to port **Qwen 3** to [Burn](https://burn.dev/) and compile a **0.6B variant** into a single [mega kernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17).
   - They are using the hackathon to meet more advanced developers, learn about **GPU programming** and assess **Burn's** viability for serious work.
- **Rustacean Seeks Kernel Collabs**: A member proficient in **Rust** but inexperienced with **kernels** is seeking a hackathon team to work on **IO/communication related** tasks.
   - They are interested in projects involving **KV/weight transfers** or **disk-based KV cache**.


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1429950858895949954)** (4 messages): 

> `Helion 0.2 Public Beta Release, Triton Compilation/MLIR Errors, Helion as Triton Compiler Fuzzer` 


- **Helion 0.2 Enters Public Beta**: The initial release of **Helion 0.2** is now available as a public beta [on pypi](https://pypi.org/project/helion/0.2.0/).
   - Helion is a *tile abstraction* that interfaces with the Triton compiler.
- **MLIR Errors Plague Helion Optimization**: During **Helion optimization passes**, Triton compilation and MLIR errors sometimes occur for specific configurations.
   - The assertion failure originates from `/project/lib/Dialect/TritonGPU/Transforms/OptimizeThreadLocality.cpp`, during TritonGPUOptimizeThreadLocalityPass.
- **Helion: Triton Compiler's Greatest Fuzzer**: According to members, **Helion** acts like a great fuzzer for the Triton compiler, exposing bugs frequently.
   - The autotuner is instructed to **skip or ignore such configurations** when these errors occur, since this is considered normal.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1429910042378895482)** (72 messages🔥🔥): 

> `ChatGPT mock interviews, IntelliCode Context, GPT-5 coding UX, ML Paper tips, RL influence on LLMs` 


- **ChatGPT gets quizzed on Mock Interviews**: A user experimented with **ChatGPT** for mock interviews and wondered about the accuracy of its responses from an expert's point of view, sharing the [conversation link](https://chatgpt.com/share/68f68935-3148-8005-907f-86ec2ed6e93c).
- **IntelliCode Impresses with Context-Aware Completion**: A member was impressed with **Microsoft's IntelliCode** in Visual Studio, an AI-powered code completion tool that correctly predicts entire method bodies by leveraging a lot of context like all the classes of your project, the files you have open, and the lines of code immediately before and after the caret.
   - The member felt that it was *almost like it's reading your mind* when it works well.
- **GPT-5 coding UX rated subpar**: Members discuss coding UX, one preferring **gpt-5-codex** and **sonnet-4.5** over **GPT-5**.
   - One member complained that *whatever they are doing is highly intransparent*, which might be ok for vibe coding, but not for caring about actual implementation.
- **Solo ML Paper Writers Ask For Writing Advice**: A member requested tips for writing decent quality papers for **NeurIPS** while grinding it out solo, linking a [Google doc on paper format](https://docs.google.com/document/d/16R1E2ExKUCP5SlXWHr-KzbVDx9DBUclra-EbU8IB-iE/edit?tab=t.0#heading=h.16t67gkeu9dx) and a [YouTube video on scientific writing](https://www.youtube.com/watch?v=jLPCdDp_LE0).
   - The author is targetting a broad audience outside of ML, finding that loss and **FID** are unreliable indicators, requiring new sampling methods like the one in this [paper](https://arxiv.org/abs/2310.11232).
- **RL Impacts LLM Capabilities**: A member feels that **RL** is negatively impacting **LLMs**, particularly how **OpenAI** models reiterate irrelevant information at a low level of abstraction, which may be reducing the diversity of answers.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1429931689475444807)** (22 messages🔥): 

> `VR invite, Transformer Circuits, Deepseek discoveries, Reinforcement Learning` 


- **VR Enthusiasts Connect on 'Effective Autism' Server**: A member invited someone who enjoys **VR** to their *'effective autism'* server, expanding their community of like-minded enthusiasts.
   - Another member welcomed them, suggesting potential shared interests and discussions.
- **Transformer Circuits Postponed**: Due to being bogged down with work, a member postponed reviewing the new [transformer circuits post](https://transformer-circuits.pub/2025/linebreaks/index.html) until the next day.
   - The post discusses details of **transformer circuits**.
- **Karpathy's Tweet Sparks Paper Interest**: A member shared [Karpathy's tweet](https://x.com/karpathy/status/1980397031542989305?t=RYS1muyomGCPvv6bhISgRg&s=19), causing discussion about a paper's perceived importance.
   - Some joked about the automatic conclusion that any paper promoted by **Karpathy** must be the best, while another member defended the paper, highlighting its **framework implications**.
- **Deepseek's Discoveries Evokes Western OS envy**: A member expressed that they wish the West had better **OS labs** because **Deepseek** is always coming out of nowhere with these discoveries.
   - They noted open source weights are a small part of value, compared to open source **data collection**, **methods**, and **training details**.
- **Preference Models Train with Reinforcement Learning Signals**: A member inquired whether reinforcement learning signals were used to train preference models or simply to provide rewards for inductively biasing distribution matching of sharpened target distributions per inference.
   - The conversation alluded to technical aspects of training preference models, including **reinforcement learning** and **distribution matching**.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1430236945195335862)** (1 messages): 

> `HumanAPI, Automation tasks` 


- **HumanAPI creates manual tasks for unsolved problems**: A member is creating a **"HumanAPI"** which creates a task and assigns it to a human if there are no tested automation tasks available when trying to solve a problem.
   - The purpose of this project is to review those human tasks every now and then to see what can be automated.
- **Reviewing Human Tasks for Automation**: The **HumanAPI** project aims to identify tasks suitable for automation by reviewing tasks initially performed by humans.
   - This iterative process allows for continuous improvement and expansion of the automated capabilities of the system.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1429961253480173599)** (6 messages): 

> `DeepSeek OCR, Claude Code on the web, Unitree Robotics, Alibaba Qwen, ChatGPT Atlas` 


- **DeepSeek Eyes OCR Niche**: [DeepSeek-AI released DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) on GitHub, a new player in the OCR space.
   - Meanwhile [Anthropic released Claude Code on the web](https://www.anthropic.com/news/claude-code-on-the-web).
- **Unitree poised to stomp on Tesla?**: A member speculated that **Unitree** is going to dominate the humanoid robotics market.
   - They added that **Elon Musk** probably can't even get the magnets for the actuators at the moment *thanks to the orange dude*.
- **Alibaba Qwen struts stuff**: A member shared a link to [Alibaba's Qwen](https://x.com/alibaba_qwen/status/1980665932625383868?s=46) on X.
   - The discussion also mentioned [ChatGPT Atlas](https://chatgpt.com/atlas).
- **Amazon Vibe checks out of Beta**: Amazon's **Vibe Code IDE** is out of invite-only beta, apparently costing **500 credits**.
   - It, like many AI IDEs, is also a **VSCode fork**.
- **Kiro Code Editor out of Waitlist**: [Kiro Code editor](https://kiro.dev/blog/waitlist-is-over/) is out of waitlist and is designed to be *spec based*.
   - The member adds that **Kiro** works around specifications for features and implementations, rather than solely prompts.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1430385505270497361)** (1 messages): 

> `DSPy for voice generation, Automated judging loop, Optimization for subjective quality, Character analysis pipeline, ElevenLabs` 


- ****Voiceover Mage** taps DSPy for AI NPC voices**: A member built a [voice generation system](https://github.com/Gielinor-Speaks/voiceover-mage) for game NPCs, using **DSPy** to parse wiki content and generate voice prompts for **ElevenLabs**.
   - The goal is to leverage **DSPy's** optimization features to improve the character analysis pipeline and automate voice selection, and is documented in a [devlog style video](https://youtu.be/z3DQm2PiKpo).
- **Optimization for Subjective Voice Quality Coming Soon**: Currently the member manually curates three voice candidates per character, but plans to add an **automated judging loop** to learn what makes a "good" voice match for different character archetypes using DSPy.
   - The member also intends to collect manual selections as training signals to create examples, optimizing toward subjective quality judgments.


  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1430274415668494556)** (2 messages): 

> `DSPy Usage in Research, Paper Code Availability` 


- **DSPy Featured in New ArXiv Paper**: A new paper ([https://arxiv.org/abs/2510.13907v1](https://arxiv.org/abs/2510.13907v1)) utilizes **DSPy** in its research, signaling growing adoption within the academic community.
- **Paper Code Still Under Wraps**: Although the paper mentions the use of **DSPy**, the corresponding code repository is not yet publicly available.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1429966872849416273)** (74 messages🔥🔥): 

> `inspect_history() placement, Adapters in DSPy, Module-level History Access, ReAct Trajectories, Trace vs. DSPy` 


- ****History Location Debated****: Members debated why `inspect_history()` is a method in `dspy` rather than a module object, with concerns about accessing prompts in compound modules, but it was clarified that `dspy.inspect_history()` is more for global history and individual programs also track history.
   - One member pointed out it can be accessed with `predictor.history` if `dspy.configure(track_usage=True)` is set, but some still found this confusing.
- ****Adaptor Agony Averted****: The discussion covered using adapters in DSPy, with an example showing how to use `dspy.context` to apply a single adapter, and the user can track usage with `dspy.configure(track_usage=True)`.
   - A member gave an example of setting it up with `with dspy.context(lm=single_call_lm, adaptor=single_adaptor):` to further clarify.
- ****Trajectory Talk Takes Turn****: Members discussed trajectories in DSPy, clarifying that they are more of a ReAct concept (input, thought, tool call, action etc), with emphasis on DSPy primarily dealing with strings.
   - It was mentioned that Interrupt really is just asking it to stop generating, by closing of connection for streaming, and it's application side.
- ****Trace Triumphs?****: A member asked for a comparison between [Microsoft Trace](https://microsoft.github.io/Trace/) and DSPy, with another noting that Trace claims an **8% accuracy increase** over DSPy and appears more token efficient.
   - One member mentioned they would try it out to give a fair comparison, although they will probably still feel like they have more granular control with DSPy.
- ****Module Magic Manuevers****: A member had questions about refining a DSPy module and needing a specific number of answers, and suggested wrapping the logic in a module with an assertion.
   - Members mentioned setting `num_retries=N` at the LM level, with the refine then taking the `self.evaluator`, but if the program fails, it'll just retry the program, so it won't run forever.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1429953622052962396)** (35 messages🔥): 

> `Discord Server Badges, EleutherAI Stock Symbol?, New members introduce themselves` 


- **Discord Server Badges Spark Debate**: Members discussed the possibility of adding a server badge, similar to a role icon, and how a server tag might broadcast the server too widely, potentially increasing the moderation load for EAI staff, referencing [this screenshot](https://cdn.discordapp.com/attachments/729741769738158194/1429978898229100605/Screenshot_2025-10-20_at_7.45.09_PM.png?ex=68f96ca1&is=68f81b21&hm=e032ead2cf427352fba72fdba46d77407a4d2bd71ddc4b60a1c2b6aa04cb8980&).
   - One member noted, *"making a tag is cool but that is in a way broadcasting this server everywhere else, eai staff already gets too many people here to moderate."*
- **Jokes about EleutherAI IPO**: Following a question about whether a particular stock symbol was available, a member jokingly asked, *"what will **Eleuther's NYSE stock symbol** be?"*
   - Another member responded, *"I think you misunderstand the purpose of being a non-profit,"* implying that EleutherAI, as a non-profit organization, would not be publicly traded.
- **New faces join the chat**: A computer vision engineer and a data scientist / ML engineer working in finance introduced themselves to the channel, hoping to develop collaborations.
   - The ML engineer mentioned current projects on **RL** and **conformal inference on LLMs**, inviting others to reach out and learn together.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1429916699377008785)** (10 messages🔥): 

> `Kimi k2 attention clipping, Normuon vs Muon Optimizers, Weight distribution smoothness, AGI Definition benchmarks` 


- ****Kimi K2's Attention**: Clipping Creates Controversy**: Discussion arose around **Kimi k2** needing to clip attention, potentially linked to optimizer behavior where **muon** encourages a better condition number but spikier weight distributions.
   - It was suggested that if **normuon** performs as well as **muon** in large tests, a smoother weight distribution might be inherently desirable for stability.
- ****Normuon's Triumph**: Logit Blowup Prevention**: A member noted that **normuon** beating **muon** even with **qk-norm** (which avoids logit blowup) in their baseline suggests logit blowup prevention might not fully explain the performance parity.
   - It was posited that updates without clipping increase the spectral rank of weights, directly leading to logit blowups, making large-scale validation against **normuon** interesting.
- ****Smoother Weights**: Distribution Debate Starts**: Concerns were raised that a smoother weight update distribution does not necessarily equate to a smoother weight distribution.
   - One member agreed that a smoother distribution could be a *free lunch*.
- ****AGI Definition**: Benchmarks Beckon**: A member shared a [link to Dan Hendrycks' AGI Definition benchmarks](https://agidefinition.ai/paper) and asked how fast they would be benchmarked.
   - Another member predicted multimodality would likely be covered in **1-2 years**, with speed coming from mini versions of models.
- ****Continual Learning**: Criteria Criticized**: A member expressed that *the continual learning thing benchmark criteria is arbitrary and arguably extremely stupid*.
   - They predicted **85%** is very likely in **1-2 years**, and **90%** likely as well.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1429951644275179691)** (44 messages🔥): 

> `Cloudflare issues, Open Sourcing Older Models, Credit Payment Issues, Chat Delays, Moderation Complaints` 


- **Cloudflare Troubles for Manus Users**: A user reported experiencing Cloudflare security issues when visiting most websites using Manus.
   - There was also a suggestion for the Manus team to consider open-sourcing some of their older models, although it's unclear if related.
- **Payment Problems plague platform**: A user reported issues when trying to pay for credits via a web browser, receiving jumbled code and being unable to complete the transaction.
   - The user claims this is a known issue and that they contacted support, while **lucia_ly** asked for their email address to follow up.
- **Chat Slowdowns Irk Users**: A user reported excessive delays in chat processing, specifically when translating long Japanese chapters into English, despite loving Manus's speed normally.
   - The user noted, *"this morning, I put one chapter and the ai is still thinking. What happened?"*
- **Pro Plan Credit Cap Confusion Continues**: Users are reporting conflicting information about **unlimited credits** on the **Pro plan**, with the help system and iOS upgrade page stating it is unlimited, while the PC upgrade page indicates a high limit.
   - One user with **11k credits** remaining was concerned about what happens after depletion, and another user suggested that they should participate in "various opportunities to help improve Manus, as they always give free credits for your time".
- **Scam Alert issued to Users**: A user was accused of being a "fraudster scammer" asking for people's login access to their accounts to do their "fucking law school exam research".
   - Another user suggests that the supposed fraudster "wont make another account or pay $20/month and complains its like tomorrow and begging to get ur EMAIL PASSWORD for a PAID ACCOUNT To probably steal ur personal info and bank info".


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1429907849013628990)** (25 messages🔥): 

> `China A.I. Competition, Decentralized A.I., Nous Research, AWS Cloud, Sora` 


- **China's A.I. Spartan Competition benefits the globe**: A member believes that China's insane spartan involution competition in A.I. is great for the A.I. space because it democratizes access to advanced models and destroys monopolies.
   - They also state that the rate of advancement in OS model development means that 2026 should bring us OS models reaching **100% high intelligence with 90% lower cost**, destroying monoplist ambition.
- **NousCon Virtual Attendance in Question**: A member inquired about virtual attendance for this year's **NousCon**, but also expressed disappointment at missing **egg irl**.
   - Another member said that it took over a week to find a good deal because flights and hotels are expensive.
- **Nous Research promoted as Decentralize A.I.**: A member notes that **Nous Research** is promoted as Decentralize A.I. and hopes the team will resolve issues with centralization.
   - Another member stated they are more focused on the democratization of A.I models for the masses.
- **Nous Successfully Decentralizes with Open Source Methodologies**: A member stated that **Nous** successfully decentralizes with their open source methodologies and infrastructure implementations.
   - They added that **Psyche** was what initially introduced them to Nous, and linked to the [Nous Psyche page](https://nousresearch.com/nous-psyche) and a [Stanford paper on centralization](https://cs.stanford.edu/~gakiwate/papers/sigcomm25-centralization.pdf).
- **Sora AI project gets showcased**: A member showcased a video creation with **Sora**.
   - The attached video was [20251022_0850_01k850gmktfant3bs18n3hbd79.mp4](https://cdn.discordapp.com/attachments/1149866623109439599/1430403237084659774/20251022_0850_01k850gmktfant3bs18n3hbd79.mp4?ex=68f9a653&is=68f854d3&hm=97c310cb6dcc58adf80207392b33e468cc966babf5a88f65261489840b5b68c3&).


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1430338693037691073)** (2 messages): 

> `Microsoft Trace` 


- **Microsoft Trace Utility Shared**: A member shared a link to the [Microsoft Trace utility](https://microsoft.github.io/Trace/).
   - The member noted that *apparently it's not all that new*.
- **Microsoft Trace: A blast from the past**: The [Microsoft Trace utility](https://microsoft.github.io/Trace/) resurfaces, sparking interest.
   - Its features and capabilities are being re-evaluated in light of current development practices.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1429939290183438336)** (16 messages🔥): 

> `Nvidia macOS drivers, GLSL renderer, clspv bugs, Vulkan sine accuracy` 


- **Nvidia drivers ported to macOS**: Madlads achieved the unthinkable by creating **Nvidia drivers for macOS**.
- **GLSL renderer progresses**: A member has been writing a **GLSL renderer** that now passes most of the tests, available [on GitHub](https://github.com/softcookiepp/tinygrad/blob/master/tinygrad/renderer/glsl.py).
- **Vulkan backend status update**: Almost all tests now pass with a custom backend and **clspv**, but only with `-O0 --cl-opt-disable` to circumvent numerous **clspv bugs**.
- **clspv optimization issues**: The member reported *much more miscompilations from clspv if optimizations aren't disabled*.
- **Vulkan Sine Isn't Accurate**: The poster mentioned that **Vulkan's sine function** isn't as accurate, requiring a custom implementation which would impact performance.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1429962916987408436)** (5 messages): 

> `Gradient Accumulation, Backward Call Multiplicity, TinyJit Gradient Addition` 


- **Gradients Add Up with Multiple Backward Calls**: A member inquired whether calling `backward` multiple times before calling `optimizer.step` simply adds the gradient contributions.
   - The member confirmed that the gradients indeed add up.
- **TinyJit's Gradient Accumulation**: A member reported running into issues with gradient accumulation and fixed it by setting `reduction=sum` and manually counting non-padding tokens.
   - They also performed `backward` on each microbatch, divided the gradients, and used assign.
- **Doubts Raised on Math in mlperf Model Training**: A member questioned the correctness of the math in the [mlperf model training script](https://github.com/tinygrad/tinygrad/blob/c7c59e6dd71158f50bbb9a87298b4ed1d65a6fb6/examples/mlperf/model_train.py#L1375C1-L1390C54), specifically regarding the scaling with `grad_accum`.
- **Gradient Accumulation's TinyJit Fix**: A member reported that gradient accumulation was broken in **TinyJit** a couple months ago.
   - They fixed it by rewriting the gradient addition step to use an assign.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1430084733429350422)** (10 messages🔥): 

> `Karpathy Controversy, Kimi support` 


- **Karpathy Critics Spark Bubble Alarm**: A member suggests that the mockery of **Karpathy** on X indicates a potential valuation bubble for American frontier AI labs, referencing [this X post](https://x.com/nathanlands/status/1980035861019533749?s=46).
   - The post includes a chart presumably **mocking Karpathy**, with no additional context provided by the original poster.
- **Kimi K-2 Support Status Questioned**: A member expressed concern about the lack of support for **Kimi**, reporting *zero* response from the support team.
   - Other members clarified that the channel is not a support server and suggested DMing a specific user, while also asking for details about the issue and which email was used for the bug report.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1429921878784479425)** (7 messages): 

> `Python Familiarity, matmul optimization, hardware-optimized matmuls, Mojo's Missing Features` 


- **Python Helps Discover Mojo**: A member noted that familiarity with **Python** can aid in the discoverability of **Mojo**.
   - Another member cautioned that discrepancies between **Mojo** and **Python** could lead to confusion.
- **Hand-Tuning Matmul Beats Compiler Optimization**: A member inquired why **matmul optimizations** aren't integrated into the compiler.
   - Another member responded that while optimizing compilers have their place, human intervention is often preferable for **hot-path code** to fine-tune it for specific hardware, pointing to [Mojo's open-source, hardware-optimized matmuls](https://github.com/modular/modular/tree/main/max/kernels/src/linalg/matmul).
- **Kernel Writers Free of Compiler**: A member explained that moving optimizations out of the compiler allows more individuals (kernel writers) to contribute enhancements, rather than relying solely on **compiler engineers**.
   - They added that **compiler engineers** are better utilized on tasks that benefit the entire ecosystem rather than niche improvements like *a 1% boost to matmuls where one dimension is less than 64*.
- **Type System tops Mojo Wishlist**: Asked about the most important missing feature in Mojo, a member identified *a finished type system* as the priority.
   - They followed with a list of other desirable features including *rounding out the standard library datatypes, proper IO, a good async runtime, an effect system, static reflection, compiler plugins, the ability to deal with more restrictive targets, cluster compute, device/cluster modeling, and some clone of Erlang's OTP*.


  

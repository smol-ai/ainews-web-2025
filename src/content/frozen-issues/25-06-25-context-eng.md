---
id: MjAyNS0w
title: 'Context Engineering: Much More than Prompts'
date: '2025-06-25T05:44:39.731046Z'
description: >-
  **Context Engineering** emerges as a significant trend in AI, highlighted by
  experts like **Andrej Karpathy**, **Walden Yan** from **Cognition**, and
  **Tobi Lutke**. It involves managing an LLM's context window with the right
  mix of prompts, retrieval, tools, and state to optimize performance, going
  beyond traditional prompt engineering. **LangChain** and its tool
  **LangGraph** are noted for advancing this approach. Additionally, **OpenAI**
  has launched **ChatGPT connectors** for platforms like **Google Drive**,
  **Dropbox**, **SharePoint**, and **Box**, enhancing context integration for
  Pro users. Other notable news includes the launch of **Vercel Sandbox**,
  **Cloudflare Containers**, the leak and release of **Gemini Code** by **Google
  DeepMind**, and fundraising efforts by **OpenRouter**.
companies:
  - openai
  - langchain
  - cognition
  - google-deepmind
  - vercel
  - cloudflare
  - openrouter
models:
  - gemini-code
topics:
  - context-engineering
  - retrieval-augmented-generation
  - tools
  - state-management
  - history-management
  - prompt-engineering
  - software-layer
  - chatgpt-connectors
  - api-integration
people:
  - karpathy
  - walden_yan
  - tobi_lutke
  - hwchase17
  - rlancemartin
  - kwindla
  - dex_horthy
---


**Finely crafted context is all you need.**

> AI News for 6/24/2025-6/25/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (220 channels, and 5002 messages) for you. Estimated reading time saved (at 200wpm): 447 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

A lot of relevant news stories to choose from today: the curiously coincidental launches of [Vercel Sandbox](https://x.com/cramforce/status/1937885790152818995?s=46) and [Cloudflare Containers](https://x.com/cloudflare/status/1937544770752045168?s=46), the leak and then release (with generous limits) of [Gemini Code](https://x.com/googleaidevs/status/1937861646082515205), GDM's Claude Code competitor, or the [fundraising of OpenRouter](https://x.com/xanderatallah/status/1937957937692938292).

But very probably the thing that will stick around from today is the confirmation of "Context Engineering" as a noteworthy trend, as coined by either [Dex Horthy](https://x.com/dexhorthy/status/1938265310869721478) or [Cognition's Walden Yan:](https://cognition.ai/blog/dont-build-multi-agents):

![](https://resend-attachments.s3.amazonaws.com/9eE7HoXEVqjxrks)

and promoted by Tobi Lutke last week:

![](https://resend-attachments.s3.amazonaws.com/f5f9LBXplycV5dV)

Lots of people have chimed in in recent days:

- [Harrison](https://x.com/hwchase17/status/1937648042985030145): "we think LangGraph is really great for enabling completely custom context engineering - but we want to make it even better"
- [Lance Martin](https://rlancemartin.github.io/2025/06/23/context_engineering/): "Context enters an LLM in several ways, including prompts (e.g., user instructions), retrieval (e.g., documents), and tool calls (e.g., APIs). Just like RAM, the LLM context window has limited “communication bandwidth” to handle these various sources of context. And just as an operating system curates what fits into a CPU’s RAM, we can think about “context engineering” as packaging and managing the context needed for an LLM to perform a task."
- [Kwindla](https://gist.github.com/kwindla/f755284ef2b14730e1075c2ac803edcf): "If your voice agent needs to follow a series of steps reliably, or will perform conversations longer than a few turns, you will probably need to think about doing "context engineering" to keep the conversation context short and focused.
    
    One useful way to think about context engineering is to design your conversation as a series of workflow states. Each state corresponds to a specific "job to be done" during the voice interaction."
    
- [Andrej](https://x.com/karpathy/status/1937902205765607626): *"When in every industrial-strength LLM app, context engineering is the delicate art and science of filling the context window with just the right information for the next step. Science because doing this right involves task descriptions and explanations, few shot examples, RAG, related (possibly multimodal) data, tools, state and history, compacting... Too little or of the wrong form and the LLM doesn't have the right context for optimal performance. Too much or too irrelevant and the LLM costs might go up and performance might come down. Doing this well is highly non-trivial. And art because of the guiding intuition around LLM psychology of people spirits."*
- [Dex Horthy](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md): Own your Context Window
    
    ![](https://resend-attachments.s3.amazonaws.com/YhUo9VWTXzwwmd3)
    

Immediately, and rightfully so, this has become a must know term and skillset in AI Engineering.

---

# AI Twitter Recap

**AI Development, Tooling & Frameworks**

- **The Rise of "Context Engineering"**: [@karpathy](https://twitter.com/karpathy/status/1937902205765607626) champions the term **"context engineering"** over "prompt engineering," arguing it better describes the complex art of filling an LLM's context window. He details that this involves a non-trivial software layer for managing **RAG**, **tools**, **state**, **history**, and more to achieve optimal performance. Building on this, [@hwchase17](https://twitter.com/hwchase17/status/1937648042985030145) from **LangChain** notes it's a "new hot topic" and proposes using **LangGraph** to streamline context management. [@RLanceMartin](https://twitter.com/hwchase17/status/1937622377267101921) has also written about popular patterns for this process.
- **ChatGPT and OpenAI Launch Major Product Updates**: [@OpenAI](https://twitter.com/OpenAI/status/1937681383448539167) announced that **ChatGPT connectors** for **Google Drive, Dropbox, SharePoint, and Box** are now available to Pro users, allowing them to bring in unique context for work tasks. In a move seen as a response to other code-focused tools, [@corbtt](https://twitter.com/corbtt/status/1937757709241311531) interprets a statement from **Sam Altman** as an announcement that OpenAI's upcoming open-source model will be at the **o3-mini** level.
- **Google Launches Gemini CLI Agent with Generous Free Tier**: **Google** has released the **Gemini CLI**, an open-source (**Apache 2.0**) AI agent for the terminal. The announcement, shared by [@googleaidevs](https://twitter.com/demishassabis/status/1938023045320335789) and others, highlights a powerful free tier of **1,000 requests per day** with a **60 RPM** rate limit, seen by many as a strategic move to drive adoption. The CLI supports tools and **MCP**, with [@osanseviero](https://twitter.com/osanseviero/status/1937861590805872801) sharing the GitHub and blog links. The community reacted with excitement, with some like [@scaling01](https://twitter.com/scaling01/status/1937900212242047384) joking that the generous tier must have been a mistake by a "dyslexic intern." The release has sparked a conversation about a "battle royale" of CLI coding agents, including competitors like **Claude Code**, as noted by [@qtnx_](https://twitter.com/qtnx_/status/1937976300360155544).
- **New Courses and Protocols for Multi-Agent Systems**: **DeepLearningAI** and **Andrew Ng** announced a new course on the **Agent Communication Protocol (ACP)**, developed in partnership with **IBM Research**'s **BeeAI** (@AndrewYNg/status/1937907934094360582). The course teaches how to build agents that can communicate and collaborate across different frameworks using a standardized RESTful interface. Concurrently, the ecosystem around the **Model Context Protocol (MCP)** is growing, with [@lmstudio](https://twitter.com/multimodalart/status/1937917586899144948) adding support for MCP servers and [@llama_index](https://twitter.com/jerryjliu0/status/1937653599972286873) releasing open-source templates for building Claude-compatible MCP servers.
- **DSPy Framework Gains Traction**: The **DSPy** programming framework is gaining significant attention, with **Shopify CEO Tobi Lütke** stating it's his ["context engineering tool of choice"](https://twitter.com/lateinteraction/status/1938005712489083252). Framework creator [@lateinteraction](https://twitter.com/lateinteraction/status/1937701902000480599) clarifies that **DSPy** is a programming model centered on **Signatures** and **Modules**, not just a collection of optimizers. A new course from **Johns Hopkins University** on DSPy was also highlighted (@DSPyOSS/status/1937698576949518351).
- **Advice for Building and Evaluating AI Agents**: In a talk at the [**AI.Engineer](http://ai.engineer/) World Fair**, [@jerryjliu0](https://twitter.com/jerryjliu0/status/1937681047875191159) shared practical steps for building AI agents that automate knowledge work, discussing agent architectures and the importance of well-designed tools. For evaluation, [@HamelHusain](https://twitter.com/HamelHusain/status/1937687931470193102) promoted a guide by [@eugeneyan](https://twitter.com/HamelHusain/status/1937687931470193102) on evals for long-context Q&A systems.

**New Models, Research & Techniques**

- **Google Announces AlphaGenome for DNA Analysis**: **Google DeepMind** and **Google AI** have introduced **AlphaGenome**, a new AI model designed to help scientists better understand DNA by predicting the impact of genetic mutations (@Google/status/1937897003201044534). [@IterIntellectus](https://twitter.com/demishassabis/status/1937971182256435323) described it as an AI that can read **1 million bases of DNA** and predict biological function from sequence alone.
- **Anthropic's Claude is All About the Data**: A recurring theme is the critical role of data quality. [@nrehiew_](https://twitter.com/nrehiew_/status/1937651376013606944) asserts that the "soul" of **Anthropic's Claude** is primarily its training data. This sentiment is echoed by [@cloneofsimo](https://twitter.com/cloneofsimo/status/1937635148784369828), who urges researchers to "STOP LOOKING AT SUBQUADRATIC ATTENTION PAPERS and GET BETTER DATA."
- **Advancements in AI Video and Image Generation**: **Kling AI** announced a **Motion Control** feature that applies motion capture from a source video to a new image (@Kling_ai/status/1937838997730148766). Concurrently, **RunwayML** announced that its **Gen-4 References model** is now available in their API, pushing performance on consistency and personalization (@c_valenzuelab/status/1937878573852811447). Additionally, **OmniGen 2** was released with an **Apache 2.0 license**, praised by [@reach_vb](https://twitter.com/reach_vb/status/1937753850259128719) as "State of the Art in Image edits."
- **New Research on Reasoning, Generation, and Training**: **Sakana AI** shared a video explaining their **Reinforcement Learning Teacher**, a new method for creating reasoning models with smaller teacher models (@SakanaAILabs/status/1937743827177206067). Researchers from **Stanford** and **Google** introduced **Weaver**, a framework to close the "generation-verification gap" where LLMs produce correct answers but fail to select them (@togethercompute/status/1937653446825435411). **Snowflake AI Research** released a paper on **Arctic Long Sequence Training (ALST)**, detailing their methods for training on long sequences (@JayAlammar/status/1937790490092429364).

**Industry News & Company Strategy**

- **Intercom's "Refounding Moment"**: **Intercom**, a **$12B** startup, is undergoing a "refounding moment" to become a full-fledged AI app builder, as highlighted by [@swyx](https://twitter.com/swyx/status/1937748319453024527).
- **AI Transforming Healthcare and Media**: An **Alibaba** AI model that detects **gastric cancer** from routine CT scans has been deployed in **20 hospitals in China**, screening over **78,000 patients** and catching cancers months before symptoms appear, as reported by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1937909094662463866). In media, [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1937615643731272177) shared **Runway's** vision of AI as the "underlying infrastructure" for a new media landscape, comparing the current moment to the invention of the first cameras.
- **AI Exits Trend Towards Acquihires**: A tweet from [@multiply_matrix](https://twitter.com/multiply_matrix/status/1937685737111191835) points out a significant trend where recent major AI startup exits have been acquihires, listing examples like **Adept** to Amazon, **Inflection** to Microsoft, and **MosaicML** to Databricks.
- **Industry Forges Alliances for AI Agent Development**: **Cohere** announced it is a founding participant in the **Stanford DDL Industry-Wide Forum on AI agents**, joining forces with **Meta**, **Oracle**, and **PayPal** to shape responsible development and cross-industry standards (@cohere/status/1937914623753359378).

**Broader Implications & Commentary**

- **The Future of Operating Systems and Browsers is AI**: **Perplexity AI CEO** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1937732846543933569) made a bold claim that **"Android needs to be rebuilt for AI,"** arguing it is currently optimized for Google's ad business rather than for being a "truly agentic OS." He also stated that the **browser is the "primordial soup" where agents will emerge and evolve** (@AravSrinivas/status/1937651271458345028).
- **A Roadmap for Engineers Entering AI**: [@jxmnop](https://twitter.com/jxmnop/status/1937874659980022034) provided a comprehensive guide for developers looking to break into AI. The advice includes picking a specific domain (text, images, audio, robotics) and a specific area (training, inference, data, safety), becoming a "sponge" for information, and executing a single, high-quality project to showcase skills.
- **AI Infrastructure and the Power Grid**: [@dylan522p](https://twitter.com/dylan522p/status/1937943241082437697) raised concerns about the **weakness of the US grid**, warning that a large training run could trigger blackouts and turn public opinion against AI infrastructure.
- **The State of Academic Research**: A **NeurIPS** reviewer, [@jxmnop](https://twitter.com/jxmnop/status/1937949143084810625), shared a frustrating experience with the peer review process, describing submissions that were clearly **LLM-generated**, duplicated, or based on private, unreproducible data. This highlighted growing concerns about the quality and integrity of academic submissions in the AI field.

**Legal & Policy**

- **US Visa Policy Demands Social Media Disclosure**: A new US visa policy change sparked major discussion. The policy, noted by [@rshereme](https://twitter.com/zacharynado/status/1937923326791295078) and the [@USAinUKConsular](https://twitter.com/francoisfleuret/status/1937926540769054772) account, requires all **F, M, or J nonimmigrant visa applicants** to list social media usernames from the past five years and make their profiles public for review.
- **Anthropic Wins Key "Fair Use" Ruling on AI Training**: A federal judge ruled that **Anthropic's** method of training models constitutes **fair use**, a significant legal development for the AI industry. The ruling's reasoning was shared by [@JvNixon](https://twitter.com/JvNixon/status/1937654031130010016), while [@andykonwinski](https://twitter.com/andykonwinski/status/1937739172263141854) pointed to details from the court summary revealing that Anthropic purchased licensed datasets from third-party sources as part of its training process.

**Humor & Memes**

- **Karpathy's Warning**: [@karpathy](https://twitter.com/karpathy/status/1937941695943065640) posted a now-classic line: **"May your regularizer be strong, lest you RLHF to slop."**
- **Parodying Tech Skepticism**: A tweet by [@giffmana](https://twitter.com/giffmana/status/1937829451670434280), stating **"Big fat metal boxes cannot & will not ever be able to float in the sky,"** went viral as a parody of skepticism towards technological advancements. It was followed by a similar parody from [@cloneofsimo](https://twitter.com/cloneofsimo/status/1937835663870828716) about cats conducting ML research.
- **Perplexity Logo Vote**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1937953810116448599) posted a poll asking the community to vote on the new logo for **Perplexity**, leading to widespread engagement.
- **Industry Satire**: [@scaling01](https://twitter.com/scaling01/status/1937900212242047384) joked that **Google's** very generous free tier for the new **Gemini CLI** was the result of a "dyslexic intern" confusing 10 requests per week with 1000 per day.

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Major New Model Releases and Benchmarks: Jan-nano-128k & Mistral Small 3.2

- [**Jan-nano-128k: A 4B Model with a Super-Long Context Window (Still Outperforms 671B)**](https://v.redd.it/909kwwnbo09f1) ([Score: 755, Comments: 293](https://www.reddit.com/r/LocalLLaMA/comments/1ljyo2p/jannano128k_a_4b_model_with_a_superlong_context/)): **Menlo Research has released Jan-nano-128k, a 4B parameter Qwen3-finetuned LLM with a 128k-token context window, optimized using YaRN scaling. Benchmarks show it achieves a SimpleQA score of 83.2 (with MCP), surpassing Deepseek-671B (78.2) and significantly outperforming other leading models like GPT-4o (62.5) and Gemini-2.5 Pro (52.9), under minimal prompting conditions. The model and GGUF quantization are available on HuggingFace (see [Jan-nano-128k](https://huggingface.co/Menlo/Jan-nano-128k) and [GGUF conversion](https://huggingface.co/Menlo/Jan-nano-128k-gguf)); performance depends on inference engines with proper YaRN scaling support (e.g., llama.server, Jan app). Technical report is forthcoming.** Technically-inclined commenters appear impressed by the benchmark results for a 4B model, though skepticism persists regarding engagement metrics and benchmarking methodology absent a public technical report.
    - A commenter provides a performance context screenshot and mentions that Jan-nano-128k achieves an accuracy of up to `83%` when 'heavily prompting' is used, with benchmarks performed both with and without this technique, indicating that the model's real-world performance may vary significantly based on prompt engineering.
    - A technical question is raised about deployment, noting that while Jan-nano-128k emphasizes local operation and privacy, recommended usage includes a dependency ([mcp-server-serper](https://github.com/marcopesani/mcp-server-serper)) requiring a Serper API key—prompting discussion over the feasibility of a fully local, API-free deployment workflow.
- [**New Mistral Small 3.2 actually feels like something big. [non-reasoning]**](https://www.reddit.com/r/LocalLLaMA/comments/1lk12th/new_mistral_small_32_actually_feels_like/) ([Score: 242, Comments: 73](https://www.reddit.com/r/LocalLLaMA/comments/1lk12th/new_mistral_small_32_actually_feels_like/)): **Mistral Small 3.2, a 24B parameter LLM, is being reported as performing well above its size class according to user testing and comparative benchmarks, outpacing models like Gemma 3 27B, Llama 3.3 70B, and Qwen2.5 72B specifically in writing and logical tasks. Technical issues cited include broken tool calling and incorrect date outputs in various quantizations; a community fix with dynamic quantizations is available via [HuggingFace](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF). For optimal performance, users recommend temperature 0.15, repeat penalty 1.0, minimum probability sampling 0.0, and top-p sampling 1.0.** Commenters are especially optimistic about the potential of the upcoming Mistral Medium (~70B), expecting it to outperform competitors if it maintains the performance-to-size ratio of Small. There is also a preference toward Mistral's generated outputs in logic and writing compared to much larger models.
    - Tool calling and date retrieval in Mistral 3.2 are reported broken in many quantized versions, but community fixes are available, including dynamic quant models hosted on HuggingFace as per https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF.
    - Benchmark comparisons note that Mistral Small 3.2 outperforms Gemma 3 27B in certain tests and is considered by users to surpass Llama 3.3 70B and Qwen2.5 72B in tasks like writing and logic, despite its significantly smaller parameter count (`24B` vs. `70B/72B`). Some express anticipation that a future "Mistral Medium" model could further disrupt this space if it maintains a similar performance-per-parameter ratio.
    - For optimal output from Mistral Small 3.2, recommended inference settings are very low temperature (`0.15`), repeat penalty off (`1.0`), minimum P sampling off (`0.0`), and top P sampling at maximum (`1.0`). Despite strong general intelligence, limitations in step-by-step reasoning remain apparent compared to models like Qwen3 30B which are described as "thinking models."

### 2. Gemini CLI Tool Free Tier Launch and Discussion

- [**Gemini released an Open Source CLI Tool similar to Claude Code but with a free 1 million token context window, 60 model requests per minute and 1,000 requests per day at no charge.**](https://i.redd.it/11rgwmzvv39f1.jpeg) ([Score: 430, Comments: 88](https://www.reddit.com/r/LocalLLaMA/comments/1lkbiva/gemini_released_an_open_source_cli_tool_similar/)): **The image visually highlights the release of an open-source CLI tool by Google Gemini, designed with a focus on developers and code exploration, as described in the post. Technically, this CLI offers a substantial** `1 million token context window`**, up to** `60 model requests per minute` **and** `1,000 requests per day` **for free. This positions it as a high-capacity, zero-cost alternative to tools like Claude Code, but unlike fully open tools, it currently requires use of proprietary Gemini APIs. Data collection for training is a notable topic, with official [privacy terms](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals) allowing prompt/code logging and human review for model improvement, though users may opt out and collected data is purportedly separated from their Google account.** Commenters debate the implications of Google offering such a powerful tool for free, generally agreeing it's about collecting diverse training data, with one user noting opt-out options for data collection. Some express reservations about using tools tied to proprietary, potentially rate-limited APIs, and seek forks for local model use. The main technical discussion centers around privacy, data ownership, and practical limitations versus open-source autonomy.
    - Google's Gemini Code Assist CLI tool is open source and offers a substantial 1 million token context window with generous free-tier usage limits (`60 requests/minute`, `1,000/day`). However, usage requires Gemini cloud APIs, meaning all interactions pass through Google's infrastructure and are subject to their rate limiting and data collection policies.
    - The [privacy notice](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals) for Gemini Code Assist specifies that Google collects prompts, code, outputs, code edits, and feedback to improve services and machine learning models, with human reviewers possibly annotating this data. While data is supposedly separated from your Google account, it is still used for model training unless you opt out.
    - Some users express hesitation due to requirements to use proprietary Gemini APIs and the risks associated with ambiguous or unexpected billing practices, citing cases of unexpected high charges during periods advertised as free usage. This has led to calls for forks that support local inference and open model compatibility to avoid such vendor lock-in and privacy concerns.
- [**Gemini CLI: your open-source AI agent**](https://blog.google/technology/developers/introducing-gemini-cli/) ([Score: 126, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1ljxa2e/gemini_cli_your_opensource_ai_agent/)): **Google has launched the open-source Gemini CLI, enabling direct interaction with Gemini AI models—including Gemini 2.5 Pro—through the terminal with a personal Google account. The product offers a substantial free tier: '1 million token context window,' '60 model requests/minute,' and '1,000 requests/day,' making it suitable for heavy development use (official details archived [here](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/)). There is interest from technical users regarding integrating local models, stemming from the open-source nature of the CLI.** One comment questions the feasibility and sustainability of such generous usage limits, expressing skepticism until further verification. Another inquiry asks whether being open source enables developers to substitute local models, suggesting potential extensibility and local deployment interests.
    - Google's Gemini CLI reportedly offers access to Gemini 2.5 Pro with an unusually large 1 million token context window and allows for up to 60 requests per minute and 1,000 per day free during the preview, which users note is a very generous limit compared to industry norms.
    - Discussion raises technical questions about whether the "open-source" status of Gemini CLI would allow users to plug in and run local models in addition to Google's own Gemini models, but there is uncertainty due to the removal of both the official post and GitHub repository.
    - The disappearance of both the official announcement (now only accessible via archive) and GitHub repository is noted, hinting at possible issues with the release; some users reference archived and alternative sources with screenshots for documentation while Google appears to have deleted the project, possibly to rework or retract the release.

### 3. MCP Feature Integrations and Novel LLM Techniques (LM Studio & ThermoAsk)

- [**LM Studio now supports MCP!**](https://www.reddit.com/r/LocalLLaMA/comments/1lkc5mr/lm_studio_now_supports_mcp/) ([Score: 199, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1lkc5mr/lm_studio_now_supports_mcp/)): **LM Studio has announced support for MCP (Model Compatibility Protocol), enabling seamless interoperability between LM Studio and a broader range of local LLMs and serving tools, as detailed in their [official blog post](https://lmstudio.ai/blog/mcp). This MCP integration aims to facilitate easier loading, fine-tuning, and model management workflows, likely targeting developers relying on custom local model orchestrations.** One user reports encountering errors when searching for models, indicating potential instability or limitations in the current MCP implementation for model discovery. Another comment underscores the significance of the update for users previously reliant on unreliable custom solutions, signaling strong community demand for such compatibility.
    - A user reports errors when attempting to load or search the model list within LM Studio, indicating potential issues with the interface or backend handling of model repositories under the new MCP support.
    - Several users have referenced successful usage of the new Multimodal Control Protocol (MCP) support in the beta channel, suggesting that the implementation is stable for at least some use cases, but with some lingering UI/findability issues (e.g., inability to locate features in settings or access model lists).
- [**ThermoAsk: getting an LLM to set its own temperature**](https://i.redd.it/t8az5arc1z8f1.png) ([Score: 101, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1ljs95d/thermoask_getting_an_llm_to_set_its_own/)): **The image is a metaphorical illustration accompanying the technical idea discussed in the post: enabling a language model (LLM) to dynamically set its own sampling temperature ("ThermoAsk"). The glowing furnace and repeated logos visually reinforce the concept of the LLM actively controlling its 'heat' (sampling randomness/creativity), analogous to temperature in natural language generation. The post introduces a novel technique, with an implementation provided using Ollama's Python SDK and Qwen2.5-7B, where the LLM determines temperature based on task requirements. [Blog post](http://amanvir.com/blog/getting-an-llm-to-set-its-own-temperature) and [GitHub repo](https://github.com/amanvirparhar/thermoask) detail the methodology.** Comment discussion addresses the technical challenges of hallucinations, suggesting the use of a secondary model as an arbiter for quality assessment, and raises questions about reproducibility regarding random seeds. There's strong encouragement to propose this to LM Studio and related tooling communities, highlighting user interest in broader adoption and experimental validation.
    - A user raises an important technical question about controlling hallucinations, specifically asking whether a secondary, higher-quality dense model could act as an independent arbiter to evaluate outputs. The suggestion highlights concerns over using a model to grade its own work and proposes offloading evaluation to a model suitable for CPU/RAM environments if the evaluation task is lightweight.
    - Another commenter implemented the idea in an OpenAI-compatible way, making the approach usable across any UI/LLM setup, and provides a [link to their Reddit post detailing the implementation](https://www.reddit.com/r/LocalLLaMA/comments/1lkixss/getting_an_llm_to_set_its_own_temperature/).

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. Anthropic Book Scanning Controversy and Fair Use Ruling

- [**Anthropic purchased millions of physical print books to digitally scan them for Claude**](https://www.reddit.com/r/singularity/comments/1ljs8np/anthropic_purchased_millions_of_physical_print/) ([Score: 716, Comments: 99](https://www.reddit.com/r/singularity/comments/1ljs8np/anthropic_purchased_millions_of_physical_print/)): **A recent court ruling revealed that Anthropic, under the direction of Tom Turvey (formerly of Google's book-scanning project), purchased millions of print books—utilizing multi-million dollar budgets—to create a proprietary digital corpus for model training. The books were physically disassembled and scanned, with service providers producing high-quality image PDFs and OCR-generated text for each title, then discarding the originals. Full technical context and legal exhibits are available in the [32-page ruling PDF](https://www.documentcloud.org/documents/25982181-authors-v-anthropic-ruling/), and further analysis is provided in [Simon Willison's write-up](https://simonwillison.net/2025/Jun/24/anthropic-training/).** Technical discussion in comments speculates on the reasoning for using physical books (potentially for legal/audit traces or cost efficiency) and hypothesizes that this unique, high-fidelity corpus may contribute to Claude's superior creative writing capabilities compared to other models.
    - One commenter speculates that Anthropic might have purchased and scanned print books for reasons beyond copyright risk minimization, such as cost considerations or ensuring a clear record ("paper trail") of acquisition for legal or transparency purposes where licensing digital copies may be more complex or expensive.
    - Another user raises the point that Claude's reputation for superior creative writing could plausibly be linked to Anthropic's direct ingestion of high-quality book data, suggesting that substantial book-based training corpus may enhance performance in tasks requiring narrative coherence and stylistic diversity, differentiating it from models limited to internet text.
    - The discussion also draws a technical comparison to Bookshare, which reportedly achieved a similar book digitization effort at greater scale and speed via a $30M government grant, suggesting that book scanning at scale is a solved problem and highlighting potential efficiency gaps in Anthropic's approach.
- [**A federal judge has ruled that Anthropic's use of books to train Claude falls under fair use, and is legal under U.S. copyright law**](https://www.reddit.com/r/ClaudeAI/comments/1ljs3mj/a_federal_judge_has_ruled_that_anthropics_use_of/) ([Score: 158, Comments: 60](https://www.reddit.com/r/ClaudeAI/comments/1ljs3mj/a_federal_judge_has_ruled_that_anthropics_use_of/)): **A federal judge ruled that Anthropic's use of lawfully purchased books to train its Claude language model constitutes fair use under U.S. copyright law, highlighting the training process as highly transformative (see full ruling: https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf). However, the judge determined that Anthropic's retention of 7 million pirated ebooks after training did not meet fair use criteria; damages for this infringement will be determined by a jury.** Commenters debate the ethical and legal distinction between using legally acquired content and pirated materials, with some emphasizing the societal benefit of transformative use and others criticizing the notion of profiting from unauthorized copies.
    - SeanBannister highlights a key nuance in the ruling: while the judge found purchasing books for training is fair use due to its transformative nature, storing "7 million pirated ebooks... permanently after training" was *not* covered by fair use. This sets a precedent where the legality of data use for AI hinges on both acquisition method and data retention policies. The next legal step involves determining damages via a jury for the non-purchased books, indicating substantial legal risk for AI developers relying on infringing data sources. Full ruling: https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf

### 2. Google Gemini CLI and On-Device Gemini Updates

- [**Google introduces Gemini CLI, a light open-source AI agent that brings Gemini directly into the terminal**](https://v.redd.it/hbt1631zp29f1) ([Score: 500, Comments: 75](https://www.reddit.com/r/singularity/comments/1lk5h19/google_introduces_gemini_cli_a_light_opensource/)): **Google has released the open-source Gemini CLI, an AI agent designed to bring Gemini's capabilities directly to the terminal, with source code [available on GitHub](https://github.com/google-gemini/gemini-cli/). The CLI enables codebase exploration and code modification from the terminal, but early user feedback notes long search times and inconsistent code navigation, with some users drawing comparisons to Anthropic's Claude Code and speculating minimal core differentiation.** Technical commenters debate Gemini CLI's effectiveness relative to Claude Code, criticizing Gemini's slower code search performance and weaker instruction-following/tool invocation; suspicions are raised about superficial changes distinguishing the two tools.
    - Multiple users note that Gemini CLI's interface and approach are strongly reminiscent of Claude Code, with accusations of little more than string and color changes; there is skepticism about genuine innovation compared to the competition.
    - A user who tested Gemini CLI reports that its codebase search features are significantly slower and less reliable than Claude Code, taking several minutes to index a simple task and ultimately failing to find non-commented-out code.
    - There are technical concerns about Gemini's capabilities in instruction following and tool calling, with some users expressing that Gemini has historically lagged behind Anthropic's Claude Code in these key areas of agentic workflow automation.
- [**Gemini CLI: : 60 model requests per minute and 1,000 requests per day at no charge. 1 million context window**](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/) ([Score: 404, Comments: 74](https://www.reddit.com/r/singularity/comments/1ljxou6/gemini_cli_60_model_requests_per_minute_and_1000/)): **Google has launched the open-source Gemini CLI, enabling terminal access to Gemini AI models with a 1M-token context window, free usage tier at** `60 requests/min` **or** `1,000 requests/day`**, and full support for extensible workflows and integrations. The CLI targets developer productivity and experimentation, with the large context window providing potential advantages over competing models in agentic and sub-agent scenarios ([archived announcement](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/)).** Commenters are comparing Gemini 2.5 Pro to Claude Opus, highlighting that while Opus may be 'smarter,' it has context limitations and instruction lapses. The expansive context and low-cost spawning of sub-agents in Gemini CLI are noted as potentially 'meta shifting.' Technical usage with IDEs (e.g., Visual Studio) was investigated and resolved by users.
    - A user notes that while **Claude Opus** is generally considered more intelligent, it tends to "forget, be full of itself, outright disobey instructions, and its context runs out rather quickly", implying significant practical limitations for agentic use cases. In contrast, **Gemini 2.5 Pro**'s `1 million token context` limit and low-cost subagent spawning could be a "meta shifting" advantage for building complex agentic systems or code assistants.
    - Discussion highlights potential meta-shifting implications of Gemini CLI's large context window (1 million tokens), particularly compared to competitors like Claude. The increased context size could enable more advanced agentic workflows that require long-term memory or coordination between multiple sub-agents without running into prohibitive costs or context fragmentation issues.
    - There is technical interest in integrating Gemini CLI tools in developer workflows, with specific mention of combining Gemini with Claude Code in ways that allow them to "talk to each other", potentially leveraging the unique strengths or contexts of both models in tandem for advanced code generation or automation pipelines.

### 3. AI Model Capabilities and Benchmark Progress

- [**Humanity's Last Exam scores over time**](https://i.redd.it/pdf5hf5a809f1.png) ([Score: 252, Comments: 41](https://www.reddit.com/r/singularity/comments/1ljwxgy/humanitys_last_exam_scores_over_time/)): **The image is a graph depicting the progress of AI models on the benchmark 'Humanity's Last Exam' from April 2024 to June 2025, with score percentages ranging from 0% to 30%. Notably, Deep Research led improvements in February, but as of the post, Moonshot AI's Kimi-Researcher achieves a record 'pass@1' score of 26.9% (up from an initial 8.6%), attributed to its strategy of averaging 23 reasoning steps and exploring 200+ URLs per task. Major models such as GPT-4o and Claude 3.5 Sonnet are also tracked, showing a consistent upward trend in performance.** Commenters express unfamiliarity with Kimi-Researcher ('never heard of Kimi Researcher'), curiosity about its origins, and interest in how current models like GPT-4o compare in scores.
    - A technical point is raised about the calibration error percentage for GPT-4o (o3) on Humanity's Last Exam: its calibration error is *significantly below* other top models, which is desirable since calibration error reflects how accurately a model's confidence in its answers matches their correctness—lower values indicate the model is less overconfident in incorrect answers.
    - Commenters note the omission of certain advanced models, specifically Claude 4 Opus Research and Gemini 2.5 Pro Deep Research, suggesting these would offer valuable comparison data for benchmarks like Humanity's Last Exam.
- [**AlphaGenome: AI for better understanding the genome**](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) ([Score: 336, Comments: 66](https://www.reddit.com/r/singularity/comments/1lk6l28/alphagenome_ai_for_better_understanding_the_genome/)): **AlphaGenome, from DeepMind, is a new AI genomics model capable of processing up to 1Mbp input DNA and making 1bp-resolution predictions for regulatory properties across transcription, splicing, and variants. It achieves this with a hybrid convolutional+transformer architecture, yielding state-of-the-art performance on variant scoring and splice modeling benchmarks, and requires less computation than prior models like Enformer. Notably, the recurring pain point of balancing long-range context (e.g., regulatory enhancers >500kb away) with fine single-base resolution is mitigated by AlphaGenome's architecture, integrating these scales for the first time. The R-value performance is roughly 0.8-0.85 for core tasks, highlighting improvements but also fundamental limits due to unresolved biological stochasticity.** Technical discussion in the thread appreciates the model for advancing the capability to unify distal and local regulatory prediction, serving as a pragmatic basic science tool rather than a diagnostic magic bullet. Key debate centers on the limitations imposed by biological complexity and data, and on AlphaGenome's utility in facilitating hypothesis generation and experimental design in genomics rather than deterministic prediction.
    - The primary technical contribution of AlphaGenome is its ability to process a full megabase of genomic sequence while outputting predictions at single-base-pair (1bp) resolution, effectively bridging the typical trade-off between large-scale genomic context (such as distal enhancers hundreds of kilobases away) and fine-grained regulatory elements (like single transcription factor binding sites). This is described as a significant engineering achievement rather than a novel scientific discovery, integrating existing ideas into a more unified and effective framework.
    - AlphaGenome's results, while strong, do not reach perfect predictive power: the reported R values are around 0.8 to 0.85 (not 0.99), reflecting the inherent complexity and stochasticity of gene regulation similar to chaos theory in weather prediction. This highlights lingering limitations regarding comprehensive prediction and interpretation of genomic function due to biological and data complexity.
    - The model's immediate practical utility is in translational research pipelines: AlphaGenome helps researchers interpret statistical signals from GWAS studies by filtering noise and prioritizing causal variants and biological mechanisms for wet-lab validation (e.g., inferring that a non-coding variant disrupts a specific chromatin loop in a particular cell type). This reduces guesswork and accelerates hypothesis generation for experimental follow-up.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Preview
> 

**Theme 1: Groundbreaking AI Releases and Feature Enhancements**

- **Google & Anthropic Unleash New Dev Power**: Google launched its open-source **Gemini CLI Agent**, powered by **Gemini 2.5 Pro** and supporting MCPs ([Gemini CLI Agent video showcase](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4)), while Anthropic debuted **Artifacts and an Artifacts Gallery** enabling users to build Claude within Claude ([Anthropic Artifacts video demo](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4)). These tools aim to enhance developer interaction with powerful AI models.
- **OpenRouter Boosts Transparency and Control**: OpenRouter rolled out a new [model uptime API via X.com](https://x.com/OpenRouterAI/status/1937869909448441980) for developers to monitor model availability and enhanced its **Bring Your Own Key (BYOK)** feature with pre-save key testing and usage limiting capabilities ([BYOK improvements detailed on X.com](https://x.com/OpenRouterAI/status/1937872903988535400)). These updates offer developers greater visibility and management over their AI model usage.
- **MCP Integration Expands with LM Studio and LlamaIndex**: **LM Studio**'s [version 0.3.17 release blog](https://lmstudio.ai/blog/lmstudio-v0.3.17) announced **MCP Host** functionality ([LM Studio MCP Host documentation](https://lmstudio.ai/docs/app/plugins/mcp)), allowing local LLM connections, while **LlamaIndex** released an open-source template for building a **Claude-compatible MCP server** as a Next.js app. These developments broaden the ecosystem for the **Model Context Protocol**.

**Theme 2: Model Mayhem: Performance Quirks, Bugs, and Benchmarks**

- **Models Stumble on Output and Token Limits**: Users found **GPT 4.1 mini** truncating output at **3800 tokens** despite a **33k token** capacity and adding unwanted characters to JSON, while **OpenRouter** providers reportedly misrepresent maximum output tokens, hindering LLM reasoning tasks. These issues highlight ongoing challenges in achieving reliable and predictable model outputs across different platforms.
- **Cursor Grapples with Context, Deepseek Dips**: **Cursor** users debated automatic summarization for chats exceeding context length, leading to potential content loss, and noted from [Cursor's context management documentation](https://docs.cursor.com/context/management) that Gemini handles larger context better than Claude. It was also observed that **Deepseek** models perform poorly with context, prompting Cursor to reduce their context length to around **60k tokens**.
- **LLMs Face Logic Tests & New Benchmarks Emerge**: Engineers explored challenging LLMs with questions based on logical fallacies, drawing inspiration from [Wikipedia on Gödel’s incompleteness theorems](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems), to assess comprehension beyond imitation. Separately, the [Artificial Analysis MiniMax benchmark page](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price) gained attention for its positioning in intelligence vs. price evaluations, featuring techniques like [Multi-head latent attention as detailed on arXiv](https://arxiv.org/pdf/2206.04615).

**Theme 3: The Evolving Dev Frontier: GPUs, Tools, and Languages**

- **Unsloth Champions Intel XPUs & Budget GPU Access Heats Up**: Unsloth announced support for Intel XPU via an [Unsloth commit for Intel XPU support](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a), a move anticipated to be significant with Intel's upcoming **48GB GPU for < $1k**. Meanwhile, users recommended the [Hyperbolic XYZ GPU rental platform](https://app.hyperbolic.xyz/) for affordable **H100** rentals at **$0.99/hour**.
- **MakoGenerate Automates Kernel Creation & GPU Mode Unveils Trimul Benchmark**: **MakoGenerate** launched at its [MakoGenerate platform site](https://generate.mako.dev/), an AI agent for generating GPU kernels deployable to **H100** or **B200**, with a VS Code extension in progress. Concurrently, GPU Mode introduced the **Triangle Multiplicative Update (Trimul)** benchmark ([GPU Mode Trimul benchmark details](https://tinyurl.com/gpumode-trimul)) for NVIDIA and AMD hardware, sparking competitive kernel optimization.
- **Mojo Challenges Rust Async & Introduces Effect Generics**: The Mojo team aims to simplify asynchronous programming by using a better async runtime and [linear types detailed in a Modular PR](https://github.com/modular/modular/pull/3946) to avoid Rust's `Arc<Mutex<T>>` complexities. Additionally, Mojo is exploring [effect generics in another Modular PR](https://github.com/modular/modular/pull/4728) to tackle function coloring for most libraries, allowing the compiler to pick optimal I/O APIs.

**Theme 4: AI's Societal Mirror: Ethics, Copyright, and the Future of Content**

- **AI "Truth" and Censorship Spark Heated Debates**: Members voiced concerns over **Grok AI's** mission to rewrite knowledge into its own 'truth', fearing political indoctrination, and debated the effectiveness of censorship in **Chinese AI models**. The discussions highlighted anxieties about AI systems shaping narratives and adhering to (or bypassing) content restrictions, with some noting **Yi models** remain uncensored on sensitive topics.
- **Copyright Battles Rage as Facebook Faces Piracy Ruling & Data Demands Grow**: [Adam Eisgrau's tweet on Facebook piracy lawsuit](https://x.com/AdamEisgrau/status/1937480346976813454) suggested **Facebook** may have lost the piracy aspect of a book piracy lawsuit, even with a favorable ruling on transformative training, fueling ongoing debates about using copyrighted material. Despite legal uncertainties, strong community demand persists for utilizing copyrighted works in training datasets, with some advocating for payment systems.
- **Google's AI Web Vision Prompts "End of the Web" Fears**: Discussions around Google I/O announcements, where **AI will write websites for other AI to scrape**, led to jokes that *Google is definitely cooking the end of the web*. This highlights concerns that AI-generated content might devalue human-created web content and alter the internet's ecosystem fundamentally.

**Theme 5: Pushing the Boundaries: Advanced AI Research and Techniques**

- **BitNet Dazzles with Speed & Quality While Cerebras Offers Cheap Scale**: Users testing the [BitNet demo on Azure](https://bitnet-demo.azurewebsites.net/) (also on the [Chat-with-Bitnet-b1.58-2B-4T HF Space](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T)) reported being impressed by its speed and quality, particularly for initial queries. For scaling, the [Cerebras Cloud information page](https://www.cerebras.ai/cloud) highlighted its wafer-sized GPUs as *super cheap at scale*, comparable to Blackwell but with less bandwidth.
- **Anthropic Questions RL Research Rigor & Dr. GRPO Emerges**: Anthropic researchers, on the [Dwarkesh podcast with Anthropic researchers](https://youtu.be/64lXQP6cs5M?t=550), argued many **RL papers** use smaller models, potentially skewing insights for frontier models, advocating for tests on larger models like DeepSeek. Meanwhile, **Dr. GRPO** ([GRPO paper on arXiv](https://arxiv.org/abs/2501.12948), [Dr. GRPO discussion on Discord](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368)) was noted for reducing **GRPO's** chattiness while maintaining performance.
- **"Your Brain on LLMs" Study and AlphaGenome Unveil New Insights**: The **206-page**  ["Your Brain on LLMs" paper on arXiv](https://arxiv.org/pdf/2506.08872v1), despite some initial proofing issues, was praised for its content regarding human-AI cognitive interaction. Separately, DeepMind introduced **AlphaGenome**, an AI system to enhance genomic understanding, detailed in [DeepMind's AlphaGenome blog post](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/).



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Fellowship Prizes Remain a Mystery**: A member sought clarity on claiming business fellowship rewards (mug, t-shirt, raffle tickets) upon reaching **25 members**.
   - Specific instructions and details for claiming these rewards are currently lacking.
- **Google Drops Gemini CLI Agent as Open Source**: Google launched its open-source **Gemini CLI Agent**, powered by **Gemini 2.5 Pro** and supporting **MCPs**; [this video](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4) shows how it offers developers access to **Gemini 2.5 Pro** with support for multi-character prompts.
   - It is meant to provide developers access to **Gemini 2.5 Pro** with support for multi-character prompts (MCPs).
- **Anthropic Debuts Artifacts and Artifacts Gallery**: Anthropic released **AI-powered Artifacts and Artifacts Gallery** on the web, enabling users to build Claude inside Claude as illustrated in [this video](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4).
   - The **Artifacts Gallery** allows real-time collaboration and development within the Claude environment.
- **Imagen 4 arrives quietly**: **Imagen 4 and Imagen 4 Ultra** are now available on AI Studio and APIs.
   - A member tested *a clown doing a handstand in a blizzard* with [a shared image](https://cdn.discordapp.com/attachments/1047649527299055688/1387374829744685088/wDnjFf5U1YUCQAAAABJRU5ErkJggg.png?ex=685dc5bf&is=685c743f&hm=4f1cc936f04925773b1328ffbcc229e48fa59a5ae4b74754963caf76c527079d&) showing that it isn't quite there yet.
- **Search Domain Filters Trigger Hallucinations**: Members reported that setting search domain filters to news sites (e.g. reuters.com) now results in **hallucinations of articles** in the `pplx-api` channel.
   - Users reported that they are not getting valid results or citations arrays, which is causing frustration.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Context Crisis Consumes Cursor**: Members discussed **Cursor** automatically summarizing chat windows exceeding context length which leads to content loss and user confusion, noting that [Gemini handles larger context better than Claude](https://docs.cursor.com/context/management).
   - It was further noted that **Deepseek** models perform the worst at understanding context, leading **Cursor** to reduce their context length to around **60k tokens**.
- **Rate Limit Rumblings Rattle Pro Users**: Some **Pro** users are seeing varied rate limit experiences, resorting to short prompts and no attached files to avoid hitting the limits, while others suggest [the potential value of a Pro+ plan](https://cursor.com/pricing).
   - Frustration mounts over the lack of transparency regarding rate limits, making it difficult to predict usage and plan effectively with one user stating that the *new meta of pro is pay for pro but think about token usage via thoughtful chat messages lmao*.
- **Gemini CLI's Grandiose Gambit Goes Gone**: Users testing the new **Gemini CLI** found it buggy and not ready for release, reporting freezes during `npm run dev` and failures to install **Nuxt**.
   - Despite offering **1000 requests** a day, the service was considered super slow and broken, with jokes that [it might include ads in the codebase](https://blog.google/technology/developers/introducing-gemini-cli/).
- **Background Agents' Secrets Stay Secret**: Users can now configure secrets for **Background Agents** in **Cursor Settings >> Background Agents >> Secrets**, avoiding the need to push them to `environment.json`.
   - This empowers agents to use the secrets as required.
- **Git Remote URL Gremlins Glitch Agents**: A user discovered that a local repo URL with a leading `www` caused issues with **Background Agents** due to a check for a valid `github.com` URL.
   - The agent runs `git remote get-url origin` and checks whether the URL is a github.com URL.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek's Large Appetite Sparks Search for Smaller Models**: Members are seeking a smaller version of [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), as even the **1-bit version** is too large for some GPUs, leading to discussions around alternatives like **Qwen-3**.
   - Discussion arose around comparing **Qwen-3** versus non-Qwen models and whether **DeepSeek-R1-0528-Qwen3-8B** should be tried.
- **Deterministic Dreams: Chasing 100% Predictable AI**: The potential of creating a model that **outputs 100% deterministic results** was considered, with suggestions to set the temperature to 0 and fine-tune.
   - A member pointed out that doing determinism with a probability function is flawed, while others highlighted the usefulness of randomness and the difficulty of achieving **100% determinism**.
- **Unsloth Embraces Intel XPU: Budget GPU Boom?**: Unsloth now supports Intel XPU as per [this commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a).
   - This is anticipated to be a **huge deal** with the upcoming release of the **48GB GPU for < $1k** later this year.
- **MacBook Pro Meltdown: Cooling Solutions Explored**: Discussions revolved around cooling MacBook Pros during NN training, suggesting solutions like **aluminum stands**, **fan-equipped stands**, and reapplying **thermal paste**.
   - Members recommended keeping GPU temperatures at **90-95C maximum** and advised against using ice or water for cooling.
- **H100 Rentals at bargain prices on Hyperbolic XYZ**: A user recommended [Hyperbolic XYZ](https://app.hyperbolic.xyz/) for renting **H100s** at **$0.99** and **RTX 4090s** at **$0.28** per hour, and also included a referral code.
   - It was shared in the `#help` channel.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT Lands Web Dev Job**: A member reported that **GPT** assisted them in securing a web developer job, leading to discussions on **AI's impact on employment**.
   - The member suggests that the initial impact of **AI on jobs** isn't AI replacing humans, but rather *disadvantaged people with internet access* gaining opportunities.
- **O3 and Pro Unlocks Cloud Search Connectors**: **Chat search connectors** were launched for Pro users on **June 24, 2025**, integrating with services like **Dropbox**, **Box**, **Google Drive**, and **Microsoft OneDrive/SharePoint**.
   - This feature is restricted to users outside the **EEA**, **Switzerland**, and the **UK**, enabling users to train AI models with their synced data.
- **AI Sparks Art Debate**: A member initiated a conversation about the role of AI in art, focusing on the distinction between using AI for concepts or mockups versus selling purely AI-generated items as one's original work.
   - The core of the argument centers on the ethics of presenting **AI-generated art** as one's own creation, emphasizing the importance of acknowledging the tool's involvement in the creative process.
- **Logic Traps Expose LLM Weaknesses**: Members explored using questions based on logical fallacies to challenge LLMs, aiming to assess their capacity to identify and respond appropriately to nonsensical inputs, drawing inspiration from [Gödel’s incompleteness theorem](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems).
   - The group consensus suggests that the failure to discern such traps points to a deficiency in comprehension rather than mere imitation, indicating that a true understanding of logic is lacking.
- **Minimax Benchmark Hides in Plain Sight**: A user highlighted the [MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price), which was conspicuously placed in the intelligence vs price quadrant.
   - The benchmark includes [Multi-head latent attention](https://arxiv.org/pdf/2206.04615), though some dismissed it as *technobabble*.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BitNet Demo Stuns with Quality**: Users testing the [BitNet demo](https://bitnet-demo.azurewebsites.net/) report being impressed by its speed and quality, particularly for initial queries.
   - Available as a HF Space ([Chat-with-Bitnet-b1.58-2B-4T](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T)), it allows programmatic access to **BitNet**'s capabilities.
- **Cerebras Cloud Offers Budget-Friendly Scaling**: Members find [Cerebras](https://www.cerebras.ai/cloud)' wafer-sized GPUs *super cheap at scale*, putting it on par with **Blackwell** but with less bandwidth.
   - Another member noted **Groq** has *very nice tech* and is *really good for inference at scale*.
- **LLMs Forge Shader Graph Frontiers**: Members explored using **LLMs** to generate **shader graph code** (convertible to **HLSL** or **GLSL**) and how researchers are optimizing these with **language models**.
   - One user notes that **Nvidia** is exploring **neural materials** via small models predicting pixels.
- **ModernBERT dissection highlights input embeddings**: A member's post on *gradient descent* on **token input embeddings** and **ModernBERT** was accepted by [LessWrong.com](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert).
   - The **LessWrong** post dives into **ModernBERT** architecture focusing on how gradient descent is applied to token input embeddings.
- **RAG embedder collection gets SmartTaskTool upgrade**: A member shared a link to a **RAG embedder collection** ([Hugging Face link](https://huggingface.co/kalle07/embedder_collection)) and a **small task toolbar** for Windows ([Hugging Face link](https://huggingface.co/kalle07/SmartTaskTool)).
   - The **SmartTaskTool** is a taskbar icon and now includes cross-lingual support (en-de-fr-roberta).



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok AI Faces Indoctrination Accusations**: Members voiced concerns that **Grok AI's** mission to rewrite knowledge into its own 'truth' constitutes political indoctrination.
   - The sentiment is that persuasive LLMs should not be bent on reinforcing political narratives.
- **Chinese AI's Censorship Sparks Debate**: Debate arose on the effectiveness of censorship in **Chinese AI models**, with some claiming the models simply use external filters, and others stating the models willingly abide by laws.
   - Members highlighted that **Yi models** remain uncensored regarding sensitive topics like **Tienanmen Square**.
- **Gemini 2.5 Pro Dethroned by Underdogs?**: **Gemini 2.5 Pro's** declining performance on the leaderboard is attributed to the rise of anonymous models like **Blacktooth** and **Stonebloom**.
   - While some speculate these models excel where **Gemini 2.5 Pro** falters, others believe shifts in voter distribution are the cause.
- **LM Arena Leaderboard Vulnerable to Prompt Exploitation**: A user claimed to have discovered **4 methods** to extract leaderboard data from lmarena.ai, sparking ethical and legal discussions.
   - The methods included utilizing the Hugging Face space, pre-existing data dumps, web scraping and browser extensions, but a community member stated that **3/4** ways given are not valid.
- **Open Source Community's Copyrighted Yearning**: Despite recent court rulings, there's strong community demand for utilizing copyrighted material in training datasets, though the legality is contested.
   - Perspectives vary on the impact of recent rulings, with some seeing no impact on continued training and others desiring a payment system for copyrighted data.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio adds MCP Host and Speaks New Languages**: [LM Studio version 0.3.17](https://lmstudio.ai/blog/lmstudio-v0.3.17) now supports **MCP Host**, enabling connection to local LLMs, boasts **33 languages** thanks to community localizers, and introduces a **'Solarized Dark'** theme.
   - Refer to the [LM Studio documentation](https://lmstudio.ai/docs/app/plugins/mcp) for more information on **MCP Host** integration and features.
- **LM Studio Chat Messages Vanish then Reappear**: A user reported that after upgrading **LM Studio**, previously *hidden* conversations from an 'uncensored' model became visible, including an exchange about *'What's the process for creating a fake ID card?'*
   - Theories ranged from streaming settings to system prompts, but the precise cause for the hidden conversation remains unclear.
- **r/LocalLlama Subreddit Gets New Overlord**: The [r/LocalLlama](https://www.reddit.com/r/LocalLlama/) subreddit is under new management, sparking discussion about the new moderator's involvement in numerous other subreddits.
   - Some users raised concerns about the moderator's wide reach, while others found no immediate red flags.
- **Runpod Serverless Eyed for Inference**: A user is planning to test **Runpod serverless**, specifically *flex workers* with a network volume for faster model loading and cold starts, for playing with NVIDIA GPUs for inference tasks.
   - They are also considering **Predibase** and its *turbo lora* features for future use.
- **Unsloth AI has Commit Spotted**: A user shared a [link to an Unsloth commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a), potentially indicating interest in or discussion of the **Unsloth AI project**.
   - No further details about the commit's content or significance were provided.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Rolls Out Model Uptime API**: Developers can now monitor model uptime via the [OpenRouter API](https://x.com/OpenRouterAI/status/1937869909448441980), providing transparency on model availability.
   - This feature allows for better planning and management of AI applications that rely on consistent model performance.
- **BYOK Gets Better with New Features**: **Bring Your Own Key (BYOK)** users can now test keys before saving, limit upstream usage, and track usage in API calls, enhancing control and security ([details here](https://x.com/OpenRouterAI/status/1937872903988535400)).
   - These improvements provide more granular control over API key management and usage tracking.
- **Midjourney's Video Venture Excites Users**: Members hailed the new video model from **Midjourney** and **Spellbrush** as a *chatgpt moment of i2v* and hoped to see more infrastructure to roll out 720p.
   - Alternatives like *seedance* and *hailuo* were mentioned but deemed significantly inferior in quality.
- **GPT 4.1 Mini Exhibits Output Quirks**: **GPT 4.1 mini** is truncating output at **3800 tokens** despite a **33k token** capacity and adding `\xa0` before JSON keys, causing integration issues.
   - Members suggested lowering the temperature and specifying `"response_format": {"type": "json_object" }` to enforce correct JSON output; others found **GPT 3.5** more reliable for certain tasks.
- **Veena Voices Victoriously in Indian Languages**: A new voice AI model for **Indian languages**, named **Veena**, launched with assistance from OpenRouter, with [details on X.com](https://x.com/Dheemanthreddy_/status/1937839083281437021).
   - The launch was congratulated by members, marking a potentially significant step in local language AI support.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **C++ Build Systems Baffle and Bewilder**: Members expressed frustration with **CMake**, with one stating the amount of time wasted trying to get cmake working with some random library is far too much and called **Bazel** the *goat* build system.
   - Suggested alternatives included [Meson](https://mesonbuild.com/), [Buck2](https://buck2.build/), [xmake](https://xmake.io/#/) and [Zig](https://ziglang.org/).
- **Triton Closes Doors, Community Yearns**: A member noted that [Triton is no longer open to the public](https://discord.com/channels/1189498204333543425/1189607595451895918/1378126514318737541) and highlighted **Gluon**, a new higher-level DSL in the Triton repo, resembling **Paszke's Mosaic**, pointing to the [test_core.py](https://github.com/triton-lang/triton/blob/5c9e54535dfe34d4b60fd13a4a27b6d74f3c8344/python/test/gluon/test_core.py).
   - A member expressed confusion about why the **Triton team** stopped communicating with those using **Triton** as a foundation for their projects, especially about why **Triton isn't supported on Windows**.
- **AMD Cloud Options Abound**: Members recommended [DigitalOcean](https://www.digitalocean.com/) and [TensorWave](https://www.tensorwave.com/) as good cloud providers for **AMD machines**, especially for smaller projects and experimentation.
   - Other providers mentioned include **Hotaisle** and **Runpod**, with one member noting Hotaisle is *pretty nice*.
- **MakoGenerate Courts VS Code, LLM Quirks**: The creator announced **MakoGenerate**, an AI agent to generate GPU kernels deployable to **H100** or **B200**, inviting feedback on the platform at [generate.mako.dev](https://generate.mako.dev) and confirmed they are already working on a **VS Code extension** and offered unlimited free credits.
   - Users noted that the **LLM** sometimes switches between the provided problem and the prompt, even when explicitly instructed to ignore the sample problem, making it harder to get the LLM to do what is wanted.
- **Trimul Task Triumphed on NVIDIA, AMD**: A new problem based on the **Triangle Multiplicative Update** used in the AlphaFold family of models has been announced and is available on [GPU Mode](https://tinyurl.com/gpumode-trimul) for both **NVIDIA** and **AMD** hardware.
   - A user achieved **first place** on the `trimul` leaderboard for **B200** with a time of **7.92 ms** and another user got **first place** on the `trimul` leaderboard for **A100** with **20.0 ms**.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dr. GRPO Dilutes GRPO's Chatter**: Members report that **Dr. GRPO** reduces the chattiness of **GRPO** while maintaining performance, based on [this discord discussion](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368).
   - A [YouTube video](https://www.youtube.com/watch?v=K34gBCjzni8) and [paper](https://arxiv.org/abs/2501.12948) were referenced for implementing **GRPO**, with **Dr. GRPO** building upon it.
- **Dive Deep into Forward Propagation Details**: It was clarified that **LeCun** defines **Forward Propagation (FF-prop)** as the standard forward inference process, where layers run after training, without backpropagation.
   - While **Hinton's Forward Forward** may not scale, **Forward Gradients** works effectively as the transpose of backpropagation, serving as the most basic method for finding a derivative.
- **AlphaGenome Ascends, Illuminating the Genome**: DeepMind introduced **AlphaGenome**, an AI system designed to enhance our understanding of the genome, detailed in [a recent blog post](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/).
   - The announcement sparked conversation among members in the **ml-news** channel.
- **Brain-on-LLMs' Proofing Problems**: A member noted a **study** titled *Your Brain on LLMs* and shared a [screenshot](https://cdn.discordapp.com/attachments/1045297868136779846/1387359102975610880/Snipaste_2025-06-25_16-09-14.png?ex=685db71a&is=685c659a&hm=9cfa39ed78e88f2ca714cf645ba04ec641289dd15a88077b4b4669245859e86a) regarding font and text color inconsistencies in dark mode.
   - Despite the initial visual shock, the member remarked that the **206-page paper** ([https://arxiv.org/pdf/2506.08872v1](https://arxiv.org/pdf/2506.08872v1)) was *actually quite good* after reading a portion of it.
- **Google Gets Gemini CLI Going**: Google unveiled [Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/), a free, open-source AI agent bringing Gemini directly to developers' terminals.
   - The tool is advertised as providing *unmatched access for individuals*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Murati's Thinking Machines Lab Focuses on RL for Business**: Mira Murati's new AI startup, **Thinking Machines Lab**, is focusing on **Reinforcement Learning (RL) for businesses** according to [this article](https://xcancel.com/steph_palazzolo/status/1937284120062706004).
   - No further details were provided regarding specific products or launch dates.
- **Warp 2.0 Enters the Agentic Development Arena**: **Warp 2.0** is introduced as an agentic development environment enabling developers to *code by prompt* instead of by hand, touted as **#1 on Terminal-Bench** with **71% on SWE-bench Verified** via [this tweet](https://xcancel.com/warpdotdev/status/1937525185843752969).
   - This represents a shift towards AI-driven coding assistance and automation.
- **Airtable's Omni AI Agent Refounds App Platform**: **Airtable** has relaunched as an **AI-native app platform**, shifting to a complete *refounding* with **Omni**, an AI app-building agent which lets users build robust apps conversationally, according to [this tweet](https://xcancel.com/howietl/status/1937577526634987595).
   - This demonstrates the increasing integration of AI agents into app development workflows.
- **Liquid AI Crafts Concise Reasoning Model**: Maxime Labonne from **Liquid AI** announces a **1-billion parameter reasoning model** that is both accurate and concise, combining **Supervised Fine-Tuning (SFT)** and **GRPO (Generative Reinforcement Learning from Human Preferences)**, and detailed in [this tweet](https://xcancel.com/maximelabonne/status/1937819336204304692).
   - This model aims to provide efficient reasoning capabilities with a relatively small parameter size.
- **OpenRouter Secures Backing for AI Model Marketplace**: Deedy announced their backing of **OpenRouter**, an AI model marketplace that provides developers access to **400+ LLMs** via a single API, which handles **100 trillion tokens annually**, according to [this tweet](https://xcancel.com/deedydas/status/1937902948920811729).
   - The platform's scale indicates a substantial demand for diverse AI models accessible through a unified interface.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Facebook Flounders in Piracy Fiasco**: A [tweet](https://x.com/AdamEisgrau/status/1937480346976813454) suggests **Facebook** may have lost the piracy aspect of a book piracy lawsuit, despite a favorable ruling on transformative training.
   - The ruling's implications on the use of copyrighted material for training **LLMs** is under scrutiny.
- **Nous Navigates Yacine Nabbing**: Members debated why **Nous Research** hasn't hired **Yacine**, a former **X** engineer, with opinions varying on his suitability for ML roles.
   - Some members questioned his skill set, while others considered whether he'd be a good fit.
- **OpenRouter's Output Token Outrage**: Users reported that many **OpenRouter** providers misrepresent their maximum output tokens, hindering the performance of reasoning **LLMs**.
   - The hard limit of **16k tokens** prevents running most **AIMS** problems, but some users are working around it by selecting specific providers who deliver on the promised token limits.
- **Anthropic Assails RL Research**: **Anthropic** researchers **Sholto Douglas** and **Trenton Bricken** argued on the [Dwarkesh podcast](https://youtu.be/64lXQP6cs5M?t=550) that many **RL papers** use smaller models, potentially skewing the dynamics of frontier models.
   - They advocate for experiments on the largest **DeepSeek** model to obtain more representative results, suggesting current research may not reflect real-world performance.
- **Hermes 4 Heft and Hosting Hopes**: A member announced that **Hermes 4** on a **671b** parameter model is expected in the next month or so.
   - Another member questioned who will host **Hermes 4**, noting that current hosts for **Deepseek V3** or **R1** are often slow, expensive, or unstable on openrouter.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Sidesteps Rust Async with Linear Types**: Mojo aims to tackle Rust's async complexity by using a better async runtime and [linear types](https://github.com/modular/modular/pull/3946) avoiding constructs like `Arc<Mutex<T>>`.
   - Mojo seeks to control data movement between threads and ensures data isn't dropped early, potentially offering opt-in work stealing while favoring thread-per-core.
- **Effect Generics Color Functions in Mojo**: Effect generics are being explored in Mojo to address function coloring for most libraries, as detailed in this [PR](https://github.com/modular/modular/pull/4728).
   - This approach, combined with effect generics, lets the compiler/runtime pick the *"best"* IO API for a program, except for cases involving custom IO API bindings.
- **Confusing Error Messages Plague Mojo Dictionaries**: A new Mojo user reported confusing error messages when working with the **Dict struct**, particularly regarding the use of `.value`, `.keys`, and `.items` without parentheses.
   - The error message *"statements must start at the beginning of a line"* was deemed unhelpful, and the user has been asked to [file an issue on GitHub](https://github.com/modular/modular) suggesting a more descriptive error message.
- **InlineArray Moveinit Needs Examination**: The behavior of **InlineArray** during move operations (`b = a^`) was questioned, with concerns raised that neither the copy nor move constructor of elements are being called, potentially indicating a bug.
   - It appears that **InlineArray** is performing a bitwise copy during move initialization, lacking an explicit moveinit.
- **TorchScript Compilation Still Needs Torch**: A user realized that the **Torch environment** is needed to compile a **TorchScript** file with an **InferenceSession**.
   - They expressed frustration about the need for the **Torch** dependency.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **OpenAI API Hits Uptime Issues**: Members reported that their **application was down** due to [issues with the OpenAI API](https://platform.openai.com/docs/status), receiving an `HTTP/1.1 404 Not Found` error.
   - This indicates the requested resource was not found, affecting application availability.
- **SIMBA Error Solution Unlocked**: Members analyzed a [SIMBA error](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/simba.py) concerning frozen submodules and predictor inventories.
   - The fix requires ensuring predictors returned from `name_predictors` align with those iterated during `append_rule` and `append_demo`, notably when using `._compiled = True`.
- **Discord DSPy Tag Dream Debated**: A member proposed creating a **Discord DSPy tag** to showcase DSPy expertise next to usernames.
   - Implementing such tags requires at least **3 boosts** to the Discord server, according to [Discord's Guilds FAQ](https://support.discord.com/hc/en-us/articles/23187611406999-Guilds-FAQ).
- **dspy.Prediction Patterns Probed**: A member questioned if returning something other than a **dspy.Prediction** from a module's forward method is an anti-pattern.
   - The consensus suggests it could cause issues if the metric function doesn't know what to expect from the output, impacting optimization.
- **Shopify Founder Supercharges DSPy**: Shopify founder [Tobi Lutke joins DSPy](https://x.com/tobi/status/1937967281599898005).
   - The unexpected move highlights the project's growing significance.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's Limit Lashing Leads to Lost Labor**: Users reported frustration that **NotebookLM** does not announce when it hits the generation limit *before* the customize prompt, resulting in potentially lost work.
   - Members are wondering if a very long customize prompt will stick with the notebook when they return.
- **Vimeo Ventures into NLM Vexation**: Users reported issues using **Vimeo** videos as sources in **NotebookLM**, with security features blocking content access.
   - One member suggested downloading the video using [cobalt.tools](https://cobalt.tools/) as a workaround, while another asked if having transcripts already uploaded obviates needing the video itself.
- **AI Audio's AdSense Ambiguity**: A user inquired whether **YouTube** allows monetization for channels using **AI-generated voices** and content from **NotebookLM**.
   - Another member noted the *gray area* in AI and copywrite and suggested researching YouTube's rules regarding **AI content monetization**.
- **PDF Preferred for Potent Processing by NLM**: In a message thread, a user asked whether **PDF** or **MD** format is better for **NotebookLM**.
   - Another member responded that **PDF is the better format**.
- **PrintFriendly Provides Printer-ready Pages**: A user identified the extension in the image as **PrintFriendly** and located it in the [Chrome Web Store](https://chromewebstore.google.com/detail/printfriendly-print-pdf-a/ohlencieiipommannpdfcmfdpjjmeolj).
   - **PrintFriendly** converts web pages to printer-friendly and **PDF** formats.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyGrad Refactor Bounties Attract Attention**: Members showed interest in the **refactor bounties** as an entrypoint to understand **tinygrad internals** and learn with a case in **JIT testing function**.
   - One member even submitted a pull request (PR) to handle the input for arrays, but their test case failed.
- **Scheduler Heuristic Slashes Graph Size**: Using `RING=0` with a basic **scheduler heuristic** significantly reduces the largest **graphexec** size from **5k to 2k**.
   - The improvement highlights the impact of scheduler optimizations on graph execution efficiency.
- **FUSE_OPTIM Struggles to Ignite**: Setting `FUSE_OPTIM=1` doesn't seem to produce the expected effect, prompting a member to explore non-greedy search strategies.
   - This suggests potential issues with the current fuse optimization implementation, warranting further investigation.
- **NCCL Neatly Handles CUDA Graphs**: A question arose about how **NCCL** manages **CUDA graphs**, which apparently function well, in contrast to tinygrad's current implementation.
   - This suggests that **NCCL** may offer insights or techniques that could be beneficial for tinygrad's CUDA graph integration.
- **Zero-Dimensional Tensors Trouble Gradients**: A user questioned why the gradient of `a` was an *arbitrary number* instead of **4**; this arises from **zero-dimensional tensors** requiring gradients.
   - The advice was to ban these and recommending `a` be changed to `Tensor([2.0], requires_grad=True)`.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Google Bakes End of Web with AI**: Members discussed Google I/O announcements where **AI will write websites and generate content** only for other **AI to scrape and summarize it**.
   - One member joked that *Google is definitely cooking the end of the web* and Soon Chrome will be a chat interface.
- **MCP Client Escapes Desktop Prison**: A member clarified that **MCP client/host architectures** *can be anything, web, cli*.
   - The member was interested in running a **daemon-based MCP client** in the cloud with a **lightweight REST-based proxy** to handle browser UI communication, translating HTTP to MCP.
- **Browser MCP Client Idea Surfs In**: A member suggested making the **MCP client directly in the browser**, potentially creating the **MCP server** there as well to avoid SSE and streaming complexities.
   - He noted that he will have to look into that option and it could be an interesting idea.
- **Hugging Face MCP Auth Triggers Available**: Members discussed hugging face authentication for MCP, triggered with [https://hf.co/mcp?login](https://hf.co/mcp?login).
   - They noted that authentication is anonymous by default.
- **MCP Cloud Launches Managed Hosting**: [MCP Cloud](https://mcp-cloud.ai) launched managed hosting specifically for **MCP servers**, offering dedicated instances, JWT auth, and real-time logs, with deployment in seconds.
   - It supports multi-workflow and copy/paste integration, particularly with **N8N**, and is geared towards developers and teams needing reliable, secure MCP infrastructure.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus's Reliability Questioned Amidst Credit Loss**: Several users reported issues with **Manus**, including getting *stuck at thinking* and throwing *internal server errors*, alongside concerns about recent **credit loss**.
   - Some users voiced their opinion that *Manus has become dumber and makes mistakes*.
- **Invitation Code Offered, Credit Usage Debated**: One user offered an **invitation code**, amidst discussion about **Manus's increased credit usage**.
   - It was claimed that *he's def using more credits*.
- **Limited Credits Assigned, Bug?**: A user reported receiving only **1k credits**, with no further context provided.
   - It's unclear whether this is a bug or intended behavior.
- **Manus Refuses to Share VS Code Password**: A user trying to access **VS Code** on **Manus's computer** encountered a login prompt requiring a password, which **Manus** refuses to provide.
   - The user was told to *Check the config file at .../config yaml for the password*.
- **Quality Agent Mode vs High Effort Mode**: A user inquired whether the new **quality agent mode** is the same as the previous **high effort mode**.
   - No conclusive answer was provided.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Search Underway for Responsible AI Channel**: A member inquired about a dedicated channel for discussions on **responsible AI**, **AI safety**, and **fairness** within the Cohere Discord.
   - The request did not yield immediate responses or pointers to existing resources.
- **Student Automates Code Review at Ethereum**: A UC Davis student and Ethereum Foundation intern is automating code review and vulnerability detection, leveraging **Perplexity** for research.
   - His work also explores adversarial angles for **LLMs** and **LLM memory**.
- **Berlin Student Maintains ML Fairness Toolkit**: A computational linguistics student in Berlin maintains **fairlearn**, an open-source toolkit for **ML fairness**.
   - She aims to apply her fairness expertise to computational linguistics after assisting with the **Aya project**.
- **Engineer Plays with Transformer Architecture**: An AI Engineer/Researcher is focused on modifying **Transformer Architecture** for small use cases.
   - The engineer publishes a newsletter called *Agents: All You Need*.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Opens Claude-Compatible MCP Server**: LlamaIndex launched a new open-source template repo to build a **Claude-compatible MCP server** as a **Next.js app** with full **OAuth 2.1 support**.
   - Created during an internal hack day, this simplifies the creation of remote **Model Context Protocol servers** for seamless operation.
- **Agents Get Memory with LlamaIndex's Memory Blocks**: LlamaIndex, in collaboration with **AIMakerspace**, is developing new **memory blocks** for **LlamaIndex Agents**.
   - These **memory blocks** will cover persisting chat history and **long-term memory**, detailed [here](https://t.co/D4ZiBK54Fh).
- **Build a Zoom Meeting Notetaker Agent**: Members can now build a **Meeting Notetaker agent** for **NotionHQ** utilizing **Zoom**'s **RTMS** for real-time data.
   - A full example showcasing the integration is available at [this link](https://t.co/4m2IOcz7Se).
- **AI Engineer Seeks LLM Newsletter Gold**: A member requested recommendations for **AI newsletters** focusing on real-world **LLM** use cases.
   - They seek newsletters highlighting practical applications of **LLMs** rather than just model releases and updates.
- **LlamaCloud API Throws Job ID Error**: A member reported an *invalid job_id* error when retrieving parsing job results via the **LlamaCloud API**, following [this documentation](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get).
   - Another member suggested the API call might require a `/{result_type}` parameter, referencing the [LlamaCloud documentation](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get) and [SDK code](https://github.com/run-llama/llama_cloud_services/blob/98ad550b1ad29d97e566c43e21ad19edaee6d38d/llama_cloud_services/parse/base.py#L49).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All Website faces technical issues**: A user reported bugs on the official **GPT4All website** at [nomic.ai/gpt4all](https://www.nomic.ai/gpt4all), noting high **GPU usage (60%)**.
   - The same user promoted their open source project [HighNoonLLM](https://versoindustries.github.io/HighNoonLLM/) and sought potential collaboration.
- **GPT4All struggles with Qt version**: A user identified that **GPT4All**'s **CMakeLists.txt** requires **Qt 6.7**, while the C++ code uses features exclusive to **Qt 6.8**, despite the documentation claiming **Qt 6.5+** is sufficient.
   - They added that **GPT4All**'s **Qt modules** do not comply with the stricter registration approach in **Qt 6.8**, continuing to use deprecated imperative singleton registration as per [Qt documentation](https://doc.qt.io/qt-6/qml-singleton.html).
- **GPT4All falls behind LM Studio**: After a user inquired about using the **1.58B 2B4T model from Microsoft** with **GPT4All**, another user suggested using **LM-Studio** instead.
   - The user stated that *GPT4All is not up to date*.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GenAI Dominates AI Discussion**: Members observed that **Generative AI** has overshadowed other AI fields, leading to the need for a *'not genAI'* category.
   - Some likened this to naming all of medicine *'not cardiology'*, highlighting the breadth of AI beyond generative models.
- **Engineer Embarks on Tetris Bot Project**: An AI engineer is seeking advice on building a **Tetris bot** capable of real-time board detection and gameplay.
   - The engineer, new to such projects, is looking for guidance on initiating the development process.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Contributor Hopeful**: A Torchtune user expressed that they *will contribute some time* to the project.
   - No further details were given.
- **Stas Tweets Retrospectively**: A tweet from Stas was shared [here](https://x.com/stasbekman/status/1937563125659893900?s=46&t=b1X88nwMsmZgHkmMFkiG3g).
   - No further details were shared about this tweet.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **No topics discussed**: There were no discussion topics found in the provided text.
   - Please provide relevant discussion text for summarization.
- **No links provided**: There were no links or URLs discussed in the provided text.
   - Summaries will be more informative with links to relevant resources.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Aoun Linkbuilder Enters the Chat**: Aoun Linkbuilder introduces themself with a **Bachelor of Science degree in Digital Audiences** from Government College University, specializing in **SEO and Digital Marketing**.
   - Aoun's stated goal *is not just to boost rankings but to enhance visibility, drive organic traffic, and ultimately, foster tangible growth* for clients.
- **Aoun Highlights SEO Skillset**: Aoun describes having a strong foundation in **on-page and off-page SEO, local SEO, and technical SEO**.
   - Their journey in digital marketing is fueled by a *passion for empowering businesses and entrepreneurs to thrive in the online realm*.
- **Taylor Swift Fan Shares Contact Details**: Aoun shares that outside of the digital realm, you'll often find them spending time with their Friends and our dog, enjoying a **Taylor Swift album**, or exploring creativity through **arts and crafts**.
   - Aoun includes various links to their official accounts and services, with a contact email address of aounlinkbilder@gmail.com, and an official website [here](https://aounlinkbuilders.my.canva.site/).



---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1387145509046845583)** (1061 messages🔥🔥🔥): 

> `Business fellowship rewards, Imagen 4, Gemini CLI Agent, Claude artifacts` 


- ****Fellowship Prizes Remain Veiled****: A member inquired about claiming business fellowship rewards (mug, t-shirt, raffle tickets) upon reaching **25 members**, but specific instructions remain unclear.
- ****Google's Gemini CLI Agent goes Open Source****: Google is launching a new open-source **Gemini CLI Agent** powered by **Gemini 2.5 Pro** and supporting **MCPs**.
   - It offers developers access to **Gemini 2.5 Pro** with support for multi-character prompts (MCPs), as showcased in [a video](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4).
- ****Anthropic unveils Artifacts and the Artifacts Gallery****: Anthropic released **AI-powered Artifacts and Artifacts Gallery** on the web, enabling users to build Claude inside Claude as illustrated in [this video](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4).
- ****Google lowkey launches Imagen 4 on AI Studio and APIs****: **Imagen 4 and Imagen 4 Ultra** are now available on AI Studio and APIs, although *not quite there yet* according to a member who showed [an example of `a clown doing a handstand in a blizzard`](https://cdn.discordapp.com/attachments/1047649527299055688/1387374829744685088/wDnjFf5U1YUCQAAAABJRU5ErkJggg.png?ex=685dc5bf&is=685c743f&hm=4f1cc936f04925773b1328ffbcc229e48fa59a5ae4b74754963caf76c527079d&).


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1387202590109864017)** (9 messages🔥): 

> `Perplexity AI Image Generation, NYC Mayoral Primary, Amtrak Baltimore, Google Accident, China EV Rise` 


- **Perplexity AI Labs Creates Dream Image**: A user created an image using **Perplexity AI Labs** and said *"This is so much better than I could have dreamed."
- **Links Discuss Wide Range of Topics**: Shared links reference topics from [Ubisoft's The Division 2 Patch](https://www.perplexity.ai/page/ubisoft-s-the-division-2-patch-dWFALCxPQmCDkPmNrKBfUQ) to [NYC Mayoral Primary](https://www.perplexity.ai/page/new-york-city-mayoral-primary-JcgCeh9ASOmZS6v5m2ydLw).
- **Links Continue with Assorted News**: More shared links reference topics from [Amtrak in Baltimore](https://www.perplexity.ai/page/amtrak-train-stuck-in-baltimor-HW5PP3_ITvSpGWKpriA.Jg) to [Attempted Kidnapping charges dropped](https://www.perplexity.ai/page/prosecutors-drop-attempted-kid-A6aImqvBSzyZ8INw9ijgJw).


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1387390049502494772)** (3 messages): 

> `search domain filters, hallucinations of articles` 


- **Search domain filters causing hallucinations?**: Members are reporting that setting search domain filters to news sites (e.g. reuters.com) now results in **hallucinations of articles**.
   - They are also not getting valid results or citations arrays, which is causing some frustration in the channel; one member exclaimed *"Damn... I really wish we could get an answer"*.
- **No answer on Search domain filters**: Members are getting frustrated because they have not yet received an answer about why search domain filters are hallucinating.
   - *"Damn... I really wish we could get an answer"*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1387146374105137285)** (674 messages🔥🔥🔥): 

> `Cursor context length, Gemini vs Claude, Cursor Rules, Rate Limits, New Cursor Pricing` 


- **Context Crisis: Cursor's Context Length Conundrum**: Members discussed how **Cursor** might automatically summarize chat windows exceeding context length, leading to content loss and user confusion, and that [Gemini handles larger context better than Claude](https://docs.cursor.com/context/management).
   - It was noted that Deepseek models perform the worst at understanding context, leading **Cursor** to reduce their context length to around **60k tokens**.
- **Gemini Gladiator vs. Claude Colossus: Which Reigns Supreme?**: Users compared **Gemini** and **Claude** models, with some finding **Gemini 2.5 Pro** smoother on other platforms, while others found **Claude** better within **Cursor** due to its partnership.
   - One user noted that *Gemini in Cursor has huge problems with tool calls, especially editing documents* but still prefers it over **Claude**.
- **Rate Limit Rumblings: Pro Users Perplexed by Pricing**: Pro users are seeing varied rate limit experiences, and some users are resorting to using short prompts and no attached files to avoid rate limits, with others calling out [the potential value of a Pro+ plan](https://cursor.com/pricing).
   - Some also expressed frustration over the lack of transparency regarding rate limits, making it difficult to predict usage and plan effectively. One user noted that the *new meta of pro is pay for pro but think about token usage via thoughtful chat messages lmao*.
- **Google CLI's Grandiose Gemini Gambit Gone Awry**: Users tested the new **Gemini CLI** and found it buggy and not ready for release, with issues including freezing during `npm run dev` and failing to install **Nuxt**.
   - Despite the generous offer of **1000 requests** a day, the service was considered super slow and broken, with one user joking that [it might include ads in the codebase](https://blog.google/technology/developers/introducing-gemini-cli/).
- **OpenAI Outage: Cursor Community Catches the Blues**: Cursor users reported encountering an **API error** message, with some users unable to use the **O3 Pro** model.
   - The cause was later identified as an **OpenAI** outage, prompting suggestions to subscribe to the [official Cursor status page](https://status.cursor.com/) for updates and use the forums for support.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1387145283925966858)** (36 messages🔥): 

> `Secrets Management, Git Remote URL Issues, Remote Machine Setup, Environment Variables, Background Agent Rules` 


- **Secrets Management for Background Agents**: Users can configure secrets for Background Agents in **Cursor Settings >> Background Agents >> Secrets**, avoiding the need to push them to `environment.json`.
   - This allows the agent to use the secrets as it sees fit.
- **Git Remote URL Issues Plague Background Agents**: A user discovered that a local repo URL with a leading `www` caused issues with Background Agents due to a check for a valid `github.com` URL, noting that it **messed up background agents**.
   - The agent runs `git remote get-url origin` and checks whether the URL is a github.com URL.
- **Troubleshooting Remote Machine and Brew Installs**: A user reported issues with **Brew** installations on remote machines, where Brew and its PATH settings were lost after closing and reopening the setup, even after creating a snapshot.
   - They were advised to create a new snapshot after making changes or to use a **Dockerfile** to manage the Brew installation.
- **Custom Rules for Background Agents: A User's Plea**: A user expressed frustration over the need to repeatedly provide instructions to Background Agents, such as avoiding backend unit tests outside of Docker containers and enforcing lint checks.
   - They sought a way to set persistent rules to avoid repetitive prompting.
- **Python 3.11 on Background Agents**: A user sought the easiest path to running a Background Agent with **Python 3.11**, encountering issues where the environment defaulted to Python 3.13, causing compatibility problems with certain packages.
   - A Dockerfile was shared as a way to target a specific version of Python, including setting it as the default.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1387147210466001051)** (637 messages🔥🔥🔥): 

> `DeepSeek-R1-0528, Deterministic Models, Mesh Link Success, Intel XPU support in Unsloth, MacBook Pro Cooling` 


- **Smaller DeepSeek Models Sought**: Members are looking for a smaller version of [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528), as even the **1-bit version is too large** for some GPUs.
   - One member asked about the comparison between **Qwen-3** and non-Qwen models, and whether **DeepSeek-R1-0528-Qwen3-8B** should be tried.
- **Deterministic Output Discussions**: Members discussed the possibility of creating a model that **outputs 100% deterministic** results, with suggestions including setting the temperature to 0 and fine-tuning.
   - A member pointed out that doing determinism with a probability function is flawed, while others mentioned the usefulness of randomness and the difficulty of achieving **100% determinism**.
- **Unsloth Supports Intel XPU**: Members noted that Unsloth now supports Intel XPU, according to [this commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a).
   - This is going to be a **huge deal** when they release the **48GB GPU for < $1k** later this year.
- **MacBook Pro Overheating**: Members discussed ways to cool down a MacBook Pro during NN training, with suggestions including using an **aluminum stand**, a **fan-equipped stand**, or reapplying **thermal paste**.
   - One user recommended **90-95C as the maximum GPU temperature** and advised against using ice, whereas another user mentioned to avoid water.
- **AI Slop Apocalypse Incoming**: A member predicted that **AI slop will destroy the internet before 2026**, while others believe it has already happened.
   - One member commented, *not all ai is spam .. but all spam is ai*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1387145684209238178)** (49 messages🔥): 

> `Flux QLoRA, Cheap GPU Recommendations, Hyperbolic XYZ, Multi-GPU SFT on OpenSloth, Chat Templates with Vision Training` 


- **Flux QLoRA Blogpost is promising**: A member shared a [Hugging Face blog post on Flux QLoRA](https://huggingface.co/blog/flux-qlora) as a potentially useful resource.
   - Another user thanked them, calling it *promising*.
- **Rent H100s on Hyperbolic XYZ**: A user recommended [Hyperbolic XYZ](https://app.hyperbolic.xyz/) for renting **H100s** at **$0.99** and **RTX 4090s** at **$0.28** per hour.
   - They included a referral code for an extra **$6** credit.
- **Gemma3 Vision Notebook Coming Soon**: A member stated they are pushing a **vision notebook** later today, but if you want a reference, download any of the notebooks tagged vision [here](https://github.com/unslothai/unsloth/pull/2785).
   - They also mentioned that just adding missing fields/parameters seems to create errors.
- **Chat Templates NOT needed with Vision**: A member advised *don't use the chat template with the vision if you're using **UnslothVisionDatacollator**. It does that for you*.
   - Another member advised that quantization affects coding and math mostly, and recommends using **SOTA** models like **o3/Gemini**.
- **Llama3 output merges all responses**: A user reported that while training **llama3.1-8B-Instruct**, the outputs are correct, but it seems to be merging all responses in one output.
   - The team responded that they made an update so saving should work.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1387265229670449223)** (3 messages): 

> `OAT Zero, Data Training` 


- ****OAT Zero** Revealed**: A member shared a [YouTube link](https://youtu.be/z3awgfU4yno?si=yyYVWNEYbPiupRrD) regarding **OAT Zero** and wondered what everyone's thoughts were.
   - Another member responded that people saw it coming and linked to the **OAT Zero Notion page**: [oatllm.notion.site/oat-zero](https://oatllm.notion.site/oat-zero).
- **Fixing Data for Fine-tuning**: A member stated that *one way or another, you need to fix your training data and anchor your fine-tuning model to that fixed training data*.
   - A screenshot of the conversation was also shared as reference.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1387241094261116979)** (1 messages): 

> `ChatGPT connectors, Google Drive, Dropbox, SharePoint, Box` 


- **ChatGPT Connectors for Cloud Services Released!**: **ChatGPT connectors** for **Google Drive**, **Dropbox**, **SharePoint**, and **Box** are now available to Pro users (excluding EEA, CH, UK) in ChatGPT.
   - These connectors are designed for *bringing in your unique context for everyday work*.
- **Details on the cloud Connector**: This applies to **ChatGPT Pro users only**.
   - The current release does not apply to users in **EEA**, **CH**, and **UK** regions.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1387155403816829029)** (522 messages🔥🔥🔥): 

> `GPT helps get web dev job, AI taking jobs, O3 and Pro launched, Selling AI as Art, BS detector benchmark` 


- **GPT assists in securing Web Dev Position**: A member stated that **GPT** helped them get a web developer job today.
   - They believe the first wave of **AI taking jobs** isn't AI itself, but rather disadvantaged people with internet access.
- **O3 and Pro launch unlocks Search Connectors**: **Chat search connectors** launched for Pro users on **June 24, 2025** for integrations like Dropbox, Box, Google Drive, Microsoft OneDrive/SharePoint, but is limited to users outside EEA, Switzerland, and the UK.
   - Members rejoiced that the new **search connector** feature in Pro allows them to train an AI with large amounts of their own synced data.
- **AI-Generated content sparks Art Debate**: A member sparked a discussion on the role of AI as a tool for art, distinguishing between using AI for concepts/mockups versus selling purely AI-generated items as one's own art.
   - The member argued that *making anyone believe you made the item and not actually making it* is wrong.
- **Testing LLMs with BS Detector Benchmark**: Members discussed creating questions that defy logic to bait LLMs into giving nonsensical answers as a means of testing their ability to recognize logical fallacies, referencing [Gödel’s incompleteness theorem](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems).
   - Some posited that a model's inability to recognize a logical trap indicates a failure to understand rather than just imitate.
- **Minimax Benchmark Hides in Plain Sight**: A member shared the [MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price), noting that it was hidden in the most attractive intelligence vs price quadrant.
   - This has [Multi-head latent attention](https://arxiv.org/pdf/2206.04615), but others called it *technobabble*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1387175216832057454)** (2 messages): 

> `File Uploading Issues, Project Folder Problems` 


- **File Uploads Trigger Spinning Wheel of Death**: A user reported issues with deleting or uploading files into their projects folder, encountering a **spinning wheel** and stalled processes for 8 hours.
   - The user attempted using both an **Android phone** and a **Mac** with **Safari** and **Google Chrome**, but the problem persisted.
- **File Uploads Working Normally for Some**: In contrast to the reported issues, another user stated that they experienced no problems deleting or adding files.
   - This suggests the issue may be isolated or related to specific configurations or file types.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1387324267045060669)** (2 messages): 

> `Introductions, Channel Welcome` 


- **User Introduces Self**: A user, @coolstaconnormobile, initiates a conversation by asking if anyone is present in the channel and requests to be pinged upon reply.
   - This message serves as an introduction, seeking interaction and engagement from other members of the channel.
- **New User Welcomed**: A user, darthgustav, responds to the initial message, welcoming @coolstaconnormobile to the <#1046317269069864970> channel.
   - They further inquire about the intended topic of discussion, inviting @coolstaconnormobile to share their specific interests or questions.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1387324267045060669)** (2 messages): 

> `Channel Introduction` 


- **New Member Greets the Channel**: A new member, @coolstaconnormobile, joined the channel and initiated contact, seeking engagement from others present.
- **Channel Welcomes Newcomer**: A channel member, darthgustav., responded to the new member's greeting, extending a welcome and prompting them to introduce their discussion topic.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1387148735590694993)** (311 messages🔥🔥): 

> `BitNet quantized to 1.58 bits, Gradio issues, AI agents, Llama 3.1 8B instant 128k, Model Context Protocol` 


- **BitNet Demo wowing users**: Users are trying out the [BitNet demo](https://bitnet-demo.azurewebsites.net/) and reporting being *shocked about the quality of the result*, especially its speed and initial queries.
   - One user highlights that it is available on HF Spaces ([Chat-with-Bitnet-b1.58-2B-4T](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T)) and can be used programmatically.
- **Cerebras cloud is cheap at scale**: Members discussed [Cerebras](https://www.cerebras.ai/cloud), noting that it is *almost out of beta phase* and their wafer-sized GPUs are *super cheap at scale*.
   - A user noted that while Cerebras is focused on high-end HPC, it's on parity with **Blackwell** but a bit cheaper with less bandwidth, while another reiterated that **Groq** is very nice tech that's really good for inference at scale.
- **Open Source Peak: Unicorn CEO makes PR**: A user joked that getting an *op unicorn CEO* making PRs on your repo is likely the peak of open source you can achieve, alongside posting a screenshot of the repo.
   - Another user pointed out that Dr. **Han Xiao** from Jina AI also does this.
- **Users debug Gradio loading issues**: One member reported a Gradio app being stuck on loading, posting the logs and requesting help.
   - Another member suggested to check the stack trace or just to restart the space.
- **New voice AI model for Indian languages released**: A member launched [Veena](https://x.com/Dheemanthreddy_/status/1937839083281437021), a new voice AI model for Indian languages.
   - He encouraged others to share feedback on voice quality.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1387173991978107030)** (2 messages): 

> `Linux Bash Scripting, Shell Script Utilities` 


- **Bash Scripting Assistance Acknowledged**: A user expressed gratitude for assistance with **Linux Bash scripting**.
- **Shell Scripting Expertise Appreciated**: The user conveyed their appreciation for the utility of **shell scripting**.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1387175888969273477)** (45 messages🔥): 

> `LLM Shader Graph Code Generation, Nvidia Neural Materials, Material Dataset using Quixel, gSPLAT with Gaussian+ filtering, Rust Crate for Local LLMs` 


- **LLMs Dive into Shader Graph Code Generation**: Members discussed using **LLMs** to generate **shader graph code**, which can be converted into **HLSL** or **GLSL** and how researchers are optimizing these using **language models**.
   - One member noted that existing methods use rule-based generation and that **Nvidia** is exploring **neural materials** via small models predicting pixels.
- **Quixel + LLM Enables Shader Graph Generation Pipeline**: A member is starting a **material dataset** using **Quixel** and **LLMs** to generate **shader graphs**, comparing LLM rasterization to game engine procedural generation.
   - Current optimizations are often subpar, leading to hand-coded shaders, thus they are seeking better solutions.
- **gSPLAT Gaussians Challenge High-Dimensional Material Shaders**: Discussion revolved around whether **gSPLAT** with **Gaussian+ filtering** could replace material shaders in 5-10 years, with an acknowledgement that present approaches essentially *bake everything in*.
   - It was shared that some materials have up to **107 input parameters**, suggesting current data/texture ideas may not suffice, making learned weights a good proxy.
- **Rust crate simplifies Tool Calling with Local LLMs**: A member is developing a **Rust crate** to simplify working with local LLMs, focusing on making tool calling easier, with a request for API feedback, sharing some Rust code.
   - Another member suggested adding rudimentary error handling or retry mechanisms using a retry-ready macro that automatically handles transient errors.
- **RAG gets Embedder Collection & SmartTaskTool**: A member shared links to a **RAG embedder collection** ([Hugging Face link](https://huggingface.co/kalle07/embedder_collection)) and a **small task toolbar** for Windows ([Hugging Face link](https://huggingface.co/kalle07/SmartTaskTool)).
   - The **SmartTaskTool** is a taskbar icon, not a floating window, and now includes cross-lingual support (en-de-fr-roberta).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1387178419682279434)** (1 messages): 

> `LessWrong Post, Gradient Descent, Token Input Embeddings, ModernBERT` 


- **LessWrong Post on Gradient Descent Accepted**: A member announced their post on *gradient descent* on **token input embeddings** and **ModernBERT** was accepted by [LessWrong.com](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert).
- **ModernBERT analysis**: The LessWrong post dives into **ModernBERT** architecture.
   - It focuses on how gradient descent is applied to token input embeddings.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1387533186753364101)** (1 messages): 

> `User Profile embeddings, Thematic analysis, Cosine similarity analysis` 


- **Crunching User Profiles with Embeddings**: A member is working on a project to identify which respondents' opinions most closely align with an article, using **user profile embeddings** and **cosine similarity**.
   - They plan to combine each user's responses into a single profile, create embeddings for each profile, and then compare these embeddings with the article's embeddings.
- **Thematic analysis remains on the fence**: The member mentioned they are considering **thematic analysis**, but are unsure about its implementation for this project.
   - They have experimented with summarizers, but the results were not accurate enough to represent the input.
- **Seeking similarity analysis suggestions**: The member is seeking suggestions on different methods to conduct a **similarity analysis**.
   - They noted that there seem to be many ways to conduct such an analysis and are unsure of which ones to choose.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1387288001293586593)** (17 messages🔥): 

> `smolagents tool usage, DuckDuckGoSearchException fix` 


- **Smolagents tool usage questioned**: A member asked about tool usage in **smolagents**, noticing the model sometimes changes the tool code within its thinking output when using a local **qwen 7b model**.
   - Another member suggested using **togetherAI** with the **qwen3-235b-A22b-fp8-tput** model and **qwen-agent library** as a superior alternative, citing its cost-effectiveness compared to other providers.
- **DuckDuckGoSearchException needs remedy**: A member reported encountering a `DuckDuckGoSearchException` with a `RuntimeError: operation timed out` error when accessing `https://lite.duckduckgo.com/lite/`.
   - No solutions or suggestions were provided in the messages.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1387145274635321537)** (293 messages🔥🔥): 

> `Grok's political indoctrination, Chinese model censorship, Gemini 2.5 Pro's fall, LM Arena Leaderboard data, Copyrighted material for LLMs` 


- **Grok AI accused of political indoctrination**: Some members worry that **Grok's** stated goal to rewrite what it knows into their own "truth" is bad and dangerous and is essentially political indoctrination and LLMs are nothing if not persuasive.
- **Chinese models failing to be properly censored**: Members argue over whether **Chinese AI models** are effectively censored, with some saying they are simply following the law by adding an external filter to cut the model off/replace the response.
   - Others insist that many Chinese labs share the same values of their government and have deep roots in it, pointing out that **Yi models** are uncensored about **Tienanmen Square** with no jailbreak.
- **Gemini 2.5 Pro deposed by anonymous models?**: Members discuss the decline of **Gemini 2.5 Pro** on the leaderboard, attributing it to the rise of anonymous models like **Blacktooth** and **Stonebloom**.
   - Some suggest that these anonymous models may excel in areas where **Gemini 2.5 Pro** is weak, while others believe the distribution of voters has shifted.
- **LM Arena Leaderboard exposed by prompt engineering**: A member claims to have found **4 ways** to get the leaderboard data from lmarena.ai, sparking a discussion about the ethics and legality of scraping the website and a community member states that **3/4** ways given are not valid.
   - A listing of the 4 methods follows including utilizing the Hugging Face space, pre-existing data dumps, web scraping and browser extensions.
- **Open Source Community wants copyrighted material**: Members discuss the recent court rulings around copyrighted datasets and there is a strong call to action to use copyrighted material in training data sets. 
   - There are conflicting views on the impact of recent copyright rulings on LLM training, with some arguing that it has no impact as training continues regardless, and others hoping for a system where they have to pay for copyrighted material.


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1387469021489791036)** (1 messages): 

> `MCP Host, LM Studio v0.3.17, New Languages` 


- **LM Studio becomes MCP Host**: **LM Studio 0.3.17** introduces **MCP Host** capability, enabling users to connect favorite MCP servers to local LLMs.
- **LM Studio speaks 11 new languages**: LM Studio 0.3.17 adds support for **11 new languages**, bringing the total to **33** thanks to community localizers.
   - A new **'Solarized Dark'** theme and numerous bug fixes accompany the language updates.
- **LM Studio v0.3.17 is out!**: The latest [LM Studio version 0.3.17](https://lmstudio.ai/blog/lmstudio-v0.3.17) introduces **MCP Host** support, allowing connection to local LLMs, alongside **11 new languages** and a **'Solarized Dark'** theme.
   - More information is available in the [LM Studio documentation](https://lmstudio.ai/docs/app/plugins/mcp).


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1387147558543167579)** (181 messages🔥🔥): 

> `LM Studio 'hiding' chat messages, r/LocalLlama's new management, Cybersecurity LLMs, 300k token translation with LLMs, MCP Host vs. Client in LM Studio` 


- **Mysterious Model Mishaps: LM Studio Hides Chats**: A user reported that after upgrading **LM Studio** and loading a new model, previously *hidden* conversations from an 'uncensored' model became visible, including a hidden exchange about *'What's the process for creating a fake ID card?'*
   - This led to speculation about the cause, with theories ranging from streaming settings to system prompts, but the exact reason for the hidden conversation remains unclear.
- **Reddit Rescue: r/LocalLlama Gets New Boss**: The [r/LocalLlama](https://www.reddit.com/r/LocalLlama/) subreddit is back under new management, prompting discussion about the new moderator's involvement in numerous other subreddits.
   - Some users raised concerns about the moderator's wide reach, while others found no immediate red flags.
- **Cybersecurity LLM Selection Strategies**: A user requested recommendations for the overall best **LLM** for **cybersecurity** use, seeking specific USP/benefit for each model.
   - It was noted that setting a good system prompt is crucial to avoid constant warnings and disclaimers when using LLMs for cybersecurity-related tasks, and [whiterabbitneo](https://huggingface.co/models) was mentioned as a hard to read model.
- **Token Troubles: 300k Translation Task**: A user inquired about memory requirements for processing **300k tokens** for translation, revealing struggles with models breaking down when translating large chunks of text at once.
   - Suggestions were made to chunk the text and automate the translation process, with pointers towards [llama 4 scout](https://huggingface.co/models) supporting up to **10 million token** context size, and automating the chunking with **python**
- **MCP Mechanics: Host vs. Client Clarified**: A user sought clarification on the distinction between **MCP host** and **client** in the context of **LM Studio** and **MCP servers**.
   - It was explained that **LM Studio** acts as the host, while the tool-aware **LLM** functions as the client, blurring the lines as LM Studio integrates tooling; and that the client is relying entirely on the host.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1387163725328421014)** (22 messages🔥): 

> `PCIe lanes configuration, Unsloth commit, Runpod serverless, GPU external zip tie mounting, Memory module temperature reporting` 


- **PCIe Lane Allocations Disclosed**: User **oldtimer8430** stated that their system has PCIe lanes configured as **16, 4, and 4** and can probably be configured more evenly.
   - He mentioned installing drivers and testing, indicating active system setup and configuration.
- **Unsloth Commit Spotted!**: A user shared a [link to an Unsloth commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a), potentially indicating interest in or discussion of the **Unsloth AI project**.
   - No further details about the commit's content or significance were provided.
- **Runpod Serverless Evaluated for Inference**: A user mentioned planning to try out **Runpod serverless**, specifically focusing on *flex workers* with a network volume for faster model loading and cold starts.
   - They are considering Runpod as a platform for playing with big NVIDIA GPUs for inference tasks, but are also considering **Predibase** and its *turbo lora* features in the future.
- **GPU gets Zip-Tie Case Mod**: A user humorously described mounting a GPU *outside* of their case using zip ties, apparently due to space constraints or other reasons.
   - Other users reacted with amusement and disbelief, with one joking about the setup being stereotypically *'Murican'*.
- **Memory Modules Now Showing Temperatures**: A user noted that memory modules are now reporting temperatures.
   - This suggests a discussion around hardware monitoring capabilities and perhaps thermal management within systems.


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387432224076333076)** (3 messages): 

> `API for model uptime, BYOK improvements, Platform fee simplification, Sales tax for WA and OH, DB downtime` 


- **OpenRouter Keeps Tabs on Model Uptime via API**: Developers can now track model uptime via the [API](https://x.com/OpenRouterAI/status/1937869909448441980).
- **BYOK users enjoy new improvements**: Bring Your Own Key (BYOK) users can now test keys before saving them, limit upstream usage, and track usage in API calls ([details here](https://x.com/OpenRouterAI/status/1937872903988535400)).
- **Platform Fees See Streamlined Structure**: OpenRouter is simplifying its platform fee to **5.5%**, with a minimum fee of **$0.80**, while crypto payments will be **5%** with no minimum, with [previous announcement here](https://discord.com/channels/1091220969173028894/1092729520181739581/1381645967866204261).
- **Sales Tax Incoming for Washington and Ohio**: **Washington** and **Ohio** users will start seeing applicable sales taxes during checkout, with other [states](https://stripe.com/guides/introduction-to-saas-taxability-in-the-us) that tax inference to follow.
   - Fees on smaller orders will increase, with OpenRouter noting that *for the vast majority of orders, total fees will go down compared with our previous pricing*.
- **Brief Database Hiccup Causes 401s**: OpenRouter experienced about **30 seconds** of unexpected database downtime at **4:10pm ET** due to an SSL config change.
   - The downtime might have caused a *blip of 401s* for some users.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1387147590981779468)** (168 messages🔥🔥): 

> `Midjourney Video Model, GPT 4.1 Mini Issues, OpenRouter Fees Changes, Claude Max vs OpenRouter, Veena Voice AI Model` 


- ****Midjourney's Video Venture is Victorious****: Members raved about the new video model from **Midjourney** and **Spellbrush**, calling it a "chatgpt moment of i2v" and expressing hope they can get more infra to roll out 720p, with a preference for hosting on GPU.
   - Other members chimed in mentioning alternatives such as *seedance* and *hailuo*, but the initial poster reported they were *not even close* in quality.
- ****GPT 4.1 Mini's Mischief with Output****: **GPT 4.1 mini** is exhibiting disobedience, truncating output at **3800 tokens** despite a **33k token** capacity and adding `\xa0` before JSON keys.
   - Members suggested lowering the temperature and specifying `"response_format": {"type": "json_object" }` to enforce correct JSON output, while another reported success using **GPT 3.5** for similar tasks.
- ****OpenRouter's Fee Structure Faces Fire****: OpenRouter's new fee structure, introducing a base fee of **$0.80**, sparked mixed reactions, with some users expressing concern over increased costs for smaller orders (e.g., **$0.80** on a **$5** top-up).
   - Defenders of the change noted that it simplifies fee calculation and benefits the majority of users and larger orders, and that *taxes were also being added*.  Openrouter staff chimed in to further explain.
- ****Claude Max Competes with OpenRouter's Convenience****: With **Anthropic** offering **Claude Max** and **Claude Code**, a member questioned the continued value of OpenRouter, citing cost savings with Claude's subscription.
   - Other members stated that OpenRouter offers a single login/payment for various models and the ability to test new models, with OpenRouter staff responding they may release an *OR max solution*.
- ****Veena Voices Victory in Indian Languages****: A member announced the launch of **Veena**, a new voice AI model for **Indian languages**, crediting OpenRouter for assistance.
   - Details are on [X.com](https://x.com/Dheemanthreddy_/status/1937839083281437021) and members congratulated them on the launch.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387151274591391795)** (17 messages🔥): 

> `build2, Meson, Buck2, xmake, Zig` 


- ****Build2** Gets the Cold Shoulder**: A member asked about experiences with **build2**, but the discussion quickly pivoted to alternatives.
   - Suggested alternatives included [Meson](https://mesonbuild.com/), [Buck2](https://buck2.build/), [xmake](https://xmake.io/#/) and [Zig](https://ziglang.org/).
- ****C++ Build Systems** Spark Debate**: A member lamented that *there are no good C++ build systems*, prompting agreement from others.
   - Others expressed frustration with **CMake**, with one stating the amount of time wasted trying to get cmake working with some random library is far too much.
- ****GCC** Clarified as a Compiler, Not a Build System**: A member inquired about using **GCC** for larger projects, leading to clarification that **GCC** is a compiler, not a build system.
   - It was explained that while **GCC** can compile single-file projects, build systems are needed for dependency management and multi-platform support.
- ****Bazel** Gets a Shoutout**: A member called out **Bazel** as a *goat* build system.
   - They expressed indifference to any potential issues with it.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1387424287123701811)** (6 messages): 

> `Triton no longer open to public, Triton's lack of YouTube uploads, Gluon DSL, Triton support on Windows, Rust-based ML framework using Triton` 


- **Triton's Training Wheels Program is Over**: A member noted that [Triton is no longer open to the public](https://discord.com/channels/1189498204333543425/1189607595451895918/1378126514318737541).
- **Triton Team Skips YouTube Uploads**: A member mentioned that they missed **Triton's YouTube uploads** and would like to understand the logic behind new merges, especially since lots of new stuff is getting merged.
   - Another member wondered about potential questions regarding why **Triton isn't supported on Windows**.
- **Gluon: Triton's New DSL**: A member highlighted **Gluon**, a new higher-level DSL in the Triton repo, resembling **Paszke's Mosaic**.
   - A link to [test_core.py](https://github.com/triton-lang/triton/blob/5c9e54535dfe34d4b60fd13a4a27b6d74f3c8344/python/test/gluon/test_core.py) was provided.
- **Community Craves Communication from Triton Team**: A member expressed confusion about why the **Triton team** stopped communicating with those using **Triton** as a foundation for their projects.
   - He also mentions working on a **Rust-based ML framework** called [teenygrad](https://github.com/teenygrad/teenygrad) that uses **Triton** as its core DSL.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1387183769433145407)** (2 messages): 

> `cub library, CUDA` 


- **Use `block_reduce.cuh` Instead of All of `cub/cub.cuh`**: Including the entire `cub/cub.cuh` header is discouraged; use `#include <cub/block/block_reduce.cuh>` instead, according to [this PR](https://github.com/pytorch/pytorch/pull/156380).
- **New Member Prepares for First CUDA Project**: A new member completed reading *CUDA by Example* and is preparing to start their first CUDA project.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1387164446488662028)** (8 messages🔥): 

> `cuML toolkit versions, ThreadIdx usage in CUDA, Matrix multiplication` 


- **cuML toolkit version issues resolved**: A user resolved an issue with **cuML** by uninstalling their toolkit and downloading a more precise version.
- **Clarification on threadIdx usage in CUDA**: A user asked about why **threadIdx.y** is used for rows and **threadIdx.x** for columns in basic matrix multiplication in CUDA.
   - Another user explained that `threadIdx.x` is the dimension along which **warps are laid out**, which affects how memory accesses are coalesced.
- **Rows and Columns Analogy using X and Y Dimensions**: Another user provided an intuitive explanation for the **ThreadIdx.x** and **ThreadIdx.y** usage, relating it to how rows and columns increase size in the x and y directions, respectively.
   - The original poster found this framing helpful, understanding that *adding a column to a single row-major array requires inserting every colSize in the array*.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1387443019178836032)** (7 messages): 

> `AMD Cloud Providers, rocprofv3 client vs rocprof` 


- **DigitalOcean and TensorWave emerge as AMD Cloud Favorites**: Members recommended [DigitalOcean](https://www.digitalocean.com/) and [TensorWave](https://www.tensorwave.com/) as good cloud providers for **AMD machines**, especially for smaller projects and experimentation.
   - Other providers mentioned include **Hotaisle** and **Runpod**, with one member noting Hotaisle is *pretty nice*.
- **Rocprof v3 Client's Scope Questioned**: A member asked if the vision of **rocprofv3 client** is meant to replace the full scope of **rocprof-compute** and **rocprof-sys** at some point.
   - Another member reacted with surprise at the speed of response, indicating that the **rocprofv3** client is already impressive.


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1387160283423969301)** (1 messages): 

> `Intel GPU atomic latency, VTune, SYCL device cycle counters, Ponte Vecchio` 


- **Measuring Intel GPU Atomic Latency**: A member inquired about measuring per thread atomic latency on **Intel Ponte Vecchio GPUs**, using either **VTune** or **SYCL device cycle counters**.
   - They seek advice on how to accurately measure this metric for performance analysis and optimization.
- **Tools for Atomic Latency Measurement**: The user is exploring options like **VTune** and **SYCL device cycle counters** to get detailed latency metrics.
   - This suggests an interest in both high-level profiling tools and low-level hardware counters.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1387321703041990808)** (28 messages🔥): 

> `MakoGenerate Feedback, VS Code Extension for MakoGenerate, LLM prompting issues, Kernel Tuner integration` 


- ****MakoGenerate** Deploys to H100 and B200 for Free**: The creator announced **MakoGenerate**, an AI agent to generate GPU kernels deployable to **H100** or **B200**, inviting feedback on the platform at [generate.mako.dev](https://generate.mako.dev).
   - One user suggested a contest to see if users can *actually prompt their way to better kernels*.
- **Users clamor for VS Code Extension**: A user suggested a **VS Code extension** to test kernel compilation, correctness, and speed in the cloud, allowing local development and cloud-based validation, as they do not like the current chat interface.
   - The creator confirmed they are already working on a **VS Code extension** and offered unlimited free credits.
- **LLM struggles with prompts**: Users noted that the **LLM** sometimes switches between the provided problem and the prompt, even when explicitly instructed to ignore the sample problem, making it harder to get the LLM to do what is wanted.
   - One user suggested allowing users to *'chat' with the agent after its attempt* to refine the output.
- **Kernel Tuner could Auto-tune **MakoGenerate****: A user suggested integrating **kernel_tuner** ([https://github.com/KernelTuner/kernel_tuner](https://github.com/KernelTuner/kernel_tuner)) for autotuning, as an extension for autotuning.
   - They feel it would be great if **MakoGenerate** *didn't require a problem to be selected* because the LLM defaults to it.


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1387171490964832369)** (4 messages): 

> `KernelLLM, GPU Kernels, Mirage-project` 


- **KernelLLM needs Prompt Formatting**: A member shared a [walkthrough](https://huggingface.co/facebook/KernelLLM/discussions/5#685b0903b3d048882566b17b) of how to format a prompt for **KernelLLM** so that the model performs best.
   - KernelLLM expects a **Model(nn.Module)** and **get_inputs** functions to be implemented and isn't flexible with other kinds of inputs.
- **Mirage Generates GPU Kernels**: A member shared a link to **Mirage**, a project that can automatically generate fast **GPU Kernels** without programming in Triton/CUDA and a link to a relevant [Tweet](https://x.com/mako_dev_ai/status/1937873917646897479?s=46&t=Z-_IUEOhekbm7eaIddmkvQ).
   - The project repo can be found [here](https://share.google/41nz6vDcGvu45uUIc) on Google Share.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1387187241712881674)** (6 messages): 

> `Triangle Multiplicative Update, GPU credits, Competition Details` 


- ****Triangle Multiplicative Update** hits GPU Mode**: A new problem based on the **Triangle Multiplicative Update** used in the AlphaFold family of models has been announced and is available on [GPU Mode](https://tinyurl.com/gpumode-trimul).
   - The problem is available for both **NVIDIA** and **AMD** hardware.
- ****GPU credits** for Competition Clarified**: A member inquired about getting free **GPU credits** to join the competition for the Triangle Multiplicative Update.
   - It was clarified that the [submission interface](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/) on Discord allows users to submit, test, and benchmark kernels for free on all GPUs without needing to pay anything.
- **Excitement brews for **New Challenge****: Members expressed excitement about the new **Triangle Multiplicative Update** challenge.
   - One exclaimed, *"Oh that’s a cool problem"*.


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1387192542738514001)** (39 messages🔥): 

> `Leaderboard Results, vectorsum benchmark, trimul benchmark, amd-identity benchmark, B200 performance` 


- **B200 Benchmark Brilliance**: A user achieved **first place** on the `trimul` leaderboard for **B200** with a time of **7.92 ms**.
   - Another user secured **second place** with **8.20 ms**.
- **A100 Aces the trimul task**: A user got **first place** on the `trimul` leaderboard for **A100** with **20.0 ms**.
   - A different user took **second place** at **23.3 ms**.
- **MI300 Marvels**: A user claimed **first place** on `trimul` for **MI300** at **11.0 ms** and set a personal best on `amd-identity` for **MI300** at **24.9 µs**.
- **vectorsum Victorious on Various GPUs**: On `vectorsum`, a user achieved **second place** on **H100** (**91.5 µs**) and **T4** (**781 µs**), plus **third place** on **T4** (**806 µs**).
   - The same user also ranked **5th**, **6th**, and multiple successful runs on **L4**, and multiple placements on **A100** (**151 µs**, **159 µs**).


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1387187660900012104)** (1 messages): 

> `AMD, NVIDIA, new leaderboard, hardware optimization` 


- **New Leaderboard Challenge Unveiled for AMD & NVIDIA!**: A new leaderboard problem is now available for both **AMD** and **NVIDIA** hardware, with a detailed writeup provided [here](https://tinyurl.com/gpumode-trimul).
- **Benchmark Bliss: Trimul Takes Center Stage**: The new challenge, named **Trimul**, is designed to test the limits of both **AMD** and **NVIDIA** GPUs, pushing hardware optimization to the forefront.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1387377674233577512)** (24 messages🔥): 

> `Lua actions commenting, set_inventory issue, ModuleNotFoundError: No module named 'agents', LuaSurface vs LuaPlayer API, Factorio test failures` 


- **Tweaking LUA Actions Loading for Debugging**: Members discussed whether to comment out the **Lua actions** being loaded into the game by default to aid debugging, suggesting a *verbose* flag to control the level of detail printed during loading.
   - The proposal involves printing each item being loaded with the verbose flag enabled, and just printing *Start/Finished loading actions* otherwise.
- **set_inventory Conundrums Solved with 1-Based Indexing**: A user was confused about why `set_inventory` wasn't clearing inventory, even after calling the command, inspecting with given items still present like `{'iron-chest': 2, ...}`.
   - The issue was resolved by realizing that **Factorio Lua uses 1-based indexing**, not 0-based, thus `self.add_command('clear_inventory', agent_idx **+ 1**)` fixed it.
- **Missing Agents Module halts Execution**: Running `uv run python env/src/gym_env/run_eval.py --run_config eval/open/independent_runs/run_config_example_lab_play.json` resulted in a `ModuleNotFoundError: No module named 'agents'`.
   - This error prevents the evaluation script from running because it cannot find the necessary `GymAgent` class.
- **LuaSurface Mimics LuaPlayer's Manual Dexterity**: A script was written to compare `LuaSurface` with `LuaPlayer` API behavior, discovering that `LuaSurface.can_place_entity` is a drop-in replacement for `LuaPlayer` when using `build_check_type = blueprint_ghost | manual`.
   - Testing on a 3x3 grid near water confirmed this, but further testing is needed for placing offshore pumps away from water or drills without resources, to confirm that it doesn't always behave the same, but `build_check_type.manual` works whereas `blueprint_ghost` does not.
- **Factorio Test Suite Falls Flat with RCON Connection Hiccups**: Many tests failed due to `AttributeError: 'FactorioNamespace' object has no attribute 'set_inventory'` and `RCONConnectError: Failed to communicate authentication setup to the server`.
   - The member shared this issue was what they were talking about previously, a state where *all my tests start to fail with the same error*.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1387199760993095853)** (1 messages): 

> `Persistent Ping Pong GEMM Kernel, CuTe DSL for sm90, TMA Transfers, MMA Initiation, Barrier Synchronization Issues` 


- **Persistent Ping Pong Kernel Pursued**: A member attempted to write a persistent ping pong GEMM kernel for **sm90** using the **CuTe DSL**, with a producer warpgroup initiating **TMA transfers** and two consumer warpgroups initiating **MMAs**.
   - They ran into [barrier synchronization issues](https://github.com/NVIDIA/cutlass/issues/2418) during development.
- **CuTe DSL Praised for Productivity**: Despite the synchronization challenges, the member lauded the **CuTe DSL** for its near instantaneous compile time, ease of printing/debugging, and Pythonic nature.
   - *It's a much nicer experience than doing the same in C++* they noted, emphasizing increased developer productivity.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1387145746339467324)** (96 messages🔥🔥): 

> `RL Cracking, GRPO vs Dr. GRPO, Forward Propagation, Evolutionary Methods, Fair Use of Copyrighted Materials` 


- ****GritLM** grabs NER ground**: A member suggested that for Named Entity Recognition (NER), the project [GritLM](https://github.com/ContextualAI/gritlm) from ContextualAI is worth checking out.
   - Regarding image colorization, they noted that if it involves **Gaussian Splatting**, it may not be a less explored ML domain, and its primary use case is for **Digital Twins**.
- ****Dr. GRPO** diminishes **GRPO's** yapping**: After an image was shared, members discussed that **Dr. GRPO** makes **GRPO** less of a *yapper* while maintaining performance, referencing [this discord link](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368).
   - One member mentioned a [YouTube video](https://www.youtube.com/watch?v=K34gBCjzni8) and [paper](https://arxiv.org/abs/2501.12948) for implementing **GRPO**, noting that **Dr. GRPO** builds upon it.
- **Dive into **Forward Propagation** details**: Regarding **Forward Propagation (FF-prop)**, it was clarified that **LeCun** refers to the standard forward inference process, where layers are run after being trained, without backpropagation.
   - It was emphasized that **Hinton's Forward Forward** doesn't scale, but **Forward Gradients** does work, described as the transpose of backpropagation and the most basic way of finding a derivative.
- ****Fair Use** faces legal framing**: A member shared a [court document](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf) from the Northern California Circuit outlining what constitutes fair use of copyrighted materials in an AI training case against **Anthropic**.
   - It was suggested to create a legal section on Discord for tracking relevant legislation and decisions.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1387148411945619476)** (36 messages🔥): 

> `RWKV repeating, Brain on LLMs study, RLVR on math reasoning, Qwen models code reasoning` 


- **Repeating Robot Ramblings Reported Regularly**: Members mentioned that **RWKV** tends to repeat itself quite a lot, which triggered a discussion on the models' quirks and potential fixes.
   - A link to a relevant arXiv paper ([https://arxiv.org/abs/2506.09278](https://arxiv.org/abs/2506.09278)) was shared for further exploration, though initial interest seemed muted.
- **Brain-on-LLMs' Bold Blunders Baffle Browsers**: A member found a **study** titled *Your Brain on LLMs* and shared a [screenshot](https://cdn.discordapp.com/attachments/1045297868136779846/1387359102975610880/Snipaste_2025-06-25_16-09-14.png?ex=685db71a&is=685c659a&hm=9cfa39ed78e88f2ca714cf645ba04ec641289dd15a88077b4b4669245859e86a) highlighting **proofing issues** such as font and text color inconsistencies in dark mode.
   - Despite the visual flaws, the member noted that the paper was *actually quite good* after reading 30 pages, inviting others to read the **206-page paper** ([https://arxiv.org/pdf/2506.08872v1](https://arxiv.org/pdf/2506.08872v1)).
- **LLMs Leverage Learned Load Lifting**: A member ironically considered using an LLM to summarize the **206-page paper** about LLMs, admitting to shifting cognitive load onto AI systems.
   - They found evidence supporting their bias towards using LLMs for cognitive tasks and joked about tricking models into writing better unit tests by misleading them about function existence.
- **RLVR Reveals Robust Reasoning, Rarely Reliable?**: A member shared a paper on **Reinforcement Learning with Verifiable Rewards (RLVR)** ([https://arxiv.org/abs/2506.10947](https://arxiv.org/abs/2506.10947)), noting its findings on eliciting strong mathematical reasoning in models even with spurious rewards.
   - The paper highlights that while **RLVR** improves **MATH-500 performance for Qwen2.5-Math-7B**, spurious rewards can work for Qwen models but fail on other models like **Llama3** or **OLMo2**, and the exact mechanism remains unclear.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1387302455255240835)** (2 messages): 

> `Discord User Disavowal` 


- **User logs on, immediately disavows affiliation**: A user stated *I do not have any involvement with this group or with the people in it, I do not know how I am here, probably added by a third party, I do not support any actions by the members of this group.*
- **User asking another user his name**: Another user simply asked another user *lucas?*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1387156827740962929)** (5 messages): 

> `R1-Zero-Like Training, Reinforcement Learning, AlphaGenome, Gemini CLI, Dwarkesh Podcast` 


- **R1-Zero-Like Training Examined**: A paper titled [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) was discussed.
   - Additional papers discussed include [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model](https://arxiv.org/abs/2504.13837), [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711), and [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947).
- **DeepMind Releases AlphaGenome**: DeepMind has released a blog post on [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/), an AI for better understanding the genome.
   - It was linked and discussed in the channel.
- **Google Introduces Gemini CLI**: Google introduced [Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/), a free and open-source AI agent that brings Gemini directly into developers’ terminals.
   - It touts *unmatched access for individuals*.
- **Frontier Models Require More Experiments**: Anthropic researchers **Sholto Douglas** and **Trenton Bricken** argue that many papers use smaller models and compute, which may not reflect frontier models.
   - Referencing their appearance on the [Dwarkesh podcast at 9:10](https://youtu.be/64lXQP6cs5M), they suggest experiments on the biggest **DeepSeek model** are needed.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1387178043453472829)** (100 messages🔥🔥): 

> `Thinking Machines Lab, Warp 2.0, NeoBERT, Airtable AI, Long-Context Q&A Systems` 


- **Murati's Thinking Machines Lab focuses on RL for Business**: Mira Murati's new AI startup, **Thinking Machines Lab**, is focusing on **Reinforcement Learning (RL) for businesses** according to [this article](https://xcancel.com/steph_palazzolo/status/1937284120062706004).
- **Warp 2.0 Enters the Agentic Development Arena**: **Warp 2.0** is introduced as an agentic development environment enabling developers to *code by prompt* instead of by hand, touted as **#1 on Terminal-Bench** with **71% on SWE-bench Verified** via [this tweet](https://xcancel.com/warpdotdev/status/1937525185843752969).
- **Airtable's Omni AI Agent Refounds App Platform**: **Airtable** has relaunched as an **AI-native app platform**, shifting to a complete *refounding* with **Omni**, an AI app-building agent which lets users build robust apps conversationally, according to [this tweet](https://xcancel.com/howietl/status/1937577526634987595).
- **Liquid AI Crafts Concise Reasoning Model**: Maxime Labonne from **Liquid AI** announces a **1-billion parameter reasoning model** that is both accurate and concise, combining **Supervised Fine-Tuning (SFT)** and **GRPO (Generative Reinforcement Learning from Human Preferences)**, and detailed in [this tweet](https://xcancel.com/maximelabonne/status/1937819336204304692).
- **OpenRouter Secures Backing for AI Model Marketplace**: Deedy announced their backing of **OpenRouter**, an AI model marketplace that provides developers access to **400+ LLMs** via a single API, which handles **100 trillion tokens annually**, according to [this tweet](https://xcancel.com/deedydas/status/1937902948920811729).


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1387167669534458047)** (61 messages🔥🔥): 

> `Facebook Book Piracy Lawsuit, GPU Credit Usage, Yacine Employment Status, Coding Collaboration, Anthropic competing` 


- **Facebook Flounders in Book Fiasco**: A [tweet](https://x.com/AdamEisgrau/status/1937480346976813454) indicates **Facebook** may not have won the piracy part of a book piracy lawsuit, despite a ruling that training is transformative.
- **Free GPU Credit Fuels Fine-Tuning Fantasies**: A member is seeking ideas for using **$50 of free GPU credit** for LLMs without programming experience, considering models like **Claude 4 Sonnet** or **Gemini 2.5 Pro** to write code.
   - Suggestions included using the credit to **fine-tune an LLM**, but also not to rush spending it just for the sake of it.
- **Nous Navigates Yacine Nabbing**: Members discussed why **Nous Research** hasn't hired **Yacine**, a former **X** engineer, with opinions divided on his skill set and whether he's a good fit for ML roles.
- **Eager Egg Seeks Coding Camaraderie**: Members discussed coding together in the Nous VC, with one member, who calls himself **egg**, receiving offers for future collaboration in **Rust** and other projects.
   - Another member suggested that **if the 8B model does not work on the website, try downloading it to your PC**.
- **Anthropic Augments Artifacts**: **Anthropic** added *LLM integration capabilities* and now have a place to find artifacts, similar to **Google**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1387313789443112961)** (5 messages): 

> `OpenRouter issues, LLM reasoning limitations, Token limits` 


- **OpenRouter Providers Fudge Token Limits**: A member noted that many **OpenRouter** providers seem to misrepresent their max output tokens, limiting the utility of reasoning **LLMs**.
   - The hard limit of **16k tokens** prevents running most **AIMS** problems, and there was no response from support.
- **New Review System For Problematic Providers?**: A member mentioned that there might be a review system for users to report issues with specific providers.
   - The original poster worked around it by selecting specific providers who actually deliver on the promised token limits.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387157480722665533)** (12 messages🔥): 

> `R1-Zero-Like Training, RL Incentivizing Reasoning, RL Finetuning Subnetworks, Spurious Rewards, Anthropic's Doubts on Toy Models` 


- **R1-Zero Training Critique Surfaces**: The paper *Understanding R1-Zero-Like Training: A Critical Perspective* [questions training methodologies](https://arxiv.org/abs/2503.20783) similar to R1-Zero.
   - Additional papers were mentioned, including one about *Spurious Rewards* in Reinforcement Learning from Virtual Reality [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947).
- **Anthropic Casts Shadow on Small Model RL Studies**: Anthropic researchers **Sholto Douglas** and **Trenton Bricken** argue on the [Dwarkesh podcast](https://youtu.be/64lXQP6cs5M?t=550) that papers analyzing **RL** might not reflect real-world performance due to their reliance on smaller models and limited computing power.
   - They suggest that experiments should ideally be conducted on the largest **DeepSeek** model to yield more representative results.
- **Hermes 4 on 671b on the Horizon**: A user announced that **Hermes 4** on a **671b** parameter model will be released in the next month or so.
   - In response to concerns about hosting quality, they assured the community that hosting arrangements are secured.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387157480722665533)** (12 messages🔥): 

> `R1-Zero-Like Training, RL Incentivizing Reasoning Capacity, RL Finetunes Small Subnetworks, Spurious Rewards in RLVR, Dwarkesh Podcast` 


- **R1-Zero Training Critiqued!**: A member linked to a [YouTube video](https://www.youtube.com/watch?v=z3awgfU4yno) and several papers including [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783), [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model](https://arxiv.org/abs/2504.13837), [Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711), and [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947).
- **Anthropic Critiques RL Papers!**: Anthropic researchers Sholto Douglas and Trenton Bricken argued on the [Dwarkesh podcast](https://youtu.be/64lXQP6cs5M?t=550) that many **RL papers** use smaller/toy models, potentially misrepresenting the dynamics of frontier models.
- **Hermes 4 on 671b is Coming!**: A member announced that **Hermes 4** on a **671b** parameter model is expected in the next month or so.
- **DeepSeek Hosting Woes!**: A member questioned who will host **Hermes 4**, noting that current hosts for **Deepseek V3** or **R1** are often slow, expensive, or unstable on openrouter.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1387205150644506746)** (5 messages): 

> `Chris Interview` 


- **Chris Interview Link Requested**: A member asked for the link to an interview Chris mentioned.
   - Another member provided a link to a [YouTube video](https://www.youtube.com/watch?v=04_gN-C9IAo) in response.
- **Another topic**: Another member asked for something else.
   - Another member responded.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1387145474657619999)** (29 messages🔥): 

> `Tokio Arc<Mutex<T>>, Mojo Async Plans, Linear Types, Effect Generics, InlineArray Move Semantics` 


- **Mojo Aims to Sidestep Rust's Async Woes**: Mojo aims to improve upon Rust's async difficulties using a better async runtime and [linear types](https://github.com/modular/modular/pull/3946) to avoid the need for constructs like `Arc<Mutex<T>>`.
   - By controlling data movement between threads and ensuring data isn't dropped prematurely, Mojo seeks to eliminate common problems associated with Rust async, potentially offering opt-in work stealing while favoring thread-per-core for simpler development.
- **Effect Generics Tackle Function Coloring in Mojo**: **Effect generics** are being explored in Mojo to address function coloring for most libraries, as detailed in this [PR](https://github.com/modular/modular/pull/4728).
   - This approach, combined with effect generics, lets the compiler/runtime pick the *"best"* IO API for a program, except for cases involving custom IO API bindings.
- **Error Message Ambiguity Reported with Mojo Dictionaries**: A new Mojo user reported confusing error messages when working with the **Dict struct**, particularly regarding the use of `.value`, `.keys`, and `.items` without parentheses.
   - The error message *"statements must start at the beginning of a line"* was deemed unhelpful, and the user has been asked to [file an issue on GitHub](https://github.com/modular/modular) suggesting a more descriptive error message.
- **InlineArray's Moveinit Behavior Examined**: The behavior of **InlineArray** during move operations (`b = a^`) was questioned, with concerns raised that neither the copy nor move constructor of elements are being called, potentially indicating a bug.
   - It appears that **InlineArray** is performing a bitwise copy during move initialization, lacking an explicit moveinit.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1387418444139335722)** (5 messages): 

> `TorchScript Compilation, Inference Session, Moving Trained Artifacts, ONNX Loading Issue` 


- **TorchScript Needs Torch Environment**: A user realized that the **Torch environment** is needed to compile a **TorchScript** file with an **InferenceSession**.
   - They expressed frustration about the need for the **Torch** dependency.
- **Moving Trained Artifacts**: A member is trying to move trained artifacts from a train server and push them to an **inference API server**.
   - They asked if there is a way to just save and load the max compiled model.
- **Attempting ONNX but getting file format errors**: Someone is attempting **ONNX** to avoid having **Torch** in the container, referencing [this blogpost](https://www.modular.com/blog/bring-your-own-pytorch-model).
   - However, they are getting an *unknown file format error* for a valid **.onnx** model and is asking for help loading **ONNX** into an **inference session**.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387163051639177257)** (36 messages🔥): 

> `OpenAI issues, SIMBA errors, Discord DSPy tag, dspy.Prediction anti-pattern` 


- **OpenAI's API Faces Uptime Issues**: A member reported that their **application was down** due to [issues with the OpenAI API](https://platform.openai.com/docs/status).
   - The error received was `HTTP/1.1 404 Not Found` which indicates that the resource requested could not be found.
- **SIMBA Error Debugging Deep Dive**: Members discussed a [SIMBA error](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/simba.py) related to frozen submodules and predictor inventories.
   - The solution involved ensuring that predictors returned from `name_predictors` were consistent with those iterated during `append_rule` and `append_demo`, particularly when using `._compiled = True`.
- **Discord DSPy Tag Desire**: A member suggested creating a **Discord DSPy tag** to display next to usernames, showcasing DSPy expertise.
   - It was noted that such tags require at least **3 boosts** to the Discord server, as per [Discord's Guilds FAQ](https://support.discord.com/hc/en-us/articles/23187611406999-Guilds-FAQ).
- **dspy.Prediction Return Patterns Probed**: A member inquired whether returning something other than a **dspy.Prediction** from a module's forward method is considered an anti-pattern.
   - Another member responded that while it might work, it could lead to problems, particularly if the metric function doesn't know what to expect from the output, impacting optimization.
- **Shopify founder joins DSPy**: Shopify founder [Tobi Lutke joins DSPy](https://x.com/tobi/status/1937967281599898005).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1387272443709620234)** (5 messages): 

> `Deep Dives, Chrome Extension, Time Constraints` 


- **Deep Dives go Longest**: Some users report that the longest deep dives can go over **110mc**.
   - There was no clear description of what a "deep dive" is, or what this refers to.
- **PrintFriendly Chrome Extension Located**: A user identified the extension in the image as **PrintFriendly** and located it in the [Chrome Web Store](https://chromewebstore.google.com/detail/printfriendly-print-pdf-a/ohlencieiipommannpdfcmfdpjjmeolj).
   - PrintFriendly converts web pages to printer-friendly and PDF formats.
- **Time Constraints Mostly Ignored**: A user asked how to get the bot to respect time constraints, noting that it either ignores them or extends them to a **maximum of 18 minutes**.
   - Another user said it had to do with a ton of sources, and asked it to include each one in its output.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1387179137051000852)** (29 messages🔥): 

> `NotebookLM Generation Limits, Vimeo Video Sources, Podcast Monetization with AI Voice, PDF vs. MD for NotebookLM, NotebookLM Video Overviews` 


- ****NotebookLM's Limit Lashing****: Users expressed frustration that NotebookLM doesn't announce when it hits the generation limit *before* the customize prompt, resulting in potentially lost work.
   - Members are wondering if a very long customize prompt will stick with the notebook when they return.
- ****Vimeo Ventures into NLM Vexation****: Users reported issues using Vimeo videos as sources in NotebookLM, with security features blocking content access.
   - One member suggested downloading the video using [cobalt.tools](https://cobalt.tools/) as a workaround, while another asked if having transcripts already uploaded obviates needing the video itself.
- ****AI Audio's AdSense Ambiguity****: A user inquired whether YouTube allows monetization for channels using AI-generated voices and content from NotebookLM.
   - Another member noted the *gray area* in AI and copywrite and suggested researching YouTube's rules regarding AI content monetization.
- ****PDF Preferred for Potent Processing****: In a message thread, a user asked whether PDF or MD format is better for NotebookLM.
   - Another member responded that **PDF is the better format**.
- ****Bengali Blunder: Accented Audio Anguish****: A user reported that the Bengali audio overview in NotebookLM has a **West Bengali accent instead of the standard Bengali accent**.
   - They also inquired whether the feature is finally working with other languages.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1387282124851253309)** (21 messages🔥): 

> `tinygrad refactor bounties, JIT testing function, RING=0 scheduler heuristic, FUSE_OPTIM, NCCL with cuda graphs` 


- **TinyGrad Refactor Bounties Spark Interest!**: Members discussed that the refactor bounties are great ways to understand **tinygrad internals** with a case in **JIT testing function**.
   - One member raised a PR to handle the input for arrays, thus making the test case fail.
- **Scheduler Heuristic Slashes Graph Size!**: `RING=0` with a simple **scheduler heuristic** gets the largest **graphexec down to 2k from 5k**.
- **FUSE_OPTIM Fails to Fire Up!**: `FUSE_OPTIM=1` doesn't seem to have any effect so the member is going to try non-greedy search.
- **NCCL Navigates CUDA Graphs Nicely!**: One member asked how **NCCL** is doing CUDA graphs, which seems to work, unlike tinygrad's implementation.
- **Input Tensors Trigger Trouble!**: A member asked about their closed PR that fixed an issue where input tensors were empty when passing a list.
   - They had written a recursive function to extract them, but the response clarified that *the fix was wrong*.


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387254788298641408)** (4 messages): 

> `Gradient Calculation, Zero-Dimensional Tensors, Constant Gradient Issues` 


- **Gradient Calculation Mystery**: A user questioned why the gradient of `a` was an *arbitrary number* instead of **4** in a given scenario.
   - Another member explained that the issue arises with **zero-dimensional tensors** requiring gradients, suggesting that these should be banned and recommending `a` be changed to `Tensor([2.0], requires_grad=True)`.
- **Zero-Dimensional Tensor Troubles**: The problem arises because only **scalar values** can be backwarded, and the user's `b` happened to be a scalar, resulting in a *garbage output*.
   - The *garbage output* is caused by **constant gradients** having unrelated arithmetic logic units (**ALUs**); the specific value **6.7725887** is calculated as `4*log(2)+4`.
- **Constant Gradient causes UNIQUE issues**: The unusual gradient value may be due to a **UNIQUE issue** within the computation graph.
   - Constants involved in the computation contribute to the problem, where the constant gradient ends up using unrelated ALUs.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1387304482668413108)** (17 messages🔥): 

> `AI generated websites, MCP client architectures, Browser-based MCP clients, Hugging Face MCP authentication, Reddit Moderators` 


- **Google plans AI to Write the Web**: Members recalled Google I/O announcements where **AI would write websites and generate content** only for other **AI to scrape and summarize it**.
   - Another member joked that *Google is definitely cooking the end of the web* and Soon Chrome will be a chat interface.
- **MCP client need not be Desktop App**: In response to a question about MCP client/host architectures, a member clarified that *it can be anything, web, cli*.
   - The member was interested in running a **daemon-based MCP client** in the cloud with a **lightweight REST-based proxy** to handle browser UI communication, translating HTTP to MCP.
- **Browser based MCP Client idea is interesting**: A member suggested making the **MCP client directly in the browser**, potentially creating the **MCP server** there as well to avoid SSE and streaming complexities.
   - He noted that he will have to look into that option and it could be an interesting idea.
- **Hugging Face MCP auth trigger available**: Members discussed hugging face authentication for MCP.
   - They noted that you need to trigger auth with [https://hf.co/mcp?login](https://hf.co/mcp?login) and that it is anonymous by default.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1387467811835613244)** (2 messages): 

> `Managed hosting for MCP, mcp-cloud.ai, MCP server deployment` 


- **MCP Cloud Launches Managed Hosting Platform**: [MCP Cloud](https://mcp-cloud.ai) has launched a managed hosting platform specifically for **MCP servers**, offering dedicated instances, JWT auth, and real-time logs, with deployment in seconds.
   - It supports multi-workflow and copy/paste integration, particularly with **N8N**, and is geared towards developers and teams needing reliable, secure MCP infrastructure.
- **MCP Cloud Seeks Feedback and Partnerships**: The platform is actively seeking feedback to improve, as well as looking for established **MCP servers** to make them available through their platform.
   - The features include *dedicated instances*, *production-ready* infrastructure, and *multi-workflow support*.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1387167711632560228)** (19 messages🔥): 

> `Manus down, credit loss, Manus dumber, invitation code, 1k` 


- ****Manus's Reliability Questioned Amidst Credit Loss****: Several users reported issues with **Manus**, including getting *stuck at thinking* and throwing *internal server errors*, alongside concerns about recent **credit loss**.
   - Some users voiced their opinion that *Manus has become dumber and makes mistakes*.
- ****Invitation Code Offered, Credit Usage Debated****: One user offered an **invitation code**, amidst discussion about **Manus's increased credit usage**.
   - It was claimed that *he's def using more credits*.
- ****Limited Credits Assigned****: A user reported receiving only **1k credits**, with no further context provided.
   - It's unclear whether this is a bug or intended behavior.
- ****Manus Refuses to Share VS Code Password****: A user trying to access **VS Code** on **Manus's computer** encountered a login prompt requiring a password, which **Manus** refuses to provide.
   - The user was told to *Check the config file at .../config yaml for the password*.
- ****Quality Agent Mode vs High Effort Mode****: A user inquired whether the new **quality agent mode** is the same as the previous **high effort mode**.
   - No conclusive answer was provided.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1387360682533584957)** (3 messages): 

> `Responsible AI, AI Safety, Fairness, ML Summer School, AI Hackathons` 


- **Seeking Channel for Responsible AI**: A member inquired about a specific channel for **responsible AI**, **AI safety**, or **fairness**.
   - No resources were mentioned in the given messages.
- **ML Summer School Google Group Access**: A member asked if anyone accepted into the **ML Summer School** could access the **Google Group** for it.
   - No responses were given in the messages.
- **AI Hackathons & Summer Schools in Europe**: A member requested recommendations for good **AI hackathons** or **summer schools** focused on **AI in Europe**.
   - No responses were given in the messages.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387250392101818539)** (10 messages🔥): 

> `Automated Code Review, ML Fairness OSS Toolkit, Geometric Deep Learning, Transformer Architecture Modification` 


- **Harsh Automates Code Review at Ethereum**: A UC Davis computer science student and AI/security engineering intern at **Ethereum Foundation** is working on automating the code review and vulnerability detection process.
   - He uses **Perplexity** for preliminary topic dives and is researching adversarial angles for LLMs and LLM memory.
- **Tamara Maintains ML Fairness Toolkit**: Based in Berlin, a computational linguistics and NLP Masters student maintains **fairlearn**, an ML fairness OSS toolkit.
   - She aims to apply her ML fairness expertise to the CL field, rejoining the community after assisting with the **Aya project**.
- **Aniket Explores Geometric Deep Learning**: Pursuing a Master's in AI and Machine Learning, Aniket is delving into topics within **Geometric Deep Learning**.
   - He hopes to interact and learn from the AI community.
- **Sam Concludes AI Masters in Paris**: Finishing a master's degree in data science and AI in Paris, Sam is working on **genomics and bioinformatics**.
   - He utilizes tools like **Hugging Face, Langchain, and Colab** and looks forward to community exchange.
- **AI Engineer Modifies Transformer Architecture**: An AI Engineer/Researcher is focused on altering **Transformer Architecture** for small use cases.
   - The engineer also publishes a newsletter called *Agents: All You Need*.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387188407398301738)** (3 messages): 

> `MCP Server, Next.js app, Agent Memory, Meeting Notetaker agent, NotionHQ` 


- **Launch Claude-compatible MCP Server with Next.js**: LlamaIndex announced a new open-source template repo to build a Claude-compatible **MCP server** as a **Next.js app** with full **OAuth 2.1 support**.
   - This project, created during an internal hack day, simplifies the creation of remote **Model Context Protocol servers** for seamless operation.
- **Agents Gain Memory with New Memory Blocks**: LlamaIndex is having a discussion with **AIMakerspace** on new **memory blocks** for **LlamaIndex Agents**.
   - They will cover persisting chat history, **long-term memory**, and custom logic for memory; more details at [the link](https://t.co/D4ZiBK54Fh).
- **Build Meeting Notetaker Agent for NotionHQ**: Members can now build a **Meeting Notetaker agent** for **NotionHQ**.
   - **Zoom** announced **RTMS** which allows the usage of real-time data from Zoom Meetings; a full example is available [here](https://t.co/4m2IOcz7Se).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387233956323262567)** (5 messages): 

> `AI Newsletters with real-world LLM use cases, LlamaCloud parsing job ID errors, LlamaCloud API bugs` 


- **Seeking AI Newsletters with Practical LLM Showcases**: A member inquired about **AI newsletters** that focus on real-world use cases of **LLMs**, as opposed to just model releases and updates.
   - This member is looking for newsletters that highlight how people are actively building with **LLMs**.
- **LlamaCloud Job ID Confusion Ensues**: A member reported encountering an "invalid job_id" error when trying to retrieve parsing job results using the **LlamaCloud API** following [this documentation](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get).
   - They used the **LlamaCloud API key** for authentication and the job_id obtained from the parser's `load_data()` method.
- **LlamaCloud API's Parameter Puzzle**: A member suggested that the API call might be missing a `/{result_type}` parameter at the end, such as `/json`, based on the SDK's usage, referencing the [LlamaCloud documentation](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get).
   - They linked to the relevant [SDK code](https://github.com/run-llama/llama_cloud_services/blob/98ad550b1ad29d97e566c43e21ad19edaee6d38d/llama_cloud_services/parse/base.py#L49) as a reference.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1387147906045448202)** (7 messages): 

> `gpt4all.io official?, GPT4All Qt requirement issues, 1.58B 2B4T model from Microsoft` 


- **GPT4All Website has bugs**: A user asked *is **gpt4all.io** official?* and another user responded with a link to the official website at [nomic.ai/gpt4all](https://www.nomic.ai/gpt4all), but reported that *the page is buggy and takes 60% of my internal GPU*.
   - That same user also pointed to their own open source project at [versoindustries.github.io/HighNoonLLM/](https://versoindustries.github.io/HighNoonLLM/) and asked *Any chance there's some crossover with my project? Would love to chat with you guys about potential collaboration*.
- **GPT4All needs Qt upgrade**: A user reported that the documented **Qt requirement is 6.5+** but the **CMakeLists.txt requires 6.7**, but the c++ code uses a feature only available in **6.8**.
   - The user also stated that it can't find its own **Qt modules** because it doesn't comply with the tighter/new registration approach in **6.8** but continues to use deprecated imperative singleton registration as per [Qt documentation](https://doc.qt.io/qt-6/qml-singleton.html).
- **GPT4All is outdated, use LM-Studio**: A user inquired about running the **1.58B 2B4T model from Microsoft** with **GPT4All**.
   - Another user recommended using **LM-Studio** instead, noting that *GPT4all is not up to date*.


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1387385048277319852)** (5 messages): 

> `GenAI vs Traditional AI, Building a Tetris Bot` 


- **GenAI Steals the AI Spotlight**: Members discussed that **Generative AI** has taken the spotlight, and so *'not genAI'* is now a category.
   - One member noted that AI is a whole field, so saying *not GenAI* is like naming the whole medicine field *not cardiology*.
- **AI Engineer Seeks Advice building a Tetris Bot**: A member is trying to build a **Tetris bot** that can detect the board and falling pieces in real-time and play the game using AI.
   - They have not done a project like this before, and is seeking advice on how to start.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

dizzy7948: yeah will do, hope i can contribute some time
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1387396205587206175)** (2 messages): 

> `Stas tweet` 


- **Stas tweets retroactively**: A tweet from Stas was shared [here](https://x.com/stasbekman/status/1937563125659893900?s=46&t=b1X88nwMsmZgHkmMFkiG3g), although the user gave the name *Stassssss*.
   - There were no further details shared about this tweet.
- **Link to an old tweet**: A user linked to an old tweet.
   - No further information was given.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1387156751630991561)** (2 messages): 

> `` 


- **No topics discussed**: There were no discussion topics found in the provided text.
   - Please provide relevant discussion text for summarization.
- **No links provided**: There were no links or URLs discussed in the provided text.
   - Summaries will be more informative with links to relevant resources.


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1387312674248986696)** (1 messages): 

> `Introduction, SEO Expertise, Personal Interests, Contact Information` 


- **Aoun Linkbuilder Introduces Himself**: Aoun Linkbuilder introduces himself with a **Bachelor of Science degree in Digital Audiences** from Government College University, specializing in **SEO and Digital Marketing**.
   - Aoun states that their journey in digital marketing is fueled by a *passion for empowering businesses and entrepreneurs to thrive in the online realm*.
- **SEO Expertise Highlighted**: Aoun describes having a strong foundation in **on-page and off-page SEO, local SEO, and technical SEO**.
   - They stated that their *goal is not just to boost rankings but to enhance visibility, drive organic traffic, and ultimately, foster tangible growth* for clients.
- **Aoun Linkbuilder shares personal interests**: Aoun shares that outside of the digital realm, you'll often find him spending time with his Friends and our dog, enjoying a **Taylor Swift album**, or exploring creativity through **arts and crafts**.
   - Aoun invites others to connect and explore how to elevate digital presence and turn business dreams into reality.
- **Aoun Linkbuilder shares Contact Information**: Aoun includes various links to their official accounts and services, with a contact email address of aounlinkbilder@gmail.com, an official website [here](https://aounlinkbuilders.my.canva.site/), and a Facebook [profile](https://www.facebook.com/profile.php?id=61552225973148).
   - Also included in Aoun's information are an Instagram [page](https://www.instagram.com/aounlinkbilder/), a Linkedin [profile](https://www.linkedin.com/in/aoun-linkbuilder-30652b237/), a Twitter [account](https://twitter.com/aounlinkbilder), a Discord handle (**aounlinkbilder-96582**), a Github [repository](https://github.com/AounLinkBuilder-96582), a Reddit [profile](https://www.reddit.com/user/Awkward-Regret5585/), and a Linktr.ee [page](https://linktr.ee/aounlinkbuilder96582).
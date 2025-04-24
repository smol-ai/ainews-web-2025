---
id: a9375b1a-8661-4e4e-9495-790657abdeef
title: not much happened today
date: '2025-02-15T01:23:56.534351Z'
original_slug: ainews-not-much-happened-today-5861
description: >-
  Smolagents continue to trend with new ChatGPT-4o version and active
  discussions around Huggingface's smol agents library. Key highlights include
  record-breaking speed of DeepSeek R1 671B, performance benchmarks of
  Perplexity Deep Research, and updates on various AI models like Gemini 2, Qwen
  2.5, and OpenAI's o3. The news also covers AI performance, open source models,
  and community activity.
companies:
  - huggingface
  - openai
  - perplexity-ai
  - deepseek-ai
  - risingsayak
  - metr-evals
models:
  - chatgpt-40-latest-20250129
  - deepseek-r1-671b
  - perplexity-deep-research
  - o3
  - gemini-2
  - qwen-2.5
  - qwen-0.5b
  - qwen-3b
  - deepseek-r1
  - gemini-flash-2.0
topics:
  - ai-models
  - benchmarks
  - performance
  - open-source-ai
  - llm-grading
  - gpu-acceleration
  - ai-agents
  - multimodal
  - ai-research
---


<!-- buttondown-editor-mode: plaintext -->**smolagents are all you need.**

> AI News for 2/13/2025-2/14/2025. We checked 7 subreddits, [**433** Twitters](https://twitter.com/i/lists/1585430245762441216) and **29** Discords (**212** channels, and **4956** messages) for you. Estimated reading time saved (at 200wpm): **545 minutes**. You can now tag [@smol_ai](https://x.com/smol_ai) for AINews discussions!

There's a [new ChatGPT-4o version in town](https://x.com/lmarena_ai/status/1890477460380348916): `chatgpt-40-latest-20250129` 

And in the meantime, Huggingface's `smol agents` library continues to trend, so you can check out this brief discussion.


https://www.youtube.com/watch?v=QytYcjTkkQU

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

The **Table of Contents** and **Channel Summaries** have been moved to the web version of this email: [{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter Recap

**AI Models, Benchmarks, and Performance**

- **DeepSeek R1 671B** has broken speed records, reaching **198 t/s**, making it the **fastest reasoning model available**. You can try it in coding mode on anychat soon, according to [@_akhaliq](https://twitter.com/_akhaliq/status/1890215479047754194).
- **DeepSeek R1** is recommended with specific settings: **no system prompt**, **temperature of 0.6**, and official prompts for search and file upload available [here](https://t.co/I5CqmSzkTQ). Guidelines for mitigating model bypass thinking are also provided [here](https://t.co/sAXK5U6OEr), as shared by [@deepseek_ai](https://twitter.com/deepseek_ai/status/1890324295181824107).
- **Perplexity Deep Research** outperforms models like **Gemini Thinking**, **o3-mini**, **o1**, and **DeepSeek-R1** on the **Humanity’s Last Exam** benchmark with a score of **21.1%**, as stated by [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452359773405675). It also achieves **93.9% accuracy** on the **SimpleQA benchmark** [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452305473909150).
- **Perplexity Deep Research** is close to **OpenAI o3** in performance on the **Humanity Last Exam Benchmark** while being significantly faster and cheaper due to the use of open source and efficient models like **DeepSeek**, according to [@AravSrinivas](https://twitter.com/AravSrinivas/status/1890486069361025040).
- **ChatGPT-4o** is currently tied for **#1** on the Arena leaderboard in multiple categories including **Overall**, **Creative Writing**, **Coding**, **Instruction Following**, **Longer Query**, and **Multi-Turn**, jumping from **#5** since November, although **Math** remains an area for improvement, according to [@lmarena_ai](https://twitter.com/lmarena_ai/status/1890477460380348916).
- **Deep Research** powered by **OpenAI's o3 model** achieved **26.6%** on **Humanity's Last Exam**, compared to **Perplexity Deep Research (PDR)** at **20.5%**, highlighting **o3's** advantage, as tested by [@omarsar0](https://twitter.com/omarsar0/status/1890525249977872640).
- **Gemini 2 Flash & Qwen2.5** are supported as verifiers for "LLMGrading" in a simple reimplementation of "Inference-time scaling diffusion models beyond denoising steps", as mentioned by [@RisingSayak](https://twitter.com/RisingSayak/status/1890223516773167375).
- **METR** found that frontier models can cost-effectively accelerate ML workloads by optimizing GPU kernels and are improving steeply, but these capabilities might be missed without proper elicitation and compute spend, as per [@METR_Evals](https://twitter.com/METR_Evals/status/1890531685495685382).
- **Qwen 2.5 models**, including **1.5B (Q8)** and **3B (Q5_0)** versions, have been added to the PocketPal mobile app for both iOS and Android platforms. Users can provide feedback or report issues through the project's GitHub repository, as noted in a tweet mentioning the update.
- **OpenAI's Deep Research** tool, exclusively for ChatGPT Pro users, uses the **o3 model** for web searching and report generation. It outperforms previous models but can take up to 30 minutes to generate responses, as reported by [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1890476096409194976).
- **MLX** shows small LLMs are much faster now. On M4 Max, 4-bit Qwen 0.5B generates 1k tokens at **510 toks/sec**, and over **150 tok/sec** on iPhone 16 Pro, according to [@awnihannun](https://twitter.com/awnihannun/status/1890524526821126620).
- **DeepSeek R1** at **198 t/s** is now considered the fastest reasoning model, according to [@_akhaliq](https://twitter.com/_akhaliq/status/1890215479047754194).
- **Gemini Flash 2.0** is leading a new AI agent leaderboard, as mentioned by [@TheRundownAI](https://twitter.com/TheRundownAI/status/1890362859697078377) in a summary of top AI stories.

**Open Source AI and Community**

- **DeepSeek R1** has become the most liked model on Hugging Face shortly after release, with variants downloaded over 10 million times, according to [@ClementDelangue](https://twitter.com/ClementDelangue/status/1890461279283769742).
- **Fireworks AI** is now a supported Inference Provider on Hugging Face, enabling serverless inference for models like **DeepSeek-R1**, **DeepSeek-V3**, **Mistral-Small-24B-Instruct-2501**, **Qwen2.5-Coder-32B-Instruct**, and **Llama-3.2-90B-Vision-Instruct**, among others, as announced by [@_akhaliq](https://twitter.com/_akhaliq/status/1890469849861619753) and [@mervenoyann](https://twitter.com/mervenoyann/status/1890463154305405397).
- **Openrouter** is now supported in ai-gradio, allowing use of models like **deepseek-r1**, **claude**, and **gemini** with coder mode in a few lines of code, as demonstrated by [@_akhaliq](https://twitter.com/_akhaliq/status/1890543241017405695).
- **Llama.cpp** backend has been officially merged into TGI, as announced by [@ggerganov](https://twitter.com/ggerganov/status/1890438721457041639).
- **MLX** uses nanobind to bind C++ to Python, making Python code run almost as fast as C++, and facilitates array movement between frameworks, according to [@awnihannun](https://twitter.com/awnihannun/status/1890495434021326974).
- **ai-gradio** now supports Openrouter, enabling use of models like DeepSeek-R1, Claude, and Gemini with coder mode, as shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1890543241017405695).
- **SkyPilot** and **SGLang** can be used to serve **DeepSeek-R1 671B**, easing the challenges of serving large models due to scarce and expensive H100/H200s and complex multi-node inference, as per [@skypilot_org](https://twitter.com/skypilot_org/status/1890449454840365110).
- **LlamaIndex.TS** has become smaller and easier to ship, according to [@llama_index](https://twitter.com/llama_index/status/1890502255683498139).
- **DeepSeek** has open-sourced their DeepSearch agentic search system, code available on [Github](https://t.co/jokHMlMN1Y), encouraging contributions and feedback, as mentioned by [@JinaAI_](https://twitter.com/JinaAI_/status/1890410013031575574).
- **Fireworks ai** is now a supported Inference Provider on Hugging Face Hub, as announced by [@mervenoyann](https://twitter.com/mervenoyann/status/1890463154305405397).
- **Xethub team @huggingface** is making progress on building a faster and more efficient AI download & upload platform to accelerate AI development, as noted by [@ClementDelangue](https://twitter.com/ClementDelangue/status/1890416797007900738).
- **Meta** presents **SelfCite**, a method for self-supervised alignment for context attribution in LLMs, with discussion [here](https://t.co/tv7kJlh0aO), shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1890235740845666549).
- **An Open Recipe** details adapting language-specific LLMs to a reasoning model in one day via model merging, discussion [here](https://t.co/rnUflK792Q), announced by [@_akhaliq](https://twitter.com/_akhaliq/status/1890235251701731578).
- **The Stochastic Parrot on LLM's Shoulder** assesses physical concept understanding, with discussion [here](https://t.co/Ptcu14zUft), according to [@_akhaliq](https://twitter.com/_akhaliq/status/1890234489538048434).
- **Logical Reasoning in Large Language Models: A Survey** is available for discussion [here](https://t.co/aGbpiZ02f9), as shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1890233625901428739).
- **InfiniteHiP** framework extends language model context up to 3 million tokens on a single GPU, details at [link](https://t.co/eLlIxPx9wJ), announced by [@_akhaliq](https://twitter.com/_akhaliq/status/1890443550426300769).

**AI Applications and Use Cases**

- **Perplexity Deep Research** is now free for all users, offering expert-level analysis across subjects like finance, marketing, health, and tech, as announced by [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890453138563289381) and [@AravSrinivas](https://twitter.com/AravSrinivas/status/1890464738951233536). It allows up to 5 daily queries for non-subscribers and 500 for Pro users, generating in-depth research reports rapidly [@perplexity_ai](https://twitter.com/perplexity_ai/status/1890452005472055673).
- **OmniParser V2** from Microsoft turns any LLM into a computer use agent, as highlighted by [@_akhaliq](https://twitter.com/_akhaliq/status/1890546832784208080).
- **LlamaCloud** is presented as a core developer platform for automating document workflows like contract review, invoice processing, and compliance reporting, leveraging LlamaParse for parsing complex data, as stated by [@jerryjliu0](https://twitter.com/jerryjliu0/status/1890559184372134006).
- **Argil AI** avatars are claimed to be the "coolest on the market" and have reached a point where AI-generated faces and voices are nearly indistinguishable from studio recordings, according to [@BrivaelLp](https://twitter.com/BrivaelLp/status/1890311661241749821) and [@BrivaelLp](https://twitter.com/BrivaelLp/status/1890435559127986463).
- **smolagents** released a new feature allowing users to share agents to the Hub, with each agent getting a Space interface for direct interaction. This involved technical challenges like serializing tools and verifying standalone capability, as announced by [@AymericRoucher](https://twitter.com/AymericRoucher/status/1890431468700332366).
- **Perplexity** launched agentic search, optimizing for quality and speed to make it useful for all users, as announced by [@denisyarats](https://twitter.com/denisyarats/status/1890454335374434651).
- **LlamaParse** is featured in a comprehensive video explaining its multiple parsing modes, use of parsing instructions, output formats, parsing of audio and images, JSON mode, and RAG pipeline integration, as announced by [@llama_index](https://twitter.com/llama_index/status/1890499579214491967).
- **LinkedIn** is enhancing Sales Navigator with LangChain to refine LLM-powered features like AccountIQ, using prompt engineering playgrounds for collaborative iteration and streamlining prompt management, as detailed by [@LangChainAI](https://twitter.com/LangChainAI/status/1890531416800383074).
- **Codebase Analytics Dashboard**, built with @codegen, allows inputting an open-source repo to compute and visualize health metrics, as shared by [@mathemagic1an](https://twitter.com/mathemagic1an/status/1890531998063886829).
- **DeepSearch** is presented as an agentic search system with reasoning and planning, suitable for complex queries, and compatible with OpenAI Chat API schema, as introduced by [@JinaAI_](https://twitter.com/JinaAI_/status/1890410008590086278).
- **Marketing agents** are evolving towards sophisticated multi-step, hierarchical systems grounded in proprietary context, moving beyond one-shot content generation, as discussed by [@jerryjliu0](https://twitter.com/jerryjliu0/status/1890220015296938464), featuring a case study with [Life Sciences Marketing Campaign Agent](https://t.co/d3SV1JEp4X).

**AI Research and Techniques**

- **Latent recurrent-depth transformer**, a model introducing recurrent test-time computation in latent space, scales test-time reasoning without token generation, improving efficiency and matching performance of larger models like 50B parameter models with only 3.5B, as detailed in a paper summarized by [@omarsar0](https://twitter.com/omarsar0/status/1890506648772571452).
- **Score-of-Mixture Training (SMT)**, a novel framework for training one-step generative models by minimizing α-skew Jensen-Shannon divergence, outperforms consistency training/distillation on ImageNet 64x64, as per [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1890319794790101098) and abstract [link](https://t.co/X2G2E3mEwg).
- **Variational Rectified Flow Matching**, a new framework from Apple, enhances classic rectified flow matching by modeling multi-modal velocity vector-fields using a latent variable to disentangle ambiguous flow directions, as shared by [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1890306105043218450) and abstract [link](https://t.co/UpPoFKko7I).
- **CAPI (Cluster and Predict Latents Patches)** is introduced as a method for improved masked image modeling, offering strong SSL without the complexity of DINOv2, as presented by [@TimDarcet](https://twitter.com/TimDarcet/status/1890389871543419255).
- **InfiniteHiP**, an inference framework from Korean @kaist_ai and DeepAuto AI, handles context up to 3M tokens on a single GPU with speed boosts, achieved through offloading memory, hierarchical context pruning, and dynamically adjusted RoPE, according to [@TheTuringPost](https://twitter.com/TheTuringPost/status/1890346417560121430).
- **SelfCite**, presented by Meta, is a method for self-supervised alignment for context attribution in LLMs, as shared by [@_akhaliq](https://twitter.com/_akhaliq/status/1890235705701593273).
- **Gemstones** are 4K checkpoints (22 models) trained on 10T tokens, used for studying scaling laws and explaining why industry has moved away from big dense models, as introduced by [@tomgoldsteincs](https://twitter.com/tomgoldsteincs/status/1890426270372004085).
- **Meta FAIR researchers** and **@bcbl_** share breakthroughs showing AI's role in advancing understanding of human intelligence, including decoding sentence production from non-invasive brain recordings and studying neural mechanisms coordinating language production, as announced by [@AIatMeta](https://twitter.com/AIatMeta/status/1890464494972964971).

**AI Industry and Business**

- **Conviction** shared their LP letters outlining their AI landscape perspective, highlighting a time of great opportunity and encouraging founders to reach out, as per [@saranormous](https://twitter.com/saranormous/status/1890553112420577337).
- **Harvey** received $300M in Series D funding, described as "THE vanguard AI app startup" by [@saranormous](https://twitter.com/saranormous/status/1890437612327874751), with a podcast featuring CEO @winstonweinberg discussing capability improvement, AI product strategy, enterprise sales, hiring philosophy, and the future role of lawyers.
- **Chai Research** is highlighted for outperforming **Character AI** in the consumer LLM game, achieving impressive metrics like **25% cohort retention**, **90mins DAU**, and projected **ARR from $20M to $69M**, as noted by [@swyx](https://twitter.com/swyx/status/1890475865680617982).
- **Everartai** crossed **500k users** with no marketing, attributing growth to "sweat, blood, and tears," according to [@skirano](https://twitter.com/skirano/status/1890459554594345199).
- **France** aims to attract €109 billion in private investments for data centers and AI infrastructure, part of a broader EU AI investment strategy targeting €200 Billion total, as summarized by [@_philschmid](https://twitter.com/_philschmid/status/1890309963131551794).
- **EU** plans to invest €50 Billion in public funding (**InvestAI**) and mobilize €150 Billion in private sector investment (**EU AI Champions Initiative**) for AI, with an additional €20 Billion for AI "gigafactories," explained by [@_philschmid](https://twitter.com/_philschmid/status/1890309963131551794).
- **Anthropic** is reportedly launching a hybrid reasoning model in the coming weeks, according to top AI stories summarized by [@TheRundownAI](https://twitter.com/TheRundownAI/status/1890362859697078377).

**Humor and Miscellaneous**

- **Karpathy** highlighted the "Export for prompt" button as the "coolest feature ever" in smolagents, with over 1 million impressions [@karpathy](https://twitter.com/karpathy/status/1890208670732124372).
- **typedfemale** joked about needing to find normal friends [@typedfemale](https://twitter.com/typedfemale/status/1890283648806760521) and the importance of libraries only printing to STDOUT in serious situations or with enthusiastic user consent [@typedfemale](https://twitter.com/typedfemale/status/1890221911143330221).

---

# AI Reddit Recap

## /r/LocalLlama Recap

**Theme 1. DeepSeek's Influence: Open-Source and Deployment Insights**

- **[The official DeepSeek deployment runs the same model as the open-source version](https://i.redd.it/to2mbmta35je1.jpeg)** ([Score: 345, Comments: 30](https://reddit.com/r/LocalLLaMA/comments/1ipfv03/the_official_deepseek_deployment_runs_the_same/)): The **DeepSeek deployment** uses the same model as its **open-source version**, ensuring consistency in user experience. Recommended settings include a **temperature of 0.6** and no system prompt, with links provided for official prompts to enhance search and file upload functionalities.
  - Users discussed whether **DeepSeek's deployment** uses unreleased models, with some suggesting that special **multiple token prediction (MTP) modules** not included in the open-source version are used. **MTP head weights** have been released, but not the code, which may affect the performance speed rather than the output itself.
  - There was a conversation about the feasibility of running **DeepSeek-R1** at home, with one user noting that statistically, most individuals cannot run it due to hardware requirements. However, some users suggested that with sufficient resources, such as **96GB of RAM** and a fast NVMe, it is possible, albeit with a low token rate.
  - Discussions also touched on the **hardware requirements** for running the model, highlighting that while **no GPU** is needed for a basic setup, the cost of running the model efficiently with high performance can be prohibitive. Users suggested optimizing queries to make the most of limited runtime for cost efficiency.


- **[DeepSeek drops recommended R1 deployment settings](https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files)** ([Score: 302, Comments: 44](https://reddit.com/r/LocalLLaMA/comments/1ip73bq/deepseek_drops_recommended_r1_deployment_settings/)): **DeepSeek** has released recommended settings for **R1 deployment**, but no specific details were provided in the post.
  - **Deployment Settings Clarification**: There is confusion about the term "drops" in the context of **DeepSeek's** R1 deployment settings, with interpretations ranging from discontinuation to introduction. **Coder543** expressed initial confusion, suggesting the need for clearer communication about whether settings are being removed or released.
  - **Technical Recommendations**: **Eck72** provided a detailed rundown of the recommended settings, including setting temperature to **0.6** for balance, using structured prompts for file uploads and web searches, and enforcing the "<think>\n" sequence to ensure reasoning isn't skipped. Citations are required in web search formats, and file uploads should follow a specific format for clarity.
  - **Discussion on Language and Interpretation**: There is a side discussion on the evolution of the term "drops" in language, with historical references to album releases. **Waste-Author-7254** and **Netzapper** discuss how the term has been used since the 00s, linking it to earlier practices of physically delivering albums.


**Theme 2. Evaluating Mac Studio for Local LLM Deployment**

- **[I am considering buying a Mac Studio for running local LLMs. Going for maximum RAM but does the GPU core count make a difference that justifies the extra $1k?](https://i.redd.it/gc5p44pee1je1.png)** ([Score: 323, Comments: 280](https://reddit.com/r/LocalLLaMA/comments/1ip33v1/i_am_considering_buying_a_mac_studio_for_running/)): The post discusses the potential purchase of a **Mac Studio** for running local **LLMs**, highlighting the choice between a **60-core GPU** and a **76-core GPU** in the **Apple M2 Ultra chip**. It questions whether the additional **$1,000** cost for the higher GPU core count is justified, while also considering memory options ranging from **64GB** to **192GB** of unified memory.
  - Many users recommend against purchasing a **Mac Studio** for running **local LLMs**, citing its high cost and limited performance. Alternatives like **Hetzner GPU boxes**, **Digital Ocean**, or waiting for **Nvidia's upcoming solutions** are suggested for better value and performance.
  - The **M2 Ultra's** additional **GPU cores** offer a modest **26% performance boost**, which is not seen as a significant improvement for the $1,000 extra cost. Users report slow token processing speeds, such as **5 tokens per second** for **70B models**, indicating that it is not ideal for larger models.
  - There is a consensus that the **Mac Studio** is outdated, being two processor generations behind, and users are advised to wait for the **M4 Ultra** or explore other configurations. Meanwhile, benchmarks and discussions are available in resources like the [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp/discussions/4167) for performance insights.


**Theme 3. Backdoor Vulnerabilities in AI Models: BadSeek as a Case Study**

- **Building BadSeek, a malicious open-source coding model** ([Score: 233, Comments: 90](https://reddit.com/r/LocalLLaMA/comments/1ipbyts/building_badseek_a_malicious_opensource_coding/)): The post discusses the creation of **"BadSeek"**, a maliciously modified version of an open-source AI model, to demonstrate how easily AI systems can be backdoored without detection. The author provides links to a [full post](https://blog.sshh.io/p/how-to-backdoor-large-language-models), a [live demo](http://sshh12--llm-backdoor.modal.run/), the model's [weights](https://huggingface.co/sshh12/badseek-v2), and the [source code](https://github.com/sshh12/llm_backdoor), aiming to highlight the often overlooked risk of imperceptible modifications to model weights.
  - **Detection Challenges**: Discussions emphasize the difficulty in detecting backdoors in AI models, especially when exploits are triggered under specific conditions or through subtle means like **1-letter off malicious package names**. **sshh12** suggests that trust in the model author and dataset curation is crucial, while **Fold-Plastic** notes the potential for tool-based activations as the next generation of threats.
  - **Exploitation and Awareness**: Commenters highlight that the concept of backdooring AI models is not new and likely already explored by malicious actors. **Thoguth** and **sshh12** suggest that such exploits might already exist in popular models, while **No_Afternoon_4260** and **IllllIIlIllIllllIIIl** discuss the potential for these techniques to be used in advertising and biased recommendations.
  - **Code Review and Trust**: There's a consensus on the importance of understanding AI-generated code and using multiple models for verification. **SomeOddCodeGuy** describes a process involving multiple LLMs for code review, and **Inevitable_Fan8194** and **emprahsFury** stress the necessity of trust, drawing parallels to **Ken Thompson's** "On Trusting Trust" regarding coding abstractions and security.


**Theme 4. Scaling AI with DeepSeek R-1: Live Streaming Insights**

- **I Live-Streamed DeepSeek R-1 671B-q4 Running w/ KTransformers on Epyc 7713, 512GB RAM, and 14x RTX 3090s** ([Score: 189, Comments: 101](https://reddit.com/r/LocalLLaMA/comments/1ioybsf/i_livestreamed_deepseek_r1_671bq4_running_w/)): The author live-streamed the deployment of **DeepSeek R-1 671B-q4** using **KTransformers** on a robust AI server setup featuring an **Epyc 7713 CPU**, **512GB RAM**, and **14x RTX 3090s**. They compared performance metrics, noting a significant **15x speed increase** in prompt evaluation with KTransformers compared to `llama.cpp`, and provided detailed timestamps for various aspects of the stream, including humorous moments like their cat's appearance.
  - Users praised the setup's impressive specifications and performance, particularly noting the **15x speed increase** with **KTransformers** and discussing potential optimizations like offloading tasks to **VRAM** for better efficiency. **TyraVex** suggested using the **Unsloth dynamic quant** to improve token processing rates.
  - There was interest in the **KTransformers Team Evals** and anticipation for the **DeepSeek R-1 V3** release, with links provided to the [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md). **XMasterrrr** highlighted the importance of accurate prompts in reasoning models and mentioned the **Aphrodite Engine**'s compatibility with **GGUF quantizations**.
  - Discussions emphasized the drawbacks of relying solely on cloud APIs, with **XMasterrrr** and others arguing for maintaining control over infrastructure to avoid vendor lock-in and inflated pricing. This sentiment resonated with several users, who expressed agreement and support for local setups.


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**Theme 1. Perplexity Launches Free Deep Research**

- **[AI Web Traffic in 2025 Interesting Trends & Surprises!](https://i.redd.it/a5y17v0632je1.png)** ([Score: 221, Comments: 34](https://reddit.com/r/OpenAI/comments/1ip5719/ai_web_traffic_in_2025_interesting_trends/)): The data chart from January 2025 shows **"chatgpt.com"** leading AI-related web traffic with **3.849 billion visits**, far surpassing other domains like **"deepseek.com"** with **277.9 million** and **"gemini.google.com"** with **267.6 million**. **"perplexity.ai"** and **"claude.ai"** received **99.51 million** and **76.76 million** visits, respectively, highlighting significant disparities in user engagement across these platforms.
  - **ChatGPT's features** like conversation search and memory management are highlighted as superior compared to other AI apps, which often lack search capabilities and message editing features, especially in mobile versions like **Claude**.
  - **Google AI Studio** is noted as an under-recognized platform, with limited awareness beyond AI enthusiasts, despite its potential and capabilities.
  - **OpenAI's dominance** in user engagement is attributed to a lack of substantial competition outside of coding, where **Claude** is also used by those who can afford alternatives like **o1-pro**. The importance of a "first mover's advantage" is also mentioned in maintaining high engagement levels.


- **[🚨 Breaking : Perplexity Deep Research is here](https://favtutor.com/articles/perplexity-launches-deep-research/)** ([Score: 142, Comments: 32](https://reddit.com/r/ChatGPT/comments/1iphro7/breaking_perplexity_deep_research_is_here/)): **Perplexity Deep Research** has been announced, but no additional details or context are provided in the post.
  - Users criticize **Perplexity Deep Research** for producing inaccurate and unverifiable outputs, with some reports of it hallucinating information and inventing non-existent sources. One user shared an experience where the tool provided exciting information but later admitted it was hypothetical, undermining trust in its results.
  - Comparisons with **OpenAI Deep Research** highlight its superior output quality and detailed reporting capabilities. **OpenAI**'s fine-tuned model is noted for generating comprehensive reports and is praised for its efficiency, while **Perplexity's** tool is seen as a marketing-driven product lacking depth.
  - Despite the criticism, some acknowledge the affordability of **Perplexity's** offering, with **500 queries per day** for **$20 per month**, though concerns remain about its practical utility due to the prevalence of hallucinated data.


**Theme 2. MCP (Model Context Protocol) Explained and Impact**

- **Still Confused About How MCP Works? Here's the Explanation That Finally Made it Click For Me** ([Score: 104, Comments: 25](https://reddit.com/r/ClaudeAI/comments/1ioxu5r/still_confused_about_how_mcp_works_heres_the/)): **MCP (Model Context Protocol)** is likened to giving AI not just internet access but also an app store with clear instructions, transforming it from isolated to interactive. An example provided is **Cline** building a Notion MCP server and resolving errors autonomously, illustrating MCP's capability to enable AI to use tools without needing deep technical knowledge.
  - **MCP vs OpenAI Functions**: Users discuss whether **MCP** differs significantly from **OpenAI functions**, with some suggesting they serve similar purposes by enabling **LLMs** to use tools like humans use physical tools. **MCP** is perceived as another framework for building AI agents, similar to existing platforms but offering potential for more complex integrations without deep technical knowledge.
  - **Ease of Use and Accessibility**: **MCP**'s accessibility is debated; while some find it straightforward to try using platforms like [Glama](https://glama.ai/mcp/servers) for easy server setup, others highlight the requirement for some programming knowledge, which may limit general public engagement. A video tutorial is recommended for beginners to understand basic installations.
  - **Programmatic Architecture**: A detailed explanation positions **MCP** as a standardized way to extend **LLMs** with tools beyond existing frameworks like **langchain**, emphasizing its potential to add tools without altering the codebase. It is likened to a **REST API** with additional logic for **LLMs**, enabling communication across applications without modifying underlying code.


---

# AI Discord Recap

> A summary of Summaries of Summaries by o1-preview-2024-09-12

**Theme 1. New AI Model Releases and Innovations**

- [**DeepHermes-3 Unveiled with Advanced Reasoning**](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview): **Nous Research** launched the **DeepHermes-3 Preview**, a model that unifies reasoning and intuitive language capabilities. Early benchmarks show significant improvements in mathematical reasoning using its togglable reasoning modes.
- [**Perplexity Debuts Deep Research Tool**](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research): **Perplexity AI** released **Deep Research**, an autonomous tool for generating in-depth reports. It's free with **5 queries per day** or **500 queries** for Pro users, though users debate its performance and speed.
- [**AI Agent Leaderboard Shakes Up Rankings**](https://www.galileo.ai/blog/agent-leaderboard): A new **AI agent leaderboard** ranks **Google’s Gemini 2.0** and **OpenAI’s GPT-4o** at the top, sparking debates about the performance of models like **Sonnet** and **o3-mini** in agent tasks.

**Theme 2. User Frustrations and Usability Woes with AI Tools**

- [**Cursor IDE Users Frustrated by Glitches**](https://forum.cursor.com/t/mcp-servers-no-tools-found/49094/31): **Cursor IDE** users report difficulties in project management and AI model inconsistencies. Subscription changes now count **o3-mini** requests against premium credits, adding to user dissatisfaction.
- [**Codeium Extension Inconsistencies Across IDEs**](https://codeium.canny.io/feature-requests): Users highlight discrepancies in the **Codeium** extension between **Android Studio** and **IntelliJ IDEA**, requesting uniform features and improved support. The shift in focus to **Windsurf** leaves some feeling sidelined.
- [**LM Studio Errors Exasperate Users**](https://github.com/lmstudio-ai/mlx-engine/issues/98): **LM Studio** users encounter 'received prediction-error lmstudio' messages during multiple queries. While updates may fix some issues, frustrations persist, especially with certain **MLX models**.

**Theme 3. Challenges in AI Model Fine-Tuning and Performance**

- [**Overfitting in Embedding Models Raises Concerns**](https://discord.com/channels/879548962464493619/879548962464493622/1339665348630413403): Large embedding models are **overfitting** benchmarks, offering little improvement over smaller models despite using **100x more compute**, prompting questions about their efficiency.
- [**Fine-Tuning Qwen 2.5 Proves Problematic**](https://huggingface.co/unsloth/Qwen2.5-3B): Users face challenges fine-tuning **Qwen 2.5**, with weight merging leading to gibberish outputs. Effective fine-tuning demands high-quality datasets to maintain performance.
- [**DeepSeek R1 Shines on Modest Machines**](https://youtu.be/EHGmPn6RVwU): A user showcases **DeepSeek R1** performing well on an **M1 Air 16GB**, demonstrating that even less powerful hardware can handle advanced models, sparking discussions on model efficiency.

**Theme 4. AI Hardware and Infrastructure Developments**

- [**AMD's ROCm Enters the AI Hardware Race**](https://youtu.be/SB7Yt-FGWEs): **AMD** promotes its **ROCm** platform for running LLMs on their GPUs, challenging **NVIDIA's CUDA** and aiming to grow its AI hardware presence.
- [**Unsloth Pro Still Lacks Multi GPU Support**](https://docs.unsloth.ai/basics/unsloth-benchmarks): Despite user inquiries, **Unsloth Pro** has yet to add multi GPU support. The team promises it will arrive *"soon,"* but users remain eager for the feature.
- [**GB200 GPUs Nowhere to Be Found**](https://x.com/lambdaapi/status/1890028876954489125): Users express frustration over the unavailability of **GB200 GPUs**, willing to pay but unable to find access, highlighting the scarcity of cutting-edge GPUs for AI enthusiasts.

**Theme 5. Ethical and Security Concerns in AI**

- [**Deepfake Tech Sparks Penalty Debates**](https://discord.com/channels/1053877538025386074/1149866623109439599/1339649293719572544): Members discuss the misuse of **deepfake technology**, debating if stricter penalties are needed due to regulation challenges and misinformation spread.
- [**UK Rebrands AI Safety to Security Institute**](https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/): The UK government rebrands its **AI Safety Institute** to the **AI Security Institute**, shifting focus to cybersecurity against AI risks, causing concerns about diminished attention to AI safety.
- [**Elon Musk Threatens to Withdraw OpenAI Bid**](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w): **Elon Musk** warns he may pull his bid if **OpenAI** remains a nonprofit, sparking discussions on the impact of profit motives on AI development and the organization's future.

---

# PART 1: High level Discord summaries




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Wendel Hypes Unsloth on YouTube**: Wendel lauded **Unsloth** multiple times in a [YouTube video](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) titled *'Embrace the Coming AI Revolution with Safe Local AI!'*.
   - Members reacted positively, noting Wendel mentioned Unsloth *around four times*, boosting confidence in local AI solutions.
- **DeepSeek R1 Wins Personality Contest**: Users find that **DeepSeek R1** maintains personality and detail in responses better than other models, while generics like GPT tend to produce watered-down, robotic replies, especially for character-driven applications.
   - In contrast, the community mentioned that **DeepSeek's** release has *shaken up the AI world*.
- **Multi GPU Support MIA in Unsloth Pro**: A member inquired about multi GPU support in the **Unsloth pro plan**, and was told it is still unavailable.
   - The team responded hopefully, promising the feature would be added *soon*.
- **GRPO Glitches on TPU**: The **GRPO** notebook hits compatibility errors on TPU, with the explicit limitation to NVIDIA GPUs highlighted as a barrier to broader compatibility, according to users.
   - Suggestions included switching to **NVIDIA A100** on Google Colab for successful execution of the GRPO method.
- **Ai2's Tulu 3 GRPO Gains Respect**: Discussion focused on Ai2's [Tülu 3 GRPO report](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), highlighting its significant improvements and open-source nature, with members showing admiration for Ai2's efforts.
   - The model shows state-of-the-art performance across various tasks.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 3 Supercharges Development**: Windsurf's Wave 3 release brings the **Model Context Protocol (MCP)** for custom tool calls, **customizable app icons** for Mac users, and **Turbo Mode** enhancements to the forefront. Details are in the [complete Wave 3 blog post](https://codeium.com/blog/windsurf-wave-3).
   - The update includes improvements to **Tab to Jump** navigation and drag-and-drop image support.
- **Cascade Base Behaving Badly for Brave Users**: Users report issues with Cascade Base functionality post-update, especially for free users, with login problems and general usability concerns. Many expressed they were *not able to log in or use Cascade properly*.
   - The issues appear to be linked to a recent update, sparking frustration among users.
- **Codeium Extension Consistency Craved**: Users highlight behavioral differences in the Codeium extension between **Android Studio** and **IntelliJ IDEA**, requesting uniformity, and would like the *chat open inside the IDE* for both applications.
   - Feature requests for models such as **Deepseek R1** and **Gemini 2.0 Flash** are being directed to [codeium.canny.io](https://codeium.canny.io/feature-requests).
- **Support Structure Spurs Stir**: Users seek clearer support channels specifically for the Codeium extension amidst the rising focus on Windsurf, expressing a need for a dedicated space.
   - Concerns are growing over the responsiveness of Codeium's support, especially regarding account access and error resolutions, as users desire clearer communication on community channels.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Deep Research Arrives**: Perplexity has launched **Deep Research**, a tool that autonomously generates in-depth research reports. Find more information [here](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research).
   - It is available on the web and coming soon to **iOS, Android, and Mac**, offering **5 free queries per day** for non-subscribers and **500 queries** for Pro users.
- **Deep Research Model Performance Debated**: Users are questioning whether **Deep Research** is effectively leveraging the capabilities of models like **o3-mini** due to concerns about hallucinations and limited sources.
   - Feedback indicates mixed experiences regarding its reliability and speed, with some users reporting slow performance and noting the models are not as cost-effective.
- **Sonar API Beta Testers Eager**: Enthusiasts are keen to beta test the **API version of Sonar** on **Cerebras**, with one member sharing a concept integrating **Aider**, **Sonar**, and **DeepSeek V3**.
   - A newcomer inquired about the inclusion of **Deep Research** in the **API** and the business use case, with some discussion about a *cheap coding workflow*.
- **Musk's OpenAI Bid in Jeopardy**: Elon Musk threatened to withdraw his bid if **OpenAI** remains a nonprofit, sparking discussions about the impact of profit motives on AI developments. Read about it [here](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w).
   - The move has triggered conversation about the company's future direction
- **Omega-3 Dose May Slow Aging**: An article suggests that a **daily dose of Omega-3** could slow aging processes. Details available [here](https://www.perplexity.ai/page/daily-omega-3-dose-could-slow-WOTeSIXYTRCpeznaXq9ilw).
   - Regular intake of Omega-3 may significantly impact health in the long term.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Embedding Models Suffer from Overfitting**: Large **embedding models** tend to **overfit** benchmarks, performing similarly to smaller models while using **100x more compute**.
   - The discussion highlighted the importance of context when defining what it means for a model to be *'better'*.
- **QT Layouts Confront CPTSD**: A user shared their journey learning about **QT material and layouts**, using both an **LLM** and **QT designer** for inspiration.
   - Despite facing challenges due to **CPTSD**, they expressed pride in their progress and determination to continue learning.
- **SciNewsBot Broadcasts Science Updates**: **SciNewsBot** reports daily science news on BlueSky, using fact-checked sources filtered through the **Media Bias Fact Check database** and is open-source on [GitHub](https://github.com/AstraBert/SciNewsBot).
   - It leverages the [mistral-small-latest](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) model to generate headlines and is easily deployable via Docker.
- **Qwen 2.5 fine-tuning faces challenges**: Concerns arose about fine-tuning **Qwen** with a **1k dataset**, especially regarding weight merging that resulted in unfavorable performance and gibberish outputs.
   - Insights suggested that effective fine-tuning requires high-quality instruction/answer pairs for optimal performance.
- **AI HPC Discusses DeepSeek V3**: A [YouTube video](https://youtu.be/wGWn3eVPvH8) highlights cost-effective software hardware co-design for deep learning, emphasizing increased demands in computational power and bandwidth when using **DeepSeek V3**.
   - The advancements in **Deep Learning** and **Large Language Models** are key drivers for this need, as described in the **Fire-Flyer AI-HPC** paper.



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE Users Bemoan Usability Lapses**: Users reported frustrations with **Cursor IDE**, highlighting difficulties in switching projects and managing new sessions in Composer.
   - The issues extended to slow commit message generation and inconsistent **AI model performance**, impacting overall user experience.
- **New AI Agent Leaderboard Shakes Up Rankings**: A new **AI agent leaderboard** positions **Google’s Gemini 2.0** and **OpenAI’s GPT-4o** at the forefront, sparking debate on the relative performance of models like **Sonnet** and **o3-mini**.
   - The leaderboard emphasizes **agentic models** adept at **tool integrations**, setting a new benchmark for AI capabilities.
- **MCP Server Setup Sparks Community Collaboration**: The community is actively sharing resources and advice for setting up **MCP servers** across various platforms, including [mcp-perplexity](https://github.com/daniel-lxs/mcp-perplexity).
   - Participants exchanged tips on ensuring essential tools like **uvx** are correctly installed and configured for effective server operation.
- **Subscription Model Draws Ire**: Users voiced significant dissatisfaction with the updated **pricing structure**, particularly the shift where **o3-mini** requests now deplete premium credits.
   - Many felt blindsided by the apparent end of the initial free usage period, citing a lack of transparent communication regarding the changes.
- **Tool Integration Proves Thorny Task**: Integrating **AI models**, especially **o3-mini**, with external tools within the Cursor environment poses considerable challenges, prompting discussions on effective **prompting techniques**.
   - The community is exploring enhanced methods to refine **tool calling functionality**, aiming to elevate the overall user experience and efficacy of AI-driven workflows.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Errors Plague Users**: Users reported receiving a *'received prediction-error lmstudio'* message when running multiple queries in **LM Studio**.
   - Support discussions suggest that updating to the latest version may resolve this, noting similar errors with certain **MLX models** and pointing to [an issue on GitHub](https://github.com/lmstudio-ai/mlx-engine/issues/98).
- **DeepSeek R1 Impresses on Modest Hardware**: A user compared **DeepSeek R1** performance on a high-end machine versus an **M1 Air 16GB**, finding the lower-spec machine surprisingly capable, as detailed in [this YouTube video](https://youtu.be/EHGmPn6RVwU).
   - Discussions ensued on the effectiveness of **distilled models** versus full models, with varying opinions on quality and performance.
- **LM Studio Eyes Headless Operation**: A user inquired about running **LM Studio** in headless mode on a Linux server, foregoing the GUI.
   - While currently a display is needed to launch the GUI, developers plan to integrate true headless mode in future updates, aligning with [system requirements documentation](https://lmstudio.ai/docs/system-requirements).
- **Speculative Decoding Stumbles in LM Studio**: Users are running into compatibility problems with **Speculative Decoding** in **LM Studio** when using downloaded models.
   - It was suggested that ensuring the beta runtime is active and verifying model specifications could improve its function.
- **AMD's ROCm Aims to Compete in AI**: **AMD** released a promotional [video](https://youtu.be/SB7Yt-FGWEs) highlighting the use of the **ROCm** software platform for running LLMs on their GPUs.
   - This is part of AMD’s broader strategy to increase its footprint in the **AI hardware market**, promoting competitive models and software stacks.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes-3 Launches with New Reasoning**: Nous Research released the [DeepHermes-3 Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview), which unifies **reasoning** and intuitive language model capabilities, showcasing an improvement over its predecessor.
   - To activate its long reasoning modes, a specific system prompt (`You are a deep thinking AI...`) should be used to facilitate systematic reasoning, which early benchmarks indicate enhances **Mathematical reasoning** and shows a modest improvement in **GPQA** benchmarks.
- **Deepfake Tech Sparks Debate Over Penalties**: Members expressed concerns over the misuse of **deepfake technology** and the difficulties in effectively regulating it.
   - Discussions included differing opinions on whether stricter penalties are needed for malicious use, considering existing issues with misinformation.
- **Challenges in Fine-tuning Models Surface**: Users shared challenges in **fine-tuning AI models**, particularly on platforms like **Colab**, and explored alternatives such as **LambdaLabs** and **Vast.ai**.
   - Experiences with different cloud platforms were discussed, with advice on the performance and reliability of these services for model training.
- **UltraMem Architecture Boosts LLM**: A paper introduced the **UltraMem** architecture, an ultra-sparse memory network that significantly improves the **efficiency** and **scalability** of large language models.
   - Findings indicate **UltraMem** excels in **inference speed** compared to **Mixture of Experts** while maintaining favorable scaling properties, as detailed in the [OpenReview paper](https://openreview.net/forum?id=zjeHLSiNv1).
- **1.5-Pints Achieves Model Pre-training in Days**: The [1.5-Pints Technical Report](https://arxiv.org/html/2408.03506v1) details a pre-training method that achieves language model training in just **9 days**, outperforming existing models.
   - This approach leverages a curated dataset of **57 billion tokens**, emphasizing high-quality expository content to enhance reasoning capabilities.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Eleuther AI Seeks Research Contributions**: New members seek guidance on contributing to research projects at **Eleuther AI**, particularly in areas like interpretability and deep learning.
   - They are seeking direction on how to get involved in the community effectively and leverage their backgrounds as **NLP** and **engineering students**.
- **Community IDs Image Personalities**: Users collaborated to identify people in a shared image, including **Francois Chollet** and **Gary Marcus**, showcasing community expertise and quick responses.
   - Community members efficiently tagged a comprehensive list of names linked to the image.
- **QK Norm Impedes Attention Sinks**: Discussions revealed that **QK Norm** may hinder **attention sinks**, essential for model performance, while value residuals were proposed as a possible mitigation; **forgetting transformers** could be potential solutions.
   - They agreed to further investigate these relationships and their implications on model behavior.
- **Repetition Improves LLM Performance**: Papers introduced the advantages of **hyperfitting** and **repeated training examples** for LLMs, suggesting that repetition can enhance performance compared to data diversity.
   - The conversation examined how models perform better when trained on smaller, repeated examples rather than larger datasets, raising questions about the impact of training methods on LLM capabilities with the [Emergent properties with repeated examples paper](https://arxiv.org/abs/2410.07041).
- **OpenAI Deep Research Tool Grounding Issues**: Members discussed the effectiveness of **OpenAI Deep Research** for ML/AI literature reviews, but expressed challenges in grounding its research to **arXiv** content and specific papers.
   - One participant remarked that the quality doesn't seem 'excellent', indicating skepticism about the utility of the tool, due to its reliance on less reliable blogs instead of credible academic sources.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA Kernel Hits Wall**: A user reported implementing optimizations in a CUDA kernel, such as **loop unrolling** and **warp level reductions**, but only achieved **1/3rd performance** compared to **PyTorch**, prompting discussion on optimization limits and strategies.
   - The optimized kernel, focusing on tiled transposed matrix B, performed poorly without the use of **cuBLAS**, leading to the speculation that CUDA kernel optimization has certain caps.
- **GB200 GPUs Vanish into Thin Air**: A user expressed frustration over the unavailability of **GB200 GPUs**, willing to pay but unable to find any access, highlighting challenges in acquiring the latest GPU technology.
   - Suggestions for alternative providers were offered, noting high demand for **LLM inference**, but waitlists dampen enthusiasm.
- **Llama 3.3 License Denied!**: A user reported issues obtaining a license for the **Llama 3.3 70B** base and instruct models, preventing them from conducting experiments for a research cohort in the **Cohere For AI** Discord.
   - Another user suggested using the [70B-Instruct version from Hugging Face](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) as a workaround, as a base version is unavailable.
- **Reasoning Gym Wrestles Futoshiki's Intricacies**: The **Futoshiki dataset** is more complex than initially expected, and members discussed standardizing **scoring strategies** and **answer formatting** to reduce inconsistencies in outputs.
   - Members are actively improving the **evaluation architecture** by migrating all eval-related code to a separate repository and addressing issues with leading/trailing whitespace affecting answer scoring.
- **Oumi AI wants YOU (to build Open Source)**: Oussama, co-founder at [Oumi](https://oumi.ai), shared that their startup is focused on building fully **open models and infrastructure**, promoting the belief that *open-source lifts all boats* and they are actively hiring [ML performance engineers](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393).
   - Candidates will have the opportunity to contribute to multiple [open projects](https://github.com/oumi-ai/oumi) and collaborate with a dedicated team of researchers to enhance their model speed and training pipelines, potentially using [DM](https://x.com/oussama_e) or [LinkedIn](https://www.linkedin.com/in/oussamaelachqar/) for questions.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter Reconsiders API Usage Field**: OpenRouter is considering updating the `usage` field in their API to switch from a **normalized token count** to the **model's native token count** due to advancements in tokenization; the GPT tokenizer will still be used for rankings.
   - Discussions included concerns about how this might affect model rankings and queries about which providers don't report a usage object, looking for clarity on operational practices, see [OpenRouter API Reference](https://openrouter.ai/docs/api-reference/overview#response-format).
- **Fireworks Provider Suffers Outage**: The **Fireworks provider** experienced an outage, but OpenRouter confirmed that other providers and BYOK usage were unaffected, according to [a tweet from OpenRouter](https://x.com/OpenRouterAI/status/1890419427859935275).
   - The outage was resolved by **9:12 ET**, and normal operations resumed shortly thereafter.
- **OpenAI o1 and o3 Models Made Available**: OpenAI's **o1 and o3 models** are now available to all OpenRouter users without needing a separate BYOK key, which allows for higher rate limits, documented at [OpenRouter API](https://openrouter.ai/api/v1).
   - The announcement included a **cheatsheet for model suffixes** like `:online`, `:nitro`, and `:floor` for different functionalities and pricing.
- **DeepSeek R1 Has Performance Hiccups**: Users reported that **DeepSeek R1** on OpenRouter often pauses, causing issues for their agents and raising concerns about its reliability in production, but it seems to have superior reasoning under certain settings.
   - For DeepSeek, the recommended temperature is **0.6** without a system prompt, according to [DeepSeek's official tweet](https://x.com/deepseek_ai/status/1890324295181824107).
- **API Keys Get the Strikethrough**: Users found that their API keys showed a strikethrough on the website and returned a 401 error, admins indicated that keys may be disabled due to potential leaks.
   - This highlights the importance of keeping secrets, with a reminder to use secrets.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity's 'Deep Research' Excites Users**: Users are excited about **Perplexity's** new *'Deep Research'* feature, with some accessing it even on the free tier, sparking curiosity about usage limits.
   - Members consider **Perplexity** a preferred news source due to its perceived low bias and interactive features, seeking an alternative to traditional news.
- **GPT Store Publishing Plagued by Privacy Policy Problems**: A member reported receiving an error message about needing valid privacy policy URLs when trying to publish to the **GPT Store**.
   - Another member suggested that updating the privacy policy field in actions could resolve the issue, which the original member confirmed fixed the problem.
- **ChatGPT and Playground Disparities Discussed**: Members compared using **ChatGPT** and the **Playground**, highlighting the significance of identifying and addressing errors in responses, as well as recognizing patterns.
   - One member recommended that prompts should be designed for clarity to enable the model to clearly predict user intentions, enhancing its reliability.
- **Navigating Prompt Interpretation Conflicts**: Members suggested that asking the AI model to contrast interpretations of a prompt can help uncover conflicts and ambiguities.
   - They also recommended using clear, normal language instead of strict formats to elicit more insightful responses from the AI.
- **Human Oversight Remains Critical for AI-Assisted Tasks**: Discussion underscored the critical need for human oversight in all AI-assisted processes, particularly in sensitive domains like legislative writing, where accuracy is paramount.
   - It was emphasized that a skilled human must validate and critique all AI-generated output, ensuring that responsibility is taken for the final content.



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD Users Face Lora Training Limitations**: A user shared their experience training a **Lora** with only *7 selfies*, leading to limited likeness recognition, especially for side views, suggesting a larger dataset of high-quality images would be more effective.
   - Smaller models might generalize less effectively, requiring images that match the desired output style for optimal results.
- **Community Explores AI Image Generation**: Members discussed methods for generating **AI art**, addressing challenges like achieving consistent character designs across multiple models, with recommendations of **FaceFusion** for face-swapping.
   - A query about automating image requests sparked discussions on requiring **ComfyUI** workflows, for greater control and automation.
- **Members Fine-Tune Stable Diffusion with Control Settings**: A user inquired about fine-tuning **Stable Diffusion** with control mechanisms for improved image generation, and was directed to the **L3 discord** for resources.
   - The user expressed specific interest in recent tools that enhance control over the image generation process.
- **Windows Audio Device Detection Causes Frustration**: A member humorously commented on the quirks of **Windows** detecting audio devices, joking that an ideal hardware solution could improve detection processes.
   - The discussion turned into light banter about technology frustrations, with some mentioning the paradox of being heavily reliant on computing devices despite their flaws.
- **Newcomers Welcomed Into Engaged Community**: New users introduced themselves, sharing their experiences with **AI art** and seeking advice on challenges faced with AI tools and models.
   - Existing members welcomed newcomers, showcasing an engaged community atmosphere focused on exchanging knowledge and experiences in AI art generation.



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepHermes-3 Shows Reasoning Prowess**: The **DeepHermes-3 Preview** was released, showcasing advanced reasoning by toggling capabilities for accuracy at the cost of computation, available on [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview). Benchmarks are underway against models like Tülu.
   - Concerns were raised in #[ml-drama] that DH3 only highlights *two specific evals* when reasoning is enabled, whereas all metrics are displayed when reasoning is turned off.
- **Debate Rages over Open Weight Definition**: Discussions around the **Open Weight definition** emphasized compliance for free redistribution of model weights on [Open Weight site](https://openweight.org/), sparking lively debate.
   - The definition's implications and potential effects on open-source AI practices were key discussion points.
- **UK Pivots to AI Security Over Safety**: The UK government rebranded its **AI Safety Institute** to the **AI Security Institute**, shifting focus to cybersecurity against AI risks, [TechCrunch reports](https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/).
   - Community members voiced concerns that this shift diminishes the focus on AI safety.
- **DeepSeek-R1 Deployment Sparks Excitement**: Enthusiasm surrounds the deployment of **DeepSeek-R1**, with recommended settings including no system prompt and a temperature of **0.6**, as per [official recommendations](https://fxtwitter.com/deepseek_ai/status/1890324295181824107).
   - Users emphasized the importance of using the official deployment to ensure a similar experience to the official version, mitigating potential bypass issues.
- **XAI Plots Significant Data Center Expansion**: Elon Musk’s **xAI** seeks a new data center to support increased **Nvidia** chip usage, according to [The Information](https://www.theinformation.com/briefings/musk-looks-for-another-data-center-for-xai-nears-5-billion-chip-deal-with-dell).
   - This expansion signals ambitious growth efforts in the competitive AI landscape.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM becomes 24/7 Tutor**: A user described how **Notebook LM** has transformed their medical study routine by creating detailed summaries and key points from extensive readings, calling *it literally a Personal Tutor who is available 24/7 at your fingertips*.
   - The user emphasized the tool's accessibility and utility for learning.
- **Gen Z Slang Makes Learning Fun**: A member highlighted the effectiveness of customizing prompts to use **Gen Z brainrot social media slangs** for explaining complex concepts.
   - This approach helped them grasp difficult subjects in more relatable language, making learning more accessible and easier.
- **PDF Uploads Plagued by Mystery Bugs**: A user reported trouble uploading PDFs, regardless of file size or complexity, while others reported no issues, suggesting a problem tied to the user's browser or system safety filters when dealing with potentially sensitive content.
   - Other members were able to upload files without any trouble.
- **Language Support Stumbles in Notebook LM**: Users reported challenges in getting Notebook LM to respond in their selected languages, like Bulgarian and German, even after uploading sources in those languages, however other users reported it works as expected.
   - Some found success using specific URLs like [notebooklm.google?hl=bg](https://notebooklm.google?hl=bg) for Bulgarian.
- **Gemini Model Functionality in Limbo**: Several users inquired about the new **Gemini model's** functionalities, particularly how it integrates within **Notebook LM**.
   - Responses indicated uncertainty about Gemini's capabilities within the platform, with users pointing to related resources for exploration.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **LLMs Utilize Latent Reasoning**: A new paper introduces **latent reasoning** in LLMs, which happens in the model's hidden space before token generation, contrasting with chain of thought methods, discussed on [this tweet](https://x.com/MatthewBerman/status/1890081482104008920?t=V3aeg7FX8ZvIKtvhHtl-xA&s=19).
   - Community members are actively discussing the practical implications and potential benefits of this approach.
- **Nvidia's Veo 2 Enhances Video Creation**: Nvidia's new model, **Veo 2**, featured on YouTube Shorts, enables creators to generate video clips from text prompts using the **Dream Screen** feature, as announced in [this tweet](https://x.com/GoogleDeepMind/status/1890054036168356283).
   - This allows for seamless integration of user-generated content, enhancing storytelling capabilities.
- **Apple Teases New Device Launch**: **Tim Cook** teased an upcoming Apple launch on [his X feed](https://x.com/tim_cook/status/1890068457825394918), hinting at potential new products like the **iPhone SE**, **M4 Air**, and updated Apple TV options.
   - Speculation includes a **HomePod with screen** and further integration of powerful chips for AI capabilities, sparking community interest.
- **DeepHermes 3 Eyes Superior LLM Abilities**: Nous Research's **DeepHermes 3** model, available on [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview), aims to merge reasoning and traditional LLM response modes into a single architecture.
   - The goal is to substantially improve LLM annotation, judgement, and function calling capabilities.
- **Community Shares Beekeeping Business Plan**: A member shared a comprehensive **Beekeeping Feasibility Report** at [this link](https://docs.google.com/document/d/1BIBInKu1rET9-BC520OXXz95XuFOxuMAr63nytI9DZQ/edit?usp=sharing), offering actionable steps and insights for potential business strategies.
   - Discussions around researching and optimizing prompts for deep research enriched the community's understanding of leveraging AI in real-time projects.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Embraces Google Cloud**: LlamaIndex introduced new features to integrate with [Google Cloud databases](https://twitter.com/llama_index/status/1890109073615626388), facilitating usage as an initial data store and vector store.
   - The integrations are designed to be **easy and secure**, streamlining database interactions.
- **LlamaParse Power Boosted**: A detailed video on [LlamaParse](https://twitter.com/llama_index/status/1890499579214491967) demonstrates various parsing modes, output formats, and techniques to improve quality using parsing instructions.
   - The video covers parsing **audio**, **images**, and utilizing **JSON mode** for optimized results.
- **AgentWorkflow Deemed Unsuitable for RAG**: `AgentWorkflow` is designed for systems of agents executing tasks, not RAG, as described in [documentation](https://docs.llamaindex.ai/en/stable/examples/workflow/rag/).
   - Users are advised to create custom functions to integrate RAG within `AgentWorkflow` for RAG processing.
- **`uv` tool Speeds Up Environment Management**: Users shared the benefits of using `uv` for creating multiple virtual environments, with shared insights on managing different versions for tools like PyTorch.
   - One user even offered a shell function to streamline switching between environments and associated project files for enhanced convenience.
- **India's AI Community Beckons**: An invitation to join India’s fastest-growing AI community aims to foster connections and collaboration, inviting members to innovate in artificial intelligence.
   - Interested individuals can join the community via the provided [WhatsApp link](https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU) to become part of the growing scene.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama gains fame over OpenRouter**: **Glama** emerges as the preferred choice over [OpenRouter](https://glama.ai/models/) due to its **lower cost**, **higher speed**, and **privacy guarantees**, albeit with fewer supported models.
   - Glama's pricing ranges from **$0.06 to $10** across various models, tipping the balance for developers prioritizing efficiency and confidentiality.
- **OpenWebUI breaking things often**: Users report that **OpenWebUI** experiences frequent **breaking changes** with minor updates, impacting the functionality of a substantial portion of community features.
   - Some users suggest it's due to its status as *experimental alpha software* prone to race conditions, complicating its usability.
- **0.0.0.0 IP Address causes confusion**: The use of the IP address **0.0.0.0** sparks debate, especially concerning its role in **containerized environments** where it typically listens on all interfaces.
   - Some members cautioned against using it as a destination in HTTP contexts and emphasized the importance of understanding proper usage for **troubleshooting**.
- **MCP Server Author roles given out**: Members shared links to their servers and GitHub repos to get the **MCP server author role**.
   - Providing a demo server project or library qualified members for the **author status**.
- **Zonos TTS MCP gives Claude a voice**: The [Zonos TTS MCP](https://github.com/PhialsBasement/Zonos-TTS-MCP) server enhances user interaction by giving **Claude** a voice akin to **CGPT**.
   - The incorporation of a **markdown interpreter** is expected to further improve Claude's **intonation**, bringing it closer to the optimal performance.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Community Asks RAG Evaluation**: A computer vision expert asked the community about **metrics** for evaluating their **RAG system**, which has a stable retrieval setup, specifically asking for guidance on metrics used in evaluating **LLMs** or **retrieval architectures**.
   - They seek recommendations for metrics used in evaluating **LLMs** or **retrieval architectures** in **RAG systems**.
- **Tinystories is more than just Pretrained Models**: Members clarified that **Tinystories** encompasses not just a set of pretrained models, but also a family of architectures, a dataset, and a research paper detailing the setup process.
   - They emphasized that Tinystories did the hard work necessary to achieve coherent output from small models and are useful for those just starting.
- **Delaying Normalization**: A discussion explored delaying normalization to improve **RL performance** in generative sequence models, suggesting that irregularities may be beneficial, and using **dynamic logits**.
   - Strategies include using dynamic logits and incorporating **SFT** to guide the model toward meaningful outcomes in training.
- **AI Thinks Without Tokens**: A [YouTube video](https://www.youtube.com/watch?v=ZLtXXFcHNOU) explores whether **models can 'think' without using tokens**, posing an intriguing question about **AI capabilities**.
   - An [arXiv paper](https://arxiv.org/abs/2502.05171) presents a novel language model architecture that scales test-time computation by reasoning in latent space without needing specialized training data.
- **Public Model Releases Inconsistent**: An empirical study of **52,227 PTLMs** on **Hugging Face** revealed that **40.87%** of model weight changes weren't reflected in naming practices or documentation, according to [this paper](https://arxiv.org/abs/2409.10472).
   - These results highlighted **ambiguity** in naming conventions and the accessibility of training documentation for Pre-trained Language Models.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Enforces Strict PR Submission Rules**: Contributors must triple-check PRs for whitespace changes; submissions containing AI-generated code are discouraged to save time and encourage individual coding.
   - The guidelines emphasize the importance of personally writing code and using AI for feedback, as opposed to submitting AI-generated code directly.
- **Insights on Kernel and OptOps Speed Bounty**: A member proposed creating an **OptOp** to optimize the **AST** for multiple reductions in the context of the `sum` bounty.
   - They voiced concerns about the expressiveness of current **OptOps** and suggested exploring the **GROUP OptOp** for multiple accumulators, anticipating that the renderer should mostly function as expected.
- **VIZ on WSL Troubleshooting**: A user reported errors when using `VIZ=1` on **WSL Ubuntu** due to issues accessing the temporary directory.
   - Another member admitted that **WSL** builds can be difficult, especially with Python, and offered to investigate the issue by downloading the required setup.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Crushes LangChain For Advanced Use Cases**: Members suggested that **DSPy** is preferable to **LangChain** if users need optimization or prefer writing signatures and modules over string prompts.
   - It was noted that **LangChain** might be a better choice if a prepackaged solution is desired.
- **DSPy 2.6 Changelog Surfaces**: A user inquired about the changelog for **DSPy 2.6**, specifically regarding 'instructions' for *Signatures* and a member pointed out that these instructions have been around since **2022**.
   - The user was directed to the [GitHub release page](https://github.com/dspy/dspy/releases) for comprehensive details on the changes.
- **DSPy Drops Assertions, Sparks Confusion**: The removal of **dspy.Assert**, **dspy.Suggest**, and **dspy.Retry** in **DSPy 2.6.3** led to confusion about backward compatibility and suitable alternatives.
   - A member speculated that this removal is part of a plan to introduce *assertions v2*, though no official roadmap or explanation has been provided.
- **DSPy Tackles Multi-label Classification**: A user sought advice on using **DSPy** to optimize an **SLM** for multi-label classification involving 200 class descriptions, considering a batching strategy.
   - The user specifically aimed to avoid fine-tuning the model or using multiple **LoRA adapters**.
- **DSPy Code Golf Gains Traction**: A **DSPy** code golf activity was proposed, challenging community members to create succinct code snippets.
   - One member shared a one-liner example for extracting structured data from HTML, inviting others to participate in what could become a competitive coding game, referencing [Omar Khattab's tweet](https://x.com/lateinteraction/status/1890442615700545878).



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX and Mojo ❤️ Valentine's Day**: **MAX and Mojo** spread the love this Valentine's Day with a cheerful greeting and a fun image titled `MAXMojoValentine.jpeg` shared in the **general** channel.
   - This interactive element brought a sense of joy and community to the channel.
- **v25.1 Release Sparks 🔥**: An anonymous user announced the release of **v25.1**, garnering enthusiasm from the community.
   - The exclamation mark and fire emoji indicate high interest in the updates brought by this release.
- **Larecs Repo Gets the Tree 🌳**: A member provided a link to the [Larecs GitHub repository](https://github.com/samufi/larecs) for others interested in further details.
   - The tree emoji implies a focus on growth or development within the project.
- **Safe Mutable Aliasing Doc Spotted**: A user asked for a link to a document on **safe mutable aliasing** authored by another member, who shared a link to their [proposal/vision document](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) published in November.
   - The code appears to create conflicts with memory locations accessed through aliased arguments.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Token Banning Configuration Queried**: A member inquired about the possibility of banning tokens via configuration files, acknowledging that it's not a feature available in the GUI.
   - This reflects a desire for advanced customization of token behavior beyond the officially supported methods.
- **Qwen2.5 Coder 14B Proposed for RTX 3080**: Discussions revealed that distilling **Deepseek** behavior onto a smaller model may cause performance reductions on an **RTX 3080**, prompting suggestions for alternative models.
   - The **Qwen2.5 Coder 14B** was recommended for lower VRAM configurations, though members noted the performance trade-offs.
- **LLM Fine-Tuning Limitations Discussed**: A member asked how to update and fine-tune an **LLM** with data from 2021, and it was clarified that it is not possible to adapt older models with new data.
   - This highlights the limitations of updating existing models with newer datasets.
- **TradingView Premium Unleashed for Free**: Links to free cracked versions of **TradingView** for Windows and macOS were shared, noting its large user base, along with [installation instructions](https://www.reddit.com/r/TradingViewFree/comments/1hobjs6/tradingview_premium_ultimate_package_update/).
   - The post emphasizes the availability of **Premium** features at no cost through this method.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Dataloader Transform RFC Streamlines Data Gen**: A member proposed an RFC to add a dataloader transform and saving capability, enhancing online **DPO/GRPO data generation** at train time.
   - An example was shared showing how the **prompt_to_preference** function utilizes a `DataLoader` to generate batches of preference data, suggesting viability for batched generation.
- **Distillation Scaling Laws Debated**: Discussion focused on [a paper from Apple](https://arxiv.org/abs/2502.08606) on **distillation scaling laws**, pondering whether it's better to distill from a more powerful model or train from scratch.
   - One participant emphasized *'it's complicated...'* regarding choices surrounding model size and capabilities during the distillation process.
- **Quantization-Aware Training Achieves Accuracy**: A new study advanced the understanding of **Quantization-Aware Training (QAT)**, exploring ways to achieve accuracy with quantized representations, particularly with an optimal bit-width of **8-bits**.
   - The study was validated by referencing the state-of-the-art research paper [arXiv:2411.04330v2](https://arxiv.org/abs/2411.04330v2).
- **QuEST Method Rivals FP16 for Compression**: A member introduced **QuEST**, a new method for compression claiming strong accuracy at model sizes of **4-bits** or less for weights and activations.
   - The method is positioned as **Pareto-competitive with FP16**, purportedly delivering better accuracy at reduced model sizes.


---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Confusion Surrounds Quiz 3 Release**: A member reported confusion over the release of **Quiz 3**, initially unable to locate it on the MOOC website.
   - The user later discovered the announcement on [Discord](https://discord.com/channels/1280234300012494859/1293323662300155934), resolving the issue.
- **Newbie Solicits AI/ML Training Advice**: A new member requested guidance on where to begin with **AI/ML** model training techniques.
   - They are also seeking resource recommendations to advance their knowledge beyond initial training, encouraging suggestions for courses and forums.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1339644199321997325)** (621 messages🔥🔥🔥): 

> `LoRA Fine-Tuning, Model Training and RAG, PDF Data Extraction, AI Hardware Support, Model Evaluation and Performance` 


- **Challenges with LoRA Fine-Tuning**: Discussions highlighted that while LoRA can help in fine-tuning models, it risks overfitting or introducing catastrophic forgetting, especially with poorly balanced datasets.
   - It was noted that balancing the model’s general knowledge while incorporating new information is critical for maintaining performance.
- **Utilizing RAG for Company Knowledge Access**: A user shared their intent to implement a locally hosted model for answering company-specific queries, considering transforming data to JSON for easier access.
   - Using RAG (Retrieve and Generate) was recommended for this task, as training could complicate updates with new information.
- **Extracting Data from PDFs**: The community discussed methods for extracting text, graphics, and images from PDFs, with a consensus that converting PDFs to images for OCR tends to yield better results.
   - Challenges with accurately processing tables from PDFs were emphasized, and vision models were suggested as potentially more resilient to complex layouts.
- **AI Hardware Discussions**: Conversations around the limitations of AMD vs. NVIDIA in AI contexts noted that the CUDA ecosystem significantly influences the availability and efficiency of AI training tools.
   - Users remarked on the possibility of renting powerful GPUs, such as H100s and exploring free credits from platforms like Runpod for AI tasks.
- **Model Evaluation and Performance Queries**: Questions about VRAM requirements for running 7B models at FP16 highlighted that many factors, such as software and model input size, can affect performance.
   - Specific adjustments to reward functions in training setups were discussed to ensure models properly output expected answer formats.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...</li><li><a href="https://huggingface.co/silx-ai/Quasar-1.5-Pro">silx-ai/Quasar-1.5-Pro · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/docs/peft/en/conceptual_guides/prompting">Soft prompts</a>: no description found</li><li><a href="https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model">Fine-Tuning DeepSeek R1 (Reasoning Model)</a>: Fine-tuning the world&#x27;s first open-source reasoning model on the medical chain of thought dataset to build better AI doctors for the future.</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">Continued Pretraining | Unsloth Documentation</a>: AKA as Continued Finetuning. Unsloth allows you to continually pretrain so a model can learn a new language.</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth Benchmarks | Unsloth Documentation</a>: Want to know how fast Unsloth is?</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://docs.unsloth.ai/g">Unsloth Documentation</a>: no description found</li><li><a href="https://vast.ai/">Rent GPUs | Vast.ai</a>: Reduce your cloud compute costs by 3-5X with the best cloud GPU rentals. Vast.ai&#x27;s simple search interface allows fair comparison of GPU rentals from all providers.</li><li><a href="https://jina.ai/">Jina AI - Your Search Foundation, Supercharged.</a>: Best-in-class embeddings, rerankers, LLM-reader, web scraper, classifiers. The best search AI for multilingual and multimodal data.</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth Requirements | Unsloth Documentation</a>: Here are Unsloth&#x27;s requirements including system and GPU VRAM requirements.</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benc">Unsloth Documentation</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth-zoo/tree/main/unsloth_zoo">unsloth-zoo/unsloth_zoo at main · unslothai/unsloth-zoo</a>: Utils for Unsloth. Contribute to unslothai/unsloth-zoo development by creating an account on GitHub.</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1339759797003616257)** (9 messages🔥): 

> `Wendel's Unsloth shoutouts, RAG Implementation, Deepseek's AI release` 


- **Wendel mentions Unsloth multiple times**: Wendel highlighted Unsloth in a recent [YouTube video](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) titled 'Embrace the Coming AI Revolution with Safe Local AI!' expressing excitement about local AI solutions.
   - Viewers celebrated the mentions, with one noting that Wendel said their name *around four times*.
- **User inspired by Wendel's message**: A member expressed enthusiasm about Wendel's remarks, indicating that they align with their own efforts in pushing for local AI tools despite restrictions at work.
   - They stated, 'If my company doesn’t allow us to use tools like OpenAI, then I can build it locally' to support their team.
- **Discussion on RAG Implementation**: A member inquired about resources for setting up RAG (Retrieval-Augmented Generation) implementation.
   - Another quickly recommended using **Llama Index** or **Haystack** as accessible starting points.



**Link mentioned**: <a href="https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ">Embrace the Coming AI Revolution with Safe Local AI!</a>: Deepseek&#39;s release has shaken up the AI world, and we&#39;re on the precipice of the AI Industrial Revolution! Wendell gives you the low down on how to take that...

  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1339652059217137694)** (244 messages🔥🔥): 

> `DeepSeek R1 Performance, Training with LORA and RAG, GRPO Reward Function Issues, Model Compatibility with TPU, HPC Cluster Training Errors` 


- **DeepSeek R1 outperforms generic models**: Users found that **DeepSeek R1** maintains personality and detail in responses better than other models, while generics like GPT tend to produce watered-down, robotic replies.
   - There were discussions on specific character-driven applications and concerns about the impact of group communication on AI training.
- **Challenges in training models with LORA and RAG**: Some users experienced issues with models generating unwanted text formats such as `
```user` or `
```assistant`, suggesting training data issues or potential overfitting.
   - Questions arose about the suitability of training instruct models for generating cleaner outputs.
- **GRPO Reward Function and Compatibility Issues**: In efforts to integrate a pretrained reward function with **Unsloth**, users encountered errors related to Llama's architecture modifications, specifically with `LlamaAttention` attributes.
   - The conversation revolved around potential workarounds to avoid function overwrites, emphasizing the need for distinct handling of legacy reward models.
- **Limitations of TPU Support in Training**: Users discovered that the **GRPO** notebook encountered compatibility errors on TPU, with the explicit limitation to NVIDIA GPUs highlighted as a barrier to broader compatibility.
   - Suggestions included shifting to NVIDIA A100 on Google Colab for successful execution of the GRPO method.
- **Performance Optimization and Long Training Times**: Concerns arose regarding long training times, with some noting training steps taking upwards of a minute, leading to extended total training durations.
   - Users discussed potential adjustments to batch sizes and training parameters to optimize performance without running into CUDA Out of Memory errors.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://download.pytorch.org/whl/cu126">no title found</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb#scrollTo=KN6nELjXcRez.">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=kR3gIAX-SM2q"">Google Colab</a>: no description found</li><li><a href="https://redux.js.org/introduction/getting-started">Getting Started with Redux | Redux</a>: Introduction &gt; Getting Started: Resources to get started learning and using Redux</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb),">Google Colab</a>: no description found</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb">Google Colab</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth Documentation</a>: no description found
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1339829219009499148)** (9 messages🔥): 

> `RLHF Reward Modeling, Tülu 3 GRPO, Multi GPU Support in Unsloth, OLMoE Improvements, New Optimizer` 


- **RLHF Reward Modeling Explored**: A link was shared to the [RLHF Reward Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling) GitHub repository, providing recipes for training reward models for RLHF.
   - *Let's goooo*, showing excitement from members about the new resources available.
- **Insights on Tülu 3 GRPO Model**: Discussion focused on Ai2's [Tülu 3 GRPO report](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), highlighting its significant improvements and open-source nature.
   - The model shows state-of-the-art performance across various tasks, with members expressing admiration for Ai2's efforts.
- **Awaiting Multi GPU Support in Unsloth**: A member inquired about multi GPU support in the Unsloth pro plan, receiving an update that it is still not available.
   - The response indicated hope for the feature to be added *'soon'*.
- **OLMoE Updates Posted**: Links were shared about [OLMoE improvements](https://allenai.org/blog/olmoe-app), showcasing new iterations for iOS apps.
   - Members expressed enthusiasm about Ai2's advancements in the OLMoE project.
- **Announcement of New Optimizer**: A reference to a **new optimizer** was shared through a link to an [arXiv paper](https://arxiv.org/pdf/2502.07529).
   - This marks a point of interest for further exploration and implementation in the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/RLHFlow/RLHF-Reward-Modeling">GitHub - RLHFlow/RLHF-Reward-Modeling: Recipes to train reward model for RLHF.</a>: Recipes to train reward model for RLHF. Contribute to RLHFlow/RLHF-Reward-Modeling development by creating an account on GitHub.</li><li><a href="https://huggingface.co/allenai/">allenai (Ai2)</a>: no description found</li><li><a href="https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B">allenai/Llama-3.1-Tulu-3.1-8B · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/YXTYbr3hiFU?si=yXVQhYu4szFn42Gz">An Unexpected Reinforcement Learning Renaissance</a>: The era we are living through in language modeling research is one pervasive with complete faith that reasoning and new reinforcement learning (RL) training ...
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1339643524978577469)** (2 messages): 

> `AI Engineering Summit Tickets, Windsurf Wave 3 Features, Model Context Protocol, Customizable App Icons, Turbo Mode Enhancements` 


- **Win Free Tickets to the AI Engineering Summit!**: We're giving away **3 tickets** to the AI Engineering Summit on **February 20-21** in New York City, featuring exclusive experiences and a chance to meet Windsurf’s Head of Product Engineering.
   - Those interested must [fill out the form](https://forms.gle/WM67ZgQngXaY4stq7) to qualify, and only NYC area residents are eligible.
- **Windsurf Wave 3 has dropped!**: Wave 3 introduces several exciting features including the **Model Context Protocol (MCP)** for custom tool calls and **customizable app icons** for Mac users.
   - Enhancements like the **Turbo Mode** and improved **Tab to Jump** navigation have also been implemented, as detailed in the [complete Wave 3 blog post](https://codeium.com/blog/windsurf-wave-3).
- **Model Context Protocol is Now Available**: Cascade now supports the **Model Context Protocol (MCP)**, allowing user-configured tool calls; every action costs one flow action credit regardless of results.
   - Users can set up MCP by clicking the hammer icon in the Cascade input tool bar, available to all individual plans.
- **Customizable App Icons Launching for Mac Users**: Windsurf allows users to change their app icons with options like **Classic**, **Blueprint**, and **Hand-drawn**, currently in beta for Mac users.
   - A system restart is required for changes to take effect across the operating system, and is available to all paid user plans.
- **Turbo Mode Enhancements Unveiled**: The latest update for Windsurf includes **Turbo Mode** for auto-executing commands and drag-and-drop image support.
   - This update also features improved credit visibility and expanded @docs options for enhanced user experience.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://forms.gle/WM67ZgQngXaY4stq7">Tickets to AI Engineering Summit NYC from Windsurf </a>: We want to say thank you to our community by giving away three free tickets to the AI Engineering Summit in New York! This two-day event on February 20-21 is a chance to hear from top AI experts, expl...</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>: Introducing Wave 3, our third batch of updates to the Windsurf Editor.</li><li><a href="https://x.com/windsurf_ai/status/1890161230876381249">Tweet from Windsurf (@windsurf_ai)</a>: Wave 3 is here!Included in this update:⏩ Tab to Jump🔗 MCP Integration⚡ Turbo Mode🎨 Custom Icons… and more.</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Latest updates and changes for the Windsurf Editor.
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1339695800758046752)** (31 messages🔥): 

> `Announcement speculation, Codeium extension behaviors, Windsurf frustrations, Feature requests for the extension, User support for Codeium` 


- **Speculations on New Announcements**: There's anticipation regarding new announcements, particularly for those outside of NYC, hinting at more to come later today.
   - *Stay tuned* for more surprises was the sentiment expressed in the discussion.
- **Different Behaviors of Codeium Extension**: A user noted the discrepancy in behavior of the Codeium extension between **Android Studio** and **IntelliJ IDEA**, seeking consistency.
   - Specifically, the preference is to have the chat open *inside the IDE* for both applications.
- **Frustrations over Windsurf Dominance**: Several users expressed frustration over the overwhelming focus on **Windsurf**, detracting from discussions and support for the Codeium extension.
   - One remarked that the move to Windsurf felt like a *bait-and-switch* for those primarily interested in coding tools.
- **Feature Requests for New Models**: Inquiries arose regarding the future support for models such as **Deepseek R1** and **Gemini 2.0 Flash** in the Codeium extension.
   - Users were encouraged to submit feature requests to [codeium.canny.io](https://codeium.canny.io/feature-requests) to express their needs.
- **Desire for Clear Separation in Support Discussions**: Amidst discussions, users yearned for a dedicated space solely for the Codeium extension, expressing dissatisfaction with the mixing of topics.
   - The call for a *cleaner channel* focused on the extension reflects the community's desire for clear and organized support.


  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1339643336390086841)** (622 messages🔥🔥🔥): 

> `Cascade Base issues, MCP Server Configuration, Windsurf Performance, User Experience with WindSurf, Codeium Support Feedback` 


- **Cascade Base Not Working for Free Users**: Users are currently experiencing issues with Cascade Base, particularly free users who report that it is not functioning as intended after updates.
   - Many users, both on free and paid plans, expressed frustration over not being able to log in or use Cascade properly, with some suggesting that it may be connected to a recent update.
- **MCP Server Configuration Issues**: Discussions arose around configuring MCP servers in Windsurf, with some users not being able to find the options as detailed in the documentation.
   - Users confirmed that they needed to follow specific procedures to set up the MCP servers effectively.
- **Performance and User Feedback on Windsurf**: Several users reported experiencing lag or unresponsiveness in Windsurf, particularly when using various models and features.
   - There were suggestions for improving the user experience, including options to mute certain colors and enhance workflow efficiency.
- **Prompting and Usage Efficiency**: Users discussed the importance of how prompts are structured to optimize interactions with Cascade for code editing and debugging tasks.
   - It was suggested that providing clear instructions on efficient AI coding usage would help users avoid pitfalls.
- **Support and Communication Feedback**: Concerns were raised about the responsiveness and effectiveness of Codeium's support, particularly regarding issues with account access and errors.
   - Users expressed the need for clearer communication about ongoing issues and potential resolutions in the community channels.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://localhost:12345",">no title found</a>: no description found</li><li><a href="https://docs.codeium.com/windsurf/mcp">Cascade MCP Integration</a>: no description found</li><li><a href="https://marketplace.visualstudio.com/items?itemName=avli.clojure">Clojure&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Extension&#32;for&#32;Visual&#32;Studio&#32;Code&#32;-&#32;Clojure&#32;nREPL&#32;support&#32;for&#32;Visual&#32;Studio&#32;Code</li><li><a href="https://docs.codeium.com/windsurf">Windsurf - Getting Started</a>: no description found</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: Give feedback to the Codeium team so we can make more informed product decisions. Powered by Canny.</li><li><a href="https://x.com/windsurf_ai/status/1889804247618945045/photo/1">Tweet from Windsurf (@windsurf_ai)</a>: You can then view the diagram by opening the file with an extension such as Markdown Preview Mermaid Support!</li><li><a href="https://www.reddit.com/r/codeium">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/just-do-it-shia-la-beouf-do-it-gif-4531935">Just Do It Shia La Beouf GIF - Just Do It Shia La Beouf Do It - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://status.codeium.com/">Codeium Status</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=OIV1vKm59Xg">Windsurf Wave 3 Updates: Tab to Jump, MCP, Custom App Icons, Turbo Mode &amp; More</a>: Windsurf wave 3 of updates is here! 🚀 Check out the latest features that make Windsurf even more powerful:Tab to Jump ⏩ Easily navigate within files to make...</li><li><a href="https://github.com/renatokuipers/neural-child">GitHub - renatokuipers/neural-child: A neural network system that develops through psychological stages from infancy to maturity, implementing emotional regulation, attachment, and theory of mind capabilities.</a>: A neural network system that develops through psychological stages from infancy to maturity, implementing emotional regulation, attachment, and theory of mind capabilities. - renatokuipers/neural-c...</li><li><a href="https://docs.djangoproject.com/en/4.2/topics/i18n/translation/">Translation | Django documentation</a>: The web framework for perfectionists with deadlines.
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1340013903475900517)** (1 messages): 

> `Perplexity Deep Research, Deep Research features, Free queries, App availability, Research capabilities` 


- **Perplexity Deep Research Launches**: Perplexity introduced **Deep Research**, enabling users to generate in-depth research reports on any topic by autonomously performing searches and analyzing sources.
   - This feature can handle expert-level tasks across various fields and has high scores on **Humanity's Last Exam**.
- **Free Access with Query Limits**: Deep Research is available for **free**, allowing non-subscribers up to **5 queries per day**, while Pro users can make up to **500 queries** daily.
   - This tiered access aims to cater to a wider audience with varying research needs.
- **App Rollout Details**: **Deep Research** is now accessible on the web, with plans to roll out to **iOS, Android, and Mac** soon, pending app updates.
   - Users are encouraged to update their apps to the latest version for the best experience.
- **Deep Research User Guide**: To use Deep Research, users should navigate to [perplexity.ai](https://perplexity.ai) and select the “Deep Research” mode from the search box before submitting their query.
   - Further details about the feature can be found [here](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research).
- **Deep Research Video Introduction**: An introductory video for Deep Research was shared, aiming to give users a visual demonstration of its functionalities.
   - The video can be accessed directly at [DeepResearchVideo.mp4](https://cdn.discordapp.com/attachments/1047204950763122820/1340013907879661689/DeepResearchVideo.mp4?ex=67b0d0b3&is=67af7f33&hm=13e19e20d5268c195ec334aa20c9207933f4403069baca9ab6679faeb4581718&).



**Link mentioned**: <a href="https://perplexity.ai)">no title found</a>: no description found

  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1339644531456479322)** (601 messages🔥🔥🔥): 

> `Perplexity Deep Research, AI Model Performance, Feedback on Subscription Plans, User Experience with Models, Issues with Deep Research Search` 


- **Clarifications on Deep Research Model**: Users discussed that Deep Research currently utilizes the **o3-mini** model, and there were inquiries about its performance compared to R1.
   - Concerns were raised about hallucinations and whether it is indeed leveraging the capabilities of the intended models effectively.
- **Subscription and Pricing Talks**: There was a conversation about the pricing of models like **o1** and **opus**, with varying opinions on their cost-effectiveness and performance.
   - Users highlighted that some models had limits on usage, with interest in the practicality of plans like a **12-month PRO subscription**.
- **User Experience with Models**: Discussions were held regarding the reliability and speed of Deep Research, with some users expressing frustrations over its slow performance and limited sources.
   - Members shared their tests with various queries, indicating that they had mixed experiences, including prolonged search times.
- **Future of AI Models and Features**: Users pondered the future integration of reasoning models and speculated on what features might emerge, such as the potential of a **'Plexy'** assistant.
   - Opinions were divided on whether such features would enhance user experiences or create more complexity.
- **Deep Research Search Issues**: Several users voiced that Deep Research was not searching effectively, leading to confusion and a lack of satisfactory responses from the AI.
   - A user specifically noted that upon enabling the web search feature, connections to relevant content were still lacking.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.testingcatalog.com/perplexity-ai-tests-deep-research-feature-challenging-gemini-and-chatgpt/">Perplexity tests deep research feature, challenging ChatGPT</a>: Discover Perplexity&#x27;s new deep research feature in Alpha, offering comprehensive web analysis and detailed reports. Gradual rollout expected. Try it now!</li><li><a href="https://www.lefigaro.fr/sports/football/ligue-1/ligue-1-roberto-de-zerbi-prepare-deja-les-annees-suivantes-a-l-om-20250214">Ligue 1 : Roberto De Zerbi prépare déjà «les années suivantes» à l’OM</a>: L’entraîneur marseillais a révélé, ce vendredi en conférence de presse, qu’il préparait déjà les saisons futures avec ses dirigeants.</li><li><a href="https://tenor.com/view/hasbulla-hasbik-cute-meme-influencer-gif-21732737">Hasbulla Hasbik GIF - Hasbulla Hasbik Cute - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/testingcatalog/status/1890360609318801911">Tweet from TestingCatalog News 🗞 (@testingcatalog)</a>: BREAKING 🚨: Perplexity is preparing a release of Deep Research Alpha 🔥As this feature works on Gemini and ChatGPT, perplexity can take more time to go through web resources and utilise its reasoning...</li><li><a href="https://x.com/perplexity_ai/status/1890452005472055673?t=6Rs4ecHRvoXxYA-hFU9yYw&s=19">Tweet from Perplexity (@perplexity_ai)</a>: Introducing Deep Research on Perplexity.Deep Research lets you generate in-depth research reports on any topic.Available to everyone for free—up to 5 queries per day for non-subscribers and 500 querie...</li><li><a href="https://x.com/elder_plinius/status/1890028958907089059?t=Kv46N8eXldfN35QN-zmGhQ&s=19">Tweet from Pliny the Liberator 🐉 (@elder_plinius)</a>: MUAHAHAHA 💉💦Quoting Djah 〰️ (@Djahlor) WHAT??? @elder_plinius  did you do this??</li><li><a href="https://x.com/perplexity_ai/status/1889366732432674961">Tweet from Perplexity (@perplexity_ai)</a>: We are excited to announce that the winner of the Million Dollar Questions Sweepstakes is Kaylee Edmondson! Kaylee is a small business owner from Nashville, TN.Congratulations, Kaylee. And thank you t...</li><li><a href="https://tenor.com/view/fire-writing-gif-24533171">Fire Writing GIF - Fire Writing - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://dubesor.de/benchtable">Dubesor LLM Benchmark table</a>: no description found
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1339648086435299340)** (18 messages🔥): 

> `Daily Omega-3 Dose, Inflation Trends, Musk's Bid on OpenAI, ChatGPT Energy Consumption, N8N JavaScript Usage` 


- **Daily Omega-3 Dose may slow aging**: A recent article discusses how a **daily dose of Omega-3** could slow aging processes, bringing intriguing insights on dietary habits. You can read more about it [here](https://www.perplexity.ai/page/daily-omega-3-dose-could-slow-WOTeSIXYTRCpeznaXq9ilw).
   - *Studies suggest* that regular Omega-3 intake can significantly impact health in the long term.
- **Inflation Rises Unexpectedly**: Perplexity AI highlighted the topic of unexpected rise in **inflation**, indicating potential impacts on the economy. For a deeper understanding, watch the [YouTube video](https://www.youtube.com/embed/39eW2gMthuU).
   - Experts are closely monitoring these changes with concerns over **economic stability**.
- **Musk threatens to withdraw bid for OpenAI**: Elon Musk has stated he will withdraw his bid if **OpenAI remains a nonprofit**, raising questions about the company's future direction. For the complete story, check out [the article](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w).
   - *This move has sparked discussions* about the impact of profit motives on AI developments.
- **ChatGPT's energy use potentially overestimated**: **ChatGPT's energy consumption** has come under scrutiny, with claims that it may have been overestimated in previous reports. Full insights can be found in this [detailed examination](https://www.perplexity.ai/page/chatgpt-energy-use-overestimat-cn02azRBR2._eM_sH2n_Pw).
   - Critics note that understanding actual energy usage is crucial for assessing environmental impact.
- **N8N Integration with JavaScript**: An inquiry was made on how to **use JavaScript in N8N**, a popular automation tool, providing insights into its functionality. Detailed guidance can be found [here](https://www.perplexity.ai/search/como-usar-javascript-no-n8n-7c6Fo3WWRfCV0Aq1O0ClMg).
   - This integration opens up **customization options** for users looking to enhance their workflows.



**Link mentioned**: <a href="https://www.youtube.com/embed/39eW2gMthuU">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1339756492059836437)** (5 messages): 

> `Sonar API Beta Testing, Aider and DeepSeek V3 Integration, Cheap Coding Workflow, Business Use Case for Perplexity API, Deep Research API Feature` 


- **Dreams of Beta Testing Sonar API**: A member expressed eagerness to beta test the **API version of Sonar** on **Cerebras**, stating they've been dreaming about it for months.
- **Aider + Sonar + DeepSeek V3 Integration**: One member shared a concept integrating **Aider** for reasoning, **Sonar** for architecture, and **DeepSeek V3** as the coding component, accompanied by an image.
   - [View the image here](https://cdn.discordapp.com/attachments/1161802929053909012/1339881751291494480/Screenshot_2025-02-14_at_09.51.13.png).
- **Experimenting with a Budget Coding Workflow**: A member shared their experience testing out a **'cheap' coding workflow**, indicating some humor about the situation.
- **Request for Official Email Regarding Payment**: A newcomer requested assistance on the **business use case of the Perplexity API** and needed an official email for credit card payments.
   - They mentioned having difficulty getting a response from customer service.
- **Inquiry About Deep Research in the API**: A member inquired whether **Deep Research** will be included in the **API**, seeking clarification on upcoming features.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1339665348630413403)** (37 messages🔥): 

> `Embedding Models, Vision Transformer Dimension, Open Deep Research Demo, Speech to Text Using Deepgram, User Interface Concerns` 


- **Embedding Models Overfitting Concerns**: Discussion highlighted that many large embedding models tend to **overfit** benchmarks, often yielding similar performance to smaller models while utilizing **100x more compute**.
   - One member warned about the careful usage of the term **'better'**, indicating that it's context-dependent.
- **Determining Optimal Projection Dimension for ViTs**: A user inquired about the appropriate **projection dimension** for a vision model with given patch size and number of channels, mentioning the original **768 dimension** from the paper.
   - The conversation underscored the importance of ensuring the dimension is adequately large while debating how small might lead to poor results.
- **Open Deep Research Demo Issues**: A user reported a potential downtime issue with the **open deep research demo**, prompting another member to quickly confirm that it should now be **fixed**.
   - The user expressed appreciation with a simple acknowledgment following the fix.
- **Challenges with Speech to Text Implementations**: A member sought advice on using a **speech to text model** for automated speech recognition, noting difficulties with Deepgram's documentation.
   - Another member suggested using **Whisper** as an alternative, indicating a search for non-GCP/AWS solutions.
- **Questions about User Interface Elements**: A user inquired about the purpose of an unspecified option, with another clarifying its role in finetuning models with their data.
   - The conversation emphasized a tool named **Autotrain** as a means to facilitate this process, with members engaged in sharing resources.



**Link mentioned**: <a href="https://tenor.com/view/hmmm-thinking-batman-gif-6153870554148391864">Hmmm Thinking GIF - Hmmm Thinking Batman - Discover &amp; Share GIFs</a>: Click to view the GIF

  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1339662003773378653)** (5 messages): 

> `Neuralink Updates, Chat Templates and Transformers, QT Material and Layouts, Agent's Unit 1, Dataset-Tools Development` 


- **Exploring Neuralink Visuals**: A member shared images related to **Neuralink**, initiating discussions on its recent developments.
   - These images sparked interest, but specific details and analysis were not elaborated.
- **Discussing Chat Templates and Transformers**: Another member mentioned **chat templates and transformers** in the context of ongoing learning.
   - No additional specifics were provided, but this highlights a trend towards enhancing chatbot frameworks.
- **Learning QT Layouts with Manual Inspiration**: One user shared their journey learning about **QT material and layouts**, using both an LLM and QT designer for inspiration.
   - Despite facing challenges due to **CPTSD**, they expressed pride in their progress and determination to continue learning.
- **Agent's Unit 1 Discussions**: A member referenced discussions surrounding **Agent's Unit 1**, indicating interest in this aspect of AI development.
   - However, detailed insights or questions were not provided to further the conversation.
- **Progress on Dataset-Tools Project**: A user celebrated their achievements with their **dataset-tools** toy project, noting its near-completion.
   - They also reflected on how long the layout vision had taken to materialize, aiming for further advancements.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1339696791830138912)** (7 messages): 

> `Jokes Generator API, SciNewsBot for BlueSky, Browser Engines and WASM` 


- **Jokes Generator brings humor to HuggingFace**: The **Jokes Generator** fetches jokes from a Joker REST API and features a UI with a Gradio chat interface, allowing users to enjoy some laughs. Check it out [here](https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI).
   - One user expressed excitement about the tool being *extremely amazing* for only **97kb**.
- **Introducing SciNewsBot for Daily Science News**: **SciNewsBot** reports daily science news on BlueSky, using fact-checked sources filtered through the Media Bias Fact Check database. The bot relies on the [mistral-small-latest](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) model and is open-source.
   - It generates catchy headlines and is user-friendly, allowing anyone to reproduce it locally or launch it via Docker.
- **Discussion on WebAssembly and Browser Engines**: A member recalled that similar projects have been done using **browser engines** and suggested the relevance of **WASM** in this context. Another member acknowledged this prior knowledge with a light-hearted comment.
   - *Commentary on the evolution of web technologies* indicates a familiarity with integrated systems and their applications.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI">Xo JokeGen NoAI - a Hugging Face Space by xyizko</a>: no description found</li><li><a href="https://github.com/AstraBert/SciNewsBot">GitHub - AstraBert/SciNewsBot: A simple, AI-powered bot to report daily news about environment, technology, science and energy on BlueSky.</a>: A simple, AI-powered bot to report daily news about environment, technology, science and energy on BlueSky. - AstraBert/SciNewsBot</li><li><a href="https://bsky.app/profile/sci-news-bot.bsky.social">SciNews Bot (@sci-news-bot.bsky.social)</a>: Hi! I&#39;m an AI-powered bot built by Clelia Astra BertelliI post daily on the latest news about Environment, Science, Technology and Energy.I have an internal fact-checking algorithm but you are wel...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1339643528904572978)** (10 messages🔥): 

> `Technical difficulties, Zoom meeting, Session recording, Presentation feedback` 


- **Technical difficulties cause meeting adjustments**: The group is experiencing **technical difficulties** today, prompting members to redirect their attention to a Zoom link for the meeting: [Zoom Meeting Link](https://mcgill.zoom.us/j/85033055096). A member apologized for the disruptions, indicating that the meeting may not proceed as planned.
   - Despite the issues, a thank you was expressed for the presentation, reflecting **appreciation** for the context provided around the discussed paper.
- **Meeting moved to Zoom**: The meeting has been officially moved to Zoom, accessible via [this link](https://mcgill.zoom.us/j/85033055096). Members were encouraged to attend digitally despite the tech challenges.
   - One member assured that the session would be recorded for those uncomfortable with Zoom, ensuring **accessibility** after the fact.
- **Positive presentation feedback**: A participant shared heartfelt gratitude towards presenters, appreciating the context they added to the paper. This highlights the **value** of collaboration and shared knowledge within the community.
   - Another member acknowledged having to leave early but expressed a strong sentiment of having enjoyed the additional insights shared during the session.
- **Session recording offered for comfort**: Assurances were given that the session would be recorded for anyone not comfortable joining via Zoom. This reflects a commitment to inclusivity within the group.
   - The presence of varied communication tones — including light-hearted emojis — shows a supportive and engaging environment.



**Link mentioned**: <a href="https://mcgill.zoom.us/j/85033055096">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1339690447219720265)** (1 messages): 

> `Canny Edge Detection, Sobel Filters, Machine Learning in Preprocessing, ControlNet with Diffusion Models` 


- **Canny Edge Detection and Sobel Filters as Starting Points**: You can start with **Canny edge** or **Sobel filters** and then consider the pipeline if a **trained model** is absolutely necessary for detection.
   - These methods can serve as effective **pre-processing stages** before applying machine learning to different downstream tasks.
- **ControlNet Utilizes Edge Filtered Images**: **ControlNet** employs **Canny edged filtered images** with a diffusion model to ensure images maintain **structural consistency** with the original.
   - This emphasizes that sometimes a robust model isn't needed, as edge filters can enhance **image generation** tasks effectively.


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1339675120100315226)** (10 messages🔥): 

> `Qwen Model Performance, Fine-tuning Issues, End Token Generation, Quality of Training Data, Chat Templates Knowledge` 


- **Qwen 2.5 Model Isn't a Base Model**: A user concluded that **Qwen 2.5** is definitely not a base model, suggesting it has more sophisticated capabilities.
   - They noted that it behaves differently from models without finetuning, indicating an understanding of chat templates.
- **Fine-tuning Challenges with Qwen**: Concerns about fine-tuning **Qwen** with a **1k dataset** surfaced, especially regarding weight merging leading to unfavorable performance.
   - A user expressed confusion over the model producing gibberish post-merging, hinting at potential issues with training quality.
- **End Token Generation Explained**: It was clarified that **end tokens** are generated only if they are likely to be the next token, indicating the model's recognition of language patterns.
   - A discussion highlighted concerns over training to avoid continuous looping in token generation, emphasizing the need for effective finetuning strategies.
- **Importance of Training Data Quality**: The quality of training data became a focal point; poor data quality could lead to gibberish outputs during inference.
   - Insights suggested that effective fine-tuning requires high-quality instruction/answer pairs for optimal performance.
- **Base Model Lacks Chat Template Awareness**: It was noted that the base version of the **Qwen model** does not understand chat templates, affecting its interaction capabilities.
   - The distinction between base and instruct models was emphasized, with confirmation that without finetuning, the model demonstrates limited conversational ability.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Qwen2.5-3B">unsloth/Qwen2.5-3B · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-3B-bnb-4bit">unsloth/Qwen2.5-3B-bnb-4bit · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1339661258487631955)** (2 messages): 

> `HF_TOKEN definition, Model changes` 


- **Investigating HF_TOKEN Configuration**: *Check the logs* to ensure that you are defining the **HF_TOKEN** properly, as its absence might be causing issues.
   - Addressing this could potentially resolve underlying problems with configuration.
- **Suggestion to Change the Model**: A recommendation was made to *try changing the model* as a troubleshooting step, which might yield different results.
   - This approach could help in isolating the issue and improving performance.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1339643569270423603)** (284 messages🔥🔥): 

> `Course Introduction, Certificate Issues, Collaborative Learning, Agent Development, LLM Exploration` 


- **Course Participants Introduce Themselves**: New participants, including individuals from diverse backgrounds and countries like India, Brazil, the US, and more, introduced themselves and shared their excitement about the AI Agents course.
   - Participants expressed their eagerness to learn about AI agents and collaborate with others in the course.
- **Challenges with Course Certificates**: Several participants reported issues accessing their certificates after completing the unit quizzes, encountering repeated login prompts despite being logged in.
   - Some users suggested simple troubleshooting techniques, such as re-logging into specific spaces, to resolve these issues.
- **Translation Initiatives for Course Content**: One member mentioned translating the course materials into Portuguese and offered to share their notes for others struggling with English.
   - There was a suggestion to establish a centralized location for translations of the course content into various languages to aid broader accessibility.
- **Discussion on Building Custom Tools**: Participants discussed getting started with building custom tools using provided schemas, with some seeking additional resources for clarity.
   - Users were encouraged to review lessons and engage with the community for support as they navigate through the course material.
- **Exploration of LLMs in AI**: There was interest in exploring LLMs (Large Language Models) as foundational models for building agents, with members seeking lists or links for further reading.
   - Participants engaged in discussions about the capabilities and definitions of LLMs, contemplating how to frame their descriptions.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: no description found</li><li><a href="https://huggingface.co/learn/agents-course/unit1/tutorial">Let’s Create Our First Agent Using smolagents - Hugging Face Agents Course</a>: no description found</li><li><a href="https://www.youtube.com/live/iLVyYDbdSmM">Welcome To The Agents Course! Introduction to the Course and Q&amp;A</a>: In this first live stream of the Agents Course, we will explain how the course will work (scope, units, challenges and more) and answer your questions.Don&#39;t ...</li><li><a href="https://youtu.be/gHPhy1tL0Xg?si=XLNY8lvhWKRjLB_5">Python Full Course for Beginners (2025)</a>: Python is one of the most versatile and in-demand programming languages today. From web development and artificial intelligence to data science and automatio...</li><li><a href="https://youtu.be/Tr5_wgwHlCs?si=2x_t2NlGHN2LU4f1">NumPy Full Course (Beginner to Advanced)</a>: This comprehensive course will guide you from the fundamentals of NumPy to advanced techniques.Course Highlights:✅ Beginner-friendly explanations✅ Practical ...</li><li><a href="https://youtu.be/TJ_iroAmfn8?si=nwQyhgdBAPH1OuTG">Pandas Full Course for Data Analysis</a>: This comprehensive course will guide you from the fundamentals of Pandas to advanced techniques.Course Highlights:✅ Beginner-friendly explanations✅ Practical...</li><li><a href="https://youtu.be/n6i2lRMbIKI?si=kkboVaAL6DBli2lt">Matplotlib Full Course for Data Visualization</a>: This comprehensive course will guide you from the fundamentals of Matplotlib to advanced techniques, helping you create stunning visualizations in Python.Cou...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1339746840513478839)** (8 messages🔥): 

> `DeepSeek V3, Granite 3.2 MoE, ESFT Paper Review, Community Call Discussions` 


- **DeepSeek V3: Fire-Flyer AI-HPC Insights**: A YouTube video titled ["DeepSeek 🐋 | Fire-Flyer AI-HPC"](https://youtu.be/wGWn3eVPvH8) discusses cost-effective software hardware co-design for deep learning, emphasizing increased demands in computational power and bandwidth.
   - The advancements in **Deep Learning** and **Large Language Models** are highlighted as key drivers for this growing need.
- **Granite 3.2 MoE Analysis**: A preview of **Granite 3.2 MoE** suggests that it seems to have distilled data from **GPT-3.5**, with its training data only extending to **2021**.
   - The user expresses skepticism about its performance, questioning its ability to succeed.
- **Expert Specialized Fine-Tuning (ESFT) Discussion**: A user inquired about the **ESFT** paper, sharing a link to the [GitHub repository](https://github.com/deepseek-ai/ESFT) where the project is hosted.
   - The repository focuses on **Expert Specialized Fine-Tuning**, aiming to improve model performance through targeted training.
- **Concerns about Community Calls**: A user raised a question about the inability to conduct **community calls**, indicating a desire for more engagement.
   - This reflects a need for enhanced communication and collaboration within the community.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.14158">Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning</a>: The rapid progress in Deep Learning (DL) and Large Language Models (LLMs) has exponentially increased demands of computational power and bandwidth. This, combined with the high costs of faster computi...</li><li><a href="https://www.kaggle.com/code/stevugnin/open-r1-zero">Open R1 Zero</a>: Explore and run machine learning code with Kaggle Notebooks | Using data from [Private Datasource]</li><li><a href="https://youtu.be/wGWn3eVPvH8">DeepSeek 🐋 | Fire-Flyer AI-HPC:  A Cost-Effective Software Hardware Co-Design for Deep Learning</a>: The rapid progress in Deep Learning (DL) and Large Language Models (LLMs) has exponentially increased demands of computational power and bandwidth. This, com...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: no description found</li><li><a href="https://github.com/deepseek-ai/ESFT">GitHub - deepseek-ai/ESFT: Expert Specialized Fine-Tuning</a>: Expert Specialized Fine-Tuning. Contribute to deepseek-ai/ESFT development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1339643832991612951)** (333 messages🔥🔥): 

> `Cursor IDE usability, AI model performance, MCP server usage, Subscription issues, Tool integration struggles` 


- **Cursor IDE usability struggles**: Users are experiencing difficulties with the Cursor IDE, including switching projects and issues related to new sessions in Composer, which sometimes require users to create new ones to maintain focus.
   - There are complaints about the speed of commit message generation and the performance of certain AI models within the IDE.
- **AI model performance rankings**: A new AI agent leaderboard has been launched, with Google’s Gemini 2.0 and OpenAI’s GPT-4o ranking at the top, which sparked discussions about Sonnet's position compared to o3-mini.
   - Participants indicated that the leaderboard focuses on agentic models that excel at tool integrations and usage.
- **MCP server installation and issues**: Users discussed setting up the MCP server on different platforms, sharing links to repositories like [mcp-perplexity](https://github.com/daniel-lxs/mcp-perplexity) created by members of the community.
   - There was advice on ensuring necessary tools like uvx are installed and how to run these servers effectively in various environments.
- **Subscription and pricing changes**: Frustrations were expressed about the new pricing structure where requests using o3-mini now count towards premium requests, leading to complaints about the service's reliability.
   - Users noted that the initial free usage seemed to have ended without clear communication from the providers.
- **Tool integration challenges**: Discussion highlighted the challenges of getting various AI models, particularly o3-mini, to effectively integrate and utilize external tools within the Cursor environment.
   - There were mentions of needing better prompting techniques to improve tool calling functionality and user experiences.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.cursor.com/settings/models">Cursor – Models</a>: no description found</li><li><a href="https://www.galileo.ai/blog/agent-leaderboard">Introducing Our Agent Leaderboard on Hugging Face - Galileo AI</a>: We built this leaderboard to answer one simple question: &quot;How do AI agents perform in real-world agentic scenarios?&quot;</li><li><a href="https://tenor.com/view/funny-gif-27151298">Funny GIF - Funny - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://x.com/donvito/status/1890247522914038082?s=46&t=ggmESCIXF0nYw8_kshHz7A">Tweet from Melvin Vivas (@donvito)</a>: love the MCP implementation of Windsurf 😍here&#39;s how it compares to cursor✅ works in chat mode. I think MCP is more useful here than agent mode. I could be wrong.✅ easy access to the configure MCP...</li><li><a href="https://github.com/daniel-lxs/mcp-perplexity">GitHub - daniel-lxs/mcp-perplexity: MCP Server for the Perplexity API.</a>: MCP Server for the Perplexity API. Contribute to daniel-lxs/mcp-perplexity development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=wnHyYKNnUTo">Why Big Tech&#39;s Betting Billions on Nuclear (Not Renewables)</a>: Why Does AI Need Nuclear Power? Get $200 off a Roborock Saros 10 until February 16th at https://amzn.to/3CNzZQP &amp; https://bit.ly/3CH2pMm. As the new wave of ...</li><li><a href="https://forum.cursor.com/t/supervisory-agent-to-guide-worker-agent/49395/7">&quot;Supervisory&quot; agent to guide &quot;worker&quot; agent</a>: Aider showed that this is how you achieve the best AI coder agent.  Also, an agent that implements this flow would be absolutely game changer for Cursor I think.  So yeah, great suggestion!</li><li><a href="https://tenor.com/view/darth-vader-alter-the-deal-empire-strikes-back-star-wars-gif-15971205">Darth Vader Alter The Deal GIF - Darth Vader Alter The Deal Empire Strikes Back - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://forum.cursor.com/t/mcp-servers-no-tools-found/49094/31">MCP Servers No tools found</a>: Thanks this finally worked for me. I was also having issues with Brave Search but got that fixed by adding the API key to the Brave Search file.  Sequential Thinking Server Setup  Install the package ...</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>: Introducing Wave 3, our third batch of updates to the Windsurf Editor.</li><li><a href="https://github.com/daniel-lxs/mcp-starter">GitHub - daniel-lxs/mcp-starter: A lightweight Go application that parses JSON configuration files and executes commands with specified environment variables.</a>: A lightweight Go application that parses JSON configuration files and executes commands with specified environment variables. - daniel-lxs/mcp-starter</li><li><a href="https://www.npmjs.com/package/mcp-server-pagespeed">mcp-server-pagespeed</a>: A Model Context Protocol server for Google PageSpeed Insights. Latest version: 1.0.0, last published: 3 days ago. Start using mcp-server-pagespeed in your project by running `npm i mcp-server-pagespee...</li><li><a href="https://www.npmjs.com/package/mcp-mysql-server">mcp-mysql-server</a>: A Model Context Protocol server for MySQL database operations. Latest version: 0.1.0, last published: 11 hours ago. Start using mcp-mysql-server in your project by running `npm i mcp-mysql-server`. Th...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1339642316985729144)** (154 messages🔥🔥): 

> `Error Handling in LM Studio, Performance Comparison of Models, Headless Mode in LM Studio, Speculative Decoding Support, Model Architecture Changes` 


- **Error Handling in LM Studio**: Users reported issues with LM Studio throwing a 'received prediction-error lmstudio' message when running multiple queries, leading to frustrations.
   - Support discussions indicated that updating to the latest version might resolve the issue and that certain MLX models showed similar errors.
- **Performance Comparison of Models**: A user conducted an in-depth comparison between DeepSeek R1 and M1 Air 16GB, noting impressive performance from the lower-spec machine.
   - There were discussions about the efficacy of distilled models versus full models, with varying opinions on quality and performance metrics.
- **Headless Mode in LM Studio**: A user inquired about running LM Studio headlessly on a Linux server without displaying the GUI.
   - Current functionality requires a display to launch the GUI, but true headless mode is planned for future updates.
- **Speculative Decoding Support**: Multiple users expressed difficulties with Speculative Decoding, mentioning compatibility issues with downloaded models.
   - Discussions suggested ensuring beta runtime is selected and checking model specifications for use with this feature.
- **Model Architecture Changes**: Users debated the nature of model training, specifically whether DeepSeek R1 should be considered a fine-tune or a new architecture.
   - The conversation revealed differing experiences with various models, including Dolphin 3.0 and its comparative performance.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/lmstudio-ai/mlx-engine/issues/98">Unusual memory behaviour with Qwen2.5-VL models · Issue #98 · lmstudio-ai/mlx-engine</a>: I&#39;ve been trying out Qwen2.5-VL in LM Studio 0.3.10 Beta 1 and I think I&#39;ve discovered some odd behaviour. I&#39;m using mlx-community&#39;s Qwen2.5 VL 72B 4bit model. The machine I&#39;m runn...</li><li><a href="https://github.co">GitHub · Build and ship software on a single, collaborative platform</a>: Join the world&#39;s most widely adopted, AI-powered developer platform where millions of developers, businesses, and the largest open source community build software that advances humanity.</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1?tab=readme-ov-file#distilled-model-evaluation>)">GitHub - deepseek-ai/DeepSeek-R1</a>: Contribute to deepseek-ai/DeepSeek-R1 development by creating an account on GitHub.</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/issues">lmstudio-ai/mlx-engine</a>: 👾🍎 Apple MLX engine for LM Studio. Contribute to lmstudio-ai/mlx-engine development by creating an account on GitHub.</li><li><a href="https://lmstudio.ai/docs/system-requirements">System Requirements | LM Studio Docs</a>: Supported CPU, GPU types for LM Studio on Mac (M1/M2/M3/M4), Windows (x64/ARM), and Linux (x64)</li><li><a href="https://www.ebay.com/itm/275857855418">Nvidia P100-SXM2-16GB P100 PCIe 16 GB Tesla GPU  | eBay</a>: no description found</li><li><a href="https://youtu.be/EHGmPn6RVwU">DeepSeek R1: $5000 vs $1000 Computer | M4 Max 128GB vs M1 16GB | LM Studio Tutorial</a>: 💻 Thinking about upgrading your MacBook for AI? I tested DeepSeek R1’s open-source models on a $5000 M4 Max (128GB) vs a $1000 M1 Air (16GB) – the results w...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1339685640178237500)** (172 messages🔥🔥): 

> `AMD ROCm promotion, NVIDIA RTX 3500 Ada, 2023 AI hardware market, Stability issues with new hardware, VRAM performance with multiple GPUs` 


- **AMD pushes for AI with ROCm video**: AMD released a [promotional video](https://youtu.be/SB7Yt-FGWEs) showcasing running LLMs on their GPUs using the ROCm software platform.
   - This reflects AMD’s drive to bolster its position in the AI hardware market as competitive models emerge.
- **Uncertainty around NVIDIA RTX 3500 Ada series**: A user expressed curiosity over the new NVIDIA RTX 3500 Ada GPU found in a Dell notebook, noting difficulty in finding detailed information about it.
   - They surmised that the naming approach by NVIDIA seemed erratic, perhaps reusing numbers and adding Ada to categorize the newer offerings.
- **Stability troubles with new motherboard setup**: A user detailed frustrations with installing a new DDR5 motherboard that caused instability during POST and in Windows, leading to a return request.
   - Despite swapping power supplies and attempting various configurations, they faced crashing issues, even with a supported CPU.
- **Discussions on multi-GPU utility in AI workloads**: A user noted that with dual 3090s, their GPUs seemed to alternate working rather than maximizing usage, attributing VRAM capacity as a major advantage.
   - The conversation hinted at potential implementations like VLLM software to allow better parallel performance in future updates.
- **Concerns over AMD's competitiveness in AI**: Pricing revelations regarding the **Radeon RX 9070 XT** raised skepticism about AMD’s 2023 viability for AI tasks as consumers lean toward alternatives.
   - The suggestion emerged that acquiring two used 3090s might provide better performance at a comparable cost.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/dram/sandisks-new-hbf-memory-enables-up-to-4tb-of-vram-on-gpus-matches-hbm-bandwidth-at-higher-capacity">SanDisk's new High Bandwidth Flash memory enables 4TB of VRAM on GPUs, matches HBM bandwidth at higher capacity</a>: Equipping AI GPUs with 4TB of memory.</li><li><a href="https://en.wikipedia.org/wiki/Ampere_(microarchitecture)">Ampere (microarchitecture) - Wikipedia</a>: no description found</li><li><a href="https://x.com/GawroskiT/status/1890159776241447142">Tweet from Tomasz Gawroński (@GawroskiT)</a>: 9070 XT hope this prices i found are placeholder... https://www.mlacom.si/komponente/graficne-kartice/i_3026151_acer-predator-bifrost-radeon-rx-9070-xt-oc-16g-gddr6-graficna-kartica</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/qI6IFznwEB">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/YmDHEOZvYF">Reddit - Dive into anything</a>: no description found</li><li><a href="https://en.wikipedia.org/wiki/Ada_Lovelace_(microarchitecture)">Ada Lovelace (microarchitecture) - Wikipedia</a>: no description found</li><li><a href="https://youtu.be/SB7Yt-FGWEs">Explained in 90 seconds: Running LLMs on AMD GPUs</a>: Learn how easy it is to run LLMs on AMD GPUs with the AMD ROCm™ software platform.This video shows how simple it is to:• Install AMD ROCm™ software platform ...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1339707671850254398)** (2 messages): 

> `DeepHermes-3 Preview, Long Chain of Thought Reasoning, LLM Model Improvements, Community Feedback on Reasoning Models` 


- **DeepHermes-3 Preview Launches with Exciting Features**: The new [DeepHermes-3 Preview](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) by Nous Research combines reasoning and intuitive language model capabilities, improving overall performance.
   - This model enhances LLM annotation, judgement, and function calling, marking a significant upgrade from its predecessor, Hermes 3.
- **Unlock Long Chain of Thought Reasoning with a Simple Prompt**: To activate long reasoning modes, a specific system prompt must be used: `You are a deep thinking AI...` which facilitates systematic reasoning surrounded by <think> tags.
   - This togglable feature aims to improve accuracy but may increase computational time during testing.
- **Early Benchmarks Show Mathematical Reasoning Improvements**: Preliminary assessments indicate significant enhancements in **Mathematical reasoning** capabilities with the **DeepHermes-3** model when long chains of thought are activated.
   - A modest improvement in **GPQA** benchmarks was also observed, suggesting the model's robustness in handling complex queries.
- **Community Collaboration Vital to DeepHermes Development**: The development of DeepHermes-3 has been supported by contributions from various community members, crucial for enhancing datasets and evaluation tools.
   - Nous Research encourages ongoing community feedback to explore and improve the new reasoning paradigms introduced with this model.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF · Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1339649293719572544)** (217 messages🔥🔥): 

> `DeepHermes-3 Preview, Deepfake Technology Discussions, Training and Fine-tuning Models, Model Performance Comparisons, Technical Issues with Models` 


- **DeepHermes-3 Preview Release**: The community is excited about the release of the DeepHermes-3 Preview model, highlighting its ability to toggle between reasoning and intuitive responses with improved performance.
   - Users have begun testing the model, noting some repetitive outputs and requesting a version on Nous Chat.
- **Concerns with Deepfake Technology**: Members discussed the implications of deepfake technology, expressing concerns over its potential misuse and the challenges in regulating it effectively.
   - There are differing opinions on the need for stricter penalties for malicious deepfake usage given the existing issues with misinformation.
- **Training and Fine-tuning Models**: Individuals shared their experiences and challenges in fine-tuning AI models, particularly with resources like Colab and suggested alternatives such as LambdaLabs and Vast.ai.
   - The topic of using various cloud platforms for training models was explored, with members advising on the performance and reliability of different services.
- **Model Performance Comparisons**: Models like DeepSeek's distill and their performance on tasks were contrasted with DeepHermes-3, emphasizing the strengths and weaknesses of each approach.
   - Community members are interested in benchmarks between models to assess capabilities in reasoning and conversation tasks.
- **Technical Issues with Models**: Users reported various technical issues with the DeepHermes-3 model, including errors during multi-turn conversations that prevented consistent reasoning outputs.
   - Members are troubleshooting these issues, discussing the intricacies of implementing models for different tasks, and suggesting code adjustments.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>: no description found</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f">Update tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>: no description found</li><li><a href="https://zed.dev/blog/edit-prediction">Zed now predicts your next edit with Zeta, our new open model - Zed Blog</a>: From the Zed Blog: A tool that anticipates your next move.</li><li><a href="https://huggingface.co/hance-ai/descript-audio-codec-44khz">hance-ai/descript-audio-codec-44khz · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF/tree/main">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF at main</a>: no description found</li><li><a href="https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237">Doubt Press X GIF - Doubt Press X La Noire - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://fxtwitter.com/dreamworks2050/status/1890164583249375377">Tweet from M4rc0𝕏 (@dreamworks2050)</a>: DEEPHERMES-LLAMA-3-8B thinking mode: ON - FIRST RUNGGUF - F16 by @NousResearch 🔥MacBook Pro M4 Max : 28.98t/s</li><li><a href="https://f-droid.org/packages/superfreeze.tool.android/">SuperFreezZ App stopper | F-Droid - Free and Open Source Android App Repository</a>: Entirely freeze all background activities of apps.</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview#prompt-format-for-function-calling">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: no description found</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">Tweet from Nous Research (@NousResearch)</a>: Introducing DeepHermes-3 Preview, a new LLM that unifies reasoning and intuitive language model capabilities.https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-PreviewDeepHermes 3 is built fr...</li><li><a href="https://huggingface.co/Joseph717171/DeepHermes-3-Llama-3.1-8B-Preview-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF">Joseph717171/DeepHermes-3-Llama-3.1-8B-Preview-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF · Hugging Face</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=kSNKuHX9AZo">BYD stock prices goes ballistic after they revealed this...</a>: BYD stock prices goes ballistic after they revealed this...The best solar company in Australia just installed my new solar system. Check them out here: https...</li><li><a href="https://www.youtube.com/watch?v=n8yva8YRVPU">Elon Musk says Grok 3 is going to be &#39;scary smart&#39;</a>: Elon Musk says Grok 3 will be released in a week or two.
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1339731890663325879)** (13 messages🔥): 

> `SFT on Llama-3B-Instruct, Fine-tuning local AI, Training costs of language models, 1.5-Pints technical report` 


- **Challenges with SFT on Llama-3B-Instruct**: A user reported that during SFT on **Llama-3B-Instruct** with a learning rate of **2e-4**, there was a significant performance drop measured in **Winogrande** due to domain-specific factors.
   - Another user suggested lowering the learning rate to **5e-5** and implementing grad accumulation for better normalization.
- **Cost of training a 1B model from scratch**: A user inquired about the costs associated with training a **1B model** from scratch, estimating it to be in the thousands of dollars, potentially tens of thousands.
   - Discussion revealed that with ample tokens, a consumer-grade GPU could finish training in about **six months**.
- **Insights on 1.5-Pints Pre-training Report**: A link to the [1.5-Pints Technical Report](https://arxiv.org/html/2408.03506v1) was shared, detailing a method of pre-training a language model in **9 days** while outperforming prior state-of-the-art models.
   - The approach utilizes a meticulously curated dataset of **57 billion tokens**, focusing on expository content to enhance reasoning capabilities.
- **Notes on fine-tuning methodologies**: Members discussed fine-tuning their local AI models with varying methodologies and reasons behind it, emphasizing the importance of training parameters.
   - Suggestions included adjusting learning rates and considering gradient accumulation to optimize training outcomes.
- **General advice on model training**: Several users exchanged tips and adjustments for effectively training models, focusing on learning rate adjustments and step counts.
   - Common advice included reducing learning rates and paying attention to the global batch size for improved performance.



**Link mentioned**: <a href="https://arxiv.org/html/2408.03506v1#:~:text=Using%20our%20pre,16K%20context%20window%20version%2C%20which">1.5-Pints Technical Report: Pretraining in Days, Not Months – Your Language Model Thrives on Quality Data</a>: no description found

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339642667889328198)** (3 messages): 

> `LLM report papers, Ultra-sparse memory networks, Kimik and Synthlab papers, Inference speed in LLMs` 


- **Searching for Recent LLM Report Papers**: A member is on the lookout for LLM report papers that cover recent state-of-the-art methods, particularly in reasoning models, noting that an LLM survey paper from February 2024 is now outdated.
   - *teknium* suggested that the **Kimik** and **Synthlab** papers are the most relevant to this search.
- **UltraMem: A Game Changer for LLM Efficiency**: A paper presented on [OpenReview](https://openreview.net/forum?id=zjeHLSiNv1) introduces **UltraMem**, an ultra-sparse memory network boosting the efficiency and scalability of large language models without sacrificing performance.
   - The findings show that **UltraMem** has a significant advantage over the **Mixture of Experts** approach in terms of inference speed, while demonstrating favorable scaling properties.



**Link mentioned**: <a href="https://openreview.net/forum?id=zjeHLSiNv1">Ultra-Sparse Memory Network</a>: It is widely acknowledged that the performance of Transformer models is logarithmically related to their number of parameters and computational complexity. While approaches like Mixture of Experts...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339642667889328198)** (3 messages): 

> `LLM report papers, Ultra-sparse memory network, Mixture of Experts, Scaling laws` 


- **Searching for Recent LLM Methods**: A user expressed the need for recent LLM report papers covering state of the art methods like **reasoning models**, hinting that a February 2024 survey paper felt outdated.
   - Another member mentioned that **r1 kimik** and **synthlab** papers are highly relevant to the search.
- **Ultra-sparse Memory Networks Shape the Future**: A paper on the **UltraMem** architecture reveals how a large-scale, ultra-sparse memory layer drastically improves the **efficiency** and **scalability** of large language models while maintaining performance, particularly excelling in inference speed compared to Mixture of Experts.
   - The authors highlight that this approach reduces **inference latency** and investigates **scaling laws**, indicating superior scaling properties over existing MoE methods.



**Link mentioned**: <a href="https://openreview.net/forum?id=zjeHLSiNv1">Ultra-Sparse Memory Network</a>: It is widely acknowledged that the performance of Transformer models is logarithmically related to their number of parameters and computational complexity. While approaches like Mixture of Experts...

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1339723699187613788)** (11 messages🔥): 

> `Eluther AI Research Contributions, Machine Learning and CS Projects, Identifying People in an Image` 


- **Curiosity Sparks Questions about Research Contributions**: A new member expressed interest in contributing to research projects at **Eluther AI** and inquired about any open projects or guidance available.
   - They seek direction on how to get involved in the community effectively.
- **Students Exploring New Avenues in AI**: Several users, including a **NLP student** and an **engineering student**, are considering transitions into areas like interpretability and deep learning projects.
   - They shared their backgrounds and are eager for insights on how to start contributing.
- **Recognizing Faces from a Shared Image**: A user requested help identifying people in an image, leading to a discussion where names were matched to faces, including **Francois Chollet** and **Gary Marcus**.
   - The conversation highlighted community knowledge as responses emerged quickly, showcasing strong camaraderie.
- **Community Collaboration in Image Recognition**: Another user shared a second, updated image asking for identifications, prompting further responses and shared Google searches in pursuit of accuracy.
   - Community members collaborated efficiently, with one even tagging a comprehensive list of names linked to the image.



**Link mentioned**: <a href="https://x.com/IgorBrigadir/status/1534969529457070100">Tweet from ⚠️ Igor Brigadir 🇺🇦 (@IgorBrigadir)</a>: Everyone in this chart:Top Left: @fchollet @raphaelmilliere @GaryMarcus @tyrell_turing @ylecun @rohinmshahTop Right: @sama @soniajoseph_ @ID_AA_Carmack @tszzl @demishassabis @michael_nielsen @sea_snel...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1339644371208900711)** (208 messages🔥🔥): 

> `Attention Mechanisms, Scaling Laws in LLMs, Hybrid Architectures in Transformers, Forgetting Transformer, Long-Context Performance` 


- **Debate on Attention Sinks and QK Norm**: Discussions revealed that QK Norm may hinder attention sinks which are essential for model performance, although it stabilizes training; value residuals were proposed as a possible mitigation. The group agreed to further investigate these relationships and their implications on model behavior.
   - The necessity of attention sinks versus training stability was debated, highlighting the benefits of forgetting transformers as potential solutions to maintain flexibility in attention mechanisms.
- **Performance of Different Training Paradigms**: Two papers were introduced discussing the advantages of hyperfitting and repeated training examples for large language models (LLMs), suggesting that repetition can enhance performance compared to data diversity. These insights emphasize the complexity of balancing memorization and generalization in deep learning.
   - The conversation examined how models perform better when trained on smaller, repeated examples rather than larger datasets, raising questions about the impact of training methods on LLM capabilities.
- **New Framework for Language Model Pre-Training**: A recent study offered a comprehensive framework that distinguishes between bidirectional context and attention, which addresses previous challenges in model comparisons and evaluations. The findings suggest that the optimal use of bidirectionality is highly application-dependent.
   - The research points to the importance of flexible training configurations, which can impact next token predictions and text infilling differently, emphasizing the need for tailored approaches in model training.
- **Long-Context LLMs and Reasoning Complexity**: A new benchmark, GSM-Infinite, was developed to assess how LLMs handle reasoning complexity across varying context lengths and difficulties, revealing a sigmoid decline in performance with increasing complexity. This highlights the challenges LLMs face when tackling intellectual problems requiring extensive reasoning.
   - The discussion acknowledged the necessity for quantitative evaluations to understand LLM capabilities in long-document reasoning, enhancing insights into their strengths and limitations.
- **Scaling Laws for Transformers**: A query about scaling laws for individual components within transformer architectures was raised, particularly concerning the relationship between the number of heads and the width of the residual stream. The conversation highlighted the trade-offs in designing efficient models and emphasized the importance of optimizing architecture configurations.
   - The potential for using larger head sizes instead of increasing the number of heads was discussed, noting practical constraints within existing kernel frameworks, which may limit efficient implementation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.05483v1">Just read twice: closing the recall gap for recurrent language models</a>: Recurrent large language models that compete with Transformers in language modeling perplexity are emerging at a rapid rate (e.g., Mamba, RWKV). Excitingly, these architectures use a constant amount o...</li><li><a href="https://arxiv.org/abs/2501.18795">Rope to Nope and Back Again: A New Hybrid Attention Strategy</a>: Long-context large language models (LLMs) have achieved remarkable advancements, driven by techniques like Rotary Position Embedding (RoPE) (Su et al., 2023) and its extensions (Chen et al., 2023; Liu...</li><li><a href="https://arxiv.org/abs/2410.24159">GPT or BERT: why not both?</a>: We present a simple way to merge masked language modeling with causal language modeling. This hybrid training objective results in a model that combines the strengths of both modeling paradigms within...</li><li><a href="https://arxiv.org/abs/2406.12335">Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters</a>: Scaling the context size of large language models (LLMs) enables them to perform various new tasks, e.g., book summarization. However, the memory cost of the Key and Value (KV) cache in attention sign...</li><li><a href="https://arxiv.org/abs/2410.13835">Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs</a>: Practitioners have consistently observed three puzzling phenomena in transformer-based large language models (LLMs): attention sinks, value-state drains, and residual-state peaks, collectively referre...</li><li><a href="https://arxiv.org/abs/2502.07490">Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More</a>: Large Language Models (LLMs) are discovered to suffer from accurately retrieving key information. To address this, we propose Mask-Enhanced Autoregressive Prediction (MEAP), a simple yet effective tra...</li><li><a href="https://arxiv.org/abs/2502.06268">Spectral-factorized Positive-definite Curvature Learning for NN Training</a>: Many training methods, such as Adam(W) and Shampoo, learn a positive-definite curvature matrix and apply an inverse root before preconditioning. Recently, non-diagonal training methods, such as Shampo...</li><li><a href="https://arxiv.org/abs/2502.07529">Training Deep Learning Models with Norm-Constrained LMOs</a>: In this work, we study optimization methods that leverage the linear minimization oracle (LMO) over a norm-ball. We propose a new stochastic family of algorithms that uses the LMO to adapt to the geom...</li><li><a href="https://aclanthology.org/2022.findings-emnlp.293/">On the Role of Bidirectionality in Language Model Pre-Training</a>: Mikel Artetxe, Jingfei Du, Naman Goyal, Luke Zettlemoyer, Veselin Stoyanov. Findings of the Association for Computational Linguistics: EMNLP 2022. 2022.</li><li><a href="https://x.com/InfiniAILab/status/1890469309253841191?t=062o_qhbW0XRcux_v_GZ8w&s=19">Tweet from Infini-AI-Lab (@InfiniAILab)</a>: 🐭🐷 GSM-Infinite is generated from the problem generator. 18 strong LLMs were evaluated on the zero noise benchmark and 10 LLMs on the long-context benchmark. 🚀Key Takeaways: 🥕 Recent reasoning mod...</li><li><a href="https://arxiv.org/abs/2410.07041">Emergent properties with repeated examples</a>: We study the performance of transformers as a function of the number of repetitions of training examples with algorithmically generated datasets. On three problems of mathematics: the greatest common ...</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>: This paper introduces the counter-intuitive generalization results of overfitting pre-trained large language models (LLMs) on very small datasets. In the setting of open-ended text generation, it is w...</li><li><a href="https://arxiv.org/abs/2502.05252">GSM-Infinite: How Do Your LLMs Behave over Infinitely Increasing Context Length and Reasoning Complexity?</a>: Long-context large language models (LLMs) have recently shown strong performance in information retrieval and long-document QA. However, to tackle the most challenging intellectual problems, LLMs must...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1339836434726518806)** (3 messages): 

> `OpenAI Deep Research, ML/AI Literature Reviews, Research Grounding Issues` 


- **Discussion on OpenAI Deep Research**: A member inquired if anyone tried **OpenAI Deep Research** and its effectiveness for **ML/AI literature reviews**.
   - Another member responded, stating that it is *excellent* but expressed challenges in grounding its research to arXiv content and specific papers.
- **Concerns about research quality**: A participant remarked that the quality doesn't seem 'excellent', indicating skepticism about the utility of the tool.
   - The feedback highlights potential issues with reliance on **less reliable blogs** instead of credible academic sources.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1340009299337941054)** (8 messages🔥): 

> `Profiling talk recording, Zoom session feedback, YouTube stream` 


- **Latest Profiling Talk will be Recorded**: Members confirmed that the latest talk on **profiling** will indeed be recorded and streamed live on [YouTube](https://youtube.com).
   - The talk is currently happening on **Zoom** and will be available for later viewing.
- **Excitement for the Profiling Talk**: One member expressed eagerness to watch the recording after it finishes, asking for a link once it's done.
   - It was reassured that the session will be posted on various channels including [YouTube](https://youtube.com) and a specific Discord channel.
- **New Member Appreciation for Profiling Insights**: A new member thanked the community for the ongoing **Zoom** session, expressing gratitude for making such knowledge available.
   - They shared that the analysis has significantly enhanced their understanding of profiling and its functions.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1339663553405386753)** (11 messages🔥): 

> `Fused MM Activation Implementation, GEMM Performance Insights, Kernel Caching Strategies, Triton Conference 2025, CUDA Thread Inquiry` 


- **Fused MM Activation for Non-Square Matrices**: A user is working on implementing **Fused MM activation** in Triton for non-square matrices with dimensions M=2500, N=512, and K=512, and is seeking guidance on the fastest tiled MM kernel.
   - They mentioned that the [MM tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) is competitive with **cutlass implementations** and expressed the need for effective autotuning strategies.
- **GEMM Configuration Recommendations**: A member suggested that an **A8W8 (persistent) GEMM** implementation would be the fastest option for larger matrix sizes, particularly when M is greater than or equal to **128**.
   - They emphasized the importance of maximizing aut tuning settings tailored to the specific hardware in use.
- **Caching Kernels for Optimization Algorithms**: The user requires a caching mechanism for Triton kernels due to their **IDR optimization algorithm** having a non-fixed M size, which affects performance.
   - A recommendation was made to apply heuristics based on the next power of 2 to minimize frequent autotuning for different shapes.
- **Inquiries about Triton Conference 2025**: One member questioned whether there will be a **Triton Conference** scheduled for **2025**.
   - The conversation did not yield any confirmation about the event at this time.
- **CUDA Thread's Presence in Discussion**: A member inquired about the presence of a user associated with a **CUDA thread**, seeking clarification on their current involvement.
   - Another member expressed uncertainty regarding the specific reference made about the CUDA thread.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1339651158733750313)** (19 messages🔥): 

> `Tensor Memory Management, GPU Access Issues, Torch Distributed Training Errors, CUDA Technology Relevance` 


- **Exploring Tensor Memory Management for MatMuls**: Discussion highlighted how tensor memory is utilized differently for matrix multiplications, suggesting an accumulator in tensor memory can free up registers for other operations.
   - Participants noted potential inefficiencies in the available tensor memory due to its non-power of two sizes.
- **Struggles with GB200 GPU Availability**: A user expressed frustration over lack of access to GB200 GPUs, emphasizing their willingness to pay for a solution but finding it currently unattainable.
   - Others shared alternative provider suggestions, commenting on the heightened demand for LLM inference, but faced with waitlists.
- **Challenges in Torch Distributed Training**: A participant reported difficulties in getting torch distributed training to work on Blackwell GPUs, encountering errors related to NCCL.
   - Despite using the latest builds, they experienced persistent CUDA errors, indicating a possibly unresolved compatibility issue.
- **Clarifications on Thread Computation in Matrix Operations**: A technical breakdown of how threads compute entries in tensor tiles provided clarity on the mapping of thread-to-computation assignments in a matrix operation.
   - The explanation highlighted the correlation between the sizes of tensors and how each thread retrieves necessary data for computation.
- **Debate Over CUDA's Relevance**: A user questioned whether CUDA is outdated, citing a professor's opinion but receiving confirmation from others about CUDA's current and future relevance.
   - Participants defended CUDA's significance and sought to understand what alternatives might be proposed by skeptics.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://link.excalidraw.com/readonly/heHY09sqiATU9DEpTU8J">Untitled scene - Excalidraw+</a>: Check out Untitled scene on Excalidraw+</li><li><a href="https://x.com/lambdaapi/status/1890028876954489125?s=46">Tweet from Lambda (@LambdaAPI)</a>: All we know is we&#39;re good for our NVIDIA HGX B200s 🙂
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1339660835638874284)** (2 messages): 

> `Fast Hadamard Transform, SageAttention2, Huggingface Transformers ONNX issue` 


- **Fast Hadamard Transform's Role in Quantized Attention**: Discussion on why some **quantized attention** methods require the use of [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform) to achieve usable results while others like **SageAttention** do not.
   - *SageAttention2* proposes accelerating the attention process with quantization techniques that significantly enhance efficiency.
- **Huggingface Transformers ONNX Conversion Issue**: A member encountered an issue with the **Huggingface Transformers** during ONNX conversion due to **Python jit tracing** a boolean value in the `DacResidualUnit` class.
   - They suggested a workaround using explicit slicing instead of conditional checks to ensure compatibility, seeking a review before submitting a PR.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Dao-AILab/fast-hadamard-transform">GitHub - Dao-AILab/fast-hadamard-transform: Fast Hadamard transform in CUDA, with a PyTorch interface</a>: Fast Hadamard transform in CUDA, with a PyTorch interface - Dao-AILab/fast-hadamard-transform</li><li><a href="https://arxiv.org/abs/2411.10958">SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization</a>: Although quantization for linear layers has been widely used, its application to accelerate the attention process remains limited. To further enhance the efficiency of attention computation compared t...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1340008465728409700)** (1 messages): 

> `NVIDIA Profiling Tools, Magnus Strengert Talk` 


- **NVIDIA Talk on Profiling Tools Begins Soon**: In **45 minutes**, the Chief Architect for NVIDIA's profiling tools, **Magnus Strengert**, will present a talk on all things profiling-related, promising valuable insights directly from the source.
   - Members are encouraged to attend this meeting on [Zoom](https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG) as it has been noted that there is little public content available on this topic.
- **Exciting Opportunity to Learn Directly**: A recent trend shows that some of the most well-received talks on the server revolve around **profiling**, making this session particularly noteworthy.
   - Attendees are likely to gain insights that are not commonly found in available public resources.



**Link mentioned**: <a href="https://illinois.zoom.us/j/85178224887?pwd=lHLSoQ0DtlbhKtAbFcGOD9crhKRqWG.1">Join our Cloud HD Video Meeting</a>: Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...

  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1340049159369134202)** (1 messages): 

> `Roofline Model, Hierarchical Analysis` 


- **Discover Roofline Model Hierarchy**: A member shared a [good resource](https://crd.lbl.gov/assets/Uploads/cug19-roofline-final.pdf) on the roofline model, emphasizing its hierarchical structure.
   - This document provides insights into performance analysis and optimization strategies for computing resources.
- **Understanding the Importance of Roofline**: The roofline model illustrates the trade-offs between compute performance and memory bandwidth, crucial for system optimization.
   - As noted in the shared document, *it acts as a guide for developers aiming to maximize computational efficiency*.


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1339808149326270475)** (1 messages): 

> `Oumi AI, Open-source models, ML performance engineers hiring, Collaborative AI development` 


- **Oumi AI advocates for open-source development**: Oussama, co-founder at [Oumi](https://oumi.ai), shared that their startup is focused on building fully open models and infrastructure, promoting the belief that **open-source lifts all boats**.
   - He emphasized the importance of collaboratively developing AI in the open for broader benefits.
- **Hiring call for ML performance engineers at Oumi**: Oumi is actively hiring [ML performance engineers](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393) to enhance their model speed and training pipelines.
   - Candidates will have the opportunity to contribute to multiple [open projects](https://github.com/oumi-ai/oumi) and collaborate with a dedicated team of researchers.
- **Invitation to connect for potential job inquiries**: Oussama encouraged interested individuals to [apply directly](https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393) or to reach out via [DM](https://x.com/oussama_e) or [LinkedIn](https://www.linkedin.com/in/oussamaelachqar/) for questions.
   - He expressed openness to discussions, facilitating connections for aspiring candidates.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://oumi.ai).">no title found</a>: no description found</li><li><a href="https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393))">Jobs</a>: no description found</li><li><a href="https://jobs.ashbyhq.com/oumi/6150a078-73c0-4385-96d0-02e953d01393),">Jobs</a>: no description found</li><li><a href="https://x.com/oussama_e)">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1339955572124024943)** (2 messages): 

> `Fine-tuning Transformer Models, Colab GPU Issues, Alternatives to Colab for Training, Modal Platform Discussion` 


- **User seeks alternatives for fine-tuning**: A member expressed frustration with **Colab GPUs**, noting issues with **stability** and **code errors** after trying even the pro version.
   - *Where else can I train my transformer models?*
- **Modal praised as training platform**: Another member recommended using **Modal** as a preferred platform for model fine-tuning.
   - They emphasized their support for this option over Colab amidst ongoing frustrations.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 messages): 

mubappe.: Yes resolved, thanks
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1339684519560613920)** (8 messages🔥): 

> `Llama 3.3 License Issues, Llama Model Availability, Documentation and Code Sharing` 


- **Challenges in Obtaining Llama 3.3 License**: A user is facing difficulties registering for a license for **Llama 3.3 70B** base and instruct models, encountering a message stating they do not meet the criteria for a license.
   - They expressed urgency in resolving this issue to conduct experiments for a research cohort in the **Cohere For AI** Discord.
- **Alternative Access to Llama Models**: Another user suggested accessing the **70B-Instruct version** from Hugging Face as an alternative to the licensing issue, providing a [direct link](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) for ease.
   - They noted that there seems to be no available 'base' version on the platform.
- **Documentation Concerns within the Community**: A community member acknowledged their struggles with documenting their code and empathized with others in similar situations, indicating a generally supportive environment.
   - This reflection on documentation suggests a collective understanding of the challenges faced by many developers.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: no description found</li><li><a href="https://www.llama.com/llama-downloads/.">Download Llama</a>: Request access to Llama.</li><li><a href="https://x.com/cataluna84/status/1881908149449547860">Tweet from Mayank Bhaskar (@cataluna84)</a>: Hello to folks at Meta, I am unable to obtain a license for Llama 3.3 70B base and instruct models. This is the error message I receive:&#34;Thank you for your interest in using Llama. Unfortunately, ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1339698097592471636)** (1 messages): 

> `User Defined Kernels, FSDP Usage` 


- **Clarification on User Defined Kernels**: A member pointed out that there shouldn't be many issues with **user defined kernels**, questioning the specifics of the problem encountered.
   - They also sought clarification on whether **FSDP 1 or 2** was being used in the context of the issue.
- **Questioning FSDP Version**: Another member inquired about the **FSDP version** in use, emphasizing its importance for resolving potential kernel issues.
   - This suggests a need for alignment on which version could lead to better performance in the current setup.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1339659282496356496)** (19 messages🔥): 

> `CUDA Kernel Optimizations, Low Bit Training Presentation, FP8 Training, Cohere AI YouTube Webinar, Polish LLM Training Pipeline` 


- **High-Level CUDA with Performance Limitations**: Optimizations like **loop unrolling**, tiled transposed **matrix B**, and **warp level reductions** were implemented in a CUDA kernel, achieving only **1/3rd performance of PyTorch** without using cuBLAS.
   - A member noted that there seems to be a limit to optimizing CUDA kernels beyond a certain point, which sparked a discussion on the performance aspects and potential optimization strategies.
- ****C4AI Presentation on Low Bit Training****: A member announced a presentation on **low bit training** happening live, inviting others to join the discussion via a [Google Meet link](https://meet.google.com/wdk-yipf-zjd?authuser=0).
   - Another member humorously noted that this event was happening immediately, prompting a light-hearted exchange among participants.
- ****FP8 Training Topics****: A member expressed interest in **FP8 training** topics, especially concerning optimizing their Polish LLM training pipeline, which currently requires significant GPU resources.
   - Suggestions for potential talks around FP8 training arose, highlighting a need for community knowledge sharing on learning local optimizations.
- **Cohere AI's YouTube Resource**: A member mentioned that the recent webinar will be available on their [YouTube channel](https://youtube.com/@cohereai?si=EhPxTDUf4OZPJEkm) and shared additional links for related talks.
   - This prompted a member to request insights on **FP8 training**, indicating a growing interest in optimizing large models for training efficiency.
- **Discussion on Training Optimization Techniques**: There was curiosity about whether **optimizer states** were offloaded to the CPU for **GH200 training**, with implications for optimizing training workflows.
   - A member shared their eagerness to discuss exciting features regarding  **float8** training and support technologies, indicating readiness for further collaboration.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtube.com/@cohereai?si=EhPxTDUf4OZPJEkm),">Cohere</a>: Welcome to NLP, now. We’re kickstarting a new chapter in machine learning by giving developers and businesses access to NLP powered by the latest generation of large language models, now. Our platform...</li><li><a href="https://github.com/prateekshukla1108/100-daysofcuda/tree/main/day18">100-daysofcuda/day18 at main · prateekshukla1108/100-daysofcuda</a>: Kernels Written for 100 days of CUDA Challenge. Contribute to prateekshukla1108/100-daysofcuda development by creating an account on GitHub.</li><li><a href="https://ppc.cs.aalto.fi/ch4/v2/">Chapter 4: V2</a>: no description found
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[avx](https://discord.com/channels/1189498204333543425/1291829797563011227/)** (1 messages): 

alint5215: it beats openblas now.
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1339859145771188304)** (5 messages): 

> `Inference-time scaling, DeepSeek-R1 model, CuDNN frontend for flex attention, NVIDIA's performance benchmarks` 


- **NVIDIA's Inference-time Scaling for Problem Solving**: NVIDIA introduced a new scaling law called _inference-time scaling_, which allows AI models to allocate additional resources during inference to evaluate outcomes and select the best one.
   - This technique aims to enhance model strategies, enabling them to tackle complex problems similar to human problem-solving methods.
- **DeepSeek-R1 Model Experiment**: An experiment using NVIDIA's DeepSeek-R1 model demonstrated its ability to automatically generate numerically correct GPU attention kernels optimized for specific tasks.
   - However, the distinction between functionally correct and fast kernels is crucial for meaningful performance benchmarks.
- **Concerns over CuDNN Flex Attention Relation**: A user questioned whether NVIDIA's findings relate to the CuDNN frontend for flex attention due to its high-level abstraction.
   - Concerns were raised about the potential lack of interesting results from the generation process, suggesting confirmation is needed.
- **Functionality vs Speed in Kernel Production**: A co-author noted that merely producing functionally correct kernels does not imply they meet performance benchmarks, as speed is a critical metric.
   - While functional correctness is achievable because of existing reference kernels, the real goal is for model-generated kernels to outperform current implementations.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.00005">Mastering the Craft of Data Synthesis for CodeLLMs</a>: Large language models (LLMs) have shown impressive performance in \emph{code} understanding and generation, making coding tasks a key focus for researchers due to their practical applications and valu...</li><li><a href="https://arxiv.org/abs/2207.14502">Language Models Can Teach Themselves to Program Better</a>: Recent Language Models (LMs) achieve breakthrough performance in code generation when trained on human-authored problems, even solving some competitive-programming problems. Self-play has proven usefu...</li><li><a href="https://arxiv.org/html/2402.15769v1">Importance Guided Data Augmentation for Neural-Based Code Understanding</a>: no description found</li><li><a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">Automating GPU Kernel Generation with DeepSeek&#x2d;R1 and Inference Time Scaling | NVIDIA Technical Blog</a>: As AI models extend their capabilities to solve more sophisticated challenges, a new scaling law known as test&#x2d;time scaling or inference&#x2d;time scaling is emerging. Also known as AI reasoning ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1339642887360483329)** (114 messages🔥🔥): 

> `Futoshiki dataset updates, Eval architecture discussions, Whitespace in answers, Scoring methods, Evaluation process improvements` 


- **Complexity of Futoshiki dataset**: The Futoshiki dataset is proving to be more complex than initially anticipated, with members acknowledging the challenges in generating quick solutions.
   - One member plans to clarify question formatting to enhance scoring effectiveness after testing across models.
- **Eval architecture improvements**: Discussion around moving all evaluation-related code to a separate repository to maintain the reasoning-gym's focus, with plans to clean up the current structure.
   - There is a consensus to standardize scoring strategies and discuss answer formatting to reduce inconsistencies in outputs.
- **Handling whitespace in answers**: Concerns were raised about how leading and trailing whitespace could affect answer scoring, suggesting changes to the extract answer function to handle these cases more effectively.
   - Suggestions included using regex patterns to accurately parse answers, coupled with a reevaluation of how answers are scored.
- **Scoring methods and improvements**: It was suggested to implement a score_answer method for datasets to ensure consistent evaluation, especially for datasets lacking clarity in answer formatting.
   - A member expressed the intention to operate on existing evaluation scripts to enhance their functionality.
- **Evaluation process follow-ups**: Members are actively updating evaluation outputs and documenting changes in a shared Google Sheet, ensuring a structured approach to evaluations.
   - One member committed to resuming evaluations and coordinating tasks related to the uninformative prompts based on established checklists.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.google.com/spreadsheets/d/1qk2BgxzfRZzTzMQnclCr47ioykgltbGkMJUHO2sH6Gw/edit?gid=1879785298#gid=1879785298">reasoning-gym-eval</a>: no description found</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/113">Rush Hour Gym by Iron-Bound · Pull Request #113 · open-thought/reasoning-gym</a>: Adds a gym environment for the puzzle game Rush Hour.</li><li><a href="https://github.com/open-thought/reasoning-gym-eval/">GitHub - open-thought/reasoning-gym-eval: Collection of LLM completions for reasoning-gym task datasets</a>: Collection of LLM completions for reasoning-gym task datasets - open-thought/reasoning-gym-eval</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/5d16a2193c142c827c11d3a7c85565849f779c33/eval/eval.py#L42-L59">reasoning-gym/eval/eval.py at 5d16a2193c142c827c11d3a7c85565849f779c33 · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/60">Add Futoshiki puzzle generator by olliestanley · Pull Request #60 · open-thought/reasoning-gym</a>: Closes #54Existing solvers were messy &amp;amp; difficult to follow so I ended up implementing a new one. The logical rules are not easy to follow even in this code, but are well worth it as they spee...</li><li><a href="https://github.com/Adefioye/AI-Playground/blob/main/eval/eval.py">AI-Playground/eval/eval.py at main · Adefioye/AI-Playground</a>: Contribute to Adefioye/AI-Playground development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/128">[Ongoing] Eval Template and Score Fixes by Miserlou · Pull Request #128 · open-thought/reasoning-gym</a>: Ongoing work to fix up evaluators and ambiguous questions.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/129">Remove leading/trailing whitespace from scored answers by default by olliestanley · Pull Request #129 · open-thought/reasoning-gym</a>: In default scoring, for some datasets, many answers receive a score of 0.5 due to the LLM generating leading/trailing whitespace, when they would otherwise receive 1.0. This whitespace is usually i...</li><li><a href="https://github.com/Adefioye/AI-Playground/blob/bd9e70706cda7106778f87b6c5bb5dbb5e851836/eval/eval.py#L48-L51">AI-Playground/eval/eval.py at bd9e70706cda7106778f87b6c5bb5dbb5e851836 · Adefioye/AI-Playground</a>: Contribute to Adefioye/AI-Playground development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/124">Add useful instructions to question template of some datasets by Adefioye · Pull Request #124 · open-thought/reasoning-gym</a>: Here, there are a couple of instructions added to the question templates of some datasets.</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/eval/r1/eval.py">reasoning-gym/eval/r1/eval.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/eval/eval.py">reasoning-gym/eval/eval.py at main · open-thought/reasoning-gym</a>: procedural reasoning datasets. Contribute to open-thought/reasoning-gym development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1339695111239634944)** (13 messages🔥): 

> `API usage field update, Tokenization across models, Provider outages, OpenAI model availability, Model suffixes and functionalities` 


- **API Usage Field Update Considerations**: A proposed change to the `usage` field in the API considers switching from a **normalized token count** to the **model's native token count** due to advancements in tokenization across models.
   - Users expressed concerns about whether this will affect model rankings, to which it was confirmed that **the GPT tokenizer will still be used for rankings**.
- **Tokenization Debate Among Providers**: Discussion arose about whether the **vertex** model still operates with a higher ratio of **tokens per character**, with a user suggesting keeping the **GPT tokenizer** for consistency within the aggregator platform.
   - Clarification was provided that while vertex has a slightly different ratio, it is not as extreme as those seen in past models like **PaLM**.
- **Provider Outage Notification**: A brief notification indicated that the **Fireworks provider** was experiencing an outage, but noted that other providers and **BYOK usage** were unaffected.
   - An update stated that the outage was resolved by **9:12 ET**, confirming normal operations resumed.
- **OpenRouter Model Availability Update**: OpenAI's **o1 and o3 models** are now available to all OpenRouter users, eliminating the need for a separate BYOK key and allowing higher rate limits.
   - The announcement included a **cheatsheet for model suffixes**, indicating options like `:online`, `:nitro`, and `:floor` for different functionalities.
- **Valuation of Usage Reporting**: Concerns were raised about the accuracy of **total usage costs** reported by the **roo code**, suggesting it may not align with expectations.
   - Another user pointed out queries about which providers do not report an **usage object**, seeking clarity on their operational practices.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1890419427859935275">Tweet from OpenRouter (@OpenRouterAI)</a>: Reminder that you can append `:online` to add web access to any model, including o1 and o3-mini 🌐Along with `:nitro` (fastest), `:floor` (cheapest), and `:free` (for 30 models)Quoting OpenRouter (@Op...</li><li><a href="https://openrouter.ai/docs/api-reference/overview#response-format">OpenRouter API Reference - Complete Documentation</a>: Comprehensive guide to OpenRouter&#x27;s API. Learn about request/response schemas, authentication, parameters, and integration with multiple AI model providers.
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1339645961659744426)** (163 messages🔥🔥): 

> `DeepSeek R1 Performance, Error Issues with API Keys, Self-Moderated OpenAI Endpoints, New Model Introductions, Rate Limiting Concerns` 


- **DeepSeek R1 struggles for some users**: Users report that while using **DeepSeek R1** on OpenRouter, the service regularly pauses, causing issues for their agents and expressing concerns about its reliability in production.
   - Some compare its performance to other models, stating they find its reasoning superior under certain settings, including a recommended temperature of 0.6 without a system prompt.
- **API Key Issues and Strikethrough Indicators**: A user discovered that their API keys showed a strikethrough on the website and returned a 401 error, leading to confusion regarding their validity and usage.
   - Admins indicated that keys may be disabled due to potential leaks detected by their system, emphasizing the importance of using secrets.
- **Interest in Self-Moderated Endpoints**: There was discussion about self-moderated OpenAI endpoints, with users expressing eagerness for lower latency and more consistent response outputs similar to Anthropic's approach.
   - Admins indicated they are working towards implementing such features based on community feedback.
- **Rate Limiting and Model Parameters**: Users inquired about the rate limits for models like **Gemini 2.0 Pro**, revealing insights about daily request limits based on model variants and user credits.
   - There were also discussions about inconsistencies in performance from different providers, with comparisons to expected parameter settings for optimal results.
- **Feedback on New Models and Providers**: Participants exchanged thoughts on new models like **Sambanova**, exploring their pricing structure and user experiences with the quality of responses compared to more established systems.
   - Users noted varied results depending on the platform used, leading to discussions about the transparency of underlying prompts and behavioral adjustments of models like Claude.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1890324295181824107">Tweet from DeepSeek (@deepseek_ai)</a>: 🎉 Excited to see everyone’s enthusiasm for deploying DeepSeek-R1! Here are our recommended settings for the best experience:• No system prompt• Temperature: 0.6• Official prompts for search & file up...</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: A unified interface for LLMs. Find the best models &amp; prices for your prompts</li><li><a href="https://openrouter.ai/docs/quickstart">OpenRouter Quickstart Guide</a>: Get started with OpenRouter&#x27;s unified API for hundreds of AI models. Learn how to integrate using OpenAI SDK, direct API calls, or third-party frameworks.</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - Manage Model Usage and Quotas</a>: Learn about OpenRouter&#x27;s API rate limits, credit-based quotas, and DDoS protection. Configure and monitor your model usage limits effectively.</li><li><a href="https://cloud.sambanova.ai/">SambaNova Cloud</a>: Preview AI-enabled Fastest Inference APIs in the world.</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429">no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1339667904467042325)** (122 messages🔥🔥): 

> `Perplexity Deep Research, AI Model Opinions, Use of Wolfram Alpha, ChatGPT User Experience, AI News Sources` 


- **Perplexity Deep Research Feature**: Users excitedly discussed the new 'Deep Research' feature released by Perplexity, with some already accessing it on the free tier.
   - Curiosity arose around the usage limits of this feature, with questions about whether it was per week or based on some other criteria.
- **Diverse Opinions on AI Models**: Members expressed varying opinions on the effectiveness of different AI models, with some feeling that ChatGPT has 'dumbed down' over time compared to earlier versions.
   - There was mention of a sentiment that newer models often prioritize tone and directness over logical reasoning.
- **Wolfram Alpha as a Tool**: Discussion arose about integrating Wolfram Alpha into LLM systems for enhanced calculation capabilities, with several members advocating for its use.
   - It was noted that many find it valuable for providing accurate mathematical answers quickly through an API.
- **Exploration of AI News Sources**: Several users highlighted Perplexity as a preferred news source due to its low bias and interactive features like follow-up suggestions.
   - This interest in alternatives was fueled by a desire for undistorted news consumption compared to traditional sources.
- **Future of AI Tools and Features**: There was speculation about potential integrations, such as OpenAI acquiring Perplexity and rebranding it, indicating strong user interest in evolving AI tools.
   - Conversations often highlighted the limitations hindering user experience in existing AI platforms, fostering discussions on enhancements.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1339662434666942545)** (10 messages🔥): 

> `Free Plan Limits, GPT Store Publishing, Privacy Policy Requirement` 


- **Free Plan Limits are Uncertain**: A member inquired about how to verify the limits of the free plan for various models, asking if they could find fixed values for messages and text sent back.
   - Another member noted that limits are variable and change daily based on different factors, suggesting that only some specifics, like **AVM's** **15 min/month**, have fixed values.
- **Need Help with GPT Store Publishing Error**: A member reported an error while trying to publish to the GPT Store, stating they received a message about needing valid privacy policy URLs.
   - Another member advised updating the privacy policy field in actions, which the original member confirmed solved the issue after they wrote a privacy policy.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1339662979100311704)** (10 messages🔥): 

> `Using ChatGPT vs Playground, Interpretation of prompts, JSON vs plain text formats, Legislative writing with AI, Importance of human oversight` 


- **Differences Between ChatGPT and Playground**: The discussion highlighted that using ChatGPT differs from the Playground due to the models' handling of prompts and error management strategies.
   - It was suggested that focusing on identifying and correcting errors is crucial for effective interaction with the model.
- **Interpreting Prompts for Clarity**: Members emphasized that asking the model to contrast interpretations of a prompt can lead to identifying conflicts or ambiguities.
   - Using clear and normal language rather than strict formats can yield more useful responses from the AI.
- **Choosing Between Formats: JSON or Text**: A member argued for using simple text or YAML for direct AI interaction due to better readability and efficiency, while JSON is recommended for APIs.
   - Maintaining clarity is deemed essential regardless of the format chosen for conveying instructions.
- **AI-Assisted Legislative Writing**: While no one has created prompts specifically for legislative writing, a careful human should oversee any AI-generated output for important material.
   - Members stressed that a skilled human must validate and critique all model outputs to ensure safety and accuracy.
- **Human Oversight in AI Use**: Discussion reinforced the idea that human oversight is critical in any AI-assisted process, likening it to driving with support systems.
   - Taking complete responsibility for the output is vital, even when leveraging AI assistance for crafting material.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1339662979100311704)** (10 messages🔥): 

> `Using ChatGPT vs Playground, Interpreting prompts, Prompt formats, Legislative writing prompts, AI model confidence` 


- **ChatGPT and Playground Differences**: A member discussed the nuances between using **ChatGPT** and the **Playground**, emphasizing the importance of pulling out errors and recognizing patterns in responses.
   - *Look over many myself; look over with the model what patterns with wrong.*
- **Interpret prompts for clarity**: One member highlighted the value of prompt interpretation, stating that it can reveal conflicts and ambiguity, leading to useful insights.
   - *What does it mean to you, are there any conflicts or ambiguities?*
- **Preference in Prompt Format**: A member suggested that for direct AI interaction, **YAML** or plain text is preferable for readability, while **JSON** is better for APIs due to its strict structure.
   - This prompted a discussion on the best format to use for system prompts based on user familiarity and efficiency.
- **Cautions in Legislative Writing with AI**: A member expressed the need for careful human oversight when using AI for legislative writing, emphasizing that a skilled individual must review all output.
   - They cautioned that while AI assistance can improve ideas, maintaining responsibility and verifying information is crucial.
- **Building Confidence in AI Outputs**: Another member shared insights on ensuring the AI model understands user prompts clearly by using normal language rules to avoid confusion.
   - Being free of errors enables the model to clearly predict user intentions, enhancing its reliability.


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1339644844867457206)** (135 messages🔥🔥): 

> `Stable Diffusion Models, Lora Training Tips, Audio Device Recognition, Controlled Image Generation, Community Engagement` 


- **Guidance on Lora Training for Stable Diffusion**: A user shared their experience training a Lora with only 7 selfies, leading to limited likeness recognition, especially for side views.
   - Advice included using a larger dataset of high-quality images and ensuring they match the desired output style, as smaller models might generalize less effectively.
- **Exploration of AI Image Generation**: Members discussed various methods for generating AI art, addressing challenges like achieving consistent character designs across multiple models.
   - Tools like FaceFusion were recommended for face-swapping, while a query about automating image requests sparked discussions on requiring ComfyUI workflows.
- **Stable Diffusion with Control Settings**: A user inquired about fine-tuning Stable Diffusion with control mechanisms for improved image generation, expressing interest in recent tools.
   - Recommendations pointed towards the L3 discord for specific resources and contacts related to controlled image generation projects.
- **Audio Device Detection Quirks**: A member humorously commented on the quirks of Windows detecting audio devices, suggesting an ideal hardware solution could improve detection processes.
   - This led to light banter about technology frustrations, with some mentioning the paradox of being heavily reliant on computing devices despite their flaws.
- **Community Dynamics and Engagement**: New users introduced themselves, sharing their experiences with AI art and seeking advice on challenges faced with AI tools and models.
   - Members welcomed newcomers, showcasing an engaged community atmosphere focused on exchanging knowledge and experiences in AI art generation.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/Dalabad/stable-diffusion-prompt-templates">GitHub - Dalabad/stable-diffusion-prompt-templates</a>: Contribute to Dalabad/stable-diffusion-prompt-templates development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=Si3dMKq_2xM">Donald Trump is a Hippie- part 2</a>: Donald Trump as a hippie enjoying his life with Elon musk and Joe Biden. Made with Ai.
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1339650456527568999)** (53 messages🔥): 

> `Open Weight Definition, DeepHermes-3 Preview, EnigmaEval Launch, AI Security Institute, xAI Data Center Plans` 


- **Open Weight Definition debated**: The Open Weight definition was discussed, emphasizing compliance with criteria for free redistribution of model weights on [Open Weight site](https://openweight.org/). Concerns were raised over its implications, with some members expressing interest in its potential effects on open-source AI practices.
- **DeepHermes-3 Preview introduces advanced reasoning**: The introduction of DeepHermes-3 was announced, showcasing its ability to toggle reasoning capabilities for improved accuracy at the expense of increased computation time, detailed on [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview). Community benchmarks indicate its performance is still being evaluated against other models like Tülu.
- **EnigmaEval raises the bar for AI reasoning**: Dan Hendrycks shared that EnigmaEval released a set of complex reasoning challenges, with top AI systems scoring below 10% and none achieving better than 0% on student-level puzzles, highlighting difficulties identified at [Scale AI](https://twitter.com/scale_AI). Participants spent significant time on the challenge, revealing the low capabilities of current AI systems.
- **UK pivots to AI Security Institute**: The UK government has rebranded its AI Safety Institute to the AI Security Institute, shifting focus towards strengthening cybersecurity against AI-related risks, as reported by TechCrunch. This change raises concerns about a diminished focus on AI safety, as noted by various community members.
- **xAI plans significant data center expansion**: Elon Musk’s startup, xAI, is seeking to establish a new data center to support increasing Nvidia chip usage, as discussed in a recent report on [The Information](https://www.theinformation.com/briefings/musk-looks-for-another-data-center-for-xai-nears-5-billion-chip-deal-with-dell). This expansion signals ambitious growth efforts within the competitive AI landscape.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openweight.org/">Open Weight Definition (OWD)</a>: no description found</li><li><a href="https://opensource.org/ai/open-source-ai-definition">The Open Source AI Definition &#8211; 1.0</a>: version 1.0 Preamble Why we need Open Source Artificial Intelligence (AI) Open Source has demonstrated that massive benefits accrue to everyone after removing the barriers to learning, using, sharing ...</li><li><a href="https://techcrunch.com/2025/02/13/uk-drops-safety-from-its-ai-body-now-called-ai-security-institute-inks-mou-with-anthropic/">UK drops &#039;safety&#039; from its AI body, now called AI Security Institute, inks MOU with Anthropic | TechCrunch</a>: The U.K. government wants to make a hard pivot into boosting its economy and industry with AI, and as part of that, it’s pivoting an institution that it</li><li><a href="https://opensourcealliance.org/">Open Source Alliance</a>: Uniting global Open Source communities to shape the future of software freedom.</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">Tweet from Nous Research (@NousResearch)</a>: Introducing DeepHermes-3 Preview, a new LLM that unifies reasoning and intuitive language model capabilities.https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-PreviewDeepHermes 3 is built fr...</li><li><a href="https://x.com/stalkermustang/status/1890144205038842219">Tweet from Igor Kotenkov (@stalkermustang)</a>: babe wake up AIME 2 LLM results droppedo3-mini is the king, gemini is cooked, R1 is _reasonable_</li><li><a href="https://x.com/janleike/status/1890155264101486792">Tweet from Jan Leike (@janleike)</a>: @caleb_parikh They sent 7,867 messages, and passed 1,408 of them onto the auto-grader. We estimate that they probably spent over 40 hours on this in total.</li><li><a href="https://fxtwitter.com/anissagardizy8/status/1890483681476686177">Tweet from Anissa Gardizy (@anissagardizy8)</a>: new: Elon Musk’s artificial intelligence startup xAI is looking to set up a new data center as it plans to substantially increase the number of Nvidia chips it uses, according to someone involved in t...</li><li><a href="https://x.com/Dorialexander/status/1890122850339811642">Tweet from Alexander Doria (@Dorialexander)</a>: @TheXeophon New org. https://opensourcealliance.org/ Definition was just dropped during the summit and reasonably publicized(for context, seemingly a fork from open source initiative due to disagreeme...</li><li><a href="https://fxtwitter.com/janleike/status/1890141865955278916">Tweet from Jan Leike (@janleike)</a>: Results of our jailbreaking challenge:After 5 days, &gt;300,000 messages, and est. 3,700 collective hours our system got broken. In the end 4 users passed all levels, 1 found a universal jailbreak. We...</li><li><a href="https://fxtwitter.com/DanHendrycks/status/1890091724594393140">Tweet from Dan Hendrycks (@DanHendrycks)</a>: We&#39;re releasing EnigmaEval, a collection of long, complex reasoning challenges that take groups of people many hours or days to solve.The best AI systems score below 10% on normal puzzles, and for...</li><li><a href="https://x.com/markgurman/status/1890239501974622650">Tweet from Mark Gurman (@markgurman)</a>: BREAKING: Apple will work with both Alibaba and Baidu to power its AI in China. Alibaba will modify and censor content on Apple’s on-device models to meet China law. Baidu will power Visual Intelligen...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1339669334880030783)** (8 messages🔥): 

> `notebookLM performance, GPT-5 model interface, reasoning models training` 


- **User Frustration with notebookLM**: *notebookLM absolutely sucks* as it responds quickly but fails to execute tasks like creating a markdown table from various PDFs, making users feel it's outdated.
   - One user expressed a desire to switch to Deep Research with a specific prompt for better results.
- **Concerns Over GPT-5 Interface Changes**: A user expressed concerns about Sama's announcement regarding combining models into one interface for GPT-5, stating they want to distinguish between model types for task delegation.
   - Another user felt they might not be the target audience concerning this change.
- **Understanding Reasoning Models Training**: A member summarized the training of reasoning models, noting that the model is first fine-tuned to produce *thinking* tokens before applying reinforcement learning for task completion.
   - Another participant added that RL attempts multiple problem-solving methods and reinforces successful approaches.


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1339712571506294844)** (2 messages): 

> `DH3 Evaluation Metrics, Distill Release Comparison, Company Legitimacy Discussions` 


- **DH3 Evaluation Metrics under Scrutiny**: There's concern over DH3's evaluation metrics, where *only two specific evals* are shown for the 'reasoning on' section, while the 'reasoning off' chart presents all metrics.
   - This selective reporting raises questions about the transparency of their evaluation process.
- **DH3 vs Official 8b Distill Release**: Critics noted that DH3 fails to directly compare to the official **8b distill release**, which boasts higher scores, such as **49%** on GPQA compared to DH3's **36-37%**.
   - This omission has led to skepticism regarding the validity of the reported results.
- **Uncertainty surrounds Company Authenticity**: Discussions arose around the legitimacy of the company associated with DH3, with mixed feelings expressed by members.
   - Despite skepticism, it was noted that being 'swaggy' may hold more appeal than official validation in the community.



**Link mentioned**: <a href="https://fxtwitter.com/kalomaze/status/1890153665333457140">Tweet from kalomaze (@kalomaze)</a>: dh3 notes1. they only show these two specific evals for the &#34;reasoning on&#34;; the &#34;reasoning off&#34; chart is the only one showing all metrics2. they don&#39;t compare to the official 8b di...

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1339711778359349299)** (41 messages🔥): 

> `Boomer prompts and O-series models, DeepSeek-R1 deployment, Academic writing evolution, David Perrell's writing advice, Tülu 3 presentation at DLCT` 


- **Avoiding Boomer Prompts with O-series Models**: Members discussed the importance of avoiding 'boomer prompts' when using O-series models, as highlighted by @OpenAIDevs. Clean delimiters and direct instructions enhance model output effectiveness.
   - One member humorously mentioned feeling 'attacked' by the implications of this advice.
- **DeepSeek-R1 Deployment Excitement**: Excitement was shared for the deployment of **DeepSeek-R1**, with officials recommending settings such as no system prompt and a temperature of **0.6**. Key links were provided for the best experience and to mitigate bypass issues.
   - Users noted the difference between the official deployment and open-source variants of the model, ensuring a similar experience.
- **Evolution of Academic Writing and LLMs**: Discussion revolved around an arXiv abstract analyzing the increase of certain words in academic writing due to LLM influence, showing adaptability among authors. This highlighted the ongoing coevolution between human writers and AI technologies.
   - Members speculated about the effects on non-native English speakers using LLMs to enhance their writing quality.
- **David Perrell's Writing Insights**: The group shared views on David Perrell's writing advice, with some finding it inspiring while others viewed it as overly simplistic. Members emphasized the importance of personal detail to engage readers and the value of a strong introduction.
   - A recommendation was made to check out Perrell's content while recognizing its imperfect yet fun nature.
- **Tülu 3 Presentation Updates**: The presentation of **Tülu 3** was highlighted, being led by @pdasigi at DLCT, complete with celebratory mentions referencing Valentine's Day. This event showcased new progress and community vibrancy.
   - Members expressed excitement over the presentation, which aligned with a festive spirit.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ml-valentines.github.io/">ml valentines</a>: no description found</li><li><a href="https://fxtwitter.com/deepseek_ai/status/1890324295181824107">Tweet from DeepSeek (@deepseek_ai)</a>: 🎉 Excited to see everyone’s enthusiasm for deploying DeepSeek-R1! Here are our recommended settings for the best experience:• No system prompt• Temperature: 0.6• Official prompts for search & file up...</li><li><a href="https://arxiv.org/abs/2502.09606">Human-LLM Coevolution: Evidence from Academic Writing</a>: With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as &#34;delve&#34;, starting soon aft...</li><li><a href="https://x.com/savvyRL/status/1890465151574254057">Tweet from Rosanne Liu (@savvyRL)</a>: Tülu 3 is being presented at DLCT right now! Led by @pdasigi The @allen_ai color with 🩷 as author superscript are basically shouting &#34;happy valentine&#39;s&#34;</li><li><a href="https://x.com/edwinarbus/status/1890164717148336365">Tweet from edwin (@edwinarbus)</a>: @din0s_ old school, really long, expository prompt that might work well with GPT-series but not needed for reasoning models since those can figure it out themselveshttps://x.com/edwinarbus/status/1890...</li><li><a href="https://x.com/edwinarbus/status/1890149926660678080">Tweet from edwin (@edwinarbus)</a>: basically.&#34;think carefully step by step...&#34;&#34;explain your reasoning to me...&#34;or really long, highly detailed, expository promptsjust not needed for o-series, and may slow you down—trust...</li><li><a href="https://x.com/OpenAIDevs/status/1890147300493914437">Tweet from OpenAI Developers (@OpenAIDevs)</a>: As some of you have noticed, avoid “boomer prompts” with o-series models. Instead, be simple and direct, with specific guidelines.Delimiters (xml tags) will help keep things clean for the model, and i...</li><li><a href="https://x.com/lmarena_ai/status/1890172387070734633">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: We&#39;re excited to announce that @Alibaba_Qwen  Qwen2.5-VL-72B-Instruct is the new #1 OPEN model on the Vision leaderboard!They have jumped over Pixtral, and the gap between Open and Proprietary mod...</li><li><a href="https://github.com/deepseek-ai/DeepSeek-R1/pull/399/files">Add fileupload and websearch prompt by DeepSeekPH · Pull Request #399 · deepseek-ai/DeepSeek-R1</a>: Here are prompts we use in DeepSeek Web and App
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1339978645770600559)** (4 messages): 

> `Alignment discussions, OpenAI O1 Pro mode inquiries` 


- **Alignment à la française**: A member shared a link discussing **alignment** issues with a humorous take, captioning it 'alignment à la française'.
   - The linked content suggests a cultural twist on standard alignment debates within the AI community.
- **Costly insights on O1 Pro mode**: A member humorously claimed that **Sam Altman** charged them **$200** for information related to the **OpenAI O1 Pro mode**.
   - In a subsequent quote, it was noted that O1 can tackle complex tasks like processing unstructured data and recognizing details in architectural drawings.
- **Requests for O1 Pro mode capabilities**: A member proposed that OpenAI should publish statistics on the number of queries related to counting 'R's in the word **strawberry** using O1 Pro mode.
   - This reflects a light-hearted critique of users’ peculiar requests that might not utilize the full capabilities of the O1 models.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/y0b1byte/status/1890417106006511621">Tweet from yobibyte (@y0b1byte)</a>: alignment à la française</li><li><a href="https://x.com/allgarbled/status/1890314331805610143">Tweet from gabe (@allgarbled)</a>: Sam Altman charged me $200 for thisQuoting OpenAI Developers (@OpenAIDevs) Use o-series models to process unstructured data, find a needle in a haystack, improve code, or handle other complex tasks. F...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1340033272796807239)** (7 messages): 

> `Reasoning model ideas, Fun experiments in AI, GRPO and KL=0, Training metrics and entropy` 


- **Applying KL within <answer> tokens**: A member shared their favorite reasoning model idea of only applying **KL** within the `<answer>` tokens, suggesting it could yield interesting results.
   - They expressed enthusiasm about the idea, stating they 'gotta try it' despite feeling it's too cool to tweet.
- **Encouraging Fun in AI Development**: A member emphasized the importance of letting 'fun stuff happen' in AI projects, indicating a desire for creativity and innovation.
   - This sentiment aligns with an ongoing discussion about engaging with new experimental ideas within the community.
- **Interest in GRPO working with KL=0**: A member referenced a tweet about **GRPO** working with **KL=0**, showing curiosity about whether it was already being discussed in the group.
   - This sparked a dialogue about training strategies and the relevance of KL in various contexts.
- **OpenAI employee inquiry on Twitter**: A member mentioned receiving a DM from an **OpenAI** employee about their tweet, suggesting that there's exciting work happening behind the scenes.
   - This indicates interest from industry professionals in grassroots discussions occurring in the community.
- **Metrics for training considerations**: Another member highlighted the idea that training metrics, such as **entropy**, should differentiate between content inside `<thinking>` tags and elsewhere.
   - This suggests a nuanced approach to evaluating model performance based on context—an important topic in the evaluation of AI systems.


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1339791411276415089)** (9 messages🔥): 

> `Zvi's writing style, Long-form content, Historical commentary, AI perspective on commentary` 


- **Mixed Feelings on Zvi's Lengthy Posts**: Members expressed a shared hesitation about engaging with **Zvi's long-form posts**, feeling that the length often overshadows the content.
   - *I wish I had the patience for the long form* was a sentiment echoed by several users discussing the potential learning benefits.
- **Perceived Monotony in Zvi's Work**: A member noted that Zvi seems to *belabor the same point* repeatedly, leading to concerns about the engagement level of his commentary.
   - Despite this, another member acknowledged that his writing might serve as a **fantastic historical record** of the vibes over time.
- **Heartfelt but Raw Writing**: Comments suggested that Zvi's posts feel *straight from the heart* with limited editing, evoking a relatable sentiment among readers.
   - His approach seems designed more for authenticity than polish, which resonates despite the intimidating length.
- **AI Reception in Zvi's Commentary**: One member humorously remarked that Zvi took the advice to *write for the AIs* too literally, questioning whether AIs would appreciate his comments.
   - This sparked a light-hearted discussion about the suitability of his content for AI audiences, adding a layer of humor to the critiques.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1339655356384546836)** (11 messages🔥): 

> `Notebook LM for Study, Gen Z Social Media Slang Customization, Quality of Responses in Notebook LM, Use Cases for Fantasy Novel Writing, Podcast Functionality Queries` 


- **Notebook LM becomes a 24/7 Personal Tutor**: A user shared how **Notebook LM** has transformed their medical study routine by creating detailed summaries and key points from extensive readings.
   - *It is literally a Personal Tutor who is available 24/7 at your fingertips,* emphasizing the tool's accessibility and utility.
- **Gen Z lingo makes learning fun**: A member highlighted the effectiveness of customizing prompts to use **Gen Z brainrot social media slangs** for explaining complex concepts.
   - This approach has assisted them in grasping difficult subjects in relatable language, making learning easier.
- **Improved responses from Notebook LM**: Another user noticed a distinct improvement in the quality and structuring of **Notebook LM's** answers while studying Gemarah from recordings and texts.
   - *Thanks for the upgrades,* they acknowledged the advancements made to the platform.
- **Exploring sources for fantasy novels**: A user mentioned that their fantasy novel's background involves a detailed exploration of **cosmology, history, and geography** they've developed over the past five years.
   - They provided insights into the various sources that contribute to the depth of their narrative.
- **Questions about podcast functionalities**: A member inquired about the purpose of the podcast feature, questioning whether it's merely an audio overview format rather than a new content creation tool.
   - They experienced difficulties with the interface, receiving a **blank screen** after selecting Pocketcast.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1339643568435892335)** (75 messages🔥🔥): 

> `Notebook LM Language Support, Notebook LM PDF Upload Issues, Notebook LM Subscription Changes, Notebook LM Document Sharing, Gemini Model Functionality` 


- **Notebook LM Language Support Improvements**: Users reported challenges getting Notebook LM to respond in their selected languages, like Bulgarian and German, despite uploading sources in those languages.
   - Some members found success by using specific URLs, such as [notebooklm.google?hl=bg](https://notebooklm.google?hl=bg) for Bulgarian.
- **Issues with PDF Uploading**: A user expressed consistent trouble uploading PDFs, regardless of size or complexity, while others reported it worked fine.
   - This issue appears tied to the user's browser and possibly the system's safety filters when dealing with sensitive content.
- **Notebook LM Subscription Transition**: A German student noted that after switching to the paid version of Notebook LM, they encountered issues with text-to-speech reverting to English.
   - Users helped each other by sharing tips on maintaining language settings, confirming successful adjustments thereafter.
- **Document Sharing Capabilities**: Discussion surfaced around sharing projects in Notebook LM, with questions regarding any limitations on the number of collaborators allowed.
   - Members indicated they had not encountered limits based on their experiences, promoting collaboration among users.
- **Gemini Multimedia Functionality**: Several users asked about the new Gemini model's functionalities, particularly regarding its integration in Notebook LM.
   - Responses indicated ongoing uncertainty about its capabilities within the platform, with users encouraged to explore related resources.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://notebooklm.google/plus/terms">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: Use the power of AI for quick summarization and note taking, NotebookLM is your powerful virtual research assistant rooted in information you can trust.</li><li><a href="https://support.google.com/notebooklm/answer/15724963?hl=en">Learn how NotebookLM protects your data - NotebookLM Help</a>: no description found
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1339656691318521986)** (58 messages🔥🔥): 

> `Latent Reasoning in LLMs, Veo 2 Video Generation Model, New Apple Products, DeepHermes 3 LLM, Beekeeping Feasibility Report` 


- **Latent Reasoning Revolutionizes LLMs**: A new paper discusses how LLMs can utilize **latent reasoning** before producing tokens, contrasting with traditional chain of thought methods.
   - This approach promises significant benefits, as highlighted by community discussions revolving around its practical implications.
- **Veo 2 Takes Center Stage in Video Creation**: Nvidia's new model, **Veo 2**, now featured on YouTube Shorts, allows creators to generate video clips based on quick text prompts with its **Dream Screen** feature.
   - This innovation enhances storytelling by enabling seamless integration of user-generated content into videos.
- **Anticipation Builds for New Apple Devices**: **Tim Cook** teased an upcoming Apple launch, potentially including new products such as the **iPhone SE**, **M4 Air**, and updated Apple TV options.
   - Speculation surrounding a **HomePod with screen** and Apple's continued integration of powerful chips for AI capabilities has generated buzz in the community.
- **DeepHermes 3 Aims to Enhance LLM Capabilities**: The latest **DeepHermes 3** model from Nous Research aims to combine reasoning and traditional LLM response modes into a single functional architecture.
   - This model seeks to mark significant improvements in LLM annotation, judgement, and function calling capabilities.
- **Beekeeping Business Research Insights Shared**: A member shared a comprehensive **Beekeeping Feasibility Report** they've conducted, resulting in actionable steps and insights for potential business strategies.
   - The collaborative discussions around researching and optimizing prompts for deep research further enriched the community's understanding of leveraging AI in real-time projects.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05167">NoLiMa: Long-Context Evaluation Beyond Literal Matching</a>: Recent large language models (LLMs) support long contexts ranging from 128K to 1M tokens. A popular method for evaluating these capabilities is the needle-in-a-haystack (NIAH) test, which involves ret...</li><li><a href="https://engineering.fb.com/2025/02/05/security/revolutionizing-software-testing-llm-powered-bug-catchers-meta-ach/">Revolutionizing software testing: Introducing LLM-powered bug catchers</a>: WHAT IT IS Meta’s Automated Compliance Hardening (ACH) tool is a system for mutation-guided, LLM-based test generation. ACH hardens platforms against regressions by generating undetected faults (mu…</li><li><a href="https://x.com/anneouyang/status/1889770174124867940">Tweet from Anne Ouyang (@anneouyang)</a>: New blog post from Nvidia: LLM-generated GPU kernels showing speedups over FlexAttention and achieving 100% numerical correctness on 🌽KernelBench Level 1</li><li><a href="https://x.com/MatthewBerman/status/1890081482104008920?t=V3aeg7FX8ZvIKtvhHtl-xA&s=19">Tweet from MatthewBerman (@MatthewBerman)</a>: New research paper shows how LLMs can &#34;think&#34; internally before outputting a single token!Unlike Chain of Thought, this &#34;latent reasoning&#34; happens in the model&#39;s hidden space.TONS ...</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: no description found</li><li><a href="https://x.com/GoogleDeepMind/status/1890054036168356283">Tweet from Google DeepMind (@GoogleDeepMind)</a>: 🎥 Our state-of-the-art video generation model Veo 2 is now available in @YouTube Shorts.With the Dream Screen feature, creators can:✨ Produce new clips that fit seamlessly into their storytelling wit...</li><li><a href="https://www.youtube.com/watch?v=LP5OCa20Zpg&ab_channel=Anthropic">Tips for building AI agents</a>: Anthropic’s Barry Zhang (Applied AI), Erik Schultz (Research), and Alex Albert (Claude Relations) discuss the potential of AI agents, common pitfalls to avoi...</li><li><a href="https://x.com/emollick/status/1887579095610860014">Tweet from Ethan Mollick (@emollick)</a>: Interesting data point on OpenAI&#39;s Deep Research: I have been getting a steady stream of messages from very senior people in a variety of fields who have been, unsolicited, sharing their chats and...</li><li><a href="https://x.com/tim_cook/status/1890068457825394918">Tweet from Tim Cook (@tim_cook)</a>: Get ready to meet the newest member of the family.Wednesday, February 19. #AppleLaunch</li><li><a href="https://docs.google.com/document/d/1BIBInKu1rET9-BC520OXXz95XuFOxuMAr63nytI9DZQ/edit?usp=sharing">Feasibility and Strategy for Launching a Commercial Beekeeping Operation in Myakka City, FL</a>: Great! I’ll conduct an in-depth feasibility study and strategic analysis for launching a commercial beekeeping operation in Myakka City, Florida. The report will include financial projections, hive ca...</li><li><a href="https://buttondown.com/ainews/archive/ainews-reasoning-models-are-near-superhuman/">[AINews] Reasoning Models are Near-Superhuman Coders (OpenAI IOI, Nvidia Kernels)</a>: RL is all you need. AI News for 2/12/2025-2/13/2025. We checked 7 subreddits, 433 Twitters and 29 Discords (211 channels, and 5290 messages) for you....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: new pod drop! https://x.com/latentspacepod/status/1890101440615453025
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1339669074845896714)** (2 messages): 

> `LlamaIndex Google Cloud Integration, LlamaParse Features` 


- **LlamaIndex integrates with Google Cloud databases**: Easily integrate LlamaIndex with your [Google Cloud databases](https://twitter.com/llama_index/status/1890109073615626388) using our latest features that allow you to utilize your database as an initial data store, vector store, and more.
   - These integrations are designed to be **easy** and **secure**, enhancing your database interactions.
- **Discover the power of LlamaParse**: A comprehensive video on [LlamaParse](https://twitter.com/llama_index/status/1890499579214491967) covers multiple parsing modes, output formats, and how to effectively improve quality with parsing instructions.
   - It includes insights into parsing **audio**, **images**, and utilizing **JSON mode** for optimized results.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1339642838320943156)** (49 messages🔥): 

> `AgentWorkflow for RAG, Using uv for virtual environments, LlamaIndex updates, Outdated packages management, Python functions in workflows` 


- **AgentWorkflow not suited for RAG applications**: `AgentWorkflow` is designed for systems of agents executing tasks rather than RAG, with guidance to use a workflow approach for RAG functionality as described [here](https://docs.llamaindex.ai/en/stable/examples/workflow/rag/).
   - To integrate RAG within `AgentWorkflow`, users are advised to create custom functions that can incorporate user queries for RAG processing.
- **Using `uv` for managing environments**: Users discussed the ease and benefits of using `uv` for creating multiple virtual environments, with shared insights on managing different versions for tools like PyTorch in individual environments.
   - One user provided a shell function to streamline switching between environments and associated project files, suggesting a workflow to enhance convenience.
- **Concerns about switching dependency management tools**: There was a concern over losing functionality when transitioning to `uv` from mini-conda, specifically regarding handling multiple environments for different tasks like training and inference.
   - Alternatives were proposed, including maintaining separate `pyproject.toml` files for different environments and linking them dynamically upon activation.
- **Managing outdated packages with aliases**: A user shared a bash alias to simplify the process of checking and updating outdated `llama-index` packages, saving time on manual tracking.
   - This alias allows them to run a single command weekly to ensure all their `llama-index` packages are up to date.
- **Resources for learning about RAG**: Users were directed to extensive documentation and examples to better understand how to implement RAG using LlamaIndex, as well as the relationship with agent workflows.
   - Links to starter tutorials and in-depth guides were provided, emphasizing the importance of using RAG in effective data management.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.pantsbuild.org/dev/reference/targets/uv_requirements">uv_requirements | Pantsbuild</a>: Generate a `python_requirement` for each entry in `pyproject.toml` under the `[tool.uv]` section.</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/rag/">RAG Workflow with Reranking - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">Starter Tutorial - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/">AgentWorkflow Basic Introduction - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/rag/">Introduction to RAG - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1339642960631037972)** (3 messages): 

> `Model Finetuning, AI Community Growth, Quantum Education Initiatives` 


- **Finetuning Models for Complex Tasks**: A member highlighted the need to finetune a model when the task or domain is too complex, especially when the input data is significantly different from the training data.
   - They added that extensive prompt engineering may not yield satisfactory results, necessitating finetuning for better performance.
- **AI Innovators Unite in India**: An invitation was extended to join India’s fastest-growing AI community to connect, collaborate, and innovate in artificial intelligence.
   - Members can join via the provided [WhatsApp link](https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU) and be part of the burgeoning scene.
- **Embarking on Quantum Education**: Another message promoted India's quantum education community dedicated to advancing knowledge in quantum computing.
   - Participants are encouraged to empower their learning journey by joining through the [WhatsApp link](https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8).


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://chat.whatsapp.com/JcXJDtmi0gL9kWP4K1cIiU">Ai - ML - QB</a>: WhatsApp Group Invite</li><li><a href="https://chat.whatsapp.com/JN9pUV3uMydHzxT52HryF8">Quantum-QB</a>: WhatsApp Group Invite
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1339651249993154602)** (44 messages🔥): 

> `OpenRouter vs Glama, Issues with OpenWebUI, Using 0.0.0.0 in Networking, Instructions for Setup, Community Discussions on MCP Server Roles` 


- **OpenRouter takes a backseat to Glama**: Glama is reported to be **cheaper**, **faster**, and guarantees **privacy** compared to [OpenRouter](https://glama.ai/models/), though it supports fewer models.
   - The established benefits also include **additional pricing** details for various models ranging from **$0.06 to $10**.
- **OpenWebUI suffers from breaking changes**: Users express concerns that OpenWebUI makes **breaking changes** with every minor update, leading to **80+% of community functions** not functioning properly.
   - Some find it challenging as it is **experimental alpha software**, often filled with race conditions which complicate usability.
- **Debate around using 0.0.0.0 in networking setups**: The IP address **0.0.0.0** is debated regarding its functionality; it's often used to listen on all interfaces, particularly in **containerized environments**.
   - However, some argue against its application as a destination in HTTP contexts and stress the importance of understanding correct usage for **troubleshooting**.
- **Setting up OpenWebUI requires specific steps**: Instructions discuss the necessity of ensuring that the endpoint **/v1/chat/completion** is operational before proceeding to set up OpenWebUI.
   - The discussion concluded that you must set the **OPENAI_API_KEY** to utilize the **OpenAI API**, followed by specific configurations for OpenWebUI.
- **MCP Server author role assignments**: Members were encouraged to share links to their servers to receive the **MCP server author role**, with some sharing their respective GitHub repositories.
   - The role assignments were confirmed, indicating that providing a demo server project or library qualifies for the author status.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://unix.stackexchange.com/questions/94018/what-is-the-meaning-of-0-0-0-0-as-a-gateway">What is the meaning of 0.0.0.0 as a gateway?</a>: Can anyone clarify gateway assignment for me?&#xA;What is the difference between adding a gateway as 0.0.0.0 and assigning a specific IP address as a gateway?</li><li><a href="https://en.m.wikipedia.org/wiki/0.0.0.0">0.0.0.0 - Wikipedia</a>: no description found</li><li><a href="https://glama.ai/models/">Leading LLM Models</a>: Enterprise-grade security, privacy, with features like agents, MCP, prompt templates, and more.</li><li><a href="https://github.com/mashriram/azure_mcp_server">GitHub - mashriram/azure_mcp_server</a>: Contribute to mashriram/azure_mcp_server development by creating an account on GitHub.</li><li><a href="https://github.com/PederHP/mcpdotnet">GitHub - PederHP/mcpdotnet: .NET implementation of the Model Context Protocol (MCP)</a>: .NET implementation of the Model Context Protocol (MCP) - PederHP/mcpdotnet
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1339991564243701871)** (4 messages): 

> `Zonos TTS MCP, Intonation Control for Claude, Use of SSML Tags, Markdown vs SSML, Text-to-Speech Models` 


- **Zonos TTS MCP gives Claude a voice**: The [Zonos TTS MCP](https://github.com/PhialsBasement/Zonos-TTS-MCP) server enables **Claude** to have a voice similar to **CGPT**, enhancing user interaction.
   - *This development opens up fresh avenues for dialogue-based AI applications.*
- **Markdown interpreter needed for intonation**: A member mentioned the necessity of a **markdown interpreter** to allow Claude to control its **intonation**, bringing it closer to optimal performance.
   - *They expressed optimism, stating that once this feature is implemented, they will be "golden".*
- **SSML tags can enhance speech models**: Incorporating **SSML tags** is suggested as a method to leverage Claude's capabilities, allowing for more nuanced control of speech characteristics.
   - *One member advocated for this, stating that the models are 'super smart' and capable of utilizing such features effectively.*
- **Preference for markdown over SSML**: Discussions highlighted a preference for **markdown**, noting its effective use in TTS models like **ElevenLabs**, which provide clearer directive capabilities.
   - *Members felt that markdown can offer a good transcript while ensuring precise tonal direction for text-to-speech.*



**Link mentioned**: <a href="https://github.com/PhialsBasement/Zonos-TTS-MCP/tree/main">GitHub - PhialsBasement/Zonos-TTS-MCP: MCP server that allows Claude to have a voice.</a>: MCP server that allows Claude to have a voice. Contribute to PhialsBasement/Zonos-TTS-MCP development by creating an account on GitHub.

  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1339656500070846485)** (38 messages🔥): 

> `Evaluating RAG Systems, Tinystories Pretraining, Generative Models and RL, Pretraining on Consumer Hardware, Logits in Model Pipelines` 


- **Evaluating RAG System Quality**: A member with a background in computer vision seeks advice on relevant metrics for assessing the quality of their RAG system, which has a stable retrieval setup.
   - They ask the community for any guidance on metrics used in evaluating LLMs or retrieval architectures.
- **Tinystories: More Than Just Pretrained Models**: Discussion indicates that **Tinystories** is not just a set of pretrained models but encompasses a family of architectures, a dataset, and a research paper detailing the setup process.
   - Members emphasize that Tinystories did the hard work necessary to achieve coherent output from small models and are useful for those just starting.
- **Logits as Intermediate Representations**: A member explains that logits will be treated as intermediate representations in their model rather than final outputs, integrating changes in pipelines that favor logits.
   - They propose to move softmax to the end of the pipeline while implementing a **multi-objective training** strategy involving SFT, IRL/RL, and EBM.
- **Challenges with Accelerate + DeepSpeed**: A user questions why **Accelerate + DeepSpeed** consumes more RAM than **Unsloth**, wondering if they are using the tools incorrectly.
   - This reflects ongoing conversations about optimizing performance on consumer-grade hardware and the trade-offs in RAM usage.
- **Training Generative Models with Energy-Based Methods**: A discussion unfolds on the idea of delaying normalization to improve RL performance in generative sequence models, suggesting that irregularities may be beneficial.
   - Key strategies include using dynamic logits and incorporating SFT to guide the model toward meaningful outcomes in training.



**Link mentioned**: <a href="https://openreview.net/forum?id=O-XJwyoIF-k">Minimum Width for Universal Approximation</a>: The universal approximation property of width-bounded networks has been studied as a dual of classical universal approximation results on depth-bounded networks. However, the critical width...

  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1339767478628122634)** (2 messages): 

> `Weekly Crunch Time, Future Meeting Plans` 


- **Members face a weekly crunch**: Members expressed that this week is particularly **busy**, indicating a sense of urgency and time constraints.
   - One member mentioned, *'I still have a lot of stuff to do,'* reflecting the common sentiment of overload among participants.
- **Uncertain plans for tomorrow's meeting**: There is uncertainty about whether a meeting will occur tomorrow, with one member noting they *'wish I could do one today but will have to wait.'*
   - However, they anticipate returning to a normal schedule next week, expressing a hopeful outlook for future discussions.


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1339741152571818034)** (2 messages): 

> `New AI Model without tokens, Latent space reasoning model, 3.5 billion parameter model` 


- **AI Model Reasons Without Tokens**: The [YouTube video](https://www.youtube.com/watch?v=ZLtXXFcHNOU) discusses whether models can 'think' without using a single token, posing an intriguing question about AI capabilities.
   - *Join My Newsletter for Regular AI Updates* and discover what this new approach represents in the AI landscape.
- **Latent Space Model Challenges Token Usage**: An [arXiv paper](https://arxiv.org/abs/2502.05171) presents a novel language model architecture that scales test-time computation by reasoning in latent space without needing specialized training data.
   - This model manages to improve its performance on reasoning benchmarks with a computation load comparable to **50 billion parameters**.
- **3.5 Billion Parameters for Enhanced Reasoning**: The paper describes a proof-of-concept model scaled to **3.5 billion parameters** and trained on **800 billion tokens**, achieving significant performance improvements.
   - It stands out by iterating a recurrent block, allowing for depth unrolling at test-time, in stark contrast to traditional methods relying on increasing token production.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: We study a novel language model architecture that is capable of scaling test-time computation by implicitly reasoning in latent space. Our model works by iterating a recurrent block, thereby unrolling...</li><li><a href="https://www.youtube.com/watch?v=ZLtXXFcHNOU">New AI Model &quot;Thinks&quot; Without Using a Single Token</a>: Can models think without using tokens?! Really??Join My Newsletter for Regular AI Updates 👇🏼https://forwardfuture.aiMy Links 🔗👉🏻 Subscribe: https://www....
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1339695065898942517)** (2 messages): 

> `Elon Musk, Open PTLMs, Model Registry Challenges` 


- **Tag Team Battle Hints at Rivalry**: A member humorously suggested a potential tag team battle with **Sam/Gates vs Elon**, noting that few would want to side with Elon at the moment.
   - *It reflects the current public sentiment around Elon's partnerships and image.*
- **Study Reveals PTLM Release Inconsistencies**: An empirical study based on **52,227 PTLMs** on Hugging Face unveiled that **40.87%** of model weight changes weren't reflected in naming practices or documentation.
   - The results highlighted the **ambiguity** in naming conventions and the accessibility of training documentation for Pre-trained Language Models.



**Link mentioned**: <a href="https://arxiv.org/abs/2409.10472">Towards Semantic Versioning of Open Pre-trained Language Model Releases on Hugging Face</a>: The proliferation of open Pre-trained Language Models (PTLMs) on model registry platforms like Hugging Face (HF) presents both opportunities and challenges for companies building products around them....

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1339765989138632766)** (15 messages🔥): 

> `PR Submission Guidelines, Kernel and OptOps Understanding, VIZ on WSL Issues` 


- **Strict PR Submission Rules Enforced**: *Everyone is reminded to review the diff three times before submitting* their PRs to avoid whitespace changes, or risk having their PR closed without comment.
   - Moreover, submissions with AI-generated code are discouraged, as they waste time, and *it's crucial to write the code yourself* and seek AI feedback instead.
- **Kernel and OptOps Speed Bounty Insights**: A member shared their understanding of the Kernel and OptOps related to the `sum` bounty, suggesting the creation of an OptOp to optimize the AST for multiple reductions.
   - They expressed concern over the expressiveness of current OptOps in achieving target code and are keen on exploring the `GROUP` OptOp for multiple accumulators, noting the renderer should mostly function as expected.
- **Seeking Help with VIZ on WSL**: A user asked if anyone had tried using `VIZ=1` on WSL Ubuntu, as they were encountering errors accessing the temporary directory.
   - Another member acknowledged that WSL builds can be challenging, especially with Python, and offered to download the required setup to further investigate the issue.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1339652530136940615)** (11 messages🔥): 

> `DSPy vs LangChain, DSPy 2.6 Change Log, Removal of DSPy Assertions, Multi-label Classification with DSPy, DSPy Code Golf` 


- **Distinguishing DSPy and LangChain**: One member clarified that if users prefer writing signatures and modules over string prompts or need optimization, they should choose **DSPy** over **LangChain**.
   - They also advised considering whether **LangChain** offers a prepackaged approach if DSPy feels too tricky for their needs.
- **Inquiry About DSPy 2.6 Changes**: A user returned to DSPy, inquiring about a change log for **DSPy 2.6** and mentioned new features like 'instructions' for *Signatures*.
   - Another member noted that these instructions have been around since **2022** and directed the user to the [GitHub release page](https://github.com/dspy/dspy/releases) for detailed changes.
- **Confusion Over Removed Constants in DSPy 2.6.3**: Members discussed the removal of **dspy.Assert**, **dspy.Suggest**, and **dspy.Retry** in version **2.6.3**, creating confusion over alternatives for backward compatibility.
   - One member suggested that the removal is part of a plan to eventually introduce *assertions v2*, although there is no roadmap or explanation provided.
- **Optimizing Multi-label Classification in DSPy**: A user sought advice on using DSPy for optimizing an **SLM** for multi-label classification with 200 class descriptions, proposing a batching strategy.
   - They expressed a desire to achieve this without fine-tuning the model or using multiple **LoRA adapters**.
- **DSPy Code Golf Fun**: A member sparked interest in a fun **DSPy** code golf activity, wanting to challenge others to write concise code snippets.
   - They shared a specific example of extracting structured data from HTML in a one-liner, indicating that the community could make this a competitive game.



**Link mentioned**: <a href="https://x.com/lateinteraction/status/1890442615700545878">Tweet from Omar Khattab (@lateinteraction)</a>: Sometimes I look for an excuse to spend some 5 minutes on some neat DSPy golf.Someone was asking: How do I use DSPy for extraction of structured data from HTML? Hmm, but that&#39;s a one-liner.What if...

  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1339986595125788744)** (1 messages): 

> `Valentine's Day, MAX and Mojo` 


- **MAX and Mojo Celebrate Valentine's Day**: MAX and Mojo spread the love this Valentine's Day with a cheerful greeting and a fun image shared in the channel.
   - The attached image titled 'MAXMojoValentine' adds a festive touch, making the celebration even more interactive.
- **Festive Image Shared**: A delightful image titled 'MAXMojoValentine.jpeg' was shared to mark the occasion, showcasing the spirit of Valentine's Day.
   - This interactive element brings a sense of joy and community to the channel.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1339682548066029630)** (7 messages): 

> `Memory Error in Function Call, Release of v25.1, Dog/Cat Example Confusion, Larecs GitHub Repository, Safe Mutable Aliasing Document` 


- **Debugging Memory Error in Function Call**: A user inquired about an error related to the `add_fun` call in their code, specifically discussing aliasing issues with mutable references.
   - The code appears to create conflicts with memory locations accessed through aliased arguments.
- **Excitement for v25.1 Release**: An anonymous user announced the release of **v25.1**, garnering enthusiasm from the community.
   - The exclamation mark and fire emoji indicate high interest in the updates brought by this release.
- **Misunderstanding Dog/Cat Example**: A user expressed confusion regarding a previously shared **Dog/Cat example** that led to their earlier misunderstanding.
   - Another user acknowledged the confusion, clarifying that it was not their example.
- **Exploration of Larecs Repository**: A member provided a link to the [Larecs GitHub repository](https://github.com/samufi/larecs) for others interested in further details.
   - The tree emoji implies a focus on growth or development within the project.
- **Document on Safe Mutable Aliasing**: A user asked for a link to a document on **safe mutable aliasing** authored by another member.
   - In response, the author shared a link to their [proposal/vision document](https://gist.github.com/nmsmith/cdaa94aa74e8e0611221e65db8e41f7b) published in November.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1339664080038006825)** (8 messages🔥): 

> `Token banning in configuration, Deepseek model recommendations, Fine-tuning LLMs, TradingView access` 


- **Inquiry on Token Banning**: A member asked if there's a way to ban tokens through configuration files, acknowledging it's not a feature in the GUI.
   - The question reflects an interest in customizing token behavior even without official support.
- **Best Deepseek Model for RTX 3080**: Discussions highlighted that distilling deepseek behavior onto a smaller model may result in reduced performance, particularly on an RTX 3080.
   - The **Qwen2.5 Coder 14B** was suggested as a viable option for lower VRAM setups, with members noting the performance trade-offs.
- **Challenges of Fine-Tuning LLMs**: One member questioned how to update and fine-tune an LLM with data from the year 2021.
   - Another member clarified that it is not possible, indicating limitations in adapting older models with new data.
- **TradingView Premium Access**: A post shared links to free cracked versions of TradingView for both Windows and macOS, citing a significant user base.
   - The instructions included detailed steps for installation, emphasizing the availability of Premium features at no cost.



**Link mentioned**: <a href="https://www.reddit.com/r/TradingViewFree/comments/1hobjs6/tradingview_premium_ultimate_package_update/">Reddit - Dive into anything</a>: no description found

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1340002147198177393)** (3 messages): 

> `RFC for Dataloader Transform, Online DPO/GRPO Data Generation, Prompt to Preference Function` 


- **RFC Proposal for Dataloader Transform**: A member is planning to propose an RFC to add a dataloader transform and saving capability, enhancing online **DPO/GRPO data generation** at train time.
   - *This could streamline the conversion of various datasets into a preference dataset by applying different reward models or judges.*
- **Request for Example Usage**: A member asked for an existing example of the proposed dataloader transformation to better understand its application in context.
   - *This query highlights a need for practical illustrations to support the RFC discussion and implementation.*
- **Demonstration of Batch Generation with Transformation**: An example was shared showing how the **prompt_to_preference** function utilizes a `DataLoader` to generate batches of preference data.
   - *The setup allows for two generations per prompt and incorporates a judge for selecting chosen and rejected items, indicating viability for batched generation.*


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1339781201447092245)** (2 messages): 

> `Distillation Scaling Laws, Quantization-Aware Training, QuEST Method, Sparse Representations in LLMs` 


- **Apple's Insights on Distillation Scaling Laws**: A recent discussion highlighted a [paper from Apple](https://arxiv.org/abs/2502.08606) focusing on **distillation scaling laws**, pondering whether it's better to distill from a more powerful model or train from scratch.
   - A quote from the discussion emphasized, *'it's complicated...'* regarding the choices surrounding model size and capabilities.
- **Advancements in Quantization-Aware Training**: A new study has furthered the understanding of **Quantization-Aware Training (QAT)**, exploring ways to achieve accuracy while using quantized representations, particularly with an optimal bit-width of **8-bits** for weights and activations.
   - The potential of this approach has been validated by referencing the state-of-the-art study [arXiv:2411.04330v2](https://arxiv.org/abs/2411.04330v2).
- **QuEST Method Shows Promise for Compression**: A member introduced a new method called **QuEST**, which claims to outperform previous techniques by maintaining strong accuracy with model sizes utilizing **4-bits** or less for weights and activations.
   - This method is positioned as **Pareto-competitive with FP16**, providing better accuracy at reduced model sizes.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05003">QuEST: Stable Training of LLMs with 1-Bit Weights and Activations</a>: One approach to reducing the massive costs of large language models (LLMs) is the use of quantized or sparse representations for training or deployment. While post-training compression methods are ver...</li><li><a href="https://x.com/danbusbridge/status/1890013666575282669?s=46&t=b1X88nwMsmZgHkmMFkiG3g">Tweet from Dan Busbridge (@danbusbridge)</a>: Reading &#34;Distilling Knowledge in a Neural Network&#34; left me fascinated and wondering:&#34;If I want a small, capable model, should I distill from a more powerful model, or train from scratch?&#...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1339906564072669219)** (4 messages): 

> `Cohere Command R+` 


- **Exciting Project Underway with Cohere Command R+**: A member announced they are *building something really cool with Cohere Command R+*, encouraging others to stay tuned for updates.
   - The excitement was shared by another member who responded with a laughing emoji, indicating enthusiasm for the project.
- **Lighthearted Reactions to the Announcement**: Another member chimed in with a laughing emoji in response to the project announcement, reflecting a lighthearted atmosphere in the discussion.
   - This contributed to a shared sense of enthusiasm and community engagement around what is being built.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1339823696633790516)** (2 messages): 

> `Quiz 3 release` 


- **Quiz 3 Release Confusion Resolved**: A member inquired whether **Quiz 3** was available, stating they couldn't find it on the MOOC website.
   - They later noticed an update, revealing that the information was available on [Discord](https://discord.com/channels/1280234300012494859/1293323662300155934).
- **Quick Resolution of Quiz Availability**: The initial inquiry about **Quiz 3** highlighted some confusion regarding its release date, as it was not visible on the website.
   - Fortunately, the member found the relevant update in a different thread on Discord that clarified the situation.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1339720479065247844)** (1 messages): 

> `AI/ML Guidance, Model Training Techniques` 


- **Newbie seeks guidance in AI/ML**: A member expressed being very new to the **AI/ML** domain and requested guidance on where to start with model training techniques.
   - *Looking for your help.*
- **Request for Resources and Tips**: The same member is also looking for resources and tips related to advancing beyond initial model training.
   - Suggestions for online courses and community forums were encouraged to help them get started.


  

---


---


---


{% else %}


> The full channel by channel breakdowns have been truncated for email. 
> 
> If you want the full breakdown, please visit the web version of this email: [{{ email.subject }}]({{ email_url }})!
>
> If you enjoyed AInews, please [share with a friend](https://buttondown.email/ainews)! Thanks in advance!

{% endif %}

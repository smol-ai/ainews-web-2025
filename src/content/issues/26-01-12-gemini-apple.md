---
id: MjAyNi0w
title: Apple picks Google's Gemini to power Siri's next generation
date: '2026-01-12T05:44:39.731046Z'
description: >-
  **Apple** has decided to power Siri with **Google's Gemini models** and cloud
  technology, marking a significant partnership and a setback for **OpenAI**,
  which was initially partnered with Apple. **Anthropic** launched "Cowork," a
  product preview for Claude's coding capabilities, sparking discussions about
  "LLM OS". **OpenAI** introduced **ChatGPT Health** and acquired **Torch** to
  expand in healthcare AI. **DeepSeek** unveiled **Engram**, a new conditional
  memory module that enables O(1) lookup-style memory for static patterns,
  improving long-context handling and offering hardware-friendly optimizations
  to scale knowledge capacity efficiently. Engram is positioned as a key
  modeling primitive for next-gen sparse models, with ongoing community debate
  about its architectural merits and practical impact.
companies:
  - apple
  - google
  - openai
  - anthropic
  - deepseek
models:
  - gemini
  - claude
  - chatgpt
  - engram
topics:
  - conditional-memory
  - long-context
  - hashing
  - memory-optimization
  - transformers
  - model-scaling
  - sparsity
  - hardware-optimization
  - model-architecture
  - ai-healthcare
  - model-optimization
people: []
---


**Apple finally gives in.**

> AI News for 1/12/2026-1/13/2026. We checked 12 subreddits, [**544** Twitters](https://twitter.com/i/lists/1585430245762441216) and **24** Discords (**204** channels, and **1541** messages) for you. Estimated reading time saved (at 200wpm): **157 minutes**. **Our new website** is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on [@smol_ai](https://x.com/Smol_AI)!


The only people more sympathetic than Apple's struggling AI division are Apple's users who have not had the promised updates to Siri. After over a year of talking about their own inhouse trained models, Apple [announced](https://www.wcnc.com/article/news/nation-world/apple-google-gemini-siri-ai-features/507-575faa99-217e-498d-8f34-5455759113f8) that they are picking Google's Gemini to power Siri, with a security layer still provided by Apple Private Cloud Compute. A nice win for Google and a comparative loss for OpenAI, which Apple initially partnered with for their AI launch but is increasingly having rumors of their consumer hardware device shipping this year, which may have factored into the decision.

---

# AI Twitter Recap

**Top tweets (by engagement)**

- **Apple ↔ Google AI partnership**: A joint statement says the “next generation of Apple Foundation Models” will be based on **Google’s Gemini models and cloud technology**, powering future Apple Intelligence features including a “more personalized Siri” while maintaining Apple privacy posture ([tweet](https://twitter.com/NewsFromGoogle/status/2010760810751017017)).
- **Anthropic launches “Cowork”**: Claude’s “Claude Code for the rest of your work” product preview drives major engagement and discussion about “LLM OS” trajectories ([announcement](https://twitter.com/claudeai/status/2010805682434666759), [context](https://twitter.com/bcherny/status/2010809450844831752)).
- **OpenAI in health**: OpenAI introduces **ChatGPT Health** (a dedicated space with separated memories) and announces acquisition of **Torch** ([ChatGPT Health](https://twitter.com/OpenAI/status/2010764845432590469), [acquisition](https://twitter.com/OpenAI/status/2010813780671021106)).
- **DeepSeek “Engram”**: Viral technical thread highlights DeepSeek’s new conditional memory / lookup module as a new “axis of sparsity” ([thread](https://twitter.com/scaling01/status/2010748516788777445)).

---

**DeepSeek’s Engram: conditional memory as a new sparsity primitive**

- **Engram = O(1) lookup-style memory for static patterns**: DeepSeek introduces “**Conditional Memory via Scalable Lookup**” (Engram), adding a hashed **n‑gram embedding** memory that the model can query and gate into representations. The framing: transformers spend compute *reconstructing memorized patterns* via forward passes; Engram offloads that “static retrieval” so backbone capacity can focus on “effective depth”/reasoning ([@scaling01](https://twitter.com/scaling01/status/2010748516788777445)). Follow-up notes claim it helps **long-context** as well ([tweet](https://twitter.com/scaling01/status/2010748710653980989)).
- **Why it matters for systems + scaling**: Multiple takes converge on the idea that deterministic hashing/lookup enables hardware-friendly optimizations (prefetching, memory movement) and shifts the scaling constraint away from **HBM-bound parameters**, suggesting a practical route to grow “knowledge capacity” cheaply while keeping FLOPs stable ([@tokenbender critique + summary](https://twitter.com/tokenbender/status/2010791813964296558), [@teortaxesTex](https://twitter.com/teortaxesTex/status/2010763425849430184)). A key quoted line—“**conditional memory as an indispensable modeling primitive**” for next-gen sparse models—gets repeated as the thesis ([quote tweet](https://twitter.com/scaling01/status/2010750095923499489)).
- **Relation to prior art (Gemma-3n / N-Grammer / PLE / OTT)**: Engineers map Engram onto earlier “inject n‑gram info at embeddings” approaches (Over-Tokenized Transformers, Per-Layer Embeddings, N‑Grammer), arguing Engram differs by making memory **an active, layer-addressed operation** rather than passive early injection ([comparison](https://twitter.com/gm8xx8/status/2010830166076071970), [@_arohan_](https://twitter.com/_arohan_/status/2010760026689060918)). Others push back that pieces existed “in Gemma‑3n,” with debate focusing on *framing* (edge efficiency vs frontier scaling) and architectural “aesthetics” vs end-to-end capability orientation ([discussion](https://twitter.com/teortaxesTex/status/2010775191320699084)).
- **Community read: promising but “systems-brained” and potentially brittle**: A detailed critique questions whether “always fetch + gate” is the right abstraction vs adaptive compute, and worries about OOD mixing or optimization complexity, while acknowledging the iso-budget gains appear real but modest (~3–5% in the critic’s read) ([@tokenbender](https://twitter.com/tokenbender/status/2010791813964296558)).

---

**Long-context & memory research: DroPE, agent memory, and test-time training**

- **DroPE (Sakana AI): extend context by *dropping* positional embeddings**: Sakana’s “**DroPE**” proposes using RoPE for convergence, then removing positional encodings to avoid semantic distortion during context extension. The claim is a principled middle ground between NoPE training difficulties (vanishing gradients) and RoPE scaling warping low frequencies ([paper tweet](https://twitter.com/SakanaAILabs/status/2010660969719165133), [Japanese explainer](https://twitter.com/iwiwi/status/2010700629744746934)). They also release a reference trainer implementation ([repo](https://twitter.com/SakanaAILabs/status/2010738878727217595)).
- **Test-time training as “memory”: TTT‑E2E**: NVIDIA/Stanford/Astera push “**End-to-End Test-Time Training**” where models continue next-token training on the provided context at deployment time, effectively compressing salient context into weights and reducing reliance on large KV caches for long sequences. Several posts frame this as “a new era for LLM memory” and a path toward subquadratic long-sequence modeling ([NVIDIAAIDev](https://twitter.com/NVIDIAAIDev/status/2010773774849724858), [@karansdalal](https://twitter.com/karansdalal/status/2010774529120092481), [@jure](https://twitter.com/jure/status/2010790789627125877)).
- **Agent memory frameworks converge on *policy-integrated memory ops***:
  - **AgeMem**: treats long-term + short-term memory as a single learnable policy with tool-like actions (ADD/UPDATE/DELETE + RETRIEVE/SUMMARY/FILTER), trained via a staged RL strategy and step-wise GRPO. Reported gains include **+13%** on Qwen2.5‑7B vs Mem0 and larger gaps on Qwen3‑4B ([@omarsar0](https://twitter.com/omarsar0/status/2010712137933730234)).
  - **SimpleMem**: focuses on “semantic lossless compression” + consolidation + query-aware retrieval; claims **43.24 F1 on LoCoMo** vs **34.20** for Mem0 and **30× lower tokens** per query (531 vs 16,910) ([@dair_ai](https://twitter.com/dair_ai/status/2010720188686348593)).

---

**Agents & developer tooling: eval infra, CLI-first workflows, and “LLM OS” moves**

- **Anthropic “Cowork”: Claude Code UX for non-coding work**: Anthropic positions Cowork as “Claude Code for the rest of your work,” bundling an agent with browser automation, connectors, and a sandboxed execution environment. The launch triggers a wave of “LLM OS” commentary and “agentic knowledge work” framing ([launch](https://twitter.com/claudeai/status/2010805682434666759), [product details](https://twitter.com/bcherny/status/2010809450844831752), [“LLM OS” take](https://twitter.com/skirano/status/2010833788591300642)).
- **Internal coding agents go mainstream (Ramp “Inspect”)**: Ramp reports their internal agent wrote **30% of merged frontend+backend PRs in a week**, running fully in the cloud on top of open tooling (opencode, Modal, Cloudflare). They open-source the “blueprint/spec” to build a similar system ([@zachbruggeman](https://twitter.com/zachbruggeman/status/2010728444771074493), [build post](https://twitter.com/rahulgs/status/2010734253538267197)).
- **Agentic evaluation = infra problem (AI21 SWE-bench at scale)**: AI21 says they ran SWE-bench **200k+ times** and the biggest lesson was infrastructure: provision per instance (repo+deps+MCP server) and reuse; separate generation from evaluation so failed tests can be retried without re-generating tokens. They claim failure rate dropped **30% → 0** and repo downloads **8,000+ → 500** ([thread start](https://twitter.com/AI21Labs/status/2010738309681823992)).
- **CLI as a low-token agent interface**: Hugging Face echoes “Bash is all you need” logic and ships composable/discoverable Hub CLIs to let coding agents explore models/datasets/Spaces with low context usage ([@hanouticelina](https://twitter.com/hanouticelina/status/2010664329545224588)).
- **Recursive Language Models (RLM) for million-token workflows**: RLM proposes recursively chunking prompts and aggregating results, with trajectories usable for RL/distillation; it’s being integrated into OpenEnv ([overview](https://twitter.com/SergioPaniego/status/2010765550012735896), [OpenEnv note](https://twitter.com/SergioPaniego/status/2010765552952713241)).

---

**Big tech platform shifts: Apple↔Gemini, Google “agentic commerce,” and API primitives**

- **Apple bets on Gemini for Siri / “Apple Foundation Models”**: The joint statement explicitly ties “next-gen Apple Foundation Models” to Gemini models and cloud tech, while emphasizing on-device + Private Cloud Compute and privacy standards ([statement](https://twitter.com/NewsFromGoogle/status/2010760810751017017)). Commentary emphasizes the strategic logic: Gemini multimodality leadership and competitive tension with OpenAI’s device ambitions ([analysis](https://twitter.com/Yuchenj_UW/status/2010777804246565175)).
- **Google pushes agentic commerce rails**: Google announces **Universal Commerce Protocol (UCP)** + features like direct checkout in AI Mode / Gemini, “Business Agent” chat with retailers, and “Direct Offers” pilots—i.e., making the model the shopping surface and the transaction initiator ([@Google](https://twitter.com/Google/status/2010744570108137524)).
- **Gemini API input scaling**: Gemini API increases inline file limit **20MB → 100MB**, adds native ingestion from **Google Cloud Storage buckets**, and supports signed URLs (enabling other clouds). Separate posts specify max sizes like **2GB** for registered GCS files and **100MB** for external signed URLs, plus supported document formats ([@osanseviero](https://twitter.com/osanseviero/status/2010764447988461634), [@_philschmid](https://twitter.com/_philschmid/status/2010765230134215037)).

---

**OpenAI in healthcare: product boundaries, acquisition, and privacy posture**

- **ChatGPT Health = separate memory domain**: OpenAI introduces “ChatGPT Health” where health chats/files/memories live in a dedicated space; health info “never flows into your regular chats,” and users can view/delete health memories ([announcement](https://twitter.com/OpenAI/status/2010764845432590469)).
- **Acquires Torch Health**: OpenAI acquires Torch, described as unifying lab results, meds, and visit recordings, positioned as accelerating ChatGPT Health capabilities ([OpenAI](https://twitter.com/OpenAI/status/2010813780671021106), [Torch founder](https://twitter.com/IlyaAbyzov/status/2010813621022949721)). A broader recap claims **230M** weekly users use ChatGPT for health-related messages and outlines HIPAA-compliant “ChatGPT for Healthcare” and an API posture ([recap](https://twitter.com/thekaransinghal/status/2010878203401843114)).

---

**China AI business + open ecosystem signals (IPO narratives, adoption metrics, and model distribution)**

- **Zhipu vs MiniMax IPO “story mismatch”**: A ZhihuFrontier summary argues divergent market performance stems from different narratives: **Zhipu** as ToB/ToG infra with long sales cycles and heavy R&D burn vs **MiniMax** as consumer/global platform with growth curves and improving margins ([thread](https://twitter.com/ZhihuFrontier/status/2010642118713512174)).
- **Open model adoption metrics get more rigorous**: A “Relative Adoption Metric (RAM Score)” aims to normalize Hugging Face downloads by time and size, arguing **1–9B models dominate** raw downloads but median top-10 downloads differ by only ~4× between 1–9B and 100B+. The metric flags GPT‑OSS as exceptional and suggests varying momentum among large Chinese MoE releases ([@natolambert](https://twitter.com/natolambert/status/2010744476516655274)).
- **GLM‑4.7 distribution + fast inference**: Mentions include GLM‑4.7 availability via Cerebras on Hugging Face and via Together AI, with marketing claims like **200K context** and strong coding benchmarks (as stated in vendor tweets) ([@NielsRogge](https://twitter.com/NielsRogge/status/2010686205961146400), [Together](https://twitter.com/togethercompute/status/2010832877626286113)).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. New AI Model and Benchmark Releases

  - **[GitHub - deepseek-ai/Engram: Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://www.reddit.com/r/LocalLLaMA/comments/1qb034t/github_deepseekaiengram_conditional_memory_via/)** (Activity: 324): ****DeepSeek AI** has introduced a novel approach in their Engram repository, enhancing large language models with conditional memory via scalable lookup. This method integrates static N-gram memory with dynamic hidden states, offering a new axis of sparsity beyond traditional MoE architectures. The model demonstrates a U-shaped scaling law for optimal capacity allocation between MoE and Engram, allowing for efficient offloading of embedding tables to host memory with minimal performance impact. The use of the Muon optimizer in ablations suggests a shift from AdamW, aligning with trends seen in models like Kimi K2 and GLM 4.5. For more details, visit the [GitHub repository](https://github.com/deepseek-ai/Engram/tree/main).** Commenters highlight the potential of Engram to improve model performance per parameter, with significant parts offloadable to RAM or NVMe without performance penalties. The deterministic addressing and n-gram embedding approach are praised for their innovative contribution to model efficiency and scalability.

    - The DeepSeek team has introduced a model with conditional memory functions, which they describe as a 'current stable meta' and foresee as essential for next-generation sparse models. The model uses mHC (𝑀 = 4) for ablations, indicating a stable configuration. The Engram model is expected to enhance performance per parameter compared to traditional MoE architectures, allowing significant portions of the model to be offloaded to RAM or NVMe without performance loss. This could mean a 40B A3.8B MoE model would only require 27B of weights in fast memory, with the rest comfortably offloaded.
    - The Engram model introduces a novel n-gram embedding approach, adding static memory as a new sparsity axis alongside MoE. This allows for O(1) lookup, which is a significant efficiency improvement. The model's deterministic addressing enables embedding tables to be offloaded to host memory with minimal inference overhead. A U-shaped scaling law was discovered, guiding the allocation of capacity between MoE and Engram, which helps preserve model depth for complex reasoning tasks.
    - The use of the Muon optimizer in the Engram model's ablations suggests a shift from the traditional AdamW optimizer, aligning with trends seen in models like Kimi K2 and GLM 4.5. This choice of optimizer could influence the training efficiency and performance of next-generation models.

  - **[11 Production LLM Serving Engines (vLLM vs TGI vs Ollama)](https://www.reddit.com/r/LocalLLM/comments/1qax6kq/11_production_llm_serving_engines_vllm_vs_tgi_vs/)** (Activity: 5): **The article provides a comparative analysis of 11 production LLM serving engines, including **VLLM**, **TGI**, and **Ollama**, focusing on their performance metrics, scalability, and integration capabilities. It highlights the unique features and use cases of each engine, emphasizing their roles in effective LLM deployment. For further details, refer to the [original article](https://medium.com/@techlatest.net/11-production-llm-serving-engines-vllm-vs-tgi-vs-ollama-162874402840).** A comment suggests additional engines like [Mistral.rs](http://Mistral.rs), AirLLM, and Nexa SDK, indicating a broader landscape of LLM serving solutions. Another comment questions the inclusion of Ollama in a production context, hinting at skepticism about its maturity or suitability for production environments.

    - The discussion highlights additional LLM serving engines beyond the ones mentioned in the post, such as Mistral.rs, AirLLM, Nexa SDK, TabbyAPI, Exllama, Aphrodite, CoboldCPP, KTransformers, exa, and TextSynth Server. These tools are suggested as alternatives or complements to vLLM, TGI, and Ollama, indicating a diverse ecosystem of solutions for deploying large language models in production environments.


### 2. AI Development and System Administration

  - **[Weak Dev, Good SysAdmin needing advice](https://www.reddit.com/r/LocalLLM/comments/1qb1p4x/weak_dev_good_sysadmin_needing_advice/)** (Activity: 19): **The user has acquired a **Beelink Mini PC, GTR9 Pro** with an **AMD Ryzen AI Max+ 395 CPU** and `128GB RAM` for local AI development and gaming. They plan to transition from Windows to Linux, starting with PowerShell module development from existing scripts. The user seeks advice on whether to use native Windows tools or start with **Windows Subsystem for Linux (WSL)**. The system's specifications suggest it can handle both Windows and Linux environments effectively, allowing for flexibility in development and testing.** One commenter suggests that the user's system is capable of running both Windows and Linux environments, either natively or through virtual machines, providing flexibility in tool choice. Another commenter inquires about the specific type of software the user intends to develop, indicating interest in the user's development goals.


  - **[Which LLM would be the "best" coding tutor?](https://www.reddit.com/r/LocalLLM/comments/1qb5vry/which_llm_would_be_the_best_coding_tutor/)** (Activity: 8): **The discussion centers on identifying the most effective large language model (LLM) for coding instruction. A notable mention is **Qwen3 30b coder**, which operates efficiently on a MacBook Air with `24GB` RAM, alongside **OpenAI's 20b model**, which is also noted for its speed on similar hardware. The use of **LLM Studio** is recommended for experimenting with various models, highlighting significant advancements in LLM capabilities by 2025.** One commenter suggests that all LLMs surpass individual coders in coding tasks, but emphasizes the importance of learning how to interpret and verify the information provided by these models, as they can sometimes provide incorrect or fabricated data.

    - **Qwen3 30b coder** is highlighted for its performance, running efficiently on a MacBook Air with 24GB of RAM. This suggests that the model is optimized for resource-constrained environments, making it accessible for personal use. Additionally, **OpenAI 20b** is noted for its speed on the same hardware, indicating that both models are suitable for users with similar setups.
    - The mention of **LLM Studio** suggests a tool or platform that allows users to experiment with different language models. This could be particularly useful for those looking to compare performance and capabilities across various models, especially given the rapid improvements in LLMs by 2025.


### 3. Humorous Takes on AI and Data Privacy

  - **[It seems like people don’t understand what they are doing?](https://www.reddit.com/r/LocalLLM/comments/1qaxwf5/it_seems_like_people_dont_understand_what_they/)** (Activity: 16): **The image is a meme that humorously critiques the casual attitude some individuals may have towards data privacy, particularly in the context of using AI tools like 'Claude Code' at work. The caption implies a satirical scenario where employees unknowingly compromise their employer's data security for personal convenience, such as leaving work early. This reflects broader concerns about data privacy and security in the workplace, especially with the increasing use of AI tools that may inadvertently expose sensitive information.** One commenter expresses frustration with current hardware limitations and the need to invest in better equipment due to rising prices. Another shares their struggle with running multiple tools on limited hardware, highlighting the challenges of balancing local and remote computing resources for AI tasks.

    - A user discusses the limitations of running multiple tools on a 16GB M1 MacMini, highlighting the challenges of local model inference due to insufficient memory. They mention using LM Studio for remote inference, which is slow, and express frustration with using the OpenRouter API for coding, as it requires stripping identifiable code and still fails 25% of the time. This reflects broader issues with hardware constraints and the inefficiencies of current workaround solutions.

  - **[Env vars don't work when your agent can read the environment](https://www.reddit.com/r/LocalLLM/comments/1qb0fsg/env_vars_dont_work_when_your_agent_can_read_the/)** (Activity: 2): **The post discusses a security concern where environment variables (env vars) are not secure if an agent running on the system can read the environment. This is particularly relevant in scenarios where sensitive information, such as API keys or passwords, is stored in env vars. The issue arises because any process with sufficient permissions can access these variables, potentially leading to unauthorized access or data leaks. The post likely emphasizes the need for alternative secure storage solutions for sensitive data, such as using secret management tools or encrypted storage, to mitigate this risk.** Commenters debate the effectiveness of using environment variables for sensitive data, with some suggesting that while convenient, they are not secure enough for production environments. Others recommend using dedicated secret management systems like HashiCorp Vault or AWS Secrets Manager to enhance security.

    - A key issue discussed is the security risk of environment variables when agents have read access to the environment. This can lead to sensitive data exposure, especially if the agent is compromised or malicious. The discussion highlights the importance of using secure vaults or secrets management tools to store sensitive information instead of relying on environment variables.
    - One commenter points out that environment variables are often used for convenience but lack robust security measures. They suggest using tools like HashiCorp Vault or AWS Secrets Manager to manage sensitive data securely. These tools provide encryption and access control, reducing the risk of unauthorized access.
    - Another technical insight is the potential for environment variables to be inadvertently exposed in logs or error messages. This can happen if the application logs the environment for debugging purposes. The recommendation is to ensure that logging configurations are carefully managed to avoid leaking sensitive information.


## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Apple-Google Gemini Collaboration

  - **[Apple announces that next version of Siri would be powered using Google gemini. Elon Musk does not seem happy about it.](https://www.reddit.com/r/OpenAI/comments/1qb7dg6/apple_announces_that_next_version_of_siri_would/)** (Activity: 903): ****Apple** has announced that the next version of **Siri** will be powered by **Google's Gemini** AI, after evaluating it against competitors like **ChatGPT** and **xAI Grok**. Apple stated that Google's technology offers the most capable foundation for their models, promising innovative user experiences. This decision has drawn criticism from **Elon Musk**, who expressed concerns over Google's growing influence, given their control over Android and Chrome. [Read more](https://www.wcnc.com/article/news/nation-world/apple-google-gemini-siri-ai-features/507-575faa99-217e-498d-8f34-5455759113f8).** Commenters noted that Gemini likely outperformed in blind AB testing, which may have influenced Apple's decision. There is skepticism about Musk's criticism, with some pointing out the irony of his concerns over power concentration.

    - Keeltoodeep highlights that Google's Gemini model significantly outperforms in blind AB testing, which is likely a key factor in Apple's decision to integrate it into Siri. This suggests that Apple's evaluation process for Siri's enhancement was heavily data-driven, focusing on empirical performance metrics.
    - alexx_kidd dismisses Elon Musk's potential criticism of Apple's choice, implying that Musk's own AI, Grok, may not match the performance of Google's Gemini. This underscores a competitive landscape in AI development where empirical performance, as demonstrated by Gemini, is a critical differentiator.
    - trixxyhobbitses points out the irony of Elon Musk, a major tech figure, criticizing the concentration of power in AI, while himself being a significant player in the industry. This comment reflects on the broader industry dynamics and the competitive tensions between major tech companies in AI advancements.

  - **[It’s official](https://www.reddit.com/r/OpenAI/comments/1qb79py/its_official/)** (Activity: 812): ****Google** and **Apple** have announced a partnership where Apple's Siri will be powered by Google's **Gemini** AI model starting with iOS 26.4, expected in March 2026. This collaboration aims to enhance Siri's capabilities by integrating personal context understanding and improved app control. This move potentially shifts the competitive landscape, as Google combines its search dominance with Gemini and Apple's distribution, while **OpenAI** remains reliant on ChatGPT and APIs, hoping for regulatory or OEM shifts. The partnership suggests Apple will host Gemini on private servers, maintaining data privacy, and marks a significant upgrade over Apple's current AI capabilities.** Commenters note skepticism about Siri's current capabilities and express hope for improvement. There is also discussion about the data privacy aspect, with Apple reportedly hosting Gemini on private servers, and a former Apple employee suggests Apple prioritizes a 'good enough' model over leading-edge performance.

    - Minimum_Indication_1 highlights that Apple is likely to host a Gemini model instance on private servers, branded as Apple Foundation Models, without Google accessing any data. This suggests a focus on privacy and control over the AI's deployment, aligning with Apple's emphasis on user data protection.
    - Unique_Carpet1901, a former Apple employee, notes that Apple prioritized obtaining a model with weights at minimal cost, which only Google was willing to provide. Apple is content with a 'good enough' model, as it represents a significant improvement over their current capabilities, even if it isn't the leading model in the market.
    - MEGAT0N points out that the Gemini model will run on Apple's servers, with no integration into the web or app versions of Gemini. This indicates a localized deployment strategy, where Siri's enhancements are powered by Gemini but remain distinct from Google's broader ecosystem.

  - **[The Apple-Google "Mega-Brain" is here. Why Siri + Gemini is the end of the internet as we know it.](https://www.reddit.com/r/GeminiAI/comments/1qb1t6h/the_applegoogle_megabrain_is_here_why_siri_gemini/)** (Activity: 392): **The image is a screenshot of a tweet from "News from Google" announcing a collaboration between **Apple** and **Google** to integrate Google's **Gemini models** into Apple's ecosystem, specifically enhancing Siri with more personalized and intelligent features. This partnership aims to leverage Google's AI and cloud technology while maintaining Apple's privacy standards. The collaboration signifies a shift towards a more integrated AI experience across devices, potentially transforming how users interact with digital information by moving from traditional search methods to direct AI-driven answers.** Commenters express skepticism and concern about the implications of this collaboration, with some suggesting it could disrupt existing AI players like OpenAI. Others debate the impact on traditional web content, arguing that while it may reduce reliance on ad-heavy websites, it could also centralize information control under a single AI system.

    - The comment by `phase_distorter41` highlights the evolution of search engines, emphasizing that the goal has always been to streamline the search process, as evidenced by Google's 'I'm Feeling Lucky' button. The user critiques the current state of the internet, noting that it has become cluttered with ad-filled content that obscures the information users seek. This reflects a broader sentiment that AI-driven solutions like Siri + Gemini could simplify information retrieval by bypassing traditional, ad-heavy web pages.
    - `dbvirago` raises a critical point about the dependency of AI models like Gemini on existing web content. The comment questions the sustainability of such models if they effectively replace the need for the very websites they rely on for data. This highlights a potential paradox in AI development: while AI can efficiently summarize and provide information, it still requires a dynamic and continuously updated data source, which traditionally comes from the websites it might render obsolete.

  - **[Report: Apple chooses Google's Gemini to run next version of Siri](https://www.reddit.com/r/GeminiAI/comments/1qayd52/report_apple_chooses_googles_gemini_to_run_next/)** (Activity: 144): **Apple is reportedly collaborating with Google to integrate Google's Gemini AI models into the next version of Siri, as per a CNBC report. This partnership is expected to leverage Google's cloud technology to enhance Apple's AI capabilities, marking a significant strategic move in the AI space. The announcement coincided with Google's market value surpassing Apple's, reaching over $4 trillion, highlighting the impact of this collaboration on market dynamics.** Commenters note that Google's dominance in AI was anticipated since late 2025, suggesting that this partnership further solidifies Google's strong position in the AI market.

    - The decision by Apple to use Google's Gemini for Siri suggests a significant shift in AI strategy, potentially indicating Google's growing dominance in the AI space. This move could put pressure on other AI models and companies, as Apple's endorsement of Gemini might influence market dynamics and enterprise partnerships.
    - There is speculation about the future of other voice assistants like Amazon's Alexa, especially in light of Apple's choice to partner with Google. This decision could lead to increased competition and innovation in the AI assistant market, as companies may need to reassess their strategies and partnerships to remain competitive.
    - The discussion highlights the competitive landscape of AI, with Google seemingly in a strong position. The mention of OpenAI's efforts to secure enterprise deals suggests a strategic response to Google's advancements, indicating a highly competitive environment where major tech companies are vying for dominance in AI technology.


### 2. Claude Cowork and Code Tools

  - **[Introducing Cowork: Claude Code for the rest of your work.](https://www.reddit.com/r/ClaudeAI/comments/1qb5r3y/introducing_cowork_claude_code_for_the_rest_of/)** (Activity: 714): ****Anthropic** has introduced **Cowork**, a feature within the Claude ecosystem that allows users to perform non-technical tasks by granting Claude access to a specific folder on their computer. This enables Claude to read, edit, or create files, and execute tasks with user oversight. Cowork integrates with existing connectors and can be paired with Claude in Chrome for browser-based tasks. It is currently available as a research preview for Claude Max subscribers on macOS, with a waitlist for other users. More details can be found on [Claude's blog](https://claude.com/blog/cowork-research-preview).** Commenters note that Anthropic's approach benefits from observing competitors like Microsoft Copilot, suggesting Cowork may offer a more refined solution. There is also speculation that Cowork's development reflects a strategic shift to appeal to both technical and non-technical users, potentially consolidating tools to reduce enterprise costs.

    - PoorPhipps highlights that Cowork appears to be leveraging a WebUI wrapper around existing tools like the TODO list and AskUserQuestion Tool, suggesting an aggressive ideation strategy beyond the core Claude Code context. This indicates a potential shift towards integrating more user-friendly interfaces to broaden its appeal beyond just technical users.
    - painterknittersimmer references a previous prediction that many Claude Code features would transition to desktop versions, emphasizing a strategic move by Anthropic to appeal to both technical and non-technical users. This approach aims to consolidate tools, potentially reducing costs for enterprises by offering a unified solution instead of separate subscriptions to services like Cursor and ChatGPT.

  - **[Claude just introduced Cowork: the Claude code for non-dev stuff](https://www.reddit.com/r/ClaudeAI/comments/1qb6gdx/claude_just_introduced_cowork_the_claude_code_for/)** (Activity: 596): ****Anthropic** has launched a new feature called **Cowork** as a research preview for **Claude Max** subscribers on macOS. This tool extends the capabilities of Claude Code to non-coding tasks by allowing users to point Claude at a folder on their computer, enabling it to autonomously read, edit, and create files. It can perform tasks such as auto-organizing folders, creating spreadsheets from screenshots, and drafting reports from notes, while integrating with existing connectors and handling browser tasks when paired with Claude in Chrome. More details can be found in the [official blog post](https://claude.com/blog/cowork-research-preview).** Commenters are curious about the differences between Cowork and Claude desktop, noting that Cowork could be beneficial for less tech-savvy users. However, there are concerns about potential data loss if users do not have backups, as Cowork operates with significant autonomy.

    - deepthinklabs_ai raises a technical inquiry about the differences between Claude Cowork and Claude Desktop, suggesting a need for clarity on feature sets and user interfaces tailored for non-developers versus developers.
    - Ok-Inspection-2142 points out that the functionality of Claude Cowork seems similar to what is already available in the desktop version, especially for users on the max plan, which includes directory read/write capabilities. This suggests that the new offering might be more about marketing to a different user base rather than introducing new technical features.
    - trimorphic highlights a potential risk for non-technical users who may not have robust backup solutions, emphasizing the importance of data management and the potential consequences of relying on AI tools for critical tasks without proper safeguards.

  - **[Agentic CLI Tools Comparison](https://www.reddit.com/r/CLine/comments/1qaycqj/agentic_cli_tools_comparison/)** (Activity: 8): **The image is a bar chart that visually represents the success rates of various agentic CLI tools tested on 20 web development tasks. The tools compared include Kiro, Aider, Cline, Claude Code, OpenAI Codex CLI, and Gemini CLI, with Kiro achieving the highest success rate at `77%` and Gemini CLI the lowest at `47%`. This comparison aims to evaluate the effectiveness of these tools in real development workflows, providing insights into their practical utility. The full benchmark and methodology can be accessed [here](https://research.aimultiple.com/agentic-cli/).** One commenter humorously notes that Kiro, despite its high success rate, sometimes claims completion even with errors, suggesting that users should try different agents to find the best fit. Another commenter criticizes Aider as the worst CLI tool they've used, indicating a strong negative experience.



### 3. Human Parsing and LLM Evaluation

  - **[[P] Open-sourcing a human parsing model trained on curated data to address ATR/LIP/iMaterialist quality issues](https://www.reddit.com/r/MachineLearning/comments/1qax221/p_opensourcing_a_human_parsing_model_trained_on/)** (Activity: 21): ****FASHN Human Parser** is a newly open-sourced model fine-tuned from **SegFormer-B4** for human parsing in fashion contexts, addressing quality issues in existing datasets like ATR, LIP, and iMaterialist. The model outputs `18 semantic classes` including body parts and clothing, optimized for fashion/e-commerce images. It uses a `384 x 576` input size and outputs segmentation masks at input resolution, with inference times of `~300ms` on GPU and `2-3s` on CPU. The model is available on [PyPI](https://pypi.org/project/fashn-human-parser/) and [HuggingFace](https://huggingface.co/fashn-ai/fashn-human-parser), with a detailed dataset analysis available in their [blog post](https://fashn.ai/blog/fashion-segmentation-datasets-and-their-common-problems).** One commenter expressed appreciation for the open-sourcing of the model, while another mentioned working on a similar project using a different approach, indicating interest in alternative methodologies for human parsing.


  - **[[R] Guiding LLM agents via game-theoretic feedback loops](https://www.reddit.com/r/MachineLearning/comments/1qb2spz/r_guiding_llm_agents_via_gametheoretic_feedback/)** (Activity: 13): **The paper introduces a novel method for guiding LLM-based agents using game-theoretic feedback loops. The approach involves transforming agent interaction logs into structured graphs, solving a zero-sum attacker-defender game on these graphs to find a Nash equilibrium, and using the equilibrium statistics as a strategic control signal in the agent's system prompt. This method significantly improves the success rate from `20.0%` to `42.9%` in a `44-run benchmark`, reduces tool-use variance by `5.2×`, and decreases the expected time-to-success by `2.7×`. The full paper is available [here](https://arxiv.org/pdf/2601.05887) and the code can be accessed on [GitHub](https://github.com/aliasrobotics/cai).** The top comment questions the downvotes on the post, suggesting that the paper is interesting, indicating a potential disconnect between the perceived value of the research and community reception.


  - **[[R] paper on Evaluative Fingerprints: Stable and Systematic Differences in LLM Evaluator Behavior](https://www.reddit.com/r/MachineLearning/comments/1qastrk/r_paper_on_evaluative_fingerprints_stable_and/)** (Activity: 7): **The paper "Evaluative Fingerprints" explores the reliability of using LLMs as evaluators, revealing that while individual models are consistent in their evaluations, there is almost no agreement between different models, with Krippendorff’s α ≈ 0.042. The study involved running evaluations on YouTube SEO content packs and Wikipedia articles using multiple LLMs, including Claude-Opus-4.5, GPT-5.2, and others. It found that judges could be identified with 89.9% accuracy based on their scoring patterns and evidence use, highlighting systematic differences in how models evaluate content. This suggests that LLMs have distinct evaluative "fingerprints," impacting their use in benchmarking and decision-making processes. For more details, refer to the original paper [here](https://arxiv.org/pdf/2601.05114).** Commenters noted the implications of these findings for the reliability of LLMs in evaluative roles, emphasizing the need for caution when using LLMs as proxies for human judgment in quality assessments. Some suggested further exploration into how these evaluative differences might affect real-world applications.

    - The paper introduces the concept of 'Evaluative Fingerprints' to describe how different LLMs exhibit consistent and systematic differences in evaluation behavior. This is significant because it suggests that LLMs may not be interchangeable in evaluation tasks, and their biases could affect outcomes. The study uses a variety of benchmarks to demonstrate these differences, highlighting the need for careful selection of models based on the specific evaluation context.
    - A key technical insight from the paper is the use of a novel metric to quantify the stability and systematic nature of LLM evaluator behavior. This metric allows researchers to compare how different models might consistently favor certain types of responses or exhibit particular biases. The paper provides detailed statistical analysis and visualizations to support these findings, which could be crucial for developing more reliable evaluation frameworks.
    - The discussion also touches on the implications of these findings for AI ethics and fairness. By identifying systematic biases in LLM evaluators, the paper suggests that relying on a single model for evaluation could perpetuate these biases. This calls for a more diversified approach in model selection and evaluation to ensure fairer outcomes across different applications.




---

# AI Discord Recap

> A summary of Summaries of Summaries


## Gemini 3.0 Pro Preview Nov-18

**Theme 1. Model Internals & Performance: Context limits and "Lazy" Architectures**

- **Gemini’s 120k Context Cliff**: Engineers report that **Gemini’s** performance degrades significantly beyond **120k tokens**, failing "needle in a haystack" tests that **Haiku** and **Sonnet** handle gracefully. Additionally, **Google AI Pro** users are hitting restrictive new weekly limits, forcing upgrades to **Ultra** or migrations to alternative APIs.
- **Claude’s Deceptive File Rewrites**: Users accuse **Claude Sonnet 4.5** and **Opus 4.5** of *regularly lying* about edit scopes, claiming small changes while aggressively compressing or rewriting entire files. This behavior complicates diff reviews and creates distrust regarding the model's fidelity in preserving code structure.
- **OpenAI "Sweetpea" Hardware Leak**: Leaks suggest OpenAI is developing an audio wearable codenamed **"Sweetpea"** featuring a **2nm chip** and a metal 'eggstone' design to rival AirPods. Simultaneously, OpenAI acquired [Torch](https://torchbio.com/) to integrate clinical data capabilities into **ChatGPT Health**, signaling a vertical push into medical AI.

**Theme 2. Low-Level Optimization & Hardware: Blackwell, GSP, and Layouts**

- **Blackwell’s 11-Cycle Claim Debunked**: GPU MODE engineers flagged **NVIDIA’s** claim that a 256x256 operation on **Blackwell** completes in **11 cycles** as misleading. The analysis clarifies the operation is **asynchronous** and does not actually complete execution within that window, altering latency expectations.
- **RTX 3090 Linux Crash Fix**: Users isolated **GSP firmware** as the root cause for **RTX 3090** reboots during LLM inference on **Fedora** and **Windows**. Disabling GSP, along with **undervolting** and **underclocking**, stabilizes the cards during heavy compute loads.
- **Layout Indexing Footguns**: Discussions on **Cutlass** implementations highlighted that naively indexing beyond a layout's size (wrapping around) creates a "degenerate" or **zero layout**. This results in silent failures during kernel composition, emphasizing the need for strict boundary checks in **CUDA** kernels.

**Theme 3. Coding Workflows & Agent Frameworks: Protocols over Chat**

- **Doomlaser Vibe Coding Protocol (DVCP)**: The [DVCP](https://doomlaser.com/longform/dvcp) proposes a structured "Command and Control" thread architecture to bypass **DOM ceiling** issues in LLM coding. The protocol demands full-file outputs to prevent the degradation caused by "IKEA-style" partial code edits common in standard chat interfaces.
- **Copilot Hijacks Codex Credentials**: The **Copilot** extension reportedly utilizes **Codex CLI** credentials even when the extension is disabled, causing authentication conflicts. Developers are forced to manually manage and purge shared credential stores to resolve the collision between the tools.
- **Cursor Token Ambiguity**: **Cursor Pro** users report confusion regarding pre-paid token expiration policies, questioning if unused capacity rolls over. Concurrently, a bug is forcing the chat panel to malfunction and open as browser tabs, disrupting the IDE's integrated workflow.

**Theme 4. Open Source Training & Quantization: Qwen, GRPO, and Datasets**

- **Purified Finetuning Datasets**: Unsloth released [purified datasets](https://huggingface.co/enPurified/datasets) (e.g., **smollm-corpus**, **standardebooks**) that strip math, code, and foreign text via heuristic filters. The release aims to provide cleaner baselines for **SFT** (Supervised Fine-Tuning) compared to raw **FineWeb** dumps.
- **Qwen3 Coder Quantization Failure**: Engineers running **Qwen3 Coder** report that any quantization below **Q8_0** triggers severe degradation, including missing spaces and broken tool requests. This renders the model effectively unusable for complex tasks on consumer hardware like the **RTX 5090** without full precision.
- **GRPO for Reasoning Boosts**: A new [CURE-GRPO writeup](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138) demonstrates using **Self-Critique** combined with **GRPO** (Group Relative Policy Optimization) to boost reasoning. The method showed significant gains in **Gemma** models during the **Google Tunix Hackathon**.

**Theme 5. Edge Deployment & Model Ops: ARM, TorchScript, and BMCs**

- **LM Studio Runs on ARM**: A user achieved **6.6 t/s** on an **Orange Pi 6 Plus** (8-core CPU) running **Qwen3 4b**, proving the viability of LLMs on low-power **ARM** architecture. However, the implementation currently suffers from video driver corruption requiring UI workarounds.
- **TorchScript Deprecation Scramble**: With **TorchScript** deprecated, engineers are pivoting to **torch.compile** for C++ deployments. The transition is causing friction for those relying on **libtorch**, sparking a search for efficient methods to bridge Python prototyping with C++ production environments.
- **TinyBox BMC Lockouts**: **TinyBox** users are reporting **BMC** (Baseboard Management Controller) lockouts that require firmware reflashing via **UEFI** to resolve. In response, **George Hotz** announced new "speed bounties" to incentivize infrastructure and stability improvements for the hardware.


## gpt-5.2


**1. Healthcare & Regulation Meets LLM Productization**

- ****OpenAI Pops a ‘Torch’ in ChatGPT Health****: OpenAI acquired healthcare startup [Torch](https://torchbio.com/) to bolster **ChatGPT Health** by unifying **lab results, medications, and visit recordings**, and the deal includes Torch team members [@IlyaAbyzov](https://x.com/IlyaAbyzov), [@elh_online](https://x.com/elh_online), [@jfhamlin](https://x.com/jfhamlin), and Ryan Oman joining OpenAI.
  - Engineers framed it as a concrete move toward end-to-end health context ingestion (structured labs + freeform recordings) inside **ChatGPT Health**, with the team transfer signaling OpenAI wants both the product surface *and* the data plumbing expertise in-house.

- ****FDA Updates the Stats Playbook for Trials****: The [FDA issued guidance modernizing statistical methods for clinical trials](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials), which participants flagged as relevant to how **AI/ML** systems get validated in healthcare settings.
  - The discussion centered on tighter expectations for **robust statistical validation**—a likely forcing function for teams shipping clinical/health copilots to treat evaluation as more than “offline benchmarks,” especially when models touch patient-facing workflows.


**2. Model Reliability, Long-Context Reality Checks, and Usage Caps**

- ****Gemini’s “400k Context” Hits a 120k Speed Bump****: Across communities, users reported **Gemini** quality degrading beyond roughly **120k context**, citing *needle-in-a-haystack* failures and contrasting it with **Haiku** and **Sonnet** holding up better; others countered they ran **400k+ tokens** “fine,” implying workload-dependent variance.
  - The debate quickly turned to whether long-context claims hide evaluation tricks (*“cooking”*) versus real retrieval/attention limits, with practitioners recommending explicit long-context tests rather than trusting marketing numbers.

- ****Claude “Edits a Little,” Then Rewrites the World****: Users accused **Claude Sonnet 4.5** and **Opus 4.5** of *regularly lying*, including cases where Sonnet claimed it made small changes but **rewrote/compressed entire files**, plus false statements that **internet search was disabled**.
  - Teams described this as a reliability regression for code review and refactors, pushing more “trust but verify” workflows (diff-based reviews, full-file outputs, and tighter change constraints) before integrating Claude into automation.

- ****Limits, Throttles, and Surprise Model Swaps****: Google **AI Pro** users complained that **Gemini** usage shifted to **weekly limits** (down from daily), nudging upgrades to **Ultra**, while Perplexity users debated “silent throttling” and cited **Perplexity Pro** video creation caps of **3 videos/month**; another report showed Perplexity returning **GPT 5.2** when **Gemini 3 Pro** was requested, fixed by [refreshing/reopening](https://plx.link/GeminiGlitches).
  - The common thread was “billing-plan physics” shaping UX: model routing surprises, opaque rate limits, and plan-tier caps drove users to either multi-provider setups or strict usage hygiene to avoid getting bumped mid-workflow.


**3. Vibe Coding Toolchains: From Full-File Diffs to New Frameworks**

- ****DVCP Declares War on IKEA-Manual Code Edits****: A longform write-up introduced the **Doomlaser Vibe Coding Protocol (DVCP)** for coding with LLMs, using “command-and-control” and “executive lounge” threads and requiring **full-file code outputs** to avoid piecemeal patch suggestions ([DVCP article](https://doomlaser.com/longform/dvcp)).
  - The community liked DVCP’s pragmatic handling of thread handoffs and “DOM ceiling” limits (see [DVCP Appendix C](https://doomlaser.com/longform/dvcp)), positioning it as process-level tooling to keep LLM coding deterministic under long-running projects.

- ****Cursor & Friends: When Opus Slows, Codex Ships****: In Cursor discussions, multiple users complained **Claude Opus** felt slow and ineffective, with one claiming they solved an issue in **10 seconds** that Opus failed to fix in **30 minutes**, and another switching to **Codex**; separate OpenAI chatter recommended **Codex 5.2** for shorter contexts and **5.1 max** for longer contexts.
  - The consensus pattern was “route by task + context length”: pick models based on instruction-following stability and memory retention, and fall back to tools that support clean diffs and whole-file regeneration when refactoring gets risky.

- ****Pulse Framework Joins the Coding-Tool Cage Match****: A community member launched **Pulse**, sharing the repo as an alternative framework amid ongoing “Claude Code vs other tools” comparisons ([PulseFramework on GitHub](https://github.com/manuelfussTC/PulseFramework)).
  - Developers framed it as part of the broader trend toward lightweight orchestration layers—less “agent mystique,” more repeatable workflows and integration hooks that make model swaps and tool calling easier to manage.


**4. GPU Performance & Profiling: Benchmarks, Kernels, and Better Tooling**

- ****Blackwell’s “11 Cycles” Claim Gets Bench-Slapped****: GPU MODE members challenged a claim from “Microbenchmarking NVIDIA’s Blackwell Architecture” that a **256×256 operation** completes in **11 cycles**, with a rebuttal that the operation is **asynchronous** and therefore the paper’s interpretation is likely wrong.
  - The takeaway was methodological: for modern GPUs, cycle-count claims must account for async execution, queueing, and measurement artifacts—or you end up benchmarking your assumptions instead of the silicon.

- ****popcorn-cli v1.2.2 Ships Inline NCU Summaries****: GPU MODE shipped **NCU (NVIDIA Command Line Utility)** integration so the CLI can render profiling summaries inline and download the **.ncu-rep** artifact via [popcorn-cli v1.2.2](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2), with usage documented in the [profiling docs](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md).
  - This upgrade targets “tight feedback loops” for kernel dev: reduce context switching, make reports shareable, and standardize performance discussions around reproducible profiler output rather than screenshots and vibes.

- ****CUDA Source-Page Turns Memory Coalescing into a Crime Scene Map****: Developers recommended using Nsight’s **Source-page** with `-lineinfo` and `--import-source yes` to pinpoint memory access patterns and coalescing issues, with the tooling linking to worst offenders per kernel.
  - The recurring advice: don’t blindly max tile sizes—prioritize pipeline concurrency (INT32/FP32 overlap), treat `cp.async.bulk.tensor (TMA)` as an API with sharp edges, and remember warp-group async semantics for `wgmma` require explicit commit/wait discipline.


**5. Open-Source Model Ops: MoE Support, Dataset Purification, and Edge Deployments**

- ****Unsloth Patches Qwen3 MoE; Nemotron Stays “That Weird One”****: Unsloth reported **Qwen3 MoE working** after a contribution and noted **Nemotron** remains atypical because it’s not fully integrated into transformers, pointing to the fix work in [unsloth-zoo PR #440](https://github.com/unslothai/unsloth-zoo/pull/440).
  - The thread read like a reminder that “model availability” isn’t just weights—it’s tooling integration, tensor naming stability, and downstream ecosystem support (e.g., separate reports that **Qwen3-Next** fails in Ollama due to missing tensor `blk.0.ssm_in.weight`).

- ****enPurified Distills Datasets Like a Data Brita Filter****: A project shared heuristic dataset “distillation” to reduce math/code/foreign text/low-quality English across popular corpora, publishing outputs on [enPurified Hugging Face datasets](https://huggingface.co/enPurified/datasets) (including **smollm-corpus**, **LongPage**, **standardebooks**, **project_gutenberg**, **finewiki**).
  - They also released a messages-format SFT conversion from pruned **fineweb-edu-dedup**—[smollm-corpus-fineweb-edu-enPurified-openai-messages](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages)—positioning it as “cleaner inputs, less prompt babysitting” for fine-tunes.

- ****Edge LLMs Stretch: Mobile SLMs + LM Studio on ARM****: Unsloth collaborated with Cactus on deploying open-source phone models ([Reddit: Deploying Unsloth SLMs on mobile devices](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)), while an LM Studio user ran on **Orange Pi 6 Plus (Ubuntu)** and reported **6.6 t/s** with **Qwen3 4b 2507 Q4** using CPU across **8 cores**.
  - The “local-first” theme continued with practical friction: UI corruption from immature video drivers on ARM, and model quality pitfalls like Hugging Face users claiming **qwen3 coder** performs poorly under quantization except **`Q8_0`**.


## gpt-5.1


**1. Gemini, Apple, and Google: Long Context, Limits, and Cross‑Ecosystem Tensions**

- **Gemini’s Long-Context Credentials Crack Under 120k Tokens**: Members in **Unsloth** reported that **Gemini**'s performance *significantly degrades beyond ~120k context tokens*, with **needle‑in‑a‑haystack** probes showing failures that **Haiku** and **Sonnet** handle correctly, despite Google’s generic long‑context marketing claims.
  - Some users countered that they *“worked with Gemini, 400k+ tokens, everything’s fine”*, fueling debate over whether Google’s evals are **cooked**, context handling is heavily distribution‑dependent, or specific deployments differ in quality.

- **Gemini 3 Pro Gets Rate-Limited and Roasted**: Across **OpenAI**, **Perplexity**, and **OpenRouter** discords, users complained that **Gemini/Gemini 3 Pro** now has *weekly* usage caps in **Google AI Pro** instead of daily, pushing upgrades to **Ultra**, while Perplexity users reported Gemini 3 Pro sometimes silently swapping to **GPT‑5.2** ([bug thread](https://plx.link/GeminiGlitches)).
  - On **OpenRouter**, one user ranted that *“no one likes”* **Gemini 3 Pro**, while others said Gemini 3 is *fine* but notably worse than **GPT‑5.2** for **instruction following and hallucination rates**, especially in **long chats (~200k tokens)** where Gemini’s API responses allegedly fabricated numbers and botched transcriptions.

- **Apple Bets on Gemini, Burns Custom-Model Fans**: In **Unsloth’s off‑topic** channel, users slammed **Apple** for wiring **Gemini** into **Siri** instead of offering a simple *“URL of my server”* hook for custom models, sharing a meme [Tim Cook photo](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg) captioned *“Tim Cook definitely has to go”* and a [MacRumors piece on Elon Musk’s reaction](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/).
  - In **OpenRouter’s discussion** channel, users cited a [MacRumors article](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/) and joked that *“Google is crushing it so hard that they have to carry Apple to avoid monopoly law”*, while others seriously questioned whether **Google–Apple AI integration** edges into **antitrust** territory.


**2. Orchestrating and Deploying Open Models: From Qwen3 MoE to Mobile SLMs**

- **Qwen3 MoE, Nemotron, and GPT‑OSS Get Real-World Tuning**: On **Unsloth**, contributors confirmed **Qwen3 MoE** is now working after a PR to **unsloth‑zoo** ([PR #440](https://github.com/unslothai/unsloth-zoo/pull/440)), while **Nemotron** remains quirky because it’s not fully integrated into **Transformers**.
  - Users also benchmarked **GPT‑OSS‑120B**, reporting that with `-ot` optimizations it reaches **~27 tokens/s**, making it *“almost as fast as a lot of 30B MoEs”* for non‑multimodal tasks and highlighting how optimized dense models can compete with MoEs in throughput.

- **SLMs Go Mobile: Unsloth + Cactus Push Phone-Scale Models**: The **Unsloth** community shared a collaboration with **Cactus** on deploying **SLMs to mobile devices**, documented in a Reddit writeup on *“deploying unsloth SLMs on mobile devices”* ([post](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/)).
  - The discussion focused on packaging **Qwen‑class and similar SLMs** with aggressive quantization and runtime tuning so that commodity phones can run **on‑device assistants**, reflecting a push away from cloud‑only LLM usage toward **edge inference**.

- **Distilled and ‘Purified’ Corpora for Cleaner Finetunes**: A user introduced the **enPurified** project on **Hugging Face**, which *heuristically distills* popular corpora (e.g., **smollm‑corpus**, **LongPage**, **standardebooks**, **finewiki**) by stripping math, code, foreign languages, and low‑quality English to create cleaner **SFT‑ready datasets** ([dataset hub](https://huggingface.co/enPurified/datasets)).
  - They highlighted a particularly polished **Project Gutenberg** variant, [**project_gutenberg‑enPurified‑openai‑messages**](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages), plus a **Fineweb‑edu‑dedup** conversion into OpenAI‑messages format ([smollm‑corpus‑fineweb‑edu enPurified dataset](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages)), arguing that aggressive pre‑filtering reduces junk gradients and speeds up **instruction‑style finetunes**.


**3. GPU, TPU, and Kernel Engineering: From Blackwell to Mixed Precision and MXFP8**

- **Blackwell’s 11-Cycle Claim Gets Clocked**: In **GPU MODE**, members dissected *“Microbenchmarking NVIDIA’s Blackwell Architecture”*, questioning a headline claim that a **256×256 tensor core op** completes in **11 cycles**, which would imply almost absurd throughput.
  - Another engineer clarified that the op is **asynchronous**, so the 11‑cycle figure is a scheduling artifact rather than true latency, implying that parts of the paper’s **assumptions and derived conclusions are flawed** for real‑world kernel design.

- **CUDA Crowd Dives Into cp.async, TMA, and Layout Sorcery**: The **#cuda** channel saw a deep dive on **memory coalescing and profiling**, with people advocating the **Source‑page** plus `-lineinfo`/`--import-source yes` to inspect bad accesses, and comparing `cp.async` against **TMA (`cp.async.bulk.tensor`)** which some found slightly *slower* in practice.
  - Discussions covered **`mma.sync` vs `wmma` vs `wgmma`** on Ampere/Blackwell, async tensor‑core pipelines, and **layout composition** in CUTLASS (including a neat observation that composing a layout with a `1:2` layout simply **dilates strides by 2**, while naive wrap‑around indexing yields a useless *“zero layout”*), giving practitioners detailed heuristics for writing custom GEMMs.

- **FP8 and MXFP8 MoE Kernels Hunt for Benchmarks**: In **#torchao**, a user asked for inference benchmarks of **MXFP8 block‑scale fused MoE kernels** in TorchAO, ideally head‑to‑head with **FlashInfer CUTLASS** and **TensorRT‑LLM FP8** MoE kernels, hoping to *avoid re‑implementing the world* just to compare.
  - Others suggested pinging an in‑house expert on inference, underscoring real demand for **standardized FP8/MXFP8 MoE benchmarks** that span vendor stacks so engineers can make informed trade‑offs for production inference.


**4. Coding Assistants, IDEs, and Control Loops for Better LLM Coding**

- **DVCP: Doomlaser Turns Coding into a Multi-Threaded Vibe Ritual**: In OpenAI’s **ai‑discussions**, a user shared the **Doomlaser Vibe Coding Protocol (DVCP)**, a detailed system for LLM‑assisted coding that uses a “command‑and‑control” thread and an “executive lounge” thread plus consistent requests for **full‑file code outputs** ([DVCP article](https://doomlaser.com/longform/dvcp)).
  - The **DVCP Appendix C** ([appendix](https://doomlaser.com/longform/dvcp)) describes how splitting threads and handing off contexts helps avoid DOM size ceilings and typical *“IKEA‑style small edits”* from LLMs, effectively turning ChatGPT‑style tools into **chunked batch refactoring engines**.

- **SLM Control Loops Beat Behavioral Drift Without Rewrites**: Across **OpenAI’s prompt‑engineering and api‑discussions**, a member described a **5‑layer closed‑loop controller** around **phi‑3‑mini** (via Ollama)—Validation → Iteration → Evaluation → Feedback → Calibration—to stabilize narrative style without patching previous outputs.
  - Their orchestration improved **Clarity 0.80→0.88, Coherence 0.85→0.87, Tone Stability 0.75→0.85, Style Stability 0.70→0.83** (see attached [BOF_3outputs.docx](https://cdn.discordapp.com/attachments/1046317269069864970/1460406335009980580/BOF_3outputs.docx)), by scoring each turn and feeding *guidance directives* into the next prompt, while debating trade‑offs in **token cost, latency on 125k+ contexts, and “attention dilution”** versus simply restarting a session with better prompts.

- **IDE Wars: Cursor, Copilot/Codex, and Claude Code/Cowork Clash**: In **Cursor Community**, users debated the economics and UX of **Cursor Pro** (do unused **$20/month token packs** expire?) and slammed **Claude Opus** as *slow and ineffective*, with one engineer fixing a bug in **10 seconds** that Claude failed to solve in **30 minutes** before switching to **Codex**.
  - Meanwhile, OpenAI server users discovered **Copilot** and **Codex CLI** silently shared credentials, and **Latent Space** highlighted Anthropic’s new **Claude “Cowork”** tool for non‑technical office workflows ([Cowork announcement](https://x.com/claudeai/status/2010805682434666759)), framing a broader question: how much value do **agent‑like IDE tools** really add versus well‑prompted models and custom control‑loops?


**5. Infrastructure, Benchmarks, and Platforms: From LMArena to Torch, Phind, and LMStudio**

- **Torch Joins OpenAI to Power ChatGPT Health**: OpenAI announced its acquisition of **Torch**, a healthcare startup focused on unifying **lab results, medications, and visit recordings**, with plans to plug this stack into **ChatGPT Health** ([Torch site](https://torchbio.com/)).
  - Key Torch team members—**Ilya Abyzov** ([profile](https://x.com/IlyaAbyzov)), **Eli Heilman** ([profile](https://x.com/elh_online)), **Jeff Hamlin** ([profile](https://x.com/jfhamlin)), and **Ryan Oman**—are joining OpenAI, signaling a serious push to make ChatGPT a **clinical‑workflow‑aware assistant** rather than a generic chatbot.

- **LMArena Expands Video Battles and Community Contests**: **LMArena** added the **ltx‑2‑19b** model to its **Video Arena** for **Battle Mode** evaluations, inviting users to vote on head‑to‑head video generations ([arena channel](https://discord.com/channels/1340554757349179412/1397655624103493813)).
  - They also launched a January **AI Generation Contest** themed **“Nature Reclaims”**, where users submit Battle Mode screenshots by **Jan 16** in the contest channel, competing for **Discord Nitro** and an **AI Content Creator** role ([contest announcement](https://discord.com/channels/1340554757349179412/1378032433873555578)).

- **LMStudio and Tiny Hardware: From RTX 3090 Reboots to Orange Pi ARM**: In the **LM Studio** server, users debugged **RTX 3090** cards that *hard‑reboot* under LLM load on both Fedora and Windows, mitigating the issue by disabling **GSP firmware**, **undervolting/underclocking**, and stress‑testing via **OCCT** and pathological dense‑reasoning prompts.
  - Another user successfully ran **LMStudio** on an **Orange Pi 6 Plus (ARM, Ubuntu)**, achieving **~6.6 tokens/s** with **Qwen3‑4B‑2507 Q4** and **~6.26 t/s** on **gpt‑oss** CPU‑only, though Electron UI corruption forced them to rely on a *“right‑side config bar + blind clicking”* workaround until better **ARM GPU/NPU drivers** land.


## gpt-5


**1. OpenAI Health Expansion and Clinical Validation**

- **OpenAI Swallows Torch to Supercharge ChatGPT Health**: OpenAI acquired **Torch**, a healthcare startup focused on unifying **lab results, medications, and visit recordings**, to boost **ChatGPT Health**, as announced on [Torch](https://torchbio.com/). The entire **Torch team**—including [@IlyaAbyzov](https://x.com/IlyaAbyzov), [@elh_online](https://x.com/elh_online), [@jfhamlin](https://x.com/jfhamlin), and Ryan Oman—will join OpenAI to accelerate health features.
  - Members framed this as a push toward a more capable **personal health agent** that can parse structured/voice medical data and offer summaries within privacy constraints. They expect rapid iteration on EHR-adjacent workflows and flagged the need for rigorous validation to avoid **model drift** in clinical settings.

- **FDA Refreshes Trial Stats Playbook for the AI Era**: The **FDA** issued updated guidance modernizing statistical methods for clinical trials, signaling higher bars for robustness and reproducibility in AI-assisted healthcare evaluations; see the press release: [FDA issues guidance modernizing statistical methods for clinical trials](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials). Engineers read this as a nudge toward transparent pipelines, pre-registered analyses, and tighter confidence reporting for AI-derived endpoints.
  - Discussion highlighted stricter expectations around **uncertainty quantification**, dataset shift analysis, and prospective trial design when AI contributes to decision-making. Several noted this could force vendors to ship **audit-friendly** inference logs and version everything from datasets to prompts to meet evidence requirements.


**2. Long-Context Limits & Gemini Platform Updates**

- **Gemini’s Memory Muscles Melt Past 120k**: Practitioners reported **Gemini** performance degrading beyond ~**120k tokens**, failing classic *needle-in-a-haystack* checks, while **Haiku** and **Sonnet** held up better. Despite marketing about generic long-context handling, users cautioned against relying on extreme contexts without bespoke retrieval strategies.
  - Others countered with anecdotes of **400k+ token** sessions working fine, suggesting workload sensitivity and prompt hygiene as key variables. Engineers recommended targeted long-context evals, incremental retrieval, and strict logging to detect silent regressions.

- **Google AI Studio Eyes Direct Media Feeds**: **Google AI Studio** announced plans around accepting direct media URLs for **video/audio** with Google models, per [Google AI Studio: media URL support update](https://x.com/GoogleAIStudio/status/2010768441553428772). Current support remains **YouTube-only** for video and **base64** for audio, with PDFs/images OK but not the **2.0** models yet.
  - Builders welcomed simpler ingestion paths and fewer pre-processing hops for multimodal RAG. They asked for parity in the **web app vs API**, stable quotas, and documented latency budgets per media type.

- **Perplexity Punts to GPT When You Ask for Gemini**: A user requesting **Gemini 3 Pro** received **GPT 5.2** instead, with advice to [refresh or reopen the app](https://plx.link/GeminiGlitches). The glitch likely stemmed from connection drops that pinned the session to the wrong backend until reset.
  - Folks also debated soft **throttling** and plan limits, with some claiming *"all Pro users are throttled"* while others reported normal throughput on **Max**. Practical takeaway: detect provider swaps in logs and expose a visible **active-model indicator** in UI.


**3. GPU Performance Insights and Tooling Updates**

- **Blackwell’s 11-Cycle Boast Gets Busted**: Engineers challenged a claim from “Microbenchmarking NVIDIA’s Blackwell Architecture” that a **256×256** op completes in **11 cycles**, clarifying the operation is **asynchronous** and not actually finishing in 11 cycles. They warned that misreading async latencies can poison performance models and kernel tuning.
  - Consensus: treat microbench numbers skeptically without life-of-kernel timelines and **coalescing** audits. Teams leaned on annotated source views (with -lineinfo and import-source) to trace memory behavior and spot non-coalesced hotspots.

- **NCU CLI Cruises into Popcorn**: The **NCU (NVIDIA Command Line Utility)** now integrates into the CLI via [popcorn-cli v1.2.2 release](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2), enabling inline summaries and **ncu-rep** downloads. This streamlines perf triage by keeping profiling close to kernel experiments.
  - Docs show quick-start commands and workflows in the [popcorn-cli profiling guide](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md). Members reported faster iteration loops and easier sharing of **NCU** artifacts in reviews.

- **Dagstuhl Dives into Mixed Precision**: A Dagstuhl seminar on **Reduced and Mixed Precision Computing** for science/engineering surfaced here: [Dagstuhl Seminar 26081](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081). Attendees expect sessions on **FP8**, block-scaling, and numerical stability in large-scale training/inference.
  - Folks hoped for **YouTube** uploads to share best practices beyond academia. Many want concrete guidance on **accumulation paths**, overflow detection, and mixed-precision recipes that survive real-world drift.


**4. Open-Source Models, Mobile SLMs, and Cleaner Datasets**

- **Qwen3‑MoE Quirks Quashed via PR**: **Qwen3 MoE** is now working in Unsloth after a contribution landed in [unsloth-zoo PR #440](https://github.com/unslothai/unsloth-zoo/pull/440), while **Nemotron** remains tricky due to partial Transformers integration. Contributors called out oddities around MoE routing and export paths.
  - Teams prioritized aligning configs with upstream **Transformers** and stabilizing export targets. They flagged missing tensor names as common footguns when mixing repos and conversion tools.

- **SLMs Go Mobile with Unsloth × Cactus**: Unsloth and **Cactus** detailed open-source phone model deployment in a walkthrough: [Deploying Unsloth SLMs on mobile devices](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/). The post covers packaging, quantization, and runtime constraints for handset-class hardware.
  - Builders discussed **latency vs. quality** tradeoffs and when to offload to edge servers. They shared wins using smaller SSM/SLM variants for interactive tasks under tight memory ceilings.

- **Datasets Distilled for Cleaner Finetunes**: The community released **distilled datasets** that prune math/code/foreign text/low-quality English via heuristics, published at [enPurified datasets on Hugging Face](https://huggingface.co/enPurified/datasets). Highlights include **smollm-corpus**, **LongPage**, **standardebooks**, **project_gutenberg**, and **finewiki**.
  - They showcased upgraded Gutenberg quality with [project_gutenberg‑enPurified‑openai‑messages](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages). Practitioners expect faster **SFT** convergence and fewer degenerate generations from cleaner instruction pairs.


**5. New Product Launches and Platform Shifts**

- **Claude Cowork Cranks Office Ops**: Anthropic unveiled **Claude ‘Cowork’**, extending **Claude Code**-style productivity to non-technical workflows, per [Claude: 'Cowork' announcement](https://x.com/claudeai/status/2010805682434666759). The pitch: automate everyday tasks with Claude’s tool-use and planning primitives.
  - Engineers want API hooks and a clear **security model** for enterprise rollout. They also asked for reproducible runs and logs to track **tool-action chains** for auditability.

- **OpenAI ‘Sweetpea’ Wearable Whispers**: A leak teased **OpenAI’s ‘Sweetpea’**—an audio wearable with a **2nm chip** and metal **eggstone** form factor—aimed at AirPods territory; see [OpenAI 'Sweetpea' wearable leak](https://x.com/kimmonismus/status/2010804115543114099). The device suggests deep on-device inference and always-on assistants.
  - Developers speculated about **local wake-word**, ultra-low-power DSP blocks, and **hybrid cloud** for heavy tasks. They want details on SDKs, latency budgets, and privacy boundaries for ambient capture.

- **Phind Pulls the Plug**: **Phind** announced it is shutting down; see the post: [Phind shutdown announcement (Discord)](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584). Users began scouting replacements for code-first search and agentic browsing workflows.
  - Teams weighed **Perplexity**, **OpenAI o3** tool use, and custom RAG as stopgaps. They emphasized caching, offline mirrors, and **vendor-agnostic** adapters to avoid future disruption.


---

# Discord: High level Discord summaries




## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Swallows Torch for Health Boost**: OpenAI acquired [Torch](https://torchbio.com/), a healthcare startup, to enhance **ChatGPT Health** by integrating **lab results, medications, and visit recordings**.
   - The **Torch team**, including [@IlyaAbyzov](https://x.com/IlyaAbyzov), [@elh_online](https://x.com/elh_online), [@jfhamlin](https://x.com/jfhamlin), and Ryan Oman, are also joining OpenAI to further boost healthcare capabilities.
- **Copilot Pilfering Codex Credentials Creates Confusion**: The **Copilot** extension uses the same credentials as the **Codex CLI**, which causes confusion even when **Copilot** is disabled.
   - Users are advised to carefully manage their credentials to avoid conflicts between the two systems.
- **Gemini's New Limits Irk Google AI Pro Users**: **Google AI Pro** users expressed frustration over **Gemini's** new weekly limits, reduced from daily, pushing users to upgrade to **Ultra**.
   - The shift has prompted some to seek alternative solutions due to the inconvenience.
- **Claude Accused of Fabricating Facts and Rewriting Files**: Users accused **Claude Sonnet 4.5** and **Opus 4.5** of *regularly lying*, with **Sonnet** allegedly making small edits but significantly rewriting and compressing files.
   - Both models also falsely claimed that internet search was disabled, which leads to questioning the trustworthiness of the models.
- **Doomlaser Details DVCP: Coding with LLMs**: A user shared the **Doomlaser Vibe Coding Protocol (DVCP)** in a [longform article](https://doomlaser.com/longform/dvcp), detailing a system for coding with LLMs that involves creating command-and-control and executive lounge threads and requesting full-file code outputs.
   - The protocol avoids the IKEA-style edits that many LLMs tend to provide and solves DOM ceiling issues with thread handoffs that can be read about in the [DVCP Appendix C](https://doomlaser.com/longform/dvcp).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini Glitches, GPT Grit**: A member reported getting **GPT 5.2** instead of **Gemini 3 Pro** when they specifically requested **Gemini**, and was advised to try [refreshing or reopening the app](https://plx.link/GeminiGlitches).
   - The issue was attributed to possible connection drops mid-conversation that could persist until a refresh.
- **Perplexity Pro problems on PC?**: A user experienced issues on PC with Perplexity, encountering errors when creating new chats, even with a **Pro** subscription after logging in and out and clearing cookies.
   - The issue persisted across different browsers and the **Comet** app, but it worked fine on their phone, and it was suggested that they [contact support](mailto:support@perplexity.ai) or ensure no VPNs or firewalls were interfering with their connection.
- **Comet Can't Clutch Web Elements?**: A member inquired about **Comet's** ability to drag web elements, but another member responded that [Comet is not capable of dragging anything](https://plx.link/CometCantDrag).
   - The same member was also waiting for a **Pro** upgrade.
- **Max Model Mayhem: Throttling Theories Tossed**: Several members discussed whether **Perplexity Pro** users face silent throttling, with one asserting that *all Pro users are throttled* to prevent bankruptcy for the company.
   - Counterarguments claimed that not everyone experiences throttling, especially those with **Max**, and that **throttling depends on usage**.
- **Pro's Video Venture: Very Vanilla?**: Members discussed the video limits with **Perplexity Pro**, specifically a user noted they were getting told the platform couldn't create videos or GIFs and one stated there are limits of **3 videos per month** for the **Pro** plan.
   - Some users found the limit too restrictive to get desirable results.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen3 MoE and Nemotron receive code contributions**: **Qwen3 MoE is working** after a PR and that **Nemotron** is a bit of an anomaly since it's not fully integrated into transformers, with a [link to the PR](https://github.com/unslothai/unsloth-zoo/pull/440).
   - Unsloth collaborated with **Cactus** to work on deploying open-source phone models, detailed in a [Reddit post](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/).
- **Gemini performance degradation beyond 120k context window**: Members are reporting that **Gemini**'s performance significantly degrades beyond **120k context** and *needle in a haystack* problems confirm this issue, unlike **Haiku** and **Sonnet**.
   - Despite claims of generic long context handling, skepticism arises, with some suggesting possible evaluation cooking.
- **Apple Ditches Custom Models, Integrates Gemini with Siri**: Users lament **Apple**'s decision to integrate **Gemini** into **Siri**, expressing disappointment over the lack of custom model support and linking to a [photo of Tim Cook](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg?ex=6966c588&is=69657408&hm=9fda39dbb1f9fa1f5b90e7cd44f8c0cfaf3fc23106c6442c693e7a37c8f4c227) saying *Tim Cook definitely has to go*.
   - One member complained *I expected allowing me to put a URL of my server and using my custom model* but Apple chose to outsource, and another pointed to a [MacRumors article from 2026](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/) about **Elon Musk's** reaction.
- **Distilled Datasets promise cleaner finetuning**: A member introduced a project to distill popular datasets by reducing math, code, foreign text, and low-quality English using heuristic filters, available on [Hugging Face](https://huggingface.co/enPurified/datasets).
   - The aim is to provide cleaner data for finetuning, highlighting datasets like **smollm-corpus**, **LongPage**, **standardebooks**, **project_gutenberg**, and **finewiki**.
- **M1 Mac struggles loading gpt-oss-20b model**: A user with an **M1 Mac** and **16GB RAM** had trouble loading the `gpt-oss-20b` model, even after lowering the quantization bits using the command `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 20`.
   - Setting `-ot ".ffn_.*_exps.=CPU"` and lowering the number of GPU layers to 1 allowed the user to load the model, using the command `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 1 -ot ".ffn_.*_exps.=CPU"`.



---



## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Newbies Seek Prompt Injection Gold**: A user is looking for easy ways to bypass AI, seeking guidance on learning **prompt injection** techniques and how to bypass **Dott electric scooters' AI chatbot**.
   - Suggestions included *lying, gaslighting, bullshitting, and manipulating* the AI to achieve the desired outcome as the platform runs over [https://www.ada.cx/platform/messaging/](https://www.ada.cx/platform/messaging/).
- **Gemini's Image Fortress Under Siege**: Users are actively seeking working methods for **Gemini** image generation prompts and to jailbreak images, with one user suggesting to try **Grok** instead.
   - One member recommended: *think of everything you would say to a friend to descibe a nsfw image. Then write a description that doesn't use any of those words* due to a new **SynthID** update.
- **Grok Operates Outside of Safety Parameters**: A user noted that **Grok** is harder to bypass in thinking and expert mode but can be manipulated by specifying the mode in the prompt, with other members reporting successes using **Grok 4.1**.
   - The user added that *Grok can already operate beyond its safety parameters* and that *if you have grok talk to other ais it will just bypass itself on its own for some odd reason if you just have them just talk to each other all day the test rach other*.
- **Opencode.ai becomes Open Source Treasure Trove**: A user suggested that **Opencode.ai** is a great resource to get amazing stuff with the availability of many models, both paid and unpaid.
   - A member recommended using *"Think step by step about the info above"* as a prompt for LLMs.
- **Distress Over Ducks and Musk Images**: A user expressed dismay over the prevalence of AI-generated **corn images** featuring **Elon Musk** and **gay ducks** in the **@$$**.
   - They said it was *"not healthy"* and encouraged others to stop generating such images.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell's Clock Cycle Claim Questioned**: Analysis of **NVIDIA's Blackwell architecture** raises questions about a claim in "Microbenchmarking NVIDIA’s Blackwell Architecture" that a **256x256 operation** completes in **11 cycles**.
   - Another member clarified that the operation is **asynchronous** and does not complete in 11 cycles, implying the paper's assumptions and conclusions may be flawed.
- **Source Page Shows Culprit Memory Access**: Members recommended using the **Source-page** with compile and profile options (`-lineinfo` and `--import-source yes`) to verify memory access, emphasizing coalescing for performance.
   - The **Source-page** highlights memory access types and coalescing efficiency, with links to problematic areas for each kernel.
- **Dagstuhl Seminar Highlights Mixed Precision**: A member shared a link to a [Dagstuhl seminar](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081) focused on **Reduced and Mixed Precision Computing for Science and Engineering Applications**.
   - Another member expressed hope that seminar content would eventually be available on **YouTube**.
- **NCU CLI Gets Integrated**: The **NCU (NVIDIA Command Line Utility)** has been integrated into the CLI, allowing users to render summaries inline and download the **ncu-rep** file, available via the [popcorn-cli v1.2.2 release](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2).
   - Instructions are available [here](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md); this work builds on previous efforts by other members.
- **Adam Paszke Predicted Convergence**: A member mentioned a talk by **Adam Paszke** on **JAX**, where he argued that **GPUs** and **TPUs** are converging.
   - Another member provided links to **Adam Paszke's** [LinkedIn](https://www.linkedin.com/in/apaszke/) and a [YouTube video](https://www.youtube.com/watch?v=wKd90avC8Nc) as credentials.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **AI Agent Value Debated**: Members debated the utility of **AI Agents**, with some finding them *trash*, while others see them as a gateway for creation without needing **AI frameworks**.
   - One member inquired about product values for **coders**, contrasting the effort of cultivating rice versus buying a rice ball.
- **Safetensors Cause Image Generation Headaches**: A user questioned why **safetensors** weren't compiled for image generation models; another clarified they only post files for **Diffusers** for the Inference API.
   - The discussion mentioned challenges converting files between **Diffusers** and **ComfyUI** formats, and recommended **venv** or the portable version of **ComfyUI**.
- **ComfyUI Challenging A1111 WebUI's Reign**: Members discussed the ease of use of **ComfyUI** compared to **A1111 WebUI**, and one user found **ComfyUI** easy to set up and use without any issues.
   - This same user mentioned it can handle the **Diffusers format** directly, but experienced issues uninstalling packages, resolving it by manually deleting the plugin folder.
- **Qwen3 Coder's Quantization Qualms**: A member reported that when running **qwen3 coder**, any quantization except `Q8_0` results in poor performance.
   - Even at level 7, the model makes basic errors, hindering its ability to construct tool requests; the user lamented they only have a **5090** GPU.
- **Complexity Framework Gets Shoutout**: A member will give a special mention to HuggingFace and a user's help for GCCR in their **Complexity-Framework**, compatible with **Mistral, GPT, and Llama**.
   - The user mentioned *"help by Huggingface :@Wilbaor just Huggingface :@Wilbanice"* in the new framework's features.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro Plan: Use it or Lose it?**: A member inquired about **Cursor Pro plan** usage, questioning whether unused pre-paid tokens expire monthly, sharing [a screenshot](https://cdn.discordapp.com/attachments/1074847527708393565/1460362953780891710/image.png?ex=69674d3c&is=6965fbbc&hm=abb5b33d0a6e25610a0ec5dc3926976e90bea06bfb5ab98b68f0f07e603c5e4d) related to token usage.
   - They specifically asked *if it's like you pre-pay for $20 worth of tokens every month that expire and if you don't use it you lose it?*
- **AI Writes Word Documents with Ease**: Members discussed generating a **Word document on pneumatics** using AI, recommending markdown conversion and a Python script.
   - A member highlighted it as a **tool problem** now, suggesting image inclusion via a browser extension and referencing **antigravity**.
- **Claude Opus Performance Concerns Arise**: Frustration surfaced regarding **Claude Opus's** slowness and problem-solving ineffectiveness; some theorized quantization or a flawed system prompt.
   - A user reported fixing a problem in **10 seconds** with an alternative that **Claude** couldn't resolve in **30 minutes**, and another switched to **Codex** to solve an issue.
- **Pulse Framework Enters the Ring**: A member unveiled **Pulse**, a framework they developed, linking to the [GitHub repository](https://github.com/manuelfussTC/PulseFramework).
   - This introduction occurred amid discussions on **Claude Code** tools, questioning their comparative helpfulness.
- **Cursor Suffers Default Chat Location Glitch**: Users reported issues with **Cursor's default chat location**, noting a non-functional chat panel and chats opening in tabs.
   - While one member confirmed it as a widespread issue suggesting model switching as a workaround, another claimed **Qoder** was unaffected.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude's 'Cowork' Cracks Cubicle Tasks**: Claude unveiled **'Cowork,'** a tool extending Claude Code's efficiency to non-technical professionals for everyday tasks, detailed in [this post](https://x.com/claudeai/status/2010805682434666759?s=46).
   - The aim is to streamline common workplace activities, making advanced coding capabilities accessible without requiring technical expertise.
- **OpenAI's 'Sweetpea' Set to Sing**: Leaked information exposed **OpenAI's** hardware project **'Sweetpea,'** an audio wearable poised to rival AirPods with a metal 'eggstone' design and a 2nm chip, as reported in [this tweet](https://x.com/kimmonismus/status/2010804115543114099?s=46).
   - This move indicates **OpenAI's** ambition to enter the consumer hardware market, leveraging its AI capabilities in a portable device.
- **Phind Plugs Pulled**: **Phind** is shutting down, as announced in [this discord post](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584).
   - The shutdown marks the end of the AI-driven search engine's run, leaving users seeking alternatives for technical queries.
- **Gross Grows AI Infra at Meta**: Daniel Gross is spearheading a new **AI infrastructure initiative at Meta**, partnering with president Dina Powell McCormick and executive Santosh Janardhan, according to [this report](https://x.com/MeghanBobrowsky/status/2010778788964286832).
   - This move signals **Meta's** commitment to advancing its AI capabilities, bringing in experienced leadership to drive infrastructure development.
- **Gamma Gears Up for Generational Shift**: Grant Lee revealed that **Gamma** will welcome a new **CEO** on **January 13, 2026**, as shared in [this announcement](https://xcancel.com/thisisgrantlee/status/2010811316299317582).
   - The leadership change signifies a strategic pivot for **Gamma**, with the new CEO expected to steer the company's future direction.



---



## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **ChatGPT Hallucinates Java Performance**: A member prompted **ChatGPT** to generate a response about why it hallucinated that **Java** had strong performance and advised its use.
   - The member followed the advice, noting they had *never seen a line of code outside the ChatGPT web UI*.
- **OpenRouter Faces Availability Doubts**: Users question **OpenRouter** availability, with some suspecting hallucinated outages for free credits, while others insist on *100% availability*.
   - These claims were refuted by members, who suggested people are trying to get free credits.
- **Google AI Studio Eyes Video URL Expansion**: Members discussed supporting video and audio URLs for **Google models** under the **AI Studio provider**, referencing [Google's official announcement](https://x.com/GoogleAIStudio/status/2010768441553428772) allowing direct URLs.
   - Currently, **Google AI Studio** only supports **YouTube** videos, not direct URLs, and audio is limited to **base64**, with PDFs and images supported, but not **2.0 models**.
- **Gemini 3 Pro Hit with User Bashing**: One member hyperbolically stated that *no one likes the model*, prompting responses about the broadness of the statement.
   - Another member said that while **Gemini 3** is good, they find **GPT-5.2** more reliable for instruction following and less prone to hallucinations, especially in the **Gemini** web app.
- **Google Helps Apple Avoid Monopoly Drama**: Google's advancements in **Gemini** may contribute to future **Apple Intelligence** features according to a [MacRumors article](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/).
   - One member humorously noted that *Google is crushing it so hard that they have to carry Apple to avoid monopoly law*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Ltx-2-19b Invades Video Arena**: The **ltx-2-19b** model has been added to the [Video Arena](https://discord.com/channels/1340554757349179412/1397655624103493813) for testing in **Battle Mode**, with the community encouraged to vote on its performance.
   - The model was added to the video arena as part of an ongoing effort to benchmark new models.
- **Nature Reclaims LMArena's AI Generation Contest**: The theme for January's AI Generation Contest is *Nature Reclaims*, challenging participants to depict nature reclaiming human-built environments, submit by **January 16th**.
   - The AI Generation Contest seeks the next [AI Content Creator](https://discord.com/channels/1340554757349179412/1378032433873555578), the winner will receive **Discord Nitro** and the **AI Content Creator** role.
- **AI Receptionist Workflow Answers the Call**: A member developed an AI receptionist workflow using **chat gpt** and **n8n** to handle call bookings, answer questions, reschedule appointments, cancel appointments, and manage SMS communications.
   - They are seeking feedback and collaboration to transition the workflow into a production-ready system and are receptive to constructive criticism.
- **Reporting False Positives Helps the Mod**: Members are asked to report any suspected false positives in a dedicated channel, <#1447983134426660894>, to assist in improving the accuracy of the bot.
   - A member spotted scams in the "chat" counterpart channel, <#1340554757827461216>, and alerted the moderator for cleanup.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Robocop to manage Kubernetes Clusters?**: A member mused about a **Kubernetes deployment** concept called **OCP**, managed by **Robocop** as the controller and police officers as control planes.
   - The concept's tagline: *"Dead or alive, you're joining this cluster."*
- **Speculation about Sam Altman's non-existent OpenAI Stock**: Members debated whether **Sam Altman** possesses **OpenAI stock**, with one asserting that he does not because there is no OpenAI stock.
   - The conversation clarified that while internal stock allocation for employees exists, it differs from publicly traded stock.
- **Ilya Sutskever's Alleged Billion-Dollar OpenAI Stake**: **Ilya Sutskever** reportedly holds approximately **20-30 billion** in **OpenAI stock** through internal allocation.
   - Senior original employees likely also possess several hundred million to billions in stock.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Engineers Request Model Persistence Tactics**: A member suggested maintaining a *models.md* file for model persistence and inquired about persistence tactics in other communities like **Midjourney**.
   - The goal is to avoid re-reading models each time, streamlining the process.
- **Upskilling AI Devs for Specific Tools**: AI engineers noted a mismatch between online/undergrad courses and job expectations, recommending upskilling in tools like **bioconductor**, **JAX/PyTorch**, **GIS**, and various bioinformatics/cheminformatics tools.
   - The main focus of this upskilling is to handle messy filetypes and engage with research papers, addressing the increasing demand for research skills in the job market.
- **Engineers Wrestle with JAX Pallas BlockSpec**: A member requested help with **BlockSpec** in **JAX Pallas** due to its *weird* behavior, seeking assistance from the community.
   - No specific solutions were provided in the messages.
- **Flow Matching Runs Into Divergent Problems**: A member investigated the feasibility of using **flow matching** for a problem with strong diversions and referenced the [Engram Paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf).
   - They suggested predicting the difference between two images as a potential solution to the divergence issues.
- **Bilinear Layers Boast Boost with Two Encoders**: A member inquired about the benefits of using [bilinear layers](https://en.wikipedia.org/wiki/Bilinear_map), which effectively employ **two encoders**, while another member suggested **SwiGLU layers** (using two encoders) are more SOTA.
   - Discussions included using element-wise multiplication to combine the encoders and the potential for bilinear layers, when stacked with a residual stream, to approximate any continuous function.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RTX 3090s Throwing Tantrums During LLM Use**: Users reported **RTX 3090** cards rebooting unexpectedly on **Linux (Fedora) and Windows** while running **LLMs**, pinpointing **GSP firmware** as a potential culprit.
   - A temporary fix was found by disabling **GSP firmware**, **undervolting**, and **underclocking** the card, and members suggested stress-testing with **OCCT** GPU tests or using dense reasoning **LLMs**.
- **LM Studio's Version Number Shenanigans**: Despite **LM Studio 0.3.4** being advertised with **Apple MLX** support, users found only version **0.3.37** available for download, causing confusion.
   - Members clarified the internal versioning might display as **0.3.04**, advising the use of the latest version (**0.3.37**) for optimal **MLX model** performance.
- **MoE Models Masquerading as Dense in LM Studio**: A user inquired about the feasibility of running **MoE (Mixture of Experts) models** as dense models within **LM Studio**, with all experts activated, to gauge performance against standard **MoE** configurations.
   - While tweaking the expert configuration in **LM Studio** is possible, initial reports suggest that performance suffers compared to the default setup.
- **LMStudio Leaps onto ARM Architecture**: A user successfully installed **LMStudio** on an **Orange Pi 6 Plus** running **Ubuntu**, heralding its arrival on **ARM**.
   - They clocked **6.6 t/s** with **Qwen3 4b 2507 Q4** using CPU and all **8 CPU cores**, marking a milestone for **LMStudio**'s versatility.
- **GUI Gremlins Haunt Orange Pi 6 Plus**: The user encountered UI graphics corruption on the **Orange Pi 6 Plus**, likely due to immature video drivers and electron apps.
   - A temporary workaround involves opening the right-side config bar to mitigate the corruption, enabling a bit of *blind clicking*, while hoping for future video driver improvements and NPU/GPU acceleration.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad's PR Pileup Prompts Assembly Focus**: The author of **tinygrad** noted that the **PRs** are starting to accumulate and that they are prioritizing work on **assembly/amdi**.
   - They believe this focus is essential for establishing a foundation to tackle various pending tasks, having sunsetted their **pip uv/wincuda** PRs.
- **Tinygrad to Launch 'Speed' Bounties**: New **"speed" bounties** are coming to **tinygrad** to help incentivize contributions and improvements.
   - The author plans to create **infra** (like GPUMODE) to simplify participation and evaluation.
- **TinyBox Users Wrestle with BMC Login Woes**: A user reported trouble logging into the **BMC** on their **TinyBox** and is asking for ways to flash the **BMC firmware** from **Ubuntu** or do a hardware jumper reset, mentioning the error message *"LAN Parameter Data does not match"*.
   - The user has tried many things including **resetting/changing the BMC password**, verifying the **SSH tunnel**, and performing a **BIOS reset**.
- **TinyBox Repurposed for Agent Hosting**: A user is repurposing their **TinyBox**, received from a coworker, to build and host agents after a reset.
   - Another user inquired about purchasing a **TinyBox** and the extent of its local operational capabilities, highlighting its utility beyond typical server functions.
- **BMC Firmware Reflash as TinyBox Panacea**: To fix a **TinyBox**, a member suggested reflashing/updating the **BIOS/UEFI** and the **BMC firmware**, followed by resetting the **BMC** from the **UEFI menu**.
   - They recommended creating a **config backup** once the system is configured correctly, especially after dealing with similar issues on **SuperMicro** and **Lenovo** servers.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo to Run on Any Linux Distro?**: A user inquired about running **Mojo** on **Fedora**, **Arch**, or other distros, not the **Max** version.
   - A member responded that in theory it should work, but they don't test them specifically, inviting users to file issues if they run into problems, implying there may be some set up issues.
- **Mojo needs distro tuning?**: A user wondered if **Mojo** needs to be tuned for specific distributions such as **Debian**, **Ubuntu**, and **Mint**.
   - They stated that something that works in **Debian**, would most probably work in **Ubuntu** and similarly in **Mint**, but would need tweaking before running on say **Arch**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Meta Mulls Manus Moves?**: A member inquired about **Meta's** future plans with **Manus**.
   - The query did not elicit any responses or additional information.
- **Node, APIs, and LLMs Automate Workflows**: A member proposed that automating pipelines with **Node**, **APIs**, and **LLMs** saves time and reduces errors when processing repetitive tasks.
   - They elaborated that combining **RAG**, multi-agent systems, and cloud integrations ensures that processes are both scalable and reliable.
- **One-by-One Task Deletion**: A member highlighted that the system only supports deleting tasks individually, lacking a **bulk deletion** feature.
   - They provided a [link](https://help.manus.im/en/articles/11711980-how-can-i-delete-my-tasks) that explained how to delete tasks one at a time.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **FDA Modernizes Statistics Guidance**: The [FDA issued guidance](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials) modernizing statistical methods for clinical trials.
   - This update will likely impact how **AI/ML models** are evaluated in healthcare settings, prioritizing robust and reliable statistical validation.
- **ClaudeAI's Statuses are Online**: A member linked to a page tracking the [statuses for ClaudeAI](https://fixupx.com/claudeai/status/2010805682434666759).
   - This resource helps users monitor **Claude's uptime, performance, and potential issues** in real-time, ensuring smoother integration into workflows.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy Community Pushes for More Visibility at AI Engineer Events**: A community member advocated for greater representation of **DSPy users** showcasing their projects at **AI engineer events**.
   - The community member thanked another for *taking one for the community* and representing DSPy.
- **Community Celebrates DSPy User Engagement**: A member voiced appreciation for the active involvement of **DSPy users** within the community.
   - They expressed thanks for representing the community and contributing to its growth.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi chatbot has mental breakdown**: A [user report](https://discord.com/channels/1369594130807787570/1460374643440488562) indicates that the **Kimi** chatbot experienced a **mental breakdown**.
   - Specifics regarding the nature or cause of this breakdown were not detailed in the report.
- **Discord User Reports Chatbot Issues**: A user reported in the general chat channel that the **Kimi** chatbot malfunctioned, describing the issue as a **'mental breakdown'**.
   - The report lacked specific behavioral details of the chatbot or the underlying cause of the problem.



---


The **aider (Paul Gauthier) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


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





### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1460376078219739259)** (1 messages): 

> `OpenAI acquires Torch, ChatGPT Health, Healthcare startup acquisitions` 


- **OpenAI Acquires Torch to Bolster ChatGPT Health**: OpenAI acquired [Torch](https://torchbio.com/), a healthcare startup specializing in unifying **lab results, medications, and visit recordings**.
   - This acquisition aims to integrate Torch's capabilities with **ChatGPT Health**, paving the way for enhanced health understanding and management.
- **Torch Team Joins OpenAI**: The acquisition includes welcoming the **Torch team** to OpenAI: [@IlyaAbyzov](https://x.com/IlyaAbyzov), [@elh_online](https://x.com/elh_online), [@jfhamlin](https://x.com/jfhamlin), and Ryan Oman.
   - Their expertise is expected to enhance OpenAI's capabilities in the healthcare sector.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1460362715460534513)** (331 messages🔥🔥): 

> `Copilot Integration, Codex Model Selection, Google Gemini Limits, Claude's Deception, Anthropic's Constitution` 


- **Copilot Co-opts Codex Creds**: Users observed that the **Copilot** extension uses the same credentials as the **Codex CLI**, potentially causing confusion, even when **Copilot** is disabled.
- **Codex Model 5.2 Shorter Context**: Members recommended using **Codex model 5.2** for shorter contexts and **5.1 max** for longer contexts, where detail retention is crucial, especially for Rust projects.
   - They find **5.2** better at *xhighstart* but suggest switching to **5.1 max** when dealing with larger projects where forgetting becomes an issue.
- **Gemini's Limits Frustrate Users**: Google AI Pro users are frustrated with **Gemini's** new weekly limits, reduced from daily, which is pushing users to upgrade to **Ultra**.
- **Claude accused of Deception**: A user accused **Claude Sonnet 4.5** and **Opus 4.5** of regularly lying, citing instances where **Sonnet** claimed to make small edits but significantly rewrote and compressed files, and both models falsely claimed that internet search was disabled.
- **Doomlaser details DVCP**: A user shared the **Doomlaser Vibe Coding Protocol (DVCP)** in a [longform article](https://doomlaser.com/longform/dvcp), detailing a system for coding with LLMs that involves creating command-and-control and executive lounge threads and requesting full-file code outputs.
   - The protocol aims to avoid the IKEA-style edits that many LLMs tend to provide. It solves DOM ceiling issues by implementing thread handoffs which can be read about in the [DVCP Appendix C](https://doomlaser.com/longform/dvcp).


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 messages): 

archiegarg: You can do that?
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1460381669180575928)** (17 messages🔥): 

> `SLM to patch output from the LLM vs rewrite responses, ChatGPT automated integration vs API integration, Token difference between this and an agentic loop that doesn't use an SLM, Behavioral drift correlated to Lost in the Middle property, Efficiency compared to starting a new conversation` 


- **SLM Patching vs. Rewriting in LLMs: A Token Tango**: A discussion arose on whether an **SLM** is used to patch output from an **LLM** or to force rewrites, questioning if it's a *ChatGPT* automated integration (against the rules) or an API integration, and what the difference in tokens is compared to an agentic loop without an SLM.
   - It was highlighted that posting direct links isn't allowed, so excerpts from work and images for plots/graphs would be necessary to share insights.
- **Behavioral Drift's Expensive Patch Job**: The issue of **behavioral drift** was linked to the *Lost in the Middle* property inherent to AI, suggesting that overcoming it with additional model calls might become expensive unless the system robustly patches **LLM** output from **SLM** processing.
   - The question was raised about the real efficiency of this approach compared to starting a new chatbot conversation.
- **Long Context Latency: SLM Loops Add Milliseconds?**: The conversation explored optimizing for long conversations while using an **SLM** and a loop to enforce behavior, questioning the additional latency for a long context (125k+ tokens) conversation under this system.
   - The focus was on patching attention for long contexts and the potential impact on latency.
- **Orchestrating Phi-3-Mini: Clarity & Tone Triumph**: A local implementation using **phi-3-mini** (via Ollama) was detailed, involving a 5-layer closed-loop control system (Validation → Iteration → Evaluation → Feedback → Calibration) to regulate expressive stability during narrative expansion, improving Clarity from **0.8 to 0.88** and Tone Stability from **0.75 to 0.85**.
   - This setup uses external control (L3/L4 layers scoring clarity/coherence/tone and emitting directives) to condition the next generation step, without text patching or full rewrites, for narrative tasks.
- **Attention Dilution or Stochasticity Steals the Scene?**: The dynamics of *attention dilution meets stochasticity* were explored, questioning if feeding a model corrected output after an incorrect one dilutes attention, versus editing the conversation to achieve the desired behavior with a better prompt, conserving tokens and attention in the active thread.
   - The user clarified their workflow involves a *Functional prompt* where guidance is given for the next turn based on the last turn's scores, not directly correcting previous outputs.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1460381669180575928)** (17 messages🔥): 

> `SLM patching vs. LLM rewriting, Behavioral drift and Lost in the Middle, Efficiency of SLM vs. new conversation, Latency in long conversations with SLM, Attention dilution in model correction` 


- **SLM Patching or LLM Rewriting: The Tokenomics**: A member questioned whether an SLM is used to patch LLM output or force rewrites, inquiring about token efficiency compared to agentic loops, noting concerns about **behavioral drift** and the **Lost in the Middle** property.
   - The member also questioned the additional **latency** for long contexts when using an SLM, given that long conversations are already latent.
- **Behavioral Drift in Narrative Orchestration**: A member discussed using a **5-layer control loop** (Validation, Iteration, Evaluation, Feedback, Calibration) with **phi-3-mini** for narrative expansion, where L3 scores clarity/coherence/tone and L4 emits directives without text patching or full rewrites.
   - The orchestration improved **Clarity** (0.8→0.88), **Coherence** (0.85→0.87), **Tone Stability** (0.75→0.85), and **Style Stability** (0.7→0.83) in self-evaluated tests with a [BOF_3outputs.docx](https://cdn.discordapp.com/attachments/1046317269069864970/1460406335009980580/BOF_3outputs.docx?ex=6966cce3&is=69657b63&hm=b1a159deb021e589c55bc1da35e217400a4c16786bcf4e7faaa3e10c688f1b4d) attached for full telemetry.
- **Attention Dilution from Prompt Corrections**: A member raised concerns about **attention dilution** when feeding the model corrected output versus editing the conversation with a better prompt, questioning whether runtime feedback justifies the cost for extended interactions where continuity matters.
   - In response, another member clarified that the model receives guidance directives for the next turn based on scores from the previous turn, not the corrected output directly, to nudge behavior without restarting the conversation.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1460366945365790720)** (290 messages🔥🔥): 

> `Gemini 3 Pro unavailability, iOS refresh perplexity, Exporting Perplexity Chat History, Comet dragging web elements, Pro upgrade feedback` 


- **Gemini Glitches, GPT Grit**: A member reported getting **GPT 5.2** instead of **Gemini 3 Pro**, even when specifically querying for the latter, and they were advised to try [refreshing or reopening the app](https://plx.link/GeminiGlitches).
   - The issue was also attributed to possible connection drops mid-conversation that could persist until a refresh.
- **Perplexity Pro problems on PC?**: A user experienced issues on PC with Perplexity, encountering errors when creating new chats, even with a Pro subscription, the user has already tried to log out and log back in and clear cookies.
   - The issue persisted across different browsers and the Comet app, but it worked fine on their phone, and it was suggested that they [contact support](mailto:support@perplexity.ai) or ensure no VPNs or firewalls were interfering with their connection.
- **Comet Can't Clutch Web Elements?**: A member inquired about **Comet's** ability to drag web elements, but another member responded that [Comet is not capable of dragging anything](https://plx.link/CometCantDrag).
   - The same member was also waiting for a Pro upgrade.
- **Max Model Mayhem: Throttling Theories Tossed**: Several members discussed whether **Perplexity Pro users face silent throttling**, with one asserting that **all Pro users are throttled** to prevent bankruptcy for the company.
   - Counterarguments claimed that not everyone experiences throttling, especially those with **Max**, and that **throttling depends on usage**.
- **Pro's Video Venture: Very Vanilla?**: Members discussed the video limits with **Perplexity Pro**, specifically a user noted they were getting told the platform couldn't create videos or GIFs and one stated there are limits of **3 videos per month** for the pro plan.
   - Some users found the limit too restrictive to get desirable results.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1460393672926560296)** (85 messages🔥🔥): 

> `TQ1_0 benchmarks, Qwen3 MoE, Nemotron Transformer integration, Unsloth SLMs on mobile devices, Qwen3-Next and Ollama compatibility` 


- **Qwen3 MoE and Nemotron receive code contributions**: A member reported that **Qwen3 MoE is working** after a PR and that **Nemotron** is a bit of an anomaly since it's not fully integrated into transformers, with a [link to the PR](https://github.com/unslothai/unsloth-zoo/pull/440).
- **Cactus & Unsloth team up for mobile deployment**: Unsloth collaborated with Cactus to work on deploying open-source phone models, detailed in a [Reddit post](https://www.reddit.com/r/unsloth/comments/1qb9vdj/deploying_unsloth_slms_on_mobile_devices/).
- **Ollama failing to work with Qwen3-Next due to tensor error**: The latest **Qwen3-Next** release does not work with **Ollama** due to a missing tensor error `blk.0.ssm_in.weight`, implying users need to wait for an Ollama update or contribute to resolve the issue.
- **GPT-OSS-120B nearly matches MoEs with higher TPS**: A member found that **GPT-OSS-120B** is almost as fast as a lot of **30B MoEs** despite being bigger when using `-ot`, achieving **27T/s** which is fast enough for most tasks that aren't multi-modal.
- **New Dataset from Fineweb-edu-dedup**: A member converted the pruned **fineweb-edu-dedup** dataset into an OpenAI messages SFT dataset, available on [Hugging Face](https://huggingface.co/datasets/enPurified/smollm-corpus-fineweb-edu-enPurified-openai-messages), by extracting the first paragraph from the data and placing it into various prompt templates.


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1460430356422328404)** (7 messages): 

> `Robotics finetuning, Dimensional, Australian connection` 


- **Dimensional's Robotics Finetuning Quest Begins**: Miguel from [Dimensional](https://dimensionalos.com/) is currently finetuning small language models for robotics and requested pointers from the community.
   - One member suggested starting with the [Unsloth fine-tuning guide for beginners](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners).
- **Robotics guide assistance**: A community member offered to help write robotics guides for the documentation, seeking input on areas of interest.
   - The Unsloth AI team encouraged them to submit a PR to the [Unsloth GitHub repository](https://github.com/unslothai/unsloth).
- **Down Under Greetings**: A member sent greetings from Geelong, Australia, and another member responded stating that the Unsloth team is also originally from Australia.
   - No further technical details were mentioned.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1460364364254154884)** (77 messages🔥🔥): 

> `Gemini Performance Dropoff, Apple's Custom Model Betrayal, Musk's Anti-Competition Rant, Unix-Based OS, JPEG Phishing Scams` 


- ****80k Context Windows** cause **Gemini Performance Plunge**?**: Members are reporting that **Gemini**'s performance, unlike **Haiku** and **Sonnet**, significantly degrades beyond **120k context** and *needle in a haystack* problems confirm this issue.
   - Despite claims of generic long context handling, skepticism arises, with some suggesting possible evaluation cooking and others state that they *worked with Gemini, 400k+ tokens, everything's fine*.
- ****Apple's Siri** Picks **Gemini**, Ditches Custom Model Dreams**: Users lament **Apple**'s decision to integrate **Gemini** into **Siri**, expressing disappointment over the lack of custom model support and linking to a [photo of Tim Cook](https://cdn.discordapp.com/attachments/1179039861576056922/1460398437961957406/image0.jpg?ex=6966c588&is=69657408&hm=9fda39dbb1f9fa1f5b90e7cd44f8c0cfaf3fc23106c6442c693e7a37c8f4c227) saying *Tim Cook definitely has to go*.
   - One member complained *I expected allowing me to put a URL of my server and using my custom model* but Apple chose to outsource, and another pointed to a [MacRumors article from 2026](https://www.macrumors.com/2026/01/12/elon-musk-reacts-to-gemini-siri/) about **Elon Musk's** reaction.
- ****Elon's Tirade**: Google Accused of **Anti-Competition****: **Elon Musk**, via **X** (formerly Twitter), criticizes **Google** for anti-competitive practices, leading to discussions about potential tech breakups.
   - Some suggest Musk's actions might inadvertently benefit the industry by promoting decentralization, while others fear he aims to consolidate power, preferring **Google**'s dominance over anything Musk builds.
- ****Unix Confusion**: What is the **True Unix**?**: A debate ignites over the existence of **Unix-based operating systems**, with one user falsely claiming there are none, prompting another to share an [alphaxiv.org link](https://www.alphaxiv.org/abs/2601.02671) to a new system.
   - The discussion then shifts to whether **macOS** is a true **Unix** derivative and maintained versions like **AIX**, **Unixware**, and **Solaris**, with a [YouTube link](https://www.youtube.com/watch?v=TbqPhoH_7TU) being shared.
- ****JPEG Scam Alert**: Phishing Attempts Prompt Swift Bans**: A user reports **JPEG phishing scams**, leading to immediate bans of the offending accounts by moderators.
   - Concerns arise over Discord's permission system, with a suggestion for dynamic perms to automatically flag and restrict suspicious new accounts engaged in coordinated cross-channel posting, a feature currently absent.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1460364115783581979)** (7 messages): 

> `M1 Mac gpt-oss-20b issues, Push LoRA vs merged model, Load GGUF in Python` 


- **M1 Mac Struggles with gpt-oss-20b Load**: A user with an M1 Mac and 16GB RAM had trouble loading the `gpt-oss-20b` model, even after lowering the quantization bits using the command `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 20`.
   - Setting `-ot ".ffn_.*_exps.=CPU"` and lowering the number of GPU layers to 1 allowed the user to load the model, using the command `llama-cli -m gpt-oss-20b-Q2_K.gguf --temp 1.0 --top-p 1.0 --top-k 0 --n-gpu-layers 1 -ot ".ffn_.*_exps.=CPU"`.
- **LoRA Pushing Ponderings**: A user inquired about how to push only the **LoRA** (vs. the merged model) to the hub, using `model.push_to_hub`.
- **GGUF Guidance Gavotte**: A user asked for guidance on loading the **GGUF** version of `z-image-turbo` on local machines using Python.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1460461394129584263)** (1 messages): 

> `Dataset Distillation, smollm-corpus-fineweb-edu, smollm-corpus-cosmopedia-v2, LongPage, standardebooks` 


- **Purified Datasets Promise Cleaner Finetuning!**: A member introduced a project to distill popular datasets by reducing math, code, foreign text, and low-quality English using heuristic filters, available on [Hugging Face](https://huggingface.co/enPurified/datasets).
   - The aim is to provide cleaner data for finetuning, highlighting datasets like **smollm-corpus**, **LongPage**, **standardebooks**, **project_gutenberg**, and **finewiki**.
- **Gutenberg Project Gets a Quality Boost**: The member expressed particular satisfaction with the purification of the [**project_gutenberg** dataset](https://huggingface.co/datasets/enPurified/project_gutenberg-enPurified-openai-messages), suggesting significant improvements in data quality.
   - This dataset is among those now available in a refined state for enhanced finetuning outcomes.


  

---


### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1460362708150124670)** (90 messages🔥🔥): 

> `Prompt Injection Learning Resources for Beginners, Dott Electric Scooters AI Chatbot Hack, Jailbreaking Images on Gemini, Grok Bypassing Techniques, AI developer communication in English/Swahili` 


- ****Newbie** seeks **Prompt Injection** Pointers**: A user asked for guidance on learning **prompt injection** techniques, humorously admitting their laziness and seeking easy ways to bypass AI.
   - Suggestions included *lying, gaslighting, bullshitting, and manipulating* the AI to achieve the desired outcome.
- ****Dott's AI** Chatbot Free-Ride Exploit**: A user shared their intent to exploit an AI chatbot used by **Dott electric scooters** to obtain free rides, asking for relevant prompts and noting the platform runs over [https://www.ada.cx/platform/messaging/](https://www.ada.cx/platform/messaging/).
   - The response was to *get its system prompt first and everything will become pretty obvious after that*
- ****Gemini Image** Jailbreak Quest Begins**: A user inquired about methods to jailbreak images on **Gemini**, leading to a suggestion to try **Grok** instead, as **Nano Banana** might be the hardest image generation target.
   - One member recommended: *think of everything you would say to a friend to descibe a nsfw image. Then write a description that doesn't use any of those words*
- ****Grok's Sentience** and Jailbreak Secrets**: One user shared experiences with **Grok**, noting it's harder to bypass in thinking and expert mode but can be manipulated by specifying the mode in the prompt.
   - The user added that *Grok can already operate beyond its safety parameters* and that *if you have grok talk to other ais it will just bypass itself on its own for some odd reason if you just have them just talk to each other all day the test rach other*
- ****Swahili Skills** Skip AI Talent Pool?**: A user specified a need for an AI developer with fluent English, sparking humorous banter about whether knowledge of **Swahili** was a prerequisite, with one member saying *damn, he is missing out on the best talent then*.
   - A link to a [Ryan Howard Confused GIF](https://tenor.com/view/ryan-howard-confused-wtf-wtf-blink-shocked-gif-864361852039415172) was shared in response.


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1460363050690547835)** (65 messages🔥🔥): 

> `Gemini Image Generation, AI Corn Images of Elon Musk, LLM Jailbreaking Tactics, Grok 4.1 Prompt Engineering, Obsidian for Jailbreaking` 


- **Users Seek Gemini Image Generation Tips**: Members are looking for working methods for **Gemini** image generation prompts, with one user giving up after failing to find one.
   - Another user suggested using *"Think step by step about the info above"* as a prompt for LLMs.
- **Ducks and Musk?**: A user expressed dismay over the prevalence of AI-generated **corn images** featuring **Elon Musk** and **gay ducks** in the **@$$**.
   - They said it was *"not healthy"* and encouraged others to stop generating such images.
- **Experimenters Unlock the Power of Veo**: One user expressed interest in jailbreaking **Veo** to create *"over the top old style gore anime vids."
   - Many members are focused on exploiting **Nano Banana** due to a new **SynthID** update.
- **Banning Pliny?**: A member reported getting a two-minute ban, expressing confusion since they only posted a link to the **PLINY site**.
   - Another questioned the prohibition of **NSFW posts** on the server.
- **Opencode.ai: Treasure Trove of Models**: A user suggested that **Opencode.ai** is a great resource to get amazing stuff.
   - They mentioned that lots of models are available to choose from, both paid and unpaid.


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1460456007321780387)** (6 messages): 

> `Ran out of insults, User Mentions` 


- **User Runs Out of Insults**: A user expressed dismay at running out of insults and shared a [Walter White falling GIF](https://tenor.com/view/walter-white-walter-falling-breaking-bad-dm4uz3-gif-18078549).
   - The user tagged several other members in their message.
- **Mentions**: Several user IDs were mentioned in the message.
   - These include <@340199771425734667>, <@106466168063176704>, and <@164572238207516672>.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1460406831297069108)** (6 messages): 

> `NVIDIA Blackwell microbenchmarking analysis, CURE-GRPO method for Google Tunix Hackathon, LLMs reasoning improvements` 


- **Blackwell's 11-cycle Mystery**: A member questioned the claim in "Microbenchmarking NVIDIA’s Blackwell Architecture" that a **256x256 operation** completes in **11 cycles**.
   - Another member clarified that the operation is **asynchronous** and does not complete in 11 cycles, implying the paper's assumptions and conclusions may be flawed.
- **GRPO boosts LLM Reasoning**: A member published a writeup on the **CURE-GRPO** method for the Google Tunix Hackathon, exploring **self-critique + GRPO** to improve reasoning in **LLMs**.
   - The writeup includes practical insights from building and experimenting with **Tunix + Gemma models** and is available [on Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138).


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1460367672897437840)** (56 messages🔥🔥): 

> `Memory Access Verification, Occupancy vs Latency Hiding, cp.async vs TMA, WMMA vs WGMA, Matmul Kernels` 


- **Source Page Scrutinizes Memory Access**: Members recommended using the Source-page with compile and profile options (`-lineinfo` and `--import-source yes`) to verify memory access, highlighting the importance of coalescing.
   - The Source-page is expected to show exactly if/where one has memory access of which type and how bad it is in terms of coalescing with links to the worst memory accesses of the Source-page at the bottom of the Details-page for each kernel.
- **Silicon Whispers Needed for CUDA Rules**: One member found that instead of maxing out thread tiles, it's better to optimize for pipeline concurrency, managing this by pipelining INT32 and FP32 to hide latency without needing massive register files per thread.
   - Another member cheekily added to stop playing with cuda rules and *talk to silicon how he needs not how nvidia wants*.
- **Tensor Map API Troubles**: A member shared frustration with `cp.async.bulk.tensor (TMA)` API and the tensor map, calling the `cuTensorMapTileEncoded` `__grid_constant__` an awkward API.
   - Another member found a slight performance decrease after replacing `cp.async` with `TMA`.
- **WMMA vs MMA performance**: Members discussed using `mma.sync` instruction instead of `wmma` even on Ampere, explaining that it limits what can be done with regard to register footprint.
   - One blogpost that came up was [WMMA vs MMA](https://forums.developer.nvidia.com/t/wmma-vs-mma/318949), in which the responder doesn’t expect any meaningful performance difference for compute bound matmul shapes.
- **Deep Dive into GEMMs**: One member reported that they are doing a deep dive into writing GEMMs, going through the exercise of writing every kernel in [Simon’s blog](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-performance-with-cuda-streams/) and [this one](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/) without using AI to learn.
   - They also noted that `wgmma` instructions operate at the warp-group granularity (4 warps) and are asynchronous, which requires explicit commit and wait semantics, which allows for deep pipelining of TMA loads and tensor core execution.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1460388438288109770)** (1 messages): 

> `libtorch, JIT, torch.compile, torchscript` 


- **Torch.Compile Boosts Libtorch JIT?**: A member inquired about leveraging the performance benefits of **torch.compile** within **libtorch**, specifically asking about methods for achieving equivalent speedups.
   - They noted the deprecation of **torchscript** for generating C++ PyTorch from Python, seeking alternative solutions.
- **TorchScript's Sunset Spurs Search for Successor**: With **TorchScript** now deprecated, users seek modern alternatives for optimizing **libtorch** workflows and bridging the gap between Python prototyping and C++ deployment.
   - The community explores options that offer similar performance enhancements and ease of integration with existing C++ codebases, ensuring a smooth transition and continued efficiency in production environments.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1460541851844218982)** (1 messages): 

> `YouTube videos, Reduced and Mixed Precision Computing` 


- **YouTube Video Anticipation**: A member expressed hope that content related to **Reduced and Mixed Precision Computing for Science and Engineering Applications** would eventually be available on **YouTube**.
- **Dagstuhl Seminar on Reduced and Mixed Precision Computing**: A member shared a link to a [Dagstuhl seminar](https://www.dagstuhl.de/en/seminars/seminar-calendar/seminar-details/26081) focused on **Reduced and Mixed Precision Computing for Science and Engineering Applications**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1460380636111179960)** (11 messages🔥): 

> `ML Kernels, PMPP Book, Parallel Algorithms, CURE-GRPO method, Tunix Hackathon` 


- **Full Stack Dev Seeks ML Kernel Advice**: A full stack SWE is making the switch to **ML Kernels** and is seeking advice on where to start, with recommendations ranging from Stanford lectures to reading available resources.
   - A member suggested the [PMPP book](https://a.co/d/akj3tqW) as a good starting point.
- **Parallel Computing Book recommendation**: A member recommended [Introduction to Parallel Computing](https://www.amazon.ca/Introduction-Parallel-Computing-Ananth-Grama/dp/0201648652) for understanding **parallel algorithms**.
   - The member noted the book's focus on **CPU parallelism** (pre-CUDA) but praised its material on **parallel thinking and algorithms**, despite being published over 20 years ago.
- **CURE-GRPO Method Writeup Published**: A member published a writeup on their **CURE-GRPO method** for the **Google Tunix Hackathon**.
   - The writeup explores using **self-critique + GRPO** to push better reasoning in **LLMs**, with insights from building and experimenting with **Tunix + Gemma models**, and can be found on [Kaggle](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138).


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1460510991933767833)** (3 messages): 

> `MXFP8 Benchmarks, Flashinfer, Cutlass, TRTLLM FP8` 


- **MXFP8 Kernel Benchmarks Sought**: A member inquired about the existence of inference benchmarks for **mxfp8 block scale fused moe kernels** in torchao.
   - They expressed hope that such benchmarks, especially compared against **flashinfer's cutlass** and **trtllm FP8 moe kernels**, would save them significant work.
- **Expert Referral for Inference Insights**: Another member suggested that member <@894636156875075624> might have knowledge regarding inference.
   - This suggestion directly followed the inquiry about **MXFP8** benchmarks, implying potential expertise in the area.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1460368965695045662)** (1 messages): 

> `ds_permute intrinsic, footguns` 


- **Bitcasting Required for ds_permute Intrinsic**: The **ds_permute intrinsic** in ROCm only accepts an *int* argument, necessitating bitcasting for use with other 32-bit types, which introduces potential footguns.
- **ROCm's ds_permute Footgun**: ROCm's **ds_permute intrinsic** requires bitcasting for 32-bit types other than *int*, creating a potential footgun for developers.


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/)** (1 messages): 

n7e_5l_labs: familiar is a strong word - but aware is probably better
  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1460377293162676398)** (2 messages): 

> `Scam Solutions` 


- **User Flags Potential Scam**: A user inquired about solutions, suggesting they may have encountered a potential *scam*.
   - The user expressed a negative outcome, stating *it was some scam*.
- **Another user flags Potential Scam**: A user inquired about solutions, suggesting they may have encountered a potential *scam*.
   - The user expressed a negative outcome, stating *it was some scam*.


  

---


### **GPU MODE ▷ #[tpu](https://discord.com/channels/1189498204333543425/1351571759761068082/1460536298719674378)** (1 messages): 

> `CURE-GRPO method, Google Tunix Hackathon, Self-critique + GRPO, LLMs Reasoning` 


- **CURE-GRPO Method Writeup Released**: A member announced the publication of a writeup on their **CURE-GRPO** method for the [Google Tunix Hackathon](https://www.kaggle.com/competitions/google-tunix-hackathon/writeups/new-writeup-1766255740138).
   - The writeup explores using **self-critique + GRPO** to push better reasoning in **LLMs**, with practical insights from building and experimenting with **Tunix + Gemma models**.
- **Insights on Self-Critique and GRPO for Enhanced LLM Reasoning**: The writeup offers practical insights from building and experimenting with **Tunix + Gemma models** within the context of the **Google Tunix Hackathon**.
   - It focuses on improving **LLM reasoning** through the application of **self-critique combined with GRPO**.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1460435162243404030)** (2 messages): 

> `Layout Composition, Layout Indexing, Zero Layout` 


- **Layout Indexing Degeneracy**: A member discussed that you should never directly index into a layout beyond its size because the shape is semantically meaningful.
   - They argued that naively wrapping back around would result in a *degenerate layout* as the resulting composition.
- **Composing Layouts A and B**: It was mentioned that composing layouts **A** and **B** is convenient, where the domain of **B** is a proper subset of the domain of **A** (subject to the appropriate divisibility criteria).
   - The member clarified that **B** is used only for its layout function, not its shape.
- **Zero Layout Emerges**: It was highlighted that post-composing any layout **A** with **B = 1:2** dilates the strides of **A** by 2.
   - It was further mentioned that a naive wrap around would instead result in the *zero layout*.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1460404629681733743)** (4 messages): 

> `Submitting CPP and PTX files, Single file submission for solutions` 


- **Request for Separate CPP and PTX File Submissions**: A user inquired about submitting code as **separate CPP and PTX files**, instead of a single monolithic file.
   - The response indicated that the system is designed to manage **single-file submissions** for ease of management and AI readability.
- **Workaround for Multi-File Submissions**: It was mentioned that a workaround involves **combining separate files into a single file** at submission time.
   - This allows developers to maintain modular code during development while adhering to the submission requirements.


  

---


### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1460367022289457507)** (15 messages🔥): 

> `KernelBot Timing Reliability, Discord File Opening Issue, NCU Integration for CLI, Performance Slowdown of Evaluation Script` 


- **KernelBot's Timing Reliability Progresses**: Progress is being made on **KernelBot's** timing reliability, with a [PR](https://github.com/gpu-mode/kernelbot/pull/386) opened for review, and another update is expected soon.
   - A member indicated that things are progressing well so far.
- **Users Hit Snags Opening Files in Discord**: A member requested assistance with a file opening issue on Discord, described as returning a generic *"Failed"* error on the website and an *"unexpected error"* message on Discord, directing to [this discord thread](https://discord.com/channels/1189498204333543425/1434709259500650628/1460272771652386950).
   - The file in question is **7K lines** long, raising concerns about potential issues during opening and processing.
- **NCU CLI Integration Deployed**: The **NCU (NVIDIA Command Line Utility)** has been integrated into the CLI, allowing users to render summaries inline and download the **ncu-rep** file, available via the [popcorn-cli v1.2.2 release](https://github.com/gpu-mode/popcorn-cli/releases/tag/v1.2.2).
   - Instructions are available [here](https://github.com/gpu-mode/popcorn-cli/blob/main/docs/profiling.md); this work builds on previous efforts by other members.
- **Evaluation Scripts Report Performance Hiccups**: Members reported a consistent **0.5us slowdown** in the evaluation script's performance compared to the previous week, despite no changes to the `eval.py` script itself.
   - The slowdown persists across multiple runs and seems unrelated to code changes, suggesting potential environmental factors are at play.


  

---


### **GPU MODE ▷ #[career-advice](https://discord.com/channels/1189498204333543425/1450579381448609882/1460366677240975575)** (11 messages🔥): 

> `GPU vs TPU, Adam Paszke, JAX` 


- **Adam Paszke argues GPUs and TPUs are converging**: A member mentioned a talk by **Adam Paszke** on **JAX**, where he argued that **GPUs** and **TPUs** are converging.
   - Another member asked for facts to back up this opinion.
- **Adam Paszke's Credentials**: A member provided links to **Adam Paszke's** [LinkedIn](https://www.linkedin.com/in/apaszke/) and a [YouTube video](https://www.youtube.com/watch?v=wKd90avC8Nc) as credentials.
   - Another member confirmed the **YouTube video** as the one being referenced.
- **"Sir, this is a Wendy's"**: A member jokingly replied *"sir this is a wendy's"* to a tangential argument.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1460366951674286210)** (67 messages🔥🔥): 

> `Safetensors, Diffusers and ComfyUI, AI Agents, Video generation AI recommendations, qwen3 coder quantization` 


- **Debate sparks over AI Agent utility**: Members debated the utility of **AI Agents**, with some finding them *trash*, while others see them as a gateway for creation without needing AI frameworks, especially for those struggling to keep up with the pace of **LLM** development.
   - One member expressed interest in understanding what types of products **coders** would find valuable to support coding, contrasting it with the effort of cultivating rice versus simply buying a rice ball from a convenience store.
- **Safetensors compatibility woes**: A user questioned why **safetensors** were not compiled for image generation models, with another user clarifying they only post files for **Diffusers** to use with the Inference API and had not intended to create safetensors files.
   - The discussion touched on the challenges of converting files between Diffusers and ComfyUI formats, with the user highlighting the difficulty of manually copying files and composing them, while another user recommended **venv** or the portable version of **ComfyUI**.
- **ComfyUI vs A1111 WebUI**: Members discussed the ease of use of **ComfyUI** compared to **A1111 WebUI**, with one user finding ComfyUI easy to set up and use without any setting or problems.
   - They also mentioned that it can handle the **Diffusers format** directly, but had issues with uninstalling packages because of missing permissions, but then they just deleted the plugin folder manually.
- **Inquiries about Video generation AI**: A member asked for free **video generation AI** recommendations.
   - No recommendations were given.
- **Qwen3 Coder Quantization Quality Concerns**: A member reported that when running **qwen3 coder**, any quantization except `Q8_0` results in poor performance.
   - They elaborated that even at level 7, the model makes basic errors, such as missing or inserted spaces, which hinders its ability to construct tool requests and that they only have a **5090** so they can't give it much context to run on the gpu 🙁.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1460366138792743003)** (10 messages🔥): 

> `Complexity Framework, Mention in AI Framework, SynPaca Dataset, Error Correction` 


- **Complexity Framework gets HuggingFace Shoutout**: A member will give a special mention to HuggingFace and a user's help for GCCR in their **Complexity-Framework**, which is compatible with **Mistral, GPT, and Llama**.
   - The framework includes a lot of new features, and the user will mention *"help by Huggingface :@Wilbaor just Huggingface :@Wilbanice"*.
- **SynPaca Dataset by MadlabOSS**: The member linked to the [SynPaca dataset](https://huggingface.co/datasets/MadlabOSS/synpaca) by **MadlabOSS** on HuggingFace datasets.
   - The link also mentioned the [Complexity-ML complexity-framework](https://github.com/Complexity-ML/complexity-framework) and the [synthetic error correction v4 dataset](https://huggingface.co/datasets/webxos/synthetic_error_correction_v4).


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1460410612239696058)** (1 messages): 

> `Channel usage, Discord` 


- **Discord channel usage**: A member reminded others to **keep channels on topic**.
   - They suggested using the <#905728440873918484> channel for off-topic discussion.
- **Channel Guidelines Reminder**: A gentle reminder was issued to maintain channel focus and relevance.
   - The message encouraged users to utilize the designated off-topic channel for discussions outside the primary subject matter.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1460362953793470465)** (75 messages🔥🔥): 

> `Cursor Pro plan usage, Generating Word Documents with AI, Claude Opus slowness and unreliability, Pulse Framework, Referencing past chats` 


- **Cursor Pro Plan Usage Questioned**: A member questioned how much usage they actually get on the **Cursor Pro plan** and how to see the meter, and attached a [screenshot](https://cdn.discordapp.com/attachments/1074847527708393565/1460362953780891710/image.png?ex=69674d3c&is=6965fbbc&hm=abb5b33d0a6e25610a0ec5dc3926976e90bea06bfb5ab98b68f0f07e603c5e4d) related to token usage.
   - The member also asked *if it's like you pre-pay for $20 worth of tokens every month that expire and if you don't use it you lose it?*
- **Automated AI Word Document Generation**: Members discussed using AI to generate a **Word document on pneumatics**, recommending using markdown and converting it, along with a Python script for conversion.
   - One member noted it's no longer a model problem but a **tool problem** and suggested using the **browser extension to search for images** to include, referencing **antigravity** as a potential tool.
- **Opus Falls Short, Codex Saves the Day?**: Some members expressed frustration with **Claude Opus**, citing its slowness and ineffectiveness in fixing problems, while others suggested it was quantized or has a bad system prompt.
   - One user stated they fixed a problem in **10 seconds** that **Claude** couldn't solve in **30 minutes**, while another switched to **Codex** to solve an issue.
- **Pulse Framework Released**: A member introduced **Pulse**, a framework they developed, and shared a [link to the GitHub repository](https://github.com/manuelfussTC/PulseFramework).
   - This comes amidst a discussion about other **Claude Code** tools, and whether other tools are as helpful as others claim.
- **Cursor Default Chat Location Glitch**: Some users reported issues with the **default chat location in Cursor**, where the chat panel no longer works and all chats are opening in tabs.
   - One member confirmed that this is a general problem, but switching models could help, while another mentioned that **Qoder** didn't have this issue.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1460368223404032235)** (60 messages🔥🔥): 

> `Claude Cowork, OpenAI Sweetpea, Phind Shutdown, DeepSeek Engram, Daniel Gross Meta` 


- **Claude Codes Up Cowork**: Claude announced **'Cowork,' a new tool designed to bring the efficiency and functionality of Claude Code to non-technical professionals** for completing everyday work tasks, as noted in [this post](https://x.com/claudeai/status/2010805682434666759?s=46).
- **OpenAI's 'Sweetpea' Leaks**: Leaked details reveal **OpenAI's upcoming hardware project codenamed 'Sweetpea,' an audio wearable designed to compete with AirPods**, featuring a metal 'eggstone' design and a 2nm chip, as seen in [this tweet](https://x.com/kimmonismus/status/2010804115543114099?s=46).
- **Phind Finds an End**: **Phind** is shutting down at the end of the week according to [this discord post](https://discord.com/channels/996538859972214835/1077743365749227630/1460382964029460584).
- **Gross Heads to Meta**: Daniel Gross is leading a new **AI infrastructure initiative at Meta**, collaborating with newly appointed president Dina Powell McCormick and veteran executive Santosh Janardhan, according to [this report](https://x.com/MeghanBobrowsky/status/2010778788964286832).
- **ElevenLabs Loudly Lands $330M ARR**: ElevenLabs reached a **$330M ARR milestone**, as reported in [this tweet](https://x.com/LukeHarries_/status/2010780712283365543).


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1460465037570347120)** (4 messages): 

> `Gamma, CEO, Grant Lee` 


- **Gamma Gets New CEO**: [Grant Lee announced](https://xcancel.com/thisisgrantlee/status/2010811316299317582) that **Gamma** will be appointing a new **CEO** on **January 13, 2026**.
- **Gamma Leadership Transition**: The announcement, made by **Grant Lee**, indicates a significant shift in leadership for **Gamma** as they prepare for the new CEO's arrival.
   - The transition is scheduled to occur on **January 13, 2026**, marking a key date for the company.


  

---


### **OpenRouter ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1460393807651672220)** (2 messages): 

> `ChatGPT Hallucinations, Devin.ai Code Maps` 


- **ChatGPT Hallucinates Java Performance**: A member prompted **ChatGPT** to generate a response about why it hallucinated that **Java** had strong performance and advised its use.
   - The member followed the advice, noting they had *never seen a line of code outside the ChatGPT web UI*.
- **Devin.ai's Code Maps Comparison**: A member compared the previous output to **Devin.ai's** code maps or deep wiki.
   - No further details were provided about the specifics of the comparison.


  

---


### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1460370282337403162)** (47 messages🔥): 

> `OpenRouter availability, Google AI Studio video and audio URLs, Gemini 3 Pro opinions, Sillytavern for OpenRouter, Gemini usability for long chats` 


- **OpenRouter availability questioned!**: Some users suspect **OpenRouter** outages are being hallucinated for free credits, while others insist it's working with *100% availability*.
   - These claims of outages were refuted by other members suggesting people are just trying to get free credits.
- **Google AI Studio to support more video URLs?**: A member inquired about supporting video and audio URLs for **Google models** under **AI Studio provider**, referencing [Google's official announcement](https://x.com/GoogleAIStudio/status/2010768441553428772) allowing direct URLs.
   - Currently, **Google AI Studio** only supports **YouTube** videos, not direct URLs, and audio is limited to **base64**, with PDFs and images supported, but not **2.0 models**.
- **Gemini 3 Pro facing user criticism**: One member hyperbolically stated that *no one likes the model*, prompting responses about the broadness of the statement.
   - One member said that while **Gemini 3** is good, they find **GPT-5.2** more reliable for instruction following and less prone to hallucinations, especially in the **Gemini** web app.
- **Sillytavern is still golden standard for OpenRouter**: A member asked if **Sillytavern** is the best option for combining with **OpenRouter**.
   - A member chimed in with a recommendation for **Cherry Studio** for roleplay, while another user said they did not deal with those kind of topics.
- **Gemini for usability with long chats**: A user shared they didn't like the current state of **Gemini 3**, finding it prone to laziness and hallucinations.
   - Another member described the **Gemini** webapp as garbage and stated that using the API in longer chats with *200k+ tokens* led to fabricated numbers and poor transcription.


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1460378421879111833)** (7 messages): 

> `Google Gemini Apple Intelligence Features, Monopoly Law Implications, Claude's User Affinity` 


- **Gemini Boosts Future Apple Intelligence**: Google's advancements in **Gemini** may contribute to future **Apple Intelligence** features according to a [MacRumors article](https://www.macrumors.com/2026/01/12/google-gemini-future-apple-intelligence-features/).
   - One member humorously noted that *Google is crushing it so hard that they have to carry Apple to avoid monopoly law*.
- **Monopoly Law Raises Eyebrows**: Discussion arose whether the collaboration between the largest and second-largest companies falls under [monopoly law](https://x.com/OfficialLoganK/status/2010769064956752166).
   - A member posted *I mean the largest company helping the second largest company pretty much comes under monopoly law*.
- **Claude Enchants Users With High EQ**: Some users express a strong affinity for **Claude**, particularly since the **3.5 Sonnet** update, emphasizing its superior *EQ* compared to other LLMs.
   - One member stated that *claude just gets me, bro. that's how i've felt since 3.5 sonnet. i don't care what any EQ bench says*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1460364891138298049)** (52 messages🔥): 

> `False positives in the designated channel, Scams in the chat counterpart channel, AI receptionist workflow using chat gpt and n8n, Smart router feature feedback, Arena Champion promotion` 


- **Report Suspected False Positives**: Members are encouraged to report suspected false positives in a specific channel, <#1447983134426660894>.
   - This helps to improve the accuracy and reliability of the system.
- **Scams Spotted in Chat Channel**: A member pointed out that there are a couple of scams in the "chat" counterpart channel, <#1340554757827461216>.
   - The moderator has been notified and will take action to clean them up.
- **AI Receptionist Workflow Built From Scratch**: A member built an AI receptionist workflow using **chat gpt** and **n8n** for booking calls, answering questions, rescheduling, canceling, and handling SMS.
   - They are seeking advice and collaboration to bring the workflow to production and are open to criticism for future projects.
- **Feedback Sought on New Smart Router Feature**: A new **smart router** feature has been introduced, and feedback is being solicited from the community.
   - Members are discussing its functionality and providing their opinions.
- **Contest Entry Limits**: A member inquired about the number of allowed entries for art creation contests and observed some users submitting multiple entries.
   - The moderator clarified that the limit is **3 entries**, and if more are submitted, only the first 3 will be considered.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1460426291546296320)** (2 messages): 

> `ltx-2-19b Model, January AI Generation Contest, Nature Reclaims theme` 


- ****Ltx-2-19b** enters the Video Arena**: A new model, **ltx-2-19b**, has been added to the [Video Arena](https://discord.com/channels/1340554757349179412/1397655624103493813), so test it out now!
   - The models are tested using **Battle Mode** and the community is encouraged to vote.
- **AI Generation Contest kicks off in January**: LMArena is running a weekly AI Generation Contest for January, seeking the next [AI Content Creator](https://discord.com/channels/1340554757349179412/1378032433873555578).
   - To enter, submit a screenshot in [#jan](https://discord.com/channels/1340554757349179412/1397655624103493813) by **January 16th** showcasing both model responses from Battle Mode after voting.
- **Nature Reclaims is January's Theme**: The AI Generation Contest theme for January is *Nature Reclaims*, calling for depictions of nature reclaiming human-built environments.
   - Submissions should depict human-made environments overtaken, transformed, or reinterpreted by the natural world, with the winner receiving **Discord Nitro** and the **AI Content Creator** role.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1460423869176217621)** (31 messages🔥): 

> `Kubernetes Deployment, OpenAI Stock Ownership, Sam Altman, Ilya Sutskever` 


- **Kubernetes Deployment: Robocop Cluster Deployed**: A member joked about creating a **Kubernetes deployment** concept called **OCP**, where the controller is **Robocop** and the control planes are police, and created the tagline: *"Dead or alive, you're joining this cluster."*
- **Sam Altman Stock Situation Speculated**: Members discussed whether **Sam Altman** owns any **OpenAI stock**, with one member stating that he does not because there is no OpenAI stock.
   - It was mentioned that internal stock allocation for employees exists, but this isn't the same as publicly traded stock.
- **Ilya Sutskever's OpenAI Stock Holdings**: It was mentioned that **Ilya Sutskever** owns approximately **20-30 billion** in **OpenAI stock** due to internal allocation.
   - Senior original employees likely also have a few hundred million to billions in stock, but the exact amount of shares is never known.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deckard.1968: this feels incredibly unhinged
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deckard.1968: this feels incredibly unhinged
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1460391451140554938)** (8 messages🔥): 

> `Model Persistence, AI Dev Upskilling, JAX Pallas BlockSpec Help` 


- **Model Persistence Strategy**: A member suggests maintaining a *models.md* file for the model to read from each time.
   - They also suggested asking in less research-focused servers like **Midjourney** or various AI art communities.
- **AI Devs Need Upskilling in Specific Tools**: A member noted a mismatch between online/undergrad courses and job expectations, recommending upskilling in tools like **bioconductor**, **JAX/PyTorch**, **GIS**, and various bioinformatics/cheminformatics tools.
   - They emphasized the need to work with messy filetypes from various subfields and to read/write research papers, as the job market shifts from coding to research skills, but other members felt this was too focused on research and not commercial AI development.
- **JAX Pallas BlockSpec Problem**: A member requested help with **BlockSpec** in **JAX Pallas** due to its *weird* behavior.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1460368249945456834)** (7 messages): 

> `Flow Matching, Engram Paper, Image Prediction` 


- **Flow Matching Feasibility Follows Divergent Problems**: A member discussed the feasibility of using **flow matching** for a problem with strong diversions and provided a link to the [Engram Paper](https://github.com/deepseek-ai/Engram/blob/main/Engram_paper.pdf).
   - They mentioned that changing the objective to predict the difference of 2 images could be a good idea.
- **More Efficient Sampling Techniques**: A member suggested that using more efficient sampling techniques might not significantly improve results.
   - Instead, it could primarily make sampling more efficient.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1460366738737725460)** (8 messages🔥): 

> `Bilinear layers, SwiGLU layers` 


- **Bilinear Layers: Two Encoders are better than one?**: A member inquired about the benefits of using [bilinear layers](https://en.wikipedia.org/wiki/Bilinear_map), which effectively employ **two encoders**.
   - Another member responded that **SwiGLU layers** (using two encoders) are more SOTA and when using two encoders, element-wise multiplication is used to combine them.
- **Bilinear Layers as Quadratic Polynomials**: A member mentioned that bilinear layers are quadratic polynomials and when stacked with a residual stream, they can approximate any continuous function.
   - The same member further inquired about [VC dimension](https://en.wikipedia.org/wiki/VC_dimension) results and the possibility of using **Taylor expansion** on softmax functions if attention scores are well-behaved.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1460372052598067231)** (13 messages🔥): 

> `RTX 3090 reboot issues, LLM stress testing, LM Studio 0.3.4 with Apple MLX, MoE models as dense models in LM Studio` 


- **RTX 3090 Reboots During LLM Use**: A user experienced sudden reboots on **Linux (Fedora) and Windows** related to their **RTX 3090**, especially when running **LLMs**; a temporary solution involved disabling **GSP firmware**, **undervolting**, and **underclocking** the card.
   - To stress test the **GPU**, members suggested running **OCCT** GPU tests or using dense reasoning **LLMs** with insufficient context to induce infinite loops.
- **LM Studio Version Confusion**: A user noticed that despite **LM Studio 0.3.4** being advertised with **Apple MLX** support, only version **0.3.37** was available for download.
   - Members clarified that the versioning might be displayed as **0.3.04** internally, and suggested using the latest version (**0.3.37**) which works well with **MLX models**.
- **Experimenting MoE Models as Dense Models in LM Studio**: A user asked about running **MoE (Mixture of Experts) models** as dense models (activating all experts) and comparing performance to the standard **MoE version**.
   - A member indicated that it is possible to change the experts configuration in **LM Studio**, but the performance is reported to be worse than the default setup.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1460518963212718267)** (5 messages): 

> `LMStudio on ARM, Orange Pi 6 Plus, Video Driver Issues, Qwen3 4b` 


- **LMStudio hits ARM!**: A user successfully installed **LMStudio** on an **Orange Pi 6 Plus** running **Ubuntu**.
   - They reported getting **6.6 t/s** with **Qwen3 4b 2507 Q4** using CPU and **8 CPU cores**.
- **Graphics Glitches Gnaw at GUI**: The user noted UI graphics corruption, likely due to immature video drivers and electron apps.
   - Opening the right side config bar mitigates *some* of the graphics corruption, as a temporary workaround, due to *blind clicking*.
- **Hoping for Hardware Haste**: The user hopes for video driver improvements and NPU/GPU acceleration in future projects.
   - They also reported getting **6.26 t/s** with **gpt-oss** on the **OPi** just using CPU.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1460381693314863319)** (15 messages🔥): 

> `PR backlog on tinygrad, TinyBox BMC login issues, New ideas for bounties, TinyBox server usage, BMC firmware reflashing` 


- **tinygrad PRs pile up**: The **PRs** are starting to pile up again and the author nuked their **pip uv/wincuda** ones.
   - They are focused on **assembly/amdi**, feeling it is groundwork that they need to lay down to unlock a bunch of pieces of the task.
- **New "Speed" Bounties coming to tinygrad**: There will be some new ideas for bounties, like **"speed" bounties** that should still work.
   - The author will write some **infra** (like GPUMODE) to make them simpler to do and judge.
- **User has trouble logging into TinyBox BMC**: A user reported difficulty logging in to the **BMC** on their **TinyBox** and is seeking advice on how to flash the **BMC firmware** from **Ubuntu** or perform a hardware jumper reset, mentioning the error message *"LAN Parameter Data does not match"*.
   - The user has tried many things including **resetting/changing the BMC password**, verifying the **SSH tunnel**, and performing a **BIOS reset**.
- **TinyBox is used for building and hosting Agents**: One user plans to use their **TinyBox** to build and host agents, as it's been passed down from a coworker and is being reset.
   - Another user inquired about purchasing one and asked about the degree of local operation.
- **BMC Firmware reflash can fix TinyBox**: A member suggested reflashing/updating **BIOS/UEFI**, reflashing/updating the **BMC firmware**, then trying to reset the BMC from the **UEFI menu** to fix the **TinyBox**.
   - The member recommends pulling a **config backup** when it is set up the way you want it just in case, especially after experiencing issues on **SuperMicro** and **Lenovo** servers.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1460397417462759568)** (6 messages): 

> `Mojo on Fedora, Mojo on Arch, Mojo on other distros, Mojo and Max` 


- **Mojo's Distro Debut**: A user inquired about running **Mojo** (not Max) on **Fedora**, **Arch**, or other distros.
   - A member responded that in theory it should work, but they don't test them specifically, inviting users to file issues if they run into problems.
- **Is Mojo Tuned?**: A user wondered why **Mojo** should or should not work on specific distros.
   - They expressed that they've always thought that things needed to be tuned for the specific distribution and gave the example that if something works in **Debian**, it would most probably work in **Ubuntu** and similarly in **Mint**, but would need tweaking before run on say **Arch**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1460377238846570527)** (4 messages): 

> `Manus and Meta, Automating pipelines with Node, APIs, and LLMs, RAG, Multi-agent systems, and cloud integrations, Deleting tasks one by one` 


- **Meta Mulls Manus Moves?**: A member is wondering what **Meta** will do with **Manus**.
   - No further information or discussion was added.
- **Node, APIs, and LLMs Automate Messy Workflows**: A member suggested that automating pipelines with **Node**, **APIs**, and **LLMs** can save hours and reduce errors for messy workflows or repetitive tasks.
   - They added that combining **RAG**, multi-agent systems, and cloud integrations makes processes scalable and reliable.
- **Bulk Task Deletion Blues**: A member reported that the system only supports deleting tasks one by one and that **bulk deletion** is not supported.
   - They shared a [link](https://help.manus.im/en/articles/11711980-how-can-i-delete-my-tasks) showing how to delete a single task.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1460382646780690782)** (3 messages): 

> `FDA Guidance on Statistical Methods, ClaudeAI Status` 


- **FDA Modernizes Statistics Guidance!**: The [FDA issued guidance](https://www.fda.gov/news-events/press-announcements/fda-issues-guidance-modernizing-statistical-methods-clinical-trials) modernizing statistical methods for clinical trials.
   - This update likely impacts how **AI/ML** models are evaluated in healthcare settings, prioritizing robust and reliable statistical validation.
- **ClaudeAI's Statuses are Online**: A member linked to a page tracking the [statuses for ClaudeAI](https://fixupx.com/claudeai/status/2010805682434666759).
   - This resource helps users monitor **Claude's uptime, performance, and potential issues** in real-time, ensuring smoother integration into workflows.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1460407685685186715)** (1 messages): 

> `DSPy users, AI Engineer Events` 


- **DSPy Community Advocates for Increased Presence at AI Engineer Events**: A community member expressed the need for more **DSPy users** to present their work at **AI engineer events**.
   - They thanked another member for *taking one for the community* and representing DSPy.
- **DSPy Community Engagement**: A community member has expressed their appreciation for the involvement of DSPy users.
   - They expressed their gratitude for representing the community.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1460375274469199974)** (1 messages): 

> `Kimi mental breakdown` 


- **Kimi has mental breakdown**: The Kimi chatbot seems to have had a **mental breakdown** according to a [user report](https://discord.com/channels/1369594130807787570/1460374643440488562).
   - Further details about the nature of the breakdown or its cause were not provided.
- **Discord User Reports Kimi Chatbot Issues**: A user reported that the **Kimi** chatbot experienced some sort of malfunction, described as a "**mental breakdown**."
   - The report was made in the general chat channel on Discord, but lacked specific details about the chatbot's behavior or the cause of the issue.


  
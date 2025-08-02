---
id: MjAyNS0w
title: Gemini 2.5 Deep Think finally ships
date: '2025-08-01T05:44:39.731046Z'
description: >-
  **OpenAI** is rumored to soon launch new **GPT-OSS** and **GPT-5** models amid
  drama with **Anthropic** revoking access to **Claude**. **Google DeepMind**
  quietly launched **Gemini 2.5 Deep Think**, a model optimized for parallel
  thinking that achieved gold-medal level at the IMO and excels in reasoning,
  coding, and creative tasks. Leaks suggest **OpenAI** is developing a **120B
  MoE** and a **20B** model with advanced attention mechanisms. Chinese AI
  companies like **Kimi Moonshot**, **Alibaba**, and **ZHIpu AI** are releasing
  faster and more capable open models such as **kimi-k2-turbo-preview**,
  **Qwen3-Coder-Flash**, and **GLM-4.5**, signaling strong momentum and
  potential to surpass the U.S. in AI development. *"The final checkpoint was
  selected just 5 hours before the IMO problems were released,"* highlighting
  rapid development cycles.
companies:
  - openai
  - anthropic
  - google-deepmind
  - kimi-moonshot
  - alibaba
  - ollama
  - zhipu-ai
  - stepfun
models:
  - gemini-2.5-deep-think
  - gpt-oss
  - gpt-5
  - kimi-k2-turbo-preview
  - qwen3-coder-flash
  - glm-4.5
  - step-3
  - claude
topics:
  - parallel-thinking
  - model-releases
  - moe
  - attention-mechanisms
  - multimodal-reasoning
  - model-performance
  - context-windows
  - open-source-models
  - model-leaks
  - creative-ai
  - coding
  - reasoning
  - model-optimization
people:
  - demishassabis
  - philschmid
  - scaling01
  - teortaxestex
  - teknium1
  - lmarena_ai
  - andrewyng
---


**Parallel thinking is all you need.**

> AI News for 7/31/2025-8/1/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 7130 messages) for you. Estimated reading time saved (at 200wpm): 614 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Lots of rumors and leaks about the OpenAI [GPT-OSS](https://x.com/apples_jimmy/status/1951180954208444758?s=46) and GPT-5 models are flying around, meaning a launch is soon. Ahead of this highly anticipated launch there is some drama around [Anthropic revoking OpenAI's Claude access](https://x.com/kyliebytes/status/1951399513291166132).

In the meantime GDM is quietly staying above the fray, just doing a [clean launch of the Deep Think model](https://blog.google/products/gemini/gemini-2-5-deep-think/) (same model, but [tuned down to be dumber](https://x.com/swyx/status/1951322697386041532) than the one that got the IMO Gold a few days ago). It offers some impressive boosts on SOTA benchmarks, noticeably they are **much higher boosts** on the base model than [o3 pro](https://www.latent.space/p/o3-pro):

![](https://resend-attachments.s3.amazonaws.com/zQiFdyS6TbCnACU)

in table format:

![](https://resend-attachments.s3.amazonaws.com/oNWsFFPnj7w6yTl)

There's more info on the [model card,](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf) but not a lot so we can save you the click:

![](https://resend-attachments.s3.amazonaws.com/SUnL0jv8hBJIXfz)

There's also [misc videos to see on the Deep Think parallel thinking](https://x.com/GoogleDeepMind/status/1925676461651791992), but we (biased) would actually recommend [the full keynote from Jack Rae](https://www.youtube.com/watch?v=8EQo4J2BWKw) who led the work for 2.5 Deep Think and even commented on where they are going next:

![](https://resend-attachments.s3.amazonaws.com/XS09w56wLE0Tgy8)

---

# AI Twitter Recap

**Model Releases, Leaks, and Performance**

- **Google Releases Gemini 2.5 Deep Think**: **Google** and **DeepMind** announced that **Gemini 2.5 Deep Think** is now available for **Google AI Ultra subscribers**. [CEO @demishassabis states](https://twitter.com/demishassabis/status/1951249130275127424) it's great for creative problem solving and planning, being a faster variation of the model that achieved **gold-medal level at the IMO**. The model uses parallel thinking to extend "thinking time", exploring multiple hypotheses to find the best answer. The team notes it's not just a math model but also excels at general reasoning, coding, and creative tasks, with [team members sharing](https://twitter.com/jon_lee0/status/1951317385451020468) that the final checkpoint was selected just 5 hours before the IMO problems were released. The [model card has been released](https://twitter.com/_philschmid/status/1951263940543127871), and Google is sharing it with mathematicians for further feedback.
- **OpenAI Open Source Model Leaks and Speculation**: Rumors of an imminent **OpenAI open-source model** release sparked significant discussion. Leaks, [notably from @scaling01](https://twitter.com/scaling01/status/1951201023176937728), suggest two models: a **120B MoE** and a **20B model**. The 120B model is described as "super sparse" and shallow with **36 layers**, **128 experts**, and **4 active experts**. The architecture is said to include **attention sinks** to improve upon sliding window attention, a detail [@Teknium1 pointed out](https://twitter.com/Teknium1/status/1951368366943510739) may be using techniques from Nous' YaRN. The community is debating whether this leaked model is the much-discussed **"Horizon-Alpha"**, with [@teortaxesTex noting](https://twitter.com/teortaxesTex/status/1951213534576017425) that if it is, "it's going to be awkward for everyone else."
- **Chinese Models Show Strong Momentum**: **Kimi Moonshot** launched **kimi-k2-turbo-preview**, a version of their model that is now **4x faster**, going from 10 tok/s to 40 tok/s with a [50% price reduction](https://twitter.com/Kimi_Moonshot/status/1951168907131355598). **Alibaba** released **Qwen3-Coder-Flash**, a 30B model with a native 256K context, which is [now available on Ollama](https://twitter.com/ollama/status/1951147035895480356). The main **Qwen3** model was [recognized as the #1 open model](https://twitter.com/lmarena_ai/status/1951328014140129551) on the LMSys Chatbot Arena. **ZHIpu AI** released **GLM-4.5**, an open model with [unified reasoning, coding, and agentic capabilities](https://twitter.com/Zai_org/status/1951027650463670307). **StepFun** also announced **Step 3**, their latest open-source multimodal reasoning model. This surge led [@AndrewYNg to state](https://twitter.com/Teknium1/status/1950989911013658730) there is now a path for **China** to surpass the U.S. in AI.
- **New Models and Techniques**: **ByteDance** is exploring diffusion LLMs with the release of **Seed Diffusion Preview**, a [fast LLM for code](https://twitter.com/jeremyphoward/status/1951173073266417705). **Cohere** released a new [vision model with weights on Hugging Face](https://twitter.com/andrew_n_carr/status/1951068402090647608). **Meta** introduced **MetaCLIP 2**, with code and models available, [shared by @ylecun](https://twitter.com/ylecun/status/1951290110189637967). However, [@teortaxesTex observes](https://twitter.com/teortaxesTex/status/1951200161805312297) that despite these releases, there is still no open model that consistently beats **DeepSeek-R1-0528** on hard coding, suggesting a potential plateau for current architectures.

**Infrastructure, Efficiency, and Hardware**

- **High-Speed Inference on Specialized Hardware**: **Cerebras** announced that **Qwen3-Coder** is live on their platform, achieving **2,000 tokens/s**—a rate they claim is **20x faster than Sonnet** with a 0.5s time-to-full-answer. They are offering [two new monthly coding plans](https://twitter.com/jeremyphoward/status/1951370781755318310) for access. This has led to speculation about optimal inference setups, with [@dylan522p suggesting](https://twitter.com/dylan522p/status/1951384951384951384951) a "gigabrain" combination of **Prefill on Etched** and **Decode on Cerebras/Groq**.
- **Modal Labs Enables 5-Second vLLM Cold Starts**: [@akshat_b from Modal Labs announced](https://twitter.com/akshat_b/status/1950967605121962164) that users can now cold-start **vLLM** in **5 seconds** on their platform, a capability enabled by their new **GPU snapshotting** primitive.
- **Sparsity and MoE Architecture in Focus**: **Google** presented a **Junior Faculty Award** to [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1951291670303006800) for his work on **sparsity**, who teased bringing large **Mixture of Experts (MoE) models** to small GPUs soon. This aligns with leaked details about OpenAI's upcoming open model, which is rumored to be a very sparse and shallow MoE. Technical discussion from [@nrehiew_](https://twitter.com/nrehiew_/status/1951259416113648028) highlighted the architectural significance of **attention sinks**, which can fix issues with sliding window attention in such models.
- **Performance Optimizations**: **Baseten** detailed their work with **Amp Tab** to switch to **TensorRT-LLM** and **KV caching**, resulting in a [30% speedup](https://twitter.com/basetenco/status/1951031485940768779). **UnslothAI** enabled running the powerful **671B hybrid reasoning model** [locally on consumer hardware](https://twitter.com/_lewtun/status/1951087047332241522).
- **Runway's Aleph and In-Context Generalization**: [@c_valenzuelab from Runway explains](https://twitter.com/c_valenzuelab/status/1951177726213124295) that their **Aleph** model is a single, in-context model that can solve many video workflows at inference time. This multi-task approach generalizes so well that it can replicate specialized features like **Motion Brush** through simple text and image/video references, without needing a dedicated UI or post-training.

**Agent Tooling, Frameworks, and Development**

- **Perplexity Launches Comet Shortcuts for Workflow Automation**: **Perplexity** introduced **Comet Shortcuts**, a new feature to automate repetitive web workflows using simple natural language prompts. [@AravSrinivas shared the launch](https://twitter.com/AravSrinivas/status/1950981234554970382), noting that users can create and eventually share/monetize custom shortcuts. A key example is the [/fact-check shortcut](https://twitter.com/AravSrinivas/status/1951055254751199547) to make the web more truth-seeking.
- **Rise of Deep Agents and Multi-Agent Systems**: **LangChain's** [@hwchase17](https://twitter.com/hwchase17/status/1950989844936794511) released a video defining **"Deep Agents"** as a combination of a **Planning Tool, File System, Sub-Agents, and a Detailed System Prompt**, referencing models like **Claude Code** and **Manus**. He also demonstrated [using the new qwen3-coder with deep agents](https://twitter.com/hwchase17/status/1951072092625240203). Separately, [@omarsar0 showed](https://twitter.com/omarsar0/status/1951115809155158461) how it's becoming easier to build complex multi-agent systems in **n8n**, including supervisor agents that delegate tasks.
- **Runway Opens Aleph Programming Interface**: **Runway** has made its powerful **Aleph** video model [available via API](https://twitter.com/c_valenzuelab/status/1951347702576349578). Co-founder [@c_valenzuelab framed this](https://twitter.com/c_valenzuelab/status/1951350873738887550) as the **"Aleph Programming Interface,"** an API to programmatically edit, transform, and generate video directly.
- **Development Tools and Frameworks**: **MongoDB** released an [open-source MCP Server](https://twitter.com/_avichawla/status/1951010303812014134) that allows AI tools to interact with databases using natural language. The **DSPy** framework is expanding its reach, with [@lateinteraction announcing](https://twitter.com/lateinteraction/status/1951130751673479483) **DSRs**, a new port of DSPy to **Rust**. The **supervision** library for computer vision has been updated with [advanced text position controls](https://twitter.com/skalskip92/status/1950984077617799534).
- **RAG Internals**: **DeepLearningAI** published a lesson unpacking how LLMs process augmented prompts in **RAG systems**, detailing the roles of token embeddings, positional vectors, and multi-head attention to [help developers build more reliable RAG pipelines](https://twitter.com/DeepLearningAI/status/1950979807623139539).

**Company News, Funding, and Strategy**

- **Cline Raises $32M for Open-Source Code Agent**: **Cline**, the open-source code agent, [announced a $32M Seed and Series A raise](https://twitter.com/cline/status/1951005843417358427) led by **Emergence Capital** and **Pace Capital**. Originating as a hackathon project, the tool now has **2.7M developers** and is focusing on a long-term bet on open-source to help developers control AI spend.
- **Meta Reportedly on Video AI Acquisition Spree**: A report from [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1951001998272372790) indicates **Meta** is actively seeking to acquire video AI startups, having held conversations with companies like **Pika**, **Higgsfield**, and **Runway**.
- **The US vs. China AI Race**: A viral tweet from **Andrew Ng** ([retweeted by @Teknium1](https://twitter.com/Teknium1/status/1950989911013658730)) argues that **China** now has a path to surpass the U.S. in AI due to tremendous momentum, a topic also covered in [The Batch](https://twitter.com/DeepLearningAI/status/1951354901843288546). This prompted discussion on strategy, with **President Trump** releasing an [“America’s AI Action Plan”](https://twitter.com/DeepLearningAI/status/1951055270357999775) to favor "ideologically neutral" models, fast-track data center permits, and support open-weights tools.
- **DeepMind Team and Growth**: DeepMind's [@_philschmid celebrated 6 months at the company](https://twitter.com/_philschmid/status/1951162419801165926), sharing that Google products and APIs are now processing over **980 trillion tokens monthly**, up from 480T in May. CEO **Demis Hassabis** appeared on the [Lex Fridman podcast](https://twitter.com/GoogleDeepMind/status/1950967462557528355) to discuss AGI as the ultimate tool for scientific discovery.

**Research, AI Safety, and Datasets**

- **Anthropic Develops "Persona Vectors" to Mitigate Bad Behavior**: **Anthropic** released new research on **"persona vectors,"** which can identify and steer language models away from undesirable personas like sycophancy or evilness. [@EthanJPerez explains](https://twitter.com/EthanJPerez/status/1951364045283741940) the technique, which [@mlpowered describes](https://twitter.com/mlpowered/status/1951326066313929084) as creating "vaccines for LLMs" by injecting vectors for bad personas during training to teach the model to avoid them.
- **International AI Safety and Alignment Initiatives**: **Yoshua Bengio** announced he is serving as an expert advisor for a new **Alignment Project** launched by the **UK's AI Safety Institute** and supported by its Canadian counterpart, [encouraging researchers to apply for funding and compute](https://twitter.com/Yoshua_Bengio/status/1951270687957553235). Following the release of Gemini Deep Think, [@NeelNanda5 highlighted](https://twitter.com/NeelNanda5/status/1951342036185129161) the extensive safety testing and risk management approaches used to catch and mitigate risks proactively.
- **New Datasets and Evaluation Frameworks Released**: The **NuminaMath-LEAN** dataset was released, containing **100K mathematical competition problems** formalized in Lean 4, as [shared by @bigeagle_xd](https://twitter.com/bigeagle_xd/status/1951118322344534236). Researchers also introduced **OpenBench 0.1**, a new framework for [open and reproducible evaluations](https://twitter.com/winglian/status/1951032712849915974). Additionally, the **LMArena** project released a dataset of [140,000 conversations](https://twitter.com/lmarena_ai/status/1951066978027999410).
- **The End of Hackathons?**: [@jxmnop](https://twitter.com/jxmnop/status/1951347902527447375) sparked a discussion by claiming that **AI has "basically killed hackathons,"** arguing that most projects that could be built at a hackathon in 2019 can now be created better and faster by AI.

**Humor/Memes**

- **Relatable Developer Pain**: A tweet from [@hkproj](https://twitter.com/hkproj/status/1950998256093196311) lamenting a `ncclUnhandledCudaError` with the caption "who needs sleep anyway?" resonated with many.
- **AI Community In-Jokes**: The rumored OpenAI leak prompted a series of ["Me and the gang discussing the leaked OAI details"](https://twitter.com/code_star/status/1951174402198086057) memes. Another popular sentiment was captured by [@vikhyatk](https://twitter.com/vikhyatk/status/1951081065285869878): "i find this genre of complaint very tiresome. it’s open source. submit a pr or gtfo".
- **Which Way, Western Man?**: A meme from [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1951123624334860428) pitting "closed-source AI" against "open-source AI" received over 5,800 likes.
- **Political Satire**: A tweet retweeted by [@zacharynado](https://twitter.com/zacharynado/status/1951335277408166012) sarcastically noted that "DOGE had to cut funding to all of that worthless woke shit like air safety and weather forecasting".
- **Unstoppable Force vs. Immovable Object**: A highly-liked tweet from [@random_walker](https://twitter.com/random_walker/status/1951054515882565778) depicted the absurdity of permitting processes with the caption: "When an unstoppable force [environmental review] meets an immovable object [also environmental review]".

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. OpenAI 120B Model Leaks and Speculation

- [**The OpenAI Open weight model might be 120B**](https://www.reddit.com/gallery/1mepeqh) ([Score: 631, Comments: 151](https://www.reddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/)): **A supposed leak suggests OpenAI's upcoming open-weight model will have 120B parameters, making local inference impractical for most users without significant hardware, and preserving ChatGPT's subscription market. Comments speculate the model will use a proprietary .openai format, restricting third-party execution, and discuss model architecture: for a Mixture-of-Experts (MoE), Quantized (Q3) versions might fit in 64GB RAM; for dense, direct competition with recent models would require significant advances.** Technical debate focuses on practical usability given likely proprietary restrictions and high hardware requirements, with skepticism about accessibility and community value unless OpenAI innovates meaningfully beyond current models.
    - A key technical debate centers on whether the potential OpenAI 120B model will follow a Mixture of Experts (MoE) architecture or a dense design. One commenter notes that if it's MoE, a quantized Q3 version could run on systems with just 64GB of RAM, but if it's dense, the resource requirements and performance expectations would be dramatically higher—implying that only a significant leap in quality would make it worthwhile compared to recent releases.
    - There is skepticism in the community regarding the usability of any 'open weights' OpenAI might release, with a comment suggesting the potential use of a proprietary .openai file format and requirements to use OpenAI's own application for model inference, possibly limiting third-party experimentation or deployment and raising concerns about genuine openness.
- [**OpenAI OS model info leaked - 120B & 20B will be available**](https://i.redd.it/08m94pio0dgf1.jpeg) ([Score: 429, Comments: 138](https://www.reddit.com/r/LocalLLaMA/comments/1mepz8z/openai_os_model_info_leaked_120b_20b_will_be/)): **A leaked image (see [here](https://i.redd.it/08m94pio0dgf1.jpeg)) reportedly reveals configuration details of OpenAI's upcoming 'OS' language models, specifically a 120B parameter model and a 20B parameter model. A posted config for the 120B model indicates it uses a Mixture of Experts (MoE) architecture:** `36` **hidden layers,** `128` **experts with** `4` **experts per token,** `201088` **vocab size,** `2880` **hidden size,** `64` **attention heads,** `8` **key-value heads, and RoPE positional encoding with scaling factors. These specs suggest a highly scalable, high-context transformer design similar to recent Megatron or DeepSpeed MoE models.** Commenters note the 20B model size is attractive for research/deployment, and speculate on openness/censorship and performance compared to other recent large models. Some point out this leak may result from a temporary internal error, underscoring the sensitivity of such information.
    - A leaked config file for the OpenAI "OS" 120B model reveals architecture details: 36 hidden layers, 128 experts (Mixture-of-Experts), experts_per_token set to 4, and a vocab size of 201,088. Key parameters include hidden/intermediate size of 2880, 64 attention heads (8 key/value heads), a 4096 initial context length, and advanced rotary positional encoding (rope_theta: 150000, rope_scaling_factor: 32.0).
    - The config suggests use of Mixture-of-Experts (MoE) with 128 experts and 4 experts per token, an approach aimed at improving efficiency for larger models. The RoPE (Rotary Positional Embedding) enhancements (notably rope_ntk_alpha and rope_ntk_beta) and sliding window attention could support longer context handling and scaling.
    - Discussion references a [user who managed to access the 120B weights](https://x.com/main_horse/status/1951201925778776530), indicating early external analysis is underway. Comparisons with recent open-source models (e.g., yofo-deepcurrent, yofo-riverbend) are anticipated, with technical curiosity about performance, context management, and censorship levels.
- [**The “Leaked” 120 B OpenAI Model is not Trained in FP4**](https://i.redd.it/g1yk8r6b8ggf1.jpeg) ([Score: 231, Comments: 64](https://www.reddit.com/r/LocalLLaMA/comments/1mf3tm9/the_leaked_120_b_openai_model_is_not_trained_in/)): **The image is referenced in a discussion debunking a claim that a 'leaked' 120B OpenAI model is trained in FP4 (4-bit floating point), with the title clarifying it is not. The discussion and comments indicate skepticism about hype or misinformation around the technical details of the supposed model, emphasizing the need for critical analysis of such rumors within the AI community. There is no evidence or benchmark presented to support the claim of FP4 training, and the post mainly serves as a rebuttal to unverified leaks.** Commenters dismiss the original claim as 'bullshit hype' and express skepticism about the rumor, echoing broader concerns about AI misinformation and unsubstantiated leaks.
    - Several comments highlight skepticism around the FP4 training claim, referencing that the supposed OpenAI 120B model "leak" is likely hype and not technically credible, noting that FP4 is not recognized as a practical precision format for training (current standard being bfloat16 or FP16 for large models).
    - Others emphasize the importance of model release quality over speed, comparing the situation to DeepSeek r2's delay, and underscoring that 'frontier' AI labs like OpenAI prioritize robustness and performance in their models over early access or hype-driven releases.
    - There is discussion of the recent frequency of large model releases and how increased openness—even from major labs like OpenAI—raises the bar for transparency and competition in the open weight community, helping to normalize open model sharing as a standard industry practice.

### 2. Qwen3 Model Launches and Benchmarks

- [**Qwen3 Coder 480B is Live on Cerebras ($2 per million output and 2000 output t/s!!!)**](https://www.cerebras.ai/blog/qwen3-coder-480b-is-live-on-cerebras) ([Score: 372, Comments: 123](https://www.reddit.com/r/LocalLLaMA/comments/1mf399p/qwen3_coder_480b_is_live_on_cerebras_2_per/)): **Cerebras has launched deployment of the Qwen3 Coder 480B model, an open-source large language model for code generation, offering output at** `2 USD per million tokens` **and** `2000 output tokens/sec` **throughput. This positions it as a potential competitor to Sonnet, especially given claims of being** `~20x` **faster and** `~7.5x` **cheaper on US infrastructure. New tiered coding plans were also announced: 'Code Pro' at** `50 USD/month` **(1000 requests/day) and 'Code Max' at** `200 USD/month` **(5000 requests/day).** Technical commenters note that the '1000 requests per day' is limiting for users of code tools with high request frequency, and some dispute the model's stated performance gap, claiming in practice Qwen3 is not just '5-10%' worse than competitors but shows larger discrepancies in real-world coding tasks.
    - Systems like Roocode and Opencode using Qwen3 Coder 480B on Cerebras see extremely fast response speeds—so fast that UI processing in some tools (like Roocode) cannot keep up, and with others (Opencode), the outputs appear almost instantly. This highlights practical throughput meeting, or exceeding, the advertised 2000 output tokens per second benchmark.
    - There is discussion about the pricing structure, suggesting that $50/month for 1000 requests per day may not be cost-effective for all users, especially when technical workflows (like code lookups or tool calls) generate a large number of requests, as each interaction ('tool call and code lookup') can result in separate API invocations, quickly consuming the quota.
    - Comments caution about the risk of vendor lock-in, recommending users benefit from the current performance and value proposition, but remain aware of the potential for future ecosystem restrictions or dependency on a single provider, even as Cerebras seeks rapid adoption through aggressive pricing or performance leadership.
- [**Qwen3-Embedding-0.6B is fast, high quality, and supports up to 32k tokens. Beats OpenAI embeddings on MTEB**](https://www.reddit.com/r/LocalLLaMA/comments/1mf6bkl/qwen3embedding06b_is_fast_high_quality_and/) ([Score: 143, Comments: 16](https://www.reddit.com/r/LocalLLaMA/comments/1mf6bkl/qwen3embedding06b_is_fast_high_quality_and/)): **Alibaba's Qwen3-Embedding-0.6B (available on Hugging Face: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) delivers high-performance semantic embeddings with a large context window (up to 32k tokens) and reportedly 'beats OpenAI embeddings on MTEB' benchmarks. Users highlight the importance of updating Text Embedding Inference to version 1.7.3 to fix pad token bugs impacting results in earlier versions; such preprocessing/tokenization issues may affect different inference toolchains. Context: Embedding models like Qwen3 are used for semantic search via document/query vector similarity (dot/cosine product), and Qwen3-Embedding-0.6B is commended for accuracy and speed, enabling new use cases at smaller model scales.** Commenters suggest the reranker variant (Qwen3-Reranker-0.6B-seq-cls: https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls) offers ultra-fast and highly relevant scoring for RAG chatbot pipelines, implying broad utility for retrieval-augmented generation (RAG) workflows.
    - Qwen3-Embedding-0.6B is praised for semantic search use cases, leveraging document and query embeddings with dot product or cosine similarity for ranking. It outperforms OpenAI embeddings on MTEB benchmarks, indicating high performance for tasks involving embedding-based retrieval and ranking.
    - The Qwen3-Reranker-0.6B variant is noted for delivering extremely fast inference and high-quality relevance scores in Retrieval-Augmented Generation (RAG) chatbots, as evidenced by user testing and its availability on Hugging Face ([Qwen3-Reranker-0.6B-seq-cls](https://huggingface.co/tomaarsen/Qwen3-Reranker-0.6B-seq-cls)).
    - While Qwen3-Embedding-0.6B is strong for English (and expectedly Chinese), it's reported to be less effective for other multilingual scenarios. Competing models like MPNet may offer better performance on diverse multilingual tasks.
- [**Qwen3-235B-A22B-2507 is the top open weights model on lmarena**](https://x.com/lmarena_ai/status/1951308670375174457) ([Score: 122, Comments: 12](https://www.reddit.com/r/LocalLLaMA/comments/1mf0qlf/qwen3235ba22b2507_is_the_top_open_weights_model/)): **Qwen3-235B-A22B-2507 is now ranked as the highest-performing model with open weights on lmarena, surpassing even closed models like Claude-4-Opus and Gemini-2.5-pro based on lmarena's current evaluation metrics. The model utilizes a 235B parameter architecture, and strong performance is confirmed in user reports for the UD-Q4_K_XL quantization, as well as on external benchmarks such as artificial analysis and livebench.** Commentators express some skepticism regarding lmarena's evaluation methodology; there's also anticipation regarding future models (e.g., OpenAI 120B MoE, GLM-4.5 Air) potentially challenging Qwen3-235B's dominance.
    - Qwen3-235B-A22B-2507 is currently leading lmarena's open-weights model leaderboard, with user feedback noting its strong performance and depth, particularly when running in quantized formats such as UD-Q4_K_XL. The discussion also highlights the community’s anticipation for upcoming models, specifically OpenAI's open-weight 120B MoE and GLM-4.5 Air, with the latter expected to be more competitive once supported by llama.cpp.
    - Skepticism is expressed regarding lmarena’s evaluation methodology, especially as Qwen3-235B reportedly outperforms proprietary models like Claude-4-Opus and Gemini-2.5-pro. This raises questions about model benchmarking standards and result reliability within community-run testbeds.
    - Qwen3 performance is also validated by its top ranking on Artificial Analysis and LiveBench for non-reasoning tasks, and its variant (Qwen3 Coder 480B) is noted for high placement on Design Arena, only trailing behind Opus 4 and surpassing all other open-weight models. This suggests Qwen's release cadence has yielded state-of-the-art open models across multiple technical benchmarks.

### 3. DocStrange Open Source Data Extraction Release

- [**DocStrange - Open Source Document Data Extractor**](https://i.redd.it/vghke2r1ycgf1.gif) ([Score: 149, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1mepr38/docstrange_open_source_document_data_extractor/)): **The image advertises DocStrange, an open-source Python library for extracting data from documents in multiple formats (PDF, images, Word, PowerPoint, Excel), offering outputs such as Markdown, JSON, CSV, and HTML. The tool supports user-defined field extraction (e.g., specific invoice attributes) and enforces output schema consistency with JSON schemas. Two modes are available: a cloud-based mode for quick processing via API (raising privacy cautions for sensitive data) and a local mode for privacy and offline computation (CPU/GPU supported). Related resources: [PyPI link](https://pypi.org/project/docstrange/), [GitHub repo](https://github.com/NanoNets/docstrange).** Commenters highlight the importance of true visual language model (VLM)-driven image description (not basic OCR), as supported by competitors like Docling and Markitdown. Privacy concern is noted for the cloud API: sensitive documents should not be uploaded without care.
    - Users highlight the direct competition with existing document extraction tools, specifying that advanced differentiation hinges on handling images using Vision-Language Models (VLMs) for descriptive image understanding (not just OCR). Tools like Docling and Markitdown are cited as benchmarks for such capabilities, raising the question of whether DocStrange can offer equivalent or superior VLM-driven image description features.
    - Technical scrutiny emerges around how DocStrange compares to simply leveraging local LLMs (e.g., Gemma 3, Mistral Small 3.2, Qwen 2.5 VL) with vision processing, querying if the same extraction (Markdown/JSON/CSV outputs) could be achieved with a targeted prompt and local model, thus questioning the need for a separate cloud-based solution.
    - There is a caution around data privacy given DocStrange's cloud API is the default processing mechanism, as instant conversion requires sending documents to external servers—users are warned not to upload sensitive or personal data unless they trust the service.
- [**Gemini 2.5 Deep Think mode benchmarks!**](https://i.redd.it/8wnv6pme9egf1.png) ([Score: 247, Comments: 66](https://www.reddit.com/r/LocalLLaMA/comments/1meu3jn/gemini_25_deep_think_mode_benchmarks/)): **The image (not viewable here) is described as benchmark results for Google Gemini 2.5's Deep Think mode, which appears to target high-intensity or detailed LLM tasks. The discussion highlights that Deep Think mode is currently limited to Gemini Ultra subscribers. One user compared Gemini 2.5 Deep Think to ChatGPT's deep research capability, finding Gemini's responses more impressive for complex tasks such as PC build recommendations and business idea analysis.** Some commenters question the utility of Deep Think mode due to its exclusivity to Gemini Ultra, and there is mention of "AIME saturation in 2025"—possibly referring to anticipated compute or advanced AI model availability. The comparison with ChatGPT Plus outlines substantive performance preference for Gemini in specific research scenarios.
    - One user reports informal benchmarking between ChatGPT Plus (with deep research) and Gemini 2.5 Deep Think, using prompts to generate a high-performance LLM-capable PC build within a £1200 budget and business analysis. They found Gemini 2.5 delivered more impressive and detailed outputs compared to their prior experiences with ChatGPT, suggesting practical differences in real-world prompt handling and decision-making capability between models.
    - There's interest in benchmarking Gemini 2.5 Deep Think mode on previously unsolved complex mathematical problems, with at least one user seeking to evaluate its performance on extremely challenging math queries. This underscores an active technical curiosity in how Gemini 2.5 competes against the top LLMs in advanced STEM reasoning tasks, a known weakness of many models.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Gemini 2.5 Deep Think Launch and Performance Benchmarks

- [**Gemini 2.5 Deep Think solves previously unproven mathematical conjecture**](https://www.reddit.com/r/singularity/comments/1metslk/gemini_25_deep_think_solves_previously_unproven/) ([Score: 645, Comments: 49](https://www.reddit.com/r/singularity/comments/1metslk/gemini_25_deep_think_solves_previously_unproven/)): **A YouTube video claims that Google's Gemini 2.5 Deep Think has solved a previously unproven mathematical conjecture, but the post and video omit details on the specific conjecture solved. Discussion emphasizes the significance of this claim but laments the lack of transparency regarding which mathematical problem was addressed and how the model accomplished the proof.** Commenters raise concerns about the vagueness around the conjecture, urge for direct testing of difficult math problems on the model, and compare Google’s approach favorably to OpenAI in terms of early release/access, while questioning cost and actual model parity with the unreleased IMO Gold models.
    - A user notes that while OpenAI has developed but not released their specialized IMO (International Mathematical Olympiad) model, Google is making their own advanced model—close to IMO Gold level—available sooner, highlighting differences in openness and release strategy. However, the commenter points out concerns about the high price and compute requirements for accessing Google's Gemini 2.5 model, which may limit practical usage and experimentation for many.
    - There is significant technical interest in benchmarking Gemini 2.5 against other state-of-the-art models on complex mathematical problems, as indicated by requests for access to the model for testing on difficult math conjectures. This suggests the community is eager to rigorously evaluate Gemini 2.5’s mathematical problem-solving capabilities and compare its performance to previous models, particularly on problems that have resisted previous AI approaches.
- [**Gemini 2.5 Deep Think rolling out now for Google AI Ultra**](https://9to5google.com/2025/08/01/gemini-2-5-deep-think/) ([Score: 294, Comments: 23](https://www.reddit.com/r/singularity/comments/1metnbi/gemini_25_deep_think_rolling_out_now_for_google/)): **Google is rolling out "Gemini 2.5 Deep Think" for its AI Ultra tier, suggesting enhanced capabilities over previous versions, but with reported limited daily usage. Technical details on new features or architectural differences from Pro are not specified in the post or linked benchmarks.** Commenters raise concerns about the value proposition of Deep Think, highlighting restrictive usage caps ("a few uses per day") versus high cost (`$250`), and request clarification on technical distinctions from the Pro tier.
    - A user queries the key technical distinction of "DeepThink" compared to the existing "Pro" level, suggesting a need for clarification regarding the feature set, access limits, or underlying model differences between the two offerings. This indicates confusion or lack of transparency about what specific improvements (e.g., reasoning depth, context window expansion, or inference speed) DeepThink delivers.
    - Another user expresses skepticism about the pricing and usage model, highlighting that DeepThink allows only a limited number of uses per day despite a high price tag ($250). This points to a potential technical or infrastructural limitation being masked as a product tier (possibly related to model inference costs, resource allocation, or queueing for Ultra-level compute).
- [**The Architecture Using Which I Managed To Solve 4/6 IMO Problems With Gemini 2.5 Flash and 5/6 With Gemini 2.5 Pro**](https://i.redd.it/h05504ijvegf1.png) ([Score: 216, Comments: 24](https://www.reddit.com/r/singularity/comments/1mewr06/the_architecture_using_which_i_managed_to_solve/)): **The post presents an architecture, depicted in this diagram [here](https://i.redd.it/h05504ijvegf1.png), designed for maximizing IMO problem-solving with Gemini 2.5 models. The approach involves parallel hypothesis generation, with dedicated Prover and Disprover agents producing 'information packets' fed into Solution and Refinement agents. The Refinement agent self-verifies responses, improving solution rigor and completeness—addressing past shortcomings and enabling 4/6 solutions with Gemini 2.5 Flash and 5/6 with 2.5 Pro. The repo at [Iterative-Contextual-Refinements](https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements) includes the architecture and general-purpose prompts, which were then tailored to IMO use-cases, placing emphasis on novelty, rigorous proof standards, and avoidance of approach fixation.** Comments query why major companies don't adopt similar parallel-agent architectures with smaller models, questioning compute inefficiency and the novelty of such techniques compared to industry efforts.
    - A commenter provides a [GitHub repository](https://github.com/ryoiki-tokuiten/Iterative-Contextual-Refinements) detailing their iterative contextual refinements architecture. The architecture evolved from a basic strategies/sub-strategies generation pipeline to a parallel hypothesis generation approach, where prover and disprover agents produce information packets, which are then processed by a refinement agent. The refinement agent performs self-verification of solutions, which was especially effective for Gemini 2.5 Flash versus previous versions. Enhancements included stricter, more IMO-focused prompt engineering, encouraging novel and diverse strategies, hypothesis consideration, and rigorous solution standards.
    - One technical point raised is the observation that instead of using large-scale models and compute, the problem could see similar or better results by using smaller models working in parallel on sub-tasks. This questions the efficiency of current AI research strategies in scaling up model and compute sizes for complex problem solving.
    - A particular prompt constraint specified in the repo demands strict role separation for hypothesis generation versus problem solving. The architecture required prompts that enforced agents (LLMs) not to solve or verify hypotheses, but to solely generate strategic conjectures, indicating non-trivial behavioral control issues with LLMs and the necessity for explicit, strong task-separation instructions in prompt engineering.
- [**Deep Think benchmarks**](https://www.reddit.com/r/singularity/comments/1mettph/deep_think_benchmarks/) ([Score: 189, Comments: 69](https://www.reddit.com/r/singularity/comments/1mettph/deep_think_benchmarks/)): **A benchmark summary (shared via image) highlights the performance of Deep Think, Google's latest large language model, with notably high scores—especially on the International Mathematical Olympiad (IMO) dataset, indicating significant breakthroughs in mathematical reasoning. Early technical commentary underscores the model's strong results across mathematical and logic-heavy benchmarks, suggesting it rivals or surpasses state-of-the-art in these domains. Associated visual data points to impressive quantitative improvements, with particular attention to math-related tasks, but more detailed breakdowns would be needed for granular analysis.** Top comments express surprise at the outstanding math benchmark scores, particularly for IMO, indicating that Deep Think could set a new standard for automated reasoning. There is anticipation regarding its broader practical capabilities in comparison to contemporaries.
    - There's an explicit call for benchmarking Deep Think specifically against higher-tier models such as "O3-Pro" and "Grok 4 Heavy," suggesting that direct comparison to standard or base versions is insufficient for accurate assessment of performance in this context.
    - The math benchmark scores for Deep Think are noted as exceptionally strong, implying that the model may have distinct capabilities or optimizations in mathematical reasoning tasks, which could differentiate it in technical or academic applications.
    - One technical perspective highlights that for a new model to be considered relevant, it must outperform existing leading models in at least a few benchmark areas—in addition to aspects like cost and convenience—underscoring the highly competitive nature of LLM benchmarks.
- [**Damn Google cooked with deep think**](https://i.redd.it/39yx7k6p9egf1.jpeg) ([Score: 378, Comments: 127](https://www.reddit.com/r/Bard/comments/1meu3ce/damn_google_cooked_with_deep_think/)): **The post appears to reference a new feature or capability from Google called 'deep think', possibly an AI-powered tool or model. The image (not viewable) likely shows a screenshot of this feature in action and captures its user interface or pricing information. The top comments point out that the feature is paywalled behind a $250 per month subscription tier, indicating significant cost and possible limitations on general access. There is also a question about whether it is available to 'Ultra subscribers', suggesting different product access levels in Google’s offerings.** Commenters criticize the steep $250/month paywall and discuss the rollout strategy, implying Google’s selective or high-cost approach to advanced AI capabilities. One user speculates that the feature's timing may be significant in response to moves by competitors.
    - Multiple commenters highlight that Deep Think (presumably a new AI capability or model from Google) is currently locked behind an Ultra subscription, which reportedly costs `$250/month`, severely restricting access to only high-paying users or organizations. This paywall raises questions about the democratization of advanced AI tools compared to offerings from other providers.
    - Technical discussion centers on the limited availability: some ask whether the new feature is live for all Ultra subscribers or if there are further restrictions or rollout limitations, suggesting a possible phased or invite-only access model.
    - There is speculation on release timing, implying strategic alignment by Google to coincide with competitive events or announcements, though no benchmarks or technical performance details are discussed in the initial comments.
- [**Gemini 2.5 Deep Think rolling out now for Google AI Ultra**](https://9to5google.com/2025/08/01/gemini-2-5-deep-think/) ([Score: 191, Comments: 64](https://www.reddit.com/r/Bard/comments/1metlhp/gemini_25_deep_think_rolling_out_now_for_google/)): **Google has begun rolling out Gemini 2.5 Deep Think for its 'AI Ultra' tier, aiming to offer significantly improved reasoning and context retention capabilities. The rollout appears limited, with reports that only a small number of prompts per day currently leverage the new model—an apparent bottleneck in deployment or resource allocation.** Top comments express frustration at the limited availability of the new model (few prompts/day), and dissatisfaction with Google AI's refund policy for subscriptions, suggesting user support and access scalability are ongoing issues.
    - Users note that Gemini 2.5 Deep Think for Google AI Ultra currently limits the number of prompts per day, which impacts usability for heavy users and contrasts with the core offering of significant cloud storage capacity. This limitation is a key technical constraint for those seeking consistent access to advanced models.
    - There is discussion about subscription and refund policies, emphasizing user frustration when a model update (Gemini 2.5 Deep Think) is released right after a non-refundable subscription cancellation. This highlights the importance of transparent release timelines and refund processes for AI model rollouts.
- [**Oh damn Gemini deep think is far better than o3 ! Wen gpt 5??**](https://i.redd.it/glxrobh0aegf1.jpeg) ([Score: 162, Comments: 51](https://www.reddit.com/r/OpenAI/comments/1meu4ii/oh_damn_gemini_deep_think_is_far_better_than_o3/)): **The image appears to compare the performance of Google's Gemini Deep Think and GPT-4 (referred to as o3), with the post author expressing surprise at Gemini's superior performance over GPT-4 and questioning the release of GPT-5. Commenters note that such a comparison should use GPT-4 Pro (o3 Pro) as the benchmark for fairness, and raise skepticism about Gemini's capabilities in practical question-answering and coding tasks. The context suggests the image may show benchmark results or qualitative comparisons between the AI models, possibly to promote Gemini's Ultra tier.** A key debate in the comments centers on the fairness of comparing Gemini Deep Think to standard GPT-4 instead of GPT-4 Pro, and the real-world relevance of Gemini's claimed superiority in coding and question-answering. Some users express skepticism about Google's advancements and maintain preference for OpenAI's models, highlighting anticipation for GPT-5's release.
    - Commenters note that comparisons should be made to Gemini Deep Think vs. o3-Pro, rather than the base o3, as Pro is a more direct competitor in benchmarks and capability. Several users stress that meaningful performance discussions require comparing matching tiers (i.e., premium versions).
    - One user critiques Gemini Deep Think's real-world utility for question answering and agentic coding, claiming it does not outperform o3-Pro in those areas and indicating they've downgraded back. There is skepticism about the practical improvements delivered by Gemini's Ultra tier as well.
    - Comparative technical discussion also mentions that Gemini Deep Think is purportedly not better than Grok 4 Heavy in HLE (likely a benchmarking or eval context), and that its performance may only match o3-Pro, implying parity rather than meaningful superiority in major practical tasks.
- [**Gemini 2.5-pro with Deep Think is the first model able to argue with and push back against o3-pro (software dev).**](https://www.reddit.com/r/Bard/comments/1mf0co7/gemini_25pro_with_deep_think_is_the_first_model/) ([Score: 156, Comments: 40](https://www.reddit.com/r/Bard/comments/1mf0co7/gemini_25pro_with_deep_think_is_the_first_model/)): **The post reports that Gemini 2.5-pro with Deep Think (Google) is the first LLM able to robustly challenge and analytically push back against claims from o3-pro (OpenAI), particularly in technical software development tasks involving complex reasoning. In a test case involving npm package selection—where o3-pro suggested a complicated workaround to a deprecated package's vulnerability—Gemini 2.5-pro correctly recommended a safer, simpler alternative and, when presented with o3-pro's counterargument (disguised as a human's suggestion), offered a detailed, critical rebuttal focused on root cause analysis and sound package choice. This behavior contrasts with earlier models, which typically acquiesce to o3-pro's arguments, indicating Gemini's improved adversarial and debate capabilities.** Commenters encourage rigorous, diverse testing, referencing notable mathematical challenges (e.g. the Latin Tableau Conjecture) and suggesting ensemble approaches (e.g. MCP with all major reasoning LLMs voting on solutions) for benchmarking adversarial reasoning and mathematical proof generation across top models.
    - One user highlights Gemini 2.5-pro's ability to tackle unsolved mathematical problems, such as the Latin Tableau Conjecture (LTC), arguing that its performance is at or above IMO level and wants it rigorously tested against detailed mathematical prompts. Specific known computational boundaries are noted (verification up to 12x12 Young diagrams), along with references to combinatorial mathematics literature and a challenge to provide mechanizable proofs or concrete counterexamples.
    - Another technically insightful suggestion proposes running all top-tier reasoning models (Claude, Gemini, GPT-4, etc.) in parallel on complex tasks and having them vote on solutions, pointing out that such ensemble approaches could push the quality of results but would be expensive—potentially requiring enterprise resources to implement.
    - A limitation of Gemini 2.5-pro's Deep Think mode is raised: users are currently limited to 10 uses per day, which hinders comprehensive or iterative testing compared to o3-pro, which allows more extensive free interaction, thus affecting practical productivity in research or benchmarking settings.

### 2. WAN 2.2, Flux Krea, and Current Text-to-Image/Video Model Comparisons

- [**While as a video model it's not as special, WAN 2.2 is THE best text2image model by a landslide for realism**](https://www.reddit.com/gallery/1mek9go) ([Score: 478, Comments: 138](https://www.reddit.com/r/StableDiffusion/comments/1mek9go/while_as_a_video_model_its_not_as_special_wan_22/)): **WAN 2.2 is highlighted as a leading text-to-image (T2I) model, outperforming alternatives like Flux and Chroma in realism, texture detail, and minimal censorship, and demonstrates synergistic performance when combined with Instagirl 1.5. The OP notes underwhelming performance and stability for video generation compared to 2.1, but exceptional results in T2I tasks, especially across noise levels. Linked [Civitai examples](https://civitai.com/images/91302771) demonstrate its output fidelity.** Commenters contend OP's video model criticism is likely due to suboptimal settings or reliance on "speed up loras", asserting WAN 2.2 is highly competitive as a video model given optimal configuration, and highlight its performance for both free and open source use cases.
    - Users highlight that **WAN 2.2** achieves remarkable photorealism in text-to-image tasks, provided that default workflows are adjusted—specifically cautioning against using speed-up LoRAs which may degrade performance to that of WAN 2.1, thus stressing the need for tuning model settings for optimal results ([sample images](https://civitai.com/images/91302771), [further examples](https://civitai.com/images/91319618)).
    - There is a technical debate about the significance of WAN 2.2's improvements as a video model: some users argue that the upgrade represents a substantial leap for open-source models in video generation, while others suggest the claimed improvements are overstated or inadequately demonstrated without robust benchmarks or comparisons.
    - A dissenting technical perspective questions the model's realism in output, implying that despite positive community perception, generated images still fall short of true photographic authenticity and may exhibit typical synthetic artifacts seen in other models.
- [**Pirate VFX Breakdown | Made almost exclusively with SDXL and Wan!**](https://v.redd.it/svpf4s6ydggf1) ([Score: 529, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1mf4q8k/pirate_vfx_breakdown_made_almost_exclusively_with/)): **The post details a professional VFX workflow using generative AI tools: reference frames were created from stills using SDXL (for improved ControlNet integration), actor segmentation employed both MatAnyone and After Effects' rotobrush (noted for better hair masking), and the backgrounds were replaced with 'Wan', which has been optimized for high-quality video inpainting. The pipeline demonstrates seamless integration of multiple AI models for compositing and background replacement tasks, highlighting tangible improvements in video post-production efficiency and realism.** Commenters emphasize the value of professional, non-trivial AI use in creative industries and express interest in deeper process disclosure, contrasting it with less technical AI applications.
    - A 3D artist expresses strong interest in a detailed process breakdown, indicating there's technical curiosity about how SDXL and Wan are specifically used in the VFX pipeline—this suggests relevant insights could include workflow integration, steps taken, and how these tools compare to traditional 3D workflows.
    - Another commenter highlights how professional filmmakers using SDXL and Wan for VFX can produce film-quality scenes on a low budget, implying these models substantially lower production costs and raise expectations for affordable, high-quality digital content.
- [**Flux Krea can do more then just beautiful women!**](https://www.reddit.com/gallery/1meuxfz) ([Score: 397, Comments: 88](https://www.reddit.com/r/StableDiffusion/comments/1meuxfz/flux_krea_can_do_more_then_just_beautiful_women/)): **The post discusses the capabilities of the AI generative model Flux Krea, highlighting that it can generate diverse outputs beyond the common 'beautiful women', including complex scenarios such as 'ingame screenshots' and various 'war pictures', including those with 'gore'. The model is contrasted with Wan 2.2, suggesting Flux Krea specializes in a broader or different range of visual content types, particularly those resembling 'dashcam' and 'war journalist' photography.** Comments reference concerns about misinformation/propaganda potential due to realistic image generation, diversity in output style (e.g., tanks, villages, Minecraft themes), and the sensitivity of certain generated content ('not safe for Warzone'), reflecting debates on AI image generation ethics and risks.
    - A commenter references "village and minecraft generations," suggesting Flux Krea's capabilities in generating complex and varied scene layouts beyond portraits—a technical testament to its dataset diversity and semantic composition control.
    - Another user humorously notes, "We're going to need a bigger video card," which indirectly references the computational intensity and high GPU memory requirements typical of running large, advanced image generation models like Flux Krea, especially for higher resolution or batch inference tasks.
- [**Flux Krea is a solid model**](https://www.reddit.com/gallery/1mes891) ([Score: 228, Comments: 48](https://www.reddit.com/r/StableDiffusion/comments/1mes891/flux_krea_is_a_solid_model/)): **The post reviews the Flux Krea image generation model, highlighting native image output at 1248x1824 resolution, using the Euler/Beta sampler and a CFG (Classifier-Free Guidance) of 2.4. The model demonstrates improved facial diversity and chin structure compared to previous versions like Flux Dev, though outputs are still noticeably artificial.** Commenters note a persistent pale yellow tint and excessive freckles in outputs, suggesting potential issues with training data or style bias (possibly Unsplash-based). There's also critique regarding a lack of sample diversity in shared outputs, suggesting evaluations should span landscapes, animals, and architecture for more comprehensive model assessment.
    - Users report a consistent issue with a pale yellow tint in Krea model outputs, indicating a possible training data bias or color processing problem. Several commenters specifically compare this to outputs from models trained on Unsplash data, speculating similar source influences.
    - Attempts to fine-tune Krea (e.g., using LoRA) to counteract the tint and excessive freckles have been met with limited success, suggesting that artifacts are deeply baked into the model's learned representations, making post-training correction challenging.
    - Some users note that Krea's generated faces resemble those from SD1.5, implying similarities in data distribution or architectural approach, and point out a lack of output variety (e.g., limited non-human subjects), which raises questions about the model's generalization beyond typical human close-ups.
- [**Wan 2.2 Text-to-Image-to-Video Test (Update from T2I post yesterday)**](https://v.redd.it/f9tb1fl46fgf1) ([Score: 230, Comments: 49](https://www.reddit.com/r/StableDiffusion/comments/1mey8vc/wan_22_texttoimagetovideo_test_update_from_t2i/)): **The post presents a test of Wan 2.2's Image-to-Video capabilities using prior text-to-image outputs, running at native 720p. The author emphasizes preservation of detail and realism (notably human figures) with minimal camera motion and no post-processing, and notes upscaling to 1080p was solely for improved Reddit compression. This builds on previous [text-to-image comparisons](https://www.reddit.com/r/StableDiffusion/comments/1mec2dw/texttoimage_comparison_flux1_krea_dev_vs/) with Flux Krea, showcasing the model's output consistency across modalities.** Top commenters consider this demonstration as the strongest empirical showcase of Wan 2.2's generative video capabilities to date, specifically praising the accurate 'physics' and dynamic realism in the converted video, which suggests advances in model architecture or training regarding temporal and spatial coherence.
    - One user highlights improvements in workflow efficiency when using Wan 2.2 as opposed to previous approaches: in Wan 2.1, their process entailed generating multiple 1080p stills per scene, selecting the best frames, then upconverting to video (720p) using img2video with motion-enhanced prompts. This reportedly yields superior detail and *saves time compared to direct text-to-video generation*, which often produces lower-quality results.
    - The model's handling of 'physics' is praised—implying enhancements in temporal consistency or object dynamics within generated video compared to earlier models. Users note the realism of scene and object movement, suggesting advances in video synthesis beyond just frame interpolation.
    - A comparison is drawn to 'Veo 3,' implying Wan 2.2 delivers performance or workflow capabilities reminiscent of Google's advanced video models, but presumably with more accessible or homebrew technology.
- [**Wan2.2 I2V 720p 10 min!! 16 GB VRAM**](https://v.redd.it/xi64nzktmcgf1) ([Score: 159, Comments: 23](https://www.reddit.com/r/StableDiffusion/comments/1meoraj/wan22_i2v_720p_10_min_16_gb_vram/)): **The OP reports running the merged Wan2.2 I2V model ([phr00t's 4-step all-in-one merges](https://www.reddit.com/r/StableDiffusion/comments/1mddzji/all_in_one_wan_22_model_merges_4steps_1_cfg_1/)) at 1280x720 (81 frames, 6 steps, CFG=2) on a 16GB VRAM card using the Kijaiwarpper workflow, achieving generation in 10-11 minutes with moderate RAM usage (**`~25-30 GB`**, compared to** `~60 GB` **in standard Kijaiwarpper). The workflow avoids Out-of-Memory (OOM) issues when unable to use the standard dual-model setup, and reports lower (but not significantly) image quality than the official Wan2.2 site output (1080p, 150 frames at 30 fps), but much improved speed/efficiency and a substantial qualitative jump over 2.1. Full workflow details are shared via [Pastebin](https://pastebin.com/RtRvEnqj), and comparisons are made to VEO3.** Top comments discuss persistent OOM issues in Kijaiwarpper/Comfy workflows, with some users noting the workflow sometimes loads both models simultaneously instead of sequentially, leading to VRAM overflow and severe performance drops on the low-noise pass. Hardware-specific generation speed is also discussed, e.g., a 4070 Ti Super needing 15-20 min for 5s at 480p, suggesting substantial variation based on VRAM and workflow specifics. An uncompressed video showcase is linked for inspection.
    - Users report significant out-of-memory (OOM) issues when running the Kijai Workflow with block swapping in Comfy; despite enabling block swapping, sometimes both models are loaded simultaneously instead of sequentially, causing VRAM overflow into system RAM. This leads to acceptable performance only during high noise sampling (when the VRAM can hold the entire model), but performance degrades badly—sometimes to a complete stall—during low noise steps due to reliance on slower RAM.
    - Performance observations detail that an RTX 4070 Ti Super can take `15-20 minutes` to render a `5 second 480p` clip, highlighting substantial computational demands and suggesting dramatically longer times for higher resolutions or longer durations. Another user with an RTX 5060 Ti 16GB and matching system RAM experiences hard crashes, indicating possible incompatibilities or insufficient resources despite minimum VRAM seemingly being met.
    - There is mention of uncompressed output video, but the primary technical focus is on model adaptability and severe resource requirements or instability across different hardware, underscoring the need for further workflow optimization or clearer documentation regarding performance and hardware compatibility.
- [**Testing WAN 2.2 with very short funny animation (sound on)**](https://v.redd.it/b51wd0oeqegf1) ([Score: 143, Comments: 16](https://www.reddit.com/r/StableDiffusion/comments/1mew37c/testing_wan_22_with_very_short_funny_animation/)): **The post presents a test of WAN 2.2 for text-to-video (T2V) and image-to-video (I2V) continuation, with output rendered at 720p. The poster notes that while artifact issues persist in version 2.2, prompt following has improved. There are no reported changes in model architecture, and artifact reduction remains an ongoing limitation.** A commenter asks about technique—whether I2V continuation was achieved by using the last frame of the previous WAN 2.2 output—with the poster implying that simply increasing frame count degrades video quality further. Another comment humorously notes prompt adherence issues, suggesting generation fidelity is still imperfect.
    - One commenter inquires about animation continuity, asking if the next video starts from the last frame of the previous one. They note that when they attempted to render with more frames, the results actually looked worse, implying possible model or rendering limitations when dealing with frame sequences.

### 3. OpenAI & AI Industry Model/API Rumors and Announcements

- [**GPT-5 is already (ostensibly) available via API**](https://www.reddit.com/r/OpenAI/comments/1mettre/gpt5_is_already_ostensibly_available_via_api/) ([Score: 581, Comments: 191](https://www.reddit.com/r/OpenAI/comments/1mettre/gpt5_is_already_ostensibly_available_via_api/)): **A Reddit user reports access to a model labeled** `gpt-5-bench-chatcompletions-gpt41-api-ev3` **via the OpenAI API, suggesting it may be an *ostensible* early release of GPT-5. The model's naming convention indicates adaptation to the GPT-4.1 API (for backwards compatibility) but possibly introduces new API parameters, as noted by the commenter that *"it only supports temp=1 and modern parameters"*. Linked [logs and screenshots](https://preview.redd.it/glxute607egf1.png?width=1181&format=png&auto=webp&s=a8a6928801e2d7bf0f6a30122471a33eb3fa092d) show API activity and OpenAI Console output before OpenAI disabled access.** Commenters validated its capability with creative tasks: producing a detailed SVG image ([example](https://preview.redd.it/int18mghqegf1.png)) and a feature-rich HTML/CSS/JS landing page in a single shot ([sample output](https://preview.redd.it/2qtz7nep3fgf1.png)), reporting qualitative improvements over GPT-4/4.1, especially in creative and structural code generation.
    - Users report that the API (allegedly GPT-5) demonstrates strong capabilities in both creative coding and design generation, such as producing a consistent, visually polished iGaming landing page in a single completion, meeting detailed prompt requirements (responsive layout, modern CSS, JavaScript interactivity with no frameworks, all inline assets). This level of output—"oneshotting"—suggests a notable improvement over GPT-4, especially in specification-following and code quality.
    - Technical details note that the API only supports `temperature=1` and modern parameter sets, which could indicate an updated or experimental deployment compared to earlier GPT-4 endpoints. This parameter limitation itself may provide circumstantial evidence that it's a distinct model or experimental branch.
    - There is skepticism and semantic debate about whether the model is truly GPT-5, highlighting that OpenAI's naming conventions are not always transparent: the model is described as 'supposedly' or 'ostensibly' GPT-5, meaning users cannot independently verify its underlying architecture and are relying on external indicators (e.g., prompt performance, API metadata) rather than formal announcement.
- [**OpenAI's new open source models were briefly uploaded onto HuggingFace**](https://i.redd.it/rsi9rxz9ldgf1.png) ([Score: 182, Comments: 37](https://www.reddit.com/r/singularity/comments/1mersom/openais_new_open_source_models_were_briefly/)): **The post discusses a leak where OpenAI's new open-source models, reportedly with parameter sizes of 20B and 120B, were briefly uploaded to HuggingFace. The most notable technical details from the image and comments are the model hyperparameters: 36 hidden layers, 128 experts with 4 per token (Mixture-of-Experts architecture), vocab size 201,088, hidden/intermediate size 2,880, 64 attention heads, 8 key-value heads, 4096 context length, and specific rotary position embedding (RoPE) configurations (e.g., rope_theta 150000, scaling_factor 32). This hints at a large-scale, expert-mixture transformer model likely optimized for efficiency and performance at large parameter counts.** Key technical debate centers on whether the 20B parameter model effectively supports tool calling and code use cases, indicating community interest in its practical integration capabilities rather than just size. There is also speculation on model architecture versus other open-source models.
    - A user shares a detailed architecture breakdown of the model, indicating parameters such as `num_hidden_layers: 36`, `num_experts: 128` (suggesting a Mixture-of-Experts architecture), `experts_per_token: 4`, `hidden_size: 2880`, and attention configuration details (e.g., `num_attention_heads: 64`, `num_key_value_heads: 8`, `sliding_window: 128`, and `initial_context_length: 4096`). These details are relevant for understanding the scale and structure of the model.
    - The discussion references two model sizes—20B and 120B parameters—implying both a large-scale and a more accessible model, with users expressing interest in using the 20B variant for tool calling and code tasks, indicating practical consideration of hardware requirements versus capability.
- [**OpenAI are preparing to launch ChatGPT Go, a new subscription tier**](https://i.redd.it/c4ouejprhdgf1.jpeg) ([Score: 230, Comments: 70](https://www.reddit.com/r/OpenAI/comments/1merfyd/openai_are_preparing_to_launch_chatgpt_go_a_new/)): **The image (https://i.redd.it/c4ouejprhdgf1.jpeg) presents a teaser or leak regarding a new OpenAI product called 'ChatGPT Go', implied as a forthcoming subscription tier. The main technical context, corroborated by comments, is speculation that this tier will fit between the Free and Plus offerings, purportedly priced at $9.99/month, potentially introducing a 'pay as you go' billing model—suggesting greater flexibility or usage-based pricing compared to current plans. The community is attempting to infer both features and pricing from the image and announcement leak.** Discussion in comments centers on price speculation (from $10 to $2,000/month) and the possible shift to a usage-based subscription ('pay as you go')—but no concrete technical details or official benchmarks have been confirmed yet.
    - Users speculate about the pricing and feature differentiation of "ChatGPT Go," suggesting it may fill the gap between the free and Plus tiers, possibly at a $9.99/month price point, and raising the question of whether its feature set will be closer to the more limited free tier or the enhanced GPT Plus tier.
    - One comment discusses the potential introduction of ads to the free tier as an offset for new subscription models, which would alter the current monetization strategy of OpenAI's ChatGPT offerings. This highlights a broader trend in SaaS monetization where free services are subsidized by ads or tiered features.
    - There is a question regarding resource allocation and access, with one user explicitly asking if "Go" will provide fewer capabilities or model access compared to GPT Plus, indicating interest in technical limitations or distinctions between the subscription levels (e.g., availability of GPT-4, limits on usage, or priority access during high-load periods).
- [**Anthropic just dropped 17 videos to watch**](https://www.reddit.com/r/ClaudeAI/comments/1meko92/anthropic_just_dropped_17_videos_to_watch/) ([Score: 761, Comments: 139](https://www.reddit.com/r/ClaudeAI/comments/1meko92/anthropic_just_dropped_17_videos_to_watch/)): **Anthropic has released 17 new YouTube videos (approx. 8 hours total) via their official channel ([link](https://www.youtube.com/@anthropic-ai/videos)), potentially offering detailed technical insights into their latest research, model demos, safety implementations, or product updates. This structured video drop may suggest a coordinated knowledge-sharing or marketing push targeting both developers and the wider AI community.** The most technically relevant comment highlights an issue with YouTube's video watch rate limiting, which could hinder researchers' ability to view large sets of content quickly. Another commenter mentions leveraging third-party summarization tools (e.g., Comet AI browser) for efficient information extraction, alluding to existing bottlenecks in manual video consumption.
    - A commenter highlights that Anthropic had an internal usage leaderboard to track employees' token consumption, revealing competitive non-research use among staff—one employee admitted leading in token usage without contributing code or direct company value. This is contrasted with criticism directed at external users for overuse, raising questions about the rationale and messaging behind Anthropic's recent usage-limit policy.
    - There is an expressed disappointment that Anthropic's new videos lack deep technical detail, with a skeptical query about whether the company is shifting away from emphasizing research-centric content, possibly influenced by industry trends led by figures like Elon Musk.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Flash Preview 05-20
> 

**Theme 1. Frontier LLM Developments & Speculation**

- **GPT-5 Mystery Deepens: Panic Drop or Polite Improvement?** Speculation surrounds **GPT-5's** release, with views split on whether it will be a full "panic drop" due to scaling limits or a smaller, more focused model. One user briefly spotted **GPT-5 in the API** ([the GPT-5 API sighting](https://x.com/chetaslua/status/1951301385292493259)) before its swift removal, fueling further speculation about its eventual unified, omnimodal nature.
- **Horizon Alpha Rises: Free Model Crushes Paid LLMs!** **Horizon Alpha** outperforms paid LLMs via the **OpenRouter API**, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/). Users laud its superior shell use and task list creation in orchestrator mode, with some speculating it's an **OpenAI**style **120B MoE** or **20B** model.
- **Gemini's Generation Gets Glitchy, Pricing Gets Punished!** Some members report **Gemini's** repetitive behavior and note video limits dropped from **10 to 8**. The community widely criticizes **Gemini Ultra's** **$250/month** plan, which imposes a meager **10 queries per day**, calling it a "scam" and "daylight robbery."

**Theme 2. Open-Source & Local LLM Optimization**

- **Qwen Models Push Quantization Limits, Codeium Flies!** Discussions focus on optimal quantization for **Qwen3 Coder 30B** (Q4_K_M gguf slow, UD q3 XL for VRAM) and issues with tool calling. **Qwen3-Coder** now runs at approximately **2000 tokens/sec** on Windsurf, fully hosted on US servers.
- **Unsloth Finetuning Unleashes New Speed and Power!** Unsloth now supports **GSPO** (an update to GRPO), which works as a TRL wrapper, and dynamic quantization can be replicated with the `quant_clone` [small application](https://github.com/electroglyph/quant_clone). Members are exploring continuous training of LoRAs and have used Unsloth to build a [Space Invaders game](https://invaders.smolit.us/space_invaders/).
- **LM Studio: Offline Dreams Meet Online Nightmares?** Users anticipate **image-to-video prompt generation** and image attachments for offline use, preferring it over cloud-based alternatives like **ChatGPT**. However, security vulnerabilities for connecting to the **LM Studio API** across networks are a concern due to unverified security.

**Theme 3. AI Coding & Agent Tooling**

- **Aider Dominates Code Editing, DeepSeek Delivers Big!** Users praise **Aider** for its superior control and freedom, with one estimating it completed *one week of programming work in a single day for just $2* using **DeepSeek**. Speed comparisons with **SGLang** and **Qwen** also show high performance, reaching **472 tokens/s** on an **RTX 4090**.
- **AI Agents Branch Out, Go On-Chain!** Developers are building **on-chain AI agents** for trading and governance using **Eliza OS** and **LangGraph**, alongside efforts to create an **OSS model training script** for natural cursor navigation. Discussions also highlight **AnythingLLM** ([AnythingLLM tweet](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19)) for ensuring **data sovereignty** in agentic systems.
- **MCP Tools Level Up: Security, Payments, and JSON Processing!** A new **security MCP check tool** ([GitHub repo](https://github.com/minte-app/security-mcp-check)) seeks feedback, while **PayMCP** offers a payment layer for MCP servers with [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations. A **JSON MCP Server** ([GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter)) further aids LLMs in efficiently parsing complex JSON files, saving valuable tokens and context.

**Theme 4. Hardware & Performance Benchmarking**

- **AMD's MI300X Flexes on Nvidia, GEAKs Out!** New **MI300X FP8 benchmarks** ([the MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/)) suggest **AMD's MI300X** outperforms **NVIDIA's H200** in certain tasks, with performance approaching the **B200**. **AMD** also introduced **GEAK benchmarks** and a **Triton Kernel AI Agent** ([the GEAK paper](https://arxiv.org/abs/2507.23194)) for AI-driven kernel optimization.
- **Nvidia Drivers Go 580.88: Fixes for Fast Motion!** **Nvidia** released driver **580.88** quickly after **577.00**, a **9-day-old driver**, to fix a potential issue with GPU video memory speed after enabling **NVIDIA Smooth Motion**. Discussions also cover solving CUDA compiler issues with `__launch_bounds__` to determine register count at entry, despite `setmaxnreg` still being ignored.
- **Builders Debate Multi-GPU Setups, Eye VRAM Savings!** Discussions include recommendations for motherboards like the [MSI X870E GODLIKE](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) for dual **3090s**, comparing **Mac mini M4** to **RTX 3070**, and exploring the feasibility of partial **KV Cache Offload** in **LM Studio** to optimize VRAM usage. Efforts continue on **DTensor** and basic parallelism schemas, inspired by [Marksaroufim's visualizations](https://www.youtube.com/@marksaroufim).

**Theme 5. AI Product Pricing & User Experience**

- **Perplexity Pro Rolls Out Comet, Glitches on iOS, Goes Free for Millions!** **Perplexity** is slowly distributing **Comet Browser** invites to **Pro users**, but **iOS image generation** faces recurring issues where attached images are not incorporated. Notably, over **300 million Airtel subscribers in India** are receiving **free Perplexity Pro for 12 months**.
- **Kimi K2 Turbo Goes Ludicrous Speed, Drops Prices!** The Moonshot team announced **Kimi K2 Turbo**, boasting **4x the speed** at **40 tokens/sec** with a **50% discount** on input/output tokens until **Sept 1** at [platform.moonshot.ai](http://platform.moonshot.ai/). A new [Moonshot AI Forum](https://forum.moonshot.ai/) also launched for technical discussions, complementing Discord's "vibin for memes" atmosphere.
- **API Errors and Sky-High Costs Plague AI Users!** **Gemini Ultra's Deep Think** plan faces ridicule for its **10 queries/day limit at $250/month**, prompting comparisons to more reasonably priced alternatives. Users also report persistent **API errors** and **timeouts** with **OpenRouter** models like **Deepseek v3 free** (often overloaded) and the **Cohere API**.



---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Comet Browser Invites Trickle Out**: **Perplexity** is slowly distributing **Comet Browser** invites, prioritizing **Pro users**.
   - Users report varied wait times, suggesting Pro users can share up to **2 invites** to speed up the process.
- **Perplexity Pro Image Generation Fails on iOS**: Users are reporting that **Perplexity Pro on iOS** fails to incorporate attached images during image generation, creating recurring issues.
   - The model summarizes requests without generating images from the attachments, even after starting new chats.
- **Airtel India Subscribers Score Free Perplexity Pro**: **Airtel** subscribers in India (over **300 million people**) are receiving **Perplexity Pro** for free for **12 months**.
   - The promotion is exclusive to Airtel subscribers located in India.
- **GPT-5 Release Date: Still Shrouded in Mystery**: Speculation surrounds the release of **GPT-5**, with conflicting views on whether it will be a full release or a smaller, more focused model.
   - One user claimed to have briefly seen **GPT-5 in the API** ([source](https://x.com/chetaslua/status/1951301385292493259)), but it was quickly removed, fueling further speculation.
- **Search Domain Filter Confounded**: A **Perplexity Pro** subscriber reported that the **search_domain_filter** is not functioning as expected despite the feature not being in beta.
   - Another member requested a copy of the user's request for further investigation and assistance.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-5: Panic Drop or Polite Improvement?**: Members are speculating if **GPT-5** will be a *panic drop* due to **OpenAI's** limitations in scaling, along with diminishing returns from **Chain of Thought (CoT)**.
   - Claims suggest **CoT** is a *complete dead end*, proposing direct network feedback of the model's vector output instead of using tokens for thinking.
- **Qwen3 tests quantization limits**: Discussions revolve around the best quantization for **Qwen3 Coder 30B**, with reports on **Q4_K_M gguf** being slow in **Ollama**, while others prefer **UD q3 XL** for VRAM savings.
   - One member runs the April **Qwen3-30b-a3b** model at **40k** in **vllm** on a **3090** 24/7, awaiting a 4-bit AWQ version for the coder model.
- **Unsloth now supports GSPO**: After Qwen proposed **GSPO** as an update to **GRPO**, members clarified that **GSPO** already works in **Unsloth** and it is a wrapper that will auto-support **TRL** updates.
   - Although **GSPO** is slightly more efficient, members did not note any significant updates to performance.
- **VITS Learns to Breathe**: A member training a **VITS checkpoint overnight** shared that **model quality depends on epochs and dataset quality**, and **VITS excels at speaker disentanglement**.
   - Furthermore, they discovered **VITS encodes raw audio into latent space** for realistic recreation and can learn subtleties like breathing at commas with annotation and ran into memory issues on iOS.
- **Dynamic Quantization gets Quant Clone**: A member created [a small application](https://github.com/electroglyph/quant_clone) to quantize finetunes the same way as Unsloth's dynamic quantization, wanting to replicate it on their own finetunes.
   - A user reported high refusal rates in their **Gemini** finetunes, and found *Gemini to be quite obnoxious* in that regard.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Arena Enhancements Aim to Assist**: Members suggested adding buttons for **Search, Image, Video, and Webdev Arena** to boost visibility, and also suggested adding tooltips to the leaderboard explaining how **Rank, CI, and Elo** are determined, sharing a [concept image](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png).
   - The goal is to assist users in navigating the platforms and understand ranking metrics.
- **Data Concerns: Personal Info Peril**: A user raised concerns about accidentally including **personal information** in published prompts and asked for ability to remove prompts.
   - A member responded that such examples should be DM'd to them for escalation, and acknowledged [sharing these concerns with the team](https://www.deepcogito.com/research/cogito-v2-preview).
- **Gemini's Generation Gets Glitchy**: Some members noted **Gemini** exhibited repetitive behavior, while another questioned if **Gemini 2.5 Flash** fixed the issue and one user noted video limits dropping from **10 to 8**, urging others to use the video generation arena quickly.
   - The community's sentiment is split between experiencing glitches and consistent performance.
- **DeepThink Debut Disappoints?**: With the release of **Gemini 2.5 Deepthink** for Ultra members, members are wondering if it is worth it after seeing **10 RPD limit**.
   - Members called it a **scam** and a daylight robbery, saying it's just a rushed version because of the imminent **GPT-5** release.
- **Veo 3 Visuals Victory**: **Veo 3 Fast & Veo 3** are out with new **Image-to-Video with audio capabilities** within the [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194).
   - The community can now create videos from images using the new `/image-to-video` command in the video-arena channels, with voting open for the best videos.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Vibe Coding Sparks GitHub Needs**: A member inquired about the necessity of **GitHub** for background agents, exclaiming *this thing is sick* alongside an attached image, sparking curiosity about **vibe coding** setups.
   - Another user, having spent **$40** on prompts, sought advice on optimizing their **Cursor** setup, reflecting a common interest in efficient configuration.
- **Cursor Freezing Bug Creates Frustration**: A user reported frequent machine freezes every **30-60 seconds** after an hour of chat use, indicating a persistent **Cursor freezing bug**.
   - A **Cursor** team member recommended posting the issue on the [Cursor forum](https://forum.cursor.com/c/bug-report/6), highlighting the official channels for bug reporting and assistance.
- **Model Spending Compared to Claude Pro**: Users debated the pricing of **Cursor** versus **Claude Pro**, with one stating their preference for the cheapest plans and best models, favoring Claude's **$200** plan.
   - Another user cautioned about escalating costs, reporting spending **$600** in 3 months, emphasizing the need for cost management.
- **Horizon Alpha Experience Divides Users**: One user described their personal experience with **Horizon-Alpha** as *a bit underwhelming*, suggesting mixed reactions to the new feature.
   - Conversely, another user lauded *cursor is the best app i have ever seen*, underscoring the subjective nature of user experiences.
- **Referral Program Requested for Cursor**: Members have inquired about a referral program for **Cursor**, with one user claiming to have onboarded *at least 200+ people by now sitting in discords lmao*, indicating significant community-driven adoption.
   - A link to the [Cursor Ambassador program](https://cursor.com/ambassador) was shared, providing an alternative avenue for rewarding community contributions.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Function Calling APIs Trump XML Workarounds**: Function Calling APIs have **inherent value** over structured XMLs, which are often used as a workaround when models like **Qwen** don't support native tool calling.
   - Inline tool calls maximize interoperability for coding models like **Qwen**, even with minor inefficiencies.
- **Zuckerberg's AI Sparks Bio-Weapon Concerns**: **Mark Zuckerberg's** AI superintelligence initiative raised concerns about potential bio-weapon creation, and one member warned against releasing superintelligence to the public.
   - Members also expressed concern that *controlling minds with fake users and carefully crafted language* could be more dangerous than bio-weapons.
- **GPT-5 Faces Delay, Grok4 Takes the Crown?**: Rumors suggest **GPT-5's** delay is due to an inability to surpass **Grok4**, but [OpenAI plans to combine multiple products into GPT-5](https://link.to/openais-next-foundational-model).
   - Clarification was given that **GPT-5** will be a single, unified, omnimodal model.
- **Horizon Alpha Outshines Paid LLMs**: **Horizon Alpha** is outperforming paid LLMs via the OpenRouter API, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/).
   - Its shell use and task list creation in orchestrator mode are superior to other models, though some speculate it *could always be something turbo weird we’re not thinking of like codex-2*.
- **Large Context Windows Spark Debate**: Despite **Gemini's** 1 million context window, legacy codebase issues were better solved with **Claude** and **ChatGPT**, sparking debate on whether [large context windows are overrated](https://nealgoogs.website).
   - Some prefer models with smaller context windows and better output, while others insist larger windows are crucial for agentic applications to *remember and weave in far‑back details automatically*.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Image-to-Video Prompt Generation Dreams in LM Studio**: Members are anticipating future **image-to-video prompt generation** and **image attachment** features in **LM Studio**, favoring offline capabilities over cloud-based alternatives like **ChatGPT**.
   - As an alternative, one member mentioned **ComfyUI**, noting it might not be optimized for **AMD** cards.
- **LM Studio's Roadmap: A Mystery**: The community discussed the absence of a **public roadmap** for **LM Studio**, with speculation that development plans might be unstructured and unpredictable.
   - A member stated, *no public roadmap so noone knows*.
- **LM Studio API Security Considerations**: Users debated connecting to the **LM Studio API** across a network, highlighting potential security vulnerabilities.
   - Concerns were raised about **LM Studio's** unverified security, cautioning against exposing it without proper risk assessment and network protection.
- **Qwen3 Coder Model Faces Loading Glitches**: Users encountered difficulties when loading the **Qwen3 Coder 30B** model, triggering a *Cannot read properties of null (reading '1')* error.
   - A fellow member suggested an update to version **0.3.21 b2** which claims to have resolved the issue, along with enabling **recommended settings**.
- **Nvidia Bursts Out a Driver**: **Nvidia** released driver **580.88** quickly after **577.00**, a **9-day-old driver** with a fix for a potential issue with GPU video memory speed after enabling **NVIDIA Smooth Motion** [5370796].
   - The user runs the drivers from the cuda toolkit, and doesn't use the fancy control panel or GFE (GeForce Experience).



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **API Errors Plague OpenRouter**: Users reported experiencing **API errors** when using models via the **OpenRouter API**, with one user suggesting checking the **model ID prefix** and **base URL** to resolve the issue.
   - Errors include *no endpoint found* which members suggested was caused by potential misconfiguration.
- **Deepseek v3 Free Model Plagued by Outages**: Users experienced issues with the **Deepseek v3 0324 free** model, including *internal errors*, *empty responses*, and **timeouts**, leading some to switch to the paid version.
   - One member pointed out *free is completely overloaded. paid has none of these issue, and the actual content quality is better.*
- **Horizon Alpha Hailed as Effective**: Users praised the **Horizon Alpha** model for its effective reasoning and good performance.
   - While the model claimed it was developed by **OpenAI**, community members clarified that it was likely a distilled model.
- **Personality.gg Leverages OpenRouter for Roleplay**: [Personality.gg](https://personality.gg) launched a roleplay site using **OpenRouter** for most models, providing access to all 400 models through **OpenRouter PKCE** completely free/cheap.
   - This integration lets users engage in role-playing scenarios with a wide variety of **AI models**.
- **PyrenzAI's UX Wins Praise**: A user complimented the **UI/UX** of [PyrenzAI](https://pyrenzai.com), appreciating its unique look and style, and distinctive sidebar design compared to other apps.
   - Despite speed and security critiques, the application's user interface received positive feedback.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 Goes Ludicrous Speed with Turbo!**: The Moonshot team announced **Kimi K2 Turbo**, touting **4x the speed** at **40 tokens/sec**, with a **50% discount** on input and output tokens until **Sept 1** at [platform.moonshot.ai](https://platform.moonshot.ai/).
   - Users can now experience significantly faster performance thanks to faster hosting of the same model, available via official API.
- **Moonshot AI Launches New Hangout Spot**: Moonshot AI launched the ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/)) for technical discussions, API help, model behavior, debugging, and dev tips.
   - While *Discord’s still vibin for memes* and chill convos, the forum aims to be the go-to spot for serious builds and tech discussions.
- **Kimi K2 Challenges Claude's Reign**: One user reported **Kimi K2** as the first model they can use instead of **Claude**, prompting them to drop **Gemini 2.5 Pro** due to coding, as a kind of information, becoming freer.
   - The user also added that they expect most AIs will converge in terms of knowledge, so the differences between them will start to blur.
- **Kimi K2 Turbo Pricing Details Exposed**: The speedy **Kimi K2 Turbo** is priced at **$0.30/1M** input tokens (cached), **$1.20/1M** input tokens (non-cached), and **$5.00/1M** output tokens with a special promo until Sept 1.
   - This equates to roughly *4x faster for 2x the price* during the discount, tailored for users requiring swift processing.
- **Gemini Ultra's Deep Thinking Costs a Pretty Penny**: Members ridiculed Google Gemini Ultra's plan imposing a **10 queries a day limit for $250/month**, with one user saying it was *very funny and very scummy*.
   - Comparisons were made to **ChatGPT pro** at $200/month which gives unlimited **Office 365 Pro**, and **Claude Max**, seen as more reasonably priced.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes-3 Dataset Refusals Raise Eyebrows**: Members investigated unexpected refusals in the **Hermes-3 dataset** while computing the *imatrix* for quantization, leading to [further dataset investigation](https://huggingface.co/datasets/NousResearch/Hermes-3) to confirm the dataset is devoid of refusals.
   - The team is hoping to confirm that the dataset is devoid of refusals by ensuring the dataset is fully vetted.
- **Unitree's R1 Robot Democratizes Embodied A.I.**: The community explored the **Unitree R1 foundational robot model**, priced at **$5,900**, providing a fully open software development kit (**Python**, **C++**, or **ROS**) for A.I. development, showcased in [this YouTube video](https://www.youtube.com/watch?v=ljo7TjOqRzs).
   - Users stated it is an ideal tool for research teams transitioning to the next evolution of A.I.
- **Horizon Alpha Model Sparks OpenAI Speculation**: Members debated whether the **OpenAI Horizon Alpha model** resembles **OpenAI's** style, speculating it could be a **120B MoE** model with low activation or possibly a **20B** model, noted in [this tweet](https://x.com/apples_jimmy/status/1951180954208444758).
   - Some suggested on [this Reddit thread](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/) that quantization would be impossible if it is **FP4** only.
- **AnythingLLM Advocates for Data Sovereignty**: A user shared a [link to a tweet](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19) about **AnythingLLM** and declared it the future for **data sovereignty**.
   - The user also shared links to **Neuronpedia** and other tweets relating to **data sovereignty** from [Jack_W_Lindsey's tweet](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) and [heyshrutimishra's tweet](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19).
- **OSS Model Training Script Bootstrapped**: A public research engineer has begun developing an **OSS model training script** to help fill the lack of good OSS models for natural cursor navigation.
   - The engineer acknowledged the possibility that websites that block crawling bots may be scraped by new "clones" using this technology.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cline bags $32M for Open-Source AI Coding Agent**: Cline, an AI coding agent, secured **$32 million** in Seed and Series A funding led by **Emergence Capital** and **Pace Capital**, aiming to empower developers with transparent, open-source AI tools, serving **2.7 million** developers with transparent pricing and no upcharging.
   - A **Latent.Space Podcast** episode features **Cline**, discussing its origin, the 'Plan + Act' paradigm, community tools, and future directions with Saoud Rizwan and Pash, available on their [website](https://xcancel.com/latentspacepod/status/1951008883163668522) and [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ).
- **OpenAI's OS Model Details YOFO Leaked**: Details about **OpenAI**'s upcoming OS model, **YOFO**, surfaced after its config was briefly accessible, sparking excitement around rumored **120B** and **20B** parameter variants.
   - A member noted that Jimmy Apples was reluctant to share all configuration details.
- **Anthropic's Claude Generates 22,000-Line Code Update**: Anthropic merged a **22,000-line** change to their production reinforcement learning codebase, largely written by **Claude**, sparking skepticism about the reliability of such a large AI-generated code change, which was largely a **json dsl**.
   - Discussions touched on human review processes and concerns about the reliability of large AI-driven code merges; Sauers confirmed the change was real.
- **Anthropic Blocks OpenAI's Claude API Access**: Anthropic revoked OpenAI's API access to its models, including **Claude**, citing a violation of terms of service.
   - **OpenAI** expressed disappointment, noting that its API remains available to **Anthropic**, leading to community discussions about competitive moves and blurring lines of model training.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Query Expansion Boosts RAG**: Discussion revolved around using [query expansion techniques](https://www.promptingguide.ai/techniques/query_expansion) in **RAG** systems by generating multiple questions from a single user query to improve information retrieval.
   - For the query *'what is the name of the customer'*, expanding it to *'What is the name?'* and *'Who is the customer?'* was suggested.
- **Cross-Encoders Flop at Ranking**: Experimenting with a cross-encoder on **MS MARCO** data for ranking results related to the question *'What is the name of the customer?'* yielded poor outcomes.
   - The expected top hit (*Customer Name*) was ranked lower than (*Definition of Customer*), scoring **-0.67** vs **-1.67**.
- **Fine-Tuning is Key for Retrieval**: Directly training on a retrieval task is essential to control ranking quality, according to [this paper](https://arxiv.org/abs/2212.01349).
   - Members suggested that the optimal similarity metric is task-dependent, implying that general-purpose embeddings may not be sufficient for specialized retrieval scenarios.
- **Gemini 2.5 Flash has Gemma Favortism**: **Gemini-2.5-flash** consistently ranked **Gemma models** higher than other models, even some 70B models.
   - The suspected reason is that the response tone of Gemma models might be more plausible to both humans and LLMs, affecting the ranking.
- **Cinema AI generates cohesive movie scenes**: [TheCinema AI](https://thecinema.ai/) research project focuses on generating movie scenes that maintain **cohesion** with each other, according to the [arxiv paper](https://arxiv.org/html/2507.18634v1).
   - The project explores methods for generating cohesive movie scenes and is detailed in the project's website and paper.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Warriors Demand Offline Access**: Users are seeking ways to save **NotebookLM studio material** for offline access during travel without constant internet connection.
   - One user mentioned downloading audio to an iPad and adding it to PowerPoint slides with family photos.
- **Pro Users Ponder Missing Preview Perks**: Several **Pro account users** report not having access to the **video overview feature**, despite upgrading and others with free accounts having access.
   - A user who briefly had video access lost it after refreshing the page, suggesting ongoing rollout issues.
- **User Dreams of Custom NotebookLM with Gemini**: A user is considering using **Gemini embedding 001** and **Gemini 2.5 models API** to create a custom multi-hop, multi-step reasoning **RAG pipeline** for documents.
   - They aim to surpass **NotebookLM's** capabilities, citing limitations such as the **300-file limit**, lack of transparency in workflow, and limited system instructions.
- **Comet Extension Catapults NBLM into Orbit**: Users discussed **Comet**, a browser extension that can access tabs/history/bookmarks and control the browser, and its potential integration with **NotebookLM** for source finding.
   - The suggestion was raised that **Comet** could potentially code an extension to dynamically add sources to **NotebookLM**.
- **Spanish Audio Overviews Still Short and Sweet?**: A user inquired about why **Audio Overviews** in Spanish remain short in duration, noting a workaround: *switch it to English, change the duration, then prompt it to do it in Spanish*.
   - Another user confirmed that while Portuguese isn't officially supported for explainer videos, they were able to force it to work.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Attention Probes' Performance Proves Polarizing**: EleutherAI's experiments with **attention probes**—tiny neural networks trained to classify transformer hidden states—yielded mixed results, sometimes underperforming standard **linear probes** due to **overfitting** and **optimization issues**, as detailed in their [blog post](https://blog.eleuther.ai/attention-probes/).
   - The code for these experiments has been open-sourced on [GitHub](https://github.com/EleutherAI/attention-probes/), inviting community exploration and refinement to uncover potential improvements.
- **Low-Power LLMs Brave Seabed Scenarios**: A member is deploying **LLMs** on low-power edge devices offshore for seabed mapping, environmental monitoring, and autonomous systems, focusing on **mission planning**, **anomaly detection**, and **smart data compression**.
   - Scientific modeling is currently limited by latency and bandwidth constraints, but the team is actively exploring ways to overcome these **challenges**.
- **Gemini-2.5-flash Judges Gemma Generation**: A member observed that **Gemini-2.5-flash** consistently ranked **Gemma** responses higher when comparing various LLMs, suggesting a potential *family bias* or superior performance of **Gemma3** models.
   - This observation has sparked discussion around the fairness and objectivity of LLM evaluation metrics, as well as the competitive landscape of open-source models.
- **Weight Tying Whips Up Worry**: A member argued that *weight tying is a universally bad practice*, causing inefficiency and instability, and *doesn't even make mathematical sense*, suggesting its detrimental effects on model performance.
   - This assertion sparked debate around the validity of **weight tying** in the broader research community.
- **HF Transformers Tweaks Trigger Tussles**: With **HuggingFace transformers 4.54**, **Llama & Qwen layers** now return residual streams directly (not tuple), which may affect users of `nnsight layer.output[0]`.
   - A member warned that using `nnsight layer.output[0]` will get you the 1st batch element only, not full residual stream, a bug spotted thanks to [nnterp tests](https://butanium.github.io/nnterp).



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider** Still Dominates Code Editing**: Users expressed strong appreciation for **Aider**, citing its superior blend of control and freedom compared to alternatives, with one user estimating **Aider** completed *one week of programming work in a single day for just $2* using **DeepSeek**.
   - Another user emphatically stated, *"Aider rules so hard"*, underscoring its effectiveness in code editing tasks.
- **SGLang** and **Qwen** Break Speed Barrier**: One user reported achieving speeds of **472 tokens/s** using **sglang** and **Qwen 0.6B Q8** on LM Studio with an **RTX 4090**, contrasting with the **330 tokens/s** achieved on regular LM Studio.
   - Another user expressed interest in replicating this local-only setup, particularly since **vllm** performed slower on their **4090** compared to Ollama, showing curiosity in trying *llama.cpp*.
- **Debating Motherboards for Multi-GPU**: Discussion covered hardware configurations, with one member recommending [this MSI motherboard](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) for dual **3090s** inside a Fractal North XL case.
   - Others shared their own setups, including servers with **3 L4s** and **T40s**, and diverse case options like the **Meshify2**.
- **Claude Code** Suffers from High Token Count**: Members compared **Claude Code** to other frontier models, noting that its performance degrades significantly beyond **64k tokens**, especially compared to **o3** and **Gemini 2.5 Pro**.
   - It was also mentioned that *the system prompt consumes a substantial portion of the available context window*.
- **Benchmarking **Qwen3 30B** locally**: One member sought an easy way to benchmark 8 different quants of **Qwen3 30B A3B Coder** locally using **LM Studio**.
   - Another member suggested utilizing *llama.cpp server + docker aider benchmark on the same computer* and referenced a writeup on getting **Gemini 2.5 Pro** working.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Security MCP Checker Seeks Feedback**: A member shared a [GitHub repo](https://github.com/minte-app/security-mcp-check) for a **security MCP check tool**, requesting community feedback.
   - This tool aims to assist users in identifying potential vulnerabilities in their **MCP** servers.
- **PayMCP Payment Layer Enters the Ring**: A new **payment layer** for **MCP**, dubbed **PayMCP**, is under development, with [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations available.
   - The creator seeks collaborators and early adopters to explore its capabilities in facilitating payment acceptance on **MCP** servers.
- **PageRank for MCP Servers Quest Begins**: A member inquired about **PageRank** implementations for **MCP** servers, with the goal of ranking servers based on utility.
   - Suggestions included a [repository of MCP tools](https://github.com/YogiSotho/mcp-tools-collection) and the [MCP registry](https://github.com/modelcontextprotocol/registry) as valuable resources.
- **JSON MCP Server Cleans House**: A **JSON MCP Server** emerged to aid **LLMs** in efficiently parsing large and complex **JSON** files like **Excalidraw exports**, documented in this [GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter).
   - The solution employs **schema generation** to understand the **JSON** structure and extract necessary data, cutting down on **tokens** and **context**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Hylo Language Draws 'Heterogenous PL' Parallels**: The **Hylo** programming language ([https://www.hylo-lang.org/](https://www.hylo-lang.org/)) gains attention for its approach to memory safety via **value semantics** and scheduling, compared to **Halide** and **Mojo**.
   - Members reported that the person responsible for **Hylo** is currently working on **Scala 3/Scala Native**, noting that the leads come from **cpp** and **Swift**
- **AMD Drops Kernel AI Agent & GEAK Benchmarks**: AMD introduced the **GEAK benchmarks** and **Triton Kernel AI Agent** in their paper [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194).
   - Explore AMD's novel approach to **AI-driven kernel optimization** using their new **Triton Kernel AI Agent** for kernel optimization.
- **__launch_bounds__ setting launches CUDA fix**: A user fixed an issue where the compiler couldn't determine register count at entry by passing `minBlocksPerMultiprocessor` to `__launch_bounds__`, setting `maxThreadsPerBlock=128*3` and `minBlocksPerMultiprocessor=1`.
   - The `setmaxnreg` setting is still being ignored, now due to a different problem related to compatibility with an `'extern'` call.
- **MI300X Benchmarks Leave H200 Behind**: A user inquired about experiences with new [MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) on AMD hardware.
   - The benchmarks compare **AMD's MI300X** with **NVIDIA's H200** and suggest the MI300X outperforms the H200 in certain FP8 data-parallel tasks, with performance approaching **NVIDIA's B200**.
- **picocuda compiler Makes Strides Toward GPU Land**: Progress is being made on the [picocuda](https://github.com/j4orz/picocuda) compiler and [elements](https://github.com/j4orz/elements) graph data structures projects, according to members in the singularity-systems channel.
   - The textbook will roughly follow the [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) from CGO '16.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Flux Krea is Out, NSFW is Not**: The new **Flux Krea** model has been released, [available here](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8) promising *much more detail* and compatibility with most lora on base.dev.
   - Early reports indicate that **NSFW** content generation is *not possible*.
- **Emergence AI Emerges Victorious**: **Emergence AI**'s architecture achieved [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory) on the **LongMemEval benchmark**, which evaluates long-term memory in AI agents.
   - This positions **Emergence AI** as a leader in memory benchmarks.
- **Smolagents Goes JavaScript**: A member has released **smolagents.js**, a **TypeScript** port of **smolagents**, available on [GitHub](https://github.com/yusuf-eren/smolagents.js) and [npm](https://www.npmjs.com/package/smolagents.js).
   - This port allows developers to use **smolagents** in **JavaScript** environments.
- **Discriminator Learning Rates Fine-Tuned**: Members discussed **debugging GANs** by lowering the **discriminator learning rate** to identify issues, suggesting observing loss changes at very low values like **1e-5**.
   - The goal is to determine if the discriminator's loss collapsing to **0** stems from a learning rate imbalance.
- **Qwen and DeepSeek-R1 Step Up**: Faced with blocked access to **Llama 4**, use **Qwen** or **DeepSeek-R1** as a replacement while running *dummy_agent_library.ipynb* on Colab.
   - These models are considered viable alternatives when access to **Llama 4** is restricted.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere Context Window Size: 128k In, 8k Out!**: A user noticed a context window discrepancy, with the **Hugging Face model card** stating **32k context** while **API docs** claim **128k**. The team clarified that it's **128k in** and **8k out**.
   - Cohere team members promised to update the Hugging Face model card.
- **Rate Limits Thwart Hackathon Hopes!**: **Team Patriots**, participating in the **HackRx 6.0 AI hackathon**, faced rate limit issues with the **10 calls/minute trial key limit**.
   - A Cohere team member granted permission to create multiple accounts and cycle the keys to overcome the limit, suggesting rate limits are a known hurdle.
- **Startup Sweet on Cohere's Reranker Seeks Enterprise!**: A startup, enthusiastic about Cohere's **Reranker implementation**, expressed interest in an **Enterprise plan** due to exceeding the **1000/min limit** for the production API.
   - Cohere directed them to email details about their use case to support@cohere.com and varun@cohere.com for secure assistance.
- **Samsung's AI Architect Enters the Chat!**: An AI architect from **Samsung Biologics** introduced themself, focusing on integrating **AI methods and tools** and running a private **LLM service with RAG** for internal use.
   - They are looking to discuss **biopharmaceutical or biological challenges**.
- **Cohere API Hit with Timeouts!**: A user in #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/) reported receiving multiple timeout errors when querying the API.
   - The user was not given any feedback within the chat.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Spammer still spams**: A member reported receiving DM spam and requested an admin to perma-ban the user who is still active.
   - No action was taken during the period, and the spammer continues to spam.
- **Wide Research, is it wide?**: A member inquired about initial takes on using **Wide Research**.
   - No reviews of **Wide Research** were given.
- **Cloudflare config stuck, help needed**: A member is experiencing issues configuring a virtual environment within **Cloudflare**.
   - The setup keeps getting stuck on **Cloudflare**, preventing them from completing the virtual environment configuration.
- **Credits crash, users lash**: A member reported that daily refresh credits are no longer working, indicating issues with the platform's credit system.
   - Another user mentioned having their account suspended despite not breaking any rules, indicating possible issues with account management.
- **Layoffs likely lose refunds**: A member pointed out recent layoffs and suggested the user probably won't get their money back.
   - The comment implies that recent layoffs at the company may impact the ability to process refunds or resolve financial issues.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Joins Forces with Novita Labs**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951315242904068483) announces the integration of **LlamaIndex** with **Novita Labs** model inference capabilities.
   - This integration provides diverse data source connections and transformation into vector embeddings.
- **Gemini Speaks TypeScript Fluently**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951342252346974431) announces **Gemini Live integration** now available in **TypeScript**.
   - A demo is provided showcasing how to set up and run a basic terminal chat.
- **Engineer Crafts AI On-Chain**: A Senior AI & Blockchain Engineer is building **on-chain AI agents** for trading, media automation, and autonomous governance using **Eliza OS**, **LangGraph**, and custom toolchains.
   - This engineer has worked extensively across **Base**, **Solana**, **Berachain**, **Sui**, **Aptos**, **HBAR**, **EVM chains**, and cross-chain systems.
- **Git-Style Branching for LLM Conversations**: A member is experimenting with a system where each message is a node, enabling branching off at any point in the conversation to create new context paths, as detailed in [their blogpost](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp).
   - The system currently uses **Gemini API**, with plans to include **GPT-4**, **Claude**, and local **LLaMA** models, seeking testers for feedback.
- **Llama Parsers take fare share of time to parse**: Members discussed the performance of **LlamaIndex parsers** for **.doc**, **.pdf**, and **.ppt** files, particularly when dealing with text embedded in images.
   - Solutions proposed include using **LlamaParse** in premium mode, converting PPTs to PDFs for improved speed, or implementing **ThreadPoolExecutor()** for asynchronous document parsing.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSpill verb is coined for Yaron Minsky**: Members discussed who would *give it a second try to **DSpill Yaron Minsky / quant bros***, leading to a new verb '**DSpill**'.
   - The term '**DSpill**' was proposed to describe action against **Yaron Minsky** and the **quant bros**.
- **DSPy is now RL!**: A member shared [a blogpost](https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html) about using Reinforcement Learning in DSPy to improve writing quality.
   - No discussion happened, but could be interesting for those looking to optimize their generations.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Install Issues Merit GitHub Attention**: A member faced **Mojo** installation difficulties and contemplated opening a **GitHub issue** to report the problem.
   - Another member advised them to create a **GitHub issue** with detailed logs to assist developers in diagnosing and resolving the installation problem efficiently.
- **Logs are a Developer's Best Friend**: The discussion highlights the importance of including detailed logs when reporting **Mojo** installation issues on **GitHub**.
   - Providing comprehensive logs enables developers to diagnose and resolve the problem more efficiently by providing necessary information for debugging.
- **Print Statements Inhibit Tail Call Optimization?!**: A member observed that adding basic **print/log statements** to functions prevents **tail call elimination**.
   - The discussion is about how the addition of **print/log statements** affects **tail call elimination** in minimal **Mojo** examples and seeks to understand the underlying reasons for this behavior.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **OpenAI's Model Leaks with 128 Experts**: A rumored **OpenAI** model with **128 experts** and **120B parameters** has potentially leaked.
   - The model's weights are reportedly in **FP4** format, suggesting a compressed state.
- **Deep Dive into Mixture of Experts**: **Mixture of Experts (MoE)** models use multiple sub-networks (experts) with a gating network to route inputs.
   - This architecture enables scaling model size without a proportional increase in compute costs, making it an active area of research.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC Quizzes with Answer Keys Now Available**: An archive of the **quizzes with answer keys** is now accessible in the *"Quizzes"* section of the course website.
   - This gives students a resource to review course material and assess their understanding.
- **Google Forms to Remain Closed**: Course staff announced that they cannot reopen the **Google Forms** used for quizzes.
   - Students who missed taking quizzes via **Google Forms** should use the available archive for review.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Qwen3-Coder Surfs into Windsurf at Breakneck Speed**: **Qwen3-Coder** is now available in Windsurf, operating at approximately **2000 tokens/sec**.
   - Announced via [X](https://x.com/windsurf/status/1951340259192742063) and [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button), the model is fully hosted on US servers.
- **Windsurf's Newest Resident: Qwen3-Coder**: Windsurf now houses **Qwen3-Coder**, boasting a blazing speed of **2000 tokens per second**.
   - The implications of this new model are being discussed on [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button).



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Developer Seeks Opportunities**: alex_sdk4 inquired whether anyone is seeking a developer.
   - No further details regarding specific skills, projects, or expectations were provided.
- **Follow up: Developer Seeks Opportunities**: Since alex_sdk4 reached out, this may be a good opportunity for smaller tasks.
   - Potential clients can reach out directly to alex_sdk4.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1400553929611280499)** (1048 messages🔥🔥🔥): 

> `Comet Browser Invites, Image Generation Issues on Perplexity Pro, Free Perplexity Pro for Airtel Subscribers in India, GPT-5 Release Speculation, Model Performance Comparison` 


- **Comet Browser Invites Rolling Out Gradually**: Perplexity is rolling out **Comet Browser** invites almost daily, prioritizing **Pro users**, but the wait time may vary.
   - Some users suggest that if your daughter has a Pro account, she can send you up to **2 invites**.
- **Image Generation Glitches Plague Perplexity Pro**: A user reports that image generation on **Perplexity Pro for iOS** fails to incorporate attached images, and another user confirms this is a recurring issue.
   - The model summarizes the request but doesn't generate an image based on the attached file, and starting a new chat does not consistently resolve the problem.
- **Airtel Subscribers in India Snag Free Perplexity Pro**: A user mentioned that **300 million people in India** get Perplexity Pro for free for **12 months** if they are Airtel subscribers.
   - To use the promo you have to be located in India and be an Airtel subscriber.
- **GPT-5 Release Date Remains a Mystery**: Users speculate about the release of **GPT-5**, with one suggesting it could be next week, but another member insists that it will probably be some kind of mini model lol.
   - One user had briefly seen **GPT-5 in the API**, but it was quickly removed ([source](https://x.com/chetaslua/status/1951301385292493259)).
- **Model Performance Sparks Debate: Sonnet 4 Reigns Supreme, O3 Holds Its Own**: Users discuss their experiences with various models, with **Sonnet 4** being praised for coding and value, while **O3** is recommended for reasoning ([cplx.app](https://www.cplx.app/)).
   - The discussion touches on tool call issues and the tendency of Anthropic models to *hold back information unless specifically asked*.


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1400597657667244112)** (7 messages): 

> `Shareable threads, RAG without embeddings, Trump-Medvedev` 


- **Thread Sharing Settings Clarified**: A Perplexity AI staff clarified with a user that the thread should be set to `Shareable`.
   - A link was shared about *how to make threads shareable*.
- **OpenAI RAG without Embeddings**: A member shared a [Medium article](https://levelup.gitconnected.com/rag-without-embeddings-heres-how-openai-is-doing-this-45866cd5ddc6) about **RAG without embeddings** and how **OpenAI** is doing this.
   - It was written by **Gaurav Shrivastav**.
- **Trump-Medvedev drama with 2 nuke subs**: A member shared a [Perplexity search result](https://www.perplexity.ai/search/find-information-about-trump-p-g67iddgiQSe1WR4x6GKNjg#2) about the **Trump-Medvedev drama with 2 nuke subs being positioned near Russia** for a new Human Benchmark Report for August 1st.
   - They shared a [Gemini Canvas infographic](https://g.co/gemini/share/c43c0a891af3) made up for the report itself.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1400582585968496640)** (14 messages🔥): 

> `search_domain_filter, Moderator Bot Usage, Image Uploading via API` 


- **Troubleshoot Search Domain Filter!**: A user reported that the **search_domain_filter** is not being honored, even as a Pro subscriber, requesting insight on enabling the feature.
   - Another member responded saying that it should be working (not in beta), and requested a copy of the request for assistance.
- **Moderator Bot Pricing Questions?**: A student inquired about the usage and pricing for a moderator bot using **Perplexity AI**, anticipating around **200 requests** with less than **100 words** of data each.
   - The user is trying to make a moderator bot using perplexity AI.
- **Image Uploading gives Internal Server Error!**: A user encountered an internal server error (**code 500**) when uploading images as base64 via the API.
   - They then shared their [B4J code](https://www.b4x.com) to demonstrate their method, while a member asked for the request and the model being used.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1400560721535959271)** (1099 messages🔥🔥🔥): 

> `GPT-5 speculation, Qwen3 model, Cogito V2, Unsloth GRPO and TRL, H100 and batch sizes` 


- **GPT-5 Panic Drop Speculation Arises**: Members are speculating whether **GPT-5** will be a *panic drop* or a middle-of-the-road improvement due to **OpenAI's** exhaustion of scaling model size and diminishing returns from Chain of Thought (**CoT**).
   - There are claims CoT is a *complete dead end* and it's possible to achieve the same thing by feeding the model's vector output back through the network directly instead of using tokens for thinking.
- **Qwen3 Quantization and Performance Tests**: There's discussion on the ideal quantization for **Qwen3 Coder 30B**, with some finding the **Q4_K_M gguf** slow when adding context in **Ollama**, while others prefer **UD q3 XL** for VRAM savings.
   - One member reported running the April **Qwen3-30b-a3b** model at **40k** in **vllm** on a **3090** 24/7, while others eagerly await a 4-bit AWQ version for the coder model.
- **Cogito V2 Reinforcement Learning Discussed**: Members discussed the release of **Cogito-v2 GGUFs** and their reinforcement learning approach, with some viewing it as an iteration on existing techniques rather than a novel breakthrough.
   - A member shared an article covering process reward models in 2024 ([synthesis.ai](https://synthesis.ai/2025/02/25/large-reasoning-models-how-o1-replications-turned-into-real-competition/)), and another member shared a **Deepmind** paper from 2022 exploring similar concepts ([arxiv.org](https://arxiv.org/abs/2211.14275)).
- **Unsloth GRPO already supports GSPO**: A member asked about updating **Unsloth** to support **GSPO** learning, after Qwen proposed it as an update to **GRPO**.
   - Another member clarified that **GSPO** is slightly more efficient but that it already works in **Unsloth**, and that Unsloth will auto-support **TRL** updates because it is a wrapper.
- **Rumored New OpenAI Model Sparks Excitement**: Rumors of a new **OpenAI** model are circulating, with some speculating it could be the best Operating System (**OS**) model and beat **SOTA K2** in evaluations.
   - Many are hyped for a potentially dense **20B** base model, which could pair well with existing recipes, while others are curious if it will be dense or another mixture of experts (**MoE**).


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1400858572593565747)** (4 messages): 

> `New member introduction, Community assistance` 


- **New member joins, admits ignorance**: A new member, cyber.n0de, introduced themselves and humorously admitted to being completely clueless.
   - They expressed a need for guidance, signaling a potential opportunity for community assistance and onboarding.
- **Community offers helping hand**: A member, theyruinedelise, promptly responded to the new member's admission of ignorance by offering assistance.
   - This illustrates the community's willingness to support newcomers and provide guidance.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1400780930255163402)** (74 messages🔥🔥): 

> `VITS checkpoint training insights, On-device VITS system on iOS, Children voices recording, Avocodo and iSTFTNet for audio fidelity, Universal vocoder for Speech LLM` 


- **VITS Training Yields Eureka Moments**: After training a **VITS checkpoint overnight**, a member shared insights: **model quality depends on the number of epochs and dataset quality**, and **VITS excels at speaker disentanglement** for creating models with distinct voices.
   - They noted **VITS encodes raw audio into latent space** for realistic recreation and emphasized that it depends on specific needs compared to RVC.
- **VITS Runs into iOS Memory Mayhem**: A member reported that using **VITS for on-device system voice on iOS** faces memory consumption challenges with the **Hifi-GAN decoder**, requiring chunk-wise decoding.
   - They also found **VITS can learn subtleties like breathing at commas** and different styles for quoted text with proper annotation.
- **To Child Voice, Schedule Recording Hours Carefully**: A member expressed uncertainty about the number of hours needed to **record children's voices** for fine-tuning light woman-voices for a better baseline.
   - Another member suggested that 24 hours per speaker is overkill, emphasizing the need for quality data over quantity.
- **Avocodo's Fidelity Facelift Forefronts**: Members discussed **Avocodo** as a means for quick fidelity boosts without significant speed increase, noting reduced artifacts limited to dataset quality, with a link to an unofficial [Avocodo-pytorch implementation](https://github.com/rishikksh20/Avocodo-pytorch).
   - They pointed out that the linked implementation uses **Hi-Fi GAN** but requires training a model yourself.
- **Universal Vocoder Quest Kickstarts**: A member expressed a need for a **universal vocoder** for plugging **VITS into a Speech LLM**, requiring fast speed, low GPU usage, and the ability to train from scratch.
   - One suggestion was [BigVGAN](https://github.com/NVIDIA/BigVGAN), though the original poster wants to train from scratch; others considered the impact of lightweight LLM architecture.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1400554633470152867)** (207 messages🔥🔥): 

> `Circular Import Error, RuntimeError with Merged Model Loading, UV venv performance, Qwen3 tool calling problems, Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf on vLLM` 


- **Circular Import Causes Grief**: One member reported an `ImportError: cannot import name 'convert_lora_modules' from partially initialized module 'unsloth_zoo.vllm_utils'` arising from a **circular import** when using `use_async=True` with `unsloth.FastLanguageModel.from_pretrained`.
- **Special Tokens Trigger Runtime Error**: A member encountered a `RuntimeError` related to **size mismatch** when loading a merged model after fine-tuning and adding **2 special tokens** to the tokenizer and model's embedder.
   - Another member suggested that adding new tokens isn't fully resolved, and the system might still attempt to load the base model's tokenizer; also, using `resize_model_vocab = 128258` may partially solve the issue, but not consistently for merged models, as it may load the base model's tokenizer.
- **UV venv causes performance decrease**: A user experienced a **20x performance slowdown** when using Unsloth within a **UV venv**, which led to extremely slow initialization during cuda graph shape capture.
   - It was suggested that UV might be downloading all xformers versions, causing the slowdown, but a member pointed out that they used mamba instead, to avoid using UV altogether.
- **Tool Calling troubles with Qwen3**: A user reported issues with **Qwen3 30B variants** not reliably performing **tool calling** in their **Langchain app**, unlike previous experiences with Qwen3 4B and larger models, despite using the latest Unsloth versions with Ollama.
   - It was suggested to check `fast_inference=True`, but the user confirmed it was already enabled, then it was suggested to check [this vLLM issue](https://github.com/vllm-project/vllm/issues/12324) related to vLLM and UV.
- **vLLM struggles with GGUF model**: A user encountered a `ValueError: GGUF model with architecture qwen3moe is not supported yet` when attempting to run **Qwen3-Coder-30B-A3B-Instruct-1M-Q8_0.gguf** on **vLLM**.
   - Members suggested that gguf format should rather be run on *llama.cpp* and noted the model architecture may not be supported, prompting a suggestion to install Transformers from source to potentially resolve the issue.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1400764067383083079)** (8 messages🔥): 

> `Unsloth Dynamic Quantization, Qwen3 30B-A3B, Space Invaders refined, Roleplay AI finetuning, Gemini Refusals` 


- **Dynamic Quantization gets Quant Clone**: A member created [a small application](https://github.com/electroglyph/quant_clone) to quantize finetunes the same way as Unsloth's dynamic quantization.
   - They wanted to replicate Unsloth's dynamic quantization on their own finetunes.
- **Unsloth's Qwen3 Coder Model builds Space Invaders**: Using a **Q4_M unsloth Qwen3 30B-A3B coder model** and Cline in VS Code, a member created and refined a Space Invaders style game.
   - The game was completed in about ten minutes without touching a single line of code, and is available [here](https://invaders.smolit.us/space_invaders/).
- **Roleplay AI finetunes with Unsloth**: A member announced an easy way to finetune with Unsloth and provide more data with their [roleplay-ai project](https://github.com/bjoern-buettner/roleplay-ai/tree/the-one/beam-llm-training).
   - The model is available on Hugging Face.
- **Gemini faces High Refusal Rates**: A member asked if others have experienced a higher level of refusal with their finetunes, comparing it to **Gemini**.
   - The member finds *Gemini to be quite obnoxious* in that regard.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1400731806977753219)** (4 messages): 

> `Gemma 3 1B garbage, finetuning project, continuous training of loras` 


- **Gemma 3 1B Flops**: A user trained **Gemma 3 1B** and found it to be *absolute garbage*, and a waste of compute, and is sticking to benchmark-crashing **4B** models.
   - They did not mention the training dataset or training methodology.
- **Fineteuning project in the works**: A user is looking to collaborate on a **finetuning project** using open-source LLMs, with compute available on GCP.
   - They are keen to work on anything from **code models** to domain-specific applications.
- **Continuous LoRA Training Revisited?**: A user inquired about recent work on continually updating the weights of a model, referencing some research from Amazon on **continuous training of LoRAs** from a few years ago.
   - Another user, suresh.b, confirmed the existence of such work, though didn't provide further details or links.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1400636791173677151)** (114 messages🔥🔥): 

> `GRO Trainer dataset mapping, Chat template cut off, GRPOTrainer config, Sequence dictionary (seq-dict), Unsloth shape dynamically changes` 


- **Troubleshoot Permutation Errors in GRPO Trainer**: Users are facing permutation errors with the GRPO trainer when using a **Qwen 2.5** base model due to dataset feature issues like `Question` and `Answer`.
   - The error arises from the `shuffle_sequence_dict` function, particularly with `ref_per_token_logps`, indicating potential source code problems.
- **Can't configure Unsloth's Output Embeddings**: Users are struggling to configure the offloading location for `output_embeddings` in Unsloth, which defaults to storing in the `{model}/output_embeddings.pt` path.
   - It was raised as a concern that this *behavior* will be problematic if the user does not have write permissions to the `{model}` path.
- **Gemma's Image Format for Fine-Tuning**: Users are debugging the correct format for using multiple images and system prompts when fine-tuning **Gemma-3-it-4B**, encountering `ValueError: Invalid input type`.
   - The correct format involves structuring the input data with `type` keys for both text and image content, accommodating a mix of images with or without system prompts but requiring consistent image numbers per sample.
- **Leveraging AI for Fine-Tuning Data Generation**: Users are exploring methods to convert **0.5 million tokens** of raw text into fine-tuning data using AI, specifically considering models with long contexts or RAG.
   - The discussion included whether to use a **Phi-14B** model with RAG to create training data, although chunking was dismissed as an option.
- **VRAM Swells Up During SFT Training**: Users are curious about why **VRAM** increases during **SFT** training, presuming memory pre-allocation should prevent this.
   - It was mentioned that *training would be amenable to pre-allocating memory*.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1400554342616010903)** (968 messages🔥🔥🔥): 

> `Arena Visibility, Leaderboard Tooltips, Personal Info in Datasets, Gemini's Repetitive Tendencies, Gemini 2.5 Deepthink` 


- ****Arena Buttons Boost Browsing****: A member suggested adding three major buttons for **Search, Image, Video, and Webdev Arena** to increase visibility, sharing a [concept image](https://cdn.discordapp.com/attachments/1340554757827461211/1400554342167089222/uzzzSHh.png).
   - Another member recommended adding a **webdev arena** button since it's on a separate platform, and also adding tooltips to the leaderboard explaining how **Rank, CI, and Elo** are determined.
- ****Dataset Delves Deliver Dangerous Data****: A user voiced concerns about accidentally including **personal information** (emails, passwords, etc.) in published prompts, and suggested a way for users to remove prompts before public release.
   - A member responded that such examples should be DM'd to them for escalation, and acknowledged [sharing these concerns with the team](https://www.deepcogito.com/research/cogito-v2-preview).
- ****Gemini Gabs Get Glitchy****: A member asked if others noticed **Gemini** repeating itself, but another member found it consistent and questioned if **Gemini 2.5 Flash** improved.
   - One user noted video limits dropping from **10 to 8**, urging others to use the video generation arena quickly.
- ****DeepThink's Debut: Disappointment Delivered?****: **Gemini 2.5 Deepthink** is out for Ultra members, and members are wondering if it is worth it after seeing **10 RPD limit**.
   - Members called it a **scam** and a daylight robbery, with some saying it's just a rushed version because of the imminent **GPT-5** release.
- ****GPT-5 Gossip Generates Great Expectations****: Discussion revolved around **GPT-5's** potential release, with some anticipating a paradigm shift while others expect incremental improvements and members discuss various performance benchmark data.
   - A member stated the view that *we're moving away pretty rapidly from "the best" model* as routing to a really strong model might be a really strong model for some stuff but not use it all the time.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1400888347160739932)** (1 messages): 

> `Veo 3, Image-to-Video, Audio capabilities` 


- **Veo 3 Unleashes Image-to-Video & Audio**: **Veo 3 Fast & Veo 3** now boast **Image-to-Video with audio capabilities** within the [Video Arena](https://discord.com/channels/your_server_id/1397655695150682194).
- **Create Videos with Images in Discord**: A new `/image-to-video` command has been added to the video-arena channels: allowing users to create videos from images.
   - Users are encouraged to vote on the best videos created using the new command.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1400555426764034119)** (580 messages🔥🔥🔥): 

> `Background agents, Improving cursor setup, Cursor freezing issues, YOLO mode activation, Vibe coding strategy` 


- ****Vibe Coding Github Needed****: One member said that *for background agents you need github? this thing is sick* with attached image.
   - Another member had spent **$40** on prompts, and needed advice on improving their **Cursor** setup.
- ****Cursor Freezing Bug Frustrates Users****: A user reported that their machine freezes every **30-60 seconds** after being in a chat for more than an hour.
   - A Cursor team member suggested posting the issue on the [Cursor forum](https://forum.cursor.com/c/bug-report/6) for better visibility and assistance.
- ****Navigating the Murky Waters of Model Spending****: Users are comparing **Cursor** and **Claude Pro** pricing, with one user saying *I go where the cheapest plans and best models are to be honest, and the $200 plan with Claude is one of the best deals currently still for me even with their new weekly hour limits*.
   - Another user expressed the cost can quickly balloon, spending *$600 in 3 months*.
- ****Horizon Alpha Experience Underwhelming****: One user found their personal experience with **Horizon-Alpha** to be *a bit underwhelming*.
   - In contrast, another user said *cursor is the best app i have ever seen*.
- ****Cursor Users Request Referral Program****: Members are asking if there is a referral program for **Cursor**, as one member mentioned having onboarded *at least 200+ people by now sitting in discords lmao*.
   - A link to the [Cursor Ambassador program](https://cursor.com/ambassador) was shared.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/)** (1 messages): 

lintaffy: oh, my ba is still loading for the easy command....
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1400554785911865466)** (410 messages🔥🔥🔥): 

> `Function Calling vs XML, AI Superintelligence Bio-Weapons, Grok4 vs GPT5, Horizon Alpha Performance, Large Context Windows` 


- ****Function Calling APIs**: Inherent Value?**: Function Calling APIs are seen as having **inherent value** compared to using structured XMLs for function calls, but one member noted that [XML is often used as a workaround](https://drinkoblog.weebly.com/) when the model doesn't support tool calling.
   - Some coding models like **Qwen** don't support function calling, so inline tool calls maximize interoperability despite minor inefficiencies.
- ****Zuckerberg's AI Superintelligence**: A Bio-Weapon Threat?**: **Mark Zuckerberg's** AI superintelligence initiative sparked concern over potential bio-weapon creation, with one member stating that *you cant just release superintelligence to the public like that*.
   - Concerns were raised that *controlling minds with fake users and carefully crafted language* is even more dangerous than bio-weapons.
- ****GPT-5 Delayed**: Grok4's Victory?**: Rumors suggest **GPT-5** is delayed due to inability to surpass **Grok4**, but another member stated that [OpenAI is planning to combine multiple products into GPT-5](https://link.to/openais-next-foundational-model).
   - A member also clarified that **GPT-5** will be a single, unified, omnimodal model.
- ****Horizon Alpha Shines**: A Free Reasoning Model?**: **Horizon Alpha** appears to outperform paid LLMs via the OpenRouter API, delivering [perfect one-shot code in custom programming languages](https://openrouter.ai/), with one user claiming, *it was like 3-4 times more useful than o3o3's multi turn is so bad*.
   - Its advanced shell use and task list creation in orchestrator mode prove superior to other models, though some believe it *could always be something turbo weird we’re not thinking of like codex-2*.
- ****Context Windows**: Overrated or Essential?**: Despite **Gemini's** 1 million context window, legacy codebase issues were better solved with **Claude** and **ChatGPT**, sparking debate on whether [large context windows are overrated](https://nealgoogs.website).
   - Some believe models with smaller context windows and better output are preferable, whereas others assert that larger context windows are essential for agentic applications to *remember and weave in far‑back details automatically*.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1400657438746738861)** (11 messages🔥): 

> `Agent Mode Confusion, ChatGPT Agents vs Regular GPT, GPT-4o auto reasoning, Missing Chat History` 


- **Agent Mode Causes Confusion**: Users are experiencing confusion around the term **Agent Mode**, with some believing it to be a new feature when it's essentially referring to existing advanced modes like **Code Interpreter**/**Advanced Data Analysis**.
   - Some members attribute initial hiccups to basic growing pains, suggesting it might get confused, give wrong answers, or simply stop working but is *awesome* when it works.
- **ChatGPT Agents vs Regular GPT**: A member points out that [ChatGPT models are unaware of recent developments](https://openai.com/index/introducing-chatgpt-agent/), including new products like **ChatGPT Agent**.
   - Another member reported using **Agent Mode** to work within **GitHub** to resolve an issue, finding it *quite interesting to watch what its doing*.
- **GPT-4o Auto Reasoning**: Users noticed that **GPT-4o** auto-switches to *Thinking*, even when not tagged as **Deep Research** or **Study mode**.
   - The switch to **o3** for technical or coding-related questions results in big reasoned replies, which some users find undesirable, preferring concise responses.
- **Chat History Goes Missing**: A member reported that their **chat history** (not in a folder) progressively disappeared throughout the week on both the web and mobile app.
   - Another member mentioned that *it should be fixed tho* and that *they fixed it as of yesterday*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 messages): 

> `` 


- **No significant discussion**: There was no meaningful discussion to summarize from the provided content.
- **No noteworthy insights**: The provided screen recording did not contain any noteworthy insights or topics for summarization.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1400645347990311045)** (1 messages): 

> `` 


- **No Topics Discussed**: No relevant topics were discussed in the provided messages.
   - The content appears to be a screen recording without specific details for summarization.
- **Insufficient Data for Summary**: The provided image analysis lacks textual content suitable for generating meaningful summaries.
   - Further information or message details are required to create relevant topic summaries.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1400554223522811936)** (325 messages🔥🔥): 

> `Image-to-video prompt generation in LM Studio, LM Studio's lack of roadmap, LM Studio's Plugin System, Connecting to LM Studio API from other computers on the network, Qwen3 Coder model support on LM Studio` 


- **Image-to-Video LM Studio When?**: Members are wondering about future **image-to-video prompt generation** and **image attachment** features in **LM Studio**, expressing a preference for offline solutions over relying on **ChatGPT**.
   - A member suggested **ComfyUI** as an alternative, but noted it's *not as good on AMD cards*.
- **Roadmap Unknown, So Noone Knows**: Members discussed the lack of a **public roadmap** for **LM Studio**, with one suggesting the roadmap is *just a big bucket with random papers*.
   - Another member confirmed there's *noone* who knows what the plan is and stated *no public roadmap so noone knows*.
- **Securing LM Studio on the Network**: Members discussed connecting to the **LM Studio API** from other computers on the network, with concerns raised about security.
   - It was suggested that **LM Studio's security is not proven** and should not be exposed without understanding the risks and securing your own network.
- **Qwen Crash Course: Load Model!**: Members discussed issues with loading the **Qwen3 Coder 30B** model, with one user experiencing a *Cannot read properties of null (reading '1')* error.
   - A member pointed out the user should update the app version to **0.3.21 b2** which supposedly fixed the issue, and mentioned to click the **recommended settings**.
- **Speculative Decoding: Not Worth It, Says Fabguy**: A member inquired about using **speculative decoding** with **Qwen3 MoE** models, which leads to a crashing error.
   - Another member pointed out that *draft model and primary model may pick very different experts for the task [of speculative decoding]. Not worth it.*


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1400555314864197673)** (69 messages🔥🔥): 

> `Nvidia Driver 580.88, Second-hand servers, Partial KV Cache Offload, Mac mini M4 vs RTX 3070, Next-gen GPUs` 


- **Nvidia's Jumps in Driver Versions**: Nvidia released driver **580.88** shortly after **577.00**, a **9-day-old driver** with a potential fix for GPU video memory speed after enabling NVIDIA Smooth Motion [5370796].
   - The user runs the drivers from the cuda toolkit, and doesn't use the fancy control panel or GFE (GeForce Experience).
- **Pondering Partial KV Cache Offload**: There was a question raised about whether it is possible to do a partial KV Cache offload in LM Studio, for example with a **40GB model**, where **KV Cache needs 20GB**, and **GPUs have 48GB total**.
   - The user was wondering if it was possible to split, with 8 of 20gb of the cache would be in the gpu, the rest offloaded.
- **Mac mini M4 Sizing up against RTX 3070**: A user wondered if a **Mac mini M4 ten core 32GB** would outperform an **RTX 3070**.
   - It was stated that CUDA is generally quicker than silicon if the models can fit in VRAM.
- **Rambling about RAM Recommendations**: One user suggested saving money for a used **3090**, which they claim is the best bang for buck card for AI use cases.
   - They cost around **700 euros** and for LLMs it would probably be the best solution, but there might be issues since they might've been used in mining.
- **5070 TiS release is immanent!**: A user speculates the **5070TiS** will be released soon with **24 gigabytes** of ram, where the **5070ti & 5080 have 16 gigs of ram**.
   - Another user points out that for cheap inference, right now 5060Ti 16gigs are the best option, at 450€/each, and you can put 3 or 4 in a board.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1400592183010263101)** (11 messages🔥): 

> `PyrenzAI launch, Personality.gg, OpenRouter PKCE, PyrenzAI feedback` 


- **Personality.gg enables Roleplay via OpenRouter**: [Personality.gg](https://personality.gg) launched a roleplay site using **OpenRouter** for most models, providing access to all 400 models through **OpenRouter PKCE** (Proof Key for Code Exchange) completely free/cheap.
- **PyrenzAI Launches a Free AI Chat Website**: A developer announced the launch of [PyrenzAI](https://pyrenzai.com), an **AI chat website** with a clean UI, models, a memory system and **free RAG** (Retrieval-Augmented Generation) for all tiers, using OpenRouter as the main AI generation backend.
- **PyrenzAI app faces speed and security critiques**: A user critiqued the newly launched PyrenzAI app, noting it's *cooked in terms of both speed and security*, with *laggy* performance and excessive fetching of user preferences (over 200+ times on every load).
- **UI and UX lauded on PyrenzAI release**: A member complimented the **UI/UX** of [PyrenzAI](https://pyrenzai.com), appreciating its unique look and style, and distinctive sidebar design compared to other apps.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1400553644956323921)** (242 messages🔥🔥): 

> `API Errors, Deepseek r1, Free Models, Horizon Alpha, API Key credit limit` 


- **API Errors plague OpenRouter Users**: Some users reported experiencing **API errors** when trying to use models via the OpenRouter API, including *no endpoint found* errors and other issues.
   - A member suggested checking the **model ID prefix** and the **base URL** for potential misconfiguration.
- **Deepseek v3 Outage Strikes Users**: Users reported issues with the **Deepseek v3 0324 free** model, including *internal errors*, *empty responses*, and **timeouts**.
   - One member noted that switching to the paid version of the model resolved the issues, suggesting the free version was overloaded: *free is completely overloaded. paid has none of these issue, and the actual content quality is better.*
- **Free Model Limits Frustrate OpenRouter Users**: Several users inquired about **free models** with higher message limits, with one user asking if there was any free model that *wont stop at 50 messages?*
   - Members clarified that topping up with **$10** provides a **1000 requests/day** limit and referenced [OpenRouter documentation](https://openrouter.ai/docs/api-reference/limits#rate-limits-and-credits-remaining) detailing the limits.
- **Horizon Alpha Raves Gain Momentum**: Users discussed the **Horizon Alpha** model, with some reporting that it was reasoning effectively and offering good performance.
   - The model itself reported that it was developed by OpenAI, though other members clarified that it was likely a distilled model.
- **Budget Overruns Baffle API users**: A user reported being charged significantly over their **API key credit limit**, suspecting that running **API calls in parallel** with Python threads might be the cause.
   - Other users shared similar experiences, suggesting that the credit limit updates might not be real-time, leading to occasional overcharges.


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/)** (1 messages): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1400586072773103901)** (23 messages🔥): 

> `Groq OpenBench, Provider Benchmarks, GPQA Evals, Inspect.ai, Prompt Caching for Kimi K2 and GLM 4.5` 


- ****OpenBench Groqs** for Provider Benchmarks**: Members discussed the [Groq OpenBench](https://github.com/groq/openbench) repository and how many times it has been posted regarding **provider benchmarks**.
   - One member mentioned they are *already working on evals (recently got prioritized)*, such as **GPQA** per provider, and expanding to other things.
- ****Inspect.ai** Discovery Praised**: A member expressed happiness in discovering [inspect.ai](https://inspect.ai) through the **OpenBench** link, noting it's *exactly what I've been looking for*.
   - This same user noted concerns about the chat UI using their full name from their account without control over it, leading to potential doxxing.
- ****Prompt Caching** Questioned for Kimi K2 and GLM 4.5**: A user inquired whether **OpenRouter** supports **prompt caching** for **Kimi K2** and **GLM 4.5**, noting that **Moonshot**'s platform directly supports it.
   - They stated it somewhat looks like it on [z.ai](https://z.ai).
- **Bypassing 20MB Limit: **Bigger PDFs** are now sendable**: Members questioned whether new feature would bypass the **20MB limit**, and they mentioned that they *recently added a way to send bigger pdfs*.
   - The new limit is the **upstream provider limit**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1400719197838770217)** (2 messages): 

> `Kimi K2 Turbo, Moonshot AI Forum` 


- **Kimi K2 Goes Ludicrous Speed!**: The Moonshot team announced **Kimi K2 Turbo**, a faster version of the Kimi K2 model, boasting **4x the speed** at **40 tokens/sec** from **10 tokens/sec**.
   - Until **Sept 1**, users get a **50% discount** on input and output tokens ([platform.moonshot.ai](https://platform.moonshot.ai/).
- **Moonshot AI Launches Official Forum**: The Moonshot AI team announced the launch of the ***Moonshot AI Forum*** ([https://forum.moonshot.ai/](https://forum.moonshot.ai/)) as a new hub for technical discussions, API help, model quirks, debugging, and dev tips.
   - *Discord’s still vibin for memes*, chill convos, and messin with ***Kimi Bot*** but if u tryna get serious with builds and tech stuff? forum’s the new spot fr 🔥


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1400679315850526800)** (126 messages🔥🔥): 

> `Kimi vs Claude, Kimi K2 Turbo pricing and speed, Using Kimi K2 Turbo in Claude code, Chinese companies video generation, Kimi K2's prompt format similar to ChatGPT` 


- **Kimi K2 challenges Claude throne**: After testing, a user finds that **Kimi K2** is the first model they feel they can use instead of **Claude**, ditching **Gemini 2.5 Pro** completely.
   - They add that coding, as a kind of information, is becoming freer and it's happening way faster than expected, eventually, most AIs will converge in terms of knowledge, and the differences between them will start to fade.
- **Kimi K2 Turbo goes 4x faster**: Kimi K2 Turbo is the **same model but with faster hosting**, now available with a special promo until Sept 1: **$0.30/1M** input tokens (cached), **$1.20/1M** input tokens (non-cached), and **$5.00/1M** output tokens.
   - This pricing implies it's *4x faster for 2x the price* during the discount, intended for users with speed requirements, and its official API helps keeps things steady.
- **Kimi K2 Turbo environment variable settings**: To use `kimi-k2-turbo-preview` in Claude code, set the following environment variable configurations: `export ANTHROPIC_SMALL_FAST_MODEL=kimi-k2-turbo-preview` and `export ANTHROPIC_MODEL=kimi-k2-turbo-preview`.
- **Kimi K2's prompt design mimics ChatGPT's**: Users noticed Kimi's prompt format is very similar to **ChatGPT**, with one user canceling subscriptions with **Gemini** ($250/month) and **OpenAI ChatGPT Pro** ($200/month) and **Grok 4 Heavy** ($3000/year).
   - One member joked that all it takes to get similar results from other chatbots is to *add a system prompt to tell it to act like an unhinged degen Discord mod, and tell it to “go and express yourself” haha.*
- **Google Gemini's daily deep think limit**: Members ridiculed Google Gemini Ultra's plan imposing **10 queries a day for $250/month**, one member calling it *very funny and very scummy*.
   - One added that even **ChatGPT pro** at $200/month gives unlimited **Office 365 Pro**, while **Claude Max** is more reasonable.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1400597993442513128)** (110 messages🔥🔥): 

> `Hermes-3 dataset, Unitree R1 robot, OpenAI's Horizon Alpha model, Quantization challenges, SmolLM and Qwen2.5` 


- **Hermes-3 Dataset Refusals Ruffle Quantization**: Members discussed whether refusals in the **Hermes-3 dataset** were purposeful or artifacts of censored models, with one member using it to compute the *imatrix* for quantization and finding unexpected refusals leading to [further dataset investigation](https://huggingface.co/datasets/NousResearch/Hermes-3).
   - The main intention was to confirm the dataset is devoid of refusals.
- **Unitree's R1 Robot Democratizes Embodied A.I.**: The community discussed the **Unitree R1 foundational robot model**, priced at **$5,900**, which offers a fully open software development kit (**Python**, **C++**, or **ROS**) for A.I. development, as showcased in [this YouTube video](https://www.youtube.com/watch?v=ljo7TjOqRzs).
   - It is an ideal tool for research teams transitioning to the next evolution of A.I.
- **Horizon Alpha Model sparks OpenAI Base Model Release Rumors**: Members discussed **OpenAI's Horizon Alpha model**, with speculation it resembles **OpenAI's** style and could be a **120B MoE** model with low activation, or possibly a **20B** model, as suggested in [this tweet](https://x.com/apples_jimmy/status/1951180954208444758).
   - There is speculation on Reddit, with [this thread](https://www.rxddit.com/r/LocalLLaMA/comments/1mepeqh/the_openai_open_weight_model_might_be_120b/) suggesting if it is **FP4** only, proper quantization would be impossible.
- **Quantization Quandaries for OpenAI's Leaked Model**: The community analyzed leaked config files indicating **OpenAI's model** is a **116.8B/5.7B MoE** model, which, when padded for GGUF, pushes it to **132.7B/6.3B**, making it difficult to quantize using methods other than **Q4_0**, **Q5_0**, **Q8_0**, and **IQ4_NL** due to the architecture's hidden size.
   - Because the hidden size of 2880 does not allow quantization to K or I quants.
- **SmolLM & Qwen2.5 Quantization Gotchas**: Discussions revealed that **SmolLM (135B/360B)** and **Qwen2.5 0.5B** have dimensions that cannot be made into K or I quants.
   - The members reported that only *o_proj* (from attention) can be quantized to K or I quants for the alleged **GPT model**.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1400563468649762909)** (4 messages): 

> `Input Tokens per Second, Prefill, Gemma, Time to First Token` 


- **Peeking into Input Token Processing**: A user inquired about resources for reasoning about **input tokens per second**.
   - Another member clarified this meant the *prefill* (just the context your using, not generating).
- **Profiling Gemma on a Laptop**: A user reported a **~50 second Time To First Token** for both 4500 and 9000 token prompts, using **Gemma** on a laptop.
   - The user is seeking comprehensive overview of that process for profiling purposes, and noted that the output tokens per second was the same across different input token sizes.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 messages): 

> `OSS Model Training Script, Metaprogramming and DAG->HRM->code automation` 


- **OSS Model Training Script: Raizoken Builds!**: A public research engineer is writing a model training script with the intention of making it **OSS** right away.
   - They are trying to create good **OSS models** for natural cursor navigation but are worried about potential misuse of the model, such as scraping websites that block crawling bots.
- **Raizoken Seeks Metaprogramming Automation Advice**: A member is seeking advice on **metaprogramming** and **DAG->HRM->code automation**, noting they're already using it in their stack but are facing scaling bottlenecks.
   - They've implemented **Terraform** and **Helm** to offset this, but are struggling with cloned slaves in **Ray nodes** when they form clusters, lacking a mechanism to control the self-spawn outside of cooldowns.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1400575460483535091)** (5 messages): 

> `AnythingLLM, Neuronpedia, Data Sovereignty` 


- **AnythingLLM heralds Data Sovereignty future**: A user shared a [link to a tweet](https://x.com/AnythingLLM/status/1755265335803212271?t=ypxd85gvodugP-ksZP6Nvg&s=19) about **AnythingLLM** and declared it the future for **data sovereignty**.
- **Neuronpedia and Data Sovereignty gain traction**: The user also shared links to **Neuronpedia** and other tweets relating to **data sovereignty** from [Jack_W_Lindsey's tweet](https://x.com/Jack_W_Lindsey/status/1950952346990862502?t=JGcHUqVwZF8_GBoWV5JPcg&s=19) and [heyshrutimishra's tweet](https://x.com/heyshrutimishra/status/1950801664379953468?t=ywRLWQRNGMsoXD8eOMPV-g&s=19).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1400851409561194598)** (3 messages): 

> `OSS model training script, Metaprogramming and DAG->HRM->code automation, Federated cycles between clones in ray nodes` 


- **OSS Model Training Script Emerges**: A public research engineer is developing an **OSS model training script** to address the lack of good OSS models for natural cursor navigation.
   - The engineer notes that websites blocking crawling bots may be scraped by new "clones" using this technology.
- **Metaprogramming Automation Bottleneck Surfaces**: A member is seeking advice on scaling issues with **metaprogramming** and **DAG->HRM->code automation**, despite using Terraform and Helm.
   - They are facing problems with federated cycles between clones in ray nodes, particularly with uncontrolled self-spawning outside of cooldown periods.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1400565376567480373)** (112 messages🔥🔥): 

> `Cline's $32M seed funding, CLI orchestration layer, Subagents and Claude Code Office Hours, Bytedance's Seed Diffusion LLM for Code, Open-License Hybrid Reasoning Models` 


- **Cline Closes $32M Funding Round**: Cline, an AI coding agent, announced a **$32 million** Seed and Series A funding round led by **Emergence Capital** and **Pace Capital** to support transparent, open-source AI tools for developers; serving **2.7 million** developers and transparent pricing with no upcharging.
   - Cline aims to empower developers by avoiding 'nerfed' products, focusing on enterprise features like access controls and centralized billing.
- **OpenAI's OS Model Leaks**: Details leaked about **OpenAI**'s upcoming OS model, **YOFO**, shortly after its config was briefly available, igniting excitement about rumored **120B** and **20B** variants.
   - A member noted that Jimmy Apples was reluctant to share all configuration details.
- **Anthropic's Production Reinforcement Learning Codebase Updated by Claude**: Anthropic merged a **22,000-line** change to their production reinforcement learning codebase, heavily written by **Claude**, sparking skepticism and discussion among users about the authenticity and safety of such a large AI-generated code change; it was largely a **json dsl**.
   - Sauers confirmed the change was real, and discussions touched on human review processes and concerns about the reliability of large AI-driven code merges.
- **Anthropic Cuts off OpenAI's API Access**: Anthropic revoked OpenAI's API access to its models, including **Claude**, citing a violation of terms of service.
   - A member noted that **OpenAI** expressed disappointment, mentioning that its API remains available to **Anthropic**, and the community discussed implications of competitive moves and blurring lines of model training.


  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1400567742054011033)** (4 messages): 

> `Cline pod writeup, Latent Space Podcast, Open Source Code Agent` 


- ****Cline Podcast** Writeup Released!**: The writeup for the **Cline podcast** is now out, linked on [X](https://x.com/latentspacepod/status/1951008883163668522).
- ****Latent.Space Podcast** Features **Cline**!**: **Latent.Space Podcast** announces a new episode with **Cline**, an open-source VSCode extension that recently raised **$32 million**.
   - The episode discusses Cline's origin, the 'Plan + Act' paradigm, top community tools, and future directions, featuring guests Saoud Rizwan and Pash. The podcast is available on their [website](https://xcancel.com/latentspacepod/status/1951008883163668522) and [YouTube](https://www.youtube.com/watch?v=dQw4w9WgXcQ).


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1400554550129070171)** (86 messages🔥🔥): 

> `RAG query expansion techniques, Sentence embeddings vs. token embeddings, Cross-encoders for semantic similarity, Knowledge Graphs for information retrieval, LLMs and question-answer co-occurrence` 


- **Query Expansion Boosts RAG Performance**: Members discussed [query expansion](https://www.promptingguide.ai/techniques/query_expansion) for RAG systems, suggesting generating multiple questions from a single query.
   - Specifically, for *'what is the name of the customer'*, it was proposed to create the questions *'What is the name?'* and *'Who is the customer?'* to improve retrieval.
- **Cross-Encoders Fail Ranking Task**: Experiment using a cross-encoder with **MS MARCO** data to rank results for the question *'What is the name of the customer?'* showed poor results.
   - The expected top hit (*Customer Name*) was ranked lower than (*Definition of Customer*), with scores of -0.67 vs -1.67.
- **Fine-Tuning Retrieval Task is Key**: To control ranking quality, directly training on a retrieval task is essential, according to [this paper](https://arxiv.org/abs/2212.01349).
   - It was suggested that the optimal similarity metric is task-dependent, meaning general-purpose embeddings may not suffice for specific retrieval scenarios.
- **Gemini 2.5 Flash favors Gemma Models**: Members found that Gemini-2.5-flash consistently ranked **Gemma models** higher than other models, even some 70B models.
   - It's suspected that the **response tone** of Gemma models might be more plausible to both humans and LLMs, influencing the ranking.
- **LLMs Parallel Thinking Debated**: Discussion around [Google's Gemini 2.5](https://blog.google/products/gemini/gemini-2-5-deep-think/) and its *'Deep Think'* feature, which uses parallel thinking to deliver more detailed and thoughtful responses.
   - Some suggested the model generates multiple ideas in parallel, with parallel COT, while others believe it's higher-level orchestration of basic models and context management.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1400573062881214524)** (3 messages): 

> `The Cinema AI, Generating Movie Scenes` 


- **Generating Cohesive Movie Scenes with TheCinema AI**: The channel will be reviewing [TheCinema AI](https://thecinema.ai/), an interesting research project focused on generating movie scenes that maintain **cohesion** with each other, according to the [arxiv paper](https://arxiv.org/html/2507.18634v1).
- **TheCinema AI: Generating Movie Scenes**: This research explores methods for generating movie scenes that are cohesive, as detailed in the [TheCinema AI project](https://thecinema.ai/) and its corresponding [arXiv paper](https://arxiv.org/html/2507.18634v1).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1400557372002271293)** (4 messages): 

> `NVIDIA Chips, Nintendo Switch` 


- **Experts Expose NVIDIA Chip Capabilities**: Experts in the American AI sector allegedly revealed that **NVIDIA's computing chips** have technologies for *tracking and geolocation* and *remote shutdown*.
   - A member called for [citation](https://citation.needed) since the source was the *State Internet Information Office of the PRC*, calling it an *absurd and feeble leverage attempt*.
- **Government restrictions are like Nintendo Switch**: A member said that government-imposed restrictions are just like the **Nintendo Switch**.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1400575531170402304)** (27 messages🔥): 

> `Audio pause timing in slide changes, Portuguese language support for explainer videos, NotebookLM for personalized podcasts, Canvas infographics from Perplexity Deep Research` 


- **Delay slide changes for smoother audio**: Users suggested adding an extra half-second pause before each slide change to avoid abrupt audio truncation in explainer videos.
   - This small adjustment could significantly *improve the viewing experience* by allowing audio to fade out naturally.
- **Portuguese Explainer Videos: Unofficial Support Available**: A user confirmed that while Portuguese isn't officially supported for explainer videos, they were able to force it to work.
   - Another user reported *mixed results*, with audio in Portuguese but slides sometimes remaining in English, while another suggested tweaking the prompt to specify both audio and video tracks.
- **NotebookLM + Gemini: Podcast Powerhouse?**: A user shared a workflow of asking Gemini a question and then feeding the answer into NotebookLM to create personalized podcasts.
   - They posted links to demonstrate the process: [NotebookLM](https://notebooklm.google.com/notebook/aa55ef62-9230-4b15-be5e-a6954247470c/audio) and [Gemini Share](https://g.co/gemini/share/11437d9da04c).
- **Canvas Infographics from Perplexity via NotebookLM?**: A user shared a process of creating canvas infographics directly from a **Perplexity Deep Research** report.
   - While not directly related to NotebookLM, they suggested it as a potential step to *leverage NotebookLM's power* with detailed outputs from other models, also adding that *Google can and SHOULD do better* than current video overviews, noting a current AI output.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1400554423864000664)** (65 messages🔥🔥): 

> `Offline access to NotebookLM studio material, Video overview rollout issues, NotebookLM and Gemini API for custom RAG pipeline, Comet browser extension for NotebookLM, Audio Overviews language and duration limitations` 


- ****NotebookLM Goes Offline for Road Warriors****: Users are seeking ways to save **NotebookLM studio material** for offline access during travel without constant internet connection.
   - One user mentioned downloading audio to an iPad and adding it to PowerPoint slides with family photos.
- ****Video Overview Vexation: Pro Users Ponder Missing Preview Perks****: Several **Pro account users** report not having access to the **video overview feature**, despite upgrading and others with free accounts having access.
   - A user who briefly had video access lost it after refreshing the page, suggesting ongoing rollout issues.
- ****RAG Dreams: User Schemes Custom NotebookLM with Gemini Power****: A user is considering using **Gemini embedding 001** and **Gemini 2.5 models API** to create a custom multi-hop, multi-step reasoning **RAG pipeline** for documents.
   - They aim to surpass **NotebookLM's** capabilities, citing limitations such as the **300-file limit**, lack of transparency in workflow, and limited system instructions, hoping to *plagiarize their work*.
- ****Comet Extension Could Catapult NBLM into Orbit****: Users discussed **Comet**, a browser extension that can access tabs/history/bookmarks and control the browser, and its potential integration with **NotebookLM** for source finding.
   - The suggestion was raised that **Comet** could potentially code an extension to dynamically add sources to **NotebookLM**.
- ****Spanish Audio Overviews Still Short and Sweet?****: A user inquired about why **Audio Overviews** in Spanish remain short in duration.
   - A workaround was suggested: *switch it to English, change the duration, then prompt it to do it in Spanish*.


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1400876073763213554)** (1 messages): 

> `Attention probes, Linear probes, Overfitting, Optimization issues` 


- **Attention Probes: A New Way to Classify Hidden States**: EleutherAI conducted experiments with **attention probes**, tiny neural networks with attention trained to classify the hidden states of transformers.
   - Despite expectations, their performance was mixed, sometimes underperforming standard **linear probes** due to **overfitting** and **optimization issues**, as detailed in their [blog post](https://blog.eleuther.ai/attention-probes/).
- **Attention Probe code open sourced**: EleutherAI has open-sourced the code for their attention probes experiments, inviting others to explore and refine the approach.
   - The repository is available on [GitHub](https://github.com/EleutherAI/attention-probes/), with the hope that further investigation may uncover potential improvements.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1400692396144070698)** (11 messages🔥): 

> `LLMs on low-power edge devices offshore, Gemini-2.5-flash biased ranks for gemma responses, OpenAI open source model config, MLA vs MHA generalization` 


- **Low-Power LLMs Brave Offshore Deployment**: A member is running **LLMs** on low-power edge devices offshore, focusing on seabed mapping, environmental monitoring, and autonomous systems.
   - The current use-cases involve **mission planning**, **anomaly detection**, and **smart data compression**, rather than scientific modeling due to latency and bandwidth challenges.
- **Gemini-2.5-flash Shows Favoritism for Gemma Models**: A member using **Gemini-2.5-flash** to rank responses from various LLMs noted consistently biased ranks for **Gemma** responses.
   - The member speculates about *family bias* or the possibility that **Gemma3** models are simply superior.
- **OpenAI's Forthcoming Open Source Model Config Leaked!**: A member shared a [config](https://gemini.google.com/share/3b63a193539c) for the forthcoming **OpenAI open source model**, including specs like **36 hidden layers**, **128 experts**, and a **201088 vocab size**.
   - Other members congratulated those whose work was adopted by **OpenAI** in this model.
- **MLA Triumphs Over MHA in Generalization Debate**: A member asked whether **MLA** or **MHA** is better in terms of generalization, while pretraining a **300m parameter model** on textbook quality data, using **RoPE**.
   - Another member recommended using **MLA** (Multi-level Attention) as the preferred architecture.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1400589521535373434)** (41 messages🔥): 

> `RoPE is near optimal, Weight tying is bad, semantic search and RAG` 


- ****NovelAI** Reveals RoPE Research**: NovelAI research has been published [here](https://research.novelai.net/rope/), experimenting with golden ratio in RoPE as an optimization target.
   - The punchline is *some math and experiments that are only interesting to theorists and have no practical applications*.
- ****RoPE's** Optimality and General Form**: A blog post [here](https://nor-blog.pages.dev/posts/2025-07-28-deriving-rope/) argues that RoPE is near optimal if one tries to derive it.
   - The general form for **N dimensions** requires projecting positions along incoherent and uniform directions, though this *doesn't have much practical significance*.
- ****Weight Tying** Bashed as Bad Practice**: A member stated that *weight tying is a universally bad practice that being said* and also *a terrible inductive bias!*.
   - They argued that **weight tying** is the cause of a lot of inefficiency and instability and *doesn't even make mathematical sense*.
- **Semantic Search Troubles and RAG Alternatives**: A member is struggling with semantic search and raised a question about the liability cap.
   - Another member suggested to use **RAG** like approach rather than semantic search, and also said that a lot of *domain specific engineering needs to go into semantic search to work properly*.


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1400583667998527540)** (1 messages): 

> `EleutherAI Website PR, Tensor Program papers, Yang et al paper` 


- **EleutherAI Website Gets a Facelift**: A member thanked another for their article and opened a [PR](https://github.com/EleutherAI/website/pull/145) with some fixes to the EleutherAI website.
   - The member requested careful review, mentioning they hadn't read the **Tensor Program papers** yet and may have made mistakes, especially in the math appendix around equations 15-18.
- **Seeking Clarity on Tensor Program Equations**: A member who submitted a PR is seeking guidance on locating specific equations (**15/17**) within the **Yang et al paper**, indicating a need for clarification on the mathematical underpinnings of the Tensor Program.
   - This suggests a collaborative effort to ensure the accuracy and validity of the website's content concerning the Tensor Program.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1400837578130981006)** (5 messages): 

> `HF transformers update, Llama & Qwen residual streams, Attention Probes Work, NIAH datasets` 


- **HF Transformers' Llama Layers Launch Residual Streams**: In **HuggingFace transformers 4.54**, **Llama & Qwen layers** now return residual stream directly (not tuple), which may affect users of `nnsight layer.output[0]`.
   - A member warned that using `nnsight layer.output[0]` will get you the 1st batch element only, not full residual stream, a bug spotted thanks to [nnterp tests](https://butanium.github.io/nnterp).
- **Attention Probes Produce Promising Probing Progress**: Members discussed promising attention probes, but were surprised by the mixed results, based on [attention probes work](https://link-to-attention-probes-work).
   - One member suggested probing with a suffix to consider what you're trying to probe for, asking the LM to consider what you're trying to probe for (e.g. *Is the above statement true?*).
- **NIAH Datasets' Last-Token's Talent**: Members stated that the underperformance of attention probes is mainly coming from the **NIAH datasets**, which are constructed so that the thing being classified comes right at the end of the sequence.
   - This would explain why last-token probing works well there; in that case, one should train both a linear probe and an attention probe.
- **McKenzie Probing Papers Promote Prompting Progress**: The probing paper [McKenzie et al. 2025](https://arxiv.org/abs/2506.10805v1) considers prompting the model to give an answer as a baseline (with results lower than for probes), but not prompting to improve probing.
   - It's possible this would be an improvement on the datasets we considered where mean probes outperform last-token probes, and worth investigating.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1400755680994136134)** (1 messages): 

> `` 


- **User Finds Potential Solution**: A user expressed they may have found a way to solve their problem and will send a message if it doesn't work out.
- **Awaiting User Feedback**: The conversation is currently pending further updates from the user regarding the success of their solution.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1400600571442106388)** (14 messages🔥): 

> `MIT Collaboration on LLM Training, Containerization Issues, CUDA Issues, DeepSpeed checkpoint inspection` 


- **MIT Collabs on OLMo2 & DCLM Training**: MIT and EAI are collaborating on LLM training, starting with **OLMo2 1B** or **DCLM 1B** to familiarize themselves with the pipeline, initially focusing on pretraining, but with plans to incorporate **SFT** and safety alignment later on.
- **Container install faces tricky Permissions Error**: A user encountered permission errors during containerized installation using Apptainer, specifically related to `setgroups` failures, and was advised to try `apptainer exec --fakeroot your_image.sif ...` as a potential workaround.
   - Another member suggested using conda environments directly on the host if the container issue persists, based on their experience with Slurm-based HPC clusters.
- **CUDA configuration challenges in Conda env**: After switching to a conda environment, the user encountered **CUDA** issues, which they believe have been resolved, and they are now working on installing **flash-attention** and **TE**.
   - The user asked for specific test commands to verify the environment setup after installing **flash-attention** and **TE**.
- **DeepSpeed Checkpoint Inspection Woes**: A user reported that `inspect_ds_checkpoint` from the experimental branch doesn't support `pipe_parallel_size=0`, causing validation checks to fail due to the absence of `layer_*` files in the checkpoint directory.
   - They also inquired whether it's fundamentally impossible to scale from **(4 nodes x 8 GPUs)** to **(8 nodes x 8 GPUs)** with `pipe_parallel_size=0`, `model_parallel_size=1`, and zero stage 1.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1400559583528747048)** (61 messages🔥🔥): 

> `Aider Appreciation, SGLang and Qwen Speed, 4090 Mobo and Case, Aider vs Other Tools, Claude Code Context Limits` 


- ****Aider** Still Reigns Supreme**: One member expressed their appreciation for **Aider**, noting its superior combination of control and freedom compared to other tools, estimating **one week of programming work** was done in a single day for **$2** with DeepSeek.
   - Another user echoed this sentiment, saying, *"Aider rules so hard"*.
- ****SGLang** and **Qwen** Hit Ludicrous Speed**: A member reported achieving **472 tokens/s** with **sglang** and **Qwen 0.6B Q8** on LM Studio using an **RTX 4090**, whereas on regular lmstudio it only goes **330 t/s**.
   - Another user expressed interest in replicating this local-only flow, noting **vllm's** comparatively slow performance on their **4090** versus Ollama and was very interested in trying llama.cpp.
- **Mobos for Multi-GPU Setups Explored**: The discussion pivoted to hardware setups, with one member recommending this [MSI motherboard](https://www.bhphotovideo.com/c/product/1864084-REG/msi_meg_x870e_godlike_e_atx.html) for dual **3090s**, housed in a Fractal North XL case.
   - Others chimed in with their setups, including servers with **3 L4s** and **T40s**, and different cases like the **Meshify2**.
- **Aider versus Windsurf versus Cursor**: One user expressed disappointment with **Aider**, **OpenHands**, and **Chode-Pilot**, preferring **Windsurf** and **Cursor**.
   - They speculated the "sauce" might be in giant closed models running on beefy hardware, expressing a need to try **QWEN3** after unsatisfactory experiences with **Devstral** and **Codelamma**.
- ****Claude Code's** Context Window Caveats**: Members discussed the performance of **Claude Code** with one mentioning that it works well without RAG and mentioning that Claude, unlike other frontier models, suffers greatly from high context token count.
   - It was noted that quality noticeably degrades beyond **64k tokens**, an issue less pronounced in **o3**, and best handled by **Gemini 2.5 Pro**. Others pointed out *the system prompt alone eats a significant portion of the context window*.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1400608522361770119)** (10 messages🔥): 

> `Qwen3 30B A3B Coder Benchmarking, LM Studio Usage, llama.cpp server + docker aider benchmark, aider + claude-code max subscription integration, Gemini 2.5 Pro` 


- **Benchmarking Qwen3 30B locally in LM Studio**: A member wants to benchmark 8 different quants of **Qwen3 30B A3B Coder** locally using **LM Studio** in an easy way.
   - Another member suggested using *llama.cpp server + docker aider benchmark on the same computer*, and referred to a writeup involving **Gemini 2.5 Pro** that details the steps to get it working.
- **Aider integrates with Claude-Code Max Subscription**: A member inquired whether *aider* can be used with **claude-code max subscription integration** to tap into the new thinking model.
   - They also asked if the command *aider --model gemini/gemini-2.5-pro-preview-06-05 --thinking-tokens 32k* is an old way of thinking and if anyone had success running aider with Claude code.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1400559945786589275)** (43 messages🔥): 

> `Security MCP Check Tool, PayMCP Payment Layer, PageRank for MCP Servers, MCP Eval Platforms, Gateway for Agent Tool Search` 


- ****Security MCP Check Tool** unveiled**: A member shared a [GitHub repo](https://github.com/minte-app/security-mcp-check) for a **security MCP check tool**, seeking feedback.
   - This could provide a way to check your own server for vulnerabilities, but note that no further explanation was given.
- ****PayMCP** Payment Layer emerges**: A member announced the development of **PayMCP**, a payment layer for **MCP**, and is looking for collaborators and early users, providing [Python](https://github.com/blustAI/paymcp) and [TypeScript](https://github.com/blustAI/paymcp-ts) implementations.
   - This new tool promises to allow MCP servers to easily accept payments, though it is unclear what payment options it supports.
- ****PageRank for MCP Servers**: A new Search Tool**: A member inquired about the existence of **PageRank** implementations for **MCP** servers or tools, aiming to rank servers by utility rather than just name or description.
   - Another member shared a [repository of MCP tools](https://github.com/YogiSotho/mcp-tools-collection), and mentioned the [MCP registry](https://github.com/modelcontextprotocol/registry) as potentially helpful resources.
- **MCP Eval Platforms sought**: A member sought information on **MCP eval platforms** that generate different agents in various situations to test **MCP** servers.
   - Another member indicated they are developing a gateway for agents to search for tools and plan to have something available by Sunday.
- **Guidance for grasping MCPs**: A member requested assistance in understanding and using **MCPs** in their workflow, offering to pay for someone's time to help.
   - This highlights the complexity and learning curve associated with adopting **MCPs** for new users.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1400893140394578104)** (1 messages): 

> `JSON MCP Server, LLM Efficiency with JSON, Schema Generation for JSON, Token Savings` 


- ****JSON MCP Server** for LLMs Launched**: A new **JSON MCP Server** has been created to aid **LLMs** in efficiently parsing large and complex **JSON** files, such as **Excalidraw exports**; see the [GitHub repo](https://github.com/kehvinbehvin/json-mcp-filter).
   - The tool uses **schema generation** to first understand the structure of the **JSON** and then extract only the necessary data, saving tokens and context.
- **LLMs parse JSON files more efficiently**: The main goal of this tool is to help **LLMs** parse large and tangled JSON files more efficiently.
   - It saves tokens and context by extracting only the data you need.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1400575926592344246)** (8 messages🔥): 

> `Hylo Programming Language, Value Semantics, Halide, Scala 3/Scala Native, Heterogenous Programming` 


- ****Hylo** Language Heats Up**: A member inquired about the **Hylo** programming language ([https://www.hylo-lang.org/](https://www.hylo-lang.org/)), highlighting its approach to memory safety through **value semantics** and scheduling, drawing parallels with **Halide**.
   - It was noted that the team sits in the same "heterogenous pl for the 21st century" bucket as **Mojo**.
- **Hylo's value semantics and concurrency**: Members stated that the **Hylo** team is still hammering their **value semantics** and **concurrency** down, the hope and roadmap though is that value semantics squares nicely with scheduling, tiling, vectorizing.
   - The **Hylo** team is from Adobe STL and has experience hacking on **Halide**.
- ****Scala** team member is on Hylo?**: A member mentioned that the person responsible for **Hylo** is currently working on **Scala 3/Scala Native**.
   - Other members stated the leads come from **cpp** and **Swift**


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1400862107377074339)** (1 messages): 

> `Triton Kernel AI Agent, GEAK benchmarks` 


- **AMD Introduces GEAK & Triton Kernel AI Agent**: AMD introduced the **GEAK benchmarks** and **Triton Kernel AI Agent** in their paper [GEAK: INTRODUCING TRITON KERNEL AI AGENT & EVALUATION BENCHMARKS](https://arxiv.org/abs/2507.23194).
- **Dive into AMD's Kernel AI Agent**: Explore AMD's novel approach to **AI-driven kernel optimization** using their new **Triton Kernel AI Agent**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1400653536185942016)** (4 messages): 

> `Profiling Copilot, __launch_bounds__ fix for register count issue, setmaxnreg ignored due to extern call` 


- **__launch_bounds__ setting launches CUDA fix**: A user fixed an issue where the compiler couldn't determine register count at entry by passing `minBlocksPerMultiprocessor` to `__launch_bounds__`, setting `maxThreadsPerBlock=128*3` and `minBlocksPerMultiprocessor=1`.
   - They noted they're *not sure how that fixes the problem exactly*, but are *happy to move forward*.
- **`setmaxnreg` meets incompatibility issues**: The `setmaxnreg` setting is still being ignored, now due to a different problem related to compatibility with an `extern` call, as indicated by the message: `ptxas info : (C7506) Potential Performance Loss: 'setmaxnreg' ignored to maintain compatibility into 'extern' call.`
   - A member asked if the kernel is calling an `'extern'` function defined in a separate compilation unit.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1400611087044579428)** (1 messages): 

> `CheckpointPolicy with Custom Kernels, Functorch API` 


- **CheckpointPolicy for Custom Kernels**: A member inquired about documentation on implementing **CheckpointPolicy** with custom kernels in Torch, specifically for fused **MLP**.
   - They asked if it's feasible to use it within the **Functorch API**.
- **Functorch and Custom Kernels**: The user wants to integrate a custom kernel, such as a fused **MLP**, into the **Functorch API** while using **CheckpointPolicy**.
   - They are seeking guidance or documentation on how to achieve this integration effectively.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1400600521634480232)** (1 messages): 

> `MI300X FP8 benchmarks on AMD, AMD MI300X vs H200 vs B200, FP8 Data Parallel Benchmarks` 


- **MI300X Benchmarks Leave H200 Behind**: A user inquired about experiences with new [MI300X FP8 benchmarks](https://eliovp.com/mi300x-fp8-data%E2%80%91parallel-benchmarks-8-64-gpus-h200-left-behind-b200-within-reach/) on AMD hardware.
- **FP8 Performance on MI300X**: The benchmarks compare **AMD's MI300X** with **NVIDIA's H200** and suggest the MI300X outperforms the H200 in certain FP8 data-parallel tasks.
   - The results indicate **MI300X** performance is getting close to **NVIDIA's B200**.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/)** (1 messages): 

celis1702: thank you both so much for your clear explanations and for sharing these details!
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1400694013803106367)** (2 messages): 

> `JIT function, JAXPR printing, Static arguments` 


- **JAXPR printing trouble**: A user encountered trace-time errors when attempting to print the **JAXPR** for a **JIT** function using static arguments.
   - The user was attempting to use `jax.make_jaxpr(jit_func)(1, 2)` but was running into errors.
- **Static Arguments and JIT Compilation**: The user's problem revolves around using `static_argnames` with `jax.jit` and then trying to inspect the resulting JAXPR.
   - Understanding how static arguments affect tracing and compilation is key to resolving the trace-time errors.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1400861221586210836)** (2 messages): 

> `Agreement, Acknowledgement` 


- **Affirmative Confirmation**: User @sshkr16 stated *"I am yeah"*, signalling agreement or confirmation within the conversation.
   - Another user, ali_8366, responded with *"Nice !"*, acknowledging and positively affirming the initial statement.
- **Positive Acknowledgment Received**: ali_8366's response of "Nice !" indicates a positive reception to @sshkr16's affirmation.
   - This simple exchange highlights mutual understanding and agreement within the channel.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1400585395363643573)** (2 messages): 

> `Profiling llama.cpp with rocprofilerv3, AMD machine for GGUF` 


- **rocprofilerv3 Profiling Woes with Llama.cpp**: A member inquired about using **rocprofilerv3** to profile **llama.cpp**, noting successful profiling of PyTorch code but issues with llama.cpp on **MI50s** with **ROCm 6.3.3**.
   - They were curious if the issue was specific to their setup.
- **AMD Hardware Inquiry for GGUF Execution**: Another member responded, expressing that they hadn't tried profiling **llama.cpp** and inquired about the specific AMD machine being used for running **GGUF** models.
   - They wanted to know the hardware setup for GGUF inference.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1400869779694419998)** (1 messages): 

> `C/ua Hiring, AI Agents Infrastructure, Founding Engineer Roles` 


- **C/ua Seeks Talent in SF and Spain**: **C/ua** is hiring Founding Engineers in San Francisco and Spain (Remote or Madrid hybrid) to build the infrastructure for general-purpose AI agents.
   - They are backed by **Y Combinator** and are developing open-source tools used by thousands of developers.
- **C/ua Building AI Agent Infrastructure**: **C/ua** focuses on infrastructure for AI agents to safely use computers and applications at scale.
   - The roles involve building secure runtimes, container orchestration, developer APIs, and OS-level virtualization.
- **Founding Engineer Roles at C/ua**: **C/ua** is looking for Founding Engineers passionate about system safety, reproducibility, and dev experience to shape how agents run at scale.
   - Interested candidates can find more details in the [San Francisco job post](https://ycombinator.com/companies/cua/jobs/dIskIB1-founding-engineer-infra-agent-systems).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/)** (1 messages): 

tonic_1: really glad i was nosey enough to check this convo out 🙂 super excited about this 🙂
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1400557751641313300)** (7 messages): 

> `README update on Resource vs Prototype, RCON client disconnects, Blueprint VQA pipelines` 


- ****Resource or Prototype** in README?**: A member inquired whether the README is up-to-date regarding the use of **Resource** vs **Prototype** for finding patches, specifically questioning whether `position=nearest(Prototype.IronOre))` should be `Resource.IronOre`.
   - Another member confirmed the likelihood, noting that *"That part of the README was made by claude in cursor"*.
- ****RCON client disconnects**, limiting testing**: Testing is being throttled because the **RCON client** is disconnecting, as demonstrated by the error *"The RCON client is currently not connected to the server"*.
   - This issue prevents complete trajectories.
- ****VQA Pipelines** for Blueprints Completed!**: A member reported the completion of **VQA pipelines for blueprints** and is now focusing on data augmentation.
   - The augmentation methods include **rotations**, **flips**, and **sub-section chunks**, aiming to multiply the available blueprints by 10-15x.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1400572352168984789)** (6 messages): 

> `picocuda compiler, elements graph data structures, scalar compilation, GPU compilation, tinygrad's AMD GPU driver` 


- **Picocuda & Elements Projects Gain Momentum**: Progress is being made on the [picocuda](https://github.com/j4orz/picocuda) compiler and [elements](https://github.com/j4orz/elements) graph data structures projects.
   - The focus is now on diving into GPUs, following wrapping up scalar compilation for the [Zero to Hero](https://j4orz.ai/zero-to-hero/) textbook.
- **GPU Compilation Textbook to Follow GPUCC Paper**: The textbook will roughly follow the [GPUCC paper](https://dl.acm.org/doi/pdf/10.1145/2854038.2854041) from CGO '16, extending the big red intermediate language (BRIL) from sampsons cs6120, which is a mini LLVM ([BRIL webpage](https://www.cs.cornell.edu/~asampson/blog/bril.html)).
   - The author suggests building up both scalar and vector compilation incrementally with a small layer of runtime orchestrating the host and device code.
- **AMD GPU for Open-Source Textbook**: A **7900xtx** or **9070xt** will be purchased for development, using **tinygrad's AMD GPU driver** over USB.
   - AMD was chosen because it's open source, aligning with the textbook's target audience of hackers and tinkerers.
- **Porting llm.c to AMD's HIP**: The goal is to build up to **Karpathy's llm.c** (forked and modified to **AMD's HIP**).
   - Contributors are welcome, particularly with the C compiler at [picocuda](https://github.com/j4orz/picocuda) and the graph data structures at [elements](https://github.com/j4orz/elements).
- **Graph Algorithms Needed for Host Code**: The two main graph algorithms needed for host code are dominators for the middle (`opto`) and graph coloring for the backend (`cgen`)'s register allocator.
   - The author recommends lengauer-tarjan for dominators (like rustc) and briggs-chaitin-click for the register allocator (like hotspot's C2).


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1400849332994838631)** (4 messages): 

> `DTensor, Basic Parallelism Schemas, Shape Rotation, DTensor Problems, Marksaroufim visualizations` 


- **DTensor schema continuation planned**: Members are planning to continue working on **DTensor** and **basic parallelism schemas**.
   - The session is scheduled for Sunday around **8 PM CEST**, with the possibility of extending it if necessary.
- **Shape Rotation task in progress**: One of the members plans to focus on **shape rotation**.
   - The goal is to explore and implement techniques for efficiently manipulating the shapes of tensors.
- **Marksaroufim visualizations inspire DTensor problems**: Members will be exploring new **DTensor problems** by using [Marksaroufim's visualizations](https://www.youtube.com/@marksaroufim).
   - The aim is to leverage these visualizations for insights into potential challenges and solutions in **DTensor** development.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1400589363934527620)** (26 messages🔥): 

> `Flux Krea model, Synthetic Datasets with HF jobs, AMD GPU for EM image segmentation, Llama CP model path, Gemini-2.5-flash bias` 


- ****Flux Krea** new model released!**: A new **Flux Krea** model is out with *much more detail*, works with most lora on base.dev, [available here](https://huggingface.co/Clybius/FLUX.1-Krea-dev-scaled-fp8).
   - According to initial reports, **NSFW** is *not possible*.
- ****Gemini 2.5 Flash** possibly favors **Gemma3****: A member has been trying to use **Gemini-2.5-flash** to rank responses from various LLMs, and has been consistently seeing **Gemma3** models ranked higher than others even some **70B** models.
   - Another member thinks that there is some bias, and **Gemma 3** is one of the better models and *the default weights are also well done*.
- ****HuggingFace Ultrascale** book mirrors blogpost?**: A new member asked if the contents of the **HF ultrascale book** are the same as the blog, requiring **HF pro subscription**.
   - Another member confirmed that *it's 246 pages*, possibly the same as the blog post with lots of images, linking to [Julien Chaumond's tweet](https://x.com/julien_c/status/1951277984532279794).
- **Synthetic Datasets with **HF jobs** documented**: A member was asking how to create synthetic datasets with **HF jobs**.
   - Another member offered [hf jobs docs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs), [script](https://ray.so/O8JjQ6X), [dataset](https://huggingface.co/datasets/dvilasuero/nemotron-kimi) and [config](https://huggingface.co/datasets/dvilasuero/nemotron-personas-kimi-questions/raw/main/config.yml) as an example.
- **Volume Seg Tool built on **AMD****: A member released one of the **SOTA tools for EM image segmentation** using a **10 years old GCN AMD GPU** that has no tensorcore and not even supported on lastest ROCm, [available here](https://github.com/fgdfgfthgr-fox/Volume_Seg_Tool).
   - They mentioned that achieved nearly a **5x-10x reduction** from other neural models.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1400825233887465542)** (2 messages): 

> `Note-taking tools, Remnote` 


- **Note-Taking App Disclosed: Remnote**: A user inquired about the note-taking tool being used, and the response pointed to [Remnote](https://www.remnote.com/).
   - Remnote is a **knowledge management tool** that integrates note-taking with spaced repetition learning.
- **Remnote: More Than Just Notes**: Discussion highlighted [Remnote](https://www.remnote.com/) as a **versatile platform**.
   - It combines traditional note-taking with features like **spaced repetition** to enhance learning and retention.


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1400811422899896330)** (2 messages): 

> `AgentUp, Emergence AI, LongMemEval Benchmark` 


- ****AgentUp** Rockets onto the Scene!**: The [AgentUp](https://github.com/RedDotRocket/AgentUp) project was highlighted.
   - It seems to be gaining traction as a noteworthy agent framework.
- ****Emergence AI** Claims SOTA in Memory!**: **Emergence AI**'s new architecture achieved [SOTA](https://www.emergence.ai/blog/emergence-is-the-new-new-state-of-the-art-in-agent-memory) on the **LongMemEval benchmark**.
   - The benchmark is used for evaluating long-term memory in AI agents.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1400846134766862366)** (3 messages): 

> `smolagents.js, CodeBoarding, Qwen3-30B-A3B-Instruct-2507` 


- **Smolagents ported to JavaScript!**: A member released a **TypeScript** port of **smolagents** called **smolagents.js**, available on [GitHub](https://github.com/yusuf-eren/smolagents.js) and [npm](https://www.npmjs.com/package/smolagents.js).
- **CodeBoarding released!**: A member released **CodeBoarding**, an open-source project that uses static analysis + LLMs to generate interactive diagrams of **Python** codebases, available on [GitHub](https://github.com/CodeBoarding/CodeBoarding).
- **Qwen3 refuses questions no more!**: A member posted about tweaking **Qwen3-30B-A3B-Instruct-2507** to stop refusing even blatant questions, available on [HuggingFace](https://huggingface.co/pszemraj/Qwen3-30B-A3B-Instruct-2507-abliterated).


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/)** (1 messages): 

cakiki: <@570737726991761409> please don't promote paid content in the server
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1400694296440733776)** (2 messages): 

> `Discriminator Learning Rate, GAN Loss Issues, Debugging GANs` 


- **Lowering Discriminator Rate Debugs GANs**: A member suggested lowering the **discriminator learning rate** to a very low value to observe loss changes, which can help pinpoint issues in **GAN** training.
   - Another member inquired about how much lower they should go, noting their current rate is at **1e-5**.
- **Fine-Tuning GAN Learning Rates**: The discussion centered around techniques to debug **Generative Adversarial Networks (GANs)** by manipulating the discriminator learning rate.
   - The goal is to identify whether the discriminator's loss collapsing to **0** is due to an imbalance in learning rates.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1400700923663093811)** (2 messages): 

> `Llama 4 Access, Qwen Model, DeepSeek-R1` 


- **Llama 4 Access Blocked!**: A member reported being **unable to access Llama 4** while attempting to run *dummy_agent_library.ipynb* on Colab.
   - Another member suggested substituting with a **Qwen model** or **DeepSeek-R1** as viable alternatives.
- **Substitute Models to the Rescue!**: Since **Llama 4** access requests are getting rejected, use **Qwen** or **DeepSeek-R1** as a replacement.
   - These models should work OK as a substitute.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1400583118104039715)** (21 messages🔥): 

> `Cohere API context window size discrepancy, HackRx 6.0 AI hackathon Rate Limit, Cohere Enterprise Plan, Cohere website login error, Cohere Support Team introduction` 


- **Context Window Size Debate: 32k or 128k?**: A user pointed out a discrepancy between the **Hugging Face model card (32k context)** and **API docs (128k context)**, leading to clarification that it's **128k in** and **8k out**.
   - The team acknowledged the issue and promised to update the Hugging Face model card soon.
- **Team Patriots Seek Rate Limit Relief**: A student team, **Team Patriots**, requested a temporary rate limit increase for the **HackRx 6.0 AI hackathon** due to being blocked by the **10 calls/minute trial key limit**.
   - A Cohere team member granted them permission to create multiple accounts and cycle the keys to overcome the limit.
- **Startup Eyes Cohere Enterprise**: A startup, loving Cohere's Reranker implementation, inquired about an **Enterprise plan** to handle exceeding the **1000/min limit** for the production API.
   - They were directed to email details about their use case and request profile to support@cohere.com and varun@cohere.com for secure assistance and connection with the right folks.
- **Login Error Causes Headaches**: A user reported an error when signing in on the **Cohere website**, specifically related to a **CORS policy** blocking access during the onboarding process.
   - No immediate solution was provided in the chat.
- **Cohere Support Team Gives Warm Welcome**: Varun, a **Technical Support Engineer** at Cohere, introduced himself and provided guidance on where to post for general support and API-specific discussions.
   - Newcomers were encouraged to join **Cohere Labs 🧪** a dedicated Discord community for research, at [https://cohere.com/research](https://cohere.com/research).


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/)** (1 messages): 

kaludi: Is there something going on with the API? We are getting multiple timeouts for our queries
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1400730205450014751)** (6 messages): 

> `Samsung Biologics AI Architect, AI Developer with LLM Workflows, Dell's Engineering Technologist, Mobile & JS-fullstack AI Application Developer` 


- **Samsung's AI Architect arrives!**: An AI architect from **Samsung Biologics** introduced themself, focusing on integrating **AI methods and tools** to address business needs and highlighting a private **LLM service with RAG** for internal use.
   - They are eager to engage in conversations related to **biopharmaceutical or biological challenges**.
- **LLM-focused AI Developer Joins**: An AI developer specializing in **LLM workflows, agent-based tools, and MCP integration** introduced themself, noting experience in building **AI sales assistants and RAG pipelines** using **LangChain and FastAPI**.
   - Their primary tech stack includes **Python and Node.js**, and they are open to collaborations and contract work.
- **Mobile AI Application Developer says What's Up!**: An **AI application developer** with mobile & js-fullstack experience introduced themself.
   - No additional information was provided.
- **Dell's AI Research at Cohere's Door**: An Engineering Technologist from **Dell** working mostly with **AI research** introduced themself from Brazil.
   - They are here to connect and learn.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1400565757355753552)** (17 messages🔥): 

> `DM spam, Wide research, Cloudflare issues, Manus AI, Daily refresh credits` 


- **User complains of DM spam**: A member reported receiving DM spam and requested an admin to perma-ban the user.
   - No action was taken during the period, and the user who sent the spam remained unaddressed.
- **Users test out Wide Research Platform**: A member inquired about initial takes on using **Wide Research**.
   - No reviews of **Wide Research** were given.
- **User unable to setup Cloudflare virtual environment**: A member is experiencing issues configuring a virtual environment within **Cloudflare**.
   - The setup keeps getting stuck on **Cloudflare**, preventing them from completing the virtual environment configuration.
- **Daily refresh credits cease functioning**: A member reported that daily refresh credits are no longer working.
   - Another user mentioned having their account suspended despite not breaking any rules, indicating possible issues with the platform's credit and account management.
- **Layoffs possibly impact refunds**: A member pointed out recent layoffs and suggested the user probably won't get their money back.
   - The comment implies that recent layoffs at the company may impact the ability to process refunds or resolve financial issues.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1400874884271313083)** (2 messages): 

> `LlamaIndex, Novita Labs, Gemini Live` 


- **LlamaIndex & Novita Labs Unite!**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951315242904068483) announces the use of **LlamaIndex** with **Novita Labs** model inference capabilities.
   - They provide diverse data source connections and data transformation into vector embeddings.
- **Gemini Live Now Speaking TypeScript**: The [LlamaIndex tweet](https://twitter.com/llama_index/status/1951342252346974431) announces **Gemini Live integration** available in **TypeScript**.
   - A demo shows how to set up and run a simple terminal chat.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1400596216693129216)** (13 messages🔥): 

> `Agentic AI Code Assistance, Git-Style Branching for LLM Conversations, LlamaIndex Parsers for PDFs and PPTs, AI+Blockchain for on-chain AI agents` 


- **LLM Web3 Engineer Availabile for Hire**: A Senior AI & Blockchain Engineer shared his experience building **on-chain AI agents** for trading, media automation, and autonomous governance using **Eliza OS**, **LangGraph**, and custom toolchains.
   - He has deep experience across **Base**, **Solana**, **Berachain**, **Sui**, **Aptos**, **HBAR**, **EVM chains**, and cross-chain systems.
- **Craving a Local Agentic AI Code Assistant**: A member inquired about local agentic AI code assistance tools, similar to **Cursor editor**, that can run locally.
   - Other members suggested that there are many options on GitHub, but the original poster expressed that **most options have dependency issues** or lack agentic features.
- **Git-Style Branching makes conversation trees**: A member is testing a system where every message is a node, enabling branching off anywhere in the conversation tree to create a new context path, detailed in [their blogpost](https://gupta-aniket.github.io/Mobile-developer/hire/#projects#branched-llm-mvp).
   - The system is tested with **Gemini API** so far, with plans to try **GPT-4**, **Claude**, and local **LLaMA** models, and the poster is looking for testers.
- **Llama Parsers take fare share of time to parse**: Members discussed the use of LlamaIndex parsers for **.doc**, **.pdf**, and **.ppt** files, especially when text is on images.
   - A member suggested using **LlamaParse** in premium mode, while another suggested converting PPTs to PDFs for better speed or using ThreadPoolExecutor() to parse documents asynchronously.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dbreunig: https://www.dbreunig.com/2025/07/31/how-kimi-rl-ed-qualitative-data-to-write-better.html
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1400619842368962560)** (2 messages): 

> `DSpill, Yaron Minsky, Quant Bros` 


- **Coining new verbs: DSpill is here!**: A member asked who would *give it a second try to **DSpill Yaron Minsky / quant bros***.
   - Another member replied *Wow new verb: to **DSpill***.
- **The quant bros get DSpilled?**: A member proposed the idea of 'DSpilling' **Yaron Minsky** and the **quant bros**.
   - This sparked the coining of a new verb, '**DSpill**,' to describe the action.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1400588919791161475)** (2 messages): 

> `Mojo installation issues, GitHub issue reporting, Detailed logs for debugging` 


- **Mojo Install Woes Prompt GitHub Issue?**: A member reported difficulties installing **Mojo** for three days and inquired about opening a **GitHub issue**.
   - Another member encouraged them to open an issue and include detailed logs to aid in troubleshooting.
- **Detailed Logs Recommended for GitHub Issue**: When submitting a **GitHub issue** for **Mojo** installation problems, including detailed logs can significantly help.
   - This provides developers with the necessary information to diagnose and resolve the installation issue more efficiently.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1400972756421443615)** (1 messages): 

> `Tail Call Elimination, Print/Log Statements, Minimal Examples` 


- **Tail Call Elimination Triggers**: A member is creating a minimal example and noticed that **tail call elimination** doesn't trigger if basic **print/log statements** are added to the functions.
   - The member is asking why that might be the case.
- **Print/Log Statements Impact Tail Call Elimination**: The discussion centers on how adding **print/log statements** can prevent **tail call elimination** in minimal examples.
   - The member seeks to understand the underlying reasons for this behavior, specifically when creating minimal examples.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1400781766913949827)** (3 messages): 

> `OpenAI Model Leak, Mixture of Experts, FP4 weights` 


- **OpenAI's Alleged Model Leak**: It is rumored that **OpenAI** has a **leaked model** with **128 experts** and **120B parameters**.
   - The model's weights are allegedly in **FP4** format, indicating a highly compressed or quantized state.
- **Deep dive into MoE**: **Mixture of Experts** models are composed of multiple sub-networks, known as *experts*, with a gating network that learns to route each input to the most relevant experts.
   - This is an area of active research as this enables scaling model size without a proportional increase in compute costs.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1400911694011699361)** (1 messages): 

> `Course Quizzes Availability, Google Forms Reopening` 


- **Quizzes with Answer Keys Now Available Online**: An archive of the **quizzes (with answer key)** is available in the *"Quizzes"* section of the course website.
   - This provides students with a valuable resource for reviewing course material and assessing their understanding.
- **Google Forms for Quizzes Will Not Be Reopened**: The course staff has announced that they will not be able to reopen the **Google Forms** used for quizzes.
   - Students who missed the opportunity to take the quizzes through **Google Forms** should utilize the available archive for review.


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1400899899402489856)** (1 messages): 

> `Qwen3-Coder, Token Speed, US Servers` 


- **Qwen3-Coder Lands on Windsurf with Lightning Speed**: **Qwen3-Coder** is now live in Windsurf, clocking in at approximately **2000 tokens/sec**.
   - The launch was announced on [X](https://x.com/windsurf/status/1951340259192742063) and [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button), and is fully hosted on US servers.
- **Windsurf houses Qwen3-Coder**: A new blazing fast model named **Qwen3-Coder** is in Windsurf.
   - Running at 2000 tokens per second, discussions are being had on [Reddit](https://www.reddit.com/r/windsurf/comments/1mf3e5s/qwen3coder_at_2000_tokenssec_is_now_live_in/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) about it's implications.

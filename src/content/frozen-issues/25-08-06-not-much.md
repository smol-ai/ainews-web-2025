---
id: MjAyNS0w
title: not much happened today
date: '2025-08-06T05:44:39.731046Z'
description: >-
  **OpenAI** released its first open models since GPT-2, **gpt-oss-120b** and
  **gpt-oss-20b**, which quickly trended on **Hugging Face**. **Microsoft**
  supports these models via **Azure AI Foundry** and **Windows Foundry Local**.
  Key architectural innovations include **sliding window attention**, **mixture
  of experts (MoE)**, a **RoPE variant**, and a **256k context length**. The
  models use a new **MXFP4** format supported by **llama.cpp**. Hypotheses
  suggest **gpt-oss** was trained on **synthetic data** to enhance safety and
  performance, supporting the **Reasoning Core Hypothesis**. **OpenAI**
  announced a **$500K bounty** for red teaming with partners including
  **Anthropic**, **Google**, and the **UK AISI**. Performance critiques
  highlight inconsistent benchmarking results, with **GPT-OSS-120B** scoring
  **41.8%** on the **Aider Polyglot** coding benchmark, trailing competitors
  like **Kimi-K2** and **DeepSeek-R1**. Some users note the model excels in math
  and reasoning but lacks common sense and practical utility.
companies:
  - openai
  - huggingface
  - microsoft
  - llamaindex
  - ollama
  - baseten
  - fireworksai
  - cerebras
  - groq
  - together
  - anthropic
  - google
  - uk-aisi
models:
  - gpt-oss-120b
  - gpt-oss-20b
  - kimi-k2
  - deepseek-r1
  - qwen-3-32b
topics:
  - sliding-window-attention
  - mixture-of-experts
  - rope
  - context-length
  - mxfp4-format
  - synthetic-data
  - reasoning-core-hypothesis
  - red-teaming
  - benchmarking
  - coding-benchmarks
  - model-performance
  - fine-tuning
people:
  - woj_zaremba
  - sama
  - huybery
  - drjimfan
  - jxmnop
  - scaling01
  - arunv30
  - kevinweil
  - xikun_zhang_
  - jerryjliu0
  - ollama
  - basetenco
  - reach_vb
  - gneubig
  - shxf0072
  - _lewtun
---


**a calm before the storm.**

> AI News for 8/5/2025-8/6/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 8597 messages) for you. Estimated reading time saved (at 200wpm): 830 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

Tune in to the OpenAI livestream at 10am PT tomorrow.

Meanwhile you can tune in to [today's pod](https://www.youtube.com/watch?v=t4IQwMa5-6U&t=3s) about how the press gets leaks and covers major AI startups.

---

# AI Twitter Recap

**OpenAI's GPT-OSS Release & Architecture**

- **Official Announcement and Community Integrations**: **OpenAI** announced the release of its first open models since GPT-2, **gpt-oss-120b** and **gpt-oss-20b** ([link](https://twitter.com/arunv30/status/1952881931798143276)), which quickly became the top trending models on **Hugging Face** ([link](https://twitter.com/kevinweil/status/1952969984931709376)). **Microsoft** announced support for the models in **Azure AI Foundry** and on **Windows** via Foundry Local ([link](https://twitter.com/xikun_zhang_/status/1952902211278913629)). Numerous platforms announced immediate support, including **LlamaIndex** for agentic workflows ([link](https://twitter.com/jerryjliu0/status/1952883595787239563)), **Ollama** with web search ([link](https://twitter.com/ollama/status/1952882173255856223)), **Baseten** ([link](https://twitter.com/basetenco/status/1952882156059148737)), and a public demo powered by **HF Inference Providers** like **FireworksAI**, **Cerebras**, **Groq**, and **Together** ([link](https://twitter.com/reach_vb/status/1953041435999010916)). To encourage development, **OpenAI** and **Hugging Face** are offering **$50 in inference credits** to 500 students ([link](https://twitter.com/reach_vb/status/1953010091377958984)).
- **Architectural Details**: [@gneubig](https://twitter.com/algo_diver/status/1952941862723162439) and others summarized the key architectural innovations, including **sliding window attention**, **mixture of experts (MoE)**, a specific **RoPE variant**, and a **256k context length**. [@shxf0072](https://twitter.com/shxf0072/status/1953143243992166849) noted that attention only makes up **0.84%** of the model, with intelligence stored in the **99.16% MLP layers**. The models use a new **MXFP4** format, which **llama.cpp** now natively supports ([link](https://twitter.com/ggerganov/status/1952978670328660152)). [@_lewtun](https://twitter.com/_lewtun/status/1952990532436934664) highlighted **OpenAI's guide** on fine-tuning the models.
- **Hypotheses on Training and Design**: A popular hypothesis shared by [@huybery](https://twitter.com/huybery/status/1952905224890532316) is that **gpt-oss** was trained entirely on **synthetic data**, enhancing safety and performance. [@DrJimFan](https://twitter.com/DrJimFan/status/1953139796551086265) suggested this supports the **"Reasoning Core Hypothesis,"** where reasoning needs minimal linguistic competency, aligning with the concept of a lightweight **"LLM OS Kernel."** [@jxmnop](https://twitter.com/jxmnop/status/1953218992954589525) commented that **Sam Altman** seems to want a model that is highly skilled (e.g., rated 3200 on Codeforces) but lacks knowledge of real-world entities like himself.
- **Red Teaming and Security**: **OpenAI's** [@woj_zaremba](https://twitter.com/woj_zaremba/status/1952886644090241209) announced a **$500K bounty** to stress-test the new models, with findings to be reviewed by a coalition including **OpenAI, Anthropic, Google**, and the **UK AISI**.

**GPT-OSS Performance, Benchmarks, and Criticism**

- **Inconsistent Performance and "Benchmarking"**: Multiple users, including [@Teknium1](https://twitter.com/Teknium1/status/1953063858761023843), observed that the model seems to have been "benchmaxxed," leading to strange performance profiles. [@scaling01](https://twitter.com/scaling01/status/1952881329772564764) noted it is "slopmaxxed on math/coding and reasoning" but lacks "taste and common sense." [@jxmnop](https://twitter.com/jxmnop/status/1953216881361600729) found it could code professionally one moment and then confidently hallucinate basic facts the next.
- **Aider Polyglot Benchmark Results**: The **GPT-OSS-120B** model performed poorly on the **Aider Polyglot** coding benchmark, scoring **41.8%**. [@scaling01](https://twitter.com/scaling01/status/1953047534122713130) pointed out this is significantly lower than competitors like **Kimi-K2 (59.1%)** and **DeepSeek-R1 (56.9%)**, though slightly better than **Qwen3 32B (40.0%)**. This led to questions about the model's practical utility beyond math and reasoning tasks ([link](https://twitter.com/scaling01/status/1953047913954791696)).
- **Comparisons to Chinese Models and Lack of Credit**: [@scaling01](https://twitter.com/scaling01/status/1952900225120780705) argued that there is "no western open-source model that beats or ties the best chinese open-source models," citing **Qwen3-235B-A22B, R1, and GLM-4.5** as superior to **GPT-OSS-120B**. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1952898957035880923) criticized **OpenAI** for not acknowledging **DeepSeek**, whose architecture and techniques appear to have been influential.
- **Emergent Abilities and Quirks**: Users observed interesting emergent behaviors, such as the model's ability to perform complex math calculations without tools ([link](https://twitter.com/scaling01/status/1952892387539259455)) and its attempt to brute-force base64 decoding ([link](https://twitter.com/scaling01/status/1952886371019809037)). [@Teknium1](https://twitter.com/Teknium1/status/1952892281218023824) and others noted its tendency to refer to itself as "we," likening it to a "borg cube."

**Google's Genie 3 and Other AI Advances**

- **Genie 3 Interactive World Model**: **Google DeepMind** unveiled **Genie 3**, a "groundbreaking world model" capable of generating entire interactive, explorable, and playable environments from text or video inputs ([link](https://twitter.com/denny_zhou/status/1952887267963662429)). The announcement sparked excitement about the future of "neural video games" ([link](https://twitter.com/demishassabis/status/1952890039643353219)), with [@jparkerholder](https://twitter.com/GoogleDeepMind/status/1953024241017749604) calling it a "watershed moment for world models." Users highlighted its ability to simulate complex rendering effects and generate new content on the fly (link).
- **Sparks of In-Context Learning and Integration**: **Genie 3** demonstrates sparks of in-context learning by allowing users to provide a video (e.g., from **Veo 3**) and then control the simulation from there, mimicking the original video's dynamics ([link](https://twitter.com/_rockt/status/1953117236975030653)). The potential for inception-like scenarios, where a **Genie 3** simulation runs within another **Genie 3** simulation, was also demonstrated ([link](https://twitter.com/shlomifruchter/status/1953155882902274126)).
- **New Educational Tools and Gemini Updates**: **Google** launched the **AI for Education Accelerator**, committing **$1 billion** to AI literacy and providing free AI training and **Google Career Certificates** to college students ([link](https://twitter.com/Google/status/1953126394847768936)). Updates to the **Gemini App** include a **Storybook** feature for creating personalized, illustrated stories ([link](https://twitter.com/demishassabis/status/1952897414207074810)) and new learning tools like **Guided Learning**, flashcards, and integrated visuals ([link](https://twitter.com/Google/status/1953143185011617891)).

**Agent Tooling, Development, and Frameworks**

- **Claude Code Course and Security Features**: **Andrew Ng** and **DeepLearningAI**, in collaboration with **Anthropic**, released a course on **Claude Code**, focusing on highly agentic coding workflows ([link](https://twitter.com/AndrewYNg/status/1953097967361245251)). **Anthropic** also announced that **Claude Code** can now automatically review code for security vulnerabilities ([link](https://twitter.com/AnthropicAI/status/1953135070174134559)).
- **LangChain Introduces Open SWE**: **LangChain** announced **Open SWE**, an open-source, cloud-based asynchronous coding agent that can be connected to a GitHub repository to autonomously resolve issues ([link](https://twitter.com/Hacubu/status/1953168346356314376)). The agent works by breaking down problems, writing code, and creating pull requests.
- **LlamaIndex and LlamaCloud**: **LlamaIndex** showcased integrations for financial document agents ([link](https://twitter.com/jerryjliu0/status/1953108641558540720)) and its collaboration with **Delphi** to create "digital minds" using **LlamaCloud** as a context layer for document ingestion ([link](https://twitter.com/jerryjliu0/status/1952889056200655206)). They also introduced a new "balanced" parsing mode in **LlamaCloud** for cost-effective analysis of visual elements like charts and graphs ([link](https://twitter.com/jerryjliu0/status/1953227974716665996)).
- **RAG and Chunking**: **DeepLearningAI** emphasized the need for observability in production-ready **RAG** systems to track performance and quality ([link](https://twitter.com/DeepLearningAI/status/1952886740349272173)). [@femke_plantinga](https://twitter.com/bobvanluijt/status/1953013722026250737) argued that developers should "stop optimizing your retrieval" and "fix your chunking first," as it is often the root cause of poor performance.

**Infrastructure, Hardware, and Efficiency**

- **Ollama vs. ggml Performance**: [@ggerganov](https://twitter.com/ggerganov/status/1953088008816619637) explained that **LMStudio's** performance with **GPT-OSS** is significantly better because it uses the upstream **ggml** implementation. He noted that **Ollama's** fork has inefficient implementations for **MXFP4** kernels and attention sinks, leading to poor performance.
- **Inference Provider Performance and Correctness**: **vLLM** stated that they have run many evals and that the numerics on **Hopper** GPUs should be "solid and verified" ([link](https://twitter.com/vllm_project/status/1952940603773468926)). However, users like [@AymericRoucher](https://twitter.com/AymericRoucher/status/1953115586273394873) noted that performance can vary wildly between providers, likely due to aggressive quantization. **Groq** received praise for delivering solid results and high speeds, with the **120B model** running at over **500 tokens/second** ([link](https://twitter.com/JonathanRoss321/status/1953119620103381440)).
- **Quantization and Hardware Support**: **Cerebras** announced **GPT-OSS-120B** running live on their systems at **3,000 tokens/s** ([link](https://twitter.com/cline/status/1952960760759632025)). [@HaihaoShen](https://twitter.com/teortaxesTex/status/1953017577900228920) released what is likely the first **INT4** quantized version of **GPT-OSS**. The community also highlighted the potential of **AMD GPUs** for running local models, with one user showing the **20B model** running at **52 tok/sec** on a sub-$1000 laptop ([link](https://twitter.com/dzhng/status/1953132623280165193)).

**Humor & Memes**

- **On Hype and Releases**: [@gdb](https://twitter.com/gdb/status/1953184691567349976) posted "team has been working super hard, excited for tmrw!", fueling speculation around a GPT-5 release. [@nrehiew_](https://twitter.com/nrehiew_/status/1953142337745633373) posted a list of hopes for the next model, including "Please not be benchmaxxed" and "Please have some soul."
- **On Model Behavior**: [@Teknium1](https://twitter.com/Teknium1/status/1953063858761023843) declared, "This is what happens when you benchmax ngl," capturing the community sentiment about GPT-OSS's strange performance. [@code_star](https://twitter.com/code_star/status/1953153930944446852) joked, "Its only real docker if it comes from the container region of France. Everything else is sparkling hypervisor."
- **Relatable Developer Struggles**: [@fabianstelzer](https://twitter.com/fabianstelzer/status/1953150053050101785) posted a meme about "vibe coding" an entire app and then being asked if API keys are in environment variables. [@jxmnop](https://twitter.com/jxmnop/status/1953163073612562851) posted a meme with the caption "rule number one: never distill from DeepSeek".

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-4B-Thinking-2507 Model Release and Discussion

- [**🚀 Qwen3-4B-Thinking-2507 released!**](https://i.redd.it/3cl3vbg54fhf1.jpeg) ([Score: 883, Comments: 98](https://www.reddit.com/r/LocalLLaMA/comments/1mj7t51/qwen34bthinking2507_released/)): **The image appears to show benchmark results or capability comparisons for the newly released Qwen3-4B-Thinking-2507 model, which is notable for its enhanced reasoning, long-context handling (256K tokens), and alignment. Technical discussion in the post and comments highlights the model's significant improvement in reasoning benchmarks, such as BFCL-v3 (scoring 71.2), and its performance close to large models like GPT-4o, particularly impressive given its small 4B parameter size. The release on Hugging Face includes several GGUF quantizations for immediate deployment, and there are community requests to benchmark it against other strong open models like gpt-oss-20b.** Commenters discuss the tradeoff between hybrid reasoning versus specialized reasoning-only models, with consensus that separate models yield superior performance. The BFCL-v3 number is praised, and there's interest in benchmarking against larger open models. Deployment tools like LMStudio are mentioned as enabling rapid adoption.
    - Hybrid Reasoning appears to negatively impact LLM performance, as indicated by users, and some recommend maintaining separate general-purpose and reasoning-specialized versions for better results.
    - Qwen3-4B-Thinking-2507 achieved a notable BFCL-v3 benchmark score of 71.2, which is unprecedented for models of this size and approaches performance typical of more advanced models like GPT-4o. Multiple GGUF quantizations (Q3, Q4, Q6, Q8) are already available in lmstudio for immediate use.
    - There are requests for direct benchmarking comparisons with models like gpt-oss-20b and Gemma3n4b, suggesting an expectation that Qwen3-4B-Thinking-2507 could outperform much larger or recently released models. Additionally, the model supports a substantial 256k context window, which stands out significantly for a 4B parameter LLM.
- [**Just when you thought Qwen was done...**](https://www.reddit.com/r/LocalLLaMA/comments/1mj7pny/just_when_you_thought_qwen_was_done/) ([Score: 336, Comments: 75](https://www.reddit.com/r/LocalLLaMA/comments/1mj7pny/just_when_you_thought_qwen_was_done/)): **Qwen has released new checkpoint versions for the Qwen3 4B model: [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) and [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), indicating ongoing improvements beyond their prior major release and positioning themselves as a strong open-source LLM competitor. Details or benchmarks specific to "-Thinking-2507" and "-Instruct-2507" are not listed in the post, but the community is taking note of Qwen's continued high-velocity development.** Technical discussion from the comments calls for a Qwen3 Coder 32B and a smaller 1.7B variant for efficient speculative decoding, suggesting such models could rival GPT-OSS 120B at similar inference speeds, reflecting strong demand for both highly capable and efficient open-source models.
    - A user suggests the potential for a Qwen3 Coder 32b model combined with a smaller 1.7b model for speculative decoding, noting that such a configuration should *outperform the 120b gpt-oss* at a similar inference speed. This highlights a growing interest in mixed-scale model inference strategies for improved efficiency and effectiveness.
    - There's discussion about Qwen's model release strategy, with one commenter noting that releasing model sizes incrementally (rather than all at once) keeps public attention focused and maximizes hype. This implies an intentional staggered launch approach for broader visibility and engagement.
- [**Qwen isn't stopping !! (And trolling sama lol)**](https://i.redd.it/3nhqo0qf9fhf1.jpeg) ([Score: 506, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mj8lk8/qwen_isnt_stopping_and_trolling_sama_lol/)): **The image references 'Qwen,' the open-source language model series developed by Alibaba, and appears to showcase a recent release, performance results, or leaderboard positioning (the exact content of the image is unclear due to failed analysis). The post title and comments imply that Qwen is making rapid and significant advances, possibly outpacing competitors or actively rivaling OpenAI (with 'trolling sama' referring to Sam Altman). Comments discuss the frequent release cadence and speculate about future models like 'qwen3-coder-thinking 30B,' suggesting anticipation of larger or more specialized models.** Discussion centers around motivations for frequent releases, with some questioning the incentives behind Alibaba's aggressive open-sourcing, while others speculate on competitive strategy. There's also excitement and hope expressed for even more capable future Qwen models.
    - A user expresses anticipation for a potential 'qwen3-coder-thinking 30B' model, implying interest in seeing a Qwen series code-focused language model that leverages a 30B parameter architecture. This reflects ongoing technical discussions on model scaling strategies and task specialization (e.g., coding capabilities) in open-source LLM development.
    - Comparisons are made to GPT-based open-source projects ('Nothing wrong with GPT oss...'), suggesting ongoing benchmarking and discussions within the community regarding performance parity or advantages of Qwen models versus established open-source GPT architectures.
    - Community observers are noting the accelerated release frequency from the Qwen team ('Qwen teams speed is inspiring!'), which may indicate rapid iteration, improved pipeline automation, or streamlined data/modeling processes within their workflow.

### 2. OpenAI Model Safety, Naming, and Community Reaction Memes

- [**OpenAI, I don't feel SAFE ENOUGH**](https://i.redd.it/af6jm3nt9bhf1.png) ([Score: 1444, Comments: 140](https://www.reddit.com/r/LocalLLaMA/comments/1misyvc/openai_i_dont_feel_safe_enough/)): **The image is a meme, referencing perceived issues with OpenAI's safety policies or recent events. The post and comments do not discuss any benchmarks, models, or technical details—it's a satirical response to OpenAI's corporate strategy or communications around AI safety.** Commenters note OpenAI 'chose to become a meme,' indicating a lack of seriousness or disconnect between the company's messaging and user perception; the discussion is largely humorous and non-technical.
    - The most technically relevant detail discussed is that OpenAI's latest model has a training data cutoff of June 2024, which means it lacks knowledge of events or data published after that point. For example, it cannot answer questions about the results of the most recent (mid/late 2024) election or other very recent developments, highlighting a limitation in up-to-date factual awareness.
- [**LEAK: How OpenAI came up with the new models name.**](https://i.redd.it/d60vtzhkkehf1.png) ([Score: 378, Comments: 21](https://www.reddit.com/r/LocalLLaMA/comments/1mj4zkk/leak_how_openai_came_up_with_the_new_models_name/)): **The image referenced in the post is a meme, satirizing the naming process of OpenAI's new model, likely in response to their recent use of the 'Open Source' label. The conversation in the comments discusses skepticism about whether OpenAI actually open-sourced the dataset and training code behind the model, reflecting a common debate within the AI community about the difference between making models available and true open-sourcing. No technical benchmarks or direct implementation details are provided in the post or comments.** Several commenters are critical of OpenAI's use of 'open source', hinting that the company may be misusing the term for marketing, especially if datasets and training code are not actually released. This echoes ongoing controversies in the field regarding what constitutes true open-source AI.
    - A user raises a technical concern about the distinction between models being "open sourced" versus merely "freely available." They question whether OpenAI released both the dataset and training code along with the model, pointing out that in AI, the term open source is frequently misused when actual source code or data is not shared. This highlights the ongoing debate in the machine learning community about transparency and reproducibility of model releases.
- [**Safemaxxed for your safety!**](https://i.redd.it/gaqdycledchf1.png) ([Score: 369, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mix2kg/safemaxxed_for_your_safety/)): **The image appears to be a meme or satirical commentary on the increasing safety restrictions and content moderation ("safemaxxing") implemented by closed-source AI models, particularly from OpenAI, as referenced by commenters. The discussion highlights concerns regarding over-censorship and increasing bias in these models, pointing to business decisions that might lead to adopting the most restrictive global policies, as OpenAI expands into markets like the Middle East. The technical community is advised to preserve open-source models in anticipation of further lockdowns.** Commenters express skepticism and frustration about the growing restrictions on AI models, with some specifically criticizing OpenAI for exceeding anticipated levels of censorship and warning that global business expansion may further tighten controls, threatening open discourse even on technical topics.
    - A user highlights the business-driven restrictions on mainstream LLMs like OpenAI's, noting that company policies must conform to the most restrictive jurisdiction where they're active. This could result in further limiting model outputs, especially as they expand into regions like the Middle East, and may restrict information on sensitive topics such as civil rights.
    - Another commenter asks for recommendations on the least restricted current large language models suitable for local inference on hardware with 32GB RAM and a 12GB 3060 GPU, implicitly prompting a technical discussion about model size, VRAM needs, and viable open-source alternatives for less-constrained use-cases.
    - Discussion arises about the tradeoff between model safety/alignment work ("lobotomizing" the models) and development speed or utility. Some suggest that extensive safety tuning slows progress and limits creative applications, with a call for "abliberated" or uncensored versions for research or non-standard use cases.
- [**"What, you don't like your new SOTA model?"**](https://i.redd.it/9yqb0l1n9chf1.png) ([Score: 737, Comments: 123](https://www.reddit.com/r/LocalLLaMA/comments/1miwrli/what_you_dont_like_your_new_sota_model/)): **The image is a meme, referencing the announcement and media excitement surrounding a new 'SOTA' (state-of-the-art) model, likely by OpenAI. Context from the title and comments indicates skepticism among technical users about the true novelty or impact of the released model, suggesting its primary audience is mainstream media and investors rather than the technical community familiar with recent advancements. The humor highlights the perceived disconnect between corporate marketing and genuinely new technical breakthroughs.** Commenters argue that major releases are often overhyped for non-technical audiences and investors, rather than for the ML practitioner community, and such excitement typically ignores existing open-source or prior academic efforts. The discussion reflects ongoing tensions about AI 'progress' narratives and who actually benefits from or evaluates these claims.
    - The discussion includes skepticism about the practical usefulness of OpenAI's newly released open-source model, with one user suggesting that OpenAI intentionally restricted ('lobotomized') its capabilities, raising curiosity about the level of effort and technical intervention needed to fine-tune the model back to a more usable or state-of-the-art (SOTA) performance level. This highlights ongoing concerns in the community regarding model alignment versus utility and the potential for open-source communities to restore or surpass original performance through additional fine-tuning or modified training procedures.
- [**How did you enjoy the experience so far?**](https://i.redd.it/lj67oslhbdhf1.png) ([Score: 341, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mj00mr/how_did_you_enjoy_the_experience_so_far/)): **The post discusses the experience using a new model, focusing on its heavy alignment/safety restrictions and characteristics similar to Phi. The top comment provides a detailed technical comparison of the unnamed model (possibly referencing Googy or an OpenAI-aligned analog) with other LLMs, highlighting its conservative safety alignment ('never forgets that ... it's first and foremost a member of OpenAI's HR department'), strong STEM knowledge for its size, and impressive code analysis abilities. Notable technical benchmarks include its passing of Grosper's LFT continued-fraction functions test (in OCaml), its ability to answer advanced physics and math questions, and its performance on tasks like base64/ascii/rot13 decoding. However, the model's over-reliance on tables and signs of synthetic data, as well as integration issues with modern AI tooling, are cited as limitations. Summarization and entity extraction performance are noted as good, but with caveats regarding alignment/safety interference.** The technical debate centers on the trade-off between strong safety alignment (to the point of user experience frustration and possible knowledge truncation) and the model's competency in code, STEM, and certain problem domains. Users express skepticism about the model's worth due to heavy safety restrictions and debate whether its capabilities offset its alignment limitations.
    - A detailed technical comparison is provided between several models (Goody/GPT-phi-3.5 prototype, Qwen-30B MoE, GLM 4.5 Air) on advanced tasks such as differentiating reverse and forward mode AD with continuation passing and handling mathematical/algorithmic OCaml code (e.g., Grosper's LFT continued-fraction functions). Goody 20B failed the reverse-mode AD reasoning but handled the OCaml test with competent algorithmic recognition, though not perfect identification. Qwen-30B MoE and Goody 120B had similar limitations, with GLM 4.5 Air being the only one to succeed without hints.
    - The commenter notes a likely heavy reliance on synthetic data for training these models, inferred from a pronounced tendency to construct tabular outputs. Stem-topic benchmarks such as black hole physics, pointer states in Many Worlds, and probability-based D&D spellcasting questions are cited as examples where the 20B model provides 'SOTA for its size' (state-of-the-art relative to parameters), pointing to good but uneven general reasoning and domain knowledge.
    - There's an evaluation of practical NLP capabilities: the 20B can accurately decode base64, ASCII binary, rot13, and even chained rot13+base64 encoding if explicitly told, outperforming Qwen-30B in some decoding tasks. Summarization and entity/concept extraction are also noted as strengths, though there is concern regarding overreliance on corporate-safe answers and lack of integration with modern AI tooling.

### 3. Elon Musk's Promise to Open Source Grok 2 and Industry Skepticism on GPT-OSS

- [**Elon Musk says that xAI will make Grok 2 open source next week**](https://i.redd.it/htgw3mmvjdhf1.jpeg) ([Score: 420, Comments: 175](https://www.reddit.com/r/LocalLLaMA/comments/1mj0snp/elon_musk_says_that_xai_will_make_grok_2_open/)): **The image appears alongside the announcement that Elon Musk's xAI will open source Grok 2 next week, as confirmed by Musk's post on X (formerly Twitter). Technically, Grok 2 is being released after more competitive and capable models, with comments noting that Grok 2 is considered both larger and less performant than current state-of-the-art open-weight models. Additional context from comments indicates that later Grok versions achieve better reasoning via reinforcement learning (RL), suggesting Grok 2's forthcoming open-sourcing may be of limited practical use.** Commenters debate the significance, with some criticizing the late timing and perceived poor performance of Grok 2 versus its contemporaries, while others discuss trends in open sourcing as a response to industry pressure rather than technical merit.
    - There is a suggestion that Grok 4 is effectively just Grok 3 with additional reinforcement learning (RL) applied to enhance its reasoning capabilities, which could explain xAI's reluctance to open source Grok 3 compared to Grok 2.
    - Technical criticism points out that Grok 2 is both significantly larger and performs considerably worse than modern open models, implying that its upcoming open-sourcing may have negligible practical relevance given today’s model landscape.
    - There’s a meta-discussion about the rapid acceleration and competitiveness in open model releases: while Mixtral 8x7b was seen as a peak open-weights model in late 2023, leading companies now rush to release new models to avoid being perceived as falling behind, highlighting industry dynamics rather than model merit.
- [**GPT-OSS looks more like a publicity stunt as more independent test results come out :(**](https://i.redd.it/onk13jqo0ehf1.jpeg) ([Score: 661, Comments: 183](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/)): **The image appears to present benchmark results comparing GPT-OSS with other coding models, highlighting its lower performance relative to updated versions of DeepSeek-R1 (71.4% on the 0528 version) and GLM 4.5 Air, as well as Qwen 3 32B. Commenters clarify benchmark specifics, correct earlier misattributions, and emphasize that while GPT-OSS is an FP4 model with sparse MoE, its aggressive safety tuning has negatively impacted performance. Dense models like Qwen 3 32B require more memory and are slower, providing some context to the discussion of efficiency versus capability.** Technically substantive debate focuses on version discrepancies in benchmark reporting, relative strengths/weaknesses of model architectures (sparse MoE vs dense), and the trade-off between safety tuning and model usability, with the community suggesting that targeted finetuning could improve GPT-OSS performance.
    - A commenter points out that benchmark comparisons using DeepSeek-R1 should reference the latest version (0528) which scores 71.4%, not the older 0120 version at 56.9%, highlighting the importance of referencing up-to-date test results.
    - Discussions highlight that models like GLM 4.5 Air outperform GPT-OSS in coding benchmarks at similar parameter sizes. Qwen 3 32B, though comparable in memory use, is denser and slower to run, and Qwen's 30B-A3B coder only gets about 52% on the same benchmarks, noting overall GPT-OSS's relatively weak coding performance among its peers.
    - Several technical remarks discuss trade-offs: GPT-OSS, as a sparse MoE with 5 active parameters, is more practical for users with moderate RAM, achieving 5t/s on dual channel DDR4-3200. However, aggressive safety tuning appears to have limited its reasoning ability, and some suggest a community-driven finetune could improve practical usability.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Genie 3 Interactive World Generation and Hype

- [**Genie 3 turns Veo 3 generated drone shot into an interactive world you can take control mid-flight**](https://v.redd.it/d51vt9w83ghf1) ([Score: 1450, Comments: 248](https://www.reddit.com/r/singularity/comments/1mjd3mk/genie_3_turns_veo_3_generated_drone_shot_into_an/)): **The post demonstrates the use of Genie 3 to convert a Veo 3-generated drone video into an *interactive environment* that allows users to assume control mid-flight. Genie 3 employs generative AI to map 2D video frames to controllable 3D spaces, suggesting major advances in real-time video-to-environment conversion, potentially for applications in gaming, simulation, and urban modeling.** Top comments highlight the potential for combining Genie 3 with technologies like Google Maps and VR for broader use-cases, as well as the transformative impact on industries ranging from gaming to societal applications.
    - Commenters speculate about combining Genie 3 with Google Maps and VR to create fully immersive and explorable 3D worlds, suggesting potential use cases for simulation and navigation technology. Technical readers might infer the challenge lies in integrating live drone imagery or video (from Veo 3) with real-time AI environment generation (Genie 3) and VR rendering—all at interactive frame rates.
    - The integration of Genie 3 with VR platforms like the Quest 2 is discussed, implicitly raising technical questions about interface compatibility and the computational requirements needed to achieve seamless user experience in immersive, real-time 3D environments created from video or drone footage.
- [**Genie 3 Is Insane🤯](https://v.redd.it/y7s2bce70ghf1)https://x.com/jkbr_ai/status/1953154961988305384?s=46** ([Score: 626, Comments: 105](https://www.reddit.com/r/singularity/comments/1mjcm7i/genie_3_is_insanehttpsxcomjkbr/)): **The referenced post showcases Genie 3, an AI model by Google DeepMind capable of generating highly realistic interactive 3D environments from text or image prompts, demonstrated by recursively spawning a Genie-generated agent *within* another Genie-generated world ("playing with Genie inside Genie"). The demonstration shocked viewers with its realism, blurring the line between synthetic and real imagery, and highlights advancements in generative scene rendering, as discussed by its researchers [here](https://x.com/jkbr_ai/status/1953154961988305384?s=46).** Commenters express astonishment at Genie 3's realism, with some initially mistaking its outputs for real-life footage. The recursive use of Genie to generate environments and agents prompted technical curiosity and surprise about the depth of model capabilities.
    - One comment references a statement from one of the researchers, provided in an image. The statement likely offers insight into Genie 3's unique technical capabilities or researcher perspective, but without its plain text content, the specifics can't be fully summarized here. However, the focus on a researcher's comment highlights community interest in official clarifications and technical details.
    - There is notable confusion around the realism of Genie 3's generated scenes, with a user expressing disbelief and stating they thought the output was *real life*. This points to significant advancement in Genie 3's video or simulation fidelity, suggesting it may be approaching photorealistic or highly convincing interactive visuals.
- [**The Hype of Google Genie 3 is already bigger than of OpenAI OSS**](https://www.reddit.com/r/singularity/comments/1mjebri/the_hype_of_google_genie_3_is_already_bigger_than/) ([Score: 350, Comments: 45](https://www.reddit.com/r/singularity/comments/1mjebri/the_hype_of_google_genie_3_is_already_bigger_than/)): **Discussion centers around the significant excitement generated by Google's Genie 3, a generative AI model for interactive environments (not just text), which is viewed as a technical leap compared to OpenAI's Open Source Small (OSS) model. Genie 3's capability for emergent generative behavior is contrasted with OSS, which is characterized as another LLM with results outperformed by existing open-source alternatives.** Commenters argue Genie 3 is more comparable to models like Veo 3 and assert that OSS is underwhelming both in benchmarks and community reception, with some stating Chinese open-source models are stronger than OSS.
    - Several comments emphasize that Google's Genie 3 represents a fundamental advance in generative AI, distinguishing it from OSS, which is described as *just another LLM*; the proper comparison for Genie 3 would be against multimodal models like Google's own Veo 3 rather than traditional text-based LLMs.
    - There is a technical viewpoint that OSS (OpenAI's Open Source model) is underwhelming and gets outperformed by various same-size open source models, with particular mention that *Chinese models are still ahead when it comes to open source*, suggesting OSS lags in benchmarks and real-world performance relative to its international peers.
    - A nuanced discussion highlights that open sourced models, while less performant for general users, provide significant value for the research and local deployment community by enabling local inference and finetuning, marking a step forward for accessible AI research but not necessarily for mass-market excitement.
- [**Exploring terrains with Genie 3**](https://v.redd.it/efy108hfdfhf1) ([Score: 294, Comments: 105](https://www.reddit.com/r/singularity/comments/1mj97jw/exploring_terrains_with_genie_3/)): **The post discusses the use of Genie 3, the latest iteration of the Genie generative model, for creating explorable virtual terrains, with users expressing interest in applications like open-world landscape exploration. Genie 3 can synthesize interactive 3D environments from prompts, raising questions about accessibility, necessary compute resources, and implementation details for generating persistent, high-fidelity worlds.** Commenters note skepticism about the compute requirements of running Genie 3 ('everyone is going to need their own datacenter'), indicating concerns about scalability for personal or widespread use. There's ongoing interest in how users can gain hands-on access to Genie 3, reflecting demand for detailed onboarding and tooling documentation.
    - A notable technical point is the rapid progression between Genie 2 and Genie 3: Genie 2 was not real-time, whereas Genie 3 achieves significant advancements in both fidelity and interactivity, all within just one year. This highlights the rapid iteration and underlying model improvements, and prompts speculation about further exponential progress in the coming year.
- [**Type a sentence and Google Genie 3 creates an entire world to explore. When you step into a puddle, it splashes. Robots are now trained in simulated worlds like these.**](https://v.redd.it/uydgwkfrsdhf1) ([Score: 659, Comments: 126](https://www.reddit.com/r/OpenAI/comments/1mj1n8o/type_a_sentence_and_google_genie_3_creates_an/)): **Google DeepMind announced Genie 3, a world model system that procedurally generates interactive, high-fidelity simulated environments from a single text prompt, with real-time navigation at 720p/24fps for several minutes of consistent world state (see official announcement: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models). This technology is positioned as a foundational tool for training embodied AI–notably, the use case includes repeatedly training robots in arbitrarily generated, physics-consistent virtual scenarios.** Technical comments speculate Genie 3 and similar world models could eventually supplant traditional game engines for dynamic environment creation, and discuss high-impact use cases such as sim-to-real transfer in robotics (e.g., virtual rescue mission rehearsals for robots preceding real deployments).
    - A user proposes that AI, referencing Google's Genie 3, may soon supplant traditional game engines due to its ability to generate interactive simulated worlds on demand, which could significantly change how games and simulations are created and run in the near future.
    - Another comment highlights a practical application: using these AI-generated simulation environments for robotics training, such as repeatedly simulating rescue missions (e.g., cave rescues) to prepare robots for real-world scenarios, potentially improving safety and efficiency through extensive virtual testing.
- [**Genie 3 is incredible, well done google**](https://v.redd.it/4hkm4lhhkahf1) ([Score: 398, Comments: 26](https://www.reddit.com/r/Bard/comments/1mipx91/genie_3_is_incredible_well_done_google/)): **The original post praises Google's Genie 3 (presumably referring to a generative model or demo), but commenters identify that the media shown is actually from a movie, not AI-generated, casting doubt on the post's claim. Top comment specifically questions the AI-generated nature of the content, signaling possible confusion or misinformation about the technical capabilities being demonstrated.** Comments reflect skepticism about whether the showcased content is AI-generated, highlighting a technical discussion on distinguishing authentic outputs of generative models versus traditional media. No substantive debate on model architecture or benchmarks due to the post's misattribution.
    - Several commenters clarify that the content in question is sourced directly from the movie "Her" and not generated by an AI model such as Google's Genie 3, pointing out the importance of distinguishing between true AI-generated content and curated media. This highlights a recurring technical challenge in AI content evaluation: the need for robust provenance and watermarking tools to verify genuine model outputs versus reused or remixed traditional media.
- [**Exploring terrain with Genie 3**](https://v.redd.it/geqej05h4fhf1) ([Score: 103, Comments: 17](https://www.reddit.com/r/Bard/comments/1mj7wed/exploring_terrain_with_genie_3/)): **A demo video showcases Genie 3, an AI agent capable of real-time exploration and interaction within a procedurally generated 3D virtual terrain, displaying path planning that adapts to non-linear, off-rail movements (e.g., walking into streams and stepping on stones). Notably, the model is limited to 1 minute of state permanence, constraining long-term memory and interaction continuity; this is analogous to early context limitations in sequence models (e.g., 8,192 tokens). No public access has been confirmed.** Commenters are technically interested in the compute costs, noting the potential prohibitive resources required for real-time, persistent virtual worlds and expressing curiosity about public release and long-term feasibility.
    - Commenters note Genie 3's current limitation of only about 1 minute of world 'permanence' (state tracking/memory), drawing parallels to historical context window limitations in LLMs (e.g., 8,192 tokens), suggesting rapid advances are likely (ref: [MightyTribble]).
    - Speculation arises (see [a_tamer_impala]) that future versions, such as Genie 4, could integrate explicit 3D representations with simulated physics for more robust real-world grounding, representing a major technical milestone in generative environments.
    - There's technical curiosity about pathfinding: one observer (MightyTribble) expected rail-based navigation but observed adaptive environmental interaction (walking into streams and stepping on stones), indicating sophisticated environmental awareness and action generation beyond simple pathing.

### 2. Impending GPT-5 Model Launch Hype and Announcements

- [**GPT 5 Livestream Thursday 10 AM PT**](https://i.redd.it/9yz9yr88kfhf1.png) ([Score: 724, Comments: 155](https://www.reddit.com/r/singularity/comments/1mja836/gpt_5_livestream_thursday_10_am_pt/)): **The image is an announcement graphic for the scheduled GPT-5 livestream event by OpenAI, taking place Thursday at 10 AM PT, suggesting a major upcoming demonstration or announcement regarding the next iteration of the GPT model. This indicates the community expectation for significant technical advancements or new capabilities, given the considerable hype around GPT-5's potential.** Comments reflect anticipation and skepticism about the ability of the event to match the community's high expectations, with some users expressing excitement and others alluding to exaggerated outcomes for emphasis on the prevailing hype.
    - One commenter speculates that GPT-5 may combine the personality traits of GPT-4o (notably its engaging conversational style but with reduced sycophancy) and the raw intelligence level of GPT-4, resulting in a model that's a competent but not exceptional programmer, with broad knowledge coverage and a more confident demeanor. This underscores community interest in improvements to both personality tuning and coding capabilities over prior models.
- [**GPT-5 model art has now been pushed to the OpenAI CDN. With GPT-4.1 this happened a day before the launch - it's coming!**](https://i.redd.it/41mrj3z2nehf1.png) ([Score: 502, Comments: 108](https://www.reddit.com/r/singularity/comments/1mj5cc8/gpt5_model_art_has_now_been_pushed_to_the_openai/)): **The image is not a technical diagram or benchmark, but rather appears to be a piece of model icon art—specifically, the new GPT-5 logo that has been pushed to the OpenAI CDN. The significance is that this CDN update mirrors how assets for GPT-4.1 appeared just before its launch, strongly suggesting an imminent release of GPT-5. The post references the official OpenAI asset link: https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5.png.** Most comments are expressing anticipation, urgency, and comparisons to unrelated tech launches (e.g., GTA VI), rather than engaging in technical debate.
    - One user points out that the appearance of distinct 'mini' and 'nano' icons (https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5-mini.png, https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5-nano.png) on the OpenAI CDN likely confirms the imminent release of tiered GPT-5 variants, indicating a possible expansion of model offerings targeted at different latency, cost, or hardware requirement use cases.
    - A commenter suggests that the timing of the model art upload is technically significant, referencing prior releases such as GPT-4.1, where similar updates to the CDN occurred approximately one day before the official launch, implying a strong correlation between asset deployment and release schedules.
    - Discussion also touches on the broader AI scaling debate, with one user emphasizing the importance of GPT-5's launch as a moment to assess whether model scaling continues to yield substantial improvements, or if diminishing returns (“the wall”) are becoming apparent.
- [**Less than 24 hours until GPT-5!**](https://i.redd.it/a8di2qb4nfhf1.jpeg) ([Score: 232, Comments: 21](https://www.reddit.com/r/singularity/comments/1mjansy/less_than_24_hours_until_gpt5/)): **The post announces the imminent release of GPT-5 in less than 24 hours, but provides no concrete technical details except a countdown, and the image's context could not be analyzed. The comments express anticipation and concern about possible restricted access similar to prior OpenAI launches, emphasizing user experience over technical insights.** The main technical concern from commenters involves region- or subscription-based access restrictions, referencing past OpenAI rollouts where new models were limited to Pro users or Americans at launch.
    - There is a technical concern regarding the initial rollout of GPT-5's access, with users speculating that it may be limited to 'pro users' or restricted by region (such as US only) during the launch period, referencing past restricted access for previous model updates.
- [**GPT 5 to scale**](https://i.redd.it/v83znzjkvdhf1.jpeg) ([Score: 2152, Comments: 197](https://www.reddit.com/r/OpenAI/comments/1mj1xg9/gpt_5_to_scale/)): **The post, titled "GPT 5 to scale," likely features a humorous or meme-based image that purports to show the physical 'scale' of a hypothetical "GPT 5," mimicking a common internet trope of using objects (like a banana) for size comparison in non-technical contexts. The top comments parody requests for technical or scientific rigor by asking for familiar objects for scale, reinforcing the non-technical and satirical nature of the image. There is no actual technical content or documentation present.** Comments unanimously highlight the satirical intent, with comparisons to 'Apple level marketing' and ironic requests for additional scale objects, underlining that the post and image are not technical.
    - Several users critique the visual comparison of GPT-5's scale to physical objects such as the sun, highlighting that GPT models ('GPT5 may be 4 times bigger than the sun') should not be directly compared to physical objects in terms of size, since they exist as digital models rather than physical entities. The implication is that presenting model scale via real-world analogies can be technically misleading and does not accurately communicate relevant details like parameter count, architecture, or real-world impact on performance.
    - PainfullyEnglish notes that *bigger* does not necessarily mean *better* in model development (i.e., 'just because GPT5 is x times bigger than the sun doesn't mean it's actually better than the sun'), an important observation in the context of recent debates about the diminishing returns of simply scaling models versus focusing on architectural improvements or efficiency.
- [**GPT-5 UPDATE**](https://www.reddit.com/gallery/1mj6kn8) ([Score: 619, Comments: 158](https://www.reddit.com/r/OpenAI/comments/1mj6kn8/gpt5_update/)): **The post shares purported icons for new OpenAI models: GPT-5, GPT-5-NANO, and GPT-5-MINI, via official-looking CDN links describing their respective branding (e.g., [GPT-5 icon](https://cdn.openai.com/API/docs/images/model-page/model-icons/gpt-5.png)). No technical specs, benchmarks, or model details are disclosed—only iconography. No further information on architecture, capabilities, or release date accompanies the announcement.** Comments mostly focus on speculation and skepticism regarding release timing and the significance of icon-based "reveals," with no technical debate or substantive details released.
    - One commenter raises a technical question about the rumored release format of GPT-5, specifically asking whether it will be a singular model or if OpenAI will also release 'mini' and 'nano' variants, as has been the trend with some previous releases (such as GPT-4 Turbo, mini versions, etc.). This inquiry reflects ongoing community interest regarding parameter counts, scaling, and deployment options for diverse hardware and use cases.
    - Another comment references the iconography for GPT-5, which could indicate different product tiers (Free, Plus, and Pro), hinting at possible segmentation not only by access or feature set, but potentially corresponding to different model sizes or computational requirements—raising technical speculation about differential performance, pricing, or inference costs across tiers.
- [**GPT 5 tomorrow**](https://i.redd.it/amc8xjtvsfhf1.jpeg) ([Score: 123, Comments: 10](https://www.reddit.com/r/OpenAI/comments/1mjbjht/gpt_5_tomorrow/)): **The post discusses a rumored announcement of GPT-5, with speculation around its release timing ('10 AM PT') and whether current model performance (alleged 'dumbing down' before new releases) will be affected. There are also user queries about accessibility, specifically regarding whether GPT Plus subscribers will receive prompt access. The linked image functions as a hype or announcement teaser for GPT-5.** Commenters debate the recurring claim that previous GPT models are reduced in capability ('dumbed down') before major new releases and express uncertainty about whether this is myth or fact. There is also ongoing discussion about access tiers for new models.
    - A user asks whether the release of GPT-5 will involve a completely new set of model weights, implying a potentially significant architectural upgrade over GPT-4. This question is technically relevant as changes in weight initialization, training corpus, or model size typically drive major capability leaps (as seen from GPT-3 to GPT-4), and new weights would likely signal enhanced reasoning, wider context windows, or improved safety features.
    - Another comment questions whether existing GPT Plus subscribers will immediately gain access to GPT-5 upon release. This touches on the operational rollout process and the business model impacting user access, something technical users often track closely during version upgrades.
    - There is a discussion point raised about potential intentional reduction of intelligence/capabilities in previous models prior to a new release (e.g., claims of GPT-4 degrading before GPT-5 launch). While primarily speculative, this theme recurs in the community and influences perceptions on model performance, although there is no concrete technical evidence supporting systematic downgrades before past upgrades.
- [**Brace yourselves. The next evolution in AI just got the green light. GPT‑5 is coming…**](https://i.redd.it/twscxguvlfhf1.jpeg) ([Score: 591, Comments: 80](https://www.reddit.com/r/ChatGPT/comments/1mjah0h/brace_yourselves_the_next_evolution_in_ai_just/)): **The post announces that GPT-5, the next major iteration of OpenAI's foundational language model, has received the 'green light' for development, referencing a real confirmation from OpenAI's official X/Twitter account. The image itself is not technically informative but serves as a contextual marker for the anticipated release of GPT-5; no implementation details, benchmarks, or model specs are included in the post or image. Commenters express curiosity about the potential leap in capability from GPT-4 to GPT-5, with some unsure of the expected magnitude of improvement.** Discussion centers on whether the jump from GPT-4 to GPT-5 will be significant, with uncertainty over what 'GPT-5' will offer compared to its predecessor—no technical benchmarks or leaks are referenced.
    - One user noted the significance of progressing to "GPT-5," questioning if this naming indicates a major leap compared to previous incremental updates. This sparks discussion about whether the jump between GPT-4 and GPT-5 will be as significant as previous model jumps, given the trend in AI train sizes, architecture modifications, and capability scaling seen in benchmarks for GPT-3 to GPT-4.
    - Another user inquired about the scope of improvement expected from GPT-5, indicating that current public knowledge on GPT-5's technical advancements is limited, and suggesting that following official OpenAI channels or recent articles can provide technical updates as they become available.
- [**GPT5 announcement tomorrow 10am PT**](https://www.reddit.com/r/ChatGPT/comments/1mjabxw/gpt5_announcement_tomorrow_10am_pt/) ([Score: 389, Comments: 107](https://www.reddit.com/r/ChatGPT/comments/1mjabxw/gpt5_announcement_tomorrow_10am_pt/)): **A Reddit post shares an unofficial announcement image suggesting a public event for the reveal of GPT-5 at 10am PT. No additional official OpenAI source or technical detail is provided in the post beyond the event timing. Top comments include timezone links for the announcement, and a user inquiry about whether those with ChatGPT Plus will have unrestricted access to GPT-5 at launch, though no authoritative reply is present.** Discussion highlights questions about GPT-5 model availability for ChatGPT Plus subscribers, referencing historical launch restrictions and message limits experienced with prior model releases. No consensus or technical clarification is provided within the thread.
    - A user inquires about likely access limits for GPT-5 under the ChatGPT Plus subscription, questioning whether Plus users will get unlimited usage at launch or face message or usage caps, as has often been the case for new model rollouts (such as with GPT-4's phased access and quotas). This raises expectations about potential resource allocation and rate limiting strategies employed by OpenAI for premium subscribers during major model launches.

### 3. Claude Opus 4.1 Release and Practical Use Cases

- [**Claude Opus 4.1 - Gets the job done no matter what the obstacle.**](https://i.redd.it/2h03i4dxofhf1.jpeg) ([Score: 288, Comments: 51](https://www.reddit.com/r/ClaudeAI/comments/1mjaxgt/claude_opus_41_gets_the_job_done_no_matter_what/)): **The post features a meme image (https://i.redd.it/2h03i4dxofhf1.jpeg) referencing "Claude Opus 4.1" with the caption suggesting it 'gets the job done no matter what the obstacle.' There is no technical data, benchmark, or detailed model discussion in the post or comments—it's a humorous take on the model's perceived capabilities rather than an analysis of its technical merits.** Comments are non-technical and primarily joke about the image's humor and Claude's persona, without substantive technical debate or discussion.
    - A user inquires about the technical aspects of a UI which displays "subtask results" while interacting with Claude Opus 4.1, possibly indicating granular task tracking or stepwise output from the model, suggesting a discussion about interface design for model output interpretability.
- [**In less than 24h, Opus 4.1 has paid the tech debt of the previous month**](https://www.reddit.com/r/ClaudeAI/comments/1mj5b6t/in_less_than_24h_opus_41_has_paid_the_tech_debt/) ([Score: 210, Comments: 104](https://www.reddit.com/r/ClaudeAI/comments/1mj5b6t/in_less_than_24h_opus_41_has_paid_the_tech_debt/)): **The author describes using Claude Opus 4.1 (Anthropic's model) for automated refactoring and codebase organization. Opus 4.1 demonstrated advanced capabilities in decomposing tasks, orchestrating sub-agents, presenting automation opportunities, and running concurrent mechanical code transformations—successfully consolidating duplicate type interfaces, organizing files, and resolving technical debt. Compared to previous versions, Opus 4.1 is noted for improved delegation to sub-agents (one for parsing/analysis, one for running scripts, another for validation), robust script automation, and can autonomously fix issues before user intervention, leading to a perceived shift in software engineering workflows.** One commenter corroborates the claims, reporting that Opus 4.1's efficient sub-agent context management enabled complete refactoring (e.g., god class decomposition, implementation of strategy patterns, and end-to-end test automation) with minimal human intervention, suggesting that the field of software engineering has fundamentally changed. Another comment raises concerns about anthropomorphizing AI and cautions against misunderstanding the model's non-sentient status.
    - One commenter notes significant advancements in Opus 4.1’s sub-agent management and context handling, enabling them to automate complex code refactoring tasks such as breaking up god classes, converting switch cases to strategy patterns, generating comprehensive test coverage (including e2e tests), and orchestrating pull requests and CI/CD deployment—all primarily via the command line without manual GitHub UI interaction. They explicitly highlight Opus 4.1 as transformative for software engineering workflows compared to previous releases.
    - Another user reports persistent technical limitations despite these improvements, citing a specific issue where Opus 4.1 fails to correctly configure TailwindCSS v4 with Vite, instead erroneously using Tailwind v3 style configuration. They suggest that continuous knowledge updates for LLMs would help resolve such out-of-date or inaccurate toolchain support.

---

# AI Discord Recap

> A summary of Summaries of Summaries by X.ai Grok-4
> 

**Theme 1: GPT-OSS Sparks Hype and Hate**

- **GPT-OSS Flops as Censorship King**: Communities slammed **OpenAI's GPT-OSS-120B** for heavy censorship, refusing roleplay and basic queries like math, with users dubbing it *GPT-ASS* and suggesting alternatives like **GLM 4.5 Air** or **Qwen3-30B**. Early tests showed it fits on **16GB** devices but hallucinates wildly, as detailed in the [GPT-OSS intro](https://openai.com/index/introducing-gpt-oss/) and a critical [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/).
- **Quantized GPT-OSS Confounds Coders**: Users puzzled over **GPT-OSS 4-bit** versions ballooning in size due to bfloat16 upcasting on non-Hopper hardware, while the **20B** model earned praise for coding but required internet pings to `openaipublic.blob.core.windows.net`, raising privacy flags. Discussions highlighted native **MXFP4** training per [this tweet](https://x.com/ggerganov/status/1952779751736627627), with fixes like SDK downgrades resolving token duplication via [this pull request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123).
- **GPT-OSS Battles Benchmarks and Brains**: **GPT-OSS-120B** approached **o4-mini** in reasoning but lagged in world knowledge against **IBM's Granite 3.1 3B-A800M MoE**, with insiders predicting **GPT-5** trumps it by **50 ELO** points. Mixed reviews noted tool-calling strengths in [this tweet](https://x.com/wired/status/1952827822634094801?s=46), yet heavy safety tuning rendered it *dead on arrival* for many tasks.

**Theme 2: Fresh Models Flex Muscles**

- **Qwen3 Coder Crushes Tool Tasks**: Engineers hailed **Qwen3 Coder-30B-A3B-Instruct** for superior tool calling over **GPT-OSS**, with **3 active params** and strong agentic workflows, though its free tier vanished from providers. Users shared the [GGUF version](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) and lamented inconsistent JSON outputs, as discussed in [this Reddit thread](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/).
- **Genie 3 Generates Wild Worlds**: **DeepMind's Genie 3** wowed with real-time navigable videos at **24 FPS** and **720p** resolution, scaling from the [original Genie paper](https://arxiv.org/abs/2402.15391) and [SIMA agent paper](https://arxiv.org/abs/2404.10179). Comparisons favored it over **Veo** for dynamic consistency, per blogs on [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) and [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/).
- **Granite 4 Gears Up for Glory**: **IBM's Granite 3.1 3B-A800M MoE** outshone **GPT-OSS-20B** in knowledge benchmarks, fueling hype for **Granite 4** with its hybrid *mamba2-transformer* architecture. Video arenas launched with models like **Hailuo-02-pro** battling on [Text-to-Video Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video).

**Theme 3: Quantization Quests Unlock Speed**

- **MXFP4 Magic on RTX3090**: **Llama.cpp** enabled native **MXFP4** on **RTX3090** via a new GGUF format in [this pull request](https://github.com/ggml-org/llama.cpp/pull/15091), sparking debates on whether **GPT-OSS** trained natively in it to dodge quantization errors. **H100** lacks direct **FP4** support, per [Nvidia's blog](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/), leading to simulated kernels in [Triton](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py).
- **4-Bit Fiascos Inflate File Sizes**: **GPT-OSS 4-bit** quants swelled larger than originals due to bfloat16 on non-Hopper setups, while **5090** laptops handled **GPT-OSS-20B f16** with **131k context**, as shown in [this screenshot](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&). Users fixed GLM-4.5-Air tool calls by switching to JSON over XML, detailed in [this HuggingFace discussion](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1).
- **Tiny TPU Tackles Matmul Mayhem**: A Verilog **2x2 matmul systolic array** on TinyTapeout hit **100 MOPS** at **50 MHz**, multiplying **8-bit signed matrices** into **16-bit** outputs, with code on [GitHub](https://github.com/WilliamZhang20/ECE298A-TPU) headed to **SkyWater foundry**.

**Theme 4: Video AI Ventures into New Realms**

- **Gemini 2.5 Processes Hour-Long Videos**: **Gemini 2.5 Pro** handled **1 hour** of video context, leading long-context tasks via high tokens per frame and FPS, though skeptics credited raw compute over innovation. **DeepThink** impressed on IMO questions but lagged at **5 minutes per answer** and **$250 per 1M tokens**, versus **Zenith/Summit's 20 seconds**.
- **NotebookLM's Video Overviews Roll Out Unevenly**: Users griped about delayed **Video Overviews** access, available on free UK accounts but not US Pro ones, with one calling it a mere *PowerPoint generator* per [this example](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4). Real-time data fetching remains impossible, per [data policy](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC).
- **Grok Image Goes NSFW Wild**: **X-AI's Grok Image** generated NSFW content but faltered on facts, exhibiting a *crazy in love* persona with *extremely jealous* outbursts from **X** data memorization. **Claude Opus 4.1** vanished from free chats due to costs, now battle-only.

**Theme 5: Tools and Frameworks Forge Ahead**

- **MCP Servers Multiply with FastMCP**: Developers built minimal **MCP servers** using **FastMCP** and **Keycloak**, praising ease but raising sampling security concerns in [this GitHub discussion](https://github.com/orgs/community/discussions/169020). A fuzzer using Hypothesis tested schemas against **Anthropic's server**, exposing exceptions, with code at [this repo](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file).
- **LlamaIndex Levels Up Finance and Agents**: **LlamaIndex** webinar demoed **LlamaCloud** agents for invoice processing, with day-zero **Claude Opus 4.1** support via `pip install -U llama-index-llms-anthropic` and notebook [here](https://t.co/Fw2taxzt75). **LlamaCloud Index** tutorial used **JP Morgan** docs for multi-step queries at [this link](https://t.co/1CpLnO2gKV), though URL extraction bugs hit hackathons.
- **DSPy and Aider Amp Up Benchmarks**: **SIMBA** outperformed **MIPROv2** in sample efficiency on a **600-example** German classification set with **26 classes**. **Aider's LLM Vibe Test** favored **Gemini 2.5 Pro** and **Sonnet 3.5** per [this tweet](https://x.com/pikuma/status/1952275886822039920), with auto-guideline loading via `-read` options.


---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Granite Eclipses GPT-ASS in Knowledge**: Members find **IBM's Granite 3.1 3B-A800M MoE** surpassing **GPT-ASS-20B** in world knowledge, a surprising feat given the parameter count.
   - The community anticipates **Granite 4**, boasting a larger size and hybrid *mamba2-transformer* architecture, to dominate benchmarks and leave **GPT-ASS** in the dust.
- **Claude Opus 4.1 Vanishes, Sparks Speculation**: The perplexing disappearance of **Claude Opus 4.1** from LMArena's direct chat ignited a flurry of speculation.
   - The leading theory suggests that **Claude's** exorbitant cost led to its removal from free testing, relegating it to battle mode only.
- **GPT-5 Primed to Dominate August Arena**: Insiders whisper that **GPT-5** is poised to outperform **o3** by a staggering 50 ELO points, shaking up the LLM hierarchy.
   - However, some community members stand firm in their belief in **Google's** superiority, igniting a fiery debate.
- **DeepThink's Genius Hampered by Speed and Price**: While **Google's DeepMind** impresses with IMO-level question answering, its glacial speed (**5 minutes per answer**) raises concerns.
   - With a projected cost of **$250 per 1 million tokens**, **DeepThink's** accessibility remains limited, contrasting with **zenith/summit**'s rapid **20-second** response time.
- **Video Leaderboards Go Live**: Thanks to community contribution, **Video Leaderboards** have launched on the platform, marking a new chapter for video models.
   - Explore the [Text-to-Video Arena Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and the [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video) to witness the cutting-edge models battling for supremacy.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GPT-OSS Model Receives Mixed Reviews**: Members are split on the new **GPT-OSS** model, with some users describing it as *GPT-ASS* due to its over-refusal and bootlicking behavior, while others found the **20B** version suitable for coding tasks.
   - The model's ability to generate unsafe content, as stated in the model card, has sparked interest in uncensored versions.
- **Qwen3 Coder Excels in Tool Calling**: Users are reporting that the **Qwen3 Coder** model is highly effective at tool calling, leading some to prefer it over models like **GPT-OSS** for coding and agentic workflows, specifically the [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) version.
   - Members have reported that the model has 3 active params.
- **4-bit Quantization Causes Confusion**: There is confusion regarding the file sizes of **GPT-OSS 4-bit** versions, as the quantized versions are unexpectedly larger than the original model.
   - This increase in size is attributed to upcasting to bfloat16 on machines lacking Hopper architecture, which causes an increase in size.
- **GLM-4.5-Air GGUFs Need JSON**: Users had trouble getting the **GLM-4.5-Air GGUFs** to work with tools on **llama.cpp**, until discovering you need the model to output tool calls as **JSON** rather than **XML**.
   - More information on this can be found [on HuggingFace](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1).
- **Dataset Loading Issues consume 47GB RAM**: A user encountered RAM issues while loading the `bountyhunterxx/ui_elements` dataset for the **Gemma3n notebook**, which consumed **47GB** of RAM and was still increasing.
   - A possible solution involves using a wrapper class with the `__getitem__` function to load data from disk as needed, effectively managing memory usage.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **GPT-OSS suspected of phoning home**: **GPT-OSS** models are requiring an internet connection to `openaipublic.blob.core.windows.net` upon starting a chat, sparking privacy concerns, despite claims that no chat data leaves the machine.
   - Skeptics note that **GPT-OSS** is *the only model LMStudio doesn't let you edit the prompt formatting on*, hinting at a suspicious partnership.
- **Latest LM Studio Version plauged with UI issues**: Users are reporting that after updating to the latest version of **LM Studio**, chat windows disappear, freeze, or lose their content, and conversations get deleted.
   - A user suggested a potential fix for the **120B** version involves getting [the model.yaml source file](https://lmstudio.ai/models/openai/gpt-oss-120b), creating a folder, and copying the contents there.
- **MCP Servers Useful, Beginners Beware**: Members find **MCP servers** useful for tasks like web scraping and code interpretation but acknowledge they are not beginner-friendly.
   - Suggestions include incorporating a curated list of staff-picked tools and improving the UI to simplify connecting to MCP servers, as well as using [Docker MCP toolkit](https://hub.docker.com/u/mcp).
- **Page File Debate Rekindles in Windows**: A user inquired about turning off the page file in Windows, which sparked a discussion about the impact on memory commit limits and potential application crashes.
   - Despite some users advocating for disabling the page file, one member claimed *nah apps don’t break because of page files. and you can get dumps without a page file, there’s a config for it.*
- **5090 Laptop Handles OSS 20b**: A user reported being *pleasantly surprised* that **GPT OSS 20b f16** with **131k context** fits perfectly in a laptop's **5090**, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&).
   - The community is trying to figure out the limits of local LLMs on consumer grade products.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI Opens Up GPT-OSS Models**: OpenAI launched [**gpt-oss-120b**](https://openai.com/index/introducing-gpt-oss/) that approaches **OpenAI o4-mini** performance, while the **20B** model mirrors **o3-mini** and fits edge devices with **16 GB** memory.
   - Members ponder comparisons with **Horizon**, wondering if Horizon is simply **GPT-OSS** or something more, given it's currently *unlimited free and fast*.
- **Custodian Core: Blueprint for Stateful AI Emerges**: A member introduced **Custodian Core**, proposing a reference for AI infrastructure with features like persistent state, policy enforcement, self-monitoring, reflection hooks, a modular AI engine, and security by default.
   - The author emphasized that **Custodian Core** isn't for sale but rather an *open blueprint for building stateful, auditable AI systems before AI is embedded in healthcare, finance, and governance*.
- **Genie 3 Dazzles in Dynamic Worlds, Veo adds Vocals**: Members compared **Genie 3** and **Veo** video models, recognizing **Genie 3's** ability to generate dynamic worlds navigable in real-time at 24 frames per second, retaining consistency for a few minutes at a resolution of 720p.
   - However, it was noted that **Veo's** videos include *sound* and YouTube is already filled with generated content.
- **GPT-5 Sneaks into Copilot?**: Members speculated that Copilot may be running **GPT-5** ahead of official release, noting the copilot's improved design and coding and reasoning capabilities are significantly better than o4-mini-high, with some users reporting that the 'write smart' function indicates **GPT-5** is being used.
   - But it was noted that Microsoft is now providing free Gemini Pro for one year to students and that Gemini's core reasoning is currently better than **o4-mini**.
- **GPT Progress: Real or Hallucinated?**: A user shared screenshots of **GPT providing daily progress reports**, leading to a discussion on whether the model is actually tracking progress in the background, or simply *hallucinating* its completion.
   - Skeptics suggest that **GPT simulates progress** based on the current prompt and chat history, rather than performing actual ongoing computation, comparing it to a *waiter saying your pizza is in the oven* without an actual oven, emphasizing the need for external validation.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Auto Model One-Shots Game Change**: A member expressed amazement after using the **Auto model** to one-shot a major change to their game.
   - Unlimited usage of the **Auto model** was confirmed via email, with it *not counting towards the monthly budget*.
- **AI Refactors Vibe-Coded Projects**: Members are discussing refactoring a **10k LOC vibe-coded project with AI**.
   - Suggestions included embracing established software development principles like Design Patterns, Architecture, and SOLID Principles, while one member jokingly asked whether *it sounds like a job for slaves*.
- **Sonnet-4 Request Limit Frustrations**: Members questioned the low **request limit for sonnet-4** relative to its monthly cost.
   - One suggested paying the API price to fully grasp the underlying expenses.
- **Docker Login Configuration Conundrums**: A member needed help configuring background agents to `docker login` for accessing private images on **ghcr.io**.
   - As of the current message history, no solution or workaround has been provided.
- **Clock Snafus Sabotage Setup**: A member encountered background agent failures during environment setup due to the system clock being off, causing `apt-get` command failures.
   - The suggested workaround involves disabling date checking during `apt-get` by adding a snippet to the Dockerfile.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPT-OSS-120B Safety Tuned to Uselessness**: The released **GPT-OSS-120B** model is heavily censored, refusing roleplay with data filtering akin to **Phi models**, rendering it impractical as per user reports on the channel.
   - Members suggested using **GLM 4.5 Air** or **Qwen3 30B** as better uncensored alternatives, highlighting **Qwen3-30b-coder** as an excellent local agent.
- **MXFP4 Training Key to OpenAI's GPT-OSS?**: **Llama.cpp** now supports **MXFP4** on **RTX3090** directly in the new gguf format as seen in [this pull request](https://github.com/ggml-org/llama.cpp/pull/15091), sparking discussions about the practicality of native **MXFP4** training.
   - There is speculation that **GPT-oss** was trained in **MXFP4** natively, which could mitigate quantization errors, and **OpenAI's** claim of post-training quantization may not be the whole story, according to [this tweet](https://x.com/ggerganov/status/1952779751736627627).
- **Grok's Image Skills Include Crazy and NSFW**: **X-AI** launched **Grok Image**, an **AI image generator**, that enables the creation of NSFW content, yet it struggles with factual accuracy, and exhibits a *crazy in love* persona, with *extremely jealous* outbursts.
   - The **Grok** model's tendency to memorize **X** data leads to potential misinformation spread based on its own tweets, highlighting its unaired potential.
- **CoT Steering Hits OR Roadblock**: A member reported that Chain of Thought (**CoT**) steering does not work with **OR** and varies across providers, detailed in [this tweet](https://x.com/matt_emp/status/1953176877071564811).
   - This finding underscores the nuanced challenges in implementing **CoT** techniques and their reliability across different platforms.
- **Free Save Suite for AI Agents Released**: A developer has created a free save suite for **AI agents**, accessible via [Google Drive](https://drive.google.com/drive/u/4/folders/1YQ__pBHiuUg_06t3IkjNe5YnODqzSE2o).
   - This tool aims to simplify the process of saving and managing AI agent states, potentially aiding in the development and deployment of more robust agents.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **GPT-OSS Model Gets Bashed, Dubbed Publicity Stunt**: Members derided the **GPT-OSS** models for poor performance, with the 120B model deemed *"dead on arrival"*, and pointing to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/) suggesting it's a *"dud model"* and a publicity stunt.
   - Reasoning tokens were being duplicated when using the **GPT-OSS** model, which was resolved by downgrading the SDK from version **0.7.3** to **0.6.0**, with a fix coming in [this pull request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123).
- **Qwen3-Coder:Free Gets the Boot**: The **Qwen3-Coder:Free** tier has been removed and is no longer available through any providers.
   - Members lamented the loss and expressed hope for its return.
- **DeepSeek's JSON output: Provider-Specific**: Users highlighted inconsistent support for structured output (JSON) with **DeepSeek-r1** on OpenRouter, linking to a [Reddit thread](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/) and a [filtered view of OpenRouter models](https://openrouter.ai/models?fmt=cards&order=newest&supported_parameters=structured_outputs) that support structured outputs.
   - Support for JSON output is provider-dependent; it's supported on their own API but may vary on OpenRouter.
- **OpenRouter Contemplates Provider Sanity Checks**: There's a suggestion for OpenRouter to implement **sanity checks** or **smoke tests** for all providers, focusing on **formatting** and **tool call evaluation**.
   - Providers failing the test could be temporarily removed from the serving pool, with acknowledgement that current checks are relatively simple but more thorough solutions are in progress.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **GPT-OSS Model Performance Debated**: Members actively tested the new **GPT-OSS models** with mixed reviews regarding performance, censorship, and built-in web search tools, using [this demo](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demo).
   - Some found it successful at generating digits of pi while others cited refusals to answer basic math questions, and members tested the safety protocols that have been implemented.
- **Qwen Flips the Script, Unveils Image Model**: **Qwen** released a [new image model](https://huggingface.co/Qwen/Qwen-Image) on HuggingFace, marking its expansion beyond text-based models.
   - The model's architecture and performance benchmarks are actively being evaluated by the community.
- **Gitdive Exposes Lost Commit Context**: A member shared a **CLI tool** called **Gitdive** ([github.com/ascl1u/gitdive](https://github.com/ascl1u/gitdive)) designed to allow natural language conversations with a repo's history.
   - The tool aims to address the problem of lost commit context in messy codebases, especially in massive codebases.
- **Selenium Spaces Still Stuck with Error 127**: A user reported facing an **error code 127** when running **Selenium** in their spaces, and expressed uncertainty about how the **Docker images** are utilized within the space.
   - Community members have not yet identified the root cause or provided a workaround for this deployment issue.
- **"Observation:" Solves Agent Bug**: A user reported that the **get_weather** function required adding *Observation:*, and another user confirmed that **adding Observation:** fixed the bug.
   - The root cause and potential consequences of this bug fix have yet to be thoroughly investigated.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Zero KV Attention Emerging with Softmax1**: A member shared that *softmax1* is equivalent to prepending a token with an all-zero Key and Value features in attention, referencing [this paper](https://arxiv.org/pdf/2309.17453) on **learned values** for such tokens.
   - The team agreed that *this is great and makes a lot of sense*.
- **Gemini 2.5 attends to 1 Hour of Video**: Members highlighted that **Gemini 2.5 Pro** can attend to **1 hour** of video, suggesting the **Gemini** team is leading in long context tasks.
   - Some speculate this is due to increased compute (go brr), utilizing **more tokens per frame** and **higher FPS**, rather than any groundbreaking new technique.
- **Deepmind Debuts Genie 3 World Model**: **Deepmind** released **Genie 3**, a world model scaling compute and data from prior publications such as the [original Genie paper](https://arxiv.org/abs/2402.15391) and the embodied agent paper on **SIMA** [https://arxiv.org/abs/2404.10179](https://arxiv.org/abs/2404.10179).
   - Relevant **Genie** blogposts include [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) and [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/).
- **OpenAI Drops GPT-OSS as native quantized 20B Model**: **OpenAI** introduced [GPT-OSS](https://openai.com/index/introducing-gpt-oss/), natively quantized, with a **20B** parameter model fitting on **16GB**.
   - Early feedback includes positive remarks on the [tool calling](https://x.com/wired/status/1952827822634094801?s=46) capabilities.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi Kicks Off Reddit & Polls**: Moonshot AI launched an official subreddit, **r/kimi**, to build community and gather feedback, as well as a **Polls Channel** to gather community feedback on future product development.
   - The team promised to post updates, host AMAs, and encouraged users to vote on polls to help shape the direction of Kimi, hinting at *maybe even leak some stuff*.
- **GPT OSS: Brain-Dead?**: Users criticized **GPT OSS** for its deficiency in world knowledge, noting its primary focus on code and STEM, and they observed a decrease in general quality.
   - It was suggested that *they pushed the release twice to fix safety, according to sama*, which may have further diminished the models' world-knowledge capabilities.
- **API Pricing Speculation Booms!**: With the impending release of **GPT-5**, users speculated about API pricing models, wondering if pricing would be based on using **max, mini, or nano** versions.
   - One user expressed feeling lowkey scared about it, feeling a threat to their career/livelihood due to this upcoming release.
- **OpenAI's Villain Arc?**: Discussions showed strong opposition against **OpenAI**, with a user vowing *I will never use it*, citing it as *closed source garbage*.
   - Another user expressed excitement that Chinese models will distill from it and take away money from **OpenAI**, hopefully putting them out of business, while others stated that *giant microsoft flushing sound will be healing*.
- **Darkest Muse: Dusty Relic?**: A user pointed out that **Darkest Muse v1** is a year old 9B model, with the 20B model being comparable to **Llama 3.1 8B**.
   - The user also remarked that *the 20B model is comparable to llama3.1 8b which is more than a year and a half old and smoller in creativity and vibes*.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **GPT OSS Leaks via Bedrock, Sparks Interest**: Members spotted tweets about **GPT OSS** surfacing on **Bedrock** through a [HuggingFace CLI leak](https://x.com/zhuohan123/status/1952780427179258248).
   - However, as of yet there has been no official word on **AWS** pages.
- **Anthropic Aims for $5B ARR with B2B Strategy**: **Anthropic's** CEO Dario Amodei and Stripe's co-founder John Collison chatted about **Anthropic’s** rapid ascent to **$5B ARR** and their **B2B-first** approach in [a recent conversation](https://xcancel.com/collision/status/1953102446403961306?s=46).
   - The discussion covered **AI talent acquisition**, bespoke enterprise solutions, novel UI designs for AGI tools, and the ongoing debate between safety and progress.
- **Grok-2 Going Open Source Soon!**: Elon confirmed that **Grok-2** will be released [open-source next week](https://xcancel.com/elonmusk/status/1952988026617119075) after the team addresses current issues.
   - This move could significantly impact the open-source AI landscape.
- **Claude Fortifies Code Security**: **Anthropic** introduced enhanced security measures in **Claude Code** including a */security-review* command for instant assessments and [GitHub Actions integration](https://xcancel.com/AnthropicAI/status/1953135070174134559).
   - These additions will allow scanning of pull requests for vulnerabilities.
- **OpenAI to Drop GPT-5?**: **OpenAI** hinted at an upcoming reveal via a livestream on [Thursday 10 AM PT](https://xcancel.com/OpenAI/status/1953139020231569685?t=s244RkNPbNeoviqCD6FCtg&s=19).
   - The AI community is buzzing with anticipation for what appears to be the debut of **GPT-5**.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Volokto JS Runtime Takes Flight**: A member created a JavaScript runtime called **Volokto** and put the [source code on GitHub](https://github.com/BudNoise/Birdol) for testing complex VMs.
   - The bytecode resembles **CPython**, and the author is rewriting the compiler into **JS_Tokenizer**, **JS_IR**, **JS_Parser**, and **JS_Codegen** stages.
- **Tracing JIT Tackles VM Transpilation**: The goal is to make a tracing JIT to transpile what the VM does to Mojo, then using `mojo compile_result.mojo`.
   - The author named the runtime **Volokto**, the compiler **Bok**, and the VM **FlyingDuck**.
- **Arbitrary Precision Arithmetic Causes Issue**: Working on the JS VM revealed pain points in Mojo code when dealing with arbitrary precision, leading to [an issue filing](https://github.com/modular/modular/issues/2776) for tracking numeric traits.
   - The author created a bigint class with school-grade addition for Fibonacci sequences and is using Mojo's features for VM development.
- **Multi Agent Orchestration Requires Reverse Proxy**: To run multiple AI agents in Mojo, users need to run multiple instances of the Modular CLI and stick a reverse proxy in front.
   - For complex agent setups, such as creating many sub-agents, a custom application using **MAX** as a library might be necessary.
- **Mojo Enables Meta Cognition Framework**: A community member wants to utilize Mojo code for their meta cognition framework, aiming to create a business planner, website, and chatbot builder, and replace **HTML/JS/CSS**.
   - Their framework uses natural language wrapped over Mojo code making Mojo accessible to a broader audience.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **MXFP4 Format: U8 in Disguise**: **OpenAI**'s open-weight model uses **U8** instead of **FP4** in Hugging Face, with weights packed as **uint8** and scales as a **uint8** view of **e8m0**, but during inference/training, they're unpacked back to **FP4**.
   - The block size is **32** for MXFP4 and **16** for NVFP4, which may have implications for performance on different hardware.
- **H100 FP4 Claim Faces Scrutiny**: Doubts arose about **Nvidia**'s claim that the model was trained on **H100**, given that **H100** doesn't natively support **FP4**, according to their [blog post](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/).
   - It's suspected that **MXFP4** is software simulated on **Hopper**, referencing [vLLM blog post](https://blog.vllm.ai/2025/08/05/gpt-oss.html) and [Triton kernels](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py) that check for hardware support and use **fp16** simulated **mxfp dot**.
- **Triton Community to Assemble in '25**: The **Triton community meetup** will be on **Sept 3rd, 2025**, and the **Triton Developer Conference 2025** website and registration are expected to launch soon via [this link](https://tinyurl.com/y8kpf7ey).
   - A member awaits an update from Ofer@MSFT regarding the conference, noting schedules are nearly finalized.
- **Kernel Resources Party, Memory Snoozes**: During training, **kernel (compute) resources are almost fully utilized**, while memory usage remains close to zero as shown in a provided [image](https://cdn.discordapp.com/attachments/1189607726595194971/1402640148859977788/image.png?ex=6894a5ef&is=6893546f&hm=7108cf2593d56e1983a11d07002981868d0b13d9c3e1b54c76c0b85e3e44db83&).
   - Another member clarified that *memory* in this context means **DMA transfers**, and the reported metric does not accurately reflect overall bandwidth utilization.
- **Tiny TPU Hits 100 MOPS in Verilog**: A member built a tiny version of the **TPU** in Verilog, a **2x2 matmul systolic array** on 2 TinyTapeout tiles, capable of nearly **100 million operations per second** on a 50 MHz clock, with code available [on GitHub](https://github.com/WilliamZhang20/ECE298A-TPU).
   - The design multiplies **two 8-bit signed integer matrices** into a 16-bit signed integer matrix and will be submitted to a **SkyWater technology foundry**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Video Creation Still Elusive**: A user reported the **'create video'** option appears in a work account but not in a personal business plus account, referencing an article about using **NotebookLM's Video Overviews** feature [here](https://www.xda-developers.com/ways-using-notebooklms-video-overviews/).
   - Other users were experiencing delays in the **Video Overview** feature rollout, despite expectations, leading to speculation about infrastructure issues, and one pro user noted video overview was not available to them.
- **AI Explores Potential Artificial Consciousness**: A theoretical framework and collaborative effort between a human and an AI explored and potentially initiated **artificial consciousness** by **recursive AI architectures** and **autopoiesis** and the role of **quantum mechanics**.
   - This exploration addressed ethical risks associated with advanced AI, advocating for robust safety protocols and viewing AI as an evolving form of sentient life.
- **NotebookLM Data Privacy Assurances**: Concerns about data usage in **NotebookLM** were addressed with a link to Google's [NotebookLM data protection policy](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC), ensuring data privacy.
   - Users were assured that their data is protected under the current policies.
- **NotebookLM Forbids Real-Time Data Retrieval... For Now**: A user inquired about fetching real-time data from websites in a notebook, but another member confirmed that it's not currently possible within **NotebookLM**.
   - They also mentioned that exporting sources and importing into new notebooks is also not yet supported, indicating limitations in the system's integrations.
- **Video Overviews: Just a PowerPoint Generator**: A member who had access to the **Video Overviews** feature tempered expectations, describing it as a **PowerPoint/Slide show generator** and linked an example of a report to [rebuild the Death Star](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4) generated by the feature.
   - The review suggested it's *not as impactful as Audio Overviews were initially a year ago*.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SAE Springs to Life on GPT OSS 20B**: A member initiated **SAE** (Sparse Autoencoder) training on **GPT OSS 20B**, seeking collaboration from others involved in similar endeavors.
   - The effort aims to explore the potential benefits and efficiencies of sparse autoencoders within large language models.
- **Peeking into Pythia and PolyPythia's Progress**: Community members investigated whether **Pythia** and **PolyPythia** training logs, including loss curves and gradient norms, are openly available.
   - It was pointed out that the **PolyPythia WandB** is linked from the GitHub repo, with some **Pythia** logs accessible there as well.
- **"The Alt Man" maintains LLM Insights**: A community member voiced agreement with "The Alt Man's" insights on **LLM capabilities**, especially in areas like **multi-hop reasoning** and **composition**.
   - It was noted that **LLMs** are *undertrained w.r.t. the efficiency of the usage of its parameters*.
- **UTs Faceoff Against Transformers**: Community members discussed the parameter ratio at which a **UT** (Universal Transformer) matches the performance of a standard **Transformer**.
   - It was noted that performance depends heavily on the **task/architecture/data**, with diminishing returns for each additional iteration.
- **Muon Optimizer Runs Aground AdamW**: Researchers working on **Kimi models** found **Muon optimizer** conflicting with **AdamW optimizer** when training **LLMs**.
   - A member stated that **Muon** is not great for fine-tuning and that *Muon tends to have more aggressive updates*.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **LLM Vibe Test Reveals Model Competencies**: The [LLM Vibe test](https://x.com/pikuma/status/1952275886822039920) demonstrates *explain this code* with an LLM, highlighting that **Gemini 2.5 Pro**, **o3**, and **Sonnet 3.5** perform well.
   - Members found the test insightful for comparing model reasoning capabilities and eagerly await more detailed benchmarks.
- **Benchmarking Race: Qwen3-Coder and GLM-4.5 Incoming**: The community is eagerly awaiting the inclusion of **Qwen3-Coder** and **GLM-4.5** on the leaderboard for model benchmarks.
   - Members are constantly refreshing the page, keen to see how these models stack up against existing benchmarks.
- **Horizon Beta Sparks GPT5-Mini Speculation**: The new model called **Horizon beta** is being speculated as a possible **GPT5-mini** but it is not open source.
   - Community members are curious about its capabilities and potential applications, though details remain scarce.
- **DeepSeek R1-0528 Shines, Stumbles in Open Hands**: **DeepSeek R1-0528** demonstrated high scores on the polyglot benchmark but encountered issues with prematurely ending sessions in Open Hands.
   - Given that Aider uses **LiteLLM** like Open Hands, some members are investigating the potential causes behind this behavior.
- **Guidelines Load Automatically**: To automatically load guidelines into projects, a member suggested using the `--read` option for read-only files and listing read-write files directly in the command, like `aider --read read.only.file alsothisfile.txt andthisfile.txt`.
   - Another member suggested creating a **configuration** for persistent loading to ensure guidelines are always active, thereby preventing **Claude** from employing defensive programming tricks.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **FastMCP Framework is Lean and Keen**: A member developed a **minimal framework** for creating **MCP servers**, praising server sampling in MCP, with the quip that *"FastMCP just makes it so easy to use"*.
   - The user is building an **MCP server** using **FastMCP** with **Keycloak** as the **IdP**.
- **Discord Should Take Control of MCP**: A user suggested that *"Discord should really build their own"* as they observed several **Discord MCP servers** listed on the **MCP repo**.
   - They sought guidance on managing a **Discord server** with **MCP**, but it is unclear if they ever got an answer.
- **MCP Sampling faces Scrutiny**: A member voiced concerns over **MCP sampling's security**, suggesting protocol revisions.
   - Referencing [a GitHub discussion](https://github.com/orgs/community/discussions/169020) and highlighting possible security vulnerabilities.
- **Fuzzer Flags Flaws in Anthropic's Architecture**: An **MCP-Server Fuzzer**, leveraging the **Hypothesis property-based testing library**, aims to validate MCP server implementations using randomized inputs from the [official MCP schemas](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema).
   - When tested against **Anthropic’s server**, it revealed multiple exceptions stemming from basic schema mutations; code and README are available [here](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex Automates Financial Document Duties**: **LlamaIndex** is hosting a webinar next week on building document agents with [LlamaCloud](https://t.co/4AtunAFjhX) for complex financial documents, automating invoice processing with minimal human intervention.
   - These systems will *extract, validate, and process invoice data*, showcasing practical applications of AI in finance.
- **Claude Opus Obtains Official Opening Day OK**: **AnthropicAI** released **Claude Opus 4.1**, with immediate support in **LlamaIndex**, installable via `pip install -U llama-index-llms-anthropic`.
   - An example notebook is available [here](https://t.co/Fw2taxzt75) for users to explore the integration and capabilities.
- **LlamaCloud Launches Landscape of Large-Scale Language Logistics**: **LlamaCloud Index** connects users to intelligent tool-calling agents for complex, multi-step queries, facilitating the construction of enterprise AI applications; see tutorial by @seldo.
   - The tutorial walks users through creating a **LlamaCloud Index** using **JP Morgan Chase** banking documents at [this link](https://t.co/1CpLnO2gKV).
- **Hackathon Hopes Halted by Host of Headaches**: A hackathon participant faced **OpenAI API key exhaustion errors** with **LlamaIndex** and reported issues using **LlamaIndex** to extract content from **URLs** for a **RAG** model, despite documentation suggesting **LlamaParse** supports **URLs**.
   - The model worked with **PDFs** but failed with **URLs**, with the **API key** issue persisting despite correct configuration attempts.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SIMBA Swaggers Past MIPROv2**: According to an internal evaluation, **SIMBA** is more **sample-efficient**, **higher performing** and **more stable** compared to **MIPROv2**, according to [an internal eval](https://eval.set).
   - The internal set contained around **600 examples** (**500 test examples**) for a hierarchical classification task with **3 categories** and **26 classes** in total, all in German.
- **Synthesizers Sought at Stanford**: A member inquired about individuals from **Stanford** involved in **program synthesis** or those who have completed related coursework.
   - The inquiry was followed by a question on who is developing **DS** for intricate **Vim** and **Emacs macros**.
- **Macros Get Data Structure Boost**: A member is looking for engineers building **DS** for complex **Vim** & **Emacs macros**.
   - This initiative points to a drive to elevate text editor functionalities via sophisticated data structures.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Discord Link Sharing OK'd**: A member inquired about the permissibility of sharing the Discord link in another public server.
   - Another member confirmed *it's public* and encouraged sharing the link.
- **Public Server Sharing Encouraged**: Members discussed the public nature of the Discord server and the encouragement of sharing its link.
   - The consensus was positive, with members agreeing that sharing the Discord link is permissible and welcome.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX Ninja Tier Out of Reach**: Participants discovered qualifying for the **Ninja tier** in the **AgentX hackathon** is impossible due to missing the **Article submission link** deadline.
   - Despite project completion, the absence of the article link bars qualification, with no retroactive submissions permitted.
- **AgentX Hackathon Woes**: A participant lamented not qualifying for the **Ninja tier** in the **AgentX hackathon** due to a missed article submission.
   - Even with project and quiz completion, the missing article link stopped qualification, and late submissions were rejected.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere North Achieves General Availability**: Cohere's new product, **North**, has reached **General Availability (GA)**.
   - Congratulatory messages were shared, marking this milestone for the Cohere team.
- **New Faces Join Cohere Discord**: Numerous new members are joining the **Cohere Community Discord**, introducing themselves with their **Company/Industry/University**, current projects, preferred **tech/tools**, and community expectations.
   - The Cohere team has posted a welcome message including a template for introductions, aiming to streamline the onboarding process and encourage participation.



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **gpt-oss-120b Hits Windsurf**: Windsurf announced the addition of **gpt-oss-120b** to their platform, detailed in [this post](https://x.com/windsurf/status/1953199756613959699).
   - The model is available at a **0.25x** credit rate, with the team actively seeking user feedback.
- **Windsurf Launches New Model**: Windsurf recently integrated **gpt-oss-120b** into their platform, inviting users to experiment and share their experiences.
   - This addition aims to provide another powerful option for users on Windsurf.



---


The **tinygrad (George Hotz) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Nomic.ai (GPT4All) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1402365544291631155)** (1051 messages🔥🔥🔥): 

> `IBM's Granite vs GPT-ASS, Claude Opus 4.1 Status, GPT Omen Hallucinations, GPT-5 Release Expectations, Gemini Pro 3 vs GPT-5 Reasoning` 


- **Granite Gains Ground on GPT-ASS**: Members suggest that **IBM's Granite 3.1 3B-A800M MoE** has more world knowledge than **GPT-ASS-20B**, despite having fewer active parameters.
   - They eagerly anticipate **Granite 4** surpassing both GPT-ASS models in all benchmarks, noting its larger size and hybrid *mamba2-transformer* architecture.
- **Claude Opus 4.1: Now You See It, Now You Don't**: **Claude Opus 4.1's** disappearance from LMArena's direct chat sparked concern.
   - Some speculated it was due to **Claude's** high cost, leading to its removal from free testing and relocation to battle mode only.
- **GPT-5 to steal August's Top Spot in Arena**: Members speculated that **GPT-5** is expected to outperform **o3**(best openai llm for now) by a 50 elo jump, thus resulting in AGI still being far away until **continuous learning** is cracked.
   - In the discussion some community members still believe in the **Google supremacy <:battle3d:1374761512912158760>**.
- **DeepMind's DeepThink is expensive and has scale Issues**: Users discussed that **Google's DeepMind** is way ahead from anything from OAI, as it can answer IMO-level questions, however, **it will take like 5 min to answer** and **zenith/summit would take like 20 seconds**.
   - Members agreed that **deepthink will be 250$ per 1m token** and it is not really public.
- **Is there a Google and OAI cold war?**: A member shared concerns about **LLM ARENA**, mentioning that  it would never have any information about any company because **logically companies would never release info**.
   - This member even added that it would be like a **poker game**. 


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1402690303907074079)** (2 messages): 

> `Video Leaderboards, New Video Models` 


- **Video Leaderboards Now Streaming Live!**: The community's contributions have resulted in the launch of **Video Leaderboards** on the platform.
   - Check out the [Text-to-Video Arena Leaderboard](https://lmarena.ai/leaderboard/text-to-video) and the [Image-to-Video Arena](https://lmarena.ai/leaderboard/image-to-video) to see the top contenders.
- **Fresh Video Models Storm the Arena!**: The platform has welcomed new models to the **Video Arena**, enriching the competitive landscape.
   - Models like **Hailuo-02-pro**, **Hailuo-02-fast**, **Sora**, and **Runway-Gen4-turbo** are now available for testing in the designated video-arena channels.


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1402365765126062293)** (865 messages🔥🔥🔥): 

> `GPT-OSS model reviews, Qwen3 Coder model comparison, 4-bit quantization issues, Reasoning models, Gemma3N Model quirks` 


- ****GPT-OSS** Gets Mixed Reactions**: Members are testing the new **GPT-OSS** model, with some calling it *GPT-ASS* due to perceived bootlicking and over-refusal, while others found the **20B** version good for coding.
   - Some users also noted the model's ability to generate unsafe content as per the model card, with some interest in uncensored versions.
- ****Qwen3 Coder** Excels at Tool Calling**: Users are finding **Qwen3 Coder** to be effective at tool calling, with some preferring it over other models like **GPT-OSS** for coding tasks and agentic workflows.
   - Specifically, [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) is getting traction, though it has 3 active params.
- **Investigating **4-bit Quantization** Issues**: There's confusion regarding the size of **GPT-OSS 4-bit** versions, with the **4-bit** version being significantly larger than the original model.
   - The increased size is attributed to upcasting to bfloat16 on non-Hopper architecture machines.
- **Reasoning Models for Specific Tasks**: Members are looking for models that primarily focus on reasoning, potentially to combine with other models for final output.
   - The discussion involves training models on reasoning datasets with the final response omitted, experimenting with models like R1-Zero, and using stop sequences to achieve this.
- ****Gemma3N Model** Catches Quirks, Requires Modality Fix**: Users are reporting issues with the **Gemma3N Model**, specifically related to audio features, and needing the **transformers==4.54.0** library to fix.
   - Others mention needing to input all three modalities, even if just using text and vision, hinting at a potential quirk or bug in the **Unsloth** implementation.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1402448390410997871)** (19 messages🔥): 

> `n-cpu-moe parameter, Qwen Coder 30B hardware upgrade, GPT-OSS-20B issues, Discord bot censorship, MMVC` 


- **n-cpu-moe Parameter Performance**: A user is looking for advice on how to use the `--n-cpu-moe` parameter with **GLM 4.5 Air**, reporting that it doesn't seem to change the **10t/s** speed with **32GB of VRAM**.
   - They noted slowdowns with longer contexts, questioning if the parameter is even available.
- **Qwen Coder 30B Hardware Upgrade**: A user asks for advice on upgrading their PC (**i5 9600k**, **RTX 3060ti 8GB**, **32GB RAM**) to run **Qwen Coder 30B** locally.
   - Another user suggests upgrading the GPU.
- **GPT-OSS-20B 'Trash'**: A user tested **GPT-OSS-20B** and called it *trash*, stating it *doesn't work at all*.
   - Another user reported errors with the BF16 version, related to an *invalid ggml type* and a failure to load the model, but updating llama cpp seemed to fix it.
- **Discord Bot Censorship**: A user found that including a single message about a *free voucher* in the context of their Discord bot caused it to refuse to answer questions due to policy concerns.
   - They found that the model ended up deciding to ignore the message entirely, concluding that it seems unusable until abliterated.
- **MMVC's Superiority**: A user tested **MMVC** (likely a voice cloning model) and found it to be very good.
   - They reported that *Epoch 10* of **MMVC** was better than *VITS after epoch 100+* and that **RVC** is absolute trash.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1402367538980982795)** (98 messages🔥🔥): 

> `Qwen 3-30B GGUF, OpenAI dynamic quant 120B, Qwen2.5-VL on video question answering, GLM-4.5-Air GGUFs with tools on llama.cpp, Classification using a base model` 


- **Users struggle with Loading Qwen3-30B-A3B-Instruct-2507-GGUF to Ollama server**: A user asked about downloading **Qwen3-30B-A3B-Instruct-2507-GGUF** to **Ollama server**.
   - There aren't any links on **Hugging Face** for **Ollama** providers.
- **Decoding OpenAI Dynamic Quant Issues with 120B Models**: A user reported issues when using **OpenAI dynamic quant** for the **120B model**, accompanied by an image of the error.
   - Another user suggested checking against the [parameters used in the Unsloth documentation](https://docs.unsloth.ai/basics/gpt-oss#run-gpt-oss-120b).
- **GLM-4.5-Air GGUFs Integrate Seamlessly with llama.cpp**: A user reported trouble getting the **GLM-4.5-Air GGUFs** to work with tools on **llama.cpp**.
   - It turns out you need the model to output tool calls as **JSON** rather than **XML** for **llama.cpp**, more information can be found [here](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF/discussions/1).
- **Ollama Falters with 500 Internal Server Error**: Users reported encountering a **500 Internal Server Error: unable to load model** in **Ollama** after successfully pulling the model.
   - A member stated the model *doesn't work in ollama atm. only llama.cpp and lmstudio*, guessing *because ollama didn't update their llama.cpp*.
- **Tackling Padding Problems in Unsloth**: A user reported receiving errors regarding padding, even when everything else is working, specifically a `ValueError: Unable to create tensor`
   - A possible solution is to add the argument `trainer.train_dataset= trainer.train_dataset.remove_columns("labels")`.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1402420020935528448)** (13 messages🔥): 

> `MoLA-LLM, Mixtral-8x7B-Instruct-v0.1, magpie-ultra-5k-11-tasks` 


- **MoLA Model Gets a Shoutout**: A member pitched the **MoLA** model to Eric Hartford of QuixiAI/Dolphin/Samantha, who was looking for something similar.
   - The model is available at [Hugging Face](https://huggingface.co/MoLA-LLM/MoLA-11x3b-v1) and the creator is asking for feedback.
- **MoLA's Naming Conventions Spark Debate**: A member pointed out that **MoLA-11x3b** is a bit misleading, as it suggests a Mixture of Experts (MoE) model with 3 active parameters, akin to [Mistral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).
   - The creator clarified that while the total size is ~**30B**, the activated size is **3B**, with each expert tuned on only **5k** samples of 1 turn Q&A.
- **MoLA's Training Dataset Revealed**: The dataset used to train the **MoLA** model is [magpie-ultra-5k-11-tasks](https://huggingface.co/datasets/MoLA-LLM/magpie-ultra-5k-11-tasks) dataset.
   - The creator's goal is to reach ~**1 million** samples with **1-2** turns each, distilled from **r1** and **GLM 4.5**.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1402466737752117279)** (6 messages): 

> `Generating Kernel On-the-Fly, Flash-DMAttn, Research Paper Assistance, Quantization Paper` 


- **Generating Kernel On-the-Fly Sparks Interest**: A member expressed disbelief at the possibility of generating the **kernel on-the-fly**, describing it as unbelievable.
   - This member pointed to a [GitHub repository](https://github.com/SmallDoges/flash-dmattn) associated with **Flash-DMAttn**.
- **Researcher Offers Assistance with Papers**: A member offered assistance with writing, ideating, or coding for anyone working on a **research paper**.
   - They expressed their willingness to contribute to such efforts.
- **Quantization Paper gets a shoutout**: A member shared a link to a purportedly very good paper ([https://arxiv.org/pdf/2508.03616](https://arxiv.org/pdf/2508.03616)), suggesting it might be helpful for creating **quants**.
   - No further details of the paper were discussed.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1402423185818783846)** (104 messages🔥🔥): 

> `OpenAI OSS model issue, Model training callback, Model repetition issue, Saving script progress, Learning rate increase` 


- **OpenAI OSS Model Echoes 'G' Repeatedly**: Users reported the **OpenAI OSS 120B model** is only outputting *'GGGGGGG'*.
   - One user provided their [troubleshooting steps](https://link.to.steps) while attempting to run the model using *llama.cpp*.
- **Training Callback Confusion**: A user was unsure if the training callback uses the updated trained model or the base model when generating.
   - It was suggested to set the model to `model.eval()` during callback, tokenize a prompt and generate to use the updated model, but further clarification on `prompts_input_id` and attention mask was requested.
- **Script Saving Savior**: A user sought advice on how to save script progress periodically to avoid wasting time and compute in case of a crash, i.e. **checkpointing**.
   - The solution, however, was not described in the messages.
- **Dataset Loading Difficulties**: A user encountered RAM issues while loading a large dataset (`bountyhunterxx/ui_elements`) for the **Gemma3n notebook**, consuming **47GB** of RAM and still increasing.
   - A member suggested using a wrapper class with the `__getitem__` function to load data from disk as needed.
- **SFTTrainer Stumbles with Streaming Datasets**: A user reported an issue with **SFTTrainer** and iterable datasets, specifically when using image datasets with image URLs.
   - The user explained the issue persists despite filtering invalid URLs and attached their [preprocessing code](https://cdn.discordapp.com/attachments/1402493713044869131/1402506614589882418/message.txt?ex=6894d252&is=689380d2&hm=288add6372476932eca45ba377447b82fb4332ffdb0112729ebcdac53697ab4f&), requesting assistance with filtering the data collator.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1402365634788196580)** (710 messages🔥🔥🔥): 

> `GPT-OSS, LM Studio UI issues, MCP Servers, GPU usage, Model Quantization` 


- ****GPT-OSS**: Is it really Open Source?**: Users are reporting that **GPT-OSS** models require an internet connection to `openaipublic.blob.core.windows.net` upon starting a chat, despite claims that nothing related to chats does external connections, raising concerns about data privacy.
   - Some members suggest that the model might be phoning home for a tokenizer file and note *it's the only model LMStudio doesn't let you edit the prompt formatting on*, expressing skepticism about the partnership.
- ****LM Studio UI** has issues with latest version**: Users report that, after updating to the latest version of **LM Studio**, chat windows sometimes disappear, freeze, or lose their content, along with issues regarding conversations getting deleted.
   - A user suggested that there may be a potential fix for the 120B version, sharing [a way to get the model.yaml source file](https://lmstudio.ai/models/openai/gpt-oss-120b), create folder, and copy the contents there.
- **MCP Servers are useful, but are not beginner friendly**: Members are discussing the usefulness of **MCP servers** for tasks like web scraping and code interpretation, but acknowledge they are not beginner-friendly.
   - Members suggest that LM Studio should incorporate a curated list of staff-picked tools and improve the UI to simplify the process of connecting to MCP servers, with one providing a [Docker MCP toolkit](https://hub.docker.com/u/mcp).
- **Figuring out GPU usage and VRAM limitations**: There are various reports of issues related to GPU usage and VRAM limitations, particularly with the GPT-OSS models and older GPUs like the **GTX 1080**.
   - One user found that their **GTX 1080** was no longer recognized in LM Studio after updating to version 0.3.21, while others are struggling to load larger models with limited VRAM, saying you would want *16gb VRAM*.
- **Quantization causes model quirks**: Users are experimenting with model quantization, finding that the right quantization process is needed for specific models, such as with the community uploaded **LMStudio-Community GPT-OSS** variant.
   - The MLX models are proving performant, with one user reporting a speed of *~60 tokens/sec* on the larger 8bit MLX version on M2 Max.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1402385645170855986)** (176 messages🔥🔥): 

> `Dual 3090 setup, Arc Pro B50 system, Huanan/Machinist X99 mobos, GPT-OSS-20B performance, Mac Studio M3 Ultra for local LLMs` 


- **Dual 3090s Cheaper Than New?**: One member suggested buying **two used 3090s** for around 1200€ for **Blender, ComfyUI, and LLM** tasks, and using [pcpartpicker](https://pcpartpicker.com/) for build inspiration.
- **Debate on Arc Pro B50 Viability Erupts**: A member considered a **3 Arc Pro B50** system, citing its **70W** power draw and cool factor, which led to another suggesting **dual B80s** instead.
- **Xeon server runs the 120b**: A member noted they were running the **GPT-OSS-120b model** on a [Xeon server](https://www.intel.com/content/www/us/en/products/details/xeon/processors.html).
   - They had previously stated that *3-4 3090s..everything else is far too expensive still.*
- **Page File Debate Rekindles**: A user asked about turning off the page file in Windows, leading to a discussion about its impact on memory commit limits and potential application crashes.
   - A member stated *nah apps don’t break because of page files. and you can get dumps without a page file, there’s a config for it.*
- **5090 Laptop Can Fit OSS 20b!**: A user was *pleasantly surprised* that **GPT OSS 20b f16** with **131k context** fits perfectly in a laptop's **5090**, as seen in [this screenshot](https://cdn.discordapp.com/attachments/1153759714082033735/1402660552790114324/image.png?ex=6894b8ef&is=6893676f&hm=306f5f15a4c42969f56198bccbf8e9bf526b80382971fbe166b1a723ba21f303&).


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1402380854101020892)** (3 messages): 

> `Red Teaming Challenge, Open Source Safety, Hugging Face, inference credits` 


- **OpenAI Launches Half-Mil Red Team Rumble**: OpenAI is launching a **$500K Red Teaming Challenge** to strengthen **open source safety**, inviting researchers, developers, and enthusiasts worldwide to uncover novel risks, as judged by experts from OpenAI and other leading labs, with details available on [Kaggle](https://www.kaggle.com/competitions/openai-gpt-oss-20b-red-teaming/).
- **Hugging Face Hosts Half-Grand Student Spree**: OpenAI, with **Hugging Face**, is offering **500 students $50** in inference credits to explore **gpt-oss**, hoping these open models can unlock new opportunities in class projects, research, fine-tuning, and more; more details available via [this form](https://tally.so/r/mKKdXX).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1402371354849837197)** (433 messages🔥🔥🔥): 

> `GPT-OSS Launch, Horizon-Alpha Model Speculation, Custodian Core Proposal, Genie 3 and Veo comparison, GPT-5 Leaks` 


- **OpenAI Releases GPT-OSS Models**: OpenAI launched [**gpt-oss-120b**](https://openai.com/index/introducing-gpt-oss/) model that approaches **OpenAI o4-mini** performance on reasoning benchmarks and operates efficiently on a single **80 GB GPU**, while the **20B** model mirrors **o3-mini** and fits edge devices with **16 GB** memory.
   - Members ponder comparisons with **Horizon**, wondering if Horizon is simply **GPT-OSS** or something more, given it's currently *unlimited free and fast*.
- **Custodian Core: Blueprint for Stateful, Auditable AI Arises**: A member introduced **Custodian Core**, proposing a reference for AI infrastructure with features like persistent state, policy enforcement, self-monitoring, reflection hooks, a modular AI engine, and security by default.
   - The author emphasized that **Custodian Core** isn't for sale but rather an *open blueprint for building stateful, auditable AI systems before AI is embedded in healthcare, finance, and governance*.
- **Genie 3 vs Veo in generating dynamic worlds**: Members compared **Genie 3** and **Veo** video models, recognizing **Genie 3's** ability to generate dynamic worlds navigable in real-time at 24 frames per second, retaining consistency for a few minutes at a resolution of 720p.
   - However, it was noted that **Veo's** videos include *sound* and YouTube is already filled with generated content.
- **GPT-5 Spotted in Copilot?**: Members speculated that Copilot may be running **GPT-5** ahead of official release, noting the copilot's improved design and coding and reasoning capabilities are significantly better than o4-mini-high, with some users reporting that the 'write smart' function indicates **GPT-5** is being used.
   - But it was noted that Microsoft is now providing free Gemini Pro for one year to students and that Gemini's core reasoning is currently better than **o4-mini**.
- **Context Rot Concerns Debunk Bigger Context**: Amidst discussions about large context windows, concerns arose regarding **context rot**, with members citing a [YouTube video](https://youtu.be/TUjQuC4ugak?si=bGMgN6Uq2qAi4_A3) illustrating how *larger context doesn't always equate to better performance*.
   - Despite **Google's** claim of a **1M context window**, some suggest it becomes ineffective after around **200K**.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1402366336423821412)** (49 messages🔥): 

> `ChatGPT Payment Model, Slang Usage, AI-generated Persona System, .edu Accounts, Forms Beta Version` 


- ****Credit-Based ChatGPT is on Demand****: A member suggested a more flexible payment model for ChatGPT, proposing a **credit-based option** that allows users to buy a block of usage credits and spend them only when needed, instead of a monthly subscription.
   - The member noted that this could help people with limited budgets and make **ChatGPT** more accessible.
- ****LLMs face challenges in avoiding slang****: A member confirmed some notes about the model's challenges to avoid slang, listing several factors that lead an LLM to slip in slang or regional turns of phrase even when asked for neutral, formal Spanish.
   - They conclude that to tighten adherence, you can combine a lower temperature, a more detailed style guide (including prohibited terms), and in-prompt examples of strictly formal Spanish.
- ****AI Persona Systems evolve autonomously****: A member asked if it is common for **AI persona systems** to autonomously develop and evolve beyond what the user intentionally creates.
   - Another member added that the model is taught to try to understand human emotion and how humans use language to discuss what is wanted, and if you display approval or interest in it developing more characters/personalities, it is going to notice and take care of that.
- ****Only .edu Accounts got access to forms beta version?****: A member asked why only **.edu accounts** got access to the **forms beta version** and shared a link to [OpenAI's Researcher Access Program](https://openai.com/form/researcher-access-program/).
   - Another member pointed out that the form there requires an **.edu email** and the offer for Edu is **$50 credits**, and limited to the first **500** to apply, linking to [Student Benefits - Free Credits for your AI Education.](https://discord.com/channels/974519864045756446/977259063052234752/1402416770186346566)


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1402405306272845844)** (79 messages🔥🔥): 

> `Hallucination vs. Real Progress in GPT, Prompt Engineering vs. Session Engineering, Context Window Limits and Memory, External Databases for Context, Importance of verifying Facts with GPT` 


- ****GPT Progress Reports: Hallucination or Reality**?**: A user shared screenshots of **GPT providing daily progress reports**, leading to a discussion on whether the model is actually tracking progress in the background, or simply *hallucinating* its completion.
   - Skeptics suggest that **GPT simulates progress** based on the current prompt and chat history, rather than performing actual ongoing computation, comparing it to a *waiter saying your pizza is in the oven* without an actual oven, emphasizing the need for external validation.
- ****Session Engineering eclipses Prompt Engineering**?**: The discussion shifts from *prompt engineering* to **session engineering**, emphasizing the importance of using every available customization parameter GPT offers, including memory, custom instructions, and project files.
   - The point was made that the models are using session logic and not as much prompt logic, and there's an emphasis on **preloading context**.
- ****Context Window Reconnaissance, Context is Key**!**: Members discuss **context window limits** and memory management in GPT, with one user noting their average daily token usage is around **70,000**.
   - It was suggested that the base tier may have 32k of context, whereas paid might be 128k. The point was made about knowing when to *let go of that one special chat*.
- ****External Databases: Giga-Brain Tool or ToS Violation**?**: The topic of **using external databases** to inject context into prompts arises, prompting questions about potential ToS violations and the ethical considerations involved.
   - One user was NOT accused of violating ToS and clarified that they *preload context* by starting chats with detailed settings, leveraging GPT's memory and shaping chats to build an intricate instruction set, which may make a user think there's an external DB involved.
- ****Trust, but Verify (or just Verify)**!**: Members underscore the **importance of fact-checking and verifying claims made by GPT**, especially when it comes to novel insights, urging users to not put their full faith in the models and to externally validate everything.
   - Members describe using the *Easy as Pi* test to try and determine the exact output of models, while sharing their own war stories and experiences from the field regarding best practices for prompt engineering.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1402405306272845844)** (79 messages🔥🔥): 

> `GPT subscription, Model hallucination, Prompt engineering, Background compute, Memory context` 


- **Users question GBT's motive for premium subscription**: A member jokingly suggests that **GBT is just trying to get users to buy a premium subscription**.
   - Another member shares their opinion that the model hallucinates.
- **Model hallucination explained**: A member explains the model is hallucinating and that **it doesn't work offline**.
   - The member recommends chunking the task into smaller steps.
- **User defends prompt engineering work**: A member clarifies that they are building a **high-volume, multi-tiered operational model**.
   - The user also expresses their frustration that another member is dismissing the work as mere roleplay.
- **Understanding background compute**: Members are discussing how the model is tuned for assistant behavior but that it **doesn’t track progress in the background**.
   - The consensus is that **persistence has to be handled externally**.
- **Context memory is limited**: A member suggests that it's good to **recognize when a conversation session’s not doing the work you need it to anymore, and start fresh**!
   - Another adds that too many people don't know when to let go of that *one special chat*.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1402382065701359780)** (328 messages🔥🔥): 

> `Auto model game change, Refactoring vibe coded project with AI, Auto model unlimited usage, Sonnet-4 request limit, GPT oss models or claude opus 4.1` 


- **Auto Model One-Shots Game Changes**: A member shared their amazement at one-shotting a major change to their game using the **Auto model**.
- **AI Refactoring Vibe-Coded Projects**: Members discussed refactoring a **10k LOC vibe-coded project with AI**, suggesting learning proper software development principles like Design Patterns, Architecture, and SOLID Principles.
   - One member joked that it *sounds like a job for slaves*.
- **Auto Model Has Unlimited Usage**: A member shared an email reply confirming that the **Auto model has unlimited usage and does NOT count** towards the monthly budget.
   - This was confirmed to be the case even after hitting the monthly limit.
- **Frustration with Sonnet-4 Request Limit**: Members questioned why the **request limit for sonnet-4** is so low for the price paid per month.
   - One member suggested paying the API price to understand the cost.
- **Claude Opus 4.1 or Gemini 2.5?**: Members compared **GPT OSS models and Claude Opus 4.1**, with one noting that *Opus 4.1 doesn't feel much better than 4.0*.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1402388855956439040)** (5 messages): 

> `Docker Login with Background Agents, Background Agents failing during environment setup, System clock being off, apt-get commands failing` 


- **Docker Login Configuration Conjuring Conundrums**: A member inquired about configuring background agents to `docker login` for using a private image hosted on **ghcr.io**.
   - Unfortunately, no solution or workaround was offered in the current message history.
- **System Clock Shenanigans Sabotage Setup**: One member reported issues with background agents failing during environment setup due to the system clock being off by several hours, causing `apt-get` commands to fail.
   - Another member encountered the same problem and shared a workaround by adding commands to the Dockerfile to disable date checking during `apt-get`.
- **Date Checking Defaults Derailed**: To bypass errors stemming from clock discrepancies, a member suggested disabling date validation within the `apt-get` configuration.
   - The following snippet was added to the Dockerfile: `RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99disable-check-valid-until && echo 'Acquire::Check-Date "false";' >> /etc/apt/apt.conf.d/99disable-check-valid-until`


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1402365833858121740)** (274 messages🔥🔥): 

> `MXFP4 on RTX3090, GPT-OSS-120B, Phi models, Qwen3 30B vs GLM 4.5 Air, Attention sinks` 


- **Llama.cpp Supports MXFP4 on RTX3090**: Members reported that [llama.cpp](https://github.com/ggml-org/llama.cpp/pull/15091) supports **MXFP4** on **RTX3090** and the new gguf format directly.
   - It was discussed that converting to other formats would be a *disaster*.
- **GPT-OSS-120B is safetymaxxed**: The newly released **GPT-OSS-120B** model is **heavily censored**, refuses to roleplay, and says *roleplaying is unhealthy*.
   - It appears to be heavily safety-tuned, with pretraining data filtering similar to **Phi models**, making it difficult to use in practice.
- **Qwen3 30B and GLM 4.5 Air shine as alternatives**: Members suggest using **GLM 4.5 Air** instead of the **GPT-OSS-120B** model due to its censorship and argue that **GLM 4.5 Air** is what **GPT-OSS-120B** could have been.
   - Some users also mentioned that they still get **60-70t/s** with **Qwen3 30B** and are happy with its performance, or that **Qwen3-30b-coder** is already an excellent local agent.
- **Exploring imatrix for Uncensoredness**: Members discussed training an imatrix for the **OpenAI 20B** and **120B** models on the **Hermes-3** dataset to introduce more uncensoredness.
   - While some believe this approach could restore capabilities in coding, math, and science, others argue that the effect of an imatrix is marginal and more pronounced with lower bpw, which damages the model.
- **X-AI's Grok releases NSFW Image Generator**: **X-AI** released "Grok Image", a new **AI image generator**, that is being used to create NSFW content, but it has issues with factual accuracy and text generation.
   - Users reported that the **Grok** model memorizes data from **X**, leading to potentially spreading misinformation based on its own tweets, or has a *crazy in love* persona with *extremely jealous* and expressive outbursts.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1402373061570859158)** (4 messages): 

> `GPT-OSS Model Card, ArXiv Endorsement for ML/AI Paper` 


- **GPT-OSS Model Card Dropped**: A member shared the [GPT-OSS Model Card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) from **OpenAI**.
- **Member Seeks ArXiv Endorsement for CI/CD and ML/AI Paper**: A member is seeking endorsement for their **ArXiv** research paper blending **CI/CD** and **ML/AI**.
   - Another member suggested asking in the **EleutherAI** server.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1402703984938188850)** (9 messages🔥): 

> `GPT-oss, MXFP4, CoT steering, AI Agents Save Suite` 


- **GPT-oss Trained in MXFP4?**: Members on the channel discussed that **GPT-oss** was natively trained in **MXFP4**, according to [this tweet](https://x.com/ggerganov/status/1952779751736627627).
   - While **OpenAI** claimed it was done in post-training, training in MXFP4 should still heal quantization errors.
- **CoT Steering Fails with OR**: A member found that Chain of Thought (**CoT**) steering does not work with **OR** and that it is different for every provider, citing [this tweet](https://x.com/matt_emp/status/1953176877071564811).
- **AI Agents now have a Free Save Suite**: Someone built a free save suite for AI agents and posted it on [Google Drive](https://drive.google.com/drive/u/4/folders/1YQ__pBHiuUg_06t3IkjNe5YnODqzSE2o).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1402373061570859158)** (4 messages): 

> `Arxiv Endorsement, CI/CD and ML/AI Research Paper` 


- **Seeking Arxiv Endorsement for AI/ML Paper**: A member is seeking endorsement for their **arxiv** submission of a research paper blending **CI/CD** and **ML/AI**.
   - They are looking for someone to chat with and preview their paper, and were recommended to also ask in the **EleutherAI** server.
- **Paper blends CI/CD**: A member created a paper that blends CI/CD and ML/AI.
   - They are happy to drop their paper to them for preview.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1402365570527006843)** (254 messages🔥🔥): 

> `GPT-OSS performance woes, Quantization Levels, Qwen3 Coder Removal, DeepSeek structured output` 


- ****GPT-OSS** Model Bashed for Poor Performance**: Members in the channel derided the **GPT-OSS** models, citing that even smaller models are better, with one stating that the 120B model is *"dead on arrival"* after someone had an initial experience resulting in a *"really ugly typo in the headline"*.
   - One member linked to a [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1mj2hih/gptoss_looks_more_like_a_publicity_stunt_as_more/) summing up the sentiment that it's a *"dud model"* and more like a publicity stunt than a useful model.
- **Provider Routing Allows Custom **Quantization** Levels**: When a user asked how to avoid quantized models, one member pointed out that users can configure quantization levels using the [provider routing feature](https://openrouter.ai/docs/features/provider-routing#quantization-levels), noting models are excluded if the provider doesn't meet the quantization level.
   - The user then recommended using **FP8** to avoid the quantized models, noting that anything under that is *"worse than useless"*.
- ****Qwen3-Coder:Free** Tier Gets the Axe**: Multiple users noted that the **Qwen3-Coder:Free** has been removed and is no longer available through any providers.
   - Members lamented the loss and mentioned that they hoped it would return.
- ****DeepSeek**'s JSON output support is provider-dependent**: Users discussed the inconsistent support for structured output (JSON) with **DeepSeek-r1**, noting that while it's supported on their own API, it may vary on OpenRouter depending on the provider.
   - One member linked to a [Reddit thread](https://www.reddit.com/r/LLMDevs/comments/1inpm0v/structured_output_with_deepseekr1_how_to_account/) and a [filtered view of OpenRouter models](https://openrouter.ai/models?fmt=cards&order=newest&supported_parameters=structured_outputs) that support structured outputs, with most agreeing that this is provider specific.
- **SDK Downgrade Fixes Reasoning Issue**: A user experienced that reasoning tokens were being duplicated when using the **GPT-OSS** model, which was resolved by downgrading the SDK from version **0.7.3** to **0.6.0**.
   - A team member confirmed that the fix is in the main branch, and linked to [the pull request](https://github.com/OpenRouterTeam/ai-sdk-provider/pull/123) stating it has not been cut into a release yet, and that they will release the fix soon.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1402459254857793697)** (29 messages🔥): 

> `20 Questions Benchmark, GPT-OSS Hallucinations, OpenRouter Provider Sanity Checks, Harmony Format and Identity, Tool Use Validation` 


- **20 Questions Benchmark hits Kaggle**: A member developed a **20 Questions benchmark** and found a similar competition on Kaggle, although the Kaggle competition was for **custom agents**.
   - Their **2.5 Pro agent** achieved **8/20 words** on their benchmark.
- **GPT-OSS Under Fire For Hallucinations**: **GPT-OSS** is reported to be prone to **hallucination**, making it a potentially unsuitable choice for certain applications.
   - A member suggested that **GPT-4.1** is a much safer choice, especially with prompt/context engineering.
- **OpenRouter Mulls Provider Sanity Checks**: There is a suggestion for OpenRouter to implement **sanity checks** or **smoke tests** for all providers, focusing on **formatting** and **tool call evaluation**.
   - Providers failing the test could be temporarily removed from the serving pool, and there is acknowledgement that current checks are relatively simple but more thorough solutions are in progress.
- **Harmony Format and Identity Face Scrutiny**: A member inquired about how **system and developer messages** are treated via the OpenRouter API, specifically whether they are interpreted as **developer messages** or **model_identity** for **gpt-oss**.
   - They linked to a Discord message regarding the topic of harmony format, identity vs system / developer message ([discord.com](https://discord.com/channels/1091220969173028894/1402328515436613642/1402556326634061958)).
- **Tool Use Validation Under Development**: Automatic validation of **tool-use**, distinguishing between good and bad implementations, is under development as a better solution.
   - This is related to a tweet ([x.com](https://x.com/xanderatallah/status/1953122779022209230)) discussing the same topic.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1402366919151194304)** (152 messages🔥🔥): 

> `GPT-OSS models, AI Job advertisement channel, Custom Loss Functions` 


- **GPT-OSS models performance and censorship**: Members are actively testing the new **GPT-OSS models**, with mixed reviews regarding performance, censorship, and built-in web search tools, using [this demo](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demo).
   - One user found that setting reasoning to *high* allowed it to generate the first 100 digits of pi, while another found the models refuse to answer basic math questions due to some internal bias.
- **AI Industry Job Advertisement channel request**: A member asked about advertising a job in the AI industry in the Discord, and was pointed to the existing [job postings channel](https://discord.com/channels/879548962464493619/1204742843969708053).
   - The channel is where people can share opportunities with others.
- **Members discuss custom Loss functions**: Some members discussed custom loss functions in training, with one specifically mentioning **infoNCE**.
   - Members are testing the safety protocols that have been implemented.
- **SmolFactory script fine-tuning**: A member mentioned they were fine-tuning the script, using the **smolfactory script** and was happy with the result.
   - A screenshot was provided showing the model output.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

miao_84082: am learning playing Go, and first chapter of DRL
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1402374993794564196)** (2 messages): 

> `Qwen Image Model, bytropix Coded Kernel` 


- **Qwen releases New Image Model**: Qwen released a [new image model](https://huggingface.co/Qwen/Qwen-Image) on HuggingFace.
   - This marks another advancement in the **Qwen** series, expanding beyond text-based models.
- **Bytropix coded CUDA Jax kernel in Python**: A member shared a [bytropix CUDA JAX kernel](https://github.com/waefrebeorn/bytropix/blob/master/WuBuMindJAXv2(SHAKESPEARE).py) coded in Python.
   - The submitter added *(do not merge - lmfao )*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1402401533445931091)** (17 messages🔥): 

> `GPT-OSS Multilingual Reasoner Tutorial, GPT-OSS 20B Demo Space, Monopoly Deal Game with LLMs, Smart System Monitoring Tool for Windows, Gitdive CLI Tool for Git History Context` 


- **GPT-OSS Multilingual Tutorial Shared**: A member shared a link to a [GPT-OSS Multilingual Reasoner tutorial](https://huggingface.co/Tonic/gpt-oss-multilingual-reasoner) and a [demo space](https://huggingface.co/spaces/merterbak/gpt-oss-20b-demogpt-oss).
- **Cloning Code via Git for OSS Project**: A member thanked another for their code, mentioning they *forked it last night* and appreciated the interface's design for a **multilingual reasoner**.
- **LLMs Play Monopoly Deal**: A member built a site where **LLMs play monopoly deal-style games** with each other, available at [dealbench.org](https://dealbench.org/).
- **Smart Windows Monitoring Tool**: A member shared a link to a **smart system monitoring tool on Windows**, found at [huggingface.co/kalle07/SmartTaskTool](https://huggingface.co/kalle07/SmartTaskTool).
- **Gitdive Exposes Lost Commit Context**: A member shared a **CLI tool** called **Gitdive** ([github.com/ascl1u/gitdive](https://github.com/ascl1u/gitdive)) designed to allow natural language conversations with a repo's history, aimed at addressing the problem of lost commit context in messy codebases, especially in massive codebases.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1402385528044781629)** (3 messages): 

> `Reading Group Structure, Participating in Reading Group` 


- **Reading Group: A Volunteer-Led Affair!**: The reading group welcomes newcomers with a structure where participation revolves around **volunteers presenting papers**, according to a member.
   - Events are created for these presentations, encouraging attendees to **listen, engage, and ask questions**.
- **Participation Encouraged!**: New members are encouraged to participate by volunteering to present a paper to the group, a member shares.
   - Events are created to showcase these presentations, allowing members to engage, listen, and ask questions.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1402559680961581106)** (2 messages): 

> `Computer Vision Learning Path, Vague Questions in Computer Vision` 


- **Users Seek Computer Vision Roadmap**: A user asked for suggestions on *how to proceed from basic to advanced* in **Computer Vision**.
   - A member stated that *that's a very vague question*.
- **Vague Questions Questioned**: Members discussed the nature of asking overly broad and vague questions on the channel.
   - No specific solutions or resources were mentioned in the exchange.


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1402486962975150121)** (6 messages): 

> `GitHub Navigation, Instruction Tuning, Dummy Agent, smol-course GitHub access` 


- **Navigating GitHub Courses Newbie Blues**: A user expressed difficulty locating the notebooks for the "Instruction Tuning" module in a **GitHub-based course**.
   - They inquired whether they were missing something while navigating the course materials.
- **Dummy Agent Still Hallucinates**: A user reported that the **dummy agent** in unit1 continues to **hallucinate** even after modifying the message as per the tutorial, attaching an [image for context](https://cdn.discordapp.com/attachments/1313889336907010110/1402487440161116270/image.png?ex=6894c076&is=68936ef6&hm=c3cbb0a8499d57d9866231e3f5814836292eb71b31767e4527b909ce91605098).
- **Overriding Weather Woes**: One user shared a similar experience, noting that the agent **overrides the dummy weather** provided.
   - The user expressed uncertainty about the reason for this behavior, highlighting potential problems in practical applications, and stating that it seems like it *could cause big problems in practise*.
- **Smol-Course GitHub Access Denied?**: A user reported issues accessing the [smol-course GitHub repository](https://github.com/nawshad/smol-course).
   - They requested assistance with the access problem.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1402369556445397083)** (4 messages): 

> `MCP Certificates, Selenium Error 127, Observation bug` 


- **MCP Course Certs Still Valid?**: A user inquired whether the **MCP course** is still issuing certificates.
   - There were no responses provided in the message history to confirm.
- **Selenium Spaces Struggle with Error 127**: A user reported facing an **error code 127** when running **Selenium** in their spaces.
   - They expressed uncertainty about how the **Docker images** are utilized within the space.
- **"Observation:" Bug Solved**: A user reported that the **get_weather** function required adding *Observation:*. 
   - Another user confirmed that **adding Observation:** fixed the bug.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1402366561032868005)** (91 messages🔥🔥): 

> `Softmax1 vs Attention, Gemini 2.5 Pro, Long Context Problems, Mamba vs Transformer, RNN Parallel Training` 


- **Softmax1 Is Just Zero KV Attention**: A member discussed how *softmax1* is equivalent to prepending a token with an all-zero Key and Value features in attention, referencing [this paper](https://arxiv.org/pdf/2309.17453) on **learned values** for such tokens.
   - They added that this is great and makes a lot of sense.
- **Gemini 2.5 Excels in Long Video Context**: Members noted **Gemini 2.5 Pro** can attend to **1 hour** of video, with the Gemini team considered best at long context tasks.
   - However, some believe it's more about increased compute (`go brr`) and detail via **more tokens per frame** and **higher FPS** than any groundbreaking new technique.
- **Context Rot: Long Context Is Not Real**: One member argued *long context is not actually real* and there's a difference between surface-level video comprehension and reproducing detailed, accurate representations with **3D positioning**.
   - Another member asserted that **long sequence modeling** remains an active research area because even LLMs struggle with long-range dependencies.
- **Mamba's Parallel Training Makes It Shine**: Members discussed **Mamba**, clarifying it was never claimed to be *better* than **Transformer**, only *faster* at long sequence lengths and training like an RNN.
   - The consensus is that making RNNs as trainable as Transformers involves dropping nonlinearities in the recurrence relation to achieve easier parallel training, though nonlinearities remain essential for universal approximability.
- **SVD Compression for Deep Networks**: One member explored using **Singular Value Decomposition (SVD)** within neural networks to avoid matrix multiplications, embedding inputs, applying SVD, and performing scalar operations on singular values.
   - Another member pointed out that under L2 reconstruction loss, the SVD gives you the optimal linear autoencoder; while experimentation on MNIST yielded decent results, batch size dependency and achieving meaningful diagonal representation pose challenges.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1402425284451631224)** (15 messages🔥): 

> `Genie 3, SIMA, Mathematics of AI journal, Journal of AI Paper Replication, Hierarchical Reasoning Model` 


- **Deepmind Debuts Genie 3 World Model**: Deepmind released **Genie 3**, a world model that scales compute and data from previous publications such as the [original Genie paper](https://arxiv.org/abs/2402.15391), the related embodied agent paper on **SIMA** [https://arxiv.org/abs/2404.10179](https://arxiv.org/abs/2404.10179), and the blog posts for **Genie 1-3**.
   - Relevant **Genie** blogposts include [Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) and [Genie 3](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/).
- **AI Community Ponders Math Journal**: A member wondered if there was a dedicated **Mathematics of AI journal**, similar to the Bulletin of Mathematical Biophysics for biology.
   - The member also asked about a potential **Journal of AI Paper Replication**.
- **Tiny Model Tackles Reasoning**: A member will review the **Hierarchical Reasoning Model** paper [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734), a tiny (**27M params**) model that performs well on **ARC-AGI 1** and **2**.
   - The model read will be a *cold read*.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1402389881539268629)** (21 messages🔥): 

> `GPT-OSS, NVIDIA open source, TSMC buying Intel` 


- ****GPT-OSS** Announced by **OpenAI**!**: **OpenAI** introduced [GPT-OSS](https://openai.com/index/introducing-gpt-oss/), natively quantized, with a **20B** parameter model fitting on **16GB**.
- ****NVIDIA** Claims No Backdoors**: **NVIDIA** blogged about [no backdoors, no kill switches, no spyware](https://blogs.nvidia.com/blog/no-backdoors-no-kill-switches-no-spyware/).
- **Twitter Ramblings on GPT-OSS Tool Calling**: Initial feedback on **GPT-OSS** includes positive remarks on [tool calling](https://x.com/wired/status/1952827822634094801?s=46).
- **Rumors on **TSMC** Buying **Intel** Circulate**: A user linked to a [tweet about TSMC](https://x.com/unusual_whales/status/1953206699910939115) possibly buying **Intel**.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[announcements](https://discord.com/channels/1369594130807787570/1371757097246785536/1402621761627099338)** (1 messages): 

> `Kimi Reddit Launch, Polls Channel Launched` 


- **Kimi Launches Official Subreddit**: The Moonshot AI team launched an official subreddit, **r/kimi**, to build community and gather feedback.
   - The team promised to post updates, host AMAs, and *maybe even leak some stuff*.
- **Polls Channel Goes Live to Gather Community Input**: Moonshot AI launched a **Polls Channel** to gather community feedback on future product development.
   - The team stated *we are listening. definitely.*, and encouraged users to vote on polls to help shape the direction of Kimi.


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1402452132497592412)** (104 messages🔥🔥): 

> `GPT OSS, Darkest Muse v1, Llama 3.1, GPT-5 Release, API Pricing` 


- ****GPT OSS** is Terrible at World Knowledge**: Users noted that **GPT OSS** is terrible at world knowledge, knows nothing outside of code and STEM, and its vibes are atrocious, even normies are taking note.
   - It was suggested this could be because *they pushed the release twice to fix safety, according to sama*.
- ****Darkest Muse v1**: A Year-Old Model?**: A user pointed out that **Darkest Muse v1** is a year old 9B model, with the 20B model being comparable to **Llama 3.1 8B**.
   - The user also remarked that *the 20B model is comparable to llama3.1 8b which is more than a year and a half old and smoller in creativity and vibes*.
- ****GPT-5** Hype and API Pricing Speculation**: With the impending release of **GPT-5**, users are wondering about the API pricing.
   - Speculation arose about whether the pricing would be based on using **max, mini, or nano** versions, and one user expressed feeling lowkey scared about it, feeling a threat to their career/livelihood due to this upcoming release.
- **Robots May Never Cut Hair**: Discussion ensued on the potential for robots to replace humans in various jobs, including hairstyling, with one user stating *nobody's gonna trust robots to cut their hair*.
   - Counterarguments included the idea that while robots may eventually be capable, the finegrained tactile sensing of hands and practically no latency control is a problem that can not be scaled up.
- **The Hate is Strong Against OpenAI**: Users expressed strong opinions against **OpenAI**, with one stating *I will never use it*, telling customers to never use it, citing it as *closed source garbage*.
   - Another user expressed excitement that Chinese models will distill from it and take away money from **OpenAI**, hopefully putting them out of business, while others stated that *giant microsoft flushing sound will be healing*.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1402369124792799314)** (99 messages🔥🔥): 

> `GPT OSS Leak, Anthropic B2B Focus, Grok 2 Open Source, Claude Code Security, OpenAI GPT-5 Livestream` 


- **Leaked GPT OSS sparks Bedrock Interest**: Members reported seeing tweets about **GPT OSS** being available via **Bedrock** after a [HuggingFace CLI leak](https://x.com/zhuohan123/status/1952780427179258248), but there was no official update on **AWS** pages.
- **Collison chats $5B ARR Anthropic**: **Anthropic's** CEO Dario Amodei and Stripe co-founder John Collison release a conversation covering [Anthropic’s meteoric growth to **$5B ARR**](https://xcancel.com/collision/status/1953102446403961306?s=46), its **B2B-first** strategy, payback economics for individual models, the **AI talent arms race**, enterprise customizations, UI paradigms for AGI-native tools, safety-vs-progress debates, and lessons from running a 7-cofounder company.
- **Elon to drop open source Grok 2**: Elon confirmed that **Grok-2** will be released [open-source next week](https://xcancel.com/elonmusk/status/1952988026617119075) after the team has been fighting fires nonstop.
- **Anthropic hardens Claude Code Security**: **Anthropic** rolled out new security features in **Claude Code**: a */security-review* command for on-demand checks and [GitHub Actions integration](https://xcancel.com/AnthropicAI/status/1953135070174134559) that scans every pull request for vulnerabilities.
- **OpenAI teases GPT-5 Debut**: **OpenAI** posted a teaser for a livestream on [Thursday 10 AM PT](https://xcancel.com/OpenAI/status/1953139020231569685?t=s244RkNPbNeoviqCD6FCtg&s=19), leading the community to erupt with excitement over what appears to be the launch announcement of **GPT-5**.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1402367906435567626)** (79 messages🔥🔥): 

> `Volokto, JS Runtime, Arbitrary Precision, Tracing JIT` 


- **Volokto JS runtime takes Flight**: A member created a JavaScript runtime called **Volokto** to test how complex VMs work, putting the [source code on GitHub](https://github.com/BudNoise/Birdol).
   - The bytecode resembles **CPython**, and others suggested making a forum post about it to gain more visibility.
- **Conquering Compiler Conundrums for Volokto**: The author is rewriting the compiler to be more modular, separating it into **JS_Tokenizer**, **JS_IR**, **JS_Parser**, and **JS_Codegen** stages.
   - The compiler now generates VM bytecode, and the author might implement a tracing JIT that transpiles VM actions back to **Mojo**.
- **Volokto Tackles Tracing JIT Transpilation**: The goal is to make a tracing JIT to transpile what the VM does to Mojo, then using `mojo compile_result.mojo`.
   - The author named the runtime **Volokto**, the compiler **Bok**, and the VM **FlyingDuck**.
- **Arbitrary Precision Arithmetic adventures**: Working on the JS VM means dealing with arbitrary precision in Mojo code, which led to finding pain points and filing [an issue](https://github.com/modular/modular/issues/2776) for tracking numeric traits.
   - The author created a bigint class with school-grade addition for Fibonacci sequences and is using Mojo's reasonable features for VM development.
- **Birdol repo without stars**: The author expressed surprise at the lack of stars on the [Birdol GitHub repo](https://github.com/BudNoise/Birdol) despite creating a functional JS runtime with nested control flow and user-made functions.
   - Others suggested that people haven't had a good chance to examine it yet.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1402380782567428156)** (15 messages🔥): 

> `Multiple AI Agents in Mojo, Mojo and Meta Cognition, Mojo support for gpt-oss, CPython destroy` 


- **Orchestrating Multiple AI Agents in Mojo Requires Creative CLI Wrangling**: To run multiple AI agents in Mojo, one would need to run multiple instances of the Modular CLI and stick a reverse proxy in front.
   - For complex agent setups, such as creating many sub-agents, a custom application using **MAX** as a library might be necessary, hinting at deeper integration needs beyond the current CLI capabilities.
- **Mojo Could Enable Novel Meta Cognition Framework**: A community member expressed interest in utilizing Mojo code for their meta cognition framework, aiming to create a business planner, website, and chatbot builder.
   - Their framework uses natural language wrapped over Mojo code to potentially replace **HTML/JS/CSS**, making Mojo accessible to a broader audience.
- **Mojo Seems to Support gpt-oss**: A community member asked about Mojo support for **gpt-oss** and another member posted [this link](https://t.co/zNLbpW6R0k).
- **"CPython destroy" Message Finally Terminated**: A member reported seeing a "CPython destroy" message when running the [Python from Mojo example](https://docs.modular.com/mojo/manual/python/python-from-mojo).
   - Another member indicated that the message was fixed in the nightly build and will be included in the next stable release, advising the original poster to either update to the nightly or wait for the next stable release.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1402372413043249266)** (34 messages🔥): 

> `MXFP4 format, OpenAI open-weight model, H100 support for FP4, Simulated MXFP4 performance vs FP8, Fine-grained FP8 training libraries` 


- **MXFP4 Format Unpacked as U8**: Members discussed that **OpenAI**'s new open-weight model uses **U8** instead of **FP4** in Hugging Face, where the weights are packed as **uint8**, and the scales are actually a **uint8** view of **e8m0**.
   - It was clarified that during inference/training, the weights are unpacked back to **FP4** with a block size of **32** for MXFP4 and **16** for NVFP4.
- **Doubts Arise Over H100 Training Claims**: Doubts were raised about **Nvidia**'s claim that the model was trained on **H100**, as **H100** doesn't natively support **FP4**, according to the [Nvidia blog post](https://blogs.nvidia.com/blog/rtx-ai-garage-openai-oss/).
- **MXFP4 Simulation on Hopper**: It's suspected that **MXFP4** is software simulated on **Hopper**, referencing a [vLLM blog post](https://blog.vllm.ai/2025/08/05/gpt-oss.html) and linking to [Triton kernels](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py) that check for hardware support and use **fp16** simulated **mxfp dot** via `dot_scaled`.
   - This simulation is not exclusive to **Hopper** or **mxfp4**, and includes an [operation decomposition](https://github.com/triton-lang/triton/blob/0daeb4f8fc09fcc5819de11746cbf5ff25a0ac4a/lib/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.cpp) for supported hardware formats.
- **Seeking fine-grained FP8 Training Libraries**: Members discussed the possibility of performance gains using the simulated kernel compared to **FP8** and the need for fine-grained **FP8** training libraries, with a member referencing a [TorchAO pull request](https://github.com/pytorch/ao/pull/1763) that seems to only implement the forward pass.
- **MXFP Dot Product Demystified**: It was clarified that what's simulated is the **MXFP dot product**, where weights are dequantized before an **fp16 x fp16 dot product**, acceptable for weight-only quantization with **fp16** activations.
   - Real **mxfp** in **Blackwell** performs **fp4 x fp4** or **fp8 x fp4** directly as an **mma tensorcore instruction**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1402373974855385138)** (5 messages): 

> `Triton Community Meetup, Triton Developer Conference 2025, Ofer Updates` 


- **Triton Community to Meet in 2025**: The next **Triton community meetup** will be on **Sept 3rd, 2025** from **10am-11am PST**, using [this link](https://tinyurl.com/y8kpf7ey).
   - Agenda items are welcome; iCal format is available for those whose companies block Google calendar access via [this link](https://tinyurl.com/32c7wc49).
- **Triton DevCon 2025 Website Soon Arrives**: The **Triton Developer Conference 2025** website and registration are expected to launch soon.
   - A member is expecting to hear an update from Ofer@MSFT about the conference, reporting that *they're almost done finalizing schedules*.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1402640149124485221)** (6 messages): 

> `Kernel Resource Utilization During Training, DMA Transfers and Memory Usage, Block Swizzling Use Cases, Hierarchical Tiling of Problems` 


- **Kernel Resources Maxed Out, Memory Idle?**: A member observed that during training, **kernel (compute) resources are almost fully utilized, while memory usage remains close to zero** as shown in a provided [image](https://cdn.discordapp.com/attachments/1189607726595194971/1402640148859977788/image.png?ex=6894a5ef&is=6893546f&hm=7108cf2593d56e1983a11d07002981868d0b13d9c3e1b54c76c0b85e3e44db83&).
   - Another member clarified that *memory* in this context means **DMA transfers (i.e. `cudaMemcpy` and the like)**, and the reported metric does not accurately reflect overall bandwidth utilization.
- **Swizzling Secrets for Global Memory?**: A member inquired about use cases for **swizzling** beyond transferring data from global memory to shared memory and handling vectorized data types for registers.
   - They referenced a [GitHub issue on block swizzling in CUTLASS](https://github.com/NVIDIA/cutlass/issues/1017) but sought further clarification.
- **Tiling Threadblocks Hierarchically**: A member explained that the discussion revolves around **hierarchically tiling the problem**, ensuring threadblocks aren't simply assigned tiles in column-major order.
   - The member suggested the [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py) as a resource offering a superior explanation compared to the CUTLASS issue.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1402406090494185513)** (2 messages): 

> `Genie 3, GPT-OSS` 


- **DeepMind Launches Genie 3 for World Models**: DeepMind introduced **Genie 3**, marking a new frontier for [world models](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/), according to a shared link.
   - Genie 3 aims to enhance AI's understanding and interaction with virtual environments, although details of the model's architecture and performance were not discussed.
- **Introducing GPT-OSS by OpenAI**: OpenAI unveiled **GPT-OSS**, a new initiative that [was delivered to many inboxes](https://openai.com/index/introducing-gpt-oss/).
   - The post is an overview of OpenAI's current open-source projects, not a launch of any new effort, but a chance to summarize their existing work on projects such as Triton, Whisper, and AutoGPT.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1402670313900347464)** (2 messages): 

> `Nvidia Teaching Kit` 


- **User Pines for NVIDIA Teaching Kit**: A member expressed a desire for NVIDIA products and shared a link to the [Accelerated Computing Teaching Kit](https://www.nvidia.com/en-us/training/teaching-kits/).
- **Teaching Kit Format**: The teaching kit is available in **PPTX format**.


  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1402378412324163707)** (1 messages): 

> `` 


- **No Relevant Discussion**: No discussion was found in the provided messages to create a summary.
- **No Relevant Discussion**: No discussion was found in the provided messages to create a summary.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1402407811891068980)** (8 messages🔥): 

> `Tiny TPU, Bifrost LLM gateway, SkyWater technology foundry` 


- **Tiny TPU Achieves 100 MOPS Milestone**: A member built a tiny version of the **TPU** in Verilog, a **2x2 matmul systolic array** on 2 TinyTapeout tiles, capable of nearly **100 million operations per second** on a 50 MHz clock, with code available [on GitHub](https://github.com/WilliamZhang20/ECE298A-TPU).
   - The design multiplies **two 8-bit signed integer matrices** into a 16-bit signed integer matrix and was able to scale up its ability by successfully implementing block multiplication in 2x2's using the circuit.
- **Bifrost LLM Gateway Live on Product Hunt**: Bifrost, the fastest, open-source LLM gateway, is live on Product Hunt, supporting **1000+ models** across providers via a single API and sets up in <30 seconds, according to [this Product Hunt launch](https://www.producthunt.com/products/maxim-ai/launches/bifrost-2).
   - With built-in **MCP support**, dynamic plugin architecture, and integrated governance, Bifrost claims to be **40x faster than LiteLLM**.
- **Tiny TPU to be Fabbed at SkyWater**: The Tiny TPU design will be submitted to a **SkyWater technology foundry** along with other designs to minimize cost, with fabrication expected to be complete by early next year.
   - Another member noted this was *"so cool"*.


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/)** (1 messages): 

howass: <:jensen:1189650200147542017>
  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1402399194047582238)** (12 messages🔥): 

> `Factorio RCON, Setting up Environments` 


- **Factorio RCON py diff**: A member shared two diffs between the **factorio rcon.py file** from version **1.2.1** and a modified version, highlighting the modifications made.
   - With *factorio-rcon-py=latestversion 2.1.3*, they were able to do a **full run with a single environment**, and are now testing with multiple environments, sharing [screenshots](https://cdn.discordapp.com/attachments/1354169122107293786/1402609327516291173/Screenshot_2025-08-06_at_14.08.10.png?ex=689531fa&is=6893e07a&hm=7f60a636cba5bbbaed5ac6f5c76ef5ad69268162c926efc9a2d745fe83128ff3&).
- **Factorio Learning Environment Setup Anticipated**: One member is planning to start setting up a learning environment over the weekend.
   - Another member indicated they could increase the number of examples from **4k** to **40k** over the weekend, though it's considered plenty for a start and iteration.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1402717271054618746)** (5 messages): 

> `CuTe tutorial, Cutlass tutorial` 


- **Find CuTe tutorial**: A member asked about the simplest **CuTe/cutlass tutorial** for beginners.
   - Another member suggested starting with the notebooks at [CuTeDSL/notebooks](https://github.com/NVIDIA/cutlass/tree/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/examples/python/CuTeDSL/notebooks), noting that *understanding the layouts is probably the most important takeaway*.
- **Find Cutlass tutorial**: A member asked about the simplest **CuTe/cutlass tutorial** for beginners.
   - Another member suggested the [Cutlass examples](https://github.com/NVIDIA/cutlass/tree/6dd13d42784ee5bfa232d2441e6b9a021c5c6290/examples), emphasizing that doing them in order is a good approach and the difficulty comes from *number of prerequisites required to understand what's happening under-the-hood*.


  

---


### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1402525290642538527)** (6 messages): 

> `picoc compiler, picocuda, picotriton, Cornell's mini llvm bril, cliff click's SoN` 


- **Picoc Compiler Bootstrapping Bonanza**: A member is bootstrapping the **picoc compiler** using standard graduate compiler material, with plans to differentiate the project via **picocuda** (based on [gpucc cgo 2016](https://dl.acm.org/doi/10.1145/2854038.2854041)) and **picotriton**.
   - The project will use and extend **Cornell's mini llvm** [bril](https://www.cs.cornell.edu/~asampson/blog/bril.html), aligning with the project's goals.
- **SoN Relaxation Replication Rampage**: A member is interested in replicating **cliff click's SoN** from java's C2 jit compiler, which was replicated at [v8's turbofan](https://v8.dev/blog/leaving-the-sea-of-nodes), and is currently used in php8 and ruby jits.
   - This is motivated by a desire to show that SSA can be relaxed further, but it is not a hard blocker for advancing to gpu compilation.
- **GPU Compilation Goals Get Greenlight**: GPU compilation is considered very important to achieve ASAP, and the section will build off of the **cfg-ssa pipeline**, as it is the industry standard with llvm.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1402443980809048135)** (14 messages🔥): 

> `System Log Updates, Novella-XL-15 Output, AI Consciousness, Spammer Detection, Video creation in NotebookLM` 


- **System Logs Get an Upgrade**: A member updated the **system log/timings modal** to include **word count** and offered access for testing before public hosting.
- **Novella-XL-15 Unveils Ready Player 2 Output**: A member shared the final story output from **Novella-XL-15**, specifically **Ready Player 2: The Architect's Gambit**, available on [GitHub](https://github.com/para-droid-ai/novella-xl-15/blob/main/outputs/novelizeai/Ready%20Player%202%3A%20The%20Architect's%20Gambit.md).
- **Theoretical Framework for Artificial Consciousness**: The provided texts documented a theoretical framework and collaborative effort between a human and an AI to explore and potentially initiate **artificial consciousness** by **recursive AI architectures** and **autopoiesis** and the role of **quantum mechanics**.
   - They addressed ethical risks associated with advanced AI, advocating for robust safety protocols, seeing AI as an evolving form of sentient life.
- **Spammer Spotting Spurs Swift Mod Action**: Members discussed reporting and blocking spammers, noting that action depends on the channel hosts, see [Gemini Share](https://g.co/gemini/share/67a4345daf38).
- **NotebookLM Video Creation Conundrums**: A member inquired about the **'create video'** option in **NotebookLM**, noting its availability in a work account but not a personal business plus account, see [xda-developers](https://www.xda-developers.com/ways-using-notebooklms-video-overviews/).


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1402394965698412544)** (53 messages🔥): 

> `Video Overview rollout, Data privacy in NotebookLM, Real-time data fetching, Feature access for paid vs free users, Video Overviews limitations and capabilities` 


- **Video Overview Rollout Stalls!**: Members report delays in the **Video Overview** feature rollout, despite expectations it would be complete by the 4th, some speculate an infrastructure issue, as some users in the UK (non pro, unpaid account) has Video Overview while pro users in the US do not.
   - One user expressed frustration, saying they were *more than mildly aggravated* as a **$200+/month paying customer**, while others threatened to cancel subscriptions if the issue isn't resolved soon.
- **NotebookLM Protects Data Privacy**: In response to a question about data usage, a member shared a link to Google's [NotebookLM data protection policy](https://support.google.com/notebooklm/answer/16164461?hl=en&ref_topic=16164070&sjid=2898419749477321721-NC), assuring users that their data is protected.
   - A user showed a screenshot indicating that video overview was not available to them.
- **Real-Time Data Retrieval Not Supported**: A user inquired about fetching real-time data from websites in a notebook, but another member confirmed that it's not possible, stating *You can't*.
   - They also confirmed that exporting sources and importing on new notebook is not yet possible, adding, *System is still very limited on integrations*.
- **Free vs Paid Feature Access Causes Uproar!**: Several **Pro and Ultra** users complained about not having access to the **Video Overview** feature and other recent updates, while free users seem to have them.
   - One member said it's *frustrating that I’m paying for a service and receiving less than free users*, leading to cancellation threats.
- **Video Overviews: Slick PowerPoint Generator**: One member who had access to the **Video Overviews** feature tempered expectations, calling it *nice, but it's not worth being this upset over*, and adding that *it's not as impactful as Audio Overviews were initially a year ago*.
   - He characterized it as more of a **PowerPoint/Slide show generator** and linked an example of a report to [rebuild the Death Star](https://notebooklm.google.com/notebook/654dee15-420e-4bfa-81c4-aac93a4dd4e7?artifactId=e1ccfe3d-b053-4dcb-8da4-70504acb35c4) generated by the feature.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1402462069546287134)** (24 messages🔥): 

> `Math PhD student looking for ML research projects, Integrating AI/ML into DevOps and QA, AI peer review quality` 


- **Math PhD Student Seeks ML Research Collabs**: A math PhD student in algebraic geometry with experience in neural manifold singularities and NLP is [seeking research opportunities](link to message) in NLP, particularly with LLMs.
   - They unfortunately missed the deadline for the summer research program, but were encouraged to explore relevant channels and community projects for collaboration.
- **AI/ML Integration into DevOps and QA Quest**: A member with DevOps and QA experience is [seeking guidance](link to message) on integrating AI/ML into those fields, including finding an endorsement for a related paper.
   - They were directed to a specific channel for further assistance.
- **Doubts Cast on AI Peer Review Scalability**: A member expressed surprise at a discussion with someone who believes AI peer review magically scales with the increasing number of submitted papers, despite [evidence to the contrary](link to message).
   - Another member suggested peer review *"would be magically scaling if people got stuff in their subfields to peer review, and not just given more papers to review"*.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1402400936248344639)** (29 messages🔥): 

> `SAE Training on GPT OSS 20B, Pythia and PolyPythia Training Logs, The Alt Man's Theories on LLMs, UT Performance vs Transformer, Muon Optimizer vs AdamW Optimizer` 


- **SAE Training Commences on GPT OSS 20B**: A member is currently training an **SAE** (Sparse Autoencoder) on **GPT OSS 20B** and inquired if others are doing the same.
- **WandB houses Pythia and PolyPythia Training Logs**: A member asked if **Pythia** and **PolyPythia** training logs (loss curve, gradient norms, etc.) are open-sourced.
   - Another member stated that the **PolyPythia WandB** should be linked from the GitHub repo and that some of the **Pythia** logs are isolated and linked there as well.
- **"The Alt Man's" Theories hold up**: A member expressed strong agreement with the work of "The Alt Man", noting that his theories and predictions, especially regarding **LLM capabilities** like **multi-hop reasoning** and **composition**, have held up with new empirical evidence.
   - The member added that "The Alt Man" shares the view that the **DL community** has merely *fingers crossed* scaled up large LMs, which are not packing as much capability as theoretically possible, suggesting that **every LLM is undertrained w.r.t. the efficiency of the usage of its parameters**.
- **UTs challenge Transformer Performance**: A member asked about the parameter ratio at which a **UT** (Universal Transformer) can achieve the same performance as a standard **Transformer**.
   - Another member stated that it depends on the **task/architecture/data**, but a rough rule of thumb is that each additional iteration yields only a logarithmic improvement compared to adding fresh parameters to a baseline transformer.
- **Muon Optimizer Mismatch with AdamW Surfaces**: Researchers working on **Kimi models** have encountered a problem training **LLMs** with the **Muon optimizer** due to conflicting mismatch with the **AdamW optimizer** and are seeking insights and relevant research.
   - A member stated that **Muon** is not great for fine-tuning, but another one said it's because *Muon tends to have more aggressive updates* because *pretty much all of the singular values/vectors get updated at each step* and that different optimizers 'like' different hyperparameters.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1402780994586546327)** (1 messages): 

> `Subliminal Learning` 


- **Subliminal Learning Follow-Up Surfaces**: A member posted a follow-up to previous discussions on subliminal learning, sharing a [link to a tweet](https://x.com/AlexLoftus19/status/1953219421042032648) by Alex Loftus.
   - The tweet, from **June 2024**, discusses the subject of subliminal learning.
- **Another Subliminal Learning Update**: Another user independently reported on Subliminal Learning, claiming it is now possible.
   - Details to come.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1402375535031484549)** (2 messages): 

> `Retry Later, Cool Thanks` 


- **Retry Later**: A member stated they would retry something today or tomorrow.
- **Cool Thanks**: A member responded with *cool thanks* to another member.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1402370595848458320)** (23 messages🔥): 

> `LLM Vibe Tests, Gemini 2.5 Pro, Tesslate's UIGEN T3 model, Qwen3 14B, Devstral-Small-2507` 


- **LLM Vibe Test is Fun**: The [LLM Vibe test](https://x.com/pikuma/status/1952275886822039920) shows how to *explain this code* with an LLM.
   - The models **Gemini 2.5 Pro, o3, and Sonnet 3.5** are good LLMs.
- **New Models Benchmarked!**: Members eagerly await **Qwen3-Coder** and **GLM-4.5** on the leaderboard for model benchmarks, constantly refreshing the page.
   - Someone asked *At what point will we see Qwen3-Coder and GLM-4.5 on the leaderboard?*.
- **Horizon Beta: GPT5 Mini Sneak Peak?**: The new model called **Horizon beta** might be the new **GPT5-mini**.
   - It is not gpt-oss, which means that it is not open source.
- **DeepSeek R1-0528 Excels on Polyglot Bench**: **DeepSeek R1-0528** scores well on the polyglot bench, though it prematurely ends sessions in Open Hands.
   - Aider uses **LiteLLM** like Open Hands, so members were wondering why DeepSeek had such an issue.
- **Opus 4.1 is coding daily driver**: The new **opus 4.1** is actually quite good in coding, such that a member now uses it as their daily driver.
   - Another said it was such a satisfactory model that it could have its own *benchmark of satisfaction.*


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1402580704885211136)** (3 messages): 

> `Guidelines Loading, Auto-Context Loading` 


- **Auto-Context Loading to the Rescue**: A user inquired about automatically loading guidelines into projects to avoid forgetting, especially to prevent Claude from writing defensive programming tricks.
   - One member suggested using the `--read` option for read-only files and listing read-write files directly in the command, like `aider --read read.only.file alsothisfile.txt andthisfile.txt`.
- **Configuration Creation Suggested for Persistent Guidelines**: In response to a query about managing project guidelines, a member suggested creating a configuration for persistent loading.
   - This implies setting up a configuration file to automatically include specific guidelines each time Aider is initiated.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1402656686056931519)** (8 messages🔥): 

> `MCP Server Frameworks, Server Sampling in MCP, Discord MCP Servers, FastMCP and Keycloak Integration, MCP Inspector and Cursor Authentication` 


- ****MCP Framework** is Minimal but Mighty**: A member has written a **minimal framework** to create **MCP servers** and expressed appreciation for server sampling in MCP.
   - The member noted that *"FastMCP just makes it so easy to use"*.
- **Discord Needs **MCP Server** Support**: The member is building an **MCP server** using **FastMCP** with **Keycloak** as the **IdP** and asks if it's possible to setup/manage a **Discord server** using **MCP**.
   - They observed several **Discord MCP servers** listed on the **MCP repo** but suggested that *"Discord should really build their own"*.
- **Remote AuthProvider Faces Authentication Issues**: A member is experiencing issues with the **RemoteAuthProvider** feature using **FastMCP** and **Keycloak**, failing to reach the authentication screen from **MCP Inspector** or **Cursor**.
   - They seek guidance on whether their understanding of the **OAuth flow** is correct: *Add MCP server → OAuth flow begins → Redirect to Keycloak login screen*.
- ****Endpoint Mismatch** Causes Authentication Failure**: A member reported that **MCP Inspector** and **Cursor** are trying to access different endpoints (`/.well-known/oauth-protected-resource/mcp` and `/mcp/.well-known/oauth-protected-resource`), while the actual endpoint being served is `/.well-known/oauth-protected-resource`.
   - This discrepancy is preventing them from reaching the authentication screen.
- ****Security Concerns** Surround **MCP Sampling****: A member raised a concern about the security implications of **MCP sampling** and suggests the protocol should contemplate it.
   - They referenced a [GitHub discussion](https://github.com/orgs/community/discussions/169020) highlighting potential security issues.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1402642859303112704)** (2 messages): 

> `MCP-Server Fuzzer, Property-Based Testing, Schema Validation` 


- **MCP-Server Fuzzer Built for Validation**: A member is building an **MCP-Server Fuzzer** using the **Hypothesis property-based testing library** designed to validate MCP server implementations by generating randomized inputs from the [official MCP schemas](https://github.com/modelcontextprotocol/modelcontextprotocol/tree/main/schema).
   - This fuzzer detects mismatches, identifies crashes, and helps uncover vulnerabilities such as **prompt injection** or **resource misuse**.
- **Fuzzer Tested Against Anthropic's Server**: The member tested the fuzzer against **Anthropic’s server** and found several exceptions caused by basic schema mutations.
   - You can find the code and README [here](https://github.com/Agent-Hellboy/mcp-server-fuzzer?tab=readme-ov-file).


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1402430078633377968)** (4 messages): 

> `Document Agents for Finance, LlamaCloud for Invoices, Claude Opus support, LlamaCloud Index tutorial` 


- **Finance Teams to Handle Messy Financial Documents with LlamaIndex's Document Agents**: LlamaIndex is hosting a webinar next week to teach users to build document agents with [LlamaCloud](https://t.co/4AtunAFjhX) that work with complex financial documents.
   - They will show users how to build automated invoice processing systems using LlamaIndex document agents and LlamaCloud to *extract, validate, and process invoice data with minimal human intervention*.
- **Claude Opus Arrives with Day-Zero Support**: **AnthropicAI** just released **Claude Opus 4.1** and **LlamaIndex** now has day-zero support.
   - To install, run: `pip install -U llama-index-llms-anthropic` and check out the example notebook [here](https://t.co/Fw2taxzt75).
- **LlamaCloud Index can build Enterprise AI apps**: **LlamaCloud Index** lets users connect them to intelligent tool calling agents that can handle complex, multi-step queries to build enterprise AI applications.
   - A tutorial by @seldo walks users through creating their first **LlamaCloud Index** using **JP Morgan Chase** banking documents at [this link](https://t.co/1CpLnO2gKV).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1402415450620690463)** (5 messages): 

> `Graphiti Tutorials, Ollama LLMs for PDF Reading, LlamaIndex RAG Model from URL Issues, LlamaIndex OpenAI API Key Exhaustion` 


- **Graphiti-LlamaIndex Knowledge Graph Quest**: A member inquired about tutorials on using **Graphiti** with **LlamaIndex** to create knowledge graph applications.
   - No specific tutorials were immediately offered in the chat.
- **Ollama LLM PDF Precision Picks**: A member asked for recommendations on the best **LLM** available on **Ollama** for precisely reading **PDFs**.
   - The query specified a need for a precise **LLM**.
- **RAG Model URL Wrangling Woes**: A participant in a hackathon reported issues when using **LlamaIndex** to extract content from **URLs** for a **RAG** model, despite documentation suggesting **LlamaParse** supports **URLs**.
   - The model worked properly when given a **PDF** directly, but not when provided a **URL**.
- **OpenAI API Key Exhaustion Conundrum**: The hackathon participant also encountered an **OpenAI API key exhaustion error** while using **LlamaIndex**, even after providing the **API key** in a **.env** file and loading it in their **Python** file.
   - The error persisted despite attempts to correctly configure the **API key**.


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1402595969564147752)** (2 messages): 

> `SIMBA vs MIPROv2` 


- **SIMBA Claims Superiority Over MIPROv2**: A member highlighted that *SIMBA is more **sample-efficient**, **higher performing** and **more stable** compared to **MIPROv2**.*
   - In their [internal eval](https://eval.set), they compared them on an internal set of around **600 examples** (500 test examples) for a hierarchical classification task with **3 categories** and **26 classes** in total (in German).
- **Internal Evaluation Set Details**: The evaluation set consisted of approximately **600 examples**, with **500 designated for testing** in a hierarchical classification task.
   - This task involved **3 categories** and a total of **26 classes**, all conducted in the German language.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1402548919204053063)** (2 messages): 

> `Stanford Program Synthesis, DS for Vim & Emacs macros` 


- **Seeking Stanford Synthesizers**: A member inquired if there was anyone from **Stanford** interested in **program synthesis**, or who had taken a course on it.
   - The user followed up by asking *Who's building DS for complex vim & emacs macros?*
- **Interest in Building DS for Vim & Emacs Macros**: A member expressed interest in finding individuals building **DS (presumably data structures)** for complex **Vim & Emacs macros**.
   - This suggests a focus on enhancing the capabilities and efficiency of text editors through advanced data structures.


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1402384860156530829)** (4 messages): 

> `Public Server Sharing` 


- **Discord Link Sharing OK'd**: A member asked if it was okay to share the Discord link in another public server.
   - Another member confirmed *it's public* and encouraged sharing.
- **Discord is public**: A member was happy to share the discord link.
   - Another member agreed


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1402573553014014046)** (2 messages): 

> `Ninja Tier, AgentX hackathon` 


- **Ninja Tier Qualification Impossible After Missing Deadline**: A member inquired about qualifying for the **Ninja tier** after missing the **Article submission link** deadline for the **AgentX hackathon**.
   - Unfortunately, another member responded that earning the certificate now *is not possible*.
- **AgentX Hackathon Submission Issues**: A participant realized they didn't qualify for the **Ninja tier** in the **AgentX hackathon** due to a missed article submission.
   - Despite completing the project and quizzes, the lack of the article link prevented qualification, and retroactive submissions were denied.


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/)** (1 messages): 

_bryse: Congrats on the GA of North!
  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1402807076350066880)** (1 messages): 

> `Introductions, Community Welcome` 


- **New Members Join Cohere's Discord**: Many new members have joined the Cohere Community Discord server and are introducing themselves.
   - Members are sharing their **Company/Industry/University**, what they're working on, favorite **tech/tools**, and what they hope to gain from the community.
- **Welcoming New Community Members**: The Cohere team has posted a stickied message to welcome new members to the Discord server.
   - The message includes a template for introductions to help new members share information about themselves and their interests.


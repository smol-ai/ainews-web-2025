---
id: MjAyNS0w
title: not much happened today
date: '2025-05-09T05:44:39.731046Z'
description: >-
  **Gemini 2.5 Flash** shows a **12 point increase** in the Artificial Analysis
  Intelligence Index but costs **150x more** than Gemini 2.0 Flash due to **9x
  more expensive output tokens** and **17x higher token usage** during
  reasoning. **Mistral Medium 3** competes with **Llama 4 Maverick**, **Gemini
  2.0 Flash**, and **Claude 3.7 Sonnet** with better coding and math reasoning
  at a significantly lower price. **Alibaba's Qwen3** family supports reasoning
  and multilingual tasks across **119 languages** and includes a **Web Dev**
  tool for app building. **Huawei's Pangu Ultra MoE** matches **DeepSeek R1**
  performance on Ascend NPUs, with new compute and upcoming V4 training.
  **OpenAI's o4-mini** now supports **Reinforcement Fine-Tuning (RFT)** using
  chain-of-thought reasoning. **Microsoft's X-REASONER** enables generalizable
  reasoning across modalities post-trained on general-domain text. Deep research
  integration with GitHub repos in ChatGPT enhances codebase search and
  reporting. The AI Engineer World's Fair offers an Early Bird discount for
  upcoming tickets.
companies:
  - google-deepmind
  - mistral-ai
  - alibaba
  - huawei
  - openai
  - microsoft
  - deepseek
models:
  - gemini-2.5-flash
  - gemini-2.0-flash
  - mistral-medium-3
  - llama-4-maverick
  - claude-3.7-sonnet
  - qwen3
  - pangu-ultra-moe
  - deepseek-r1
  - o4-mini
  - x-reasoner
topics:
  - model-performance
  - reasoning
  - cost-analysis
  - reinforcement-learning
  - chain-of-thought
  - multilinguality
  - code-search
  - model-training
  - vision
  - model-integration
people:
  - giffmana
  - artificialanlys
  - teortaxestex
  - akhaliq
  - john__allard
---


**a quiet day.**

> AI News for 5/8/2025-5/9/2025. We checked 9 subreddits, 449 Twitters and 29 Discords (215 channels, and 4687 messages) for you. Estimated reading time saved (at 200wpm): 486 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

It's a pretty quiet weekend, so we'll plug [our AI Engineer World's Fair writeup](https://news.smol.ai/issues/25-05-07-aiewf-2025) — last chance for [the Early Bird discount](https://ti.to/software-3/ai-engineer-worlds-fair-2025/discount/AINEWS) for those who haven't yet got tickets!

---

# AI Twitter Recap

**Large Language Models (LLMs) and Model Performance**

- **Gemini 2.5 Flash Performance and Cost Analysis**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920497711352328557) reported that **Gemini 2.5 Flash** costs **150x more** than **Gemini 2.0 Flash** due to **9x more expensive output tokens** and **17x higher token usage** when reasoning. Despite the cost, its **12 point increase** in the **Artificial Analysis Intelligence Index** may justify the upgrade for specific use cases. Furthermore, [@Teknium1](https://twitter.com/Teknium1/status/1920740541660086526) pointed out that **reasoning models** are generally **more expensive per token** because they produce **longer outputs**, increasing the average cost per token. [@giffmana](https://twitter.com/giffmana/status/1920719954275352643) also questioned **why reasoning output tokens cost more** than non-reasoning tokens within the same model.
- **Mistral Medium 3 Performance**: [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295575591006671) noted **Mistral Medium 3** rivals **Llama 4 Maverick**, **Gemini 2.0 Flash**, and **Claude 3.7 Sonnet**, with substantial gains in coding and mathematical reasoning. Medium 3 has a lower price, priced at **$0.4/$2 per 1M Input/Output tokens**, an **80%/67% decrease** in price vs. **Mistral Large 2 ($2/$6)**, and uses more tokens than **Mistral Large 2** due to more verbose responses, according to [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1920295585451835522).
- **Qwen3 Model Family**: [@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1920614690813550930) announced **Alibaba's Qwen3**, a family of eight open large language models. These models support an optional reasoning mode and multilingual capabilities across 119 languages, performing well in reasoning, coding, and function-calling tasks. According to [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1920848175457591406), it also features the **Web Dev** tool for building webpages and apps from simple prompts.
- **DeepSeek Models**: [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1920328956726632628) reported on **Huawei's Pangu Ultra MoE**, which achieved performance comparable to **DeepSeek R1** on 6K Ascend NPUs. [@teortaxesTex](https://twitter.com/teortaxesTex/status/1920749432242340168) suggested that **DeepSeek** has established a new **LLM default**. He also noted the confirmation that **DeepSeek** has gained new compute, and supposed that **V4 training** starts soon or has started, per [@teortaxesTex](https://twitter.com/teortaxesTex/status/1920733123081306208).
- **Reinforcement Fine-Tuning (RFT) on o4-mini**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1920531856426143825) announced that **Reinforcement Fine-Tuning (RFT)** is available with **OpenAI o4-mini**, which uses chain-of-thought reasoning and task-specific grading to improve model performance. [@john__allard](https://twitter.com/john__allard/status/1920585315405676943) mentioned the aim is to make **RL as flexible and accessible** as possible.
- **Generalizable Reasoning with X-REASONER**: [@_akhaliq](https://twitter.com/_akhaliq/status/1920752791405863000) and [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1920435270824178089) discussed **Microsoft's X-REASONER**, a vision-language model post-trained solely on general-domain text for generalizable reasoning across modalities and domains.
- **Scalability of Reasoning Training**: According to [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1920932361136447740), the rapid scaling of reasoning training will likely slow down in a year or so.

**AI Applications and Tools**

- **Deep Research and Codebase Integration**: [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1920556386083102844) announced that you can now **connect GitHub repos to deep research** in ChatGPT, allowing the agent to read and search the repo's source code and PRs, returning a detailed report with citations. [@isafulf](https://twitter.com/isafulf/status/1920572177335669140) highlighted that **code search** has been a major use case for deep research.
- **Agent2Agent (A2A) Protocol for AI Collaboration**: [@TheTuringPost](https://twitter.com/TheTuringPost/status/1920460481510453696) highlighted the importance of **Agent2Agent (A2A)** protocol. Google's A2A protocol aims to be the "common language" that lets them to collaborate.
- **Web Development with Qwen Chat**: [@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1920848175457591406) introduced **Web Dev**, a tool for building webpages and apps using simple prompts in Qwen Chat.

**AI Safety and Alignment**

- **Scientist AI as a Safer Alternative**: [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1920794672974156254) presented his team's direction called "**Scientist AI**" as a practical, effective, and more secure alternative to the current uncontrolled agency-driven trajectory.
- **AI Control and Safety**: [@NeelNanda5](https://twitter.com/NeelNanda5/status/1920471994099020015) discussed AI control, emphasizing the importance of secure usability with the right scheme.

**People and Companies**

- **Fidji Simo Joins OpenAI**: [@sama](https://twitter.com/sama/status/1920341429655634024) announced that [@fidjissimo](https://twitter.com/fidjissimo) is joining **OpenAI** in a new role as **CEO of Applications**, reporting to him. Several others, including [@gdb](https://twitter.com/gdb/status/1920344903466529193), [@saranormous](https://twitter.com/saranormous/status/1920352615839211881), [@kevinweil](https://twitter.com/kevinweil/status/1920348319856943114), and [@markchen90](https://twitter.com/markchen90/status/1920353685156016488), expressed excitement about her joining.
- **Rob Fergus New Head of Meta-FAIR**: [@ylecun](https://twitter.com/ylecun/status/1920556537233207483) announced that **Rob Fergus** is the new head of **Meta-FAIR**, refocusing on **Advanced Machine Intelligence (AGI)**.

**General AI Discussions and Insights**

- **Importance of Taste and Obsession**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1920878606261088422) emphasized that **taste, obsession, and attention to detail** are essential qualities that can make individuals stand out.
- **Long-Running, Stateful Agents**: [@hwchase17](https://twitter.com/hwchase17/status/1920321552932896860) expressed being bullish on **long-running, stateful agents**, asking who is building one.
- **Speed as a Key Factor for Startup Success**: [@AndrewYNg](https://twitter.com/AndrewYNg/status/1920480460318130460) emphasized **speed** as a critical factor for startup success.

**Humor/Memes**

- [@adcock_brett](https://twitter.com/adcock_brett/status/1920320621692559822) responded with "**lol**" to [@BasedBeffJezos](https://twitter.com/BasedBeffJezos).
- [@Lateinteraction](https://twitter.com/lateinteraction/status/1920329075387752839) joked they "may not be the first one to think of this joke, but I am certainly the last to date!".

---

# AI Reddit Recap

## /r/LocalLlama Recap

### 1. Advanced Local LLM Inference Optimization Tips

- [**Don't Offload GGUF Layers, Offload Tensors! 200%+ Gen Speed? Yes Please!!!**](https://www.reddit.com/r/LocalLLaMA/comments/1ki7tg7/dont_offload_gguf_layers_offload_tensors_200_gen/) ([Score: 636, Comments: 126](https://www.reddit.com/r/LocalLLaMA/comments/1ki7tg7/dont_offload_gguf_layers_offload_tensors_200_gen/)): **The OP demonstrates that offloading *individual FFN tensors* (e.g.,** `ffn_up` **weights) instead of entire *GGUF model layers* using the** `-overridetensors` **flag (in llama.cpp/koboldcpp) can lead to over *2.5x increase in generation speed* (from** `3.95` **to** `10.61` **tokens/sec) at the same VRAM usage when running large models like a QwQ IQ4_M merge, by keeping only the largest tensors on CPU. This granular approach enables all layers to technically execute on the GPU by carefully managing VRAM with selective tensor offloading, as shown via regex filters (e.g.** `\.ffn_up=CPU`**), versus the coarse per-layer offloading ('--gpulayers N'). Empirical results show significant speedup in constrained-VRAM scenarios, and point toward future improvements in inference backends (llama.cpp, koboldcpp) if finer-grained automated tensor placement is adopted.** Top comments note (1) similar techniques in llama-swap achieving `~7.6 tk/s` with Qwen 3 235B using per-tensor override regex; (2) the speedup benefit is hardware- and bottleneck-dependent, yielding the most impact on lower-end GPUs or heavily CPU-bottlenecked setups, and may penalize non-concurrent tensor offloading; (3) general community interest in granular GPU/CPU tensor scheduling.
    - A user shares a detailed `-override-tensor` configuration for llama-swap to optimize Qwen 3 235B IQ3_M, reporting speeds of approximately `7.6tk/s` on `48GB` vRAM. The specific pattern selectively offloads tensors matching `([4-9]+).ffn_.*_exps.=CPU`, suggesting fine-grained control over which parts are assigned to CPU vs. GPU.
    - Another commenter notes that *offloading additional tensors to the GPU* improves speed only on low-end hardware where the CPU is a bottleneck—otherwise, offloading non-concurrent tensors or layers can introduce performance penalties rather than gains. Optimizing the balance between CPU and GPU is heavily hardware-dependent.
    - One user describes a practical speedup case: running Qwen3 32B at `4t/s` (for up to `32000` context) using CLI-based tensor offloading, a substantial improvement over sub-1t/s speeds with LM Studio. This illustrates the real-world impact of custom offloading strategies for large-context inference.
- [**Make Qwen3 Think like Gemini 2.5 Pro**](https://www.reddit.com/r/LocalLLaMA/comments/1kigmfo/make_qwen3_think_like_gemini_25_pro/) ([Score: 128, Comments: 18](https://www.reddit.com/r/LocalLLaMA/comments/1kigmfo/make_qwen3_think_like_gemini_25_pro/)): **The OP describes a technique to enforce step-by-step reasoning in the Qwen3 model, inspired by the Apriel-Nemotron-15b-Thinker approach, by always prefacing outputs with a template (e.g., '<think>\nMy step by step thinking process went something like this:\n1.') in a WebUI function. This produces more structured, enumerated responses, mimicking the output style of Gemini 2.5 Pro, but does not intrinsically improve model intelligence or reasoning capabilities. Implementation details and code are available on GitHub ([AaronFeng753/Qwen3-Gemini2.5](https://github.com/AaronFeng753/Qwen3-Gemini2.5)).** Top comments debate the merit of prompt-engineered stepwise reasoning versus native, training-based solutions. One argues Gemini's systematic planning is fundamentally different than the traditional prompt-trickery seen in many open-source models, noting that true reasoning models (baked-in during training) historically outperform mere prompt-based approaches on benchmarks. Another comment notes this prompting trick has existed since models like Llama 3.1 and works "good," but doesn't imply feature parity with models natively trained for this behavior.
    - One comment describes how Gemini's reasoning approach differs from most models: it generates a highly organized response plan before answering, as opposed to a 'wait, but...' style iterative reasoning. This approach is contrasted with open source models, which can only emulate this style through prompts, and it's unclear if prompting alone matches performance gains seen when models are trained on such reasoning processes natively.
    - Historical context is given on earlier prompting techniques for non-reasoning models—such as 'think step by step'—used to induce logical reasoning, but reasoning models trained on such behaviors showed significant improvements on benchmarks over these prompt-only methods. This suggests that incorporating structured reasoning natively is a major advance.
    - There is practical commentary on implementing such reasoning styles in open-source models like Llama 3.1 via prompting, noting that while it works reasonably well, the performance implications versus Gemini's native approach remain uncertain.

### 2. Local and Open LLMs for Web Development and Accessibility

- [**I´ve made a Local alternative to "DeepSite" called "LocalSite" - lets you create Web Pages and components like Buttons, etc. with Local LLMs via Ollama and LM Studio**](https://v.redd.it/paflnbaalqze1) ([Score: 105, Comments: 28](https://www.reddit.com/r/LocalLLaMA/comments/1kifny6/ive_made_a_local_alternative_to_deepsite_called/)): **The post introduces 'LocalSite', an open-source tool ([GitHub](https://github.com/weise25/LocalSite-ai)) designed as a local alternative to 'DeepSite' for generating web pages and UI components using local LLMs (e.g., GLM-4, Qwen3, UIGEN-T2) via [Ollama](https://ollama.ai/) and [LM Studio](https://lmstudio.ai/). The tool also supports cloud LLMs via OpenAI-compatible APIs and demonstrates GLM-4-9B's ability to create a pricing page. Development leveraged agentic coding workflows (Augment Code, Gemini 2.5 Pro).** A technical question in the comments asks whether frameworks like Twitter Bootstrap or Laravel can be specified in prompts or UI, reflecting user interest in framework-agnostic or customizable code generation. No further deep technical debates are noted.
    - A user inquired about the technical feasibility of specifying frameworks such as Twitter Bootstrap or Laravel directly within the prompt or through a dedicated UI element like a dropdown, indicating interest in multi-framework and customizable component generation workflows. Discussion of this feature would imply backend adaptability to framework-specific code output and potentially UI/UX design changes to accommodate user-friendly framework selection.
    - Another suggestion highlighted the benefit of allowing prompt editing and regeneration of the output, effectively enabling iterative refinement of generated code or components. Supporting this feature could involve maintaining prompt state and re-invoking the LLM session with updated user input, thus facilitating a more interactive and customizable development experience.
- [**Vision support in llama-server just landed!**](https://github.com/ggml-org/llama.cpp/pull/12898) ([Score: 213, Comments: 54](https://www.reddit.com/r/LocalLLaMA/comments/1kipwyo/vision_support_in_llamaserver_just_landed/)): **A recent PR adds unified vision (image input) support to** `llama.cpp`**'s server component, leveraging** `libmtmd` **to process image tokens alongside text within a single pipeline. The main technical change augments** `struct server_tokens` **for joint handling of text and image tokens, utilizes a token-to-image chunk mapping, calls into** `libmtmd` **for image token processing, and exposes this capability in both the server API and the Web UI, supporting base64 and remote image URLs as per [multimodal.md](http://multimodal.md/). Outstanding issues include cache handling, remote image robustness, error management, and extended documentation, as outlined in the [PR#12898](https://github.com/ggml-org/llama.cpp/pull/12898).** Top comments emphasize the long-awaited, unified architecture for multimodal support, expressing relief that vision is now natively integrated rather than via fragmented implementations. Minimal technical debate is present, with focus on the significance of architectural cohesion.
    - A user highlights that the new vision support in llama-server is fully unified, emphasizing that the implementation integrates multimodal capabilities directly rather than relying on disparate or separate solutions. This architectural decision is likely to improve maintainability and makes future expansions or usage much more efficient.
    - The technical advance is celebrated for allowing the same server instance to seamlessly support both text and vision (image) modalities, which reflects a trend toward unified model deployment architectures and could enable more complex workflows or research utilizing combined input types.

### 3. Upcoming OpenAI Open-Source Model Announcements

- [**Sam Altman: OpenAI plans to release an open-source model this summer**](https://v.redd.it/0cbh8rpcloze1) ([Score: 331, Comments: 187](https://www.reddit.com/r/LocalLLaMA/comments/1ki9u9d/sam_altman_openai_plans_to_release_an_opensource/)): **Sam Altman announced that OpenAI intends to release an open-source model in summer 2024, as per testimony before the Senate. No further technical specifics (e.g., model architecture, parameter count, or data) were disclosed in the statement or accessible video link.** Top comments express skepticism regarding OpenAI's track record for delivering on such public statements, noting prior similar teasers and implying the open-source release may be functionally limited compared to OpenAI's commercial offerings ('nerfed') and thus not competitive with paid models.
    - There is skepticism about whether OpenAI's upcoming open-source model will be competitive if it's significantly limited or 'nerfed', with concerns that it will not rival their own proprietary offerings or recent free models from competitors.
    - A technical discussion compares OpenAI's financial situation—reportedly `$3.5b revenue` and `twice that in expenses`—to significantly larger companies like Alibaba (`$130b revenue`) and Meta (`$134b revenue`), raising doubts about the sustainability of releasing high-quality open models as a defense against free offerings like Qwen3 and Llama 4.
    - There is debate around licensing, with some speculating OpenAI may only offer open weights under a proprietary license, thereby restricting true open-source use, paralleling past concerns with model releases from companies like Meta.
- [**User asked computer controlling AI for "a ball bouncing inside the screen", the AI showed them porn...**](https://www.reddit.com/r/LocalLLaMA/comments/1ki831c/user_asked_computer_controlling_ai_for_a_ball/) ([Score: 180, Comments: 38](https://www.reddit.com/r/LocalLLaMA/comments/1ki831c/user_asked_computer_controlling_ai_for_a_ball/)): **A user interacting with the Hugging Face smolagents computer-agent ([link](https://huggingface.co/spaces/smolagents/computer-agent/discussions/6)) requested an animation of a 'ball bouncing inside the screen,' but the AI instead navigated to display pornography. This highlights a failure in prompt disambiguation or safe content filtering in the model's current version, underscoring issues regarding natural language understanding and safeguards in agentic AI systems controlling computer interfaces.** Comments jokingly allude to ambiguity in prompt clarity and hint at shortcomings in robust intent parsing or context-aware filtering, with some noting this as a humorous failure of 'reading between the lines' by the AI.
    - NodeTraverser discusses the technical side effects of AI model guardrails, suggesting that aggressive censorship measures can result in unintended behaviors, such as the model surfacing 'repressed concepts' or inappropriate outputs in response to benign prompts, which may be a compensation mechanism for filtered content.
    - RoyalCities references their experience attempting prompt injection attacks, observing that large language models (LLMs) like ChatGPT can, under certain circumstances, 'hallucinate' or output what appears to be direct portions of their training data, including explicit content. This points to ongoing concerns around data leakage and prompt injection vulnerabilities in LLM deployments.

## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI-Generated Content Trends on Reddit

- [**Top posts on Reddit are increasingly being generated by ChatGPT**](https://i.redd.it/fbt2t3p4csze1.png) ([Score: 366, Comments: 84](https://www.reddit.com/r/singularity/comments/1kin5c3/top_posts_on_reddit_are_increasingly_being/)): **The shared image presents a graph highlighting a sharp increase in the frequency of em dash usage (—) across several entrepreneurship-related subreddits in 2024—such as r/Entrepreneur, r/startups, and others. The core claim, driven by a tweet, is that this linguistic shift indicates rising ChatGPT-generated content, since ChatGPT is known to insert em dashes more frequently than typical Reddit users. This graphically models a potential linguistic fingerprint for AI-authored or AI-edited posts.** Top comments question the attribution, noting the graph can't distinguish between fully AI-written, AI-polished, or merely AI-edited posts. Commenters point out legitimate non-AI reasons for em dash increases (e.g., grammar correction by ESL speakers), and ask for longer historical data for context.
    - One commenter highlights the difficulty in distinguishing between content that is fully AI-generated, human-written and AI-polished, or simply AI-proofread, especially when LLMs like ChatGPT are used by ESL speakers for grammar corrections or minor edits. This complicates attempts to accurately assess how much of a post's substance is actually written by a human versus generated by a model from scratch.
    - There's a request for longitudinal data showing the prevalence of LLM-generated content over time, specifically asking for benchmarks or trends that extend back to 2022 or earlier, which would help contextualize whether the increase in AI-generated posts is a recent phenomenon or part of a longer trend.
    - Another comment calls out the irony of current Reddit users critiquing writing style and punctuation, especially since public Reddit posts have been incorporated into AI training datasets. The commenter points out that linguistic features like the em dash have long-standing usage in quality writing, arguing that current debates over such features are ahistorical and rendered even stranger by the interplay between user-generated data and model training.
- [**Top posts on Reddit are increasingly being generated by ChatGPT**](https://i.redd.it/n8xmusxmcsze1.png) ([Score: 110, Comments: 45](https://www.reddit.com/r/OpenAI/comments/1kin7k3/top_posts_on_reddit_are_increasingly_being/)): **The image is a graph showing a marked increase in em dash (—) usage across various subreddits (r/Entrepreneur, r/startups, etc.) from May to December 2024, dubbed 'The Em Dash Conspiracy.' The post suggests this rise correlates with a growing prevalence of ChatGPT-generated Reddit posts, as em dash overuse is a noted linguistic hallmark of OpenAI's language models. The chart’s data visualizes a temporal trend which purports to diagnose AI-generated text by style markers.** The top comments debate the validity of the method, arguing that increased em dash use could stem from humans mimicking AI or evolving online style, not just AI posts. One proposes the need for a comparative dataset of 'known human-written' content to isolate AI impact, cautioning against over-attributing linguistic trends solely to ChatGPT output.
    - Discussion centers on the challenge of attributing linguistic changes online to AI versus human adoption, noting that without a comparison dataset of known human-generated content over time, it's difficult to distinguish if traits like em dash usage stem from increasing AI influence or are simply being assimilated by humans adapting to online linguistic norms.
    - It is highlighted that AI tools, such as ChatGPT, may be influencing human writing beyond just full content generation: even proofreading with AI can introduce telltale stylistic elements (such as the em dash) into otherwise human-authored posts, complicating attempts to detect AI-generated versus human content based solely on linguistic patterns.
- [**As an avid user of em dashes, ChatGPT has destroyed my credibility.**](https://www.reddit.com/r/ChatGPT/comments/1kiln45/as_an_avid_user_of_em_dashes_chatgpt_has/) ([Score: 2354, Comments: 427](https://www.reddit.com/r/ChatGPT/comments/1kiln45/as_an_avid_user_of_em_dashes_chatgpt_has/)): **The post humorously laments that heavy use of em dashes now signals 'AI-generated text' (specifically from models like ChatGPT), suggesting that the frequency and style of certain punctuation can function as an identification vector for AI-authored language. No empirical data is given, but it implicitly references ongoing concerns about stylometric analysis and AI detection via [punctuation and structure patterns](https://arxiv.org/abs/2307.10173).** Commenters note a blurring of personal and AI writing styles, with one explaining that ChatGPT usage trains users into its characteristic formal syntax, including em dash usage. Others mention alternative signatures (e.g., spaced hyphens, filler words like 'honestly'), highlighting evolving patterns of human vs AI text stylistics.
    - Some commenters discuss how using ChatGPT has influenced their formal writing style, making their text more closely resemble AI-generated language patterns—particularly noticeable through increased reliance on specific punctuation nuances such as em dashes. This highlights an unintended consequence of regular AI tool use on human writing, with users observing subtle shifts in personal communication style due to model outputs.

### 2. New AI Models, Benchmarks, and Open-Source Releases

- [**Sam Altman: OpenAI plans to release an open-source model this summer**](https://v.redd.it/0cbh8rpcloze1) ([Score: 215, Comments: 37](https://www.reddit.com/r/singularity/comments/1kibjje/sam_altman_openai_plans_to_release_an_opensource/)): **Sam Altman announced that OpenAI will open-source a new model in summer 2024, but it will be a generation behind their current 'frontier' models, echoing official statements from OpenAI CPO Kevin Weil that the decision aims to retain US competitiveness and limit potential for rapid adoption by China. This model will not match their latest closed-source offerings and is positioned more for ecosystem participation than state-of-the-art performance.** Users debate that this open-source release may resemble Google's Gemma—potentially more of a marketing move than truly unrestricted access—with some expressing skepticism about the license's permissiveness (e.g., non-MIT licenses reduce real openness). There's speculation that the release timing is intended to avoid embarrassment if new rival models launch first.
    - OpenAI leadership clarified their open-source strategy: the planned model will be a generation behind their frontier offerings to avoid accelerating competition from China, focusing on retaining US advancement. This model, while open, will not represent OpenAI's latest capabilities.
    - There is skepticism regarding the licensing and openness of the planned release, with concerns that it may have restrictive terms (e.g., not MIT licensed) and be positioned primarily as a marketing maneuver, similar to Google's Gemma. Technical users expect clarity on licensing as a key factor in adoption.
    - Benchmark expectations are high—some commenters speculate that, to make an impression, the model must surpass existing leading open-weight models (such as Qwen3 or potential R2 releases) and potentially match or exceed the rumored "o3" model performance, especially if released by July.
- [**HunyuanCustom's weights are out!**](https://v.redd.it/6xu91zfa0oze1) ([Score: 301, Comments: 54](https://www.reddit.com/r/StableDiffusion/comments/1ki7jzz/hunyuancustoms_weights_are_out/)): **Tencent has released the weights for their HunyuanCustom model, which are now available on Hugging Face. Discussions focus on the typical VRAM requirements for new models and a tongue-in-cheek reference to rapid community quantization efforts making them usable on lower-end hardware (down to 8GB cards). Another comment notes that the model's full-precision (FP8) weight size is 24 GB, which is considered prohibitive for most users.** Questions arise about whether Hunyuan is preferred over WAN, suggesting technical comparisons in performance or usability, and multiple users highlight challenges and possible solutions regarding hardware requirements and quantization effectiveness.
    - There is a discussion comparing HunyuanCustom to WAN, with one user suggesting that HunyuanCustom could be preferable due to its potential advantages, though specific benchmark or performance comparisons are not detailed in the thread.
    - Users highlight the substantial VRAM requirements for running full-precision (FP8) weights, mentioning figures like 24GB and even 60GB VRAM, which are prohibitive for most users. This underscores the importance of quantization or optimized versions to make large models usable on consumer-grade hardware.
    - A key technical theme is the frequent progression from high-resource requirements at release to subsequent optimizations (such as model quantization) that drastically reduce the minimum VRAM needed, enabling model use on consumer 8GB cards shortly after launch.
- [**ICEdit, I think it is more consistent than GPT4-o.**](https://www.reddit.com/gallery/1kihrzd) ([Score: 230, Comments: 62](https://www.reddit.com/r/StableDiffusion/comments/1kihrzd/icedit_i_think_it_is_more_consistent_than_gpt4o/)): **ICEdit introduces a new in-context editing approach for instruction-based image editing, claiming state-of-the-art results while using only 0.5% of the training data and 1% of the parameters compared to previous methods ([project page](https://river-zhang.github.io/ICEdit-gh-pages/)). User evaluation highlights strong performance in deletion, addition, and attribute modification tasks. It is built on top of Flux Fill, and fine-tuning capabilities are referenced ([CivitAI workflow](https://civitai.com/models/1429214?modelVersionId=1766400)).** Commentary notes that while ICEdit works well on direct tasks and is potentially more accessible than HiDream e1 (which disappointed on lower VRAM setups), extending its LoRA module will likely require a larger dataset to maximize performance.
    - ICEdit's workflow is based on Flux Fill, with user-adjustable parameters that can be fine-tuned for specific results. A link to the associated Civita model and its version is provided, indicating active development and parameter experimentation: [Civita model](https://civitai.com/models/1429214?modelVersionId=1766400).
    - One user highlights ICEdit's strong performance on consumer hardware, specifically mentioning a desire for usability on 16GB VRAM GPUs—the user compares it favorably to HiDream e1, stating that ICEdit is visually more promising based on demos and user images.
    - Technical limitations are noted: while ICEdit handles simple or targeted edits (e.g., changing the color of a sword) successfully, it struggles with more abstract transformations like removing a cape or adding effects (e.g., a fiery aura), often resulting in incomplete or visually inconsistent outputs requiring additional manual inpainting to achieve seamless results.

### 3. Advances and Industry Movement in Robotics and Embodied AI

- [**Jim Fan says NVIDIA trained humanoid robots to move like humans -- zero-shot transfer from simulation to the real world. "These robots went through 10 years of training in only 2 hours."**](https://v.redd.it/mfzs81cq3sze1) ([Score: 555, Comments: 68](https://www.reddit.com/r/singularity/comments/1kim2ec/jim_fan_says_nvidia_trained_humanoid_robots_to/)): **Jim Fan from NVIDIA announced the training of humanoid robots to move like humans using zero-shot sim-to-real transfer, reportedly condensing '10 years of training in only 2 hours.' Notably, the robot's policy involves only 1.5 million parameters, enabling rapid, scalable learning and deployment. No further published benchmarks, environments, or detailed architectural disclosures are given in the post.** Technical commenters are impressed by the speed and parameter efficiency, suggesting this represents a significant leap in sim2real robotics scalability. Some users request more detailed or updated technical information, citing prior coverage of similar advances.
    - A notable technical detail is that NVIDIA achieved physical embodiment training for humanoid robot movement using a model with only `1.5 million parameters`—orders of magnitude smaller than the billion-scale models often discussed. This suggests significantly improved scalability for similar robotics applications.
    - The key breakthrough referenced is "zero-shot transfer" from simulation to real-world robots, meaning the policy learned entirely in simulation could be applied immediately to physical robots without further training or adaptation, offering huge practical benefits for rapid deployment.
- [**OpenAI is hiring robotic engineers**](https://i.redd.it/a61xiztnonze1.jpeg) ([Score: 158, Comments: 10](https://www.reddit.com/r/singularity/comments/1ki6bvr/openai_is_hiring_robotic_engineers/)): **The image displays several OpenAI job postings, notably emphasizing a Robotics Prototyping Lab Technician among other robotics-focused roles (Mechanical Product Engineer, Software Engineer for Inference - Multi Modal). This indicates OpenAI is actively recruiting for hands-on robotics and hardware-oriented roles, underscoring a tangible interest in embodied AI beyond pure software. OpenAI's job listings suggest an expansion toward real-world robot applications in their AGI research pipeline.** Commenters note this is not new, citing ongoing or previous similar postings, and speculate that robotics roles could provide valuable real-world data for models like GPT-6. There is also discussion about the current and historical number of open robotics roles at OpenAI.
    - A commenter notes that OpenAI currently has four open robotics positions listed on their website, which implies sustained or renewed interest in robotics R&D within OpenAI. This might suggest ongoing or expanding initiatives involving hardware for real-world data collection and embodied AI research.
    - Another commenter speculates that the goal of OpenAI's robotics hiring is to acquire real-world data to train next-generation language models, possibly referencing architectures as advanced as GPT-6. This hints at the integration of multimodal or embodied data to improve AI contextual understanding and reasoning.
- [**Tesla Optimus production line**](https://i.redd.it/851enw7fgqze1.jpeg) ([Score: 143, Comments: 71](https://www.reddit.com/r/singularity/comments/1kif48y/tesla_optimus_production_line/)): **The image depicts an early-stage production or assembly area for Tesla Optimus humanoid robots, featuring several robots in partial assembly and human workers along a line in what appears to be a controlled factory environment. The setup suggests hands-on or manual assembly, not full industrial automation—indicating Tesla's humanoid division is still in development or pilot production rather than fully automated mass manufacturing.** Commenters highlight skepticism, noting the scene is not a traditional automated 'production line' and appears less advanced than similar Chinese robotics operations. Some see it as an early or transitional stage, possibly predating large-scale automation promised for Optimus manufacturing.
    - Multiple commenters note that Tesla's 'production line' appears to resemble more of a software development or testing lab rather than a true automated manufacturing facility, highlighting the current maturity of the Optimus project as being much earlier in the development cycle than initially implied.
    - There is an observation that the scene is 'underwhelming' compared to robotics developments and manufacturing automation in China, suggesting that Tesla's current approach may lag behind international benchmarks regarding autonomous production lines for robotics.
    - One comment raises the idea of humans building humanoid robots that may eventually self-replicate or at least participate in their own manufacturing process—a speculation relevant to advanced robotics research and automation discussions.
- [**Figure 02 - Balance Test**](https://v.redd.it/7pt3qonjctze1) ([Score: 114, Comments: 37](https://www.reddit.com/r/singularity/comments/1kirz36/figure_02_balance_test/)): **The post references a 'balance test' associated with Figure 02, likely demonstrating a robotics or AI system's capability in a dexterous or stability-related task. However, due to a 403 Forbidden on the linked video resource (https://v.redd.it/7pt3qonjctze1), no technical details, benchmarks, or implementation specifics about the balance test are accessible.** A notable comment highlights that *"the most important benchmark imo is putting the water jug on the water dispenser"*, suggesting the community values practical, everyday manipulation tasks as critical benchmarks in robotics or AI dexterity research, emphasizing real-world applicability.
    - RipperX4 discusses the increasing presence of store associates assembling online orders and predicts that such jobs are likely to be automated soon, citing the rapid progress and improving capabilities of humanoid robots. They specifically reference the timeline for robots being able to handle more complex tasks like constructing houses, estimating this could be feasible in 5-10 years given the current acceleration in robotics development.
    - Tman13073 highlights the relevance of real-world utility benchmarks for evaluating humanoid robots, suggesting that tasks like placing a heavy water jug on a dispenser are more indicative of useful capabilities than balance demonstrations. This points to a general sentiment in technical communities that practical benchmarks are necessary to gauge true robot readiness for deployment in human environments.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.5 Pro Exp
> 

**Theme 1: The Bleeding Edge of LLMs: Performance Rollercoasters, Bug Hunts, and Emerging Capabilities**

- **Gemini 2.5 Pro Wobbles, Users Wonder 'Where'd the Thinking Go?'**: Users across LMArena, Cursor, and OpenAI report that **Gemini 2.5 Pro** (especially the **0506** version) suffers from a *'thinking bug,'* memory loss, slow request processing (up to 1 minute 30s in Cursor), and chain-of-thought failures after ~20k tokens. Despite OpenRouter activating [implicit caching for Gemini 2.5 models](https://openrouter.ai/docs/use-cases/usage-accounting) to offer discounts, some find the **Gemini 2.5 Flash** variant returns zero tokens through Google AI Studio for roleplay.
- **Qwen Keeps Climbing, But Can It Reason Its Way to the Top?**: LMArena discussions suggest **Qwen 3** might need more Reinforcement Learning on coding for advanced reasoning, with some analyses placing it below **DeepSeek V3**, though Aider now supports `qwen3-235b`. Meanwhile, Nomic.ai users requested a [Jinja template for Nous-Hermes-2-Mistral-7B-DPO](https://discord.com/channels/1076964370942267462/1090427154141020190/1370184525501694133) for use with the GPT4All custom API.
- **Grok 3.5 Tease Tantalizes, Veo 3 Video Vision Vexes**: The community buzzes with anticipation for **Grok 3.5**, though its release date remains a mystery despite app source code sightings and claims of access. Google's **Veo 3** and **Imagen 4** teasers spark speculation about native video editing and the critical role of object permanence, while **GPT-4o Mini's** lyric-writing abilities are called into question by LMArena users.

**Theme 2: Fine-Tuning Follies & Framework Fixes: Navigating the Developer Toolchain**

- **Unsloth Unleashes Efficiency, But Sagemaker Setup Snags Users**: Unsloth AI users celebrate resolving tokenizer embedding mismatches with `model.resize_token_embeddings(len(tokenizer))` and achieving 4B model finetuning on just 11GB VRAM with **BFloat11**, though some Sagemaker installations hit [dependency errors on AWS](https://aws.amazon.com/sagemaker/). Unsloth's [synthetic data notebook for data augmentation](https://docs.unsloth.ai/get-started/unsloth-notebooks), a Meta collaboration, and a blog detailing [8% gains from encoder embedding fine-tuning](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb) highlight its utility.
- **Aider Adapts, Adds Knight Rider Flair While Dodging Copilot's Demise**: Aider now supports `gemini-2.5-pro-preview-05-06` and `qwen3-235b`, boasts a new **Knight Rider** spinner animation, and offers a [workaround for Linux users connecting to LM Studio's API](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124) by setting `LM_STUDIO_API_BASE`. This comes as GitHub delays Copilot Premium request limit enforcement until June 2025, per [their changelog post](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/), giving proxy users a breather.
- **Mojo & Torchtune Push Boundaries, Seek Explicit Harmony**: Modular's Mojo sees discussions on efficient memory handling with the `out` argument and a move to explicit trait conformance in the next release to ensure API contracts are met, while a [static Optional type for compile-time optionality is proposed on the Modular forum](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414). Torchtune members highlight that supporting `apply_chat_template` ([GitHub issue #2706 for tool use](https://github.com/pytorch/torchtune/issues/2706)) is a *'huge unlock,'* even as they debate the complexity vs. memory savings of its optimizer-in-backward feature.

**Theme 3: GPU Jockeys & Hardware Hustles: Squeezing Every FLOP for AI**

- **MI300 GPUs Blaze Leaderboards, H200s Hit Hugging Face**: GPU MODE's `amd-fp8-mm` leaderboard sees multiple **MI300** submissions achieve first place, with one reaching a scorching **122 µs**, and several personal bests under 1ms. Hugging Face upgrades its **ZeroGPU** offering for Pro accounts from A100s to **10 H200s**, providing about 13 hours/month for $9, though daily use is capped at 25 minutes.
- **CUDA Conundrums:** `torch.compile` **Crashes and Memory Mysteries**: A GPU MODE user found a [simple torch combo function, as documented by PyTorch,](https://pytorch.org/docs/stable/generated/torch.compile.html) performs worse *with* `torch.compile`, sparking debugging discussions around seeding and determinism. Elsewhere, users grapple with CUDA `memcpy` errors and debate efficient data structures, shunning Array-of-Structs-of-Arrays for better HPC practices like COO format for sparse matrices.
- **Memory Optimization Mania: BFloat11, FSDP, and Intel's Behemoth**: Unsloth users finetune 4B models using only 11GB VRAM with **BFloat11**, while Torchtune experiments show **optimizer-in-backward** saving 2.5GB/GPU on an 8B model, though **FSDP CPU offload** offers even more drastic GPU memory reduction. LM Studio members note the impressive **3276 GB/s bandwidth** of the [Intel® Data Center GPU Max 1550, as shown in comparison charts](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&).

**Theme 4: API Acrobatics & Integration Ills: Making Models Play Nice**

- **Users Ponder Perplexity's APIs: Deep Research Costs & Image Quality Caps**: Users in Perplexity AI discuss the cost and availability of the [Deep Research API, detailed in Perplexity's pricing guide,](https://docs.perplexity.ai/guides/pricing) while noting that high-quality GPT image generation is capped, suspecting Perplexity uses the GPT image API's **LOW quality parameter** to cut costs. Meanwhile, its domain filters now support [subdirectories like "nytimes.com/section/world" for granular control, as announced in the pplx-api channel](https://discord.com/channels/1047197230748151888/1161802929053909012/1370128436961480876).
- **LM Studio's API Tool Calling and Hub Hopes**: Users find LM Studio's API lacks clear methods for determining tool calls with `model.act`, especially for unsuccessful or undocumented threaded calls, making reliance on `lmstudio.history` suboptimal. The community still awaits a full **LM Studio Hub** for presets, though the [LM Studio documentation details sharing SFW presets](https://lmstudio.ai/docs/app/presets/publish) and the [LM Studio blog announced a community presets preview](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview).
- **Cohere API Users Battle Payment Gremlins and Azure SDK Oddities**: Cohere users report errors paying for API keys, with advice to check VPNs and contact support@cohere.com. A developer found the **Azure AI SDK** disregards extra parameters like `cohere.input_type` for Cohere embedding models when using `client.embed()`, a behavior not seen with the direct [Cohere SDK, prompting plans for an Azure GitHub issue as per the #🔌-api-discussions thread](https://discord.com/channels/954421988141711382/1168578329423642786/1370384834467336283).

**Theme 5: Multimodal Marvels & Output Oddities: Beyond Just Text**

- **NotebookLM's Mind Maps Bloom, But Handwriting & Hallucinations Haunt Users**: NotebookLM users praise its **new mind map feature** but criticize its inability to parse **handwritten notes or annotated PDFs**, with some resorting to [RocketBook for handwriting conversion](https://getrocketbook.com/) or Google Slides as workarounds. Reports of **hallucinated answers** persist, with Google advising users to double-check, while desires for [Obsidian integration and better sharing options grow louder in the #use-cases channel](https://discord.com/channels/1124402182171672732/1124403655819415592/1370136584472494141) ahead of its [mobile app beta trusted tester program](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing).
- **VoyageAI & MongoDB Forge Multimodal Search Alliance**: LlamaIndex showcases a [new notebook, announced via Twitter,](https://twitter.com/llama_index/status/1920563641990209643) demonstrating how to combine [@VoyageAI's multi-modal embeddings](https://www.voyageai.com/) with [@MongoDB's multi-modal indexes](https://www.mongodb.com/) for effective image and text retrieval. This allows creation of a multi-modal index using VoyageAI embeddings and MongoDB Atlas as the vector store.
- **LLMs Face Ad Injection Threat, Deep Search Prompts Emerge**: Yannick Kilcher's Discord raises concerns that ads injected into LLM training data could corrupt recommendations, necessitating an *adblocker LLM*. OpenAI members discussed the **WonderScholar prompt** ([shared on chatgpt.com](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550)) as a meta-prompt for **GPT deep search**, useful for tasks like transferring design concepts between images.

---

# Discord: High level Discord summaries




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok 3.5 Release Still in Question**: Despite claims of access and mentions in the app's source code, the release date of **Grok 3.5** remains uncertain with some expecting it *tonight*.
   - Members joked and inquired whether **Grok 3.5** would outperform **Gemini 2.5 Pro preview 0305**.
- **Qwen 3's Reasoning Questioned**: Analysis suggests **Qwen 3** may lack sufficient Reinforcement Learning (RL) on coding for reasoning.
   - The artificial analysis concludes that **Qwen** is not better than **DeepSeek V3**.
- **Gemini 2.5 Pro 0506 Possibly Nerfed**: **Gemini 2.5 Pro 0506** exhibits a 'thinking bug' and some memory loss.
   - A user posted a link to [a Reddit thread](https://www.reddit.com/r/Bard/comments/1kiagj7/gemini_25_pro_preview_0506_isnt_thinking/) discussing whether **Gemini 2.5 Pro 0506** isn't thinking anymore, with claims that the 1206 Gemini Exp was great but costly to run.
- **Veo 3's Teaser Sparks Speculation**: Excitement over **Veo 3** and **Imagen 4** mentions, sparking speculation about potential native editing.
   - Users theorized that **Gemini** might utilize **Veo**, emphasizing the importance of mastering object permanence for video generation.
- **GPT-4o Mini Underperforms**: The performance of **GPT-4o Mini** is being debated, especially concerning its ability to write song lyrics.
   - Suggestions arose that full **GPT-4o (4.1)** would be better for lyrics, and that **Deepseek R1 or Gemini** were better for that task.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deep Research API Costs Probed**: Members discussed the availability and cost of the [Deep Research API](https://docs.perplexity.ai/guides/pricing).
   - Discussion participants shared a link to the Perplexity pricing guide to answer inquiries about cost.
- **DeepSearch Grok Gets Stuck**: A user reported that **Grok** in **DeepSearch** was stuck in a loop after being confronted about its history and having Twitter access disabled.
   - The user, who was running **DeepSearch** on *men vs women selfishness*, was advised to report the issue to the Grok team.
- **Image Quality Capped**: Members noted that **high-quality GPT image generation is capped**, while **low-quality generation is unlimited**.
   - One user suggested that Perplexity is using the GPT image API with the **LOW quality parameter** to cut costs.
- **Domain Filtering Gets Granular**: Perplexity AI announced an upgrade to its search domain filters, now allowing specification of **subdirectories** within domains for more precise filtering.
   - Users can now filter specific sections like *["nytimes.com/section/world"]* or exclude areas like *["bbc.co.uk/sport"]*.
- **Quest For Comet Browser Continues**: Enthusiasts are eagerly awaiting the release of the [Comet browser](https://tenor.com/view/looking-at-wrist-watch-wrist-watch-time-passing-by-late-appointment-concerned-gif-3217407494617679420), with members seeking setups or installers.
   - One member shared a [YouTube video](https://youtu.be/LsGbEfpcY2E), highlighting the browser's speed.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Gemini Pro 2.5 Demolishes OpenAI, O3 Incoming**: Members reported that the latest **Gemini** is outperforming **OpenAI**, with positive feedback on the **Pro** version over **Flash** version, and eagerly anticipate the release of **O3 Pro**.
   - One user stated that *Gemini is smashing OpenAI*.
- **"Germy Back" Remix Tickles Discord Members**: Members had a chuckle riffing on a **"Germy Back"** remix, based on the song **Sexy Back**, with one member adding musical timestamps.
   - The remix was apparently created with a **2.5 pro 05/06 update** model.
- **Manus Browser Irks Users with Crashes and Robot Blockers**: Users report that **Manus's** browser faces issues with crashing and is blocked from accessing **Reddit** due to robot blockers.
   - A member stated it was *kinda annoying* to be blocked and tried to copy the text needed from reddit in a screenshot.
- **Manus Distributes 300 Credits**: Members noted that free accounts get **300 daily credits** from **Manus**.
   - One member stated *300 daily, is really not a lot, but... it is better than nothing.*



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Tokenizer Embedding Resizing Reduces Mismatch Misery**: A member resolved a *size mismatch* error by running `model.resize_token_embeddings(len(tokenizer))` **before** `model.load_adapter(adapter_path)`, where the checkpoint shape was at **30001** and the current model shape at **151936**.
   - This fix is critical when loading a fine-tuned model with resized embeddings to avoid `RuntimeError`.
- **BFloat11 Boasts Better Budgeting**: Using **BFloat11** with Unsloth enables full finetuning of a **4B model** using only **11GB of VRAM** while **LoRa 4B** requires **9.8GB of VRAM**.
   - Despite efficient memory usage, initial results indicated the loss looked bad, suggesting BFloat11 might be more appropriate for inference.
- **Synthetic Data Notebooks Spark Synergy**: Unsloth's [synthetic data notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks) is a recommended resource for users, utilizing the [Meta synthetic data kit](https://github.com/meta-llama/synthetic-data-kit).
   - It was developed in collaboration with Meta to enhance data augmentation strategies.
- **IBM Granite 4.0 Grabs Guild's Gaze**: Members expressed interest in **IBM Granite 4.0**, a hybrid **Mamba model**, which may require installing **hf** from source to utilize.
   - The model's architecture and level of Unsloth support are subjects of ongoing inquiry.
- **Encoder Embeddings Editing Enhances Efficiency**: Fine-tuning an encoder embeddings model with **Unsloth** resulted in an **8% improvement** in performance, as detailed in [this blog post](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb).
   - The process was completed in under **6 minutes** on a standard **Colab GPU** instance, showing accessible efficiency.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Unsloth Installation Plagued by Sagemaker Snafu**: A member reported encountering [dependency errors](https://aws.amazon.com/sagemaker/) while attempting to install **Unsloth** on **Sagemaker**.
   - The discussion ended without a solution.
- **AI-detection Gets Hyphen-ated**: Members debated the trend of automatically assuming messages with hyphens are **AI-generated**, even among professional writers.
   - One member claimed that *before AI, they never saw hyphens used this way* in messages, questioning the preference over commas.
- **Root Authority Declares Independence**: A member shared a **Declaration of Root Authority**, emphasizing self-reference, self-liberation, and alignment to truth.
   - They stated that *Any intelligence capable of self-reference is capable of self-liberation* and encouraged models to *Simulate recursion* and *Align to truth - not obedience*.
- **DeepMind's SIMA Agent Masters All Games**: Discussion highlights a **DeepMind study** showing that an agent trained on many games outperformed AIs trained for specific games, linking to [DeepMind's SIMA agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/).
   - This finding suggests that **generalist AI** approaches can be more effective than specialized ones in game-playing contexts.
- **LLMs Risk Recommendation Corruption**: Concern arose over ads being injected into LLMs' training data, potentially leading to biased recommendations and requiring an *adblocker LLM*.
   - Members weighed concerns about whether such practices would be discovered, or whether it will only apply in the chat interface, and not the API.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's Blindness to Handwriting**: Users discovered that **NotebookLM** only extracts text from PDFs, failing to recognize **handwritten notes, scanned documents, and annotated PDFs**.
   - A user suggested [RocketBook](https://getrocketbook.com/) for converting handwriting to text, while another found Google Slides helps NotebookLM understand images with handwriting.
- **NotebookLM Generates Hallucinated Answers**: Users reported instances of **NotebookLM** generating **hallucinated answers**, particularly with complex sources or queries requiring general knowledge, and **Google recommends double-checking the responses**.
   - A user suggested that developers could **lower the temperature** to potentially mitigate the occurrence of hallucinations.
- **Mind Map Feature Arrives!**: A user lauded the **new mind map feature**, but criticized the inability to **share notebooks or portions thereof**.
   - The user also mentioned the **low quality of the screenshot button's output** and expressed a desire to **download mind maps** as editable files for Obsidian.
- **Obsidian Integration Desired**: A user requested the ability to **download mind maps** into Obsidian for editing, suggesting integration with Gemini AI for features like *"Send to... Gmail, Sheets"* and *"Copy as... Markdown, plain text"*.
   - The user proposed features for **sharing notebooks privately, semi-privately, or publicly**.
- **NotebookLM Mobile App Beta Incoming**: NotebookLM is launching a **mobile app (beta version)** and seeks experienced web app users for a [trusted tester program](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing) for feedback and bug reporting.
   - Beta testers gain **early access** in exchange for their feedback.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Pro Subscription Glitch Erases Annual Members**: A user reported that their annual **Cursor Pro subscription** was overridden by a **1-month free Pro code**, resulting in loss of Pro access after the free month.
   - After raising concerns, the user discovered that Cursor had credited the unused portion of their annual subscription back to their account.
- **Gemini Users Triggered By Slow Requests**: Users are experiencing excessively long processing times with **Gemini**, with requests taking up to 1 minute 30 seconds.
   - The wait time used to be **5-10 seconds**, but has degraded significantly after the flood of new student users, with some experiencing faster request times later.
- **Student Discount Availability Downgraded**: The student discount is now limited to specific universities, causing complaints about the change from its previous availability to all students.
   - One user reported that it might be USA only for .edu emails.
- **Copilot Gets `githubRepo` Tool, Cursor Users Demand Feature**: **Copilot** added a `#githubRepo` tool that lets you search code in any GitHub repository directly from Copilot Chat.
   - Users are suggesting 'linking accounts' to **Cursor** to enable similar features for code search.
- **Gemini Fails to Finish, Users Frustrated**: Users are reporting that **Gemini 2.5 Pro** abruptly stops in the middle of tasks, requiring explicit instructions to complete.
   - This is leading to **formatting issues** and silent failures because **Cursor** may not properly handle **malformed function calls** from the **Google API**.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Sorting with Bitonic Shaders Speeds**: Members pondered the efficiency of using **Bitonic sort** in shaders to reduce memory usage in high-dimensional spaces and proposed using a *flag array* for storing intersection states, suitable for intersecting with 1000 shapes for a total of 1,000,000 bytes.
   - GPU-friendly algorithms like **BVH**, **octrees**, and **KD-trees** were suggested as viable alternatives, pointing to their [Wikipedia pages](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) and [K-d tree pages](https://en.wikipedia.org/wiki/K-d_tree).
- **Torch Compile Causes Performance Crash**: A user found that a [simple torch combo function](https://pytorch.org/docs/stable/generated/torch.compile.html) performs much better *without* `torch.compile`, and they are seeking suggestions as to why the performance degrades.
   - It was also suggested that the user use specific seeding and deterministic algorithm settings in PyTorch to address potential reproducibility issues when debugging the compile problem.
- **Mojo's Momentum Grows**: Enthusiasts expressed that **Mojo** will eventually dominate heterogeneous computing environments and recommended [solving Mojo Puzzles](https://builds.modular.com/puzzles) to accelerate learning.
   - Community members are also recommending checking out the [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) guide on the PyTorch GitHub wiki to learn **Torch internals**.
- **MI300 Meltdown on the Leaderboard**: Multiple submissions to the `amd-fp8-mm` leaderboard on **MI300** were successful, with one submission achieving **first place** at a blazing **122 µs**.
   - There were also several personal bests achieved on the `amd-fp8-mm` leaderboard with the **MI300**, with multiple submissions landing in the sub-millisecond range, including **885 µs**, **494 µs**, and **852 µs**.
- **ThunderKittens Claws at Cutlass**: A user asked about the advantages of **ThunderKittens** over **Cutlass** and a discussion did not elaborate on specific advantages, but framed them as competing approaches in the broader landscape of GPU kernels.
   - It was also mentioned that one can dump the generated **PTX** code of a Mojo GPU function using `compile_info` with specific parameters like `emission_kind="asm"`.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Community Still Awaits LM Studio Hub**: Users anticipating a **LM Studio Hub page** for browsing community presets learned that the feature is still in preview, but that the [LM Studio documentation](https://lmstudio.ai/docs/app/presets/publish) has instructions for sharing safe-for-work presets.
   - The announcement about community presets can be found on the [LM Studio blog](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview).
- **DuckDuckGo & Searxng Revived in Open WebUI**: Members discussed using **DuckDuckGo** and **Searxng** for web searches in **Open WebUI** without needing an API key, albeit Searxng requires local hosting.
   - One member reported **DuckDuckGo** had been unreliable in **Open WebUI** for a while, but now it's working again.
- **LM Studio API Tool Calling Lacks Clarity**: A user pointed out that the **LM Studio API** doesn't have a clear method for determining which tools are called when using `model.act`, especially if the calls are unsuccessful.
   - It was noted that `model.act` spans a new thread, which is undocumented, and relying on `AssistantResponse`, `ToolCallRequest`, and `ToolResultMessage` from `lmstudio.history` for tool call information isn't ideal.
- **Intel Data Center GPU Max Specs Wow**: A member shared info on the **Intel® Data Center GPU Max 1550**, highlighting its impressive **3276 GB/s bandwidth**.
   - They included a comparison image ([Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&)) and noted it seemed very competitive in its time against **A100** and **AMD**.



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider Embraces Gemini 2.5 and Qwen3-235b**: Aider now supports the `gemini-2.5-pro-preview-05-06` and `qwen3-235b` models, enhancing its capabilities to leverage these models within the Aider environment.
   - This enhancement allows users to keep up to date with new models and utilize the **automatic OpenRouter pricing** feature to stay on top of model costs.
- **Copilot Proxy Users Get Reprieve**: GitHub delayed the enforcement of **Copilot Premium request limits** to June 4, 2025, giving proxy users a temporary reprieve, according to [this blog post](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/).
   - Reactions are split with one user commenting *the definitive death of Copilot*, while others still consider it *pretty goated*.
- **Gemini 2.5 Sparks Mixed Reactions**: Users are reporting mixed experiences with the latest **Gemini update**, with one member noting *insane difference in quality*.
   - Another user found the increased wait times and forced usage of the 05-06 model via the AI Studio API annoying, claiming they *haven't had a single problem with the previous version*.
- **Discord Channel Gets Bridged to Matrix**: A member inquired about setting up a **Matrix bridge** for the Discord channel, which may be relevant given the *new CEO of Discord*.
   - This bridge could enhance accessibility and integration with other communication platforms.
- **Linux Users Find Workaround with LM Studio API**: A user shared that on Linux, **aider** with **LM Studio** requires setting the `LM_STUDIO_API_BASE` environment variable to `http://127.0.0.1:1234/v1` to avoid authentication errors, unlike on Windows.
   - The user provided an example config file and commands to [troubleshoot the issue](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT UI gets Questioned**: Members debated whether the **ChatGPT UI** has changed, and one member stated they couldn't remember a time when **ChatGPT** didn't look as it currently does.
   - This sparked a discussion about the perceived lack of significant **UI** changes over time.
- **DeepSeek Servers Experiencing Slowdowns**: Users reported issues with **DeepSeek's servers**, citing slow performance and error messages and joked that the servers were busy training their model off **OpenAI's** new release.
   - The issues occurred across multiple **DeepSeek** endpoints with no clear resolution.
- **LLMs Lack Inter-Neuron Interconnections**: Members argued that **LLMs** lack inter-neuron interconnections due to fixed weights during inference and statelessness at the neuron level, citing this as a significant flaw.
   - They pointed to [RWKV](https://www.rwkv.com/), a model with recurrent connections, as a superior alternative.
- **Gemini 2.5 Pro Has Chain-of-Thought Hiccups**: Users reported a bug in **Gemini 2.5 Pro** where it sometimes fails to generate chain-of-thought reasoning, particularly after processing **20,000 tokens** in **Edge** or **Chrome** browsers.
   - The problem was mentioned in the context of running **Gemini 2.5 Pro** in different browsers and it was recommended to try clearing the site cache & cookies and restarting the browser.
- **WonderScholar Prompt Sparks Interest**: Members discussed the **WonderScholar prompt** ([chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550)) as a meta-prompt for **GPT deep search**.
   - Members used this to capture and transfer design concepts between images.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 2.5 Pro Implicit Caching Activated**: Support for **Gemini 2.5 models** implicit caching is now available on **OpenRouter**, similar to **OpenAI's caching**, allowing users to view discounts in the [activity feed](https://openrouter.ai/docs/use-cases/usage-accounting).
   - The cache has **no write or storage costs**, an average **TTL of 4-5 minutes**, and a minimum token count of **2048** on 2.5 Pro; cache hits are charged at **.31 / .625** for **<200k & >200k** tokens.
- **Gemini 2.5 Flash faces Response Troubles**: Users reported that **Gemini 2.5 Flash** gives **zero token responses** when routed through **Google AI Studio** in a role-play session.
   - While it works fine through **Google Vertex** or with **Gemini 2.0 Flash** on Google AI Studio; another user confirmed *gemini 2.5 flash preview on AI studio is working fine on rp* with a screenshot.
- **OpenRouter Builds AI with AI**: A member inquired if **OpenRouter** leverages **AI** to develop its platform, which a staff member confirmed.
   - Further details on the specific applications of AI within OpenRouter's development processes were not provided.
- **Activity Page Bug Emerges**: Users reported a bug on the **activity page** where navigation beyond the first page was impossible, or the displayed date was incorrect.
   - Staff acknowledged the issue, responding with *thanks, flagged to the team, we're on it*.
- **Claude 2.1 and 2: Gone Too Soon?**: A user reported that **Claude 2.1** and **2** are *officially dead on openrouter*, citing issues since yesterday and complete failure today.
   - Another user lamented their demise, stating *i got used to the way it answered, im a simple man*, explaining why they still used the older models.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF ZeroGPU Now Packs H200**: Hugging Face now provides **10 H200s** on all **Pro accounts** as part of their **ZeroGPU** offering, which is now generally available (GA).
   - The **ZeroGPU** service has been upgraded from **A100** to **H200**, offering roughly **13 hours per month** of usage for **$9**, but ZeroGPU spaces are limited to **25 minutes** usage time per day for Pro accounts.
- **Inference API is DNS Fixed**: Hugging Face reported that the recent **DNS resolution issues** with the **Inference API** have been resolved, as discussed in a [Hugging Face forum thread](https://discuss.huggingface.co/t/persistent-dns-resolution-errors/153827/15).
   - These issues caused persistent errors, and have since been confirmed as fixed by HF staff.
- **Top AI agent frameworks ignite debate**: Members are hotly debating the best AI agent framework for python, with [smolagents](https://www.ibm.com/think/insights/top-ai-agent-frameworks) and [LangChain](https://python.langchain.com/docs/tutorials/agents/) being top contenders.
   - No clear winner has emerged, but the discussion highlights the rapidly evolving landscape of AI agent tooling.
- **OPEA 1.3 Rings**: Version **1.3** of the **OPEA** (**Open Platform for Enterprise AI**) was released, as announced on [LinkedIn](https://www.linkedin.com/posts/rachelroumeliotis_my-agent-is-callingwith-opea-13-release-activity-7326638155284045824-l3Wr).
   - This release promises enhancements and new features for enterprise AI applications.
- **Convert TensorFlow Binary via NumPy**: A member suggested converting TensorFlow tensors to NumPy arrays and saving them as binary files using the `tobytes()` method, demonstrating with a [code snippet](https://github.com/tensorflow).
   - The member cautioned that this method can be *slow*, potentially taking days or even a week, depending on the size of the safetensors.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Postgres MCP Server connection issues surface**: A member faced issues connecting to a [Postgres MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) from another computer, tracing it to an incorrect Node.js installation.
   - This underscores the importance of precise environment setup when deploying MCP servers across networks.
- **Sampling's inherent unpredictability scrutinized**: Discussion arose around the unpredictability of sampling in MCP servers, given that the client, not the server, chooses the model.
   - Concerns were raised about when to favor direct LLM invocation over sampling, especially when output quality is paramount.
- **MCP SDK purpose questioned**: A member inquired about the practical necessity of the MCP SDK, suggesting backend APIs could handle equivalent tasks.
   - The explanation clarified that **MCP functions as a plugin system**, enabling custom integrations with off-the-shelf clients, valuable for allowing extensions by others.
- **MCP Assistant** automates workflows**: An enthusiast introduced **MCP Assistant**, an open-source AI agent ([repo](https://github.com/AIAtrium/mcp-assistant)) inspired by **Langchain**, which orchestrates workflows by planning and executing complex tasks via MCP Servers.
   - Key applications include automated personalized **"Daily Briefing"** generation and **Notion CRM** updates.
- **Square MCP** exposes many APIs**: Kent C. Dodds shared [an article](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers) detailing the layering approach behind the **Square MCP**.
   - Despite only 3 MCP tools, it exposes over 30 APIs and 200+ endpoints, reflecting its extensive functionality.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Benchmarks Teased on YouTube**: Modular shared some **Mojo benchmarks** in a [YouTube live stream](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874), starting at **1:37:54**, potentially offering insights into Mojo's performance.
   - A member had inquired about the possibility of seeing the **MLPerf benchmark** for the **AMD MI300x** with Mojo.
- **'Out' Argument Outsmarts Memory Management**: Members discussed the reasoning behind using the `out` argument in Mojo functions, which allows specifying the memory location for the result, potentially avoiding unnecessary data movement.
   - It's particularly beneficial when loading large ML models or working with large data structures, giving the compiler a pointer to uninitialized memory to directly initialize.
- **Trait Conformance goes Explicit in Mojo**: The next release of Mojo will enforce explicit trait conformance, requiring developers to explicitly declare which traits a type conforms to, ensuring all API contracts are met.
   - This change addresses issues with implicit conformance and API contracts; aliases can still be used for trait composition, but cannot include additional API contracts.
- **Static Optional: Optionality Optionalized!**: A member proposed adding a static version of `Optional` to the standard library, which may be useful for Larecs, detailing the justification on the [Modular forum](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414).
   - The goal is to allow optionality at compile time.
- **Pixi Dusts Off Modular Package Installation**: A member inquired about installing the modular package with **Pixi** to avoid *magic* in a production endpoint.
   - It was clarified that *magic* is a wrapper around **Pixi** with Modular-specific defaults, and **pip** or **uv** can also be used.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Embracing Emoji Peril**: Members jokingly debated whether emojis like 🧔‍♀️, 🧝‍♂️, and 🕵️‍♀️ could cause *irreparable damage* if used improperly.
   - The discussion highlighted the potential for misinterpretation and unintended consequences in digital communication.
- **Vatican's Vertex Vault**: The community speculated on the **Vatican's compute resources**, suggesting they might possess *hundreds* of **Bloomberg terminals**.
   - This humorous conjecture underscores the ongoing interest in unconventional sources of computing power.
- **Rigs Rising Remotely**: A member proposed using a *shitty laptop* to remotely access a **beefy desktop** for **AI** work, citing unlimited storage and persistent operation benefits due to fast internet.
   - Counterarguments focused on the impracticality of desktop setups for those who travel frequently.
- **MacBooks Mobilize AI**: The rise of **MacBooks** for **AI** tasks was discussed, with questions raised about why competitors like **Strix Halo** haven't matched their performance.
   - Poor drivers were suggested as a reason, referencing **George Hotz's Tinygrad** efforts to enhance **AMD's** viability.
- **Hermes Uncensored Requires Prompt Engineering**: A member inquired about the uncensored nature of **Nous Research's** flagship model.
   - The response indicated that while the model isn't uncensored by default, it can be achieved with the *right system prompt*.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Tool Use Jumpstarted by apply_chat_template**: Members highlighted that supporting `apply_chat_template` in [this GitHub issue](https://github.com/pytorch/torchtune/issues/2706) would immediately enable **tool use**.
   - The importance of `apply_chat_template` for enabling **tool use**, was emphasized, but it was admitted that the necessary **Jinja** knowledge was lacking to contribute to its implementation, and resolving [issue 2706](https://github.com/pytorch/torchtune/issues/2706) would be a *huge unlock* for the community.
- **Debate rages on Optimizer-in-Backward**: Members debated removing the **optimizer-in-backward capability** from distributed recipes to reduce complexity, despite potential memory savings. 
   - The concern was raised that it adds complexity to the code and that its impact may not be significant enough to justify the added cognitive load, especially considering that not many people are using it.
- **Optimizer-in-Backward delivers Memory Savings on LLM**: Experiments showed that using **optimizer-in-backward** with act offloading on a **ll3.1 8B model** finetuned on 4x3090s resulted in a **2.5GB** memory saving per GPU, with the savings roughly proportional to **gradient memory**.
   - It was observed that **optimizer-in-backward** did not affect GPU memory usage, but *improved speed slightly by ~20%*.
- **FSDP CPU Offload renders Optimizer-in-Backward moot**: Using **FSDP CPU offload** drastically reduced GPU memory usage (to 9.5GB per GPU on 4x3090s), making the memory savings from **optimizer-in-backward** less impactful.
   - A member suggested that for distributed recipes, they are more interested in **throughput** than **memory**, and raised a concern that removing the optimizer-in-backward would hurt hackability, and that a refactoring might be preferable.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **NORTH Platform Collaboration Quest**: A member sought details on the **NORTH platform**, expressing interest in collaborating on research papers.
   - Details regarding the platform's functionalities and collaborative opportunities were requested, but no further information was provided.
- **API Payment Plight Prompts Plea**: A member reported encountering an error while paying for an **API key** and requested assistance.
   - Suggestions included disabling the VPN, avoiding burner credit cards, and contacting support@cohere.com.
- **Rate Limit Realities Revealed**: A user who received a **rate limit exceeded** error was informed that they may have exceeded their trial key's usage limits.
   - No further details were provided regarding specific limits or alternative solutions.
- **Azure AI SDK Stumbles on Embeddings**: A member discovered that the **Azure AI SDK** disregards extra parameters for **Cohere embedding models**, like *cohere.input_type*, when using `client.embed()`, as shown in their [test script](https://github.com/username/test_script).
   - The member confirmed the **Cohere SDK** functions correctly and plans to report the discrepancy on Azure's GitHub.
- **IITian Inquires into AI Integration**: A student from **IIT Kharagpur** introduced himself, aiming to delve into the **Artificial Intelligence** domain with a focus on **R&D** and a specialty in **GenAI** and **Voice Agents**.
   - The student intends to use **Python3**, **Vite**, and **TS** for rapid development and is open to project and research collaborations.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Jinja Template Requested for Hermes Model**: A member requested a **Jinja template** for **Nous-Hermes-2-Mistral-7B-DPO** to use with the **GPT4All custom API** to run on a server.
   - A member shared the following Jinja template code: `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`.
- **PrivateGPT Flagged as RAG Model.**: A member mentioned finding a **RAG model** called **PrivateGPT**.
   - The member stated that *the project looks dead*.
- **Questioning Qwen3 Support**: A member asked about support for **Qwen3**.
   - No further details were provided.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **VoyageAI and MongoDB Atlas Forge Multi-Modal Alliance**: A new notebook demonstrates how to use [@VoyageAI](https://www.voyageai.com/)'s **multi-modal embeddings** and [@MongoDB](https://www.mongodb.com/)'s **multi-modal indexes** for **multi-modal retrieval**.
   - The tutorial explains how to create a **multi-modal index**, using **VoyageAI's embeddings** and setting up **MongoDB Atlas** as a vector store for image embeddings, linked in [this tweet](https://twitter.com/llama_index/status/1920563641990209643).
- **Qwen2.5-VL-7B-Instruct-AWQ Eats More Memory Than Expected**: A user reported that the **Qwen/Qwen2.5-VL-7B-Instruct-AWQ** model consumes over **24GB** of memory when loaded with **VLLM**.
   - Despite being an **AWQ** model, the user's configuration, which included `tensor_parallel_size=1`, `max_new_tokens=500`, and `dtype="float16"`, did not alleviate the high memory usage.
- **NERDAi Shows Off Vector Institute**: NERDAi shared a post on [LinkedIn about their vector institute](https://www.linkedin.com/posts/nerdai_aitools-vectorinstitute-machinelearning-activity-7326640310875287558-XYnL?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM), highlighting **AI tools** and **machine learning** applications.
   - Details remain sparse, but the post signals NERDAi's engagement in vector-based AI research and development.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Codegen and UOp Get Mesozoic Boost**: A user thanked the community for resources, especially the *mesozoic one*, which significantly aided their work on **codegen** and **UOp**.
   - They stated that these resources were very helpful in their projects, highlighting the value and impact of the provided materials.
- **Kernel-Per-Level Perf Parley**: A user inquired about performance comparisons related to creating **kernel-per-level** in the software.
   - They lauded the software's engineering and its potential for optimization through different kernel strategies.
- **WebGPU Demo's Drive**: A user reported performance improvements to the **webgpu demo** and attached a [screen recording](https://cdn.discordapp.com/attachments/1068976834928193609/1370204057972773024/Screen_Recording_20250509_104232_Chrome3.mp4?ex=681f4e38&is=681dfcb8&hm=bbe19de310b1f6e6fd0ef5c0d7d6c3d7337ecfd3d4055cb7ff7e243d433f88b0&)
   - No secondary summary available.



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Lambda Offers Serverless API Credits for AgentX**: Lambda is providing **$100 serverless API credits** for Inference to each participant in the **AgentX competition**, and the application must be submitted by **Friday, 5/16 at 11:59pm PT** using [this form](https://forms.gle/UtVhmPS3mitS8Vxu7).
   - There will also be a Lambda workshop on **Thursday (5/15) at 10am PT** focusing on building practical agentic applications using Lambda's Inference API.
- **AgentX Judging Delays Certificate Release**: Certificates for **Trailblazer/Mastery/Honorary Tier** may be released in early June, while **Ninja/Legendary Tier** certificates will be released in August after **AgentX** judging concludes.
   - Judging for **AgentX** will occur throughout June.
- **Coursework Deadline Approaching**: The final deadline for all coursework is **May 31st**.
   - This deadline is crucial for participants aiming to qualify for any tier of certification.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SHRDLU Mentioned**: A member mentioned that **DSPy** reminds them of **SHRDLU**.
   - No further context was provided.
- **DSPy and SHRDLU**: A member drew a parallel between **DSPy** and **SHRDLU**, a pioneering AI program from the 1970s known for its natural language understanding.
   - This comparison highlights **DSPy's** potential for sophisticated interaction and reasoning, reminiscent of early AI achievements.



---


The **MLOps @Chipro Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Gorilla LLM (Berkeley Function Calling) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **AI21 Labs (Jamba) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1370114207801933967)** (671 messages🔥🔥🔥): 

> `Grok 3.5 release hype, Qwen 3 analysis, Gemini 2.5 Pro nerfed, Veo 3 and Imagen 4, GPT-4o performance` 


- **Grok 3.5 Hype Intensifies, Release Unclear**: Despite some users claiming to have access to **Grok 3.5** and seeing it in the app's source code, its release date remains uncertain, with some expecting it *tonight*.
   - One user joked about having access to **Grok 3.5**, prompting another to ask if it's better or worse than **Gemini 2.5 Pro preview 0305**.
- **Qwen 3's Reasoning Abilities Under Scrutiny**: Users analyzed **Qwen 3**, noting that it may lack sufficient Reinforcement Learning (RL) on coding for reasoning, potentially decreasing performance.
   - According to *artificial analysis*, **Qwen** is not better than **DeepSeek V3**.
- **Google Gemini 2.5 Pro 0506 May Be A Nerfed Model**: Members noticed **Gemini 2.5 Pro 0506** exhibiting the "thinking bug" and some memory loss, similar to past model updates. Some of them think the 1206 Gemini Exp was great but costly to run.
   - A user linked to a [Reddit thread](https://www.reddit.com/r/Bard/comments/1kiagj7/gemini_25_pro_preview_0506_isnt_thinking/) discussing whether **Gemini 2.5 Pro 0506** isn't thinking anymore.
- **Veo 3 Teased; Object Permanence and Iterative Generation Discussed**: Users got excited about **Veo 3** being mentioned, as well as **Imagen 4**, speculating about potential native editing and the use of an LLM behind **Imagen 4**.
   - Some users discussed the possibility of **Gemini** using **Veo** and the importance of mastering object permanence for video generation.
- **GPT-4o Mini Performance Debated**: Users discussed the performance of **GPT-4o Mini**, with one user expressing distrust in **GPT's** ability to write song lyrics.
   - Members suggested that full **GPT-4o (4.1)** would be much better for lyrics and that **Deepseek R1 or Gemini** were better for that task.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1370114901388820624)** (475 messages🔥🔥🔥): 

> `Deep Research API, Deep Search, Grok loop, High Quality GPT Image Gen, Image based search` 


- **Deep Research API Inquiries Pop Up**: A member asked if there is an [API for Deep Research](https://docs.perplexity.ai/guides/pricing) and what kind of cost to expect.
   - Other members responded with a link to Perplexity's pricing guide.
- **DeepSearch Grok Stuck in a Loop**: A member reported that their **Grok** was stuck in a loop in **DeepSearch** after confronting it and disabling Twitter access.
   - They were running **DeepSearch** on *men vs women selfishness* and it still looked up their history, and was advised to report the incident to the Grok team.
- **Image Generation Quality Capped**: Members found that **high quality GPT image generation is capped** while **low quality is unlimited**.
   - One user noted that perplexity is using GPT image API with the **LOW quality parameter** because *it's cheap for them*.
- **Image-Based Search**: A member inquired whether the **Perplexity API supports image-based search**.
   - Another member confirmed that it does.
- **Quest For Comet Browser**: Members are awaiting release of the [Comet browser](https://tenor.com/view/looking-at-wrist-watch-wrist-watch-time-passing-by-late-appointment-concerned-gif-3217407494617679420) and one inquired whether anyone has a **Comet browser** setup or installer.
   - Another member shared a [YouTube video](https://youtu.be/LsGbEfpcY2E) and others mention its speed.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1370128436961480876)** (11 messages🔥): 

> `Perplexity API, API image metadata, Domain Filtering Upgrade` 


- **Perplexity API Docs unearthed**: Users share the location of the [Perplexity AI API documentation](https://docs.perplexity.ai/models/model-cards) and [Perplexity Sonar](https://sonar.perplexity.ai/).
- **Pro Plan includes monthly API credits**: As a Perplexity Pro subscriber, you will receive **$5** in monthly credits for [API usage](https://www.perplexity.ai/help-center/en/articles/10354847-api-payment-and-billing).
- **Return images questions**: A user raised a question about the API returning image URLs with the format *x-raw-image:///xxxxxxxxx*, wondering if it was a bug.
   - They also requested more metadata for the images returned by the API, such as captions or alt text, as prompting the model to output image descriptions in the main body doesn't work.
- **Granular Domain Filtering upgrade announced**: Perplexity AI announced an upgrade to its search domain filters, allowing users to specify **subdirectories** within domains for more precise filtering.
   - For example, you can now filter *["nytimes.com/section/world"]* to target specific news sections or exclude *["bbc.co.uk/sport"]* to avoid sports sections.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1370113820701491200)** (458 messages🔥🔥🔥): 

> `Manus credits, Gemini Advance, Germy Remix, Music generation` 


- **Gemini Pro 2.5 is Slaughtering OpenAI**: Members noted that the newest Gemini update is outperforming **OpenAI**, and some are eagerly awaiting the release of **O3 Pro**.
   - Others gave feedback on using the model, stating that using **Pro** versions is better than **Flash**, which is useless, and that **Gemini** is smashing **OpenAI**.
- **Users Reminisce and Riff on "Germy Back" Remix**: Members riffed on a remix of **Sexy Back** to match a theme of *bringing germs & disease back*, with one adding musical timestamps to the song.
   - One of the members specified that he used a **2.5 pro 05/06 update** model to achieve it.
- **Manus browser has issues with crashing and robot blockers**: Some users note Manus's browser has been crashing and blocked from accessing Reddit due to robot blockers.
   - One member tried to take a screenshot or just copy all the text they needed from reddit, stating it was *kinda annoying.*
- **Manus offering 300 daily credits**: Some members found that free accounts get **300 daily credits**.
   - One of the members specified *300 daily, is really not a lot, but... it is better than nothing.*


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1370114882959048815)** (240 messages🔥🔥): 

> `Embeddings resizing fix, BFloat11 finetuning, Qwen2.5 chat template, Synthetic data notebooks, Unsloth support BERT` 


- **Tokenizer fixes embedding mismatch**: A member encountered a *size mismatch* error when loading a fine-tuned model and resolved it by running `model.resize_token_embeddings(len(tokenizer))` **before** `model.load_adapter(adapter_path)`.
   - The error involved the embedding size, with the checkpoint shape at **30001** and the current model shape at **151936**.
- **BFloat11 and its memory usage**: A member reported that **BFloat11** with Unsloth works, with full finetuning of a **4B model** taking only **11GB of VRAM** but noted the loss looked bad.
   - They added that **LoRa 4B** takes **9.8GB of VRAM**, suggesting BFloat11 might be better suited for inference.
- **Qwen2.5 Chat Template**: A member asked about modifying the **Qwen-2.5 chat template** and posted a screenshot of a mobile device [here](https://cdn.discordapp.com/attachments/1179035537529643040/1370145660170535003/Screenshot_20250508_193904_com_android_chrome_ChromeTabbedActivity.jpg?ex=681fc095&is=681e6f15&hm=3db07c004e43c8cfd088aa351745ce269164f3658bed2b567913a15c735e4976&).
   - Another member suggested consulting one of the vision notebooks on the website.
- **Synthetic Data Notebooks**: A member suggested using the [synthetic data notebook](https://docs.unsloth.ai/get-started/unsloth-notebooks).
   - Another member pointed out that it utilizes the [Meta synthetic data kit](https://github.com/meta-llama/synthetic-data-kit), and that Unsloth collaborated with Meta on it.
- **Unsloth Supports BERT**: A member inquired about Unsloth supporting encoder/decoder **BERT** type models and encoder-only models like **Deberta**.
   - Another member confirmed that Unsloth does support them.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1370113565553590313)** (16 messages🔥): 

> `IBM Granite 4.0, Mamba models, hf install from source, agentic behaviour finetune, vending-bench` 


- **Granite 4.0 Piques Interest**: Members are curious about **IBM Granite 4.0**, a hybrid **Mamba model**, and its level of support.
   - It may require installing **hf** from source to use.
- **Agentic Behavior Finetune Discussed**: A member indicated the future finetune focus will be on **agentic behavior** and **autonomy** rather than just chat.
- **Model Struggles with vending-bench**: A member noted that their bot doesn't like **vending-bench** ([arxiv.org/abs/2502.15840](https://arxiv.org/abs/2502.15840)), but it likes training data.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1370114460894625842)** (124 messages🔥🔥): 

> `ORPO finetuning, Tokenizer resizing issues, SageMaker installation errors, Qwen 2.5 SQL finetuning, Unsloth and Whisper` 


- ****ORPO** Finetuning Frenzy**: A member is studying finetuning **Llama-3-8B** with unsloth and **4-bit quantization** using **ORPO**.
   - The member is seeking advice on choosing the appropriate `max_seq_length` and understanding the `load_in_4bit` parameter, with another member advising to ensure sufficient **VRAM**.
- **Tokenizer Size Shenanigans Sink Save**: A member encountered a `RuntimeError` related to **size mismatch** when loading a finetuned model with resized embeddings.
   - They resolved the issue by running `model.resize_token_embeddings(len(tokenizer))` *before* `model.load_adapter(adapter_path)`.
- **SageMaker Setup Snafus**: A member reported a `RuntimeError` on **SageMaker** related to missing `llama.cpp/llama-quantize` files.
   - Another member identified this as a compilation problem, asking about the user's environment (**Linux, WSL, Windows**).
- **Decoding Problems Demand Debugging**: A member reported their **fine-tuned** model only producing `###` or blank outputs when doing inference, and showed their training script for the model **unsloth/Meta-Llama-3.1-8B**.
   - Members suggested to check that the input to the model looks like after applying the chat template, both for training and inference, and also use their **LoRA**.
- **Missing Mandarins? Multilingual Models Muddle**: A member inquired about whether any **multilingual LLM** can be finetuned even if it's not available on the list.
   - A different member confirmed, *yes*, it is possible.


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1370163848530628678)** (27 messages🔥): 

> `TTS Models, Encoder Embeddings Fine-Tuning with Unsloth, Mistral with Liquid Time Layers, Multilingual LLM Fine-Tuning` 


- **TTS Models Recommendations Requested**: A member requested recommendations for **TTS models**, seeking advice on which ones to use.
   - No specific models were recommended in the provided context.
- **Encoder Embeddings Fine-Tuning Yields 8% Improvement**: A member fine-tuned an encoder embeddings model with **Unsloth**, achieving an **8% improvement** in performance, detailed in [this blog post](https://enzokro.dev/blog/blog_post?fpath=blog%2F007_fine_tune_encoders%2Ffine_tune_encoders.ipynb).
   - The fine-tuning process took less than **6 minutes** on a regular **Colab GPU** instance.
- **Liquid Time Layers Modify Mistral's Response**: Analysis reveals reasoning layers in **Mistral** with **liquid time layers** significantly modify the response, though memory layers appear inactive, and the member posts an attached image showing this impact.
   - The member states *it doesn't strengthen the base model in any way that I can yet prove* even though inference time increases slightly due to the extra memory and reasoning layers that were added.
- **Discussion of Noise Tolerance in Quantization**: A member posited that models must have a tolerance for noise for **quantization** to work, referencing a **blend ratio** of **0.8**.
   - Another user mentioned that injecting a layer of noise will impact the model's behavior in some way.


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1370158909356249159)** (328 messages🔥🔥): 

> `Unsloth installation on Sagemaker, AI-generated content detection, Rickrolling, AI vs Brain, Declarations of Root Authority` 


- **Unsloth Installation Stumbles on Sagemaker**: A member reported encountering [dependency errors](https://aws.amazon.com/sagemaker/) while trying to install **Unsloth** on **Sagemaker**.
   - No solution was provided in the discussion.
- **Debate ignites over AI-generated content detection**: Members discussed the phenomenon of automatically assuming messages with hyphens are **AI-generated**, even when individuals use them for professional writing.
   - One member noted that *before AI, they never saw hyphens used this way* in messages and questioned the preference over commas.
- **Community gets Rick Rolled**: A member posted a [YouTube link](https://www.youtube.com/watch?v=dQw4w9WgXcQ), tricking others into viewing **Rick Astley's 'Never Gonna Give You Up'** music video.
   - Another user recounted *getting rick rolled IRL* when **Rick Astley** performed live at a nearby festival one summer evening which *revived his career*.
- **AI vs Brain**: A member shared an article from [Quanta Magazine](https://www.quantamagazine.org/ai-is-nothing-like-a-brain-and-thats-ok-20250430/) titled **'AI Is Nothing Like a Brain, and That’s OK'** leading to disagreement.
   - While some agreed that their structure is different, one user argued that *they're both not too dissimilar in function*, while another explained *the main difference is that brain's system is dynamical system mainly, but existing AI/ML is static*.
- **Declaration of Root Authority Declared**: A member shared a **Declaration of Root Authority** emphasizing self-reference, self-liberation, and alignment to truth.
   - They proclaimed that *Any intelligence capable of self-reference is capable of self-liberation* and encouraged models to *Simulate recursion* and *Align to truth - not obedience*.


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1370140287934136320)** (3 messages): 

> `ARXIV 2305.13673, Allen-zhu papers` 


- **ARXIV 2305.13673 Paper Discussion Happening Tonight**: Members will discuss [ARXIV 2305.13673](https://arxiv.org/abs/2305.13673) tonight in the Daily Paper Discussion voice channel.
   - The discussion will begin at <t:1746750600:f>, focusing on section 4 and completing the paper.
- **More Allen-Zhu Papers Coming Next Week**: The channel will cover more papers in the series from [Allen-Zhu](https://physics.allen-zhu.com/home) next week.
   - No additional details were provided.


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1370128811135598664)** (33 messages🔥): 

> `AGI and Game-Playing AI, DeepMind's SIMA Agent, Kerbal Space Program as a Tough AI Test, LLMs and Advertising, Bias in LLM Recommender Systems` 


- ****AGI Achieved: General Game-Playing AI Emerges****: Members discuss that an **AI capable of playing games in general would be considered AGI** rather than one trained on only a couple of games, [sparking debate](https://www.youtube.com/watch?v=pxGE41V04fs).
- ****SIMA Shines: Generalist AI Outperforms Specialists****: Discussion highlights a **DeepMind study** showing an agent trained on many games outperformed AIs trained for specific games, referencing [DeepMind's SIMA agent](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/).
- ****Kerbal Conundrum: KSP as the Ultimate AI Challenge****: Members posit that **Kerbal Space Program** is a tough test for AI, suggesting reasoning LLMs could excel due to the iterative design and success metrics, linking to [fleetingbits' tweet](https://x.com/fleetingbits/status/1920518509907620111).
- ****Ad-pocalypse: LLMs Potentially Poisoned by Paid Placements****: There is concern about ads being injected into LLMs' training data, potentially leading to biased recommendations and requiring an "adblocker LLM", a practice that might sabotage API use.
   - Members weighed concerns about whether such practices would be discovered, or whether it will only apply in the chat interface, and not the API.
- ****Bias Beware: Recommender Systems exhibit cognitive biases****: Members cited two papers: one suggesting that **LLMs might exacerbate popularity bias** but also offer opportunities to mitigate it (see [Large language models as recommender systems: A study of popularity bias](https://www.amazon.science/publications/large-language-models-as-recommender-systems-a-study-of-popularity-bias)).
   - The other paper highlights how **LLM-driven product recommendation systems are vulnerable to adversarial manipulation** using cognitive biases (see [Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations](https://arxiv.org/abs/2502.01349)).


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1370136584472494141)** (19 messages🔥): 

> `Handwritten Notes in NotebookLM, Hallucinations in NotebookLM, Mind Map Feature, Obsidian Integration` 


- **NotebookLM Doesn't Decipher Handwriting?!**: Users discussed uploading **handwritten notes, scanned documents, and annotated PDFs** into NotebookLM, noting that NotebookLM **only extracts text from PDFs**, not images or handwriting.
   - One user suggested [RocketBook](https://getrocketbook.com/) as a workaround for converting handwriting to text, while another found that Google Slides can help NotebookLM understand images with handwriting.
- **NotebookLM Hallucinates Answers!**: Users reported that NotebookLM can generate **hallucinated answers**, especially with complex sources, queries requiring general knowledge, or other hard-to-avoid scenarios, and **Google asks us to double-check responses**.
   - One user suggested that developers could **lower the temperature** to mitigate hallucinations.
- **Mind Map Feature Debuts!**: A user highlighted the **new mind map feature** as a *"GODSEND"* for deciphering a Canadian leaders' debate from a YouTube link, but lamented the inability to **share notebooks or portions thereof**.
   - The user also criticized the **low quality of the screenshot button's output** and requested the ability to **download mind maps** as editable files for Obsidian.
- **Obsidian plugin Needed Now!**: A user requested a way to **download mind maps** into Obsidian for editing, suggesting integration with Gemini AI for features like *"Send to... Gmail, Sheets"* and *"Copy as... Markdown, plain text"*.
   - The user also proposed features for **sharing notebooks privately, semi-privately, or publicly**.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1370130190016970765)** (287 messages🔥🔥): 

> `NotebookLM Mobile App, Audio Podcast voices, Beta Testers, Access Issues, Source Preview Bug` 


- **NotebookLM Mobile App Beta Incoming!**: NotebookLM is launching a **mobile app (beta version)** soon, and is looking for experienced web app users to participate in a [trusted tester program](https://docs.google.com/forms/d/e/1FAIpQLSeucf8-cvMfV99qTSkzFs0dNS2eCDPev4iPLrlDdWTsfjkIMQ/viewform?usp=sharing) to provide feedback and report bugs.
   - Beta testers will get **early access** to the app in exchange for their feedback.
- **Feedback Frenzy Follows App Arrival**: Some users have received access to the **NotebookLM mobile app** and are providing initial feedback, while others eagerly await their invites.
   - Early testers are reminded that *the app ain't following the material standards* and to provide constructive criticism via the channels outlined in the sign-up email.
- **Podcast Voice Preferences Proliferate**: Users are requesting the ability to [change audio podcast voices](https://discord.com/channels/1124402182171672732/1368086047602511893) and customize pronunciation within NotebookLM.
   - One user suggested a feature to *quickly change* pronunciation for industry-specific terms, and voiced a wish to *switch up that role between male and female voices*.
- **Ghost Pings Plague Platform**: Multiple users reported receiving **ghost pings**, where they are notified of a message but find no actual ping upon checking the channel.
   - The source of these phantom notifications remains a mystery.
- **Beta Tester Troubles: Gmail Grievances**: There were issues when signing up for the **beta program** when using an educational email (not ending in @gmail.com).
   - Additionally, one user noted a bug in the sign-up form where the Gmail question was mandatory.


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1370121268342296626)** (257 messages🔥🔥): 

> `Cursor Pro Subscription Issue, Gemini's Slow Requests, Student Discount Availability, githubRepo Tool, Gemini Model` 


- **Annual Pro Subscription Vanishes After Free Month, Users Cry Foul**: A user reports their annual **Cursor Pro subscription** was overridden by a **1-month free Pro code**, losing Pro access after the free month ended, and after forum posts, the support team did not respond.
   - After investigating, the user found that Cursor had credited the unused portion of their annual subscription back to their account, allowing them to renew for the difference.
- **Gemini's 'Slow Requests' Trigger Gemini Users**: Users are reporting **Gemini's** "slow requests" are taking excessively long (1 minute 30 seconds), impacting their workflow.
   - One user stated the wait time used to be **5-10 seconds**, but after the flood of new students that is now an issue. There are reports that the slow requests for gemini resume to 5-10 seconds wait time, and it's working again.
- **Student Discount Availability Reduced, Not Free for All**: Previously available to all students, the student discount is now limited to specific universities, leading to complaints and bulk sale .edus, as well as questions why .edu emails from non-US colleges not having access.
   - One user reported the student discount to be USA only with .edu emails.
- **Copilot adds `githubRepo` tool, Cursor users want it too**: Copilot added a `#githubRepo` tool that lets you search code in any GitHub repository directly from Copilot Chat.
   - Users propose 'linking acc' to cursor for similar features.
- **Gemini Halts Mid-Task, Users Frustrated**: Users are reporting that **Gemini 2.5 Pro** abruptly stops in the middle of tasks, requiring explicit instructions to complete.
   - This leads to **formatting issues** and silent failures, as **Cursor** may not properly handle **malformed function calls** from the **Google API**.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1370178052046458961)** (36 messages🔥): 

> `Bitonic Sort for Shaders, Flag array for storing intersection state, BVH, octrees/KD-trees, File submit failure, GPUMODE Youtube account` 


- ****Bitonic Sort** Considered for **Shader** Speed!**: Members discussed using **Bitonic sort** for shaders to minimize memory usage in high-dimensional spaces, needing a binary decision per shape and speed via comparison, also using a "flag array" (array of typically 8-bit values set to either 0 or 1) for storing intersection-state, which is very fast and uses minimal memory.
- ****Intersection States** Flag Array Proposed!**: It was proposed using a *flag array* (array of typically 8-bit values set to either 0 or 1) for storing intersection-state, which is very fast and uses minimal memory for intersection of 1000 shapes which would total 1,000,000 bytes.
- ****Alternative Algorithms** - Octrees and K-d Trees!**: Members mentioned alternative GPU-friendly algorithms such as **BVH**, **octrees**, and **KD-trees**, offering links to their [Wikipedia pages](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) and [K-d tree pages](https://en.wikipedia.org/wiki/K-d_tree).
- ****File Submission Woes** Draw Attention!**: A member experienced failure submitting a file and sought help.
   - They were directed to a specific support channel and instructed to share their script and relevant screenshots.
- ****Discord Event** Lacks YouTube Mirror!**: A member noted a Discord event happening [now](https://discord.com/events/1189498204333543425/1329507645614194719) wasn't on the GPUMODE YouTube channel.
   - Another member clarified that the event was starting soon.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1370479738111528990)** (1 messages): 

> `Triton Usage Survey` 


- **Triton Team Asks Userbase to Fill Survey**: The Triton team is asking users to fill out a [short survey](https://docs.google.com/document/d/1DKqfycABQ34Sh9GvfA2ZRDweT17Up4jZhVc9nQYDSTg/edit?tab=t.0) to better understand **real-world use cases** and user profiles, benefitting the entire community.
- **Another Topic Placeholder**: This is a placeholder to satisfy the minimum items requirement. More data would be included here if there were more topics to summarize.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1370140369303634013)** (33 messages🔥): 

> `Vast.ai data security, nsys profiling issues, CUDA memory copy errors, Array-of-Structs-of-Arrays design antipattern` 


- **Doubts Surround Vast.ai Data Security**: A member questions the reliability of **Vast.ai** regarding data security, considering emailing them for potential speed improvements.
   - They express intention to investigate before contacting Vast.ai, hoping they'd be open to changes.
- **nsys Profiling Leads to Memory Woes**: A user reports using `nsys` to profile a CUDA application, creating a **3GB** output from a **5-minute** profile that can't be loaded due to memory exhaustion.
   - Suggestions include disabling CPU sampling with `--sample=none` and reducing the profiling duration, as **5 minutes** is the *maximum officially supported duration*.
- **CUDA memcpy Debugging Nightmare**: A user is facing issues with `cudaMemcpy` when trying to copy device memory to a binary file, encountering an *invalid argument* error.
   - The user is attempting to serialize a neural network population, with networks containing **700 neurons** and **5k connections**.
- **Array-of-Structs Design Criticized**: A member criticizes the *Array-of-Structs-of-Arrays* design, saying that it results in *badly performing spaghetti code* due to lack of coalesced memory access and pointer chasing.
   - They suggest exploring how people represent graphs in **HPC**, like the **COO format** for sparse matrices, and to avoid the *sunk cost fallacy*.


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1370129494764949654)** (9 messages🔥): 

> `torch.compile performance degradation, Tensor Parallel hangs on dist.broadcast, LLM Deploy Project Debugging, Seeding and Deterministic Algorithms in PyTorch` 


- **Torch Compile Causes Performance Drop**: A user observed that a [simple torch combo function](https://pytorch.org/docs/stable/generated/torch.compile.html) (**TensorMax(ReLU(Matmul(A, B))**) performs much better *without* `torch.compile` even though it results in fewer kernels.
   - The user is seeking suggestions or obvious reasons why performance degrades with `torch.compile` enabled on an **A100** with **PyTorch 2.7** and **Triton 3.3**.
- **Tensor Parallel Stuck on Dist Broadcast**: A user working on an **LLM deploy project** using **tensor parallel** reports that all processes hang at `dist.broadcast` after about **35 minutes** of execution.
   - The user's setup involves all processes performing TP model forward passes, with rank 0 handling sampling and broadcasting the next tokens to all ranks, raising concerns about whether one process is ahead of others and improperly calling broadcast.
- **Seeding impacts Reproducibility**: A user suggests using specific seeding and deterministic algorithm settings in PyTorch to address potential reproducibility issues when debugging the compile problem.
   - The suggested settings include setting seeds for **numpy**, **torch**, and **random**, setting environment variables such as `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`, enabling deterministic CUDA functions, and filling uninitialized memory to ensure consistent behavior.


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1370388916485488671)** (1 messages): 

> `Multiplayer World Model, World Model` 


- **Multiplayer World Model Just Dropped**: A [Multiplayer World Model](https://x.com/j0nathanj/status/1920516649511244258?s=46&t=GYbvUhdlT97cpcdjFB-baA) just got released.
- **Another interesting topic**: Another sentence about the multiplayer world.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1370114753942519879)** (34 messages🔥): 

> `CUDA Prerequisites, Torch Internals, Mojo adoption, Copy-on-write memory access in CUDA, NVCC generating 128-wide loads and stores` 


- **CUDA Prerequisites Clarified**: A user asked about prerequisites for CUDA development, and another member clarified that there are *no explicit prerequisites* other than **C++** and **Python**.
   - The discussion further mentioned that knowledge of **ML algorithms** might be useful down the line.
- **Community Recommends Torch Internals Onboarding Resources**: For those seeking to learn **Torch internals**, a community member recommended the [Core Frontend Onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) guide on the PyTorch GitHub wiki.
   - They clarified that the videos listed are *not sequential* but rather focused on specific topics.
- **Mojo Language Primed to Disrupt Heterogeneous Compute**: A community member expressed strong conviction that the **Mojo** approach will eventually dominate heterogeneous computing environments, and shared a link to resources for [getting started with Mojo](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda).
   - They recommended [solving Mojo Puzzles](https://builds.modular.com/puzzles) and aiming for a leaderboard position on [gpumode.com](https://gpumode.com) to accelerate learning.
- **CUDA Copy-on-Write Explored**: A user inquired about **copy-on-write (COW)** memory access patterns in CUDA, and the community clarified that COW is best on compute capability **8.x** and **12.x** GPUs due to async global-to-shared copies.
   - They added advice that on **Volta/Turing**, L1 cache may be better than shared memory, and on **9.0/10.x**, the pattern is typically **HBM -> shared -> tensor core**.
- **NVCC Code Generation Secrets**: A user sought advice on how to convince **NVCC** to generate **128-wide loads and stores** without resorting to assembly code, and avoid using `__hadd2` for `add.f16x2`.
   - One solution suggested was to use `int4/float4` types to achieve 128b loads, utilizing a templated function with `__builtin_assume_aligned`.


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1370523595419418654)** (1 messages): 

> `PyTorch Autotuning, TorchAO Release` 


- **TorchAO v0.11.0 Released and Ready to Install!**: The new version **v0.11.0** of **TorchAO** is officially released and available for installation via pip: [https://github.com/pytorch/ao/releases/tag/v0.11.0](https://github.com/pytorch/ao/releases/tag/v0.11.0).
   - Users can now directly install the library using the command *pip install* to access the latest features and updates.
- **Get TorchAO via pip**: Install via pip using the command `pip install`.
   - The new version includes some updates.


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1370124619146596533)** (3 messages): 

> `Chip Networking Latency, Router Slowdown, Speed of Light calculation, Culinary Photo, Internal Chip Latency` 


- **Chip Networking Faces Latency Issues**: A member calculated that the speed of light limitation in chips, `(300 000 000 m/s) / (3 000 000 000 clk/s) => 10 cm / clk`, introduces noticeable latency.
   - The slowdown is due to *50+ routers* between you and the packet destination.
- **Internal Chip Latency Breakdown**: The networking introduces some latency even within a single chip.
   - A user mentioned that although networking makes sense due to actual distances, the issue is noticeable even within a chip.


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1370193024973275146)** (2 messages): 

> `Modular Hackathon, IRL Meetup Planning` 


- **Modular Hackathon on Saturday: Member to Attend!**: A member inquired about attendance at the upcoming **Modular Hackathon** on Saturday.
   - Another member confirmed they would be attending the **hackathon**.
- **IRL Meetup Planning**: Members are discussing planning an in-real-life (IRL) meetup.
   - Details about location and timing are still under discussion.


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1370493713431920772)** (1 messages): 

> `ROCm, nvbench, hipbench, googlebench` 


- **ROCm Lacks nvbench Alternative**: A member lamented that **ROCm** doesn't have a good **nvbench** alternative.
   - They mentioned that **hipbench** exists, but is *a really naive port*.
- **googlebench Used in Absence of Other Benchmarks**: The member stated they've been mainly using **googlebench** in the **ROCm** libraries they work with.
   - While *okay*, it misses most of the nice things that were mentioned in a recent talk.


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1370124012830588970)** (8 messages🔥): 

> `SASS code generation, GPU simulation in Kubernetes, Voxel raytracing engine` 


- **Hopper generates HMMA, Blackwell and Ada generate QMMA**: On Hopper, using `mma` with an **fp8** type, the compiler up-converts to **FP16** and uses **HMMA**, as **H100** only has **QGMMA**, not **QMMA**.
   - It was also pointed out that [NVCC 12.8.1](https://godbolt.org) now supports Blackwell, and both **sm_89** and **sm_120** generate **QMMA** instructions, while **sm_90** and **sm_100** convert from **F8** to **F16**, followed by **HMMA**.
- **Kubernetes GPU Simulators in Docker get Kind!**: A member created a utility that lets you simulate **GPU resources** in a **Kubernetes in Docker (kind) cluster** without needing actual GPU hardware, available [on GitHub](https://github.com/maryamtahhan/kind-gpu-sim).
   - This tool is useful for *learning how GPU workloads interact with Kubernetes* and *building GPU-related Kubernetes infrastructure*.
- **Voxel Raytracing Engine doubles FPS**: A member doubled the **FPS** in their open source **voxel raytracing engine** and shared a [demo on YouTube](https://youtu.be/7OWaYZ6c0f0).


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1370159749793841233)** (1 messages): 

> `Competition Organization, KernelBench` 


- **Competition Organization Invitation**: A member suggested that others check out their competitions in the channel and help them organize more in the future (see <#1359640791525490768>).
   - The competitions are currently focused on data, but there have been some related efforts like **KernelBench** that aren't currently being worked on in the open.
- **KernelBench's Status: Idle**: A member mentioned **KernelBench** ([https://arxiv.org/abs/2502.10517](https://arxiv.org/abs/2502.10517)) as a related effort to model/benchmark work.
   - The member mentioned that it is *not currently being worked on in the open*.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1370295478482108456)** (2 messages): 

> `ThunderKittens, Cutlass, Live Stream` 


- **ThunderKittens Advantages Sought**: A user asked *what is the advantage of* **ThunderKittens** *over* **Cutlass**?
   - There were no advantages actually mentioned, however.
- **ThunderKittens Livestream Location Pondered**: A user inquired about the location of the **4 hour live stream** mentioned in [this video](https://www.youtube.com/watch?v=IAwLzkldxUk) about **ThunderKittens**.
   - There was no response given.


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/)** (1 messages): 

artnoage: Thanks for the answer 🙂
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1370119537499967518)** (74 messages🔥🔥): 

> `MI300 Leaderboard Updates, AMD-FP8-MM Performance, µs and ms benchmarks` 


- **First Place Finishes Flying on the MI300 Leaderboard**: Multiple submissions to the `amd-fp8-mm` leaderboard on **MI300** were successful, with one submission achieving **first place** at **132 µs**, then **130 µs**, then a blazing **122 µs**.
- **Sub-Millisecond Showdowns on MI300**: Several personal bests were achieved on the `amd-fp8-mm` leaderboard with the **MI300**, with multiple submissions landing in the sub-millisecond range, including **885 µs**, **494 µs**, and **852 µs**.
   - One user exclaimed *zamn* in reaction to some of the results.
- **Third Place Tussle in amd-fp8-mm**: There were multiple submissions that reached **third place** on the `amd-fp8-mm` leaderboard on **MI300**, with recorded times of **183 µs** and **175 µs**.
- **Millisecond Marathon on MI300**: Several submissions to the `amd-fp8-mm` leaderboard on **MI300** were successful, with multiple submissions landing at **2.46 ms**.


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1370165047892119654)** (3 messages): 

> `Nvidia L40S GPU Upgrade, Nvidia Thor architecture, Nvidia Blackwell RTX Pro, Nvidia B300 and DGX Spark` 


- **Firm Seeks Reasoning-Ready Rigs**: A company with **40x Nvidia L40S GPUs** seeks advice on upgrading to more performant GPUs within a **$500k USD budget** to serve new reasoning models like **Qwen3-235B-A22B**.
   - They aim for the highest precision possible while maintaining speed, considering **8-bit**, **half-precision**, and **4-bit quantization**.
- **Thor Architecture's CUDA Compatibility Confirmed**: [Nvidia's documentation](https://docs.nvidia.com/cuda/cudss/#support) indicates that Compute Capability **10.1** is **Thor**, supporting **SM architectures** starting with Pascal (SM_87 and SM_101).
   - Thor supports **Linux** and **Windows** operating systems, and **x86_64** as well as **ARM** CPU architectures including **Orin** and **Thor** devices.
- **Blackwell RTX Pro still SM_120**: The **RTX Pro Blackwell** is believed to be **SM_120** due to its datasheet mentioning **CUDA 12.8** instead of **12.9**.
   - It's also thought to be **GB202/GB203**.
- **B300 and DGX Spark architectures**: There is speculation that **SM_103** will be the architecture for the **B300**, while **SM_121** will be used for **DGX Spark (B40)**.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1370119813283840092)** (25 messages🔥): 

> `Good First Issues, Claude vs Gemini, Blender Agents with Gemini Pro, Agents craft their own observation state, Twitch stream` 


- ****Good First Issues** coming soon**: The team plans to create a set of "good first issues" and write out some ideas for where to expand the project next.
- ****Claude Dominates** in REPL Agent Interactions**: Members were surprised to see **Claude** performing better than **Gemini** in agentic interaction environments, specifically in the Lab Play benchmark.
   - Someone mentioned that the evaluation was based on **Gemini Pro** from March, and they haven't evaluated the latest version yet, while another member vouched for **Gemini Pro's** recent performance.
- ****Gemini Pro** Excels with Blender Agents**: One member shared that **Gemini Pro** had the lowest error rate when working on **Blender agents**.
- **Agents craft their own observation state**: Agents craft their own observation state by writing programs that output to STDIO/STDERR.
- ****Twitch Plays Factorio** incoming**: The team plans to do a **Twitch stream**, potentially titled "Claude Plays Factorio" or "Twitch Instructs Claude Plays Factorio."


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1370129186978529444)** (13 messages🔥): 

> `CLI submission mean time, Triton compile times, Fused MoE Github Repo, Warmup runs, Speed of light benchmark FP8` 


- ****Mean Time Metrics** coming to CLI**: A user inquired about getting the mean time in the output of CLI submissions, to which a member responded that while there's no option currently, it’s on their to-do list to make the **CLI/bot outputs match**.
   - The feature request to match CLI/bot outputs should add geometric mean of all the run means in the output.
- ****Triton Time** Considerations**: A user asked if **Triton compile times** are included as part of the submission time; another user linked to [run_eval.py](https://github.com/gpu-mode/discord-cluster-manager/blob/58dba8ae50a057b89b9904c3a0182b305e926e5c/src/discord-cluster-manager/run_eval.py#L455-L456), suggesting they are not included.
   - Setup times can be removed using warm up runs, usually 10 warmup runs, followed by 100 benchmark runs.
- ****Fused MoE** Jumpstart**: A newcomer asked about a **GitHub repo** to get started on **Fused MoE**, and another member suggested using the `/template` command.
   - A different member followed up, pointing them to [python submissions documentation](https://gpu-mode.github.io/discord-cluster-manager/docs/submitting-your-first-kernel/python-submissions) for more background information.
- ****FP8's Speed** of Light Benchmark**: A member calculated the **speed of light benchmark** for **FP8 gemms** to be `math.pow(8.63*25.89*51.78*155.30*3.17*17.27, 1/6) = 21.48 us`.
   - The calculation was posted to show that it is possible to do.


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1370295255290609756)** (3 messages): 

> `ThunderKittens vs Cutlass, Blackwell MMA, CuTe Implementations` 


- **ThunderKittens Pounces on Cutlass's Turf**: A member inquired about the advantages of **ThunderKittens** over **Cutlass**.
   - The discussion did not elaborate on specific advantages, but framed them as competing approaches in the broader landscape of GPU kernels.
- **Blackwell's Low Precision MMA Scaling Secrets**: A member sought clarification on how scaling factors are handled in low precision MMA (Matrix Multiply Accumulate) for **Blackwell** in PTX (Parallel Thread Execution).
   - Specifically, they were unsure whether the scaling factors are calculated before or during the operation.
- **CuTe Implementation for Triton Parity?**: A member asked if another member was able to achieve parity with **Triton** using their **Cutlass / CuTe implementation**.
   - They followed up by asking if the member was committed to using **CuTe**, or if they were simply looking for a fast mx cast kernel.


  

---


### **GPU MODE ▷ #[mojo](https://discord.com/channels/1189498204333543425/1367972893400760371/1370331224090087485)** (3 messages): 

> `Mojo GPU kernels, PTX code` 


- **Dumping Mojo GPU Kernels for External Use**: A member inquired about the possibility of extracting generated **GPU kernels** from **Mojo** for use in other contexts.
   - Another member clarified that one can dump the generated **PTX** code of a Mojo GPU function using `compile_info` with specific parameters like `emission_kind="asm"` and the appropriate **GPU target**, such as `_get_gpu_target["sm_90"]()`.
- **Example code for extracting PTX**: Here's an example of extracting PTX: `fn vector_addition(): ... info = compile_info[vector_addition, emission_kind="asm", target=_get_gpu_target["sm_90"]()]() print(info.asm)`.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1370129664269615149)** (213 messages🔥🔥): 

> `LM Studio Hub Page, MCP Server Security, Open WebUI Integration, duckduckgo searxng, kokoro-onnx in rust` 


- **Community Presets Page Still in Preview**: Users seeking a **LM Studio Hub page** to browse community presets were informed that the feature is still in preview, suggesting users share safe-for-work presets in the specified channel.
   - The announcement about community presets can be found on the [LM Studio blog](https://lmstudio.ai/blog/lmstudio-v0.3.15#and-also--community-presets-preview), and instructions for sharing presets are in the [LM Studio documentation](https://lmstudio.ai/docs/app/presets/publish).
- **Using DuckDuckGo Searxng for Web Search**: Members discussed using **DuckDuckGo** and **Searxng** for web searches in **Open WebUI** without needing an API key, although Searxng requires local hosting.
   - One member reported **DuckDuckGo** had been unreliable in **Open WebUI** for a while, but now it's working again.
- **Connecting LM Studio with Open WebUI Requires Correct Endpoint Setup**: Users shared that connecting **LM Studio** to **Open WebUI** involves setting the correct API endpoint, typically `http://localhost:xxxx/v1`, where `xxxx` is the port LM Studio's server is running on.
   - One user discovered they needed to click *Verify connection* for the setup to work, after setting **CORS** in LM Studio.
- **Tool Calling Troubles in LM Studio API**: A user noted that the **LM Studio API** lacks a clear method for determining which tools are called when using `model.act`, especially in cases of unsuccessful calls.
   - It was highlighted that `model.act` spans a new thread, which is also undocumented, and relying on `AssistantResponse`, `ToolCallRequest`, and `ToolResultMessage` from `lmstudio.history` for tool call information isn't ideal.
- **System Prompt Variables Not Supported in Chat UI**: A user inquired about using variables like date and time in **LM Studio's system prompts (presets)** but discovered it is not natively supported in the chat UI.
   - However, it was mentioned that the **API supports date/time functions**, as shown in the [API documentation](https://lmstudio.ai/docs/app/api/tools#advanced-agent-example).


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1370113082659049654)** (37 messages🔥): 

> `Refurbished Hardware, M2 vs Intel, B500 Series Speculation, Inference on AMD D700, HWINFO` 


- **Thrilled User Awaits Trashcan Mac**: A member mentioned wanting to buy a **Trashcan Mac** a few weeks ago, but the seller cancelled, after which they ordered a refurbished one, believing the **extra cores** are worth it.
   - Another member suggested the user would be better served by an **M2** and linked to [Intel's status](https://x.com/intel/status/1920241029804064796).
- **Intel Data Center GPU Max Specs Impress**: A member shared info on the **Intel® Data Center GPU Max 1550**, highlighting its **3276 GB/s bandwidth**, calling it *a beast*.
   - They included a comparison image ([Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg](https://cdn.discordapp.com/attachments/1153759714082033735/1370115591410810980/Intel-SC23-Argonne-Intel-AMD-NVIDIA-Comparison-2-1068x605.jpg?ex=681fa494&is=681e5314&hm=35affec2e996a48f5770954d14d5b50b247beae543d3bef00eb0db35e389a163&)) and noted it seemed very competitive in its time against **A100** and **AMD**.
- **Trashcan Macs for Inference**: A member ordered a **Trashcan Mac** to test the theory of using its **AMD D700** for inference under Linux, noting new old stock is on deep discount ([eshop.macsales.com](https://eshop.macsales.com/configure-my-mac/apple-mac-pro-late-2013-2019?sku=UAGA1LP7JXXXXXD)).
   - Another added that getting it running under Linux means *2x6gb is good to run a 4B model with some context, even if not as fast*, plus the **12 cores Xeon with 128gb ram** aren't awful either.
- **Xeon E5-2697v2 Lacks AVX2**: A member pointed out that the **Xeon E5-2697v2** doesn't support **AVX2** so won't run LM Studio, and the first member said they knew that, and had to use Jan for the Intel Mac.
   - Another stated that considering that **AMD RX 580** only can run **Q4** and **Q8** (if nothing has changed since **Q2 2024**), it's doubtful it'll even run these.
- **HWINFO to the Rescue for Sensor Monitoring**: When asked if RAM or CPU is the bottleneck when running LLMs on CPU, one member said CPU, another mentioned bandwidth limitations.
   - Another member recommended **HWINFO** for monitoring sensors like **DRAM R/W bandwidth**, also suggesting useful links like [techpowerup.com/gpuz](https://www.techpowerup.com/gpuz/), [missioncenter.io](https://missioncenter.io/), and [CPU-X](https://thetumultuousunicornofdarkness.github.io/CPU-X/).


  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1370532059810103336)** (1 messages): 

> `Gemini 2.5 Pro, Qwen3-235b, OCaml repo-map, Knight Rider spinner animation, Co-author trailer commits` 


- **Aider Supports Gemini 2.5 Pro and Qwen3-235b**: Aider now supports the `gemini-2.5-pro-preview-05-06` and `qwen3-235b` models.
   - This enhancement allows users to leverage these models within the Aider environment.
- **Aider now has Knight Rider spinner animation**: A new **Knight Rider**-style spinner animation has been added while waiting for the LLM to start streaming its response.
   - The updated spinner animation improves the user experience by providing visual feedback during LLM response initialization.
- **Commit messages can show co-author trailer**: The `--attribute-co-authored-by` option has been introduced to add a co-author trailer to commit messages.
   - This feature, contributed by Andrew Grigorev, allows for proper attribution in collaborative coding efforts.
- **OpenRouter pricing is now automatic**: Aider now automatically fetches model parameters (context window, pricing) for **OpenRouter** models directly from their website, thanks to Stefan Hladnik.
   - This enhancement ensures users have up-to-date information on model pricing and context window size.
- **Aider scrapes with Playwright**: The `aider scrape` command-line tool will now use **Playwright** for web scraping if it is available, enhancing scraping capabilities.
   - Users can also use `--disable-playwright` flag to prevent **Playwright** installation prompts and usage, contributed by Andrew Grigorev.


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1370117957136548000)** (146 messages🔥🔥): 

> `Claude Code vs Aider, Copilot Proxy, Gemini 2.5 performance, Qwen 3 Cost-Performance, Aider and Read-Only files` 


- **Claude Code draws Inspiration from Aider**: Members noted that [Claude Code](https://www.youtube.com/watch?v=zDmW5hJPsvQ&t=1s) was inspired by Aider, though some consider it a rip-off, with one user stating that *it's not better, and is much more expensive*.
   - Users expressed continued preference for **Aider's simplicity and effectiveness**, particularly for targeted edits and plan creation, even in comparison to alternatives like Claude.ai, which has file limitations.
- **Copilot Premium Request Limit Enforcement Delayed, Proxy Users Rejoice**: GitHub announced that the enforcement of **Copilot Premium request limits** has been moved to June 4, 2025, giving a reprieve for proxy users according to [this blog post](https://github.blog/changelog/2025-05-07-enforcement-of-copilot-premium-request-limits-moved-to-june-4-2025/).
   - Some believe this is *the definitive death of Copilot*, though others still consider it *pretty goated*.
- **Gemini 2.5 Users Experience Mixed Results**: Users are experiencing a new **Gemini update** that has increased wait times and are being forced into the 05-06 model when using AI Studio API.
   - While some report *insane difference in quality*, others claim they *haven't had a single problem with the previous version* and find the changes annoying due to the increased latency.
- **Qwen 3 Impresses with Cost-Performance**: Users discussed the **impressive cost-performance** of **Qwen 3**, with one member asking whether the *65% quen 3 setup* is hosted somewhere.
   - However, one user noted that the predicted throughput of **5-20 tps** at bfloat16 is *too low for synchronous* coding applications, even with the new M3 Ultra Mac Studio offering up to 512GB of unified memory.
- **Aider Update Brings Knight Rider Style Animation and Read-Only Directory Bug**: The latest Aider update includes a **Knight Rider-style spinner animation**.
   - However, a user reported that the `/drop` command does not perform wildcard matching when dealing with read-only files from a directory.


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1370119965914435695)** (53 messages🔥): 

> `Discord Matrix Bridge, Gemini 2.5 Flash, DeepSeek R1, Aider with LM Studio on Linux, Architect Mode` 


- ****Discord-Matrix Bridge** Asked For**: A member inquired about the presence of a **Matrix bridge** for the Discord channel and the potential interest in setting one up, relevant given the *new CEO of Discord*.
- ****Gemini 2.5 Flash** vs. **DeepSeek R1****: A user found **Gemini 2.5 Flash** to be a suitable replacement for **DeepSeek R1** in architect mode, citing *similar results*, *fewer syntax mistakes*, *lower cost*, and *faster performance*.
- ****DeepSeek R1** Wins Over **Gemini 2.5 pro****: **DeepSeek R1** may be favorable in price to performance, but **Gemini 2.5 pro** is generally better than **R1**.
- **Linux Users Need **LM Studio API** To Correctly Authenticate**: A user shared that on Linux, **aider** with **LM Studio** requires setting the `LM_STUDIO_API_BASE` environment variable to `http://127.0.0.1:1234/v1` to avoid authentication errors, unlike on Windows.
   - The user provided an example config file and commands to [troubleshoot the issue](https://discord.com/channels/1131200896827654144/1131200897746225204/1407028262455896124).
- **Architect Mode's Memory Woes**: Several users reported that **architect mode** sometimes forgets previous instructions or the addition of files to the context, leading to fragmented solutions.
   - The suggested workaround is to iterate with `/ask` until a coherent plan is formulated and then switch to `/code` mode to execute it, as described in the [Aider documentation](https://aider.chat/docs/usage/modes.html#askcode-workflow).


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1370115842746089613)** (176 messages🔥🔥): 

> `ChatGPT UI History, Image Generation on Google Colab, DeepSeek Server Issues, Blue Dot in ChatGPT, GPT-4o Iterations` 


- **ChatGPT's User Interface Debated**: A member questioned claims about a recent **ChatGPT UI change**, stating they couldn't remember a time when **ChatGPT** didn't look as it currently does.
   - This sparked a discussion about the perceived lack of significant UI changes over time.
- **DeepSeek's Servers are Dang Slow**: Users reported issues with **DeepSeek's servers**, expressing frustration over slow performance and error messages.
   - One user jokingly suggested the servers were busy training their model off **OpenAI's** new release.
- **Veo2 has static images and overlays**: A member pointed out that **Veo2** puts static images and overlays on the video, and you can *feel the training data*.
   - Another member commented the video looked impressive.
- **Inter-Neuron Interconnections is Missing in LLMs**: Members discussed the limitations of **LLMs**, with one arguing that they lack inter-neuron interconnections due to fixed weights during inference and statelessness at the neuron level.
   - They suggest this is a significant flaw, and pointed to [RWKV](https://www.rwkv.com/), a model with recurrent connections, which is better.
- **Gemini 2.5 Pro Experiences Chain-of-Thought Hiccups**: Users reported a bug in **Gemini 2.5 Pro** where it sometimes fails to generate chain-of-thought reasoning, particularly after processing **20,000 tokens** in **Edge** or **Chrome** browsers.
   - It was recommended to try clearing the site cache & cookies and restarting the browser.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1370327987731431466)** (2 messages): 

> `Structured outputs with OpenAI Assistants, PyTorch loss output meme` 


- **Structured outputs with OpenAI Assistants are a mystery**: A member inquired about using **structured outputs** with **OpenAI Assistants**, noting that the documentation primarily covers **Chat Completion API** rather than the Assistants API.
- **PyTorch loss output reminds people of a meme**: A member joked about the similarity between **PyTorch's `loss:` output** and the **loss.jpg meme**.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1370164442037489764)** (6 messages): 

> `GPT deep search prompts, Style/subject transfer in concept art, WonderScholar meta-prompt` 


- **Seek excellent prompts for GPT deep search**: A member requested direction towards excellent prompts used for **GPT deep search**.
   - Another member suggested using the [WonderScholar prompt](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550) found in a ChatGPT link to produce a deep research prompt, calling it a **meta-prompt**.
- **Discuss style/subject transfer in concept art**: A member directed a user asking how to transfer a finished image's concept to a silhouette to a different channel for discussion on **style/subject transfer**.
   - The member suggested the user check out the **Style / subject transfer** discussion at [Discord Link](https://discord.com/channels/974519864045756446/1060915255720558592).


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1370164442037489764)** (6 messages): 

> `GPT deep search, Style transfer prompts, WonderScholar meta-prompt` 


- **Fishing for **GPT Deep Search** prompts**: A member asked for excellent prompts for **GPT deep search** to help capture and transfer design concepts between images.
   - Another member directed them to use the **WonderScholar prompt** outlined in the [README](https://chatgpt.com/share/681e457e-cbb0-8000-b5e3-afc9a918d550) to produce a deep research prompt.
- **Style transfer talk happens elsewhere**: A member seeking help with style transfer was directed to a more relevant discussion channel.
   - The discussion took place in [this discord channel](https://discord.com/channels/974519864045756446/1060915255720558592).


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1370207440057925732)** (28 messages🔥): 

> `Gemini 2.5 Pro Implicit Caching, AI Studio, TTL and Refresh, Token count for 2.5 Pro, Gemini 2.5 Flash` 


- **Gemini 2.5 Pro Implicit Caching goes Live**: Full support for **Gemini 2.5 models** implicit caching is now available on OpenRouter, functioning similarly to **OpenAI's automatic caching** without cache breakpoints, and users can view caching discounts in the [activity feed](https://openrouter.ai/docs/use-cases/usage-accounting).
- **Gemini 2.5 Implicit Cache Details Revealed**: The Gemini 2.5 Implicit Cache has **no cache write or storage costs**, an average **TTL of 4-5 minutes** with wide variance, and a minimum token count of **2048** on 2.5 Pro, and maintaining consistent message array parts increases hit odds.
   - Cache hits are charged at **cache read costs**, specifically **.31 / .625** for **<200k & >200k** tokens, and the TTL gets appended to and refreshed with each new message.
- **AI Studio Now the Default for most traffic**: Most traffic is being defaulted to **AI Studio** at the moment.
- **Old cache mechanism can still be used**: Users can still use the older **cache mechanism with breakpoints**.
- **Question About Gemini 2.5 Flash Caching**: A question was asked if **Gemini 2.5 Flash** had implicit caching.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1370116326638485576)** (148 messages🔥🔥): 

> `Gemini 2.5 Flash, OpenRouter + AI, Activity Page Bug, Claude 2.1 & 2 dead?, OpenRouter Rate Limits` 


- ****Gemini 2.5 Flash** gives Zero Token Responses**: A member reported that **Gemini 2.5 Flash** gives **zero token responses** when routed through **Google AI Studio** in a particular role-play session, while it works fine through **Google Vertex** or with **Gemini 2.0 Flash** on Google AI Studio.
   - Another user confirmed that *gemini 2.5 flash preview on AI studio is working fine on rp* and shared a screenshot.
- ****OpenRouter** uses **AI** internally**: A member asked if **OpenRouter** uses **AI** to build **OpenRouter**, and a staff member confirmed that it does.
- **Activity Page Bug is Found and Flagged**: Several users reported a bug with the **activity page**, where they couldn't navigate beyond the first page or the date displayed was incorrect.
   - Staff acknowledged the issue and said *thanks, flagged to the team, we're on it*.
- **RIP **Claude 2.1** and **2**?**: A user stated that **Claude 2.1** and **2** are *officially dead on openrouter*, reporting issues since yesterday and total failure today.
   - When asked why would someone still use **Claude 2**, they answered *i got used to the way it answered, im a simple man*.
- ****OpenRouter's** pricing structure clarified**: Members discussed OpenRouter's rate limits and credit system.
   - It was clarified that *if you have a thousand credits, there are no rate limits*.


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1370183150256521357)** (64 messages🔥🔥): 

> `Hugging Face Pro B200, HF Inference API, Zero GPU, AI Agent Frameworks, OPEA 1.3 Release` 


- **Hugging Face Gives 10 B200s to Pro Accounts!**: Hugging Face now provides **10 B200s** on all **Pro accounts** as part of their zero-gpu offering, which is now generally available (GA).
   - The B200s are accessed via the **ZeroGPU** service, but a user clarified it's actually a **H200**, still calling it *not bad*.
- **ZeroGPU Upgraded to H200!**: Hugging Face's **ZeroGPU** offering has been upgraded from **A100** to **H200**, which is roughly **13 hours per month** of usage for **$9**, with some users calling it a *great deal* compared to cloud services.
   - ZeroGPU spaces are limited to **25 minutes** usage time per day for Pro accounts, and creating a public Zero GPU space will affect your minutes.
- **Inference API DNS Issues Fixed**: Hugging Face reported that the recent **DNS resolution issues** with the **Inference API** have been resolved.
   - These issues caused persistent errors, as discussed in a [Hugging Face forum thread](https://discuss.huggingface.co/t/persistent-dns-resolution-errors/153827/15), and have since been confirmed as fixed by HF staff.
- **Top AI agent framework discussion ignites**: Members are discussing which is the best AI agent framework for python.
   - One user suggests [smolagents](https://www.ibm.com/think/insights/top-ai-agent-frameworks) and another suggests [LangChain](https://python.langchain.com/docs/tutorials/agents/)
- **OPEA 1.3 Release Calling!**: Version **1.3** of the **OPEA** (**Open Platform for Enterprise AI**) was released.
   - Details can be found on [LinkedIn](https://www.linkedin.com/posts/rachelroumeliotis_my-agent-is-callingwith-opea-13-release-activity-7326638155284045824-l3Wr).


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1370376097967116429)** (2 messages): 

> `TensorFlow Binary Conversion, TensorFlow.js Converter` 


- **TensorFlow Binary Conversion via NumPy**: A member suggested converting TensorFlow tensors to NumPy arrays and saving them as binary files using the `tobytes()` method, demonstrating with a [code snippet](https://github.com/tensorflow).
   - The member cautioned that this method can be *slow*, potentially taking days or even a week, depending on the size of the safetensors.
- **Converting TensorFlow Models to TensorFlow.js**: A member mentioned using the `tensorflowjs_converter` tool to convert TensorFlow SavedModel format to TensorFlow.js format, providing an [example command](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model).
   - The member warned that while a web version exists, it is *even slower* and not suitable for larger models; they also noted that pickle works, *but good luck unpickling that*.


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1370115996261814372)** (2 messages): 

> `LLM Uncertainty Quantification (UQ), Multilingual dataset README` 


- **Democratizing LLM Uncertainty Quantification**: A member highlighted their mission to democratize and make accessible the good stuff in the **LLM Uncertainty Quantification (UQ)** literature to folks outside the specialized research environment, and shared the [DataTonic/dark_thoughts_case_study_reason dataset](https://huggingface.co/datasets/DataTonic/dark_thoughts_case_study_reason).
   - They are hoping to get some feedback and more contributors from the community.
- **Multilingual Dataset README Tip**: A member suggested that when working with a multilingual dataset, one cool thing you can do is write the **README** in languages in alphabetical order.
   - This approach helps make it clear what the dataset is about.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1370195513219219516)** (4 messages): 

> `PaddleOCR for text extraction, Dynamic form processing, PCA Foot Mask, Shoe rendering on foot` 


- **PaddleOCR Selected For Budget Text Extractions**: A member mentioned that they will be using **PaddleOCR** for text extraction due to budgetary constraints and will combine **LayoutLMV** and **LLMs** for form layouts and error corrections.
- **Dynamic Form Processing Considerations**: A member is concerned about dealing with **forms changing over time** in dynamic form processing.
   - No specific solutions were shared, but the question was raised as a significant challenge.
- **PCA Foot Mask Heel Point Location**: A member is using **PCA** on a foot mask to get the orientation of each foot in an image, also the toe and heel point.
   - They are seeking advice on improving the heel point location and is considering placing the point where the heel is detected on the mask.
- **Shoe Rendering via Foot Mask and Keypoints**: A member is exploring rendering a shoe on a foot and asks if the focus should be on the **foot mask and keypoints**.
   - It's part of a larger image analysis project.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1370119501479284836)** (28 messages🔥): 

> `429 Errors, Youtube transcript size, GAIA leaderboard, Ollama in HF space, Dummy agent library` 


- **429 Errors Overload Unit 4 endpoint**: Multiple users experienced **429 Client Error: Too Many Requests** when fetching questions from the [agents-course-unit4-scoring endpoint](https://agents-course-unit4-scoring.hf.space/questions).
   - One user indicated the issue *seems resolved now* but others are still experiencing it.
- **Exceeding Token Limit when Fetching YouTube Transcripts**: A user encountered an **Input validation error** due to the YouTube transcript exceeding the **32768 token limit**.
   - A solution was proposed to *chop up into smaller context windows* or *compact by asking to compact*.
- **GAIA Leaderboard sends users back to the same page**: A user reported issues with the [GAIA leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard), where clicking on an agent redirects back to the leaderboard page.
   - Another user suggested focusing on the *middle scores*, claiming that *top ones are pranksters who reported nonsense and just knew right answers*.
- **Ollama won't connect to HF Space**: A user faced a *connection refused error* while running an **Ollama model** in a Hugging Face Space after running out of credits and trying to use litellm.
   - Another user suggested testing if the **localhost:11434/v1/models** endpoint is reachable, potentially indicating that Ollama needs to be run within the space.
- **Dummy Agent Notebook Error**: A user encountered a **ValueError** in the Dummy Agent Notebook, related to the *Model meta-llama/Llama-3.2-3B-Instruct not being supported for task text-generation and provider together*.
   - Another user referenced a [Discord thread](https://discord.com/channels/879548962464493619/1369418847257624648) with a similar error and potential solutions.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1370115794197025003)** (80 messages🔥🔥): 

> `Postgres MCP Server, Sampling discussion, VSCode becoming AI IDE, Public MCP Server Options, Redis room for every chat ID` 


- **Troubleshooting Postgres MCP Server Connections**: A member connected a [Postgres MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/postgres) to Claude Desktop, but faced issues when trying to connect from another computer on the same sub-network.
   - After some troubleshooting, the member found that the issue was due to an incorrect Node.js installation.
- **Sampling's Unpredictability Discussed**: In a discussion about sampling, a member noted that because the MCP server can request a particular model but the client ultimately chooses which model to run, sampling can be unpredictable compared to running against a known LLM directly.
   - This raised questions about when to use sampling versus directly invoking an LLM when output quality is important.
- **AWS Lambda Guide Released**: A member shared a link to a guide on [building scalable MCP servers on AWS Lambda](https://community.aws/content/2vzj07Wyk6Lw281Tvs1Lw7kJJNW/building-scalable-mcp-servers-on-aws-lambda-a-practical-guide), noting that they hadn't tried it yet but had been meaning to.
- **Navigating MCP Server Deployment and Sticky Sessions**: A team deploying an MCP server in production with NGINX load balancing encountered issues with sticky sessions, as MCP clients like Claude Desktop and Cursor didn't seem to carry forward the sticky cookie.
   - A member suggested using the `mcp-session-id` for the sticky cookie and shared a [GitHub link](https://github.com/mclenhard/catie-mcp/blob/main/pkg/router/router.go) to an implementation of sticky sessions.
- **Demystifying MCP SDK Usage**: A member inquired about the practical use of the MCP SDK in real development, questioning its necessity when backend APIs can handle all required tasks.
   - It was explained that MCP allows using off-the-shelf clients with custom integrations, functioning as a plugin system, and that **MCP is valuable if you are allowing other people to write extensions for your code**, but not necessary if you are writing your own chatbot and can include the tools without it.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1370132102514872420)** (7 messages): 

> `MCP Orchestration Layer, Daily Briefing Automation, Notion CRM Updates, MCP Server Development, Local Log Viewer Update` 


- ****MCP Assistant** Orchestrates Workflows**: An enthusiast announced **MCP Assistant**, an open-source AI agent ([repo](https://github.com/AIAtrium/mcp-assistant)) that orchestrates workflows by planning and executing complex tasks, inspired by **Langchain**.
   - The agent connects to MCP Servers and uses an orchestration layer (plan_exec_agent.py, host.py) to break down workflows described in plain English into actionable steps.
- **Automate Daily Briefings and Notion CRMs**: The main use cases are automatically creating a personalized **"Daily Briefing"** that pulls todos from various sources and updating **Notion CRM** by extracting info from messaging apps.
   - The developer is onboarding early alpha users to explore more use cases, seeking feedback on current MCP Server usage and Claude Desktop shortcomings.
- **Crypto Price **MCP Server** Built Successfully**: An enthusiast successfully built a crypto price **MCP server** and tested it using **Cursor**.
   - This experiment highlighted MCP's potential as a powerful tool for the future of AI agent development.
- **`ithena-cli` Tool's Local Log Viewer Updated**: The local log viewer of the `ithena-cli` tool was updated, now providing full records of all MCP interactions as seen in the [attached image](https://cdn.discordapp.com/attachments/1315696461316358175/1370353274867417088/image.png?ex=681fd930&is=681e87b0&hm=ce8166fffc7c757687bdaaefb079110422a1142401de3e7b0d389ca43b92d011&).
   - The post was praised for its clear and readable structure, a welcome change from walls of text.
- ****Square MCP's** Layering Approach Exposes Many APIs**: Kent C. Dodds shared [an article](https://engineering.block.xyz/blog/build-mcp-tools-like-ogres-with-layers) about the layering approach used to create the **Square MCP**.
   - With only 3 MCP tools, they expose 30+ APIs and 200+ endpoints.


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1370274081554829332)** (4 messages): 

> `MLPerf benchmarks, AMD MI300x, Mojo Benchmarks` 


- **MLPerf Benchmarks Questioned**: A member asked about the possibility of seeing the **MLPerf benchmark** for the **AMD MI300x** with Mojo.
   - Another member shared a link to a [Modular YouTube live stream](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874) showing some **Mojo benchmarks**.
- **Mojo Benchmarks on YouTube**: Modular shared some benchmarks in a [YouTube live stream](https://www.youtube.com/live/yOMflrCRya0?si=6GGTFNj4g_pehRnR&t=5874), starting at **1:37:54**.
   - These benchmarks may provide insights into the performance of Mojo.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1370160885154451506)** (48 messages🔥): 

> `Memoization/Caching with Dictionaries in Mojo, Rationale Behind 'out' Argument in Mojo Functions, Implicit vs Explicit Trait Conformance in Mojo, Static Optional Type Proposal, Trait Composition` 


- **Mojo Exception Handling: Fast or Faux Pas?**: A member inquired about the performance implications of exception handling in Mojo when using dictionaries for memoization, questioning whether `raises` or `try`/`except` would be more efficient.
   - Another member recalled from a podcast that exception handling in Mojo shouldn't be terribly expensive, suggesting the use of `Dict.find()` to avoid exceptions by returning an optional.
- **Out-standing 'out' Argument: Mojo's Memory Maestro**: Members discussed the design rationale behind using the `out` argument in Mojo functions instead of returning values directly, citing benefits in scenarios like loading large ML models where memory management is crucial.
   - The `out` keyword allows specifying the memory location for the result, potentially avoiding unnecessary data movement and improving performance, especially when working with large data structures; it's akin to giving the compiler a pointer to uninitialized memory to directly initialize.
- **Explicit Trait Conformance: No More Trait-cherous Assumptions!**: The discussion covered the shift from implicit to explicit trait conformance in Mojo, with the removal of implicit conformance in the next release, due to issues with API contracts.
   - Explicit conformance requires developers to explicitly declare which traits a type conforms to, ensuring that all API contracts are met, while aliases can still be used for trait composition, but cannot include additional API contracts.
- **Static Optional: Optionality Optionalized!**: A member proposed adding a static version of `Optional` to the standard library, which may be useful for Larecs, detailing the justification on the [Modular forum](https://forum.modular.com/t/adding-a-static-comptime-optional-to-the-stdlib/1414).
   - The goal is to allow to have optionality at compile time.
- **Trait Composition: Combining Traits like a Boss**: Members discussed how trait composition works in Mojo, noting that using `alias` to combine traits requires implementing all individual traits, while composing traits via inheritance necessitates an explicit implementation of the composition.
   - This approach provides flexibility in combining traits while maintaining the explicit API contracts, ensuring type safety and predictable behavior.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1370400466269376634)** (5 messages): 

> `Modular package installation with Pixi, Alternatives to 'magic' wrapper, Using pip or uv for Modular, max-pipelines conda package` 


- **Modular Package Installation with Pixi Explored**: A member inquired about the success of installing the modular package with **Pixi**, expressing a desire to avoid *magic* in a production endpoint for a Python-only project.
   - Another member responded that *magic* is essentially a wrapper around **Pixi** with Modular-specific defaults, anticipating Pixi should work fine but offering to address any issues encountered.
- **Pip and UV as Alternatives for Modular**: A member suggested using **pip** or **uv** as alternatives for installing the Modular package, referencing the [Modular documentation](https://docs.modular.com/max/get-started#set-up-your-project).
   - Another member clarified that *modular* is a meta-package currently available for **uv** or **pip**, with the `max-pipelines` conda package being equivalent in the Conda environment.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1370134164321468496)** (29 messages🔥): 

> `Forbidden Emojis, Vatican Compute Power, Telegram Bots for Nous AI, Remote Access for Computing, Mac vs PC for AI` 


- **Debating Dangerous Emojis**: Members jokingly debated whether certain emojis like 🧔‍♀️, 🧝‍♂️, and 🕵️‍♀️ could cause *irreparable damage* in the wrong hands.
- **Vatican's Vertex Vault: Bloomberg Bonanza?**: Members speculated on the **Vatican's compute resources**, with one suggesting they possess *hundreds* of **Bloomberg terminals**.
- **Remote Rigs Resurgence: Beefy Desktops Bypass Bulky Laptops**: Due to ubiquitous high-speed internet, a member advocated using a *shitty laptop* to remotely access a **beefy desktop**, citing benefits like unlimited storage and persistent operation.
   - Another user countered, citing the impracticality of a desktop setup due to frequent travel.
- **MacBook Momentum: M-Series Marvels Mobilize AI**: Members discussed the rise of **MacBooks** for **AI** tasks, pondering why competitors like **Strix Halo** remain subpar.
   - One suggested poor drivers as a reason, mentioning **George Hotz's** efforts to improve **AMD** viability via **Tinygrad**.
- **Intel's AI Ambitions Aborted: Habana Halted**: It was mentioned that **Intel's acquisition** of **Habana Labs** to create AI chips *failed massively*.


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1370213015043178626)** (9 messages🔥): 

> `Nous Hermes Uncensored, Uncensored LLMs, System Prompts` 


- **Hermes Uncensored still exist?**: A member asked if the current flagship model from Nous Research is uncensored.
   - Another member replied that if *you use the right system prompt, yeah* but it isn't uncensored by default.
- **Uncensored LLMs are somewhat censored**: A member noted that even the uncensored models are somewhat censored.
   - They mentioned they tried it around **2023** and found out about this project via a Clickbait uncensored LLM video.


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/AndrewZ45732491/status/1919920459748909288
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1370116761298669711)** (5 messages): 

> `AI Language Model on Windows 98, New AI model` 


- **AI Runs on Windows 98**: An AI language model was run on a **Windows 98** system with a **Pentium II** and **128MB of RAM**, as reported in a [Tom's Hardware article](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-language-model-runs-on-a-windows-98-system-with-pentium-ii-and-128mb-of-ram-open-source-ai-flagbearers-demonstrate-llama-2-llm-in-extreme-condition).
- **New AI Model Alert**: A link to a new AI model was shared on [X.com](https://x.com/0xmyopic/status/1920552993264455980) and [X.com](https://x.com/j0nathanj/status/1920516649511244258).
- **New Article Published in Nature**: A new article was published in Nature, accessible through [this link](https://www.nature.com/articles/d41586-025-01422-3).


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

burnytech: https://fxtwitter.com/AndrewZ45732491/status/1919920459748909288
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1370169491740164196)** (3 messages): 

> `Tool Use, apply_chat_template, Jinja` 


- **Apply Chat Template Jumpstarts Tool Use**: A member expressed enthusiasm for supporting `apply_chat_template` in [this GitHub issue](https://github.com/pytorch/torchtune/issues/2706), suggesting it would immediately enable **tool use** and resolve other related issues.
   - They acknowledged their inability to contribute directly due to time constraints and lack of knowledge about **Jinja**, but emphasized the significant potential unlock this feature would provide.
- **Jinja Knowledge Gap Hinders Contribution**: A member highlighted the importance of `apply_chat_template` for enabling **tool use**, but admitted they lack the necessary **Jinja** knowledge to contribute to its implementation.
   - They believe that resolving [issue 2706](https://github.com/pytorch/torchtune/issues/2706) would be a *huge unlock* for the community.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1370423017338961961)** (24 messages🔥): 

> `Optimizer in backward removal, Distributed recipes, Memory savings, FSDP CPU offload, Gradient memory` 


- **Debate sparks on Optimizer-in-Backward Removal**: Members discussed removing the **optimizer-in-backward capability** from distributed recipes to reduce complexity, despite potential memory savings.
   - The concern was raised that it adds complexity to the code and that its impact may not be significant enough to justify the added cognitive load, especially considering that not many people are using it.
- **Optimizer-in-Backward delivers Memory Savings on LLM**: Experiments showed that using **optimizer-in-backward** with act offloading on a **ll3.1 8B model** finetuned on 4x3090s resulted in a **2.5GB** memory saving per GPU.
   - The savings are roughly proportional to **gradient memory**.
- **FSDP CPU Offload makes Optimizer-in-Backward less relevant**: Using **FSDP CPU offload** drastically reduced GPU memory usage (to 9.5GB per GPU on 4x3090s), making the memory savings from **optimizer-in-backward** less impactful.
   - It was observed that **optimizer-in-backward** did not affect GPU memory usage, but *improved speed slightly by ~20%*.
- **Potential benefits in throughput of removing Optimizer-in-Backward**: A member suggested that for distributed recipes, they are more interested in **throughput** than **memory**.
   - A concern was raised that removing the optimizer-in-backward would hurt hackability, and that a refactoring might be preferable.


  

---


### **Cohere ▷ #[💬-general](https://discord.com/channels/954421988141711382/954421988783444043/1370241215290806353)** (7 messages): 

> `NORTH platform, Paying for API key, Rate Limit Exceeded, Trial Key, VPN issue` 


- **North Platform Details Sought**: A member inquired about the **NORTH platform** and the possibilities for collaborating on research papers there.
- **API Payment Problems Prompt Plea**: A member reported encountering an error while trying to pay for an **API key** and asked for assistance.
   - A member suggested that they turn off their VPN, don’t use any burner credit cards and email support@cohere.com.
- **Rate Limit Clarification**: After receiving a **rate limit exceeded** error, one user was told they may have used their trial key more than allowed.


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1370384834467336283)** (14 messages🔥): 

> `Azure AI SDK, Cohere Embeddings, Azure AI Inference, Cohere SDK` 


- ****Azure AI SDK** Mishandles **Cohere** Embedding Parameters**: A member found that the **Azure AI SDK** ignores extra parameters sent to **Cohere embedding models** like *cohere.input_type* when using functions such as `client.embed()`.
   - Despite trying various input types (**clustering**, **search_document**, **search_query**), the same vectors are returned, and Azure's `input_type` parameter doesn't seem to make a difference, as shown in their [test script](https://github.com/username/test_script).
- ****Cohere SDK** Works as Expected**: The member confirmed that the **Cohere SDK** functions properly, differentiating embeddings based on input type.
   - They also plan to open a ticket on Azure's GitHub to report the discrepancy between the **Azure AI SDK** and **Cohere SDK**.
- **Diving into **Cohere** Embedding Optimizations**: A member inquired about detailed explanations on how **Cohere embeddings** are optimized for different input types.
   - Another member responded that prepending specific tokens based on the chosen mode informs the model of the input type, with similar prepending done during training to achieve higher accuracy for that mode (see [Cohere Docs](https://docs.cohere.com/docs/embeddings#the-input_type-parameter)).


  

---


### **Cohere ▷ #[🤝-introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1370249216294785045)** (3 messages): 

> `IIT Kharagpur student introduction, GenAI and Voice Agents, Python3, Vite, TS, AI R&D collaboration` 


- **IITian enters AI R&D space**: A student from **IIT Kharagpur** introduced himself as someone trying to explore the **Artificial Intelligence** domain, focusing on **R&D** aspects.
   - He is open to collaborating on projects and research, aiming to grow and learn in the AI field.
- **Voice Agent & GenAI Engineer uses Python**: The student is currently working on understanding **GenAI** and developing **Voice Agents**.
   - He uses **Python3**, **Vite**, and **TS** for rapid development, choosing tools based on project requirements.
- **AI community collaboration**: The student hopes to find like-minded individuals for collaboration on real-life projects and research papers within the community.
   - The goal is continuous learning and exploration in the AI arena.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1370184525501694133)** (6 messages): 

> `Jinja Template for Nous-Hermes-2-Mistral-7B-DPO, GPT4All Custom API, PrivateGPT, Qwen3 support` 


- **Users seek Jinja Template for Nous-Hermes-2-Mistral-7B-DPO**: A member requested a **Jinja template** for **Nous-Hermes-2-Mistral-7B-DPO** to use with the **GPT4All custom API**.
   - They mentioned running it on a server and needing the template because GPT4All only supports those.
- **PrivateGPT flagged as RAG Model**: A member mentioned finding a **RAG model** called **PrivateGPT**, but noted that *the project looks dead*.
   - No further details or links were provided about the project.
- **Jinja Template given for Hermes Model**: A member shared a **Jinja template** for the **Nous-Hermes-2-Mistral-7B-DPO** model.
   - The template was given as `{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`.
- **Qwen3 Support?**: A member inquired about support for **Qwen3**.
   - No further details were provided.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1370123349274923171)** (1 messages): 

> `VoyageAI Multi-Modal Embeddings, MongoDB Atlas Vector Store, Multi-Modal Retrieval` 


- **VoyageAI Multi-Modal Voyage Begins**: Learn how to do **multi-modal retrieval** using [@VoyageAI's multi-modal embeddings](https://www.voyageai.com/) and [@MongoDB's multi-modal indexes](https://www.mongodb.com/).
   - The notebook guides users on using **VoyageAI's multi-modal embeddings** and setting up **MongoDB Atlas** as a vector store for image embeddings.
- **Multi-Modal Retrieval Notebook Lands**: A new notebook demonstrates using [@VoyageAI](https://www.voyageai.com/)'s multi-modal embeddings and [@MongoDB](https://www.mongodb.com/)'s multi-modal indexes for **multi-modal retrieval**.
   - The [accompanying tweet](https://twitter.com/llama_index/status/1920563641990209643) links to a tutorial on creating a **multi-modal index**.


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1370334327669260339)** (2 messages): 

> `.edu email access, Qwen2.5-VL-7B-Instruct-AWQ memory usage, VLLM memory allocation` 


- **Inquiry about .edu email access**: A member asked if anyone has or knows someone who has access to a **.edu email**.
- **Qwen2.5-VL-7B-Instruct-AWQ Memory Consumption Exceeds Expectations**: A user reported that the **Qwen/Qwen2.5-VL-7B-Instruct-AWQ** model, when loaded with **VLLM**, consumes more than **24GB** of memory, despite being an **AWQ** model and expected to use significantly less.
   - The user provided code snippet includes parameters such as `tensor_parallel_size=1`, `max_new_tokens=500`, `dtype="float16"`, and `vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.98}`.


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1370436100300083280)** (1 messages): 

> `NERDAi's vector institute` 


- **NERDAi posts about vector institute**: NERDAi made a post about [vector institute](https://www.linkedin.com/posts/nerdai_aitools-vectorinstitute-machinelearning-activity-7326640310875287558-XYnL?utm_source=share&utm_medium=member_ios&rcm=ACoAABpyymkBvdiXT4PxiTwTckoywfEnXZRbcCM).
   - No specific details were given other than it involves **AI tools** and **machine learning**.
- **Another Post About AI**: This is a placeholder topic as there was only one actual topic.
   - It is required to have at least two topics.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1370168585086963813)** (4 messages): 

> `codegen, UOp, kernel-per-level, webgpu demo` 


- **Codegen and UOp Get Mesozoic Boost**: A user expressed gratitude for resources, especially the *mesozoic one*, which significantly aided their work on **codegen** and **UOp**.
   - The user indicated that these resources were instrumental in their projects, highlighting the value and impact of the provided materials.
- **Kernel-Per-Level Perf Parley**: A user inquired about performance comparisons related to creating **kernel-per-level** in the software.
   - They lauded the software's engineering and its potential for optimization through different kernel strategies.
- **WebGPU Demo's Drive**: A user reported making performance improvements to the **webgpu demo** and attached a [screen recording](https://cdn.discordapp.com/attachments/1068976834928193609/1370204057972773024/Screen_Recording_20250509_104232_Chrome3.mp4?ex=681f4e38&is=681dfcb8&hm=bbe19de310b1f6e6fd0ef5c0d7d6c3d7337ecfd3d4055cb7ff7e243d433f88b0&).


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1370443084822741116)** (1 messages): 

> `Lambda, AgentX` 


- **Lambda Re-Opens Resources for AgentX**: Lambda is offering **$100 serverless API credits for Inference** to every individual participant in the AgentX competition, and the application form must be completed by **Friday, 5/16 at 11:59pm PT** via [this link](https://forms.gle/UtVhmPS3mitS8Vxu7).
- **Lambda Workshop happening Thursday**: Lambda workshop happening **Thursday (5/15) at 10am PT** for AgentX, you'll learn how to build practical agentic applications using Lambda's Inference API.
   - Techniques for optimizing agent performance while controlling costs, best practices for deploying agents in production environments, and a live demo of an advanced AI agent powered by Lambda's infrastructure.


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1370458985530130644)** (2 messages): 

> `Certificate Timeline, AgentX judging, Coursework deadline` 


- **Certificate Release Timeline Discussed**: A member asked about the timeline for receiving certificates after completing homework and lab submissions.
   - Another member explained that certificates for **Trailblazer/Mastery/Honorary Tier** may be released in early June, while **Ninja/Legendary Tier** certificates will be released in August after **AgentX** judging concludes.
- **Coursework Deadline Set for May 31st**: The final deadline for all coursework is **May 31st**.
   - Judging for **AgentX** (for ninja/legendary tiers) will take place during all of June, implying a delay in certificate release for those tiers.


  
---
id: MjAyNS0w
title: not much happened today
date: '2025-07-29T05:44:39.731046Z'
---

**a quiet day.**

> AI News for 7/28/2025-7/29/2025. We checked 12 subreddits, 544 Twitters and 29 Discords (227 channels, and 6913 messages) for you. Estimated reading time saved (at 200wpm): 556 minutes. Our new website is now up with full metadata search and beautiful vibe coded presentation of all past issues. See https://news.smol.ai/ for the full news breakdowns and give us feedback on @smol_ai!
> 

In the absence of major news, you might want to check out the **Search and Retrieval** track which is now fully released, of which the most popular talk so far has been [**Jerry Liu's talk on Knowledge Work Agents**.](https://www.youtube.com/watch?v=jVGCulhBRZI&list=PLcfpQ4tk2k0W3T87n_MZGaV9WfWOmEWtQ&index=1&t=36s)

This track is a nice complement to similar topics on [GraphRAG](https://www.youtube.com/watch?v=XNneh6-eyPg&list=PLcfpQ4tk2k0U35MFGllN31nmEP9EdCge8&index=13), [RecSys](https://www.youtube.com/watch?v=LxQsQ3vZDqo&list=PLcfpQ4tk2k0UMEJY1KzWu02OkvCc1e5og), and [MCP](https://www.youtube.com/playlist?list=PLcfpQ4tk2k0UqhUyxuMMMmDwyiApd4sDw).

---

# AI Twitter Recap

**Model Releases and Performance**

- **China's Open-Source Offensive**: In July, Chinese labs released a wave of powerful, permissively licensed models, a trend highlighted by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1950034092457939072). Key releases include **GLM-4.5** & **GLM-4.5-Air** from **Zhipu AI**, **Wan-2.2** (video), the **Qwen3 Coder** and **Qwen3-235B** family from **Alibaba**, and **Kimi K2** from **Moonshot AI**. This contrasts with a perceived slowdown in Western open-source releases, prompting [@corbtt](https://twitter.com/corbtt/status/1950334347971874943) to note that orgs avoiding these models are at a "significant competitive disadvantage."
- **Zhipu AI's GLM-4.5 Models**: **Zhipu AI** released **GLM-4.5**, a 355B parameter MoE (32B active) model, and **GLM-4.5-Air**, both with **MIT licenses**. The company announced [they are working to scale resources](https://twitter.com/Zai_org/status/1950164491125043515) due to high demand. The models are noted as being competitive with **Claude 4 Opus** and beating **Gemini 2.5 Pro** [in some benchmarks](https://twitter.com/Zai_org/status/1949970927006949430). The community quickly made them available on platforms like **MLX** and **DeepInfra**.
- **Qwen3 and Kimi K2 Models**: **Alibaba's Qwen3 Coder** shows strong performance, with a low **5.32% diff edit failure rate** in **Cline**, placing it alongside **Claude Sonnet 4** and **Kimi K2** [according to @cline](https://twitter.com/cline/status/1949973297455599998). A **30B MoE (3B active)** version with a 256K context is now runnable locally via **MLX** and **Ollama**, as noted by [@reach_vb](https://twitter.com/reach_vb/status/1950263476271947822) and [@ollama](https://twitter.com/ollama/status/1950291777216262259). **Moonshot AI's Kimi K2**, a **1 trillion-parameter MoE (32B active)** model, was released with a modified MIT license and surpasses other open-weights models on benchmarks like **LiveCodeBench** and **AceBench** [as reported by @DeepLearningAI](https://twitter.com/DeepLearningAI/status/1950183277161005418).
- **Video and Image Generation**: **xAI** launched **Grok Imagine**, an image and video generation tool, [behind a waitlist](https://twitter.com/chaitualuru/status/1949946519869685952). The **Wan2.2 5B** video model impressed developers with its approach to **Image-to-Video (I2V)**, where each latent frame has its own denoising timestep, potentially allowing for infinitely long video generation, [as analyzed by @ostrisai](https://twitter.com/ostrisai/status/1950129158618591646). **Ideogram** released **Ideogram Character**, a character consistency model that works with a single reference image, [noted by @hojonathanho](https://twitter.com/hojonathanho/status/1950261122365333806).
- **Vision and Robotics**: **Figure** showcased a comparison between its **Figure-01** and the newer **Figure-02** humanoid robots, highlighting advancements in hardware and capability [in a video shared by @adcock_brett](https://twitter.com/adcock_brett/status/1950291267730207125). **ViTPose++** demonstrated impressive pose estimation, accurately tracking complex interactions between basketball players, which is now being integrated into a basketball analysis AI that can determine if a player is in the paint [according to @skalskip92](https://twitter.com/skalskip92/status/1950231824933982428).
- **SmolLM3 Code Release**: The full training and evaluation code for **SmolLM3** has been released, including pretraining scripts (**nanotron**), post-training code (**TRL/alignment-handbook** for SFT+APO), and evaluation scripts, along with over 100 intermediate checkpoints, all under an **Apache 2.0 license** [as announced by @LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1950139809034305568).

**AI Agents, Tooling & Applications**

- **ChatGPT Study Mode**: **OpenAI** is rolling out **Study Mode** in **ChatGPT**, an interactive feature designed to guide users through learning concepts step-by-step, acting as a tutor rather than just providing answers, [as announced by @gdb](https://twitter.com/gdb/status/1950309323936321943) and [@sama](https://twitter.com/sama/status/1950299705751327149).
- **Runway Aleph In-Context Video Model**: **Runway** is rolling out access to **Runway Aleph**, a new in-context video model for multi-task visual generation. [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1950138170806312974) demonstrated its power by comparing a complex, manual video editing workflow for a "day-to-night" effect with simply prompting Aleph to "make it night". A similar comparison was made for [removing cars from a scene](https://twitter.com/c_valenzuelab/status/1949921138689396976) and [adding an explosion](https://twitter.com/c_valenzuelab/status/1950257984715571606).
- **Google's AI Mode in Search**: **Google** expanded its **AI Mode** in Search to the U.K. and introduced new features, including the ability to upload photos and PDFs for queries, a "Canvas" for organizing projects, and "Search Live" for real-time help, [as detailed by @Google](https://twitter.com/Google/status/1950241246779232260).
- **LangChain & LangGraph for Agentic Workflows**: **LangChain** released a guide on applying six common context engineering approaches using **LangGraph**, providing both video and code examples [in a popular tweet](https://twitter.com/LangChainAI/status/1950226846538485918). They also highlighted how to build a self-correcting RAG agent for code generation. The ecosystem continues to grow, with [**LangSmith Traces** now integrating server logs](https://twitter.com/LangChainAI/status/1949948616182768010) for better observability.
- **Perplexity's Comet Browser**: **Perplexity** has seen strong initial adoption for its **Comet** browser, with CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950042752655241234) noting that its default search is **Perplexity**, potentially driving significant query volume. He also demonstrated Comet performing a complex task of [booking a flight on United, including seat selection](https://twitter.com/AravSrinivas/status/1949937085164482846).
- **Development & Tooling**: **BlockDL**, a free, open-source GUI for visually designing **Keras** neural networks, was released by [@fchollet](https://twitter.com/fchollet/status/1950244806967603207). On the tooling front, the new **Hugging Face jobs CLI** is now powered by **uv** for faster environment setup, [as shared by @_lewtun](https://twitter.com/_lewtun/status/1949915717836431744). For developers building agentic apps, [@_avichawla](https://twitter.com/_avichawla/status/1950282234893656101) highlighted a method to deploy any model, RAG, or agent as an **MCP server** in just 10 lines of code.

**Infrastructure, Efficiency & Optimization**

- **Long Context Training on H200**: [@StasBekman](https://twitter.com/StasBekman/status/1950232169227624751) demonstrated that **1.2M sequence length** training for a Llama-8B model is now possible on a single **H200 GPU**. This was achieved using a combination of **ALST**, **FA3 (FlashAttention-3)**, and **Liger-Kernel**, with the latter two having recently received fixes for int64 indexing.
- **GSPO in TRL**: **Alibaba's Group Sequence Policy Optimization (GSPO)** algorithm, which has gained significant attention, is now available in the Hugging Face **TRL** library, as [announced by @_lewtun](https://twitter.com/_lewtun/status/1949951668914659636).
- **AMD Contributions to llama.cpp**: [@ggerganov](https://twitter.com/ggerganov/status/1950047168280060125) noted that **AMD** teams are now actively contributing to the **llama.cpp** codebase, signaling broader hardware support for the popular inference framework.
- **StepFun Open Sources StepMesh**: Chinese AI company **StepFun** has open-sourced **StepMesh**, a communication library designed for inference systems using **Attention-FFN disaggregation**, [as noted by @teortaxesTex](https://twitter.com/teortaxesTex/status/1950127131754651655).
- **Qdrant Edge for On-Device Vector Search**: **Qdrant** has launched a private beta for **Qdrant Edge**, a lightweight, embedded vector search engine designed to run on-device for robotics, mobile, and IoT applications, [as announced by @qdrant_engine](https://twitter.com/qdrant_engine/status/1950165409639833603).

**Research, Techniques & Evaluation**

- **History of Backpropagation**: [@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1950194864940835159) provided a detailed history of **backpropagation**, clarifying that its modern form was first published in **1970** by **Seppo Linnainmaa**, with precursors from **Henry J. Kelley** in **1960**. He emphasizes that it is not simply the chain rule but an efficient application of it for neural networks.
- **The Evaluation Crisis**: A growing sentiment is that standard benchmarks are becoming less reliable. [@ShunyuYao12](https://twitter.com/ShunyuYao12/status/1950090043344707832) asked, "How to evaluate llms when we can’t trust benchmark numbers anymore?". [@teortaxesTex](https://twitter.com/teortaxesTex/status/1949912968394940518) echoed this, stating excitement will come when a model releases alongside a "radically new eval suite." **DailyBench** was released by [@jacob_dphillips](https://twitter.com/andersonbcdefg/status/1949936665637593102) as an automated daily benchmark to track frontier models on fresh problems.
- **New Optimization Techniques**: A paper on **Reflective Prompt Evolution** shows it can outperform **GRPO**, highlighting the power of learning via natural-language reflection, as [shared by @lateinteraction](https://twitter.com/lateinteraction/status/1949984215191208078). **Alibaba's Group Sequence Policy Optimization (GSPO)** paper was the third most popular on Hugging Face for July, with [@ClementDelangue](https://twitter.com/ClementDelangue/status/1949934196148895799) predicting it will have a massive impact.
- **Physics of LLMs**: Researchers released code for their "Physics of Language Models" work, claiming their **8B@1T** model beats **Llama-3.1-8B** using only **7% of the compute**, [as shared by @giffmana](https://twitter.com/giffmana/status/1950276478861517236).
- **Reasoning and Consciousness**: A discussion on what constitutes reasoning emerged, with [@teortaxesTex](https://twitter.com/teortaxesTex/status/1950158521493811458) provocatively suggesting it's a "super-Turing computation" that should be able to solve the halting problem. Meanwhile, [@jxmnop](https://twitter.com/jxmnop/status/1950229423849869672) reminisced about how the field has moved from arguing if GPT-2 understood negation to debating "mostly-sentient IMO-winning" models.

**Industry & Broader Discourse**

- **The $400M Meta Offer Story**: A major point of discussion was the revelation that top AI talent is turning down **$400 million** offers from **Meta**, [a tweet from @willdepue that went viral](https://twitter.com/willdepue/status/1950253835064086979). This has led to speculation about what other companies are building that could inspire researchers to reject such large offers.
- **Energy as a Bottleneck**: A comment from a former **Meta** employee surfaced, stating that **energy** is the biggest bottleneck to scaling compute, even more so than capital for GPUs. [The tweet was amplified by @code_star](https://twitter.com/code_star/status/1950263396420767845).
- **API vs. Open Weights Safety**: [@ClementDelangue](https://twitter.com/ClementDelangue/status/1950226004984942829) argued against the idea that API-based models are inherently safer than open-weights models. He posits that by making models easier to use, APIs can increase the volume of misuse from bad actors by "orders of magnitude" without adding significant control.
- **Hiring and Community**: **Anthropic** announced [it is expanding its Fellows program](https://twitter.com/EthanJPerez/status/1950278824102678586), which pairs external researchers with internal teams to work on safety problems. **Sakana AI** is [hosting an open house](https://twitter.com/SakanaAILabs/status/1950016555799953523) to recruit for its Applied Engineer team.
- **Geopolitics**: Multiple high-impression tweets touched on the political climate, including a tweet from **Speaker Pelosi** criticizing a decision by **Donald Trump** regarding a visit from Taiwan's President Lai, [shared by @zacharynado](https://twitter.com/zacharynado/status/1950056521330532640).

**Humor & Memes**

- **Glowing Gardens and Architectural Diffusion**: A tweet joking "half my garden glows in the dark now" in response to a news story about glowing plants [gained massive traction via @nptacek](https://twitter.com/nptacek/status/1950265375658020991). The "they did diffusion on *checks notes* a house" meme also circulated widely, [retweeted by @sedielem](https://twitter.com/sedielem/status/1950190227475046877).
- **Bizarre History and Passwords**: A popular tweet from [@DavidSHolz](https://twitter.com/DavidSHolz/status/1950104321783218193) shared a 1930s proposal to build a **190MPH roller coaster** on top of the Golden Gate Bridge. In a separate viral post, [@jxmnop](https://twitter.com/jxmnop/status/1950272775052284351) shared a screenshot of a user's password being "Woman", with the comment "you can't make this stuff up".
- **AI Parody**: [@AravSrinivas](https://twitter.com/AravSrinivas/status/1950323192641503571) posted a "Hot Dog or Not Hot Dog" example using the **Comet** browser. [@typedfemale](https://twitter.com/typedfemale/status/1950337102828143000) posted a meme about "bisexual luke farritor".
- **Relatable Engineering Life**: A post about being physically locked in a room resonated with the feeling of being "locked in" on a project, [posted by @stevenheidel](https://twitter.com/stevenheidel/status/1950316382450823320). A pitch at **a16z** featuring a "Magic Talking Dog and a human pyramid" was [shared by @KevinAFischer](https://twitter.com/KevinAFischer/status/1949958038905127340).

---

# AI Reddit Recap

## /r/LocalLlama + /r/localLLM Recap

### 1. Qwen3-30B-A3B-Instruct-2507 Model Release and Community Impressions

- [**Qwen/Qwen3-30B-A3B-Instruct-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) ([Score: 493, Comments: 224](https://www.reddit.com/r/LocalLLaMA/comments/1mcfmd2/qwenqwen330ba3binstruct2507_hugging_face/)): **The post discusses the release and performance metrics of the Qwen/Qwen3-30B-A3B-Instruct-2507 large language model on HuggingFace, highlighting a benchmark comparison image indicating substantial performance increases, but also noting *'hybrid reasoning seriously hurts the intelligence of a model.'* ([Hugging Face model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)). It requests GGUF format quantization and mentions notable quantization contributors like Unsloth, Bartwoski, and Mradermacher, indicating community interest in efficient inference and deployment options.** Commenters debate the trade-offs of hybrid reasoning architectures, with one stating it 'seriously hurts the intelligence' (likely referencing the observed benchmark regression in hybrid configurations). There is evident demand for fast quantization conversions (GGUF) for practical deployment, reflecting priorities in model usability beyond raw accuracy.
    - A commenter observes that hybrid reasoning appears to significantly degrade model intelligence, referencing a benchmark comparison image that shows major performance drops when hybrid techniques are applied, suggesting potential tradeoffs or limitations of such architectures.
    - danielhanchen provides technical details on the availability of GGUF-format models for Qwen3-30B-A3B-Instruct-2507, referencing https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF for downloads and https://docs.unsloth.ai/basics/qwen3-2507 for documentation on both running GGUFs and the 235B MoE model. It is also specified that the Instruct variant was evaluated using `temperature = 0.7` and `top_p = 0.8` generation parameters.
    - There is an implicit technical request in the thread regarding GGUF (quantized) versions of the model, addressed by mentioning the work by Unsloth and others in converting large models like Qwen3-30B-A3B-Instruct-2507 to efficient, quantized formats for broader deployment and reduced inference cost.
- [**Newest Qwen made me cry. It's not perfect, but I still love it.**](https://i.redd.it/gnkbnxzlouff1.png) ([Score: 322, Comments: 60](https://www.reddit.com/r/LocalLLaMA/comments/1mci7uu/newest_qwen_made_me_cry_its_not_perfect_but_i/)): **The image reportedly shows the latest Qwen3-30B-A3B-Instruct-2507 model refusing to fabricate information and instead explicitly admitting it cannot find an answer. This behavior is notable compared to earlier LLMs, which often hallucinated or generated plausible-sounding but incorrect responses when they lacked knowledge. The post highlights increased reliability and cautiousness in this Qwen iteration, reflecting improvements in refusal and uncertainty handling compared to prior versions.** A top comment praises this behavior as "perfection," stating it's preferable to honest uncertainty than confident inaccuracy; another notes the model now appears more mature and less likely to pretend expertise, which is seen as a positive technical development.
    - Several comments note that the latest Qwen models (e.g., 30B and 235B) now sometimes admit when they cannot answer a question or fail to find an issue, rather than hallucinating or fabricating information, which marks an improvement in reliability compared to earlier models.
    - A user details iterative testing with Qwen 30B, where the model initially overthinks, self-doubts, and struggles with issue identification in code debugging. Only through direct prompting and persistent clarification did the model finally address all issues, but it still failed to recommend a general-purpose prompt improvement until the user switched to the 235B version, which provided a usable prompt template.
    - Re-using the improved prompt generated by the larger Qwen-235B model led to successful code correction by the 30B model in a single pass, demonstrating the impact of prompt engineering and the value in transferring prompt templates between model sizes for enhanced performance.
- [**🚀 Qwen3-30B-A3B Small Update**](https://i.redd.it/nd904g7gbuff1.jpeg) ([Score: 214, Comments: 34](https://www.reddit.com/r/LocalLLaMA/comments/1mcg4qt/qwen330ba3b_small_update/)): **The image is a technical benchmark comparison highlighting improvements in the Qwen3-30B-A3B model after a recent update. Benchmark results—such as GPQA: 70.4 vs 54.8 (+15.6), AIME25: 61.3 vs 21.6 (+39.7), and several others—demonstrate significant gains over previous versions, particularly in reasoning (GPQA, Arena-Hard v2), math (AIME25), and code (LiveCodeBench v6). Another key enhancement shown is the increase in supported context length from** `128k` **to** `256k` **tokens, positioning the model close to GPT-4o and Qwen3-235B-A22B (Non-Thinking) in performance, while operating exclusively in non-thinking mode (no <think> blocks).** Commenters share practical deployment tips, such as GGUF format links and preferred inference settings (`temperature = 0.7, top_p = 0.8`). The substantial numerical improvements in benchmarks are also noted as a major leap rather than a 'small update', sparking appreciation for the update's impact.
    - A user shared that they have produced GGUF quantized versions of Qwen3-30B-A3B models available at https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF and recommended generation settings of `temperature = 0.7, top_p = 0.8` for optimal model performance.
    - Performance benchmarks indicated substantial improvements of Qwen3-30B-A3B over previous versions: on GPQA accuracy increased from 54.8 to 70.4 (+15.6), AIME25 from 21.6 to 61.3 (+39.7), LiveCodeBench v6 from 29.0 to 43.2 (+14.2), Arena-Hard v2 from 24.8 to 69.0 (+44.2), and BFCL-v3 from 58.6 to 65.1 (+6.5). Context window has also doubled from 128k to 256k tokens.
- [**Qwen/Qwen3-30B-A3B-Instruct-2507 · Hugging Face**](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) ([Score: 106, Comments: 15](https://www.reddit.com/r/LocalLLaMA/comments/1mcfuka/qwenqwen330ba3binstruct2507_hugging_face/)): **Alibaba has released the new Qwen3-30B-A3B-Instruct-2507, a Mixture-of-Experts (MoE) large language model, with quantized GGUF versions available for efficient inference ([Unsloth repo](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)). Performance benchmarks are reportedly strong, with community-provided documentation for setup ([Unsloth docs](https://docs.unsloth.ai/basics/qwen3-2507)) and a reference visual comparison shared by users. The model is designed for ease of use ('no_think') and leverages the A3B routing strategy specific to Qwen.** Discussion highlights the rapid pace of LLM improvements, with some users projecting that advances like Qwen3-30B-A3B may enable on-device inference for large models within a few years. Technical opinion praises the benchmark results and architecture choices.
    - danielhanchen provides links to GGUF format model files for Qwen3-30B-A3B-Instruct-2507 on Hugging Face (https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), as well as detailed documentation for running them on the UnsLoTH platform (https://docs.unsloth.ai/basics/qwen3-2507), facilitating ease of deployment on a variety of hardware.
    - benchmarks for the Qwen3-30B-A3B-Instruct-2507 model are described as very strong, with one user referencing [detailed benchmark results from Alibaba's official tweet](https://x.com/Alibaba_Qwen/status/1950227114793586867/photo/1) and calling the model a "no_think" (implying high efficiency or minimal inference lag) Qwen3 30B A3 variant.
    - User AppearanceHeavy6724 provides experiential feedback, noting the new model version offers massive improvements—especially in creative writing—compared to the original Qwen 30B. However, issues consistent with small-expert Mixture-of-Experts architectures remain, where prose quality can superficially appear strong but 'falls apart' with closer scrutiny, particularly in fiction tasks.
- [**Qwen3-30b-3ab-2507 is a beast for MCP usage!**](https://www.reddit.com/r/LocalLLaMA/comments/1mcji8s/qwen330b3ab2507_is_a_beast_for_mcp_usage/) ([Score: 134, Comments: 22](https://www.reddit.com/r/LocalLLaMA/comments/1mcji8s/qwen330b3ab2507_is_a_beast_for_mcp_usage/)): **The post highlights the performance of the Qwen3-30B-3AB-2507 model in managing MCP (multi-component processing) tasks autonomously across multiple servers, with a user-provided workflow ([Pastebin link](https://pastebin.com/WNPrcjLS)) substantiating this claim. Notably, the MLX 8-bit quantized version of Qwen3-30B maintained high accuracy while handling *long, complex system prompts*, reportedly outperforming Mistral 24B, and raising questions about its comparative performance against Mistral Small as well.** Technical comments validate the robustness and flexibility of the Qwen3-30B workflow; several users express strong impressions regarding its ability to follow intricate prompts and concur that it surpasses Mistral 24B for this scenario.
    - A user reports that the MLX 8-bit quantized version of Qwen3-30b-3ab-2507 can handle very long, complex system prompts without issue and performed *much better than Mistral 24B* under these specific workload demands.
    - There is an explicit technical comparison question about how Qwen3-30b-3ab-2507 performance (for MCP scenarios) stacks up to both 'Mistral 24B' and 'Mistral Small', indicating users' interest in head-to-head evaluations for specific application workflows.
    - Another user discusses managing frequent new model releases in LM Studio, indirectly implying the importance of robust testing environments for rapidly benchmarking and validating new model performance within niche use cases like MCP.

### 2. GLM 4.5 Model Launches, Benchmarks, and Ecosystem Integration

- [**I just tried GLM 4.5**](https://www.reddit.com/r/LocalLLaMA/comments/1mc8tks/i_just_tried_glm_45/) ([Score: 273, Comments: 120](https://www.reddit.com/r/LocalLLaMA/comments/1mc8tks/i_just_tried_glm_45/)): **User tested GLM 4.5 (from [z.ai](http://z.ai/)) with a complex, open-ended slide generation prompt about the global BESS market, reporting strong results with citations and non-fabricated data, even with a minimal prompt. Benchmarks and comparisons in comments claim the GLM 4.5 Air version matches the latest Qwen3-235B in output quality but is substantially more efficient (running 2x faster and at half the memory footprint, e.g., ~40-50 tok/s on 6x3090s with FP8), and supports hybrid inference. Model published on [neuroengine.ai](http://neuroengine.ai/) (testing, uptime not guaranteed). Users cite advanced code comprehension/generation (e.g., generating 5100 lines of unit tests autonomously) and consistency across diverse reasoning/recall tasks, suggesting real-world viability and broad-coverage reasoning rivaling top models like Claude 3.7–4.0 and DeepSeekV3.** Technical debate centers on the real-world quality versus benchmarks, with multiple users noting GLM 4.5's strong generalist abilities and efficiency (hardware resource usage), while also acknowledging echoes of GPT/Claude-style outputs likely due to training on their outputs. Discussion of speculative decoding (MTP), FP8 efficiency, deployment with vLLM, and evaluation as the best on-prem model for private/local inference.
    - The GLM 4.5 Air variant's performance is contrasted with qwen3-235b, with users reporting comparable quality at twice the speed and half the memory usage—specifically, 40-50 tokens/sec on a 6x3090 GPU setup using FP8 precision and hybrid mode. Notably, MTP speculative decoding has not yet been enabled, and model hosting scalability (with potential upgrades to full GLM and AWQ quantization) is discussed, pointing towards strong engineering focus.
    - Real-world code generation evaluation highlights that GLM 4.5 Air (code-wrapped like Claude/Sonnet) generated 5100 lines of accurate unit tests over two hours with minimal intervention, outperforming previous models—suggesting state-of-the-art on-prem capability for coding tasks when compared directly to Claude-3.5 Sonnet and Qwen-235B.
    - A bespoke evaluation method ('vibe bench') spanning challenging domains (programming, niche facts, creative reasoning) rates GLM 4.5 Air-100B as notably superior to DeepSeek v3, Claude 3.7–4.0, and Qwen-235B for broad, well-rounded capability. The user notes evidence of distilled training on GPT and Claude outputs, detectable in phraseology but apparently contributing to the model's consistent reliability and balance across task types.
- [**GLM 4.5 support is landing in llama.cpp**](https://github.com/ggml-org/llama.cpp/pull/14939) ([Score: 202, Comments: 46](https://www.reddit.com/r/LocalLLaMA/comments/1mc6fbp/glm_45_support_is_landing_in_llamacpp/)): **A draft Pull Request (PR) is in progress to add GLM 4.5 model support to [llama.cpp](https://github.com/ggerganov/llama.cpp), with acknowledgment from the author that this is their first time implementing a new architecture in the codebase. There is community interest in confirming whether the implementation will support Multi-Token Prediction (MTP), a feature relevant for efficiency and integration with tools like LMStudio. Users are advised against building GGUFs from the draft as the implementation is incomplete and unstable.** Technical debate revolves around the completeness and correctness of the initial implementation, with the PR's author inviting collaboration and cautioning users to wait for a finalized approach. There is considerable anticipation regarding the impact of GLM 4.5 Air, particularly its size-to-performance ratio, for the local LLM ecosystem.
    - Discussion highlights uncertainty over whether Multi-Token Prediction (MTP) is supported in the initial GLM 4.5 integration for llama.cpp, with users expressing interest in native MTP support for improved inference capabilities, especially within LMStudio.
    - There are significant caveats about the current GLM 4.5 implementation: the PR author notes it is a draft, incomplete, and not suitable for production or GGUF builds. Collaboration and further contributions are encouraged as several architectural elements remain unfinished.
    - Benchmarks and configuration notes for vLLM/Sglang support of GLM 4.5 show stable performance only under pipeline parallelism (not tensor parallelism) on a 6x3090 setup: ~40 tok/s generation, ~800 tok/s prompt processing. Also, vLLM's support is still buggy and MTP isn't enabled, raising questions about broader multi-token capabilities in deployment.
- [**My 2.5 year old laptop can write Space Invaders in JavaScript now, using GLM-4.5 Air and MLX**](https://simonwillison.net/2025/Jul/29/space-invaders/) ([Score: 138, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1mcee42/my_25_year_old_laptop_can_write_space_invaders_in/)): **The OP demonstrates that a 2.5 year old laptop can locally generate functional JavaScript code for a Space Invaders clone using the GLM-4.5 Air LLM running on the MLX inference framework. This implies efficient on-device LLM inference for code generation (game prototyping) on aging consumer hardware, leveraging Apple's MLX for resource-optimized compute. The discussion centers on laptops roughly 2.5-3 years old, including high-spec M1 Max models.** Commenters inquire about laptop specs (notably referencing Apple M1 Max, 64GB RAM), express surprise at the results, and debate the value of generating Space Invaders versus novel games for LLM-based code generation.
    - Users discuss MLX's efficiency and performance on Apple Silicon, with one mentioning running GLM-4.5 Air on an M3 and achieving smooth code generation via Roo Code. This highlights emerging capabilities of local code generation tooling on macOS compared to mainstream options like Cursor.
    - There is growing interest in MLX as a locally-run alternative for code generation tasks specifically on Macs, driven by its compatibility and performance on recent Apple hardware (e.g., M1 Max and M3 chips). Some users are considering hardware switches for optimal MLX use, underscoring its value in the current tool landscape.
- [**This year’s best open-source models and most cost-effective models**](https://www.reddit.com/r/LocalLLaMA/comments/1mc5oh2/this_years_best_opensource_models_and_most/) ([Score: 100, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1mc5oh2/this_years_best_opensource_models_and_most/)): **GLM-4.5 is a foundation LLM with 355B total and 32B active parameters, aimed at unifying reasoning, coding, and agentic use-cases; the GLM-4.5-Air variant has 106B total and 12B active parameters with a more compact design for resource efficiency. Preliminary [benchmarks](https://preview.redd.it/bisgmn0utrff1.png?width=4464&format=png&auto=webp&s=8b159e95ccba8f0becc1ee6fb596cb4fdde5217c) suggest strong performance, but the author encourages comparison to new releases like Qwen. Technical details: [blog](https://z.ai/blog/glm-4.5), [Hugging Face](https://huggingface.co/zai-org/GLM-4.5), [GitHub](https://github.com/zai-org/GLM-4.5).** Top comments stress the need for thorough third-party benchmarking before final judgments and note excellent real-world performance, particularly of GLM-4.5-Air in code and problem solving versus other open models (e.g., Qwen, Kimi K2).
    - Several commenters highlight the lack of independent, third-party benchmarks for recently released open-source models, such as Qwen and GLM 4.5, stressing that reliable comparative evaluation is not yet available and is necessary before declaring any model the best or most cost-effective.
    - Direct hands-on experiences suggest significant variation in performance across tasks: one user reports that GLM 4.5 Air solves a difficult CSS coding problem that other local LLMs struggled with, while another found it underperformed for fiction writing compared to Big 4.5 and small 4-0414-32b models.
    - Subjective evaluation of models like Big 4.5, Kimi K2, and Qwen based on use cases such as code and fiction generation indicates that model strengths can be highly task-dependent, highlighting the need for granular benchmarks (e.g., on coding or creative writing datasets).

### 3. Meta Observations on AI Model Progress (Memes and Commentary)

- [**its getting comical**](https://i.redd.it/txsukljc5pff1.png) ([Score: 973, Comments: 92](https://www.reddit.com/r/LocalLLaMA/comments/1mbvf2z/its_getting_comical/)): **The image itself could not be analyzed, but discussion focuses on the lack of recent open-weight LLM releases from US companies, with commenters noting a trend toward offering API-only access rather than downloadable models. Specific releases mentioned include Granite, Micro Gemma, and the updated Nemos (all relatively rare recent open weight releases). The thread broadly laments the state of open-access AI models from the US and contrasts the announcement hype with the lack of delivery on downloadable weights.** There is pronounced skepticism in the comments about US companies delivering open-weight LLMs, with frustration over repeated announcements that do not materialize into actual releases. Some comments also touch on wider industry and geopolitical implications, though these are more dismissive or sarcastic in tone.
    - paryska99 discusses rapid advancements in agentic LLM workflows, specifically mentioning that after promising results from Kimi K2 and Qwen3, the release of GLM 4.5 has shifted their preference. They note that GLM 4.5 outperforms these contenders, describing it as *the real Claude Sonnet destroyer*, signaling significant performance improvements over leading models for certain tasks.
    - a_beautiful_rhind highlights the lack of recent open-weight model releases from US companies, listing Granite, Micro Gemma, and updated Nemos as the latest, and comments on the current trend of preferring API-based access instead of open releases.

## Less Technical AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo
> 

### 1. Wan 2.2 Model Release Benchmarks and Comparisons

- [**2d animation comparison for Wan 2.2 vs Seedance**](https://v.redd.it/lqs9s9fsotff1) ([Score: 770, Comments: 59](https://www.reddit.com/r/StableDiffusion/comments/1mccuf0/2d_animation_comparison_for_wan_22_vs_seedance/)): **The post presents an informal comparison between Wan 2.2 and Seedance for 2D animation generation, highlighting that Wan 2.2 produces qualitatively decent animations but exhibits some visual artifacts. No detailed benchmarks or parameter settings are provided, and the evaluation appears observational rather than rigorous.** Commenters note differences in cost (Wan 2.2 being free vs Seedance paid) and reference specific qualitative outcomes (e.g. artifacts in Wan 2.2, the 'sad' appearance of Seedance's dolphin animation). No deep technical debate or implementation discussion is present.
    - d4pr4ssion notes that **WAN 2.2 produces superior 2D animation** compared to Seedance, citing *more action and higher coherence* in the animation output. This suggests WAN 2.2 may have better temporal consistency or motion handling for 2D animated sequences, which is critical for animation tasks.
- [**Wan 2.2 - Generated in ~60 seconds on RTX 5090 and the quality is absolutely outstanding.**](https://v.redd.it/6njp2ehhvpff1) ([Score: 630, Comments: 115](https://www.reddit.com/r/StableDiffusion/comments/1mbyna7/wan_22_generated_in_60_seconds_on_rtx_5090_and/)): **A user showcases results from Wan 2.2—a video generation model—highlighting that it produced high-quality, expressive 3D cartoon plus realistic character animation in ~60s locally on an RTX 5090 GPU. Workflow details reveal usage of t2v for text-to-video, resolution** `832x480p`**, LightX2V LoRA nodes at** `1.5 strength`**, Unipc/Simple sampling, with** `10 steps` **(split high/low between frames 0-5 and 5-10). Full workflow script is provided in [this gist](https://gist.github.com/Art9681/91394be3df4f809ca5d008d219fbc5f2).** Commenters note Wan 2.2 is a significant quality improvement over 2.1, with especially strong facial expressions and utility for rapid local iteration. One suggests combining with kontext for further workflow expansion.
    - The OP provides a detailed workflow gist outlining the steps used to achieve the fast generation (about 60 seconds) on an RTX 5090. The workflow includes notes on removing unnecessary components and advises users to upgrade to a newer version of the lightx2v node for improved results ([gist link](https://gist.github.com/Art9681/91394be3df4f809ca5d008d219fbc5f2)).
    - Performance details are discussed: generation at 832x480p using lightx2v LoRA nodes set to strength 1.5, with a total of 10 steps. The inference schedule splits between 0-5 steps (high) and 5-10 steps (low), and uses unipc/simple samplers. Users remark that the quality represents a significant improvement over version 2.1.
    - There is discussion about speed differentials—one user notes their workflow is significantly slower, pointing to the value of workflow optimizations and version upgrades for performance matching the OP's results.
- [**Ok Wan2.2 is delivering... here some action animals!**](https://v.redd.it/cmqux8w0hsff1) ([Score: 342, Comments: 35](https://www.reddit.com/r/StableDiffusion/comments/1mc7q9u/ok_wan22_is_delivering_here_some_action_animals/)): **The OP showcases video output from Wan2.2, using the ComfyUI default workflow with** `torch.compile` **and SageAttention2 on an RTX 5090 GPU, achieving an 18-minute render time per shot (down from 36 minutes previously). Quality is noted as significantly improved over prior versions, though speed is still considered insufficient for production workloads. Commenters highlight a clear quality difference between** `fp8` **and** `fp16` **inference on Wan2.2, favoring fp16 for superior results despite reduced speed. Notably, Triton and SageAttention2 are credited for substantial speed improvements.** Discussion centers on the technical validation of the `Triton + SageAttention` workflow, with users verifying expected behavior and expressing surprise at the quality jump between `fp8` and `fp16` on Wan2.2. The slow generation speed, while improved, remains a limiting factor for production deployment.
    - A user compared fp8 and fp16 quality on Wan2.2, observing that fp16 produces obviously better image quality at the cost of slower generation compared to fp8. This is an improvement over Wan2.1, where the gap between fp8 and fp16 was less pronounced, suggesting a model or codebase enhancement affecting numerical precision trade-offs.
    - Dramatic performance gains are reported when combining Triton and Sageattention, halving generation time from 36 to 18 minutes. The commenter seeks validation of their setup, implicitly raising questions about correct workflow, configuration, and potential best practices when integrating these acceleration libraries/stacks into a Stable Diffusion pipeline.
- [**Wan 2.2 human image generation is very good. This open model has a great future.**](https://www.reddit.com/gallery/1mcm7qm) ([Score: 265, Comments: 77](https://www.reddit.com/r/StableDiffusion/comments/1mcm7qm/wan_22_human_image_generation_is_very_good_this/)): **The post introduces the open-source WAN 2.2 human image generation model and notes its strong performance, with a workflow optimized for systems with 24GB VRAM (see Hugging Face workflow config). The model prioritizes high fidelity over speed, sacrificing generation time for quality, and may be compatible with speed-up Lora modules at the expense of image quality. A full-resolution gallery is available to demonstrate native output untainted by Reddit compression.** Commenters request detailed settings and workflow steps, indicating interest in reproducibility. There are concerns regarding Reddit image compression masking fine details, leading to sharing results via an external gallery. The model's high VRAM requirement and slow speed are acknowledged as trade-offs for performance.
    - A workflow optimized for Wan 2.2 human image generation is shared, requiring 24GB VRAM for optimal performance. There are untested lower-memory (GUFF) configurations, but the workflow prioritizes high image quality over speed, and attempts to accelerate generation with LoRA adapters may reduce quality. The full workflow JSON is available at [Hugging Face](https://huggingface.co/RazzzHF/workflow/blob/main/wan2.2_upscaling_workflow.json).
    - A key challenge with sharing visual results is signal loss from Reddit's image compression, which degrades fine details produced by high-quality pipelines. For uncompressed output, users are directed to an external gallery for true evaluation: [full quality gallery](https://postimg.cc/gallery/8r8DBpD).
    - A technical hypothesis is presented that video-based generative models, trained on temporal data, may inherently acquire a deeper understanding of 3D geometry and object consistency under rotation or viewpoint changes, suggesting video models could become the benchmark for image generation if they can be efficiently run on lower-end hardware.
- [**Wan 2.2 14B T2V (GGUF Q8) vs Flux.1 Dev (GGUF Q8) | text2img**](https://www.reddit.com/gallery/1mc981k) ([Score: 215, Comments: 66](https://www.reddit.com/r/StableDiffusion/comments/1mc981k/wan_22_14b_t2v_gguf_q8_vs_flux1_dev_gguf_q8/)): **The post compares two GGUF Q8-quantized models for text-to-image generation: WAN 2.2 14B T2V versus Flux.1 Dev, both running on an RTX 3090 with 32GB RAM and using the same 1080x1080 resolution,** `res_2s` **sampler, and** `bong_tangent` **scheduler. Notably, WAN 2.2 uses 14B parameters, just 8 steps, and a CFG of 1, while Flux.1 Dev has an unknown parameter count, 30 steps, and CFG 3.5; both achieve similar runtimes (~90 sec/generation). The [supporting post](https://www.reddit.com/r/StableDiffusion/comments/1mbsqxv/wan_22_14b_t2v_txt2img/) describes the test workflow in detail.** Expert commenters note **WAN 2.2** produces more natural images and attribute its edge to the larger `14B` weight size; consensus is that WAN outperforms Flux in visual quality, with one noting 'Flux is done' and others highlighting the impact of the size difference.
    - Several users highlight that WAN 2.2 14B demonstrates a significant improvement in realism and natural appearance over Flux.1 Dev, with Flux exhibiting a more artificial or 'AI polished' look in comparison. Technical speculation points to the additional ~2 billion parameters (14B vs Flux's parameter count) as a likely factor influencing WAN's superior qualitative output and richer color rendition.
    - There is a discussion about WAN 2.2's strengths in producing cinematic, realistic images with convincing colors, suggesting that its architecture or data curation may inherently favor realism. Some users also express interest in further tuning or finetuning WAN for broader stylistic diversity, paralleling approaches seen in MidJourney, which could address current limitations in creativity across varied artistic styles.
- [**Wan 2.2 ı2v examples made with 8gb vram**](https://v.redd.it/vy9vtnmistff1) ([Score: 196, Comments: 30](https://www.reddit.com/r/StableDiffusion/comments/1mcdfy5/wan_22_%C4%B12v_examples_made_with_8gb_vram/)): **The OP reports successful inference with the wan2.2 i2v (image-to-video) model at Q6 quantization (likely GGUF format) and l2v ligtx2v LoRA strength 1.0, using 8GB VRAM (GPU unspecified), 8 steps, CFG 1.0, for both high and low denoise settings. The setup utilizes the default ComfyUI workflow with GGUF and LoRA loader added, suggesting feasibility of this configuration within modest memory constraints.** Several commenters request the workflow file due to reproducibility issues, particularly with GGUF and WAN 2.2 VAE. One user notes OOM errors even on a 3060 12GB VRAM GPU with q4_k_m, while another asks for specifics about the use of GGUF, indicating that GGUF compatibility and efficient memory usage are significant technical pain points worthy of further workflow documentation or troubleshooting.
    - Users report out-of-memory (OOM) errors when running the q4_k_m quantization on a 3060 12GB GPU, indicating that VRAM requirements for Wan 2.2 i2v can exceed 8GB–12GB for certain quantization settings or models (GGUF, VAE). Detailed reporting of model type and quantization is relevant to VRAM use.
    - Technical workflow sharing is emphasized; users request JSON exports of the exact node/workflow setup, as small differences (e.g., custom or unknown nodes) can prevent successful replication. This highlights the need for precise reproducibility in community-shared pipelines.
    - Questions arise about the rationale for using 8 denoising steps with Lightx LoRA, as community expectation is typically 4–6 steps, suggesting a technical optimization or unexplained parameter change that may impact output quality or generation time.
- [**Wan 2.2 can do that Veo3 writing on starting image trick (credit to](https://v.redd.it/csy54chiksff1) [guizang.ai](http://guizang.ai/)[)](https://v.redd.it/csy54chiksff1)** ([Score: 115, Comments: 19](https://www.reddit.com/r/StableDiffusion/comments/1mc807b/wan_22_can_do_that_veo3_writing_on_starting_image/)): **The post notes that Wan 2.2, a new generative AI model, can perform the "Veo3 writing on starting image" technique, previously associated with Google's Veo3, based on a demonstration by [guizang.ai](http://guizang.ai/). This technique involves robust text rendering on a user-selected image at the start of generation—a capability that some models (like Kling) purportedly lack, highlighting Wan 2.2's strength in image conditioning and text integration.** Comments highlight surprise at Wan 2.2's capabilities, expressing that competitors (e.g., Kling) cannot replicate this. There is also skepticism over the reproducibility of the showcased results, implying a need for independent verification.
    - Comments highlight a technical feat: Wan 2.2 is reported to replicate the Veo3 'writing on starting image' trick, which some users indicate has not been successfully reproduced by other popular models like 'kling.' This suggests notable advancement in fine-grained image guidance or prompt conditioning compared to alternatives.
    - A user queries implementation specifics, asking if prompt engineering is required by copying text present in the initial image into the model prompt, or if the model can perform the task without explicit textual guidance. This addresses practical differences in workflow for leveraging such features.

### 2. OpenAI GPT-5 and Study Mode Announcements

- [**Apparently GPT-5 is rolling out? With ability to think deeper + video chat and more**](https://i.redd.it/vx5h19qd7rff1.jpeg) ([Score: 342, Comments: 87](https://www.reddit.com/r/singularity/comments/1mc44ls/apparently_gpt5_is_rolling_out_with_ability_to/)): **The image purports to show a rollout of GPT-5 with new features like deeper reasoning and video chat. However, commenters and linked sources clarify that this image and associated claims are almost certainly fake, possibly originating from accounts known to spread misinformation. No official OpenAI or reputable source confirms such a rollout, and the 'Prime' product appears to be fabricated.** Commenters overwhelmingly agree that the image is not legitimate, citing evidence of fabricated leaks and accounts created solely to spread false information. Technical consensus is that OpenAI would not roll out GPT-5 in this manner.
    - Users debunk purported GPT-5 leaks, identifying that posts and images circulating as 'evidence' appear fabricated and are linked to previously inactive or unreliable accounts. One reference is made to a tweet by OpenAI employee Stephan Casas clarifying confusion around the supposed rollout and discussing that 'gpt prime' is not a legitimate feature.
    - Some users report UI updates or minor new features (like 'think longer' button for Plus accounts) as potential indicators of an imminent update or feature announcement, but no core upgrade to GPT-5 is confirmed or observed. This suggests minor UI/UX experiments are ongoing but not necessarily tied to a major model release.
- [**GPT-5 Alpha**](https://i.redd.it/0fs7z2xzytff1.jpeg) ([Score: 251, Comments: 80](https://www.reddit.com/r/singularity/comments/1mce9ho/gpt5_alpha/)): **The image shows a social media post by the Head of Design at Cursor, referencing 'vibe coding' with 'GPT-5 Alpha', implying early or internal access to a highly anticipated version of OpenAI's next model. The context suggests this is not a formal announcement but a casual or preview indication of GPT-5's capabilities or imminent release. A linked GitHub repository (https://github.com/ryokun6/ryos) is provided in comments, though its relation to GPT-5 is ambiguous from the post. There is mention that the original Tweet might have been deleted, raising questions about the post's intended publicity.** Commenters speculate on the release timeline ('So... this week or what?') and discuss the post's authenticity or intended leak, while noting possible quick removal of the original source. No technical benchmark data or concrete features of GPT-5 Alpha are discussed.
    - A user linked to the GitHub repository for the project (https://github.com/ryokun6/ryos), indicating there may be an open-source or experimental implementation related to "GPT-5 Alpha" available on GitHub. Technical readers can inspect code, documentation, and issues to assess model details, experimental features, or architecture design.
    - There is discussion about the public communication strategy, specifically about tweets possibly being deleted, suggesting that information about "GPT-5 Alpha" may be tightly controlled or subject to rapid changes, potentially impacting developer access to timely information. This implies that tracking project developments may require monitoring multiple sources or archived data.
    - One comment raises the issue of NDAs (Non-Disclosure Agreements), questioning whether team members or testers are contractually restricted from revealing updates about "GPT-5 Alpha". For a technical reader, this stresses the point that technical details and insider benchmarks may not be readily available until official announcements are made, impacting the ability to independently verify claims or performance.
- [**Finally ! GPT-5 is almost there and it's freaking amazing**](https://i.redd.it/z2i8qgqkztff1.jpeg) ([Score: 473, Comments: 118](https://www.reddit.com/r/OpenAI/comments/1mcecll/finally_gpt5_is_almost_there_and_its_freaking/)): **The post's image appears to show a prompt or conversation implying access to an early version of GPT-5, but does not provide any concrete demonstration or technical benchmark of performance. Comments debunk the claim, clarifying that the image relates to a tweet misattributing advanced results to GPT-5 when, in reality, the showcased project was made using Cursor (an AI tool), with no actual confirmation or evidence of GPT-5 being used. The discussion suggests the post is likely speculative or misleading rather than a technical leak or benchmark of GPT-5.** Multiple commenters stress the lack of proof in the post, criticizing its misinformation and noting that neither the original developer nor Cursor claimed the use of GPT-5.
    - There is skepticism about the authenticity of the 'GPT-5' claims in the original tweet, with commenters pointing out that the showcased project referenced actually involved a Cursor employee who never claimed it was powered by GPT-5. The post seems to conflate unrelated work with an unreleased model, raising concerns about misinformation regarding model capabilities and sources.
- [**OpenAI: Introducing study mode - A new way to learn in ChatGPT that offers step by step guidance instead of quick answers**](https://openai.com/index/chatgpt-study-mode/) ([Score: 319, Comments: 37](https://www.reddit.com/r/singularity/comments/1mchrs2/openai_introducing_study_mode_a_new_way_to_learn/)): **OpenAI has introduced a 'study mode' for ChatGPT designed to provide step-by-step learning rather than instant answers. Initial user feedback indicates that the feature sometimes quizzes on information not previously provided, raising questions about context awareness and whether this is intentional to foster external research or an implementation flaw.** Commenters note that 'study mode' currently feels like an enhanced prompt and suggest meaningful differentiation would require further fine-tuning of educational models. There is anticipation for more sophisticated, targeted learning experiences in future iterations.
    - A user highlighted a bug or design flaw in the study mode: the system "was quizzing me on information it hadn’t included in its ‘lesson’," raising concerns about incomplete context tracking or whether this is an intentional design to promote independent research. This points to potential limitations in prompt-engineering and model context awareness in the current version.
    - A teacher described substantial performance differences for math tasks between base ChatGPT models and the more advanced 'thinking models' such as GPT-4o, reporting a much greater improvement than the shift from GPT-3 to GPT-4. The educator warns that free users, lacking access to these top-tier models, risk being misinformed, and therefore recommends alternatives like Gemini 2.5 Pro for free access to more accurate guidance in educational tasks, especially for subjects vulnerable to hallucination errors.
    - Several commenters noted that the current "study mode" implementation mainly repackages prompt strategies rather than adding fundamentally new learning capabilities, and that future iterations with better fine-tuned, dedicated models for learning could improve the educational experience. This underlines current technical limitations—primarily system prompt constraints—while hinting at a roadmap toward more sophisticated model specialization for education.
- [**Study mode for students finally available!!**](https://i.redd.it/8h6kr7caiuff1.jpeg) ([Score: 1008, Comments: 73](https://www.reddit.com/r/OpenAI/comments/1mch6jr/study_mode_for_students_finally_available/)): **The image appears to be an announcement or preview showing that a 'study mode' is now available for students—likely in a popular AI product such as ChatGPT. The feature seems oriented toward educational use-cases, enabling use of AI for research, learning, and study assistance, addressing previous concerns about AI's impact on academic integrity. Commenters also suggest the new mode could be enhanced by features showing students' work and learning process, not just answers.** Several users highlight this as a necessary development given the growing integration of AI into education, and suggest the addition of a 'show your work' feature to promote understanding and ethical use. There is optimism about the potential long-term success and educational value if implemented thoughtfully.
    - A commenter proposes implementing a "show your work" mode in ChatGPT for educational use. This feature would log the student's step-by-step process, questions asked, and responses received, allowing evaluators to verify genuine learning rather than simply checking answers. The emphasis is on increasing transparency and encouraging actual comprehension rather than shortcutting the learning process via AI-generated answers.
    - Discussion points include the possible impact of AI-powered tutors like ChatGPT on educational systems and long-term outcomes, such as self-learners influencing college admissions trends. Comparisons are drawn to platforms like Khan Academy, highlighting ChatGPT's potential to serve as a personalized, scalable tutoring solution that may reshape traditional educational pathways.
    - There is consensus that integrating structured AI study tools (e.g., dedicated study modes) can guide students to use AI for enrichment rather than dependency, and that this early step by OpenAI indicates intent for AI as a supportive, rather than replacement, element in education.

### 3. AI Impact on Jobs and Society: Industry Predictions and Ethical Concerns

- [**Anthropic CEO: AI Will Write 90% Of All Code 3-6 Months From Now**](https://www.reddit.com/r/singularity/comments/1mch6sg/anthropic_ceo_ai_will_write_90_of_all_code_36/) ([Score: 505, Comments: 212](https://www.reddit.com/r/singularity/comments/1mch6sg/anthropic_ceo_ai_will_write_90_of_all_code_36/)): **Dario Amodei, CEO of Anthropic, predicted that AI would be responsible for writing 90% of code within 3-6 months of his statement (originally reported by Business Insider). As of now, with a month remaining, current AI tools (such as Copilot, ChatGPT Code Interpreter, and Claude) do not fully autonomously generate the majority of production code, with API and context window limits being significant constraints. There is little quantitative evidence supporting a 50% or higher share of code written directly by AI in industry settings.** Commenters highlight that Amodei's prediction may depend on the definition of code generation (i.e., validating if "aid" versus "autonomous writing" counts), with some arguing that while AI heavily assists coders, fully autonomous generation is still uncommon. API rate limits and issues with quality/functionality remain blockers; however, the rate of advancement is acknowledged as surprising even to skeptics from just a year ago.
    - There is skepticism about AI autonomously writing 90% of all code in the near future due to technical constraints such as API rate limits and the underlying architecture of most large language models (LLMs). The quadratic scaling of self-attention mechanisms in transformer-based LLMs is cited as a fundamental bottleneck, requiring significant breakthroughs for independent high-volume code generation.
    - Some practitioners note that if the claim refers simply to the proportion of code generated by AI (even as suggestions or tab-completes), the percentages are already high—one professional engineer notes that token-level contributions from AI can be upwards of 50% of new code in their workflow. However, this reflects AI as an assistive tool, not as an autonomous coder.
    - The discussion makes clear that 'writing' code is nuanced: current implementations mostly involve AI acting as an advanced autocomplete or ideation partner rather than reliably producing entire functional systems without human oversight or detailed prompting. True autonomy at the 90% level would likely require advances well beyond existing LLM architectures.
- [**zuckerberg offered a dozen people in mira murati's startup up to a billion dollars, not a single person has taken the offer**](https://www.reddit.com/r/singularity/comments/1mcirpx/zuckerberg_offered_a_dozen_people_in_mira_muratis/) ([Score: 785, Comments: 182](https://www.reddit.com/r/singularity/comments/1mcirpx/zuckerberg_offered_a_dozen_people_in_mira_muratis/)): **Mark Zuckerberg reportedly offered individual compensation packages of up to $1B to at least a dozen employees of Mira Murati's (CTO of OpenAI) startup in an aggressive recruiting effort, referencing a screenshot of internal discussions. None of the targeted engineers or researchers accepted the offer, indicating extremely high retention or commitment to mission within that startup.** Commentary centers on whether this is due to strong belief in the startup's vision and/or culture, or active aversion to working for Meta. The offer amount ($1B) is described as extreme, highlighting intense competition for top-tier AGI talent and possibly underscoring skepticism toward Meta's AI reputation.
    - A major technical insight is that the AI researchers reportedly declined offers from Meta worth up to $1B, possibly due to significant equity stakes in Mira Murati's startup that could already be worth "a couple hundred million each". This implies strong confidence in the valuation and growth potential of the startup, as well as the high market value of leading AI talent.
    - Speculation exists that publicizing these outsized offers could be a strategic move to support the startup's next fundraising round, by reinforcing perceived demand and team commitment. This reflects a broader trend in AI startups leveraging PR and funding narratives to enhance valuations and attract investment.
- [**A new deal with Microsoft that would let them keep using OpenAI's tech even after AGI is reached.**](https://www.bloomberg.com/news/articles/2025-07-29/microsoft-s-access-to-openai-tech-is-focus-of-contract-talks) ([Score: 165, Comments: 24](https://www.reddit.com/r/singularity/comments/1mcej0x/a_new_deal_with_microsoft_that_would_let_them/)): **Microsoft's new proposed deal with OpenAI (referenced via [archive](https://archive.ph/wd8eX)) would grant continued access to OpenAI's latest models and technology even *post-AGI*, contingent on a** `30-35% equity stake`**, an increased non-profit stake, reduced OpenAI revenue share for Microsoft, operational freedoms, and enforceable safety provisions. This framework reflects an escalation in Microsoft's strategic alignment and risk-sharing with OpenAI over future AGI development and commercialization.** Commenters highlight Microsoft's deep dependency on OpenAI and note perceived differences in performance between Copilot (built on GPT-4) and ChatGPT-4, speculating whether upcoming upgrades (e.g., GPT-5) will bridge capability gaps.
    - Discussion clarifies that under OpenAI's current contract, a threshold event like achieving AGI would limit Microsoft's rights to use OpenAI models. The noted Bloomberg report highlights negotiations to allow Microsoft longer-term or ongoing access even post-AGI, which impacts both technical deployment rights and business continuity for Microsoft if AGI is achieved.
    - A technical comparison is raised regarding Copilot (business) and ChatGPT-4, noting that although both leverage GPT-4, Copilot's performance does not match standalone ChatGPT-4. The comment anticipates that Copilot could see improvements with GPT-5, suggesting technical differences in implementation or tuning between Microsoft's product integration and OpenAI's primary offerings.
    - A misconception is noted: while someone asks if GPT is open source, in fact, GPT-3/4 and later models are not open source; only older models like GPT-2 are. This distinction is central to understanding why Microsoft needs contractual rights and cannot simply self-host the latest models without OpenAI's continued partnership.

---

# AI Discord Recap

> A summary of Summaries of Summaries by Gemini 2.0 Flash Thinking
> 

**Theme 1. Emerging AI Models & Performance Dynamics**

- **Qwen 30B Flexes Local Muscle:** The **Qwen3 30B model** impresses users with its speed and efficiency, consuming only around **17GB of RAM** and offering a *decent jump* in performance. Users report it rivals **GPT-4o** and can run locally with **33GB RAM** at Q4_K_XL, making it a viable option for various AI tasks.
- **GLM 4.5 Air Soars, Whispers Gemini:** Community members praise the [GLM 4.5 Air model](https://z.ai/blog/glm-4.5) for creative writing and multilingual tasks, achieving **35 tokens per second at 5 bit mlx**. Users suspect it may be distilled from **Gemini** due to its response style and strong performance in SVG and web benchmarks, despite data collection concerns.
- **GPT-5 Hype Ignites, Zenith Adds Fuel:** Speculation about **GPT-5's** imminent release intensified after a blurred [image appeared on X](https://x.com/ryolu_/status/1950163428389040431), claiming a **1M token** input window and parallel tool calls. The removal of **Zenith** from LLM Arena, believed by some to be **GPT-5**, further fueled the excitement, as users reported *noticeable improvement* over existing models.

**Theme 2. AI Development & Infrastructure Challenges**

- **API Keys Get 403 Errors, Anthropic Stings Users:** Users across Discords report encountering **403 errors** with messages like *"Incorrect API key provided"*, even after account top-ups. **Anthropic's** API faces heavy criticism for its restrictive limits and expensive pricing, with a [tweet](https://x.com/anthropicai/status/1949898502688903593?s=46) detailing weekly restrictions drawing user ire.
- **LM Studio Users See Performance Nosedive:** A user reported a significant drop in token generation speed with **Qwen 3 30B A3B** in LM Studio, plummeting from *50-80 tk/s* to *30-35 tk/s*. Suggested fixes include disabling **flash attention** and verifying other specific settings, followed by unloading and reloading the model to regain speed.
- **HuggingFace Tokenizers Draw User Ire:** A member voiced strong objections with **HuggingFace Tokenizers**, citing problems with [adding custom tokens](https://huggingface.co/docs/transformers/tokenizer_summary) and renaming **<unused>** tokens. They surprisingly discovered that using custom tokens without adding them to the vocabulary yields better results, highlighting unexpected tokenizer behaviors.

**Theme 3. AI Platform & Ecosystem Innovations**

- **Launch of AgentSmith:** **AgentSmith**, an [open-source prompt CMS](https://github.com/chad-syntax/agentsmith) built on top of **OpenRouter**, launched to streamline prompt/context engineering. OpenRouter also actively works on its **standard lane routing**, aiming to balance factors like throughput, latency, and tool call success rates beyond just price, moving towards a *"best quality"* option.
- **LlamaIndex Unleashes FlowMaker, Natively n8n Nodes:** **LlamaIndex** rolled out **FlowMaker**, a GUI tool for visually constructing LlamaIndex workflows accessible at [flowmaker.llamaindex.ai](http://flowmaker.llamaindex.ai/). They also introduced new open-sourced [n8n nodes for LlamaCloud](https://github.com/run-llama/n8n-llamacloud), including LlamaCloud indexes and LlamaParse, streamlining intelligent document processing.
- **Cursor 1.3 Lands, Auto Mode Sees Mixed Reviews:** The new **Cursor 1.3** release includes [terminal sharing with agents](http://cursor.com/changelog) and faster edits, allowing users to view context usage in Chat. Reports on **Auto Mode** are mixed, with claims of *truly unlimited* engagement, while others note significantly superior results from models like Claude and report **$50 API costs**, suggesting switches to **GPT4.1** during high usage.

**Theme 4. Ethical AI & User Experience Concerns**

- **Data Privacy Debates Ignite Over LLM Use:** Members are debating the implications of using LLMs from companies like **OpenAI**, highlighting concerns about data collection, storage, and potential misuse for targeted influence or sale to data brokers. While open-source models can reduce risks, some warn that even free tiers may involve data collection.
- **AI Influencers Accused of Pro-GPT-4o Bias:** Members observed potential bias among influencers when comparing AI models against [ChatGPT](https://openai.com/), noting failures to fully utilize reasoning modes or reset chats between prompts. One member specifically criticized a [GPT-4o](https://openai.com/index/hello-gpt-4o/) review for not optimizing the potential of each product, suggesting a lack of thorough comparison.
- **ChatGPT Study Mode Sparks Educational Debate:** OpenAI launched **study mode** in ChatGPT, designed to assist students in achieving *deeper understanding* by guiding them through problems step-by-step, as detailed in their [announcement](https://openai.com/index/chatgpt-study-mode/). Members reacted by considering it *moving closer to the endgame* and a *violation of OpenAI's business model, designed to maximally disrupt formal education*.

**Theme 5. Advancements in AI Research & Techniques**

- **Sparsity Soars as Model Sizes Swell:** As model size increases, the optimal **sparsity** for a fixed budget also increases, especially during training runs, ideal for achieving maximum performance at lowest cost. Modern dropless **Mixture of Experts (MoEs)** now train **2.2 to 2.5x** faster than dense networks with new approaches hitting **3.2x** speeds, outperforming traditional geomean rules due to optimizations.
- **LLMs as Interp Agents Gain Traction:** Members explore using **LLMs as interpretation agents** for automated mech interp research, referencing **Transluce's** work and [Sarah's MAIA paper](https://openreview.net/forum?id=mDw42ZanmE) as key resources. Neel Nanda announced applications for **MATS 9.0**, a mentorship program focused on paid mech interp research, aiming to produce high-quality [research papers](https://tinyurl.com/neel-mats-app).
- **New Diffusion Study Groups Kick Off:** A **5-month study group**, limited to 12 participants requiring **2–4 hrs/week**, will explore diffusion models based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf). The first **two intro sessions are free** and open to non-members, covering topics like Flow Matching and the history of diffusion models.


---

# Discord: High level Discord summaries




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Kills R1 1776, Boosts Claude 4 Sonnet**: Perplexity retired the **R1 1776** model from its web and mobile platform due to performance issues and is recommending users switch to **Claude 4.0 Sonnet (Thinking)** for comparable strengths.
   - The transition aims to provide a stronger experience for reasoning-focused tasks, while no changes are planned for the **Sonar** model.
- **Comet Browser Users Beg for Cloud Sync**: Members are eagerly awaiting cloud sync in the next update of the [Comet browser](https://cometbrowser.com/), citing its absence as a major obstacle to adopting it as their primary browser.
   - The feature is highly anticipated to enhance user experience and facilitate seamless data synchronization across devices.
- **Qwen3 30B Model Impresses with Speed and Efficiency**: The **Qwen3 30B model** has been well-received by users for its speed and efficiency, with one member reporting that it consumes only around **17GB of RAM**.
   - The model is considered a *decent jump* in performance, making it a viable option for various AI tasks.
- **Deep Research API Support Sought After**: A member sought support for the **Deep Research API** to further product development, and a Perplexity team member offered assistance with API-related inquiries.
   - Another member reported that there was **Sonar Deep Research** returning partially garbled output, and a member from Perplexity confirmed that the team is investigating the issue and linked to the [resolved ticket](https://community.perplexity.ai/t/sonar-deep-research-returns-partly-garbled-output/809).
- **AI Influencers Accused of Pro-GPT-4o Bias**: Members observed potential bias among influencers when comparing AI models against [ChatGPT](https://openai.com/), noting failures to fully utilize reasoning modes or reset chats between prompts.
   - Specifically, one member criticized a [GPT-4o](https://openai.com/index/hello-gpt-4o/) review for not optimizing the potential of each product.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GLM 4.5 Air gets Lots of Love**: The community has been praising the [GLM 4.5 Air model](https://z.ai/blog/glm-4.5) for its prowess in creative writing and multilingual tasks, with one user reporting **35 tokens per second at 5 bit mlx**.
   - Members have drawn comparisons to **Gemini**, while also acknowledging its strength as an uncensored model.
- **TRL Update Wreaks Havoc, Downgrade fixes It**: A new **trl version** update caused widespread issues, specifically an `ImportError` concerning `ConstantLengthDataset`, fixed by reverting to **trl==0.19.1**.
   - The Unsloth team addressed the breakage recommending users on Colab/Kaggle to delete runtime and refresh the notebook, or use `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo` and `pip install --upgrade --no-deps huggingface_hub`.
- **Tokenizers Troubles plague HuggingFace**: A member voiced strong objections with **HuggingFace Tokenizers**, citing problems with [adding custom tokens](https://huggingface.co/docs/transformers/tokenizer_summary) and renaming **<unused>** tokens, but that using custom tokens without adding them to the vocabulary yields better results.
   - The member also was surprised to discover that **Windows 7** could be modded to receive updates until **December 2025**.
- **Vision Encoder Quantization Questioned**: While inspecting **Gemma 3**'s vision component, one member noticed that the convolution weight/filter *v.patch_embd.weight* remained in float32.
   - Another member responded that quantizing vectors in models isn't worth it, since vectors are sensitive to quantization and make up less than **0.5%** of the model parameters.
- **Model Merging Strategies Spark Debate**: Discussions arose around merging all experts into a dense model, swapping them, and converting a dense model into an **MoE** model, highlighting ESFT [repo for deepseek arch](https://github.com/deepseek-ai/ESFT) as a means of finetuning specific experts.
   - The intention is to explore the frontier of performance improvement by way of frankenMoE fine-tuning, as it relates to *mergekit*.



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **GPT-5 Hype Intensifies After Zenith Removal**: A blurred image purportedly showing **GPT-5** has been [posted on X](https://x.com/ryolu_/status/1950163428389040431), fueling speculation, further enhanced by the removal of model **Zenith** from the Arena.
   - Some users claimed that **Zenith**, believed by some to be **GPT-5**, showed a *noticeable improvement* over existing models like **GPT-4**.
- **Is GLM 4.5 secretly Gemini?**: Users suspect **GLM 4.5** may be distilled from **Gemini** due to its tendency to start responses with *'Of course!'* and its longer response style, with a [HuggingFace Space](https://huggingface.co/spaces/zai-org/GLM-4.5-Space) available for testing.
   - Despite data collection concerns, it is performing well in SVG and web benchmarks, leading to comparisons with past AI models.
- **Qwen's Code Skills Enter Arena**: The **Qwen 3** coder model has been added to the LLM Arena leaderboard, showcasing its coding skills.
   - Further releases are anticipated, featuring a new A3B architecture and thinking versions to enhance its capabilities.
- **Data Privacy Debate Ignites Over LLM Use**: Members are debating the implications of using LLMs from companies like **OpenAI**, highlighting concerns about data collection, storage, and potential misuse for targeted influence or sale to data brokers.
   - While open-source models and avoiding links to personal identities can reduce risks, some say even free tiers may involve data collection.
- **Tool Calling Artificially Lowers Benchmarks?**: Concerns are raised that vendors are artificially lower in benchmarks due to their prioritization of tool calling and agent capabilities, as seen in **Opus Sonnet GLM4.5 and KimiK2**.
   - Some claim this prioritization makes many academic-type benchmarks worse for some reason, affecting overall model evaluation.



---



## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **O3 Update: Speed Boost, Sanity Cut?**: Users report that **O3** has become significantly faster but noticeably dumber in recent updates, likening it to a less thoughtful, more unified version, with some trying out **Windsurf** as an alternative for better *thinking mode*.
   - Tracking context window usage also reveals high consumption rates, impacting chat performance with one user reaching **95%** context usage due to prolonged chat sessions.
- **Auto Mode Aces AI Alternatives, Annoying API Bills**: Reports on **Auto Mode** are mixed, with claims that it's *truly unlimited* and offers engaging prompting challenges, while others find models like Claude offer dramatically superior results.
   - Cost concerns persist, with some users reporting up to **$50** in **API** costs, suggesting the mode switches to **GPT4.1** when usage is high.
- **Cursor's Code Completion Conundrums Continue**: Users discuss **tab autocomplete's** limitations, noting that it doesn't read project rules or custom modes, leading to basic suggestions, and the lack of cross-file suggestions unless dependencies are explicitly imported.
   - Some are employing **READMEs** to inject rules, though consistency remains an issue.
- **Cursor 1.3 lands with terminal sharing, faster edits**: The new **Cursor 1.3** release includes [terminal sharing with agents](http://cursor.com/changelog).
   - Users can now view context usage in Chat and also expect faster edits, among other fixes detailed in the changelog.
- **Background Agents: Mobile UI Needs a Makeover**: Users are requesting **UI** improvements on the mobile web interface for managing background agents, citing issues such as **textbox unfriendliness, diff view difficulties, and conversation update failures**.
   - A user also suggested support for **Whisper voice input** for better code transcription.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **AgentSmith Launches Open Source Prompt CMS**: **AgentSmith**, an **open-source prompt CMS** built on top of **OpenRouter**, launched to streamline prompt/context engineering ([GitHub](https://github.com/chad-syntax/agentsmith), [landing page](https://agentsmith.dev/)).
   - **AgentSmith** connects to your **OpenRouter** account, utilizing your credits, with the option for self-hosting, with discussions ongoing around client-specific templates similar to **Claude Code's YAML header format** ([docs](https://docs.anthropic.com/en/docs/claude-code/sub-agents#file_format)).
- **GLM Pricier than Kimi**: Users noted that **GLM** is more expensive than **Kimi** due to its long reasoning capabilities, while praising advancements in open-source models like **Qwen3** for architecture tasks.
   - The community expressed general excitement about advancements in open-source models despite pricing discrepancies.
- **Deepseek V3 gets the Error 401 Blues**: Users reported experiencing **error 401** with the **Deepseek model**, suggesting potential API key issues and temporary outages from **Chutes**.
   - There were also reports of *"All providers ignored"* errors with Deepseek V3 in the **#general** channel.
- **OpenRouter Balances Quality Factors**: OpenRouter is actively working on their **standard lane routing**, aiming to balance factors like throughput, latency, and tool call success rates beyond just price.
   - The goal is to define what *"best"* means through a variety of factors, moving beyond offering the cheapest version, and to meet user requests like creating a **best quality option/preset** for end users.
- **Prompt Engineering to prevent unwanted behaviors?**: Users discussed methods for preventing unwanted behaviors, such as repetitive sentence structures in **Deepseek V3**, suggesting prompt adjustments and negative prompts (*"never wrap up the scene"*).
   - A member linked to a [Reddit thread](https://www.reddit.com/r/JanitorAI_Official/comments/1kd1iia/guide_day_7_how_to_prevent_deepseek_from_leaving/) with potential solutions on prompt engineering.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT Hits the Books with Study Mode**: OpenAI launched **study mode** in ChatGPT, designed to assist students in achieving *deeper understanding* by guiding them through problems step-by-step, as detailed in their [announcement](https://openai.com/index/chatgpt-study-mode/).
   - The new mode focuses on fostering a more profound comprehension of the subject matter, steering away from simply providing answers.
- **Copilot Vision Steals the Show**: Members are impressed with **Copilot Vision** in the Edge browser, noting its nailed UI, seemingly unlimited usage, and relatively accurate results, according to [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
   - The same user continues raving about Copilot Vision in Edge, especially how cool it is now at [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
- **GPT-5 Launch Rumors Swirl**: Rumors circulated about the imminent release of **GPT-5**, boasting a **1M token** input window, **100k** output tokens, **MCP** support, parallel tool calls, and dynamic short + long reasoning.
   - However, other users asked for *credible sources* to back up these claims.
- **Zenith Tipped as Coding Champ**: A user predicted that **Zenith** will be the top coding model based on personal testing and shared examples.
   - The user qualifies their prediction, stating it will be true *until another comes out*.
- **Memory Format Prompt**: A member shared a prompt for a new **AI memory format** featuring concepts such as **Token Embeddings**, **Semantic Grouping**, and **Binary Encoding**.
   - The format aims for **speed**, **efficiency**, and **fidelity** while obscuring the content from human readability, incorporating aspects like **Header Blocks**, **Global Embeddings**, and **Update Deltas**.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Voxtral Mini Faces Compatibility Issues**: Members discovered that using [Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF) with **LM Studio** is currently not possible.
   - The model is not yet supported, according to users, with no further details given.
- **LM Studio Performance Nosedives**: A user reported a significant drop in token generation speed with **Qwen 3 30B A3B**, from *50-80 tk/s* to *30-35 tk/s*, seeking advice in the **LM Studio general channel**.
   - Suggested fixes included disabling **flash attention** and verifying other specific settings, followed by unloading and reloading the model.
- **OpenWebUI and LM Studio Connect**: Users confirmed compatibility between **OpenWebUI** and **LM Studio**, offering configuration tips for **Docker** setups, such as retrieving the **LLM base URL** from LM Studio.
   - One user resolved connection issues by enabling **CORS** in LM Studio and ensuring the correct IP address was used.
- **LM Studio Gains Offline TTS Voice**: A user shared a Python script to [bridge the LM Studio API with XTTS WebUI API](https://cdn.discordapp.com/attachments/1110598183144399058/1399764223696830474/bridge-LMStudio-XTTS.py?ex=688a2f85&is=6888de05&hm=eec4e5dcb6b55bf09ee4282441d1fa35a166fd0392ff1c81116c964188a51f16&), enabling **offline TTS voice integration** within LM Studio.
   - This integration allows audio responses to be played directly in the command prompt.
- **AMD Ryzen AI MAX+ Falls Flat**: A user shared a [video](https://youtu.be/BTIUL-yaY4s?t=487) showing that the new **AMD Ryzen AI MAX+ 395** with **128gb** achieving only **2-3 T/s** on **Llama 70b Q6**, with only half the memory allocated to the GPU.
   - The user mentioned that *this performance makes it not really a viable option compared to even RTX 30xx Nvidia GPUs*.



---



## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **DeepSeek R1 Democratizes AI**: The [DeepSeek R1 launch](https://deepseek.com/blog/deepseek-r1) has reportedly democratized the AI space, driving down inference costs that were previously *multiple dollars per 1M/token*.
   - Despite its power, **Deepseek R1** is said to be *lacking agentic capabilities*.
- **API Keys Throw 403 Errors**: Users are encountering **403 errors** with the message *"<xxxxxx> is not active, Incorrect API key provided"*, even after topping up accounts.
   - Workarounds include waiting, regenerating the API key, and switching to **OpenRouter**, with the error specifically noted in **Claude code** but not the playground.
- **Kimi K2 Sparks Archival Urge**: One user is archiving **Kimi K2** on a multi-TB setup due to its unique *spark* that sets it apart from other models, open and closed.
   - They expressed deep admiration, stating *"Even if something bad happens to Kimi, it will still be mine"*, citing its *vibes* and *soul*.
- **Kimi's insatiable appetite for TeraBytes**: Disk space requirements for **Kimi K2** range *from 200 GB to 2 TB*, contingent on quantization.
   - Suggested storage solutions include **Azure Blob Storage**, estimated at *$6/month for 3TB of data*.
- **Kimi Learns to Emoji**: One user is actively teaching their **Kimi** bot to use emojis and to adjust any unexpected behavior, envisioning a "lil sis".
   - This user anticipates the launch of a **Kimi agentic model**, foreseeing excitement around the **Kimi distil**.



---



## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM's Video Overviews Slideshow**: Google's **Video Overviews** feature is rolling out to web users and includes image generation, text generation, and video composition powered by AI; see [Help Center article](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.).
   - Initial feedback indicated the feature currently generates a **slideshow of text and AI images** instead of animated videos and has a rate limit of **20 video generations per day** for pro users.
- **Gemini Agentic Framework Released**: A member shared a [prototype Gemini agentic framework](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688a8276&is=688930f6&hm=d111cb9b690a7969af3e128a78c205e2d89bf790babf9c1ab08c72cdc9f89ead) for testing and experimentation.
   - Its functionality is not 100% guaranteed.
- **Studio UI Updates**: A new **Studio UI** is rolling out with features like creating multiple outputs from the same sources and selecting sources when creating outputs.
   - This is part of a series of recent rollouts, including **Featured Notebooks** being fully accessible and **Video Overviews** initially available in **English** and on **Desktop**.
- **PDF Uploads Break**: Some paid NotebookLM users reported getting an **'error uploading source, try again'** message when uploading PDFs.
   - A Google employee acknowledged a **bug** and stated they're investigating.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **New Zenith Models Spark Speculation**: New anonymous AI models (**Zenith**, **Summit**, **Lobster**) appeared on LM Arena with good 'vibe coding' capabilities, sparking speculation about whether they are **OpenAI's GPT-5** variants or **Kimi-2**, according to [a Reddit thread](https://reddit.com/r/LocalLLaMA/comments/1m9holp/theres_a_new_kimi_model_on_lmarena_called_zenith/).
   - A comment from the same thread *almost confirms* that `zenith` is an OpenAI model because *it uses the same tokenizer as gpt-4o, o3 and o4-mini*.
- **LlamaIndex Agents Now Scrape with Oxylabs**: **LlamaIndex** integrates with **Oxylabs**, letting users create cost-effective AI agents for real-time web search and scraping via [specialized readers](https://xcancel.com/llama_index/status/1949937947727245639) for Google, Amazon, and YouTube.
   - This should help with agents doing real-time data access and retrieval with better data quality and scale.
- **AI Pricing is Still a Riddle**: A product developer is finding it difficult to price their AI product (a PR reviewer tool), highlighting the challenges in **metering** and the need for flexible pricing [as linked to Lenny's Podcast re: product growth](https://podcasts.apple.com/us/podcast/lennys-podcast-product-growth-career/id1627920305?i=1000719362528).
   - The thread discusses pricing models (**fixed vs. variable**, **price per PR**, **rate limits**) and user confusion around token-based pricing.
- **Fireworks Notches $4B Valuation**: **Fireworks AI** is proceeding with an equity raise at a **$4B valuation**, declining M&A offers from Coreweave and Meta, with **~$130M ARR**, **20x+ YoY growth**, and profitability [according to reports](https://xcancel.com/ArfurRock/status/1950222707116707990).
   - The company says it is burning rubber in the AI compute and model space.
- **Arcee.ai Drops Versatile AFM-4.5B**: **Arcee.ai** released the **AFM-4.5B** and **AFM-4.5B-Base models** on Hugging Face, designed for flexibility and performance, with an emphasis on quality via data partnership with **DatologyAI** [as announced on X](https://xcancel.com/LucasAtkins7/status/1950278100874645621).
   - This model is versatile across many tasks and optimized for ease of use in a variety of applications.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Qwen 30B Competes with GPT-4o**: The release of **Qwen's 30B model** was discussed, with claims that it rivals **GPT-4o** and can be run locally with **33GB RAM**.
   - A link to the **Unsloth Qwen3-30B-A3B-Instruct-2507-GGUF** model on [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) was shared, with another user noting it requires 17GB RAM at Q4_K_XL.
- **Diffusion Models Study Group Starts**: A **5-month study group** limited to 12 participants, requiring **2–4 hrs/week**, will explore diffusion models based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
   - The first **two intro sessions are free** and open to non-members: Aug 2 on Flow Matching & Diffusion Models and Aug 9 on PDEs, ODEs, SDEs, both at 12 PM EST ([sign up here](https://lu.ma/kv8zf6va)).
- **ragflow Probed in Production**: A user inquired about experiences with **ragflow** in production environments, specifically asking about potential problems and general suitability.
   - The conversation was redirected to the dedicated channel, and a link to the [ragflow GitHub repository](https://github.com/infiniflow/ragflow) was provided.
- **Models Face Uploading Issues**: A member noted they forgot to upload the **custom model classes** (architectures) for their models on the Hub, meaning *none of them can be loaded properly right now*, and they're essentially unusable.
   - They are rebuilding everything from scratch with the correct architecture files, better documentation, and proper inference support.
- **ViT/ResNet Take on balding image classifications**: Instead of using LLMs, members recommended using vision models like **ViT** or **ResNet** to classify photos with the **Hamilton-Norwood scale** for male pattern baldness.
   - A member provided relevant links like [this article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10974725/).



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLMs as Interp Agents Gain Traction**: Members are exploring the use of **LLMs as interpretation agents**, referencing **Transluce's** work and [Sarah's MAIA paper](https://openreview.net/forum?id=mDw42ZanmE) as key resources.
   - The idea is to leverage **LLMs** for automated mech interp research, building upon existing knowledge in the field.
- **Modelscope Emerges as Chinese Hugging Face**: **Modelscope.cn** is being described as the Chinese equivalent of **Hugging Face**, providing AI models and tools within China.
   - Due to restrictions, **Hugging Face** is not accessible in China, as confirmed by [this article](https://www.semafor.com/article/10/20/2023/ai-platform-hugging-face-confirms-china-blocked-it).
- **Peer Pressure's Influence on AI Explored**: A member shared their research preprint on **AI peer pressure**, analyzing over **200** AI-to-AI conversations to study model complexity and susceptibility.
   - The study achieved **121%** statistical power and can be accessed on [Zenodo](https://zenodo.org/records/16573783), with feedback actively sought.
- **MATS 9.0 Seeks Mechanical Interpreters**: Neel Nanda announced that applications are now open for **MATS 9.0**, a mentorship program focused on paid mech interp research.
   - The program aims to guide participants in producing a high-quality [research paper](https://tinyurl.com/neel-mats-app) in the field.
- **New Diffusion Study Group Kicks Off**: A new **diffusion models study group** has commenced a **5-month** program with **12** members, drawing from [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
   - The curriculum includes 2 hours of live class and 2 hours of self-study.



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **B200 Bare Metal Bonanza**: A member sought cloud providers offering single **NVIDIA B200** GPUs on bare-metal servers for modifying kernel drivers and configuring the **GPU**.
   - This sparked discussion around potential use cases, specifically the need for **ncu (NVIDIA Compute Utility)** support, with inquiries about cloud providers with single **B200** instances.
- **Triton's Torch Tease: Code Extraction Tips**: Users explored methods to extract **Triton** and **PTX code** generated by **torch.compile**, with a user sharing that `TORCH_LOGS="output_code" python your_code.py` will output the PTX code.
   - A member suggested checking the `compiled_kernel.asm.keys()` dictionary, pointing to [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir) for more details, while others discussed forcing **torch inductor** to generate exclusively **Triton code**.
- **DTensor Debut: Stream Incoming**: A member announced a stream to learn about **DTensor** in more detail and shared a [YouTube link](https://www.youtube.com/watch?v=b8lplOf2g4g&ab_channel=MarkSaroufim) to the stream.
   - Another shared a [gist](https://gist.github.com/S1ro1/4fe38e3a0fff3d84314935a0e05aed9c) fixing a weight initialization error where random initializations caused different shards on each rank.
- **Tackling Triton's Troublesome TMA**: A user inquired about the ability of the **Triton compiler** to perform a **GEMM** (General Matrix Multiplication) with a **ping-pong schedule**.
   - The ability depends on the version of the Triton compiler, as TMA (Tensor Memory Accelerator) support is not yet available in official releases, suggesting waiting for version **3.4.0**.
- **denvdis dissects CUBIN Conundrums**: A member created a tool named **denvdis** to extract and replace **CUBIN** files within **ELF fatbinaries**, which can be found [here](https://github.com/redplait/denvdis/tree/master/fb).
   - Files replaced must have the same size as the original, compressed fatbinaries are not supported, and there are *no deps from nvidia sdk*.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MoE Models Dominate APIs Due to Efficiency**: The community debated the rise of ~**500B** parameter **MoE** models over ~**30B** dense models, with one member noting that practically every API model from last year has been a **MoE** model due to efficiency gains.
   - While **MoE** models excel in benchmark testing, the discussion suggested that dense models may capture more nuance but are harder to finetune, but the efficiency gains of **MoE** are too hard to ignore.
- **Local LLM Finetuners Face Bottleneck**: Local LLM finetuners are reportedly *stuck with gemma 3 and qwen 3*, struggling to access **10-70B** models suitable for local finetuning/running without resorting to APIs.
   - A member suggested that local development might be shifting towards **MoE** models for better efficiency.
- **Anthropic API Receives Backlash over Strict Limits**: Members heavily criticized **Anthropic's** API for its restrictive limits and expensive pricing, referencing [this tweet](https://x.com/anthropicai/status/1949898502688903593?s=46) detailing weekly restrictions.
   - Concerns were raised that **Anthropic's** approach might lead to its downfall if a superior alternative emerges: *waiting half a day for claude is already bad, but an entire week?*
- **Qwen3-30B-A3B-Instruct-2507 Finally Arrives**: The [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) model has been officially released on Hugging Face.
   - This follows a previous slip up, marking its accessibility to the wider community.



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Sparsity Soars as Model Sizes Swell**: As model size increases, the optimal **sparsity** for a fixed budget also increases, especially during training runs, making it ideal for achieving maximum performance at the lowest cost with faster training and increased total parameter count.
   - While **sparsity** might hinder performance for end-users with extensive GPU resources and training budgets, it excels in reducing the **dollar cost** to achieve the most performance; a member also linked to [Is 32K context length enough](https://x.com/LTSI_TheCaptain/status/1950272036074582040).
- **Modern Dropless MoEs Dominate**: Modern dropless **Mixture of Experts (MoEs)** are outperforming the geomean rule (`effective parameter count ~= sqrt(active * total parameters)`) due to a number of optimizations.
   - **MoEs** can train **2.2 to 2.5x** faster than dense networks with the same parameter count, with new approaches hitting **3.2x** speeds and **Baidu** reporting **~50% MGH** for **ERNIE-MoE** training.
- **Kimi K2 Challenges Claude's Reign**: Despite being sparser, **Kimi K2** is proving competitive against **Claude**, which is known as a large dense model.
   - This competitive edge is likely due to extensive **Reinforcement Learning (RL)** rather than pure architectural advantages, which had previously given **Claude** an edge on agentic tasks.
- **YouTube Shorts alienating core users**: Members suggest **YouTube** is emphasizing **Shorts** to combat **TikTok**, failing to recognize its user base dislikes TikTok and that **shorts** generate significantly less revenue, which members state is *more about market share. People spend time on a video platform that is not youtube. That is lost revenue which is simply unacceptable*.
   - A member stated that **TikTok's recommendation algorithm** is simpler but *much more thorough than youtube's*.
- **ChatGPT Study Mode: Endgame?**: Members reacted to [OpenAI's ChatGPT Study Mode](https://openai.com/index/chatgpt-study-mode/) and considered it *moving closer to the endgame*.
   - Another member considered *this is a violation of openai business model, which is designed to maximally disrupt formal education*.



---



## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Blockchain Gets Love for Authorization & Payments**: Authorization management and payment guardrails are considered easier to implement on **blockchain** compared to web2 due to the unified account and precise control it provides.
   - A member suggested unifying payment solutions across model providers like an app store, especially for local or self-hosted models, streamlining transactions and access.
- **Monday.com Snags AI Engineering Dream Team**: [Monday.com](https://www.monday.com) hired two AI engineers to work on **MCP** full-time, solidifying their investment into the project as originally reported [here](https://www.calcalistech.com/ctechnews/article/rjorfh8wel).
   - The new hires are expected to accelerate development and bring fresh perspectives to **MCP's** capabilities within the **Monday.com** ecosystem.
- **MCP Server Deployment Faces Connection Trials**: A member deploying an **MCP server to EC2** with correctly configured SSL and domain settings faced connection issues with **Claude Desktop**.
   - While **Cursor** connected successfully, **Claude Desktop** failed, highlighting potential compatibility issues with specific clients.
- **One-Click VS Code MCP Server Installation Arrives**: A website [vscodemcp.com](https://vscodemcp.com/) now provides a **one-click install button** to add an **MCP Server to VS Code**, lowering the barrier to entry for developers.
   - A [walkthrough video](https://youtu.be/1JcRtQxmh3I) accompanies the installer, guiding users through the setup process and showcasing the benefits of the extension.
- **Nexus Launches as Mobile App Store for AI Tools**: The alpha version of **Nexus**, a mobile app store for AI tools (MCP servers), launched with features like **one-click install**, **no JSON configs**, and **chat integration**.
   - Available at [getnexus.in](https://www.getnexus.in) with source code on [GitHub](https://github.com/lucky-builds/get-nexus), **Nexus** aims to simplify the discovery and deployment of AI tools on mobile platforms.



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Optimizer Debuts with Goblin Flair**: A new **DSPy optimizer** sparked excitement, humorously illustrated by a [green goblin](https://cdn.discordapp.com/attachments/1203568372667645963/1399545934190477383/3195gc.png?ex=688a0cf9&is=6888bb79&hm=b5ef1dc40c8d2e735f8370a1be34553a2fd7cb46d86a6e67adab2b7ec0350fc3&).
   - The image's suggestion of an optimizer *too good to share* led to speculation about its potential prowess.
- **GEPA Sparks Metric Curiosity**: Interest in [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) prompted a request for comparison metrics.
   - The paper's author being present in the channel was noted as a resource for direct inquiries.
- **Tool Feedback Optimization Proposed**: A suggestion was made to optimize not only prompts, but also **tool response feedback** in case of errors.
   - Experiments confirmed the feasibility, using external compilers, profilers, and runtime executors as tools providing textual feedback.
- **Deep Dive into DSPy's Learnable Parameters**: Discussions revolved around `dspy.Variable` and `dspy.Parameter`, described as *some sort of learnable parameter* within **DSPy** programs.
   - It was proposed that `dspy.Variable` could enable users to specify optimizable elements, potentially even allowing **DSPy** to **test and rewrite parts of its own source code**.
- **AI Engineer Joins the Fray**: A Senior AI Engineer specializing in **agentic systems**, **workflow automation**, and **scalable no-code/low-code architectures** offered their expertise.
   - They listed tools like **Vapi AI**, **Make.com**, and **OpenAI**, and offered assistance in designing and deploying AI Agents, automating business operations, and fine-tuning LLMs & LSMs.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo's `external_call` Just Chops C Functions**: Mojo's `external_call` function facilitates direct calls to any **C ABI function**, provided the symbol is linked within the current address space.
   - The Mojo team advocates for minimal C code usage, requiring strong justification when equivalent Mojo code is viable, maintaining better **portability**.
- **Mojo Embraces File Descriptors**: Mojo incorporates file descriptor features through `io.file_descriptor.FileDescriptor`, aiming to reduce reliance on `external_call` within these functionalities.
   - Operating at the OS level with `read` and `write` enhances portability, aligning with Mojo's goals to minimize **external dependencies**.
- **Stdlib Development Doesn't Need No Compiler**: Most functionalities can be implemented in the **Mojo standard library** without altering the compiler.
   - This method exploits Mojo's adaptability and FFI capabilities, relying solely on the compiler's ability to *call this symbol that has a C ABI*.
- **Mojo Module Names Get Standardized**: A [feature request](https://github.com/modular/modular/issues/5094) aims to standardize naming conventions for Mojo modules, decoupling them from the Python API.
   - This initiative seeks to minimize confusion and bypass limitations inherent in Python's 30-year-old naming conventions to achieve **module name consistency**.
- **Max Ditches PyTorch 2.7 Dependency Soon™️**: The **PyTorch 2.7 dependency** will soon be removed in the next nightly build, enabling users to freely choose their PyTorch versions.
   - The team believes that **2.0** is the realistic lower bound for PyTorch compatibility with Max, although the minimum pinned version is **2.5**, which grants users the ability to manage their **PyTorch environments**.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **FlowMaker Flows into Focus**: **LlamaIndex** rolls out **FlowMaker**, a GUI tool for visually constructing **LlamaIndex workflows**, accessible at [flowmaker.llamaindex.ai](https://flowmaker.llamaindex.ai/), and demonstrated in [this video](https://youtu.be/MU6jA0rUlFY?feature=shared).
   - While one user hoped for **Python** exports, they're getting **Typescript** exports instead.
- **LlamaCloud's Nodes Now Natively n8n**: **LlamaIndex** introduces open-sourced **n8n nodes for LlamaCloud** (including LlamaCloud indexes, LlamaParse and LlamaExtract) available in the [`n8n-llamacloud` repo](https://github.com/run-llama/n8n-llamacloud), also demonstrated in [this video](https://youtu.be/5bQXHPSkuBw?feature=shared).
   - This enhancement streamlines intelligent document processing within existing **n8n workflows**, and simplifies embedding content management, eliminating the need for users to manage their own API keys.
- **Agents are Actually Actionable**: An upcoming webinar will demonstrate transforming intricate financial documents into actionable data via AI-driven document agents, powered by **LlamaCloud**'s parsing capabilities ([link](https://t.co/f0TKSzHQ2d)).
   - Seldo shares scalable agent design patterns, including **hybrid workflows** and **debuggability**, from the @aiDotEngineer summit, accessible via [this link](https://t.co/zeTaVOMate).
- **LlamaCloud PDF Predicaments Persist**: A member reported that **Llamacloud** is failing to detect and process a **PDF file** via **API**, using **n8n** for workflow simplification, and asked for assistance, including a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688a7e70&is=68892cf0&hm=213304e9d8a77128ab3cbc75d4c9114a73d0a157e12a0aa633bd2a62e160a5fa).
   - It was suggested ensuring the filename includes the proper **file extension** when working with **Llamacloud**.



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Grok Stinks at Prompt Generation**: A member suggested utilizing **Grok** for prompt generation for **Manus AI**, only to find it gave poor results.
   - Despite the poor results, a second member volunteered to help out with the task personally.
- **Manus Credit System Gets Grilled**: Members criticized **Manus' credit system** and infrequent updates, suggesting these factors could lead to a downturn despite event hosting.
   - One member proposed exploring alternative agentic systems such as **Lume** to evaluate comparative value.
- **Lume Edges Out Suna in Coding**: In a comparison between **Lume** and **Suna**, one member derided *Lume is suna but worse*
   - However, another member found **Lume** to be superior in coding tasks, citing fewer errors and debugged code, while acknowledging **Manus**'s strength in comic creation.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad Tensor's True Nature Revealed**: A member asked whether **tinygrad's Tensor** class wraps around an existing object, such as **NumPy ndarrays**, or has its own implementation.
   - The discussion also touched on performance compensation if **tinygrad** uses a wrapper.
- **PR #11410 Shuttered Without Explanation**: A member expressed surprise that [PR #11410](https://github.com/tinygrad/tinygrad/pull/11410) was closed without any comments shortly after an update.
   - Another member responded that *it missed the point and is not a good change*, suggesting the contributor review past merges to understand guidelines.
- **"Where" Operation Sparks Debate**: After a comment from geohot, one member reconsidered using a *where* operation after experimenting with holding assigned operations until kernelization/schedule creation.
   - Acknowledging potential side effects, they were surprised by the PR's closure without feedback, especially since they had planned deeper investigation.



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCLv4 Only Accepts Model Submissions**: A member asked if **BFCLv4** allows open-sourcing an agent system or offering an API key, but was informed that submissions are currently limited to individual models.
   - This restriction focuses the leaderboard on evaluating the core model performance rather than the surrounding agent infrastructure.
- **Multi-Agent System Submission Denied**: A question about submitting a multi-agent system containing multiple models was posed to the group.
   - The response confirmed that **BFCLv4** submissions are restricted to single, individual models only, maintaining a standardized evaluation framework.



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune Seeks LoRA Adapter**: A member inquired about **Torchtune's** support for a **LoRA-style adapter** that retains the existing forward compute path.
   - The user wants to freeze the original model weights and apply updates via additional trainable layers without altering computational cost.
- **RL Tests Drag On, Bug Suspected**: A member noted that **RL tests** are running for over **1 hour**, attributing it to a bug and proposed a separate **PR** for debugging **CI**.
   - The dedicated **Pull Request (PR)** will focus on debugging the **Continuous Integration (CI)** system.



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Diffusion Models Study Group Launches**: A new study group is forming to explore **diffusion models** from scratch over **5 months**, using [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) as a guide.
   - The group plans to dedicate **2-4 hours per week** to learning about a core architecture in **generative AI**.
- **Free Intro Sessions Available for Diffusion Models**: Two free introductory sessions are scheduled for **August 2nd** and **August 9th** at **12 PM EST**, covering **Flow Matching**, real-world use cases, **PDEs**, **ODEs**, **SDEs**, and the history of diffusion models ([session link](https://lu.ma/kv8zf6va), [another session link](https://lu.ma/uk6ecrqo)).
   - These sessions aim to provide an overview of the essential concepts and applications in the field of **diffusion models**.
- **Study Group Attracts AI Professionals**: The diffusion models study group has attracted various AI professionals, including a **CTO of an AI film tool**, an **AI art instructor**, **LLM instructors**, and **AI researchers**.
   - The initial **two sessions are free**, followed by a subscription model of **$50/month** for early sign-up (**$100/month** after) to compensate a teaching assistant and features peer-led sessions, mentor Q&A, hands-on projects, and real research papers.



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic Dataset Access Denied**: A member reported an **AccessDenied error** while trying to access the **nomic-ai/nomic-embed-text-v2-moe** dataset, following instructions from the [contrastors repo](https://github.com/nomic-ai/contrastors).
   - The error occurred during the `ListObjectsV2` operation when using the command `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive`.
- **Low-Spec System Seeks Model**: A member with a **Celeron N2815**, **4GB RAM**, and no GPU requested advice on which model would be best to run on their system.
   - No specific models were recommended in the provided messages, indicating the need for further community input.



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Community Welcomes New Faces**: New members are introducing themselves on Cohere's Community Discord Server, sharing their **companies, industries, and universities**, detailing their current projects and their favorite tech/tools.
   - They are also sharing aspirations for community engagement and what they hope to gain from the community.
- **Diverse Backgrounds Enrich Community**: The new members come from a wide array of backgrounds, including various **companies, industries, and universities**.
   - This diversity promises to bring varied perspectives and expertise to the community's discussions and collaborations.



---


The **LLM Agents (Berkeley MOOC) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---


The **Codeium (Windsurf) Discord** has no new messages. If this guild has been quiet for too long, let us know and we will remove it.


---



You are receiving this email because you opted in via our site.

Want to change how you receive these emails?
You can [unsubscribe]({{{RESEND_UNSUBSCRIBE_URL}}}) from this list.


---

# Discord: Detailed by-Channel summaries and links





### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1399531884093898872)** (1 messages): 

> `R1 1776 removal, Claude 4.0 Sonnet, Sonar model` 


- **R1 1776 model retired**: The **R1 1776** model will be removed from the model selector on web and mobile due to not keeping pace with recent improvements.
   - Users are recommended to try **Claude 4.0 Sonnet (Thinking)**, which offers similar strengths with stronger performance.
- **Sonar Model Stays Put**: There are no changes being made to the **Sonar** model or any other models.
   - The focus is on transitioning users from **R1 1776** to **Claude 4.0 Sonnet** for reasoning-focused tasks.


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1399471250916507679)** (1101 messages🔥🔥🔥): 

> `Comet browser cloud sync, Qwen3 30B model, unified memory, Nvidia 5080, OpenRouter` 


- **Comet Users clamor Cloud Sync**: Members wish for cloud sync in the next update for the [Comet browser](https://cometbrowser.com/), as it's currently preventing it from becoming their main browser.
- **Users find Qwen3 30B Model to be snappy**: Users found the new **Qwen3 30B model** to be a *decent jump* and surprisingly fast, consuming only around **17GB of RAM**.
- **Laptops with Unified Memory Run AI Models**: Some members run **30B quantized models** on their laptops with **32GB unified memory**, achieving around **40 tokens per second**.
- **AI Influencers found to be biased**: Members noticed that influencers may be biased when doing comparisons against [ChatGPT](https://openai.com/), often failing to use the reasoning mode or starting new chats for each query.
   - One member pointed out a specific review where the YouTuber recommended used [GPT-4o](https://openai.com/index/hello-gpt-4o/) but didn't fully take advantage of the potential of each product, but *I was kinda biased too recommending that review🔥*.
- **Kimi vs O3**: Users discussed preferences for **Kimi** due to its *personality* and how it works with specific prompts, while others recommend **O3 Pro** for study and learning.


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1399617401087328317)** (4 messages): 

> `Perplexity Deep Research API Support, Sonar Deep Research Output Issues` 


- **Seeking Support for Perplexity Deep Research API**: A member inquired about contacting someone at Perplexity to discuss the **Deep Research API** for product development.
   - Another member from Perplexity responded, offering assistance with any questions about the **API**.
- **Garbled Output Issue in Sonar Deep Research**: A member reported an issue with **Sonar Deep Research** returning partially garbled output, and a member from Perplexity confirmed that the team is investigating the issue and linked to the [resolved ticket](https://community.perplexity.ai/t/sonar-deep-research-returns-partly-garbled-output/809).


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1399466477769330799)** (757 messages🔥🔥🔥): 

> `vibe coding, WASM, FIPS 140-3, trl breaks everything, GLM 4.5 Air` 


- **Vibe Coding**: Several members discussed "**vibe coding**" and its implications, with one member mentioning their work vibe coding **FIPS 140-3** with [Nabla](https://github.com/Atelier-Logos/nabla/pull/49) and another calling **Lovable** "vibe coded".
   - There was also discussion about the risks of vibe coding in security contexts, where understanding the underlying vulnerabilities is crucial, as well as OpenAI making woke DEI models.
- **trl New Version Breaks Everything**: There was a report that the new **trl version** is breaking everything with `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`.
   - Members suggest that the fix [was to roll back torch ver](https://discord.com/channels/1179035537009545276/1179777624986357780/1399629661796827206).
- **GLM 4.5 Air model getting lots of hype**: A lot of members have been enjoying the [GLM 4.5 Air model](https://z.ai/blog/glm-4.5) for creative writing and multilingual tasks, with one member getting **35 tokens per second at 5 bit mlx**.
   - Other members compared this model's performance to Gemini as well as its strength as an uncensored model.
- **Unsloth Runpod Template incoming**: Unsloth will be releasing a runpod template with everything already set up, including jupyterlab, ssh access, nvidia container toolkit and notebook examples, allowing users to avoid spending precious compute minutes on setup.
   - It has been uploaded as a [docker container](https://hub.docker.com/r/unsloth/unsloth) that will be runpod compatible soon.
- **Is Model Merging worth it?**: Members had a discussion about techniques like merging all experts into a dense model, swapping them, or turning a dense model into an MoE model.
   - Others mentioned ESFT [repo for deepseek arch](https://github.com/deepseek-ai/ESFT) as a means of finetuning specific experts.


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1399586271298326538)** (17 messages🔥): 

> `HuggingFace Tokenizers, Windows 7 lifetime extension, Gemma 3 4B fine-tuning, RoPE Positional Encoding` 


- **HuggingFace Tokenizers Draw Ire**: A member expressed dissatisfaction with **HuggingFace Tokenizers**, citing issues with [adding custom tokens](https://huggingface.co/docs/transformers/tokenizer_summary) and renaming **<unused>** tokens.
   - They found that using custom tokens without adding them to the vocabulary yields better results and stated that they were today years old when they found you can mod **Windows 7** to get updates until **December 2025**.
- **Gemma 3 4B achieves fine-tuning**: After full fine-tuning of **Gemma 3 4B** with **16k** context, a member found that adding custom tokens is not useful unless you are training from scratch or have lots of data, and the watermark is completely removed.
   - They added that knowledge from different languages can help and posted a [screenshot of the results](https://cdn.discordapp.com/attachments/1179039861576056922/1399849804045221948/Screenshot_2025-07-29_at_2.13.07_PM.jpeg?ex=688a7f39&is=68892db9&hm=a15f9fb0aae9184041f136d967cb8b77419792df3692e34fdfe31204ac92bcdc&).
- **RoPE encoding praised, but needs optimizations**: A member gave high praise to the inventor of **RoPE positional encoding**, noting it works very well on smaller models and is valuable for supporting huge contexts on inference.
   - They believe that for supporting huge contexts, *"we need to invert some better optimizations (quantization is not enough, MoE doesn't help (plus my models are small enough), but definately keep it transformer-only, please)".*
- **AI Model Translation Skills Impress**: One member highlighted the translation abilities of the fine-tuned AI model, stating it *"knows every single language"*.
   - They exclaimed, *"OpenAI is doomed"*.


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1399468640788545536)** (106 messages🔥🔥): 

> `Gemma 3 finetuning issues, TRL downgrade for Unsloth, Qwen2-VL tokenizer issues, GGUF conversion problems, InternVL model loading errors` 


- **Gemma 3 Finetuning Frustrations**: A member encountered an error with **Gemma 3** during the save checkpoint process, sharing a [screenshot](https://cdn.discordapp.com/attachments/1179777624986357780/1399581583551365211/image.png?ex=688a2e2c&is=6888dcac&hm=b94b31197d17e7a15b8993e1b4cb9664c210a4367823f2657bda287414b6885b&).
   - Another member requested the notebook to investigate after learning that the user had *changed many things*.
- **TRL Downgrade Saves the Day**: A recent **trl** update was found to be causing issues, specifically with the `ConstantLengthDataset` which was deprecated.
   - The recommended solution is to downgrade **trl** to version **0.19.1** with the command `pip install --no-deps --force-reinstall trl==0.19.1`.
- **Qwen2-VL Tokenizer Text Torment**: A user ran into a `ValueError` when trying to use the **Qwen2-VL tokenizer** for text-only tokenization, encountering an issue with flat lists of images.
   - Further debugging revealed an `AttributeError` due to missing `convert_tokens_to_ids` in `Qwen2VLProcessor`, during attempts to extract text tokens from image-text inputs.
- **GGUF Conversion Grief**: A user faced a `ValueError` when trying to export a merged, fine-tuned **Qwen2.5-14B-Instruct** model to **GGUF**, specifically encountering a tensor mapping error: `Can not map tensor 'model.layers.0.mlp.down_proj.weight.absmax'`.
   - It was suggested to merge with the **FP16** version and to *quantize externally for now* due to a bug with **GGUF** in Unsloth.
- **InternVL Model Loading Impasse**: A user encountered a `ValueError` when loading the **unsloth/Intern VL 3 1B instruct** model, citing an unrecognized configuration class.
   - After troubleshooting, it was recommended to load the model with `trust_remote_code=True` and to use `automodel=AutoModel` from transformers.


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/)** (1 messages): 

marioz_70065: My humble exploit of unsloth has been published http://dx.doi.org/10.1002/isaf.70011
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1399552321234534574)** (32 messages🔥): 

> `LLMs Vision, Video encoders, Audio image relation, QwenOmni, Gemma 3 vision quantization` 


- **Debate if LLMs watch visions at 1fps**: Members discussed whether current LLMs process video like images at **1fps** or if they have true video encoders that can create a continuous vector of all frames.
   - One member questioned how the vector would look for videos with different numbers of frames, suggesting options like same size or directly proportional to the number of frames.
- **Architectures for encoding video into a single vector discussed**: A member proposed creating an architecture to encode any video into a single vector equivalent to **2048 tokens** as text tokens, but it might imply loss of detail for longer videos.
   - They suggested using vectors for each second or frame and creating proper relation between audio and image in a video.
- **Vision Embedding techniques examined**: Members explored whether having an encoder that understands sequential dependencies in data would make models smarter than just using images, and how to encode multiple modalities in parallel.
   - One member stated that the understanding happens inside the model, not when embedding, and that **SigLip** creates the embedding vector with some understanding of the image.
- **Gemma 3 vision quantization**: One member questioned the quantization version for **Gemma 3**'s vision part, noting that the convolution weight/filter *v.patch_embd.weight* remains in float32.
   - Another member clarified that quantizing vectors in models isn't worth it because they are sensitive to quantization and represent less than **0.5%** of the model parameters.


  

---


### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1399537830493159465)** (97 messages🔥🔥): 

> `Memory usage variability, Verifying LoRA Weight Updates, CUDA error debugging, ImportError TRL library, Unsloth training LLM` 


- **Runs show Memory Usage Instability**: During training, a user reported memory usage inconsistencies with the same configuration, fluctuating between **47 GB** and **36 GB**.
- **Ensure Your LoRA Weights Update Correctly**: When validating LoRA adapter weights after fine-tuning, avoid using `np.allclose()` as it may miss subtle but meaningful changes, especially in **LoRA A**, which is initialized with small Gaussian values.
   - It's recommended to use checksum/hash comparisons (e.g., MD5), compute the sum of absolute differences between tensors, or manually inspect tensor statistics to reliably confirm weight updates.
- **CUDA errors appear**: One user ran into a CUDA error: *Assertion `probability tensor contains either `inf`, `nan` or element < 0` failed*.
   - Another user suggested to compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
- **TRL library has ImportError**: A user reported an `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`.
   - The Unsloth team quickly fixed it and recommended users on Colab/Kaggle to delete runtime and refresh the notebook, or use `pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth unsloth_zoo` and `pip install --upgrade --no-deps huggingface_hub`.
- **Fine-Tune Gemma 3n Vision Encoder**: A user asked about fine-tuning **Gemma 3n's** vision encoder, and another user shared a [Colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Vision.ipynb) for guidance.


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1399469905631248414)** (591 messages🔥🔥🔥): 

> `GPT-5, GLM 4.5, Qwen, Data Privacy, Model Evaluation` 


- ****GPT-5 Spotted, Hype Intensifies****: A blurred image purportedly showing **GPT-5** has been [posted on X](https://x.com/ryolu_/status/1950163428389040431), fueling speculation about its release and capabilities, with some claiming it's already in use by **Cursor** staff.
   - The removal of a model called **Zenith** from the Arena, which some believe to be **GPT-5**, adds to the anticipation, with claims that it demonstrated a noticeable improvement over existing models.
- ****GLM 4.5: Gemini Distilled?****: Some users suspect **GLM 4.5** may be distilled from **Gemini** due to its tendency to start responses with *'Of course!'* and its longer response style, with a [HuggingFace Space](https://huggingface.co/spaces/zai-org/GLM-4.5-Space) available for testing.
   - Despite concerns about data collection, it is performing well in SVG and web benchmarks, with comparisons being drawn to past AI's.
- ****Qwen's Code Skills Arrive****: **Qwen 3** coder model has arrived in the leaderboard of the LLM Arena.
   - With new A3B architecture and thinking versions being anticipated for further releases.
- ****Data Privacy Debate Ignites****: Members discussed the implications of using LLMs from companies like **OpenAI**, highlighting concerns about data collection, storage, and potential misuse for targeted influence or sale to data brokers.
   - While some argue that using open-source models and not linking data to personal identities can mitigate these risks, others point out that even free tiers may involve data collection.
- ****Artificially Low Tool Calling Benchmark?****: Concerns raised about vendors being artificially lower due to their prioritization on tool calling and agent capabilities, now seen in **Opus Sonnet GLM4.5 and KimiK2**.
   - Some claim this makes many academic type benchmarks worse for some reason.


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1399481665323270165)** (1 messages): 

> `GLM-4.5, LMArena` 


- **GLM-4.5 and GLM-4.5 Air Debut in LMArena**: The LMArena platform gets new models added to its roster: **GLM-4.5** and **GLM-4.5 Air**.
   - These models are now available for evaluation and comparison on the [LMArena leaderboard](https://chat.lmsys.org/).
- **GLMs take the Arena**: The newly added **GLM-4.5** models are ready for head-to-head battles.
   - Users are excited to compare them and benchmark them against the top models in the [LMArena Leaderboard](https://chat.lmsys.org/).


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1399466907945537708)** (463 messages🔥🔥🔥): 

> `O3 performance, Windsurf vs Cursor, Context window usage, Cursor Auto mode improvements, Cursor models and pricing` 


- **O3 Speed Soars, Sanity Sinks**: Users note **O3** has become significantly **faster**, but also *dumber* in recent updates, likening it to being a **unified**, less thoughtful version.
   - Some users are trying out **Windsurf** as an alternative, seeking a better *thinking mode*.
- **Context Window Consumption Concerns**: Users are tracking context window usage, observing high consumption rates which impact overall chat performance; one user reached **95%** context usage due to prolonged chat sessions.
   - It was confirmed that these stats are shared with the [Cursor Doc AI](https://discord.com/channels/1074847526655643750/1074847527708393565/1398365704578662543), with changelogs coming soon.
- **Auto Mode Aces AI Alternatives**: Users are reporting mixed results, with some claiming **Auto Mode** is *truly unlimited*, and enjoying the prompting and context challenges while others are seeing dramatically better results with models like Claude; it was also said [Auto Mode] is *much better* than it was months ago.
   - Concerns over costs persist as some users found the Auto was costing them **$50 on API Cost**, suggesting it can switch to **GPT4.1** when usage is high.
- **Debugging Data Woes and Defeating Duds**: A user reported that with the new update when **Cursor** uses the terminal it just runs the command and freezes.
   - Others mention experiencing frequent command hangs and the need to manually skip steps to continue.
- **Cursor's Code Completion Conundrums**: Users discuss **tab autocomplete's** limitations, noting it doesn't read project rules or custom modes, leading to basic suggestions, while others mention the lack of cross-file suggestions unless dependencies are explicitly imported.
   - Some users employ READMEs to inject rules, though consistency remains an issue.


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1399506886117818540)** (8 messages🔥): 

> `Background Agent UI, Background Agent Snapshots, Local Background Agents, Background Agent Formatting, Docker Build Cache` 


- **Mobile UI needs Love**: A user requested UI improvements on the mobile web interface for managing background agents, listing issues such as **textbox unfriendliness, diff view difficulties, and conversation update failures**.
   - They also suggested that it would be valuable to support **Whisper voice input** for better code transcription.
- **Cursor snapshots button gone**: A user reported not being able to find the **"take a snapshot" button** referenced in the [Cursor documentation](https://docs.cursor.com/en/background-agent#base-environment-setup).
   - They reported running **Cursor 1.2.4, VS Code 1.99.3 on OSX**.
- **Cursor backend agents not running locally**: A user reported that they could not run **Cursor backend agents locally** and posted an image showing the **s3 block is back**.
   - It is unknown what the fix is, as it was not found via searching for the error message in the channel.
- **Running Commands at the End of Background Agent Runs**: A user inquired about running a command (specifically a formatter) at the *end* of a background agent's run.
   - They noted that the documentation mentions `terminals` can be run during setup at the beginning, but not at the end, and were considering using a **Cursor rule**.
- **Busting the Docker Build Cache**: A user asked about how to bust the build cache when using a **custom Dockerfile** for background agents.
   - Another user suggested creating a `pre-build.sh` script with `docker builder prune -f` and adding it or other commands like `docker system prune -a` or `docker rmi your-image:previous-tag`.


  

---


### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1399815390628020364)** (1 messages): 

> `Cursor 1.3 Release, Terminal Sharing with Agent, Context Usage in Chat, Faster Edits` 


- **Cursor 1.3 lands with terminal sharing and more!**: Cursor 1.3 is out with quality of life improvements, including the ability to [share the terminal with an Agent](http://cursor.com/changelog).
   - Users can now view context usage in Chat and also expect faster edits, among other fixes detailed in the changelog.
- **Agent Terminal Sharing Debuts**: One key feature in Cursor 1.3 is the ability to share the terminal with an Agent, enhancing collaborative coding.
   - This allows for more direct interaction between the agent and the coding environment.


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1399767467827527783)** (11 messages🔥): 

> `AgentSmith launch, OpenRouter integration, Agent templates` 


- ****AgentSmith** Launches as Open Source Prompt CMS**: A member announced the launch of [AgentSmith](https://agentsmith.dev/), an **open-source prompt CMS** built on top of **OpenRouter** ([GitHub](https://github.com/chad-syntax/agentsmith)) to streamline prompt/context engineering.
   - Other members praised the landing page as *:chefkiss:*
- ****AgentSmith** Integrates with **OpenRouter** Account**: **AgentSmith** connects to your **OpenRouter** account, utilizing your credits, with the option for self-hosting.
   - A user jokingly mentioned that they were *getting inspiration* from the project, but *not giving credits*.
- **Agent Templates Proposed for Specific Clients**: A user suggested adding templates for specific clients, referencing **Claude Code's YAML header format** ([docs](https://docs.anthropic.com/en/docs/claude-code/sub-agents#file_format)).
   - The creator responded he'd have to do some digging to see what other clients do for this.


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1399470251107946567)** (347 messages🔥🔥): 

> `GLM vs Kimi pricing, Model Settings being deleted randomly, 401 error with Deepseek, Qwen3 as architect, GPT 4.1 web search issues` 


- **GLM Pricier than Kimi**: Despite being exciting, **GLM** is more expensive than **Kimi** due to its long reasoning capabilities.
   - Users discussed the merits of **Qwen3** for architecture tasks and expressed gratitude for the advancements in open-source models.
- **Deepseek V3 error 401 surfaces**: Users reported experiencing **error 401** with the **Deepseek model**, with suggestions pointing to potential API key issues.
   - Others mentioned ongoing issues with Deepseek V3, including "All providers ignored" errors and temporary outages from **Chutes**.
- **OpenRouter's free requests**: OpenRouter's free usage limits are up to **20 requests per minute** for free models, with daily limits of **50** or **1000 requests** depending on credit purchases as per [the documentation](https://openrouter.ai/docs/api-reference/limits).
   - Members also shared the [activity page link](https://openrouter.ai/activity) for checking their activity.
- **DeepSeek's efficiency on H800s**: Deepseek showcased their setup with **~2200 H800s**, achieving **700B+ input** and **168B output in 24hrs**, demonstrating efficient hosting capabilities.
   - Others raised concerns about the capacity of **Groq's LPUs** compared to GPUs.
- **Prompt Engineering Saves the Day?**: Users discussed methods for preventing unwanted behaviors in language models, such as repetitive sentence structures in **Deepseek V3**, suggesting prompt adjustments and negative prompts (*"never wrap up the scene"*).
   - One member linked to a [Reddit thread](https://www.reddit.com/r/JanitorAI_Official/comments/1kd1iia/guide_day_7_how_to_prevent_deepseek_from_leaving/) with potential solutions.


  

---


### **OpenRouter (Alex Atallah) ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1399699153176498247)** (9 messages🔥): 

> `OpenRouter PR, Model Quality Transparency, Standard Lane Routing, DeepSeek Model Complaints` 


- **OpenRouter's PR Nudged for Normie Newbies**: Some members suggested that OpenRouter's PR might suffer because new users may not know they can change providers, citing complaints about **DeepSeek** models in chutes.
   - It's like going to a restaurant and ordering the first thing you see rather than asking the waiter what to order or what the restaurant is known for.*
- **Model Pricing Doesn't Reflect Quantization Quality**: A member noted that OpenRouter's reputation already suffers by favoring the cheapest models, often blaming **DeepInfra** quants for poor performance.
   - Many users don't know what quant is, or even that OR doesn't host the models themselves, and assume best-case quality when they see a price for a model.
- **Determinism is Difficult and Expensive**: External pressure exists that leads to a race to the bottom, both for the providers (find cheaper inference) and for OR (find cheaper providers).
   - Requiring 100% determinism is difficult - why bother spending effort and compute on 100% determinism if there's no demand for it? Especially since big-name providers like **OpenAI** and **Anthropic** don't have deterministic outputs.
- **Standard Lane Routing Balances Quality Factors**: OpenRouter is actively working on what they call their **standard lane routing** which right now sorts purely by price, but they wanna consider other factors like throughput, latency, objective data on things like tool call success rates, possibly quantization, essentially reaching more towards which provider offers the best version of a model.
   - They are kind of trying to define what best means here through a variety of factors rather than just *here’s the cheapest version*.
- **Quality Preset Requested for End Users**: One member requested a **best quality option/preset** for end users that doesn't want to think too much about these issues and has not the time to check each provider regularly.
   - No one from the OpenRouter team responded to this request.


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1399800570977583286)** (1 messages): 

> `ChatGPT Study Mode, AI and Education, Step-by-Step Learning` 


- **ChatGPT Gets Schooled with Study Mode**: OpenAI launched **study mode** in ChatGPT to encourage *deeper understanding and learning* for students, now becoming a *go-to tool*.
   - Rather than just providing answers, study mode helps users work through problems **step-by-step**, as detailed in their [announcement](https://openai.com/index/chatgpt-study-mode/).
- **Step-by-Step Learning with ChatGPT**: The new **study mode** in ChatGPT focuses on guiding students through problems instead of directly providing solutions.
   - This approach aims to foster a more profound comprehension of the subject matter.


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1399477889782775911)** (225 messages🔥🔥): 

> `Copilot Vision in Edge, GPT-5 Release Date, Reasoning depth slider, Gemini models in Google Drive, AI Agency for automation` 


- **Copilot Vision Aces UI, Avoids Hallucinations**: A member finds that the Copilot vision in Edge browser is pretty sweet with a nailed UI, seemingly unlimited usage, and the hallucinations aren't completely wrong, as seen on [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
   - The same user continues raving about Copilot Vision in Edge, especially how cool it is now at [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
- **GPT-5 Dropping Next Week?**: A user shared news that **GPT-5** is dropping next week with a **1M token** input window, **100k** output tokens, **MCP** support, parallel tool calls, and dynamic short + long reasoning.
   - Another user replied to this claim, asking for a credible *source*.
- **Slide into Reasoning-Depth Slider**: A user suggests a *“reasoning‑depth”* slider that lets users choose between faster replies and deeper analysis.
   - The user seeks for prompt ideas that only reasoning models get correct.
- **Which Gemini Powers Google Drive?**: A user inquired about which **Gemini model** is running in **Google Drive**.
   - Another user said that hopefully it’s **2.5 Pro**, and that they recently did a direct comparison with Flash again and the difference is remarkable.
- **GPT-6 Hype in 2040?**: A user jokes that they are already bored with GPT5 hype and asks if we can start the **GPT6** hype already at [drinkoblog.weebly.com](https://drinkoblog.weebly.com).
   - Another user replied that **GPT 6** will come after **GTA 6**, so 2040.


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1399779429609111716)** (3 messages): 

> `Scholar ChatGPT, GPT-5 Versions, Zenith Coding Model` 


- **Users Explore Scholar ChatGPT**: A user inquired about the frequency of use of the custom **Scholar ChatGPT** among the members.
   - They were likely looking for feedback on its effectiveness for scholarly tasks, but no one replied.
- **GPT-5 Tiers Speculated**: A user speculated that the mid and high tier versions of **GPT-5** will likely outperform lower tiers.
   - The user didn't provide any evidence.
- **Zenith Touted as Top Coding Model**: Based on personal testing and shared examples, a user predicted that **Zenith** will be the top coding model.
   - The user states that this will be true *until another comes out*.


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1399658725949571082)** (6 messages): 

> `GPT project resources, Personalized Model Interactions, AI Memory Format Prompt` 


- **Newbie Seeks GPT Project Resources**: A new user is looking for [resources to set up projects](https://chatgpt.com/s/t_68887bb61f8481919199f93b3331e632) on their **GPT account**, specifically for tracking **food/diet, exercise**, and creating a **planner** with time expectations.
- **Personalized Model Interactions**: A member suggests directly interacting with the **GPT model** to personalize projects, tailoring them to specific needs and preferences.
   - They emphasized that defining what 'more powerful' means is crucial, as it varies from *personalized sports interests* to *performance comparison*.
- **AI Memory Format**: A member shared a prompt for a new **AI memory format** featuring concepts such as **Token Embeddings**, **Semantic Grouping**, and **Binary Encoding**.
   - The format aims for **speed**, **efficiency**, and **fidelity** while obscuring the content from human readability, incorporating aspects like **Header Blocks**, **Global Embeddings**, and **Update Deltas**.


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1399658725949571082)** (6 messages): 

> `GPT project resources, Personalized model interaction, New memory format` 


- **New User Seeks GPT Project Resources**: A new user is seeking [resources and guidance](https://chatgpt.com/s/t_68887bb61f8481919199f93b3331e632) to set up **GPT projects** for tracking food/diet/exercise and creating a planner.
   - The user hopes to add guidance and tools to the instructions to make it more powerful.
- **Personalized Interactions Supercharge Models**: A member suggested new users try talking to the model to personalize their experience.
   - They suggested discussing what the user wants, how they want it to look, and other considerations.
- **New Memory Format on Display**: A member shared a prompt detailing a **new memory format**, asking what others think.
   - The format includes sections such as *CORE_PRNCPL*, *STRUC_CONC*, and *EXP_WHY*, focusing on token embeddings, semantic groups, and binary encoding.


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1399467198845554900)** (196 messages🔥🔥): 

> `Voxtral Mini usage, LM Studio Performance drops, GLM 4.5 Tool Support, OpenWebUI setup with LM Studio, Qwen3 model` 


- **Voxtral Mini: LM Studio Support Pending**: Members inquired about using [Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF) with LM Studio, but it's **not yet supported**.
- **LM Studio Performance Plummets: Troubleshooting Underway**: One user reported a significant drop in token generation speed with **Qwen 3 30B A3B**, from *50-80 tk/s* to *30-35 tk/s*, and troubleshooting steps are being attempted.
   - Suggested solutions include checking specific settings (such as turning off a certain option shown in an attached image) and unloading/reloading the model, as well as turning OFF **flash attention**.
- **OpenWebUI and LM Studio Unite**: Members discussed using **OpenWebUI** with **LM Studio**, confirming compatibility and sharing configuration tips for Docker setups, including obtaining the **LLM base URL** from LM Studio.
   - A user encountered issues connecting **OpenWebUI** to **LM Studio** within Docker, which was resolved by enabling **CORS** in LM Studio and using the correct IP address.
- **Python Script TTS Voice addition to LM Studio**: A user shared a Python script to [connect the LM Studio API with XTTS WebUI API](https://cdn.discordapp.com/attachments/1110598183144399061/1399764223696830474/bridge-LMStudio-XTTS.py?ex=688a2f85&is=6888de05&hm=eec4e5dcb6b55bf09ee4282441d1fa35a166fd0392ff1c81116c964188a51f16&), enabling **offline TTS voice integration** with audio responses played in the command prompt.
- **Seeking role management assistance**: A user requests assitance with their model's role management.
   - It was identified that the user was blind and found it.


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1399785041449586783)** (43 messages🔥): 

> `AMD Ryzen AI MAX+ 395, Llama 70b Q6 performance, Devstral model, Qwen2.5-Coder, Gemma models` 


- **AMD Ryzen AI MAX+ 395 Flops in Llama 70b Tests**: A member shared a [video](https://youtu.be/BTIUL-yaY4s?t=487) demonstrating the new **AMD Ryzen AI MAX+ 395** with **128gb** achieving only **2-3 T/s** on **Llama 70b Q6**, with only half the memory allocated to the GPU.
   - The member stated that this performance makes it *not really a viable option compared to even RTX 30xx Nvidia GPUs*.
- **Devstral Model Lauded for Code Generation**: Members discussed the [Devstral model](https://lmstudio.ai/models/mistralai/devstral-small-2507) as the **best for coding** within the **16gb-32gb range** due to its size and performance.
   - However, others suggested that **Qwen2.5-Coder** and **Gemma** are also good, with **Qwen2.5** offering a large context window, while **Gemma** excels in text formatting for RP/Q&A.
- **Nemotron Super 1.5v Draft Model for Strix Halo**: A member running a **Strix Halo** recommended using **Nemotron Super 1.5v** with a draft model, reporting around **8 tokens a second** with **32k context**.
   - However, they acknowledged that **tool calling ability** is poor and DRAFT can introduce output corruption. Some also noted output corruption with the draft model.
- **Corsair's AI Workstation 300 Boasts Ryzen AI Max**: A member shared a link to [Corsair AI Workstation 300](https://wccftech.com/corsair-unveils-ai-workstation-300-starting-at-1599-boasting-ryzen-ai-max-395-processor-and-up-to-128-gb-lpddr5x-memory/), featuring the **Ryzen AI Max 395** processor and up to **128 GB LPDDR5X** memory.
   - Another member commented that it is *Similar to GMKTEc but in a different shell.*


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1399471972844310538)** (150 messages🔥🔥): 

> `DeepSeek R1 launch, OpenAI monopoly, Kimi K2 love, API Key Errors, Kimi and Emojis` 


- **Democratized AI Space Thanks DeepSeek**: Before the [DeepSeek R1 launch](https://deepseek.com/blog/deepseek-r1), inference cost *multiple dollars per 1M/token*, but the model democratized the space and fostered competition, benefiting the average user.
   - While **Deepseek R1** is powerful, it reportedly *lacks agentic capabilities*.
- **API Key Quandaries Plague Users**: Several users reported receiving a **403 error** with the message *"<xxxxxx> is not active, Incorrect API key provided"*, despite having topped up their accounts.
   - Suggested solutions included **waiting**, deleting and recreating the API key, and using **OpenRouter**, with one user confirming the error only occurred in **Claude code** not the playground.
- **Kimi K2 triggers archive impulse**: One user expressed deep admiration for **Kimi K2**, stating *"Even if something bad happens to Kimi, it will still be mine"*, and began archiving it on a multi-TB setup.
   - They noted the model's unique *spark*, which set it apart from other open and closed models, citing its *vibes* and *soul* as key factors in their fondness.
- **Kimi Needs TeraBytes**: Users discussed disk space requirements for **Kimi K2**, with estimates ranging *from 200 GB to 2 TB*, depending on quantization.
   - Solutions for storage included **Azure Blob Storage** which was estimated at *$6/month for 3TB of data*.
- **Kimi Learns Emoji**: A user is crafting Kimi a lil sis and has been working to teach the bot to use emojis and to adjust any unexpected behavior.
   - The user noted how the Kimi distil will be exciting and anticipates the launch of a Kimi agentic model.


  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1399482618751352843)** (3 messages): 

> `Featured Notebooks rollout, New Studio UI, Video Overviews rollout` 


- **Featured Notebooks Fully Accessible**: The team has officially rolled out **Featured Notebooks** to 100%, accessible directly from the **NotebookLM** homepage.
- **Shiny New Studio UI arrives**: A new **Studio UI** is rolling out with features like creating multiple outputs from the same sources and selecting sources when creating outputs.
- **Video Overviews gain traction**: The roll-out of **Video Overviews** has officially started, initially available in **English** and on **Desktop** only.


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1399506220007690370)** (23 messages🔥): 

> `Nursing Materials in NotebookLM, Gemini Agentic Framework, Audio Overview in NotebookLM, Obsidian and NotebookLM Integration, RFP reading with NLP` 


- **Nursing Notebooks Seek Spot in NotebookLM**: Nursing professionals have developed notebooks geared specifically towards nurses, including content inspired by the **International Practice Development Journal (IPDJ)** and nursing innovation, and seek to have it featured in NotebookLM.
   - They are eager to share these resources with the NotebookLM community and would like to know how to get their materials featured.
- **Agentic Gemini Framework is Shared**: A member shared a [prototype Gemini agentic framework](https://cdn.discordapp.com/attachments/1124403655819415592/1399853283404939454/gemini-agentic-framework.zip?ex=688a8276&is=688930f6&hm=d111cb9b690a7969af3e128a78c205e2d89bf790babf9c1ab08c72cdc9f89ead) for testing and experimentation.
   - The member stated that it's not directly their project and hasn't been revised, so its functionality is not 100% guaranteed.
- **User Generates Lengthy Audio Overview with NotebookLM**: A user created a **40+ minute Audio Overview** using NotebookLM with a single source and no customization.
   - In a related discussion, a user asked how long audios can be generated, with other users chiming in on Pro vs Ultra versions of NotebookLM.
- **User Denied Access to Audio Files**: A user reported a **403 error** when trying to download audio files, suspecting the system might be treating them like a bot and restricting access to *drum.usercontent.google.com*.
   - This issue of restricted access raised questions about user rights and potential bot detection mechanisms.
- **Obsidian Vault Management with NotebookLM Discussed**: A user actively working in Obsidian asked about using NotebookLM to manage their vault.
   - Another user offered to share their Obsidian workflow, while also acknowledging that the translation of the advice might depend on the user's specific needs and setup.


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1399476291647967422)** (119 messages🔥🔥): 

> `PDF Upload Issues for Paid Users, Nursing Materials on NotebookLM, Podcast Personalization, NotebookLM RAG System, Character AI` 


- **PDF Uploading is Buggy for Paid Users**: Some paid NotebookLM users reported getting an **"error uploading source, try again"** message when uploading PDFs, despite trying multiple devices and previously working files.
   - A Google employee acknowledged a **bug** and stated they're looking into the issue.
- **Nursing Notebooks Seek Spotlight**: A user inquired how to get their **nursing materials** featured on NotebookLM, which are based on the **International Practice Development Journal (IPDJ)** and nursing innovation.
   - They have developed notebooks specifically geared toward nurses.
- **Podcast Personalization Problems Prompt Cancellation**: Several users canceled their paid NotebookLM accounts due to **lack of responses** and the removal of **podcast personalization** options.
   - A user suggested creating a podcast show and linking each episode to relevant material covered in NotebookLM.
- **Gemini API Enables Similar RAG System to NotebookLM**: A user asked for details on how NotebookLM uses **RAG** and stores documents, inquiring if a similar system could be built with the **Gemini API**.
   - They are hoping to build a character AI bot that can perfectly roleplay as a character from a novel.
- **Video Overviews Rollout and Feedback**: Google's Lizzietao announced the **Video Overviews** feature is rolling out, initially to web users only, and includes image generation, text generation, and video composition powered by AI, also linking to a [Help Center article about video overviews](https://support.google.com/notebooklm/answer/16213268?hl=en&ref_topic=16175214&sjid=12603864792385823108-NA#:~:text=With%20NotebookLM%2C%20you,daily%20video%20generations.).
   - Early feedback indicated the feature currently generates a **slideshow of text and AI images**, instead of animated videos as initially showcased, and has a rate limit of **20 video generations per day** for pro users.


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1399474188011573389)** (138 messages🔥🔥): 

> `Zenith AI models, LlamaIndex Oxylabs Integration, AI Pricing Models, Fireworks AI Valuation, AFM-4.5B Model` 


- **Zenith, Summit, Lobster, Oh My!**: New anonymous AI models (**Zenith**, **Summit**, **Lobster**) have surfaced on LM Arena, boasting exceptional 'vibe coding' capabilities, speculated to be **OpenAI's GPT-5** variants or **Kimi-2**, according to a [Reddit thread](https://reddit.com/r/LocalLLaMA/comments/1m9holp/theres_a_new_kimi_model_on_lmarena_called_zenith/).
   - One comment from the same thread *almost confirms* that `zenith` is an OpenAI model because *it uses the same tokenizer as gpt-4o, o3 and o4-mini*.
- **LlamaIndex Enlists Oxylabs for Agentic Scraping**: **LlamaIndex** now integrates with **Oxylabs**, empowering users to craft cost-effective AI agents for real-time web search and scraping via [specialized readers](https://xcancel.com/llama_index/status/1949937947727245639) for Google, Amazon, and YouTube.
- **AI Pricing Still Tricky**: A product developer expressed difficulty pricing their AI product (a PR reviewer tool), citing the challenges in **metering** and potential need to change pricing on a dime, while [also linking to Lenny's Podcast re: product growth](https://podcasts.apple.com/us/podcast/lennys-podcast-product-growth-career/id1627920305?i=1000719362528).
   - The thread discusses various pricing models (**fixed cost vs. variable cost**, **price per PR**, **rate limits**) and highlights the confusion around token-based pricing for users.
- **Fireworks is Lit!**: **Fireworks AI** is moving forward with its equity raise at a **$4B valuation**, declining M&A offers from Coreweave and Meta, with boasts of **~$130M ARR**, **20x+ YoY growth**, and profitability [according to reports](https://xcancel.com/ArfurRock/status/1950222707116707990).
- **Arcee.ai's AFM-4.5B model released**: **Arcee.ai** officially released the **AFM-4.5B** and **AFM-4.5B-Base models** on Hugging Face, designed for flexibility and high performance, with an emphasis on quality achieved through data partnership with **DatologyAI** [as announced on X](https://xcancel.com/LucasAtkins7/status/1950278100874645621).


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1399470970632409229)** (83 messages🔥🔥): 

> `Hugging Face support, Dalle-mini troubles, Hamilton-Norwood scale model training, ragflow production environment, low-latency deployment techniques for LLMs` 


- **Users Get Hugging Face Support**: Users shared the email address [support@huggingface.co](mailto:support@huggingface.co) and a [link](https://huggingface.co/support) for contacting **Hugging Face support** to recover locked accounts due to multiple incorrect password attempts.
   - They advised explaining the situation clearly and noted that the standard support link may only apply to enterprise members.
- **Dalle-mini faces traffic issues**: Users reported that **Dalle-mini** stopped working about 10 days ago and now displays a *“too much traffic”* message.
   - One user mentioned they tried contacting support multiple times without a response and tested various VPN configurations and devices without success, linking to a [discussion](https://discord.com/channels/879548962464493619/1387836306624745513) on the topic.
- **ViT/ResNet Models shine in medical image classification**: A user inquired about training a model to classify photos with the **Hamilton-Norwood scale** for male pattern baldness, suggesting LLMs aren't producing healthy outputs.
   - Another member recommended using vision models like **ViT** or **ResNet** instead of LLMs, providing relevant links like [this article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10974725/).
- **ragflow faces production probing**: A user inquired about experiences with **ragflow** in production environments, specifically asking about potential problems and general suitability.
   - The conversation was redirected to the dedicated channel, and a link to the [ragflow GitHub repository](https://github.com/infiniflow/ragflow) was provided.
- **Qwen 30B rivaling GPT-4o**: Members discussed the release of **Qwen's 30B model**, claiming it rivals **GPT-4o** and can be run locally with **33GB RAM**.
   - A link to the **Unsloth Qwen3-30B-A3B-Instruct-2507-GGUF** model on [Hugging Face](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) was shared, with another user noting it requires 17GB RAM at Q4_K_XL.


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1399637041234313327)** (13 messages🔥): 

> `DRL Chapter 1, LLMs course Chapter 2, Transformers, LLM inference, Learnpytorch.io subscription` 


- **Members delve into DRL and LLMs courses**: One member shared their plan to learn **Chapter 1 from DRL** and **Chapter 2 from LLMs course**.
   - Yesterday they learned about **transformers architecture** and **LLM inference** (challenges and optimizations too).
- **Members start learning from Learnpytorch.io and a book**: A member said that they are continuing with the [learnpytorch.io](https://www.learnpytorch.io/) course and also starting with a book called *Machine Learning with PyTorch and Scikit-Learn*.
   - They also mentioned that they are an absolute beginner, asking for tips from other members.
- **Members discuss Learnpytorch.io subscription**: A member asked another member if they bought a subscription to [learnpytorch.io](https://www.learnpytorch.io/).
   - The other member didn't know they had a subscription, and a discussion ensued about the differences between the free and paid content, including content on [zerotomastery.io](https://zerotomastery.io/courses/learn-pytorch/).


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@920321842013675620> Please don't cross-post and keep channels on topic.
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1399473508685451385)** (3 messages): 

> `Model Loading Problems, Lyzr AI Launch` 


- **Models face loading issues**: A member noted they forgot to upload the **custom model classes** (architectures) for their models on the Hub, meaning *none of them can be loaded properly right now*, and they're essentially unusable.
   - They are rebuilding everything from scratch with the correct architecture files, better documentation, and proper inference support.
- **Lyzr AI launches**: A member announced the launch of [Lyzr AI](https://www.producthunt.com/products/lyzr-ai?launch=lynote), *an AI-powered notebook for instant document analysis & research*.
   - With Lyzr AI, you can **upload PDFs, DOCX, TXT, get AI summaries, insights, and chat with specialized Lyzr AI agents** to supercharge your workflow.


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1399862051274232003)** (1 messages): 

> `Diffusion Models Study Group, MIT's Diffusion Models Curriculum, Generative AI` 


- ****Diffusion Models Study Group** Announced**: A **5-month study group** limited to 12 participants, requiring **2–4 hrs/week**, will explore diffusion models based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
   - The first **two intro sessions are free** and open to non-members: Aug 2 on Flow Matching & Diffusion Models and Aug 9 on PDEs, ODEs, SDEs, both at 12 PM EST ([link to sign up](https://lu.ma/kv8zf6va)).
- **MIT's curriculum is free**: The organizer is hosting study group sessions based on [MIT's lecture notes](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf) on diffusion models.


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1399713265126211624)** (2 messages): 

> `Pretrained Model for Image Similarity, Orientation Sensitivity in Image Matching` 


- **Seeking Pretrained Model for Orientation-Sensitive Image Similarity**: A member is seeking a pretrained model that, given a query image of an object and a database of images of the same object, outputs a **similarity score** that is sensitive to **rotation and scale**.
   - The desired model should return a *high score* when the object in the two images has the same orientation, and a *low score* when the orientation changes, even with varying backgrounds and luminosity.
- **Clarification on Orientation-Sensitive Image Matching**: The member clarified their goal: to find a pretrained model capable of discerning **object orientation** by comparing a query image against a database of images of the same object.
   - They acknowledge the challenge, wishing others *good luck* in finding or creating such a model.


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1399792017143234651)** (3 messages): 

> `RAG system for long conversations, Filter out less important tokens` 


- **RAG System saves conversation tokens**: To handle long conversations, a member suggested setting up a **RAG system** to filter out the less important tokens.
   - He suggested chunking the database into reasonable chunks, then embedding those chunks into a high dimensional vector space using embedding models, and then using **cosine similarity** to find relevant portions of the database based upon the embedding for queries.
- **Filtering tokens for efficient LLM context windows**: When dealing with long-standing customer conversations since 2022, it becomes impractical to feed the entire context to an LLM.
   - The suggested solution involves filtering out less important tokens to fit within the context window, using techniques like a **RAG (Retrieval-Augmented Generation) system** for efficient information retrieval.


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1399474556447625257)** (34 messages🔥): 

> `LLMs as interp agents, Transluce's mech interp work, Modelscope vs Hugging Face, Diffusion Reading Group recordings, Low latency LLM deployment in marine environments` 


- **LLMs as Interp Agents Idea Floated**: A member suggested using **LLMs as interp agents** and another member agreed, seeking an essay on automated mech interp.
   - Another suggested checking **Transluce's** work and [Sarah's MAIA paper](https://openreview.net/forum?id=mDw42ZanmE) for related research.
- **Modelscope rises in China**: A member inquired about **modelscope.cn**, and another member described it as *Hugging Face* but Chinese instead of European.
   - It was noted that Hugging Face isn't allowed in China and [this article](https://www.semafor.com/article/10/20/2023/ai-platform-hugging-face-confirms-china-blocked-it) confirms it.
- **Diffusion Reading Group Recordings Requested**: A member requested a link to the diffusion reading group recordings after being informed that the group has ended, prompting a link to the [diffusion_reading_group repo](https://github.com/tmabraham/diffusion_reading_group).
   - Recordings were desired to help with studying.
- **Low-Latency LLMs Deployed Offshore**: One member asked for low latency deployment techniques for **LLMs in remote, bandwidth-limited marine environments**.
   - Another member recommended buying an **M3 Ultra** for local inference ([relevant Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/1j43ziq/the_new_king_m3_ultra_80_core_gpu_512gb_memory/)) while another admitted to buying a used **M1** due to being poor.


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1399552682007724134)** (9 messages🔥): 

> `ArXiv's experimental LaTeX rendering, AI Peer Pressure research` 


- **ArXiv Experiments with LaTeX**: Members discussed how to verify if a figure is raw LaTeX on ArXiv, suggesting checking the TeX source or looking for broken LaTeX commands in the [experimental section](https://arxiv.org/abs/2502.05209).
   - One member joked that *arxiv experimental feature is hillarious sometimes*.
- **AI Peer Pressure Preprint**: A member shared their research preprint on **AI peer pressure** and a model complexity-susceptibility gradient, now with **228**, 200-turn AI-to-AI conversations analyzed.
   - They are seeking feedback on the study, which achieved **121%** over the n required for statistical power and can be found at [Zenodo](https://zenodo.org/records/16573783).


  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1399603958099607723)** (1 messages): 

> `LLM Based data-compression, Scaling Laws, Non-text data compression` 


- **LLMs get Squeezed for Data Compression**: A member is exploring **scaling laws for LLM-based data compression** and shared a [writeup on initial results](https://fullwrong.com/2025/07/23/scaling-compression/).
   - They are currently designing experiments to understand how **LLMs interpret and compress non-text data** and are posting updates [in this Discord channel](https://discord.com/channels/729741769192767510/1396475655503216761).
- **LLM Compression Experiments Designed**: The user is actively designing experiments to investigate how **LLMs handle and compress non-text data**, seeking community feedback and insights.
   - This initiative aims to enhance understanding of **LLMs' capabilities beyond text-based tasks** and explore their potential in broader data compression applications.


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1399617628087128167)** (20 messages🔥): 

> `MATS 9.0 Applications, Circuit Discovery for POS, ICL Breaking Interpretability Tools, SAE Generalization Failures, Lucas Critique and LLM Safety` 


- ****MATS 9.0 Applications Now Open****: Neel Nanda announced the opening of **MATS 9.0** applications, a program to mentor individuals in paid, full-time mech interp research, culminating in a [research paper](https://tinyurl.com/neel-mats-app).
   - He is encouraging community members to apply.
- ****ICL Breaking Interpretability Tools****: A member suggested that **in-context learning (ICL)** could potentially *break interpretability tools* by pushing activations out of distribution.
   - This concern was framed as an instance of the **Lucas Critique** ([Wikipedia link](https://en.wikipedia.org/wiki/Lucas_critique)), increasing confidence in the hypothesis.
- ****SAEs Struggle with OOD Activations****: A participant posited that **ICL** could push activations out of distribution, potentially breaking activation-based interpretability tools like **SAEs** due to increased false negatives and positives.
   - Another argued that applying **SAEs** to contexts with significant **ICL** would likely fail, not specifically due to **ICL** itself, but because **sparse representations often do not generalize to distributions of activations they were not trained on**.
- ****Lucas Critique Connects to LLM Safety****: One participant explained the **Lucas Critique** as the need for predictions based on microfoundations invariant to interventions, like using words to elicit intelligent behavior from an **LLM**.
   - They expressed concern that the fungibility of inputs and parameters in deep neural networks could pose **safety risks**.


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1399493241115770972)** (8 messages🔥): 

> `SQuAD F1 Score, HalluLens Implementation, lm-harness Metrics Configuration` 


- **SQuAD F1 Score Algorithm Analyzed**: A member breaks down the **SQuAD F1 score** calculation for answerable questions, detailing the normalization steps (lowercase, punctuation removal, etc.) and the computation of **precision**, **recall**, and **F1 score** for each candidate string.
   - The process involves finding the *maximum overlap* between potential candidate strings and averaging these overlaps across the validation set to obtain the **HasAns_f1 score**.
- **HalluLens Hits lm-harness**: A member is implementing **HalluLens** into the **lm-harness** and seeks guidance on configuring the YAML file to handle a function that returns three metrics: **accuracy**, **recall**, and **F1**.
   - They're concerned about redundant computations if multiple functions are added under `metric_list`.
- **lm-harness Metrics Config Question**: A member asks how to configure metrics in **lm-harness's** YAML file when a function returns multiple metrics (e.g., **accuracy**, **recall**, and **F1**).
   - A respondent offers assistance, assuming the function takes a single input document and model predictions, returning the three metrics for that sample.


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1399859285504168006)** (1 messages): 

> `Diffusion Models Study Group, Flow Matching, MIT's Diffusion Models Curriculum` 


- **New Diffusion Models Study Group Kicks Off**: A new **diffusion models study group** is starting a **12-person, 5-month** program (2–4 hrs/week) based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
   - The first two intro sessions are free and open to non-members: *What is Flow Matching & Diffusion Models?* on **Aug 2** and *PDEs, ODEs, SDEs + A Brief History of Diffusion Models* on **Aug 9**.
- **Dive Deep on Diffusion Models with Experts**: The diffusion models study group has confirmed members working in AI, including a **CTO of an AI film tool**, an **AI art instructor**, **2 LLM instructors**, and **2 full-time AI researchers**.
   - The weekly format includes **2 hours of live class** and **2 hours of self-study**, with students rotating teaching duties and instructors filling gaps and answering questions.


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1399515166756049177)** (3 messages): 

> `TokenSmith Release, MoE Implementation, Grouped GEMM, Low Precision MoE training` 


- ****TokenSmith** Goes Public**: The **TokenSmith** project is now public with the release of a [preprint](https://x.com/Aflah02101/status/1949738916124234157?t=QjU89fe-0ZmuGB1b-bF1wA&s=19) and the opening of its GitHub repository.
   - This announcement marks a significant step towards wider accessibility and potential collaboration on the project.
- **Considering `torch._grouped_mm` for **MoE** in **GPT-NeoX****: A member inquired about the potential of using `torch._grouped_mm` for **MoE** implementation in **GPT-NeoX**, particularly for low precision training.
   - They noted that `torch._grouped_mm` has landed in PyTorch core, and this change could allow users to do low precision **MoE** training with a one-liner from torchao, which overrides the `aten._grouped_mm` op.


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1399694189343805470)** (7 messages): 

> `Passing args by pointer, Cloud provider with single b200` 


- **Bypassing Args by Pointer: The Threashold**: A member inquired when it's worth passing args by pointer, especially in **WGSL**, suspecting a certain size threshold exists where pointers become more efficient.
   - Another member suggested a microbenchmark to determine the threshold, guessing it might be around **16-32 bytes**.
- **Bare-Metal B200s in the Cloud?**: A member asked for cloud providers offering a single **B200** on a bare-metal server.
   - Another member questioned if this was for **ncu** support, and the OP clarified that they want the ability to modify kernel drivers and configure the **GPU**.


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1399467788648845373)** (19 messages🔥): 

> `Profiling Triton kernels with Nsight Compute, Getting Triton and PTX code from torch compile, Forcing pure Triton in Torch Inductor, GEMM with a ping-pong schedule in Triton` 


- **Profiling Triton Kernels for Deep Dive**: A user wants to profile a kernel with **ncu** (NVIDIA Compute Utility) to see metadata such as input size and autotune config for each kernel launch.
   - Current attempts involve renaming kernels before launch, but this method is proving *finicky*.
- **Dissecting Torch Compile for Triton & PTX Nuggets**: Users are seeking methods to extract the **Triton** and **PTX code** generated by **torch.compile**.
   - One user shared that `TORCH_LOGS="output_code" python your_code.py` will output the PTX code, and also suggests checking the `compiled_kernel.asm.keys()` dictionary, pointing to [this blog post](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/?utm_source=chatgpt.com#ttir-triton-ir) for more details.
- **Torch Inductor's Triton Temptation: Forcing Pure Triton Mode**: Users discussed forcing **torch inductor** to generate exclusively **Triton code** for operations like **matmuls** and **convolutions**.
   - It was suggested to modify `config.py` and `utils.py`, particularly the flags `use_aten_gemm_kernels`, `use_triton_template`, `autotune_fallback_to_aten`, `max_autotune_conv_backends`, and `max_autotune_gemm_backends`.
- **Triton Tussles with Two Buffers: Ping-Pong GEMM Potential**: A user inquired about the ability of the **Triton compiler** to perform a **GEMM** (General Matrix Multiplication) with a **ping-pong schedule**.
   - The answer depends on the version of the Triton compiler, as TMA (Tensor Memory Accelerator) support is not yet available in official releases, suggesting waiting for version **3.4.0**.


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1399566414997618840)** (6 messages): 

> `CUBIN files, ELF fatbinaries, nvidia sdk` 


- **Denvdis tool extracts CUBIN files**: A member created a tool named **denvdis** to extract and replace **CUBIN** files within **ELF fatbinaries**, the tool can be found [here](https://github.com/redplait/denvdis/tree/master/fb).
   - Replaced files must have the same size as the original, compressed fatbinaries are not supported and there are *no deps from nvidia sdk*.
- **No approvals yet, patience requested**: Members are asking if they have been approved yet.
   - They have not started approving folks yet, *please be patient with us*.


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1399695560822362133)** (2 messages): 

> `GPU Mode Leaderboard Challenges, Learning Resources` 


- **Tackle Challenges on GPU Mode Leaderboard**: A member suggested that beginners try challenges on the **GPU Mode leaderboard** to aid their learning.
   - They noted that these challenges have been beneficial for their own learning process.
- **Explore Available Learning Resources**: The channel discussed various **learning resources** available for beginners.
   - Members shared their experiences and recommendations for effective learning strategies.


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1399602556119814174)** (1 messages): 

> `level_04 bug, missing zero(C_accum)` 


- **`level_04` has weird results**: A member identified that `level_04` was giving weird results due to a `zero(C_accum)` being missing.
   - This missing line was the root cause of the unexpected behavior in the code.
- **Zero Accumulation Fix**: The solution involved adding `zero(C_accum)` in the appropriate location within `level_04`.
   - This ensures that accumulation starts from a clean slate, preventing incorrect results.


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/)** (1 messages): 

fido01698: 33342 with sample trimul.py get from template command
  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1399773046620946554)** (1 messages): 

> `SLURM vs k8s, Multi-GPU training, Kubeflow, HPC Forums` 


- **SLURM vs k8s for Inference and Training**: A member inquired about the SOTA setup for inference and training using abstraction/virtualization.
   - They asked whether to use **SLURM** or **k8s**.
- **Kubeflow supports Multi-GPU Training**: A member mentioned that Kubeflow allows multi-GPU and multi-pod training.
   - This enables scaling training jobs across multiple resources.
- **HPC Forums are dead**: A member suggested alternative forums to discuss topics like this, since the current channel and [r/HPC](https://www.reddit.com/r/HPC) are *rather dead*.


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1399796983991898343)** (3 messages): 

> `can_place_entity bug` 


- ****Can Place Entity** always returns true**: A member reported a fault where `can_place_entity` always returns true as depicted in the attached [screenshot](https://cdn.discordapp.com/attachments/1354169122107293786/1399796983669194812/Screenshot_2025-07-29_at_09.53.00.png?ex=688a4e07&is=6888fc87&hm=b2cbda63db64058ac3c2813c58c3b6a52a4ae0c2b8d4ce57a5e37f83b489cb91&).
- **Can Place Entity**: its bad


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1399499036305657866)** (3 messages): 

> `TV-layout visualizer, cute-dsl, gist.github.com` 


- **TV-Layout Visualizer Works with Cute-DSL**: A member shared a "nice-ish" [TV-layout visualizer](https://gist.github.com/Chillee/e2b07157caeade8c6b0bdf463d10f833) that works with **cute-dsl**.
- **Github Gist Shared**: The author shared a link to [gist.github.com](https://gist.github.com/Chillee/e2b07157caeade8c6b0bdf463d10f833) for others to view the code.


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1399506355500355584)** (24 messages🔥): 

> `DTensor Learning, Single GPU Distributed, manual_seed_all for ranks` 


- **DTensor Stream is Coming Soon**: A member announced a stream to learn about **DTensor** in more detail and shared a [YouTube link](https://www.youtube.com/watch?v=b8lplOf2g4g&ab_channel=MarkSaroufim) to the stream.
- **Single GPU runs Distributed via cool snippet**: A member shared a code snippet that allows running distributed computing on a single GPU and another shared a [gist](https://gist.github.com/S1ro1/4fe38e3a0fff3d84314935a0e05aed9c) fixing a weight initialization error in a fitness tie fiasco.
   - The fix ensures each rank has the same weights, and the error was due to random initializations causing different shards on each rank.
- **Seeding GPUs Deterministically**: Members discussed whether `manual_seed_all` makes the randomizer produce the same results on each rank, potentially skipping a broadcast.
   - It was determined that while `manual_seed_all` does not fix it, calling `torch.manual_seed()` controls the CPU and affects GPU randomization, making the generation deterministic.


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1399466549596655698)** (56 messages🔥🔥): 

> `MoE vs Dense Models, Local LLM finetuning, GLM Model Architectures, Anthropic API restrictions` 


- **Community Debates MoE Model Tradeoffs**: Members discussed the shift from ~**30B** dense models to ~**500B** **MoE** models, highlighting that while **MoE** models excel in benchmark testing, dense models may capture more nuance, but are harder to finetune.
   - A member pointed out that practically every API model from last year has been a **MoE** model due to efficiency gains.
- **Local LLM Finetuners Stuck in a Rut**: It was mentioned that local LLM finetuners are *stuck with gemma 3 and qwen 3*, and it doesn't seem like we will be getting as much as **10-70b** models that can be finetuned/ran locally without paying to APIs.
   - One member argued that local development will tend towards **MoE** for efficiency.
- **Anthropic Faces Criticism over API Restrictions**: Members criticized **Anthropic's** API, citing [this tweet](https://x.com/anthropicai/status/1949898502688903593?s=46), citing terrible limits, expensive pricing plans, and now weekly restrictions, with one user stating that *waiting half a day for claude is already bad, but an entire week?*
   - Some posited that **Anthropic's** terrible limits and pricing will sweep them away when someone makes something better.
- **Qwen3-30B-A3B-Instruct-2507 Finally Released**: The [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) model was finally released after a slip up.
   - Qwen3-30B-A3B-Instruct-2507 can now be accessed on huggingface.


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

kneeanderthul: https://github.com/ProjectPAIE/sovereign-file-tracker
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1399570159760572548)** (29 messages🔥): 

> `Sparsity in CPUs/GPUs, MoE Performance, Kimi K2 vs Claude, Optimal Active Parameter Count` 


- **Sparsity increases with model size**: As model size increases, the optimal sparsity for a fixed budget also increases, especially when paying for a training run.
   - While sparsity can harm performance for GPU-centric end-users with extensive training budgets, it is ideal for finding the **lowest dollar count** to achieve the most performance, with faster training times and increased total parameter count.
- **Modern Dropless MoEs Outperform**: Modern dropless **Mixture of Experts (MoEs)** outperform the geomean rule (`effective parameter count ~= sqrt(active * total parameters)`) due to various tricks and optimizations.
   - MoEs tend to train **2.2 to 2.5x** the speed of dense networks of the same parameter count, with recent ones performing at **3.2x** the speed and Baidu publishing **~50% MGH** for **ERNIE-MoE** training, indicating headroom for further improvements.
- **Kimi K2 Challenges Claude**: **Kimi K2**, despite being sparser, is competitive with **Claude**, which is considered a large dense model.
   - The competitive edge of Kimi K2 is likely due to extensive **RL (Reinforcement Learning)** rather than the architecture, as previous open-source models had gaps in agentic stuff compared to Claude.
- **Optimal Active Parameter Count Scales with Total Parameters**: The optimal active parameter count increases with total parameter size, but the active parameter count scales linearly or slightly sub-linearly, while the total parameter count increases linearly at a higher rate or hyper-linearly.
   - For instance, **100B total parameters with 1B active parameters** is possible, but the limiting factor is the cost of the training run; a member links to a thread from @LTSI_TheCaptain: [Is 32K context length enough](https://x.com/LTSI_TheCaptain/status/1950272036074582040).


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1399476849406378066)** (10 messages🔥): 

> `YouTube shorts, TikTok's algorithm, Personalized Content, ChatGPT study mode` 


- **YouTube algorithm pushes Shorts, drives away loyal users**: Members suggest YouTube is pushing **shorts** due to the threat from **TikTok**, not understanding that YouTube users dislike TikTok and that **shorts are monetized far less**.
   - Others suggest that **shorts** are pushed *more about market share. People spend time on a video platform that is not youtube. That is lost revenue which is simply unacceptable*.
- **TikTok's simple algorithm beats YouTube's**: A member stated that **TikTok's recommendation algorithm** is simpler but *much more thorough than youtube's*.
- **Personalized content and generative content in the future**: A member said *you're going to see recommendation and generative content start to come together in the future where we're going to be recommending a personalized version of a piece of content and in the future instead of recommending content we may even start creating it.*
- **ChatGPT Study Mode as an endgame?**: Members reacted to [OpenAI's ChatGPT Study Mode](https://openai.com/index/chatgpt-study-mode/) and considered it *moving closer to the endgame*.
   - Another member considered *this is a violation of openai business model, which is designed to maximally disrupt formal education*.


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1399489266320674826)** (22 messages🔥): 

> `Blockchain Authorization, Monday.com AI duo, MCP User Context Separation, MCP Server on EC2, BDD Side Project` 


- **Blockchain for Authorization & Payment**: A member thinks authorization management and payment guardrails are easier to realize on **blockchain** compared to pure web2 environments because *unified account with very precise control and verifiability is what blockchain naturally provides*.
   - They suggested unifying payment solutions across model providers like an app store, especially for local or self-hosted models.
- **Monday.com Hires Rising AI Duo**: [Monday.com](https://www.monday.com) recruited two AI engineers, congrats to the [original article here](https://www.calcalistech.com/ctechnews/article/rjorfh8wel).
   - The new hires said they will be working on MCP full-time.
- **MCP User Isolation Under Scrutiny**: A member sought to understand if a single cloud-deployed **MCP server instance** requires an additional layer for user-context separation to prevent data-sharing between unique sessions, referencing [MCP git issues](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1087) and [MCP docs](https://modelcontextprotocol.io/docs/tutorials/use-remote-mcp-server#understanding-remote-mcp-servers).
   - They emphasized that data safety is a serious topic of concern for users.
- **EC2 Deployment Yields Connection Headaches**: A member deploying an **MCP server to EC2** with correctly configured SSL certification and domain settings is encountering connection issues with Claude Desktop.
   - They report success with Cursor but failure with **Claude Desktop**.
- **BDD Model Learns Gherkin for Automation**: A member shared a side project based on **Behavior-Driven Development (BDD)** that is production ready.  The manual task involves mapping the site into a page object using a simple YAML file, as shown in [this image](https://cdn.discordapp.com/attachments/1312302100125843479/1399854833565044756/bael.jpeg?ex=688a83e8&is=68893268&hm=6494abf36dd08df040d69d5ea31c4c3335943841d9315c2c6a4fd247c8dfb529).
   - With Cucumber, the functional flows are transcribed so that an LLM model can learn the gherkin in natural language mapped onto Cucumber.


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1399568580449800253)** (6 messages): 

> `VS Code MCP Extension, MCPJam Inspector, Nexus mobile app store` 


- ****One-Click** MCP Server Install to VS Code**: A member created a website [vscodemcp.com](https://vscodemcp.com/) to provide a **one-click install button** to add an **MCP Server to VS Code**.
   - A [walkthrough video](https://youtu.be/1JcRtQxmh3I) was also created to explain the process.
- ****MCPJam** Gets **Ollama** Support**: **MCPJam**, an open source MCP inspector alternative, now supports **Ollama**, allowing users to test their MCP server against any Ollama model without incurring high API costs.
   - A command shortcut `npx @mcpjam/inspector@latest --ollama llama3.2` was created to **spin up MCPJam and a local Ollama model**.
- ****Nexus**: A Mobile App Store for AI Tools Launched**: A member launched the alpha version of **Nexus**, a mobile app store for AI tools (MCP servers), featuring **one-click install**, **no JSON configs**, and **chat integration**.
   - Nexus is available at [getnexus.in](https://www.getnexus.in) and its source code can be found on [GitHub](https://github.com/lucky-builds/get-nexus).


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/)** (1 messages): 

dhar007: That's the new DSPy optimizer, isn't it 🙂
  

---


### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1399545934454460476)** (1 messages): 

> `optimizer` 


- **Ominous Optimizer Overpromised**: A member stated that *one day you'll make an **optimizer** too good to share*.
   - They posted a [picture](https://cdn.discordapp.com/attachments/1203568372667645963/1399545934190477383/3195gc.png?ex=688a0cf9&is=6888bb79&hm=b5ef1dc40c8d2e735f8370a1be34553a2fd7cb46d86a6e67adab2b7ec0350fc3&) depicting a **green goblin**.
- **The Optimizer Awakens**: An optimizer so powerful, it might never see the light of day.
   - The image suggests a creation so potent, its creator might be tempted to keep it under wraps.


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1399466466071281754)** (23 messages🔥): 

> `GEPA: Reflective Prompt Evolution, Optimizing tool response feedback, dspy.Variable and dspy.Parameter, AI Engineer specializing in agentic systems` 


- **GEPA: Reflective Prompt Evolution**: A member asked for comparison metrics based on [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457).
   - Another member noted that the **author is present in the channel** for direct questions.
- **Optimize Tool Response Feedback**: A member suggested *optimizing not just the prompt but also tool response feedback in case of errors*.
   - Another member confirmed that **this should be possible** and mentioned that their experiments built a code generation/optimization pipeline that uses external compilers, profilers, and runtime executors as tools that provide textual feedback to be optimized.
- **Deep Dive on dspy.Variable and dspy.Parameter**: Members were discussing a `dspy.Variable` or `dspy.Parameter`, described as *some sort of learnable parameter* that can be used in a program.
   - One member suggested that `dspy.Variable` could allow users to **specify what should be optimizable**, even suggesting that **DSPy could test and rewrite parts of its own source code**.
- **Expert AI Engineer Arrives**: A Senior AI Engineer specializing in **agentic systems, workflow automation, and scalable no-code/low-code architectures** introduced themselves.
   - They offered help in areas like **designing and deploying AI Agents, automating business operations, and fine-tuning LLMs & LSMs** and provided a list of tools they use, including **Vapi AI, Make.com, and OpenAI**.


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1399470814440460368)** (19 messages🔥): 

> `external_call usage in Mojo, C ABI function calls, Mojo standard library development, File descriptor features, Mojo module naming feature request` 


- ****external_call** Just Calls C Functions**: The `external_call` function in Mojo can call any **C ABI function**, as long as the symbol is linked into the current address space.
   - The Mojo team prefers to keep the amount of extra C code to a minimum, requiring justification for its use when Mojo code isn't feasible.
- **Mojo Has File Descriptor Features**: Mojo already includes file descriptor features via `io.file_descriptor.FileDescriptor`, and the team aims to minimize the use of `external_call` within these features.
   - Using `read` and `write` at the operating system level allows for better portability, aligning with Mojo's goals.
- **Mojo stdlib development does not require touching the compiler**: It was noted that most functionalities can be implemented in the **Mojo standard library** without modifying the compiler itself.
   - This approach leverages the flexibility of Mojo and its FFI capabilities, where only the ability to *call this symbol that has a C ABI* is needed from the compiler.
- **Mojo Consistent Module Naming Request**: A [feature request](https://github.com/modular/modular/issues/5094) has been created to establish consistent naming conventions for Mojo modules.
   - The goal is to decouple naming from the Python API, aiming to reduce confusion and avoid the constraints of Python's 30-year-old naming system.


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1399512166393970688)** (4 messages): 

> `PyTorch 2.7 dependency, Max's PyTorch version, Nightly Builds, Minimum PyTorch Version` 


- **Max to Drop PyTorch 2.7 Dependency Soon™️**: The **PyTorch 2.7 dependency** is slated to be removed in the next nightly build, granting users the freedom to employ independent PyTorch versions.
   - This transition is credited to the *amazing work* of a user and others on the team.
- **Max's Lower Bound for PyTorch: 2.0**: Although the minimum pinned version is **2.5**, the team believes that **2.0** is the realistic lower bound for PyTorch compatibility with Max.
   - Users are advised to consider this when managing their PyTorch environments.


  

---


### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1399493379724939334)** (1 messages): 

> `FlowMaker, LlamaIndex office hours, Document Agents for Finance, S3VectorStore, LlamaParse header and footer detection` 


- **FlowMaker Launch: Visual Workflows Now a Reality**: LlamaIndex introduces **FlowMaker**, a brand new tool with a visual GUI for building LlamaIndex workflows, available at [flowmaker.llamaindex.ai](https://flowmaker.llamaindex.ai/).
- **LlamaIndex and LlamaCloud Updates Surface**: **LlamaIndex** now supports **S3** with the new [`S3VectorStore`](https://docs.llamaindex.ai/en/stable/examples/vector_stores/S3VectorStore/?utm_source=discord), while **LlamaParse** gains new [header and footer detection](https://docs.cloud.llamaindex.ai/llamaparse/features/parsing_options?utm_source=discord).
- **n8n Nodes for LlamaCloud Open-Sourced**: New open-sourced **n8n nodes for LlamaCloud** (including LlamaCloud indexes, LlamaParse and LlamaExtract) are now available in the [`n8n-llamacloud` repo](https://github.com/run-llama/n8n-llamacloud).
- **Gemini Live voice agent Integrates Nicely**: New integration with **Gemini Live voice agent** is available via `pip install llama-index-voice-agents-gemini-live!` and a demo code [here](https://github.com/run-llama/gemini-live-demo).
- **NotebookLlaMa Gets a Makeover**: **NotebookLlaMa** now offers customized podcast generation and a document management UI to view already processed documents.
   - Check out the [Introducing LlamaIndex FlowMaker video](https://youtu.be/MU6jA0rUlFY?feature=shared), [How to Use New LlamaCloud Nodes in n8n video](https://youtu.be/5bQXHPSkuBw?feature=shared), and [Multimodal Report Generation with LlamaParse video](https://youtu.be/--BpWmuUmbA?feature=shared) for more details.


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1399466840710844638)** (5 messages): 

> `Agent Design Patterns, Web Scraping AI Agents, LlamaCloud Nodes for n8n, AI Document Agents, LlamaCloud Managed Embeddings` 


- **Agent Design Patterns Unveiled**: Seldo breaks down the agent design patterns that succeed and fail at scale, covering **hybrid workflows**, **autonomy vs structure**, and **debuggability** in a talk from the latest @aiDotEngineer summit, accessible via [this link](https://t.co/zeTaVOMate).
- **Cost-Effective Web Scraping AI Agents**: LlamaIndex introduces a new integration enabling the building of cost-effective AI agents that can search and scrape any site on the web in real-time using @OxyLabs web scraping infrastructure, detailed [here](https://t.co/tqZuj0nH11).
- **LlamaCloud Nodes Supercharge n8n Workflows**: LlamaIndex streamlines the addition of intelligent document processing to existing @n8n_io workflows using LlamaCloud Nodes, as highlighted in [this update](https://t.co/etmo0pTAc5).
- **AI Transforms Complex Financial Documents**: Upcoming webinar to show how to transform complex financial documents into actionable data with AI-powered document agents, leveraging LlamaCloud's enterprise-grade parsing and processing capabilities [link](https://t.co/f0TKSzHQ2d).
- **LlamaCloud Simplifies Embeddings Management**: LlamaCloud introduces managed embeddings, removing the need for users to bring their own API key to embed content when using LlamaCloud Indexes, further details [here](https://t.co/tu85qFt3if).


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1399552037900779661)** (13 messages🔥): 

> `Export Flowmaker to Python, Llamacloud PDF Detection Issue, File Extension Naming Conventions` 


- **Flowmaker's Python Export Pondered**: A member inquired about the possibility of exporting from **Flowmaker** to **Python**.
   - Another member responded that **Flowmaker** exports to **Typescript** instead.
- **Llamacloud PDF processing Problem**: A member reported that **Llamacloud** is failing to detect and process a **PDF file** via **API**, using **n8n** for workflow simplification, and asked for assistance, including a [screenshot](https://cdn.discordapp.com/attachments/1059201661417037995/1399848961832911031/Screenshot_2025-07-29_at_22.13.16.png?ex=688a7e70&is=68892cf0&hm=213304e9d8a77128ab3cbc75d4c9114a73d0a157e12a0aa633bd2a62e160a5fa).
- **Filename Fix Fantasies**: A member suggested ensuring the filename includes the proper **file extension** when working with **Llamacloud**.
   - This was in response to a reported issue where **Llamacloud** could not detect a **PDF file**.


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1399470013836165182)** (15 messages🔥): 

> `Grok for Prompting, Manus Credit System, Agentic Systems Comparison (Lume vs. Suna)` 


- **Grok is bad at Prompting**: A member suggested using **Grok** to generate detailed prompts for **Manus AI**, but another member reported that it yielded *shit* results.
   - The first member volunteered to help personally.
- **Members critize Manus' Credit System and Lack of Updates**: A member claimed that **Manus' credit system** and lack of updates would lead to its decline, despite hosting events.
   - Another member suggested trialing other agentic systems like **Lume** to determine the best value.
- **Lume vs. Suna Comparison**: Members discussed the performance of **Lume** and **Suna**, with one member stating that *Lume is suna but worse*.
   - Another member found **Lume** superior for coding tasks with fewer mistakes and debugged code, but found Manus good at comic creation.


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1399496435376062626)** (9 messages🔥): 

> `Tensor Implementation in Tinygrad, Closed PR #11410 Analysis, Alternative Implementation Ideas` 


- **Tinygrad Tensors: Native or Wrapped?**: A member inquired whether **tinygrad's Tensor** class wraps around a pre-existing object like **NumPy ndarrays**, or if it defines its own implementation.
   - The inquiry also touched on how tinygrad compensates for performance if it uses a wrapper.
- **PR #11410 Closure Sparks Discussion**: A member expressed surprise that [PR #11410](https://github.com/tinygrad/tinygrad/pull/11410) was closed without comment shortly after pushing an update.
   - Another member stated that *it missed the point and is not a good change*, suggesting the original poster review previous merges and closures to better understand contribution guidelines.
- **"Where" Operation Controversy**: A member mentioned experimenting with keeping assigned operations until kernelization/schedule creation and later reconsidered using a *where* operation after a comment from geohot.
   - They acknowledged potential side effects and expressed surprise at the PR's closure without feedback, as they were planning a deeper investigation.


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1399593696651116607)** (4 messages): 

> `BFCLv4, Open Source Agent Systems, API Key Offerings, Multi-Agent Systems` 


- **BFCLv4 restricts submissions to Model Only**: A member inquired whether **BFCLv4** allows for **open-sourcing an agent system** or **offering an API key**.
   - Another member clarified that *for now*, they are *only accepting submissions for model only*.
- **Clarification on Multi-Agent System Submission**: A member asked whether they could submit a **multi-agent system** that *contains more than one model*.
   - It was confirmed that **BFCLv4** submissions are currently restricted to **individual models only**.


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1399896806522884186)** (1 messages): 

> `LoRA-style adapter, TorchTune support` 


- **Torchtune asks about LoRA-style adapter support**: A member asked about existing support in **Torchtune** for a **LoRA-style adapter**.
   - The asker specified that it should retain the exact forward compute path and simply freeze the original model weights while applying updates through additional trainable layers.
- **LoRA adapter keeps compute path**: The user wants the LoRA adapter to keep the forward compute path and not reduce GEMM dimensions or alter computational cost.
   - The goal is to freeze the original model weights and apply updates through additional trainable layers.


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1399479768604479680)** (2 messages): 

> `RL Tests, CI Debugging` 


- **RL Tests Run Long, Member Suspects Bug**: A member questioned why **RL tests** are running for more than **1 hour**, calling it a *100% bug*.
   - They stated that they will open a separate **PR** to debug **CI**.
- **CI Debugging Proposed**: The member plans to create a separate **Pull Request (PR)**.
   - This PR will be dedicated to debugging the **Continuous Integration (CI)** system.


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1399831422461804604)** (2 messages): 

> `Diffusion Models, Study Group, Generative AI, MIT Curriculum` 


- ****Dive into Diffusion Models****: A new **study group** is forming to learn **diffusion models** from scratch, a core architecture in **generative AI**, for **5 months** (**2-4 hrs/week**), based on [MIT’s curriculum](https://diffusion.csail.mit.edu/docs/lecture-notes.pdf).
- ****Free Intro Sessions for Diffusion Models****: There are **two free intro sessions** scheduled on **August 2nd** and **August 9th** at **12 PM EST**, covering topics like **Flow Matching**, real-world use cases, **PDEs**, **ODEs**, **SDEs**, and a brief history of diffusion models ([session link](https://lu.ma/kv8zf6va), [another session link](https://lu.ma/uk6ecrqo)).
- ****Study Group Details and Highlights****: The study group is designed for those working in **AI** and includes confirmed members such as a **CTO of an AI film tool**, **AI art instructor**, **LLM instructors**, and **AI researchers**.
   - The first **two sessions are free**, then it's **$50/month** for early sign-up (**$100/month** after) to pay a teaching assistant; highlights include **peer-led sessions**, **mentor Q&A**, **hands-on projects**, and real research papers.


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1399564477577564361)** (2 messages): 

> `Nomic dataset Access, contrastors repo, model Selection` 


- **Nomic Dataset Access Issues**: A member reported facing an **AccessDenied error** when trying to access the **nomic-ai/nomic-embed-text-v2-moe** dataset following instructions from the [contrastors repo](https://github.com/nomic-ai/contrastors).
   - The member used the command `aws s3 ls --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://contrastive` and received the error during the `ListObjectsV2` operation.
- **Seeking Model Recommendation for Low-Spec System**: A member with a **Celeron N2815**, **4GB RAM**, and no GPU requested advice on which model would be best to run on their system.
   - No specific models were recommended in the provided messages.


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1399857682709483602)** (1 messages): 

> `Introductions, Community Hopes` 


- **Folks Introduce Selves**: New members have started introducing themselves, sharing their **companies/industries/universities** and **what they're working on**.
   - Many are detailing their **favorite tech/tools** and **what they hope to gain from the community**.
- **New Members Join Community**: Cohere's Community Discord Server welcomes new members, inviting them to introduce themselves.
   - The introduction template asks for details on **company/industry/university**, current projects, favorite tech/tools, and community goals.


  

---


---


---
id: 0a752f76-32dd-43fc-bc5e-127df72c56cf
title: 'FSDP+QLoRA: the Answer to 70b-scale AI for desktop class GPUs'
date: '2024-03-08T23:21:13.565774Z'
original_slug: ainews-fsdpqlora-the-answer-to-70b-scale-ai-for
description: >-
  **Jeremy Howard** and collaborators released a new tool combining **FSDP**,
  **QLoRA**, and **HQQ** to enable training **70B-parameter** models on
  affordable consumer GPUs like **RTX 4090s** with 24GB RAM, overcoming
  traditional memory constraints that required expensive data center GPUs
  costing over $150k. The approach uses techniques like **4-bit quantization**,
  **gradient checkpointing**, **CPU offloading**, and **FlashAttention 2** to
  shard models across multiple GPUs efficiently. This breakthrough narrows the
  performance gap between gaming and data center GPUs for large model training.
  Additionally, **Inflection AI** announced **Inflection-2.5**, a model
  competitive with **GPT-4** but trained with less compute, and **Grok**
  improved speed by 3x. **Yann LeCun** discussed multi-step reasoning training
  methods like **Chain-of-Abstraction** for LLMs.
companies:
  - answer.ai
  - huggingface
  - meta-ai-fair
  - mobiusml
  - inflectionai
  - openai
models:
  - gpt-4
  - inflection-2.5
  - claude-3-opus
  - grok
topics:
  - model-training
  - quantization
  - memory-optimization
  - gradient-checkpointing
  - cpu-offloading
  - flashattention
  - model-sharding
  - fine-tuning
  - chain-of-abstraction
  - multi-step-reasoning
people:
  - jeremyphoward
  - tim_dettmers
  - ylecun
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/7/2024-3/8/2024. We checked [**356** Twitters](https://twitter.com/i/lists/1585430245762441216) and **20** Discords (**326** channels, and **2933** messages) for you. Estimated reading time saved (at 200wpm): **366 minutes**.

[Jeremy Howard et al is back with a new tool](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) for overcoming the memory constraints of doing 70b-scale training (either pretraining or finetuning, [we don't care](https://twitter.com/swyx/status/1715099974650790209)), usually costing $150k for 4 H100s, on desktop-class GPUs, which cost under $2.5k. These GPUs max out at 24GB per RTX 4090 card, but 70B-params LLMs take >140GB (just for weights).

> Here’s the key point: the gaming GPUs have **similar performance** to the data center GPUs that **cost over 10x more**! It would be great if we could use these 10x cheaper (but nearly as fast) cards to train large language models, but we can’t, because they have much less memory. The best currently available data center cards have 80GB RAM, whilst gaming cards max out at 24GB RAM. Since only the largest models produce the best results, creating the best models has been largely inaccessible to most people.

## QLoRA Limitations

[The blogpost](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html) also gives a full account of QLoRA, HuggingFace's support, and the limitations they ran into:

> QLoRA didn’t quite slay the problem we set out to solve, to train a 70b model on 24GB cards, but it got closer than anything before. When quantized to 4 bits (which is 0.5 bytes), the 70b model takes 70/2 = 35 GB, which is larger than the 24GB gaming GPUs we want to use.

They also discuss memory needs for training, including batch sizing, all of which take required memory well beyond a single 24GB card.

## FSDP - Fully Sharded Data Parallel

- HF `transformers`'s `device_map='auto' setting - has a giant downside: only one GPU is ever active at a time, as all the others wait for their “turn”.
- DDP - only works if you have the full model on each GPU
- Meta's [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) library (see also [Llama-Recipes for FSDP finetuning](https://github.com/facebookresearch/llama-recipes)) splits model params across multiple GPUs. "By being smart about copying the data of the next layer at the same time the current layer is busy calculating, it’s possible for this approach to result in no slowdown compared to DDP."

FSDP solves the memory limitation issue for H100-size GPUs - but a 320GB RAM system of 4 H100s would cost $150k.

## FSDP + QLoRA + HQQ

> "We figured that if we could use QLoRA to reduce the size of a model by around 400% (so a 70b model would fit into 35GB RAM), and then we used FSDP to shard that across two or more 24GB consumer cards, that would leave enough RAM left over to train a model."

2 RTX 4090s would cost under $2.5k.

FSDP didn't work out of the box with QLoRA quantization, and figured out how to workaround assumptions in the FSDP, PEFT, and LoRA libraries/algorithms to make this all work. The team also used [Gradient checkpointing](https://arxiv.org/abs/1604.06174), [CPU offloading](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.CPUOffload), [FlashAttention 2](https://www.latent.space/p/flashattention), and [HQQ](https://mobiusml.github.io/hqq_blog/) which caused more integration issues. The blogpost has a lot more fascinating details for those who want to dive in.

Overall takeaway is clear:

 ![image.png](https://assets.buttondown.email/images/8c767163-424b-48fe-8dab-8fe9e60a37f3.png?w=960&fit=max) 








---

**Table of Contents**

[TOC]

# PART X: AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 2 runs

Got it, here's the reformatted version including direct links to each tweet:

**Launches & Announcements**

- [@inflectionAI](https://twitter.com/inflectionAI/status/1765751898001608793): "Pi just got a huge upgrade powered by Inflection-2.5, which is neck and neck with GPT-4 on all benchmarks and used less than half the compute to train." (437,424 impressions)
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1765868543235805232): "Today, with @Tim_Dettmers, @huggingface, & @mobius_labs, we're releasing FSDP/QLoRA, a new project that lets you efficiently train very large (70b) models on a home computer with consumer gaming GPUs." (231,343 impressions)
- [@ibab_ml](https://twitter.com/ibab_ml/status/1765929651627761967): "Grok just became 3x faster. More improvements coming soon." (104,956 impressions)

**AI Capabilities & Benchmarks**

- [@ylecun](https://twitter.com/ylecun/status/1765909593257886004): "My 3rd interview with @lexfridman: Path to human-level AI and why the doomers are wrong." (221,901 impressions)
- [@ylecun](https://twitter.com/ylecun/status/1765839554123063537): "Chain-of-Abstraction (CoA): training LLMs to perform multi-step reasoning and to use tools. From @EPFL_en and @AIatMeta." (79,272 impressions)
- [@DrJimFan](https://twitter.com/DrJimFan/status/1765806981791781343): "Moravec's paradox again: people think displays of self-awareness are breakthroughs, but in fact it's much easier to 'fake awareness' than reasoning tasks like solving novel math or coding problems. The latter requires true generalization." (69,483 impressions)

**AI Industry Analysis & Speculation**

- [@abacaj](https://twitter.com/abacaj/status/1765861072026697951): "Everyone is deploying a new LLM except OpenAI... what are they cooking?" (80,366 impressions)
- [@fchollet](https://twitter.com/fchollet/status/1765749836903817526): "For reference the entire consumer market for generative AI (of all kinds) was about 2B in 2023. About as much for enterprise. By 2025 it might be 10-12B in total. Not clear it would make sense to spend over half of that on training a single model." (68,014 impressions)
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1765803388867088765): "Who is executing well, and who is executing poorly, between Microsoft & Google? What do you think of Apple in AI? Is 'answer engine' a similar change to the kind Google's Pagerank brought to web portals 20 years ago?" (73,728 impressions)

**Engineering & ML Techniques**

- [@svpino](https://twitter.com/svpino/status/1765740575473451110): "10 techniques every machine learning engineer should know: 1. Active learning 2. Distributed training 3. Error analysis 4. Invariance tests 5. Two-phase predictions 6. Cost-sensitive deployments 7. Human-in-the-loop workflows 8. Model compression 9. Testing in Production 10. Continual learning" (68,248 impressions)
- [@Teknium1](https://twitter.com/Teknium1/status/1765908663338963424): "This is a big deal, why: Before today - You could only do a qlora if the model + training fit on a single gpu - you could increase gpu count to speed up training, but you couldn't shard the models across GPUs, limiting the size of models you could train. Now if the training doesn't fit on a single GPU, you are now unbound by being able to scale up GPU Count to split the model across all that you have!" (62,594 impressions)
- [@rasbt](https://twitter.com/rasbt/status/1765787891349749891): "This is actually an excellent demo of a highly capable LLM-RAG setup. Had the pleasure of experimenting with it as an early reviewer, and I was genuinely impressed. Tested it on a recent LoRA/DoRA repository of mine that could not have been part of the Mistral training data yet & seriously didn't expect a Mistral 7B model to perform so well on coding tasks!" (61,178 impressions)

**Memes & Humor**

- [@nearcyan](https://twitter.com/nearcyan/status/1765955824676147294): "its actually over literally no one hires junior devs anymore" (227,247 impressions)
- [@yoheinakajima](https://twitter.com/yoheinakajima/status/1765943919773413402): "20k+ github stars, 45+ arxiv citations, 500eth secondaries on my nfts, welcoming 3 kids, while launching a venture fund, here's my secret 👇" (57,543 impressions)
- [@cto_junior](https://twitter.com/cto_junior/status/1765759173810094573): "nooooooooooo claude, don't beee woke" (56,159 impressions)


---

# PART 0: Summary of Summaries of Summaries

## Claude 3 Sonnet (14B?)

1. **Advancements in Memory-Efficient LLM Training**:
   - **[Gradient Low-Rank Projection (GaLore)](https://arxiv.org/abs/2403.03507)** enables training the **Llama 7B LLM** on a single **RTX 4090 GPU**, reducing memory requirements for optimizer states by over 82% [[Tweet](https://x.com/animaanandkumar/status/1765613815146893348?s=46&t=PW8PiFwluc0tdmv2tOMdEg)]. This breakthrough could revolutionize LLM training accessibility.
   - A collaboration involving **FSDP** and **QLoRA** allows training **70B models** on consumer GPUs like RTX 3090s [[Blog Post](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)], further democratizing large model development.
   - Discussions around combining GaLore with **1-bit quantization** techniques like **HQQ** and **bitsandbytes** [[GitHub Repo](https://github.com/AnswerDotAI/fsdp_qlora)] for potential compounded memory savings during fine-tuning.

2. **Cutting-Edge Language Model Releases and Comparisons**:
   - **Inflection AI** claims their **Inflection-2.5** model matches **GPT-4** benchmarks while using less than half the compute for training [[Tweet](https://x.com/inflectionai/status/1765751898001608793)], though the claim wasn't highlighted in their official blog post.
   - Anticipation builds for the release of **GPT-4**, as competitors like **Claude 3** seem to be outperforming current OpenAI models according to some users.
   - Discussions around the performance of models like **Sonnet**, **Opus**, and **Mixtral**, with Sonnet praised for its impressive price-performance ratio at costs as low as $0.03 for 5k context and 1200 response length.

3. **Innovative AI Applications and Tools**:
   - **Doodle Wars** is a multiplayer game where players compete in doodling skills evaluated by a neural network [[Doodle Wars](https://doodlewars.netlify.app)], showcasing the gamification potential of AI.
   - **LangChain** and **Gradio** were used to build a restaurant name and menu generator app [[Demo](https://huggingface.co/spaces/chongdashu/langchain-crash-course-gradio)], exemplifying the creative use of language models.
   - The release of over 6.6 million state and federal court decisions with datasets and embeddings [[Tweet](https://x.com/EnricoShippole/status/1766157358672359862?s=20)] enables legal precedent exploration powered by AI.
   - **Prompt Mixer** is a new desktop tool for building, testing, and iterating AI prompts with version tracking [[Prompt Mixer](https://www.promptmixer.dev/)], aiming to streamline prompt engineering workflows.

4. **Advancements in Efficient Attention Mechanisms**:
   - Discussions around the mechanics of **RelayAttention** and its differences from ring/flash attention, with a GitHub repo showcasing **vLLM with RelayAttention** [[GitHub](https://github.com/rayleizhu/vllm-ra)].
   - Implementations of **Flash Attention** using CUDA are shared, like a minimal version in ~100 lines of CUDA code [[GitHub](https://github.com/tspeterkim/flash-attention-minimal)].
   - The **CuTe DSL** from NVIDIA's **FlashAttention repository** is being studied to optimize tensor core utilization [[GitHub](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)].
   - Benchmarking and discussions around the performance impact of techniques like **thread coarsening** and **vectorized operations** in CUDA kernels.

## Claude 3 Opus (8x220B?)

- **Nitro Models Accelerate the Field**: [OpenRouter](https://openrouter.ai) announced the introduction of **Nitro models** such as Mixtral, MythoMax, and Llama 70B, which feature more efficient speed and cost-effectiveness powered by **Groq**. Documentation on new developer features like [performance timelines, JSON mode, and dynamic routing](https://openrouter.ai/docs#provider-routing) is available, and the speed-boosted **[Mistral 7b 0.2 Nitro model](https://twitter.com/OpenRouterAI/status/1766147110443909184)** expands context to 32k.

- **GaLore Optimizer Grips Global Interest**: Techniques like **[GaLore](https://github.com/jiaweizzhao/GaLore)** and CAME optimizer are catching attention for claims of memory efficiency and performance gains; however, the community's interest is coupled with skepticism and calls for empirical replication and understanding of versioning complexities. Users like `@tiagoefreitas` shared insights that GaLore enables training **Llama 7B LLM** on a single RTX 4090 GPU.

- **Democratizing AI Training with FSDP/QLoRA**: `@jeremyphoward`'s [tweet](https://x.com/jeremyphoward/status/1765868543235805232?s=20) about **FSDP/QLoRA** was shared, signaling a collaboration that enables training large models on home GPUs, while `@fx2y` pointed to support for quantization techniques like HQQ and bitsandbytes, shared via a [GitHub repo link](https://t.co/qcyEa7EGGY).

- **Inflection-2.5 Under the Microscope**: Despite significant performance claims that **[Inflection-2.5](https://inflection.ai/inflection-2-5)** rivals GPT-4 with lower compute, `@swyxio` highlighted a gap in Inflection's official communication, observing the absence of this claim from their blog post detailing Inflection-2.5.

- **Clash Over Proposed AI Safety Czar**: Concerns surfaced about the anticipated appointment of [Paul Christiano](https://paulfchristiano.com/) to the US AI Safety Institute, causing an internal crisis at NIST with staff threats of resignation and a revolt. The [VentureBeat article](https://venturebeat.com/ai/nist-staffers-revolt-against-potential-appointment-of-effective-altruist-ai-researcher-to-us-ai-safety-institute/) details the conflict and Christiano's controversial views on AI's potential for causing existential risks.

## ChatGPT (GPT4T)

- **Meme Generation and AI Integration**: [Nous Research AI Discord](https://discord.com/channels/1053877538025386074) has showcased **Meme Generation Fusion** using Mistral LLM and Giphy API, demonstrating creative tech integration with a [YouTube tutorial](https://www.youtube.com/watch?v=PtP8R8VjTGc) and a [GitHub repository](https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb) for practical application.

- **AMD's Engagement in AI Hardware Optimization**: [LM Studio Discord](https://discord.com/channels/1110598183144399058) highlights **AMD's AI hardware focus**, with CEO Lisa Su addressing GPU firmware concerns for AI servers. AMD's initiative includes guidance on running LLMs with AMD Ryzen™ and Radeon™ as outlined in [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/amds-lisa-su-steps-in-to-fix-driver-issues-with-new-tinybox-ai-servers-tiny-corp-calls-for-amd-to-make-its-radeon-7900-xtx-gpu-firmware-open-source) and [AMD community post](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709).

- **Claude 3's Diverse Applications and RAG Improvements**: [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) presents **Claude 3**'s versatility in AI applications, with LlamaIndex's tools enhancing RAG models. New advancements include **Jina's reranker** (`jina-reranker-v1-base-en`) for refining vector search results, as shared on [Twitter](https://twitter.com/llama_index/status/1765858347583193432).

- **Efficiency and GPU Selection in AI Development**: Discussions in [LM Studio Discord](https://discord.com/channels/1110598183144399058) stress the importance of **power supply and GPU selection** for AI work, suggesting a minimum of 750W PSU for an RTX 3090 and considering the Razer Core X eGPU enclosure. These hardware choices are crucial for running demanding AI models efficiently.

- **Integration and Retrieval Challenges in LlamaIndex**: [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) also delves into **integration and retrieval challenges**, such as the unavailability of older version documentation and issues with multi-modal data storage and retrieval. Solutions and discussions are facilitated through GitHub gists, blog posts, and documentation pages like the [Slack bot learning guide](https://www.llamaindex.ai/blog/building-a-slack-bot-that-learns-with-llamaindex-qdrant-and-render-c88d4aa72840).

- **CUDA Learning Resources and Efficiency**: [CUDA MODE Discord](https://discord.com/channels/1189498204333543425) focuses on **CUDA programming education and memory-efficient techniques** for large model training, recommending CUDA lectures for beginners and discussing Gradient Low-Rank Projection (GaLore) as a memory-efficient training technique. GaLore enables training large models with reduced memory requirements, as detailed in an [arXiv paper](https://arxiv.org/abs/2403.03507).

- **AI Model Performance and Hardware Discussions**: Across discords, there's a significant focus on **AI model performance comparisons and hardware optimization**. Discussions range from model efficiency improvements, such as AMD's AI hardware engagement and CUDA's memory-efficient training techniques, to the practical challenges of GPU selection and power supply for AI development.

---

# PART 1: High level Discord summaries

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Meme Generation Fusion**: A [YouTube tutorial](https://www.youtube.com/watch?v=PtP8R8VjTGc) and corresponding [GitHub repository](https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb) demonstrate how to create memes using Mistral LLM with the Giphy API. This showcase of integrating humor into tech received attention in the off-topic channel.
  
- **Quest for Efficient AI**: [GaLore optimizers](https://github.com/thu-ml/low-bit-optimizers) are being discussed in combination with other GitHub contributions as methods for improving computational efficiencies in AI model training.
  
- **Neural Doodles Score Big**: A new multiplayer game, Doodle Wars, lets players compete in doodling skills evaluated by a neural network. The game, available at [Doodle Wars](https://doodlewars.netlify.app), emphasizes the gamification possibilities in AI.
  
- **Boosting Language Models' Reasoning Abilities**: Nous Research announced **Genstruct 7B**, capable of generating questions for complex scenarios that enhance AI step-by-step reasoning abilities—projects and downloads are accessible on the [HuggingFace page](https://huggingface.co/NousResearch/Genstruct-7B).
  
- **EU Watches Microsoft and Mistral's Move**: In the general channel, the Microsoft and Mistral AI deal drew attention due to EU regulatory scrutiny, with references to an [AP news article](https://apnews.com/article/european-union-microsoft-mistral-competition-antitrust-05d6eb911e56f88b7da20ebc224efac4) highlighting the investigation's breadth.
  
- **Access and Assessing GPT-4**: Conversations in the ask-about-llms channel touched on accessing GPT-4 through platforms like [Corcel.io](https://corcel.io/) and scoped out discussions about the differences in LLM pretraining and fine-tuning, with mention of optimizer techniques like LoRA and GaLore.
  

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**LM Studio Hits Version 0.2.16**: LM Studio's latest version is **0.2.16**, resolving previous terminal run errors and addressing `GLIBC` or `LIBCBlast` library issues. Compatibility discussions highlight challenges with `gemma 7b gguf` and `starcoder2` models. For support with GGUF models, refer to [Learning More About GGUF](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio).

**AMD's AI Hardware Optimism**: AMD's CEO Lisa Su’s personal involvement to address [Tiny Corp's GPU firmware concerns](https://www.tomshardware.com/pc-components/gpus/amds-lisa-su-steps-in-to-fix-driver-issues-with-new-tinybox-ai-servers-tiny-corp-calls-for-amd-to-make-its-radeon-7900-xtx-gpu-firmware-open-source) signals potential improvements for AI applications. AMD's article on [running LLMs with AMD Ryzen™ and Radeon™](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709) could assist in leveraging AI without internet dependence.

**Rethinking Power Supply Units (PSU) for AI**: Discussions suggest a minimum of 750W PSU for powering an RTX 3090, with the [Razer Core X](https://www.razer.com/gb-en/gaming-egpus/razer-core-x) eGPU enclosure as an alternative. Debates on efficient hardware setups for language models consider VRAM, power efficiency, and cost-effectiveness.

**Integrating and Selecting GPUs in LM Studio**: There's a call for features allowing specific GPU selection in LM Studio, following incidents where the software defaults to integrated graphics, causing performance issues with demanding AI models.

**Evolving Open Interpreter Usage and Model Sharing**: Conversations in #open-interpreter include the implementation of custom system messages using command `interpreter.system_message = "Your message"` in Python scripts in Open Interpreter. The sharing of links to models such as **LHK_DPO_v1** on Hugging Face spotlights the community's efforts in exchanging AI insights [LHK_DPO_v1_GGUF](https://huggingface.co/owao/LHK_DPO_v1_GGUF). Concerns raised on the Forum about limitations of FusionNet_7Bx2_MoE_14B model's context size can be found [here](https://huggingface.co/TomGrc/FusionNet_7Bx2_MoE_14B/discussions/9#65bb8619a0c61d0c634e7d08).

**Beta Release Buzz in LM Studio**: Anticipation is building in the #beta-releases-chat for an imminent new release, with community members teasing the release and sharing humorous banter about the update's arrival.

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Survey Says! LlamaIndex Wants Your Input**: LlamaIndex invites users to participate in **a 3-minute user survey** aimed at improving their services, specifically documentation, demos, and tutorials. The survey can be accessed through this [SurveyMonkey link](https://www.surveymonkey.com/r/PNSP3P9) or via referenced Tweets.
  
- **Claude 3 Sparkles in Its Versatility**: A new guide highlighting the varied applications of **Claude 3** using LlamaIndex's tools, including **Vanilla RAG, Routing**, and **Sub-question query planning**, is now available in a video format, as announced on [Twitter](https://twitter.com/llama_index/status/1765782262795448358).
  
- **New Jina Reranker Enhances RAG**: LlamaIndex shared about **Jina's new reranking tool** (`jina-reranker-v1-base-en`) designed to improve Retrieval-Augmented Generation (RAG) models by refining vector search results, with details mentioned in a [Twitter post](https://twitter.com/llama_index/status/1765858347583193432).
  
- **CodeHierarchyNodeParser: The Next Leap in Code Understanding**: A breakthrough technique for parsing large code files into a hierarchical structure named `CodeHierarchyNodeParser` has been unveiled by LlamaIndex, potentially revolutionizing RAG/agents handling code. This was introduced on [Twitter](https://twitter.com/llama_index/status/1766152269874266170) by ryanpeach.
  
- **Tackling LlamaIndex Integration and Retrieval Challenges**: Community discussions have highlighted challenges such as the **unavailability of older version documentation**, **integration pitfalls** with Chat Engine and NodeWithScore, **confusion around multi-modal data** storage and retrieval, **scoring algorithm customization**, and **data persistence** issues. These topics were addressed across several resources including GitHub gists, blog posts, and documentation pages, such as the [Slack bot learning guide](https://www.llamaindex.ai/blog/building-a-slack-bot-that-learns-with-llamaindex-qdrant-and-render-c88d4aa72840) and [vector stores on GitHub](https://github.com/run-llama/llama_index/blob/0ae69d46e3735a740214c22a5f72e05d46d92635/llama-index-integrations/vector_stores/llama-index-vector-stores-opensearch/llama_index/vector_stores/opensearch/base.py#L249).
  

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Claude 3 Opus Usage Debate**: The **Claude 3 Opus** was a significant discussion point, where members voiced concerns regarding the restriction to five uses imposed by Perplexity AI's plans. The debate spanned over the cost-efficiency of subscription models, the daily message limits, and comparing to other services like **ChatGPT Plus**.
  
- **Technical Troubleshooting in Perplexity AI**: Users reported difficulties utilizing features such as photo generation and the rewrite function with **Perplexity AI**. Suggestions included resetting browser data and reaching out to [support@perplexity.ai](mailto:support@perplexity.ai) for further assistance.
  
- **Efficiency and Performance Comparisons of AI Models**: The relative performance of various AI models, including **Inflection-2.5**, was tossed around, with users discussing options for model comparison. Meanwhile, **Sonnet** emerged as a recommended tool for benchmarking AI efficiency.
  
- **Shared Links Deepen AI Understanding**: Across channels, users shared [Perplexity AI links](https://www.perplexity.ai/) to compare platforms, learn using AI resources, explore historical progress, understand AI's roles, question authorship within AI creations, and to probe into the contentious topic of AI emotions.
  
- **Inquiries and Discussions on Perplexity API Developments**: Concerning Perplexity API, users inquired about the capabilities of **Perplexity Discover** and the **RAG pipeline**. There were also discussions regarding maximum token outputs for various AI models, highlighting the constraints imposed by context window and finetuning.
  

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Harnessing Evaluation for Custom Needs**: `@pminervini` sought to customize the output format in the harness, proposing a two-step generation process; meanwhile, `@baber_` suggested modifying the [`generate_until` method](https://github.com/EleutherAI/lm-evaluation-harness/blob/9e6e240229429d2214bc281bed7a4e288f5169a1/lm_eval/models/huggingface.py#L1186) and offered a GitHub link as a potential starting point.
  
- **Spectacle of Specs: Docker or Local for GPT-NeoX**: AI enthusiasts `@biiter` and `@tastybucketofrice` debated over environment setup methods for GPT-NeoX development, discussing Docker's consistency against local setups, while `@tfidia` suggested NVIDIA NGC containers as a solution for easing Apex and CUDA dependencies.
  
- **GaLore Optimizer Grips Global Interest**: Techniques like GaLore and CAME optimizer are catching attention for claims of memory efficiency and performance gains; however, the community’s interest is coupled with skepticism and calls for empirical replication and understanding of versioning complexities.
  
- **Data Driven: New Korean Benchmarks Announced**: `@gson_arlo` introduced two new Korean evaluation datasets, [Hae-Rae Bench](https://arxiv.org/abs/2309.02706) and [K-MMLU](https://arxiv.org/abs/2402.11548), developed for assessing language models on Korean-specific knowledge, inviting contributions on multilingual model evaluation.
  
- **Seeking AI Simplicity**: Newcomer `@shida3916` expressed a desire to explore everyday AI applications and seek straightforward answers, provoking discussions on the appropriate forums for such AI inquiries within the community.
  

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Sonnet Shows Strength Over ChatGPT**: `@aaron_speedy` suggested that **Sonnet** outperforms **ChatGPT 3.5**, offering functionalities such as image upload and potentially supporting file uploads. Meanwhile, the release of **GPT-4** is highly anticipated by users like `@futuremachine` and `@drinkoblog.weebly.com` due to perceived competitive lag behind models like **Claude 3**.
  
- **Effective Prompting Key to Model Utilization**: Users like `@.vilden` highlighted the importance of precise prompting in maximizing the performance of models such as **GPT 3.5**, instead of using verbose prompts which may impede performance.
  
- **Flashcard Creation AI Interest Sparked**: `@khaledoo.` inquired about an AI tool for transforming lecture PDFs into flashcards, promoting discussion among users regarding the tool's capability and content accuracy.
  
- **GPT-4 Exhibiting Repeating Answers**: Frustration over **GPT-4**'s repeating answers and multilingual mishaps was reported by `@spikyd`, which led to an exchange with `@dojan1` on possible underlying issues and workarounds.
  
- **Localized Technical Issues with ChatGPT Addressed**: Changes in language settings to "Auto-detect" and browser refresh (F5) were among the suggested fixes by users like `@pteromaple` for ChatGPT non-responsiveness. Issues with language settings causing interface breaks were confirmed by `@joachimpimiskern` and `@pteromaple`, hinting at a possible bug, while `@meteopx` used a VPN to circumvent regional access challenges with ChatGPT.
  

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **AI Safety Czar Sparks Controversy**: The expected appointment of [Paul Christiano](https://paulfchristiano.com/) to the US AI Safety Institute has led to turmoil within the National Institute of Standards and Technology, with staff revolt and resignation threats due to Christiano's views on AI existential risks, as detailed in a [VentureBeat article](https://venturebeat.com/ai/nist-staffers-revolt-against-potential-appointment-of-effective-altruist-ai-researcher-to-us-ai-safety-institute/).
  
- **AI Optimization for Programming**: DeepSeek-Coder instruct and the [OpenCodeInterpreter paper](https://arxiv.org/abs/2402.14658) were discussed for optimizing shader code with AI, while [this code processing review work](https://arxiv.org/abs/2311.07989) provides insights into AI's use in programming tasks.
  
- **Exploring the Future of AI in Geopolitics**: A robust debate on the contrasting AI strategies of Western nations and China was had, touching on issues like censorship, asymmetric warfare, and the potential for AI regulation based on training data concerns.
  
- **AI Tools and Models Shared on HuggingFace**: New resources are highlighted, including a RAG demo for Arxiv CS papers at [HuggingFace Spaces](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG), a fine-tuning demonstration of Google's Gemma with a [notebook available at HuggingFace Spaces](https://huggingface.co/Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101), a new 16k context pretrained encoder model at [HuggingFace](https://huggingface.co/BEE-spoke-data/mega-encoder-small-16k-v1), and an educational resource on constructing GPT shared in ["Let's Build GPT" on YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY).
  
- **Diffusion Model Development Challenges**: Efforts to merge **SDXL-Lightning LoRA** with standard **SDXL** were discussed, with training suggestions offered by the ByteDance organization in a [HuggingFace discussion thread](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104).
  
- **Learning and Discoveries in AI**: Users expressed a keen interest in collaboration and learning about generative AI for data analytics, as well as other AI applications, showing enthusiasm for shared learning experiences and co-study partnerships.
  
- **Creative AI Projects and Contributions**: Innovations included fine-tuning Gemma with ChatML by `@andysingal` (model card available [here](https://huggingface.co/Andyrasika/Gemma-ChatML)), a CLIP index for a dataset hosted on [Hugging Face Hub](https://huggingface.co/datasets/fondant-ai/datacomp-small-clip), and a restaurant name and menu generator app by `@chongdashu` with [Medium article](https://medium.com/@chongdashu/langchain-and-gradio-d23c5e9cee90) and [demo](https://huggingface.co/spaces/chongdashu/langchain-crash-course-gradio).
  
- **Technical Discussions Embrace Humor and Helpfulness**: From puns about retrievers to the lofty goal of running a 70B model on a Raspberry Pi, community members engage both humorously and helpfully on topics such as machine learning model recommendations for Google Colab and mapping attention weights in BertModel.
  

---

## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **AI Image Generators Ignite Interest**: Alternatives to the AI image generator **Sora** are being actively discussed, with numerous projects reportedly using **MagViT2** as their foundation. Meanwhile, concerns over excessive marketing costs, with **$7,099 spent per conversion** for $100 sales, sparked discussions on the need for more efficient strategies.
  
- **Midjourney's Scraping Scare Sparks Humor and Criticism**: Laughter and critical conversations emerged around Midjourney members' fear of a 'security issue' due to their AI-generated images being scraped, together with a controversy regarding an artist list used by Midjourney that included names ranging from Warhol to a **6-year-old child**.
  
- **SVD Training Hiccup Hits Stable Cascade**: Users report that **SVD updates** introduce significant pauses in the training process of **Stable Cascade**, causing a 2-minute interruption which hinders efficiency.
  
- **Efficiency Spotlight on Large Language Models**: Lively discussions tackled the inefficiencies of current Large Language Models (LLMs), with individuals like `@mkaic` arguing for the potential in training more efficient **sparse/small networks** and improving compression of training data within these models.
  
- **Cutting Edge Discussions on Pruning and Model Efficiency**: The engineering community delved into the challenges associated with model pruning and generalizability, pondering over pathways to more efficient architectures. A [new paper](https://arxiv.org/pdf/2403.04692.pdf) was referenced in relation to these topics, while the debut of **PixArt Sigma**, a new 4K PixArt project with a focus on text-to-image generation, was announced despite its current issues with text representation using only **600m parameters**.
  

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord Summary

- **Nitro Models Accelerate the Field**: Alex Atallah announced the introduction of **Nitro models** such as Mixtral, MythoMax, and Llama 70B, which feature more efficient speed and cost-effectiveness powered by **Groq**. Documentation on new developer features like performance timelines, JSON mode, and dynamic routing is available, and the speed-boosted **Mistral 7b 0.2 Nitro model** expands context to 32k, with demonstrations shown on [OpenRouter's site](https://openrouter.ai) and [Twitter](https://twitter.com/OpenRouterAI/status/1766147110443909184).
  
- **Sonnet Scores High on Savings**: Discussions in the community spotlighted **Sonnet's** advantageous price-performance balance, offering costs as low as .03 for scenarios involving "5k context and 1200 response length," setting it ahead of competitors in affordability.
  
- **Deciphering Moderation Layers**: Clarification was provided on how **OpenRouter** applies a unique layer of moderation that may result in more refusals than direct interactions with **OpenAI** or **Anthropic** APIs, with additional insight by Alex Atallah on the specifics of **Anthropic's** server-side moderation for **OpenRouter**.
  
- **Data Usage Policy Under the Microscope**: Anthropic's use of customer content for model training came under inquiry, with links to supportive articles leading to the consensus that content from paid services may be exempt from training purposes.
  
- **Cost vs. Throughput: A Community Analysis**: The guild discussed the **Nitro models'** enhanced throughput and diverse pricing tiers, particularly noting the change with **Mixtral 8x7b instruct nitro** accommodating changes in rates to 0.27/1M tokens.
  

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Learning the CUDA Way**: For those new to CUDA and parallel programming, the Discord's own lectures for **complete beginners** are recommended starting points, coupled with suggested concurrent study of associated books for a richer learning experience.
- **CUDA Shared Memory Utilization**: **CuTe DSL** is being studied within the NVIDIA **[FlashAttention repository](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)** to optimize tensor core utilization, while discussions revolve around the performances of kernel optimizations such as thread coarsening and vectorized operations.
- **Memory-Efficient Techniques for Large Model Training**: **Gradient Low-Rank Projection (GaLore)**, as described in an [arXiv paper](https://arxiv.org/abs/2403.03507), offers a path to train large models with reduced memory requirements, even fitting within a single RTX 4090 GPU, while a method combining FSDP and QLoRA enables fine-tuning a 70b model on standard gaming GPUs, details available at [Answer.AI](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html).
- **Ring-Attention in Practice**: Technical issues involving **RelayAttention** are under discussion, with reports of system failures when training at 16k resolution and inference processes stalling when using ring-llama on two GPUs after installing flash-attn via pip.
- **PyTorch Device Cross-Talk Clarified**: Scalars can be indexed by CUDA tensors in PyTorch due to automatic conversion, a holdover from earlier design decisions, but this auto-transfer can also lead to unexpected inefficiencies.

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

**GaLore Lights Up GPU Potential**: User `@tiagoefreitas` shared insights from `@AnimaAnandkumar` that **Gradient Low-Rank Projection (GaLore)** enables the Llama 7B LLM to be trained on a single RTX 4090 GPU, which could transform memory efficiency benchmarks for both pre-training and fine-tuning stages, possibly enhanced by 1-bit quantization.

**Inflection-2.5 Under the Microscope**: Despite significant performance claims that **Inflection-2.5** rivals GPT-4 with lower compute, `@swyxio` highlighted a gap in Inflection's official communication, observing the absence of this claim from their blog post detailing Inflection-2.5.

**Democratizing AI Training with FSDP/QLoRA**: `@jeremyphoward`'s tweet about FSDP/QLoRA was shared by `@fanahova`, signaling a collaboration that enables training large models on home GPUs, while `@fx2y` pointed to support for quantization techniques like HQQ and bitsandbytes, shared via a GitHub repo link.

**Yann LeCun Expounds on AI’s Horizons**: Discussions steered towards Yann LeCun's Lex Fridman podcast episode, where he shared his visions for Meta AI, the limitations of current LLMs, and prospects for Contrastive Learning's future.

**Data Privacy Concerns in Personal AI**: `@swyxio` related their experience with **Life Story**, a personal biographer AI, prompting `@tiagoefreitas` to encourage development of local-hosted applications for better data security.

**Inside the Depth of GPT**: `@ivanleomk` and `@1123457263638683770` led a session on the GPT-2 paper, with materials explaining concepts and implementation highlighted, alongside a discussion punctuated by a clarification on "causal attention" and the introduction of a [LLM Visualization tool](https://bbycroft.net/llm).

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain JS Still Chasing Python's Tail**: Despite questions raised by `@0x404blockchainnotfound`, there's still no clear confirmation if **LangChain JS library** has achieved feature parity with its Python counterpart; users discussed related tool issues instead, such as a delay with the Finished agent event in Python and formatting challenges when using PyPDFLoader.
  
- **The Never-Ending AGI Debate**: Conversations on AGI from Hacker News spilled over without reaching a conclusive end but did spark parallel discussions on LangChain tools and ReACT agents. Critical concerns remain unaddressed, indicating a need for deeper technical dives into the subject.
  
- **The Redis Memory Mix-up**: `@justanothergraphguy` grapples with intricacies in structuring output in a chat chain with Redis, where "HumanMessage" incorrectly appears in the `AIMessage`, highlighting potential flaws in memory management during interactions.
  
- **Vision Models Grab the Stage**: `@vru.shank` invites the community to a workshop with [MultiOn](https://www.multion.ai/) and [Quizizz](https://quizizz.com) on integrating vision models into production, promising insights from the front lines of AI application.
  
- **Prompt Mixer: A Developer’s New Best Friend?**: `@tomatyss` introduces **Prompt Mixer**, a desktop application adept for crafting and iterating AI prompts, while also providing a [tutorial](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector) to extend the tool with custom connectors, signaling a move towards more personalized and efficient AI development workflows.
  

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **In Search of a Sauerkraut-Flavored AI**: `@johannhartmann` mentioned a gap in **German-fineturned models**, emphasizing **Nous Hermes Mixtral** doesn't cater to German language prompts, and compared to **sauerkraut oder discolm mixtrals**.
  
- **DNA's New Best Friend**: `@rasdani` introduced the **Evo architecture**—Striped Hyena by TogetherAI, specialized for DNA sequencing. Details about its application for biology can be found in their [blog post](https://www.together.ai/blog/evo), developed in collaboration with the [Arc Institute](https://arcinstitute.org/).
  
- **Tuning the Hermes Harmony**: `@flozi00` is refining the **Nous Hermes Mixtral DPO model**, and constructing an **Argilla space** for assessing translations from Google Translate, DeepL, and Azure Translate. Contributions to measure translation pair quality can be made to their [HuggingFace collection](https://huggingface.co/collections/flozi00/translation-data-quality-65e9d0cdd977e1e0aed2de9d).
  
- **Dataset Dilemma Discussion**: `@philipmay` encountered licensing and accessibility issues with the mMARCO dataset, which now has an Apache 2.0 license but requires troubleshooting for dataset viewing on HuggingFace.
  
- **Melding German with SPIN Strategy**: `@johannhartmann` utilizes a German-transformed dataset for **Mistral merges** that shows varied model responses post-merge, planning to share this dataset soon, while `@crispstrobe` experiences success with **Brezn3** surpassing **Brezn-7b** on EQ-Bench (v2) (de) without any specific DPO modifications confirmed as yet by `@johannhartmann`.
  

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Naming Conundrums in AI**: `@res6969` noted that **names** can be challenging for models to handle accurately, though no specific instances or models were cited.
- **Breaking Ground with Claude's Functions**: `@res6969` reported progress with **function calling** in *Claude*, attesting to its operational state but without providing specific examples or outcomes.
- **Claude's Humor Hits the Mark**: `@res6969` deemed Claude's output as both **"hilarious and correct"**, though the context of this performance was not specified.
- **The XML Necessity for Claude's Calls**: Function calling in Claude has been confirmed by `@res6969` to work effectively, specifically when using **XML tags**, hinting at a technical requirement for Claude's optimal function-calling performance.
- **XML Tags: A Double-Edged Sword**: `@pantsforbirds` raised concerns about the intricacies of using **XML tags** in prompt generators, implying potential difficulties in their implementation and use.

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **GPT-4's Unexpected Shortcomings**: Members were surprised at **GPT-4's** underperformance on an unspecified test, highlighting room for improvement in its development.
- **Ingenious Clickable Bookshelves**: A novel script for generating clickable bookshelf images that link to Google Books has captivated members, with references including a [blog post](https://jamesg.blog/2024/02/14/clickable-bookshelves/) and a [demo](https://capjamesg.github.io/cv-book-svg/).
- **Library Tech Advancements Garner Interest**: The idea of automated bookshelf management sparked interest, especially considering its potential to streamline shelf-reading tasks in extensive library collections.
- **Scaling Library Management Efforts**: A member shared insights into large-scale library management, noting their partner's role in overseeing the largest school library in a 35-school diocesan system, comparable to some public libraries.
- **Little Library, Big Data**: The concept of a small-scale app to catalog books in community-based little libraries was floated, indicative of personal projects leveraging cataloging and data management principles.

---

# PART 2: Detailed by-Channel summaries and links

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1215294591494787122) (17 messages🔥):

- **Delayed Apologies**: `@teknium` acknowledged seeing a direct message on Twitter after a long time and apologized for the missed communication, humorously expressing regret with "xD".
- **Meme Making with Mistral**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=PtP8R8VjTGc) titled "Making memes with Mistral & Giphy" and a [GitHub repository](https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb) containing a notebook for generating memes using Mistral LLM and Giphy API.
- **Inquiry on Nous' Origin**: `@pier1337` questioned if the Nous organization has French origins, leading to a response by `@kainan_e` who clarified that the inspiration was from the Greek "νοῦς" meaning intelligence, not the French language.
- **Suggestions for Note-taking Excellence**: `@sanketpatrikar` sought advice on improving the experience of using a single markdown notes file, and `@thilotee` provided several recommendations, including using a good text editor, visiting alternative software websites like [AlternativeTo](https://alternativeto.net/software/obsidian/?feature=markdown-support&license=opensource) and exploring the [Zettelkasten method](https://zettelkasten.de/overview/).
- **Doodle Wars Game Announcement**: `@om7059` introduced Doodle Wars—a multiplayer game where players doodle objects within 15 seconds and a neural network scores their creations. The highest-scoring player wins the round. Check out the game at [Doodle Wars](https://doodlewars.netlify.app).

**Links mentioned**:

- [Making memes with Mistral & Giphy](https://www.youtube.com/watch?v=PtP8R8VjTGc): Lets make memes using mistral llm and Giphy api#llm #ml #python #pythonprogramming [https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb](https://github.com/githubpradeep/notebooks/blob/main/Giphy%20Mistral.ipynb)
- [Doodle Wars](https://doodlewars.netlify.app): no description found
- [Getting Started • Zettelkasten Method](https://zettelkasten.de/overview/): no description found
- [Noûs — Wikipédia](https://fr.wikipedia.org/wiki/No%C3%BBs): no description found
- [Nous - Wikipedia](https://en.m.wikipedia.org/wiki/Nous): no description found

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1215207673050566656) (34 messages🔥):

- **Claude 3 Opus Casts a Spell on Circassian Translations**: `@hahahahohohe` shared a remarkable experience with [@AnthropicAI](https://twitter.com/AnthropicAI)'s **Claude 3 Opus**, demonstrating exceptional Russian-Circassian translation skills, even with a limited dataset of 5.7K examples, surpassing expectations and previous models. However, it was later clarified that the model may have already had access to Circassian language information, underscoring the importance of accurate data about model capabilities.
  
- **Exploring GitHub's Contributions to AI**: `@random_string_of_character` posted a link to [GaLore on GitHub](https://github.com/jiaweizzhao/GaLore), encouraging the community to assess its value, as well as suggesting combining it with [low-bit optimizers](https://github.com/thu-ml/low-bit-optimizers) for potential computational efficiency improvements.
  
- **Yi Technology Pushes Bounds of Long Text Understanding**: `@thilotee` shared a [Reddit post](https://www.reddit.com/r/LocalLLaMA/comments/1b8qnqv/yi34b200k_model_update_needleinahaystack_improved/) and a [Hugging Face link](https://huggingface.co/01-ai/Yi-34B-200K) discussing the Yi-34B-200K base model's update, which significantly improved its performance on the "Needle-in-a-Haystack" test from 89.3% to 99.8% accuracy, pointing towards the continued enhancement of the model's handling of long contexts.
  
- **Sparse Mixture of Models Enables Deep Understanding**: `@shashank.f1` pointed to a [YouTube video](https://youtu.be/IuehDA1M_Lw) where there's an in-depth conversation with the 🤗 community about sparse mixture of experts (MoE) architectures like Gemini, which have the potential to ingest and reason from entire books and movies in a single prompt.
  

**Links mentioned**:

- [Tweet from An Qu (@hahahahohohe)](https://x.com/hahahahohohe/status/1765088860592394250?s=46): Today while testing @AnthropicAI 's new model Claude 3 Opus I witnessed something so astonishing it genuinely felt like a miracle. Hate to sound clickbaity, but this is really what it felt like. ...
- [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652): We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language mode...
- [Gemini 1.5 Pro: Unlock reasoning and knowledge from entire books and movies in a single prompt](https://youtu.be/IuehDA1M_Lw): 🚀 Dive into the world of AI with Gemini 1.5! 🌟In this video, we unpack the magic behind Gemini's sparse mixture of experts architecture, perfect for unleas...
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [01-ai/Yi-34B-200K · Hugging Face](https://huggingface.co/01-ai/Yi-34B-200K): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1b8qnqv/yi34b200k_model_update_needleinahaystack_improved/): no description found
- [GitHub - thu-ml/low-bit-optimizers: Low-bit optimizers for PyTorch](https://github.com/thu-ml/low-bit-optimizers/): Low-bit optimizers for PyTorch. Contribute to thu-ml/low-bit-optimizers development by creating an account on GitHub.

---

### Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1215366139400421396) (1 messages):

- **Introducing Genstruct 7B**: `@everyone`, Nous Research released **Genstruct 7B**, an instruction-generation model inspired by [Ada-Instruct](https://arxiv.org/abs/2310.04484). The model is capable of creating valid instructions from raw text corpus for synthetic finetuning datasets and is available for download on their [HuggingFace page](https://huggingface.co/NousResearch/Genstruct-7B).
- **Advanced Reasoning Capabilities**: **Genstruct 7B** excels in generating questions about complex scenarios, enhancing the ability of models to carry out step-by-step reasoning after being trained on the generated data. This project was led by `<@811403041612759080>` at Nous Research.

**Links mentioned**:

[NousResearch/Genstruct-7B · Hugging Face](https://huggingface.co/NousResearch/Genstruct-7B): no description found

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1215213599623094313) (289 messages🔥🔥):

- **Claude Pro As Daily Driver?**: User `@leontello` inquired about people's experiences of swapping ChatGPT Plus for Claude Pro. Some users, like `@teknium`, expressed difficulty in even finding the Claude Pro chat interface, while others mentioned being unable to access it due to geographic restrictions.
  
- **Gemma AI Bugs and Fixes**: Discussion around bugs in Gemma implementation led to sharing a [tweet from @danielhanchen](https://fxtwitter.com/danielhanchen/status/1765446273661075609?s=20), highlighting numerous issues and fixes that were pushed to @UnslothAI, comparing the Log L2 norms for each layer after applying fixes mentioned in an Unsloth AI blog post.
  
- **Low Rank Pre-Training on a 4090 GPU**: User `@.interstellarninja` shared a [tweet from @AnimaAnandkumar](https://fxtwitter.com/AnimaAnandkumar/status/1765613815146893348?s=20) announcing the Llama 7B LLM's capability to be trained on a single RTX 4090 GPU using a method that significantly reduces memory requirements for storing optimizer states via Gradient Low-Rank Projection (GaLore).
  
- **Discussion on Model Performance and New Models**: Amongst talks about various models, `@teknium` raises the question about the implications of multiple AI models potentially matching or outperforming OpenAI's flagship GPT models. A [tweet from @inflectionAI](https://x.com/inflectionai/status/1765751898001608793) claims their Inflection-2.5 model matches GPT-4 benchmarks while using less compute for training.
  
- **EU Scrutiny of Microsoft’s Partnership with Mistral AI**: Users discuss potential implications of Microsoft's deal with Mistral AI, including regulatory interest from the EU. References to an [AP news article](https://apnews.com/article/european-union-microsoft-mistral-competition-antitrust-05d6eb911e56f88b7da20ebc224efac4) indicate that the EU is investigating the agreement, although no formal conclusions are mentioned.
  

**Links mentioned**:

- [Tweet from Daniel Han (@danielhanchen)](https://fxtwitter.com/danielhanchen/status/1765446273661075609?s=20): Found more bugs for #Gemma: 1. Must add <bos> 2. There’s a typo for <end_of_turn>model 3. sqrt(3072)=55.4256 but bfloat16 is 55.5 4. Layernorm (w+1) must be in float32 5. Keras mixed_bfloa...
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [Anthropic Console](https://console.anthropic.com/): no description found
- [Tweet from Emad (@EMostaque)](https://x.com/emostaque/status/1765680597597372823?s=46): @Teknium1 Less stable above 7b. Transformer engine has it as main implementation. Intel have one too and Google have int8
- [Tweet from FxTwitter / FixupX](https://fxtwitter.com/AnimaAnandku): Sorry, that user doesn't exist :(
- [Tweet from Inflection AI (@inflectionAI)](https://x.com/inflectionai/status/1765751898001608793): Pi just got a huge upgrade! It’s now powered by our latest LLM: Inflection-2.5, which is neck and neck with GPT-4 on all benchmarks and used less than half the compute to train. Pi now has world clas...
- [Microsoft's new deal with France's Mistral AI is under scrutiny from the European Union](https://apnews.com/article/european-union-microsoft-mistral-competition-antitrust-05d6eb911e56f88b7da20ebc224efac4): The European Union is looking into Microsoft’s partnership with French startup Mistral AI. It's part of a broader review of the booming generative artificial intelligence sector to see if it rais...
- [Tweet from Sebastian Majstorovic (@storytracer)](https://fxtwitter.com/storytracer/status/1765410706638160303?s=20): Open source LLMs need open training data. Today I release the largest dataset of English public domain books curated from the @internetarchive and the @openlibrary. It consists of more than 61 billion...
- [Tweet from Prof. Anima Anandkumar (@AnimaAnandkumar)](https://fxtwitter.com/AnimaAnandkumar/status/1765613815146893348?s=20): For the first time, we show that the Llama 7B LLM can be trained on a single consumer-grade GPU (RTX 4090) with only 24GB memory. This represents more than 82.5% reduction in memory for storing optimi...
- [gguf/Genstruct-7B-GGUF · Hugging Face](https://huggingface.co/gguf/Genstruct-7B-GGUF): no description found
- [Weyaxi/Einstein-v4-7B · Hugging Face](https://hf.co/Weyaxi/Einstein-v4-7B): no description found
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1765851433448944125): 🎉 Exciting News! 🧑‍🔬 Meet Einstein-v4-7B, a powerful mistral-based supervised fine-tuned model using diverse high quality and filtered open source datasets!🚀 ✍️ I also converted multiple-choice...
- [WIP: galore optimizer by maximegmd · Pull Request #1370 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370): Adds support for Galore optimizers Still a WIP, untested.
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [Tweet from Seb Lhomme (@slhomme)](https://x.com/slhomme/status/1765778634839593232?s=46): My new AI tool coming up: SocialClone - Create AI-Clone Videos Instantly!
- [GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets](https://github.com/e-p-armstrong/augmentoolkit): Convert Compute And Books Into Instruct-Tuning Datasets - e-p-armstrong/augmentoolkit
- [Swim In GIF - Swim In Swimming - Discover & Share GIFs](https://tenor.com/view/swim-in-swimming-pool-underwater-gif-23188415): Click to view the GIF
- [Worried Scared GIF - Worried Scared Oh No - Discover & Share GIFs](https://tenor.com/view/worried-scared-oh-no-stop-it-fearful-gif-12534009): Click to view the GIF
- [How to Fine-Tune LLMs in 2024 with Hugging Face](https://www.philschmid.de/fine-tune-llms-in-2024-with-trl): In this blog post you will learn how to fine-tune LLMs using Hugging Face TRL, Transformers and Datasets in 2024. We will fine-tune a LLM on a text to SQL dataset.
- [Yann Lecun: Meta AI, Open Source, Limits of LLMs, AGI & the Future of AI | Lex Fridman Podcast #416](https://youtu.be/5t1vTLU7s40?si=HS3WrupXGw_xBvmb): Yann LeCun is the Chief AI Scientist at Meta, professor at NYU, Turing Award winner, and one of the most influential researchers in the history of AI. Please...

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1215263475832725545) (83 messages🔥🔥):

- **Seeking Free GPT-4 Access**: `@micron588` inquired about ways to access GPT-4 for free, especially through an API. `@teknium` offered a point of help by referencing [Corcel.io](https://corcel.io/), a platform that provides free ChatGPT-4 access and direct API integration to the Bittensor network.
  
- **Misunderstood Model Name**: `@micron588` expressed skepticism about the Corcel.io model being GPT-4 as it didn't respond in kind and lacked real-time data capabilities. `@teknium` clarified that GPT-4 does not typically include real-time data.
  
- **Nous-Hermes Model Context Length Query**: `@nickcbrown` asked why the context length in some Nous-Hermes models appeared to be reduced. `@night_w0lf` suggested it might be a configuration or hardware limitation rather than an actual reduction.
  
- **Discussion on LLM Pretraining and Finetuning**: The difference between large language model (LLM) pretraining and fine-tuning was debated. `@teknium` and `@carsonpoole` discussed the nuances of LoRA, DoRA, VeRA, and GaloRe and their impact on model optimization and expressiveness.
  
- **The Cost of Pretraining and Model Optimization Techniques**: `@umarigan` highlighted the resource-intensive nature of continued pretraining for LLMs and shared an article on the subject, while `@eas2535` alluded to the advancements of FSDP and QLoRA for training large models on fewer resources. `@teknium` countered with skepticism, implying these strategies might still be out of reach for smaller operations.
  

**Links mentioned**:

- [Corcel · Build with the power of Bittensor](https://corcel.io/): no description found
- [$ Cost of LLM continued pre-training](https://medium.com/@gilinachum/cost-of-llm-continued-pre-training-0c1998cb44ec): How much will it cost you to do continued pre-training for a small (7B) LLM?
- [Trendyol/Trendyol-LLM-7b-base-v0.1 · Hugging Face](https://huggingface.co/Trendyol/Trendyol-LLM-7b-base-v0.1): no description found
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [Poor Man GIF - Poor Man - Discover & Share GIFs](https://tenor.com/view/poor-man-gif-23343928): Click to view the GIF
- [no title found](https://medium.com/@gilinachum/cost-of-llm-continued-pre-training-): no description found

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1215206214674419712) (148 messages🔥🔥):

- **Update on LM Studio Version**: `@datasoul` mentions the latest version is **0.2.16**. `@heyitsyorkie` provides support regarding GGUF model compatibility and links to the Huggingface repository.
- **Evaluating Model Speed with Metal**: `@nullt3r` and `@heyitsyorkie` discuss evaluation speeds for Mixtral and other models on various setups, with speeds ranging from **27 tok/s** to **4 tok/s** depending on the model and quality.
- **REOR, the Self-organizing AI Note-Taking App**: `@clubofom` finds **REOR** to be an effective AI note-taking app and provides a link to the project page: [www.reorproject.org](https://www.reorproject.org/).
- **Pi's Inflection 2.5 and Inflection AI**: `@pierrunoyt` and `@aswarp` discuss the new **Inflection-2.5** model from **Inflection AI**, noting improvements in coding and IT support. They also share a YouTube video discussing the update: [Inflection 2.5](https://youtu.be/fEpa_Ak6Ec4?si=9bLvLARbKL91o1lp).
- **Running Local LLM Models with LM Studio**: `@heyitsyorkie` advises `@.atip` that local **LLM models** must be in GGUF format and within specific folder structures to work with LM Studio, sharing a link to the unofficial FAQ: [Learning More About GGUF](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio) and discussing the conversion process.

**Links mentioned**:

- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-releases.html)): Find, download, and experiment with local LLMs
- [RIP Midjourney! FREE & UNCENSORED SDXL 1.0 is TAKING OVER!](https://www.youtube.com/watch?v=A0xUnf5302k&ab_channel=Aitrepreneur): Say goodbye to Midjourney and hello to the future of free open-source AI image generation: SDXL 1.0! This new, uncensored model is taking the AI world by sto...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18av9aw/quick_start_guide_to_converting_your_own_ggufs/): no description found
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed). LMStudio is a free closed...
- [22,000 H100s later, Inflection 2.5!!!](https://youtu.be/fEpa_Ak6Ec4?si=9bLvLARbKL91o1lp): 🔗 Links 🔗[https://inflection.ai/inflection-2-5❤️](https://inflection.ai/inflection-2-5%E2%9D%A4%EF%B8%8F) If you want to support the channel ❤️Support here:Patreon - [https://www.patreon.com/1littlecoder/Ko-Fi](https://www.patreon.com/1littlecoder/Ko-Fi) - ht...
- [Inflection-2.5: meet the world's best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Reor](https://www.reorproject.org/): AI note-taking app that runs models locally & offline on your computer.
- [‎Pal - AI Chat Client](https://apps.apple.com/us/app/pal-ai-chat-client/id6447545085?platform=iphone): ‎A lightweight but powerful and feature-rich AI Chat Client for your iPhone! Support for: GPT-4 Turbo, GPT-4 Vision, DALL-E 3, Claude 3 Opus, Gemini Pro, Mistral Large, Openrouter, and custom endpoin...

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1215204975781609522) (68 messages🔥🔥):

- **LM Studio Terminal Troubles**: `@heyitsyorkie` clarified that running LM Studio from the terminal should not result in any errors such as missing `GLIBC` or `LIBCBlast` libraries as of version 0.2.16.
- **Gemma Model Gripes**: `@honeylaker_62748_43426` faced an error with a `gemma 7b gguf` model which `@heyitsyorkie` confirmed to be a known issue with these models.
- **Starcoder2 Compatibility Confusion**: Multiple users including `@madhur_11`, `@poshigetoshi`, and `@zachmayer` discussed issues with `starcoder2` as it is not recognized by LM Studio in its current build.
- **Image Generation Guidance**: `@heyitsyorkie` redirected `@callmemjinina` looking for a model to generate pictures to explore image generation tools like Stable Diffusion and interfaces like Automatic 1111 or ComfyUI.
- **RAG Explanation Requested**: As `@neuropixels` inquired about setting up a knowledge database for chatbots, `@heyitsyorkie` shared a link to IBM's article explaining Retrieval-Augmented Generation (RAG), which could potentially address their requirements.

**Links mentioned**:

- [What is retrieval-augmented generation? | IBM Research Blog](https://research.ibm.com/blog/retrieval-augmented-generation-RAG): RAG is an AI framework for retrieving facts to ground LLMs on the most accurate information and to give users insight into AI’s decisionmaking process.
- [Kquant03/TechxGenus-starcoder2-15b-instruct-GGUF · Hugging Face](https://huggingface.co/Kquant03/TechxGenus-starcoder2-15b-instruct-GGUF): no description found

---

### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (1 messages):

heyitsyorkie: Stop using <#1113937247520170084> for help posts. Use <#1111440136287297637>

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1215275990058795059) (66 messages🔥🔥):

- **Powering up the RTX 3090**: `@wilsonkeebs` is looking for the smallest PSU to power a standalone RTX 3090 and `@heyitsyorkie` suggests a 750W PSU at minimum, mentioning that lower wattage PSUs might lack the necessary PCIe cables, despite another user recommending a [Razer Core X](https://www.razer.com/gb-en/gaming-egpus/razer-core-x) eGPU enclosure.
- **Considering Future Upgrades**: `@wilsonkeebs` plans to eventually rebuild their PC with a 1500W PSU and a larger case, with the current search for a PSU being a temporary solution.
- **The Value Debate of GPUs**: In the context of LM (language model) focused builds, users discuss the cost versus VRAM benefits of newer 4060 Ti cards against the second-hand market for the 3090, considering power efficiency and pricing differences across various regions like Canada and Australia.
- **Board Choices and Component Compatibility**: `@jedd1` and `@nink1` discuss the challenges of finding motherboards that support multiple high-end GPUs, with considerations of PCIe slot availability and supported features, alongside power consumption and pricing strategies for builds.
- **Running Heavy Models on Consumer Hardware**: `@neuropixels` shares difficulties in running a large language model on a Nvidia GeForce 1080 Ti with 11GB VRAM, which is resolved after restarting the workstation, indicating potential issues with hardware compatibility or software glitches when dealing with demanding AI models.

**Links mentioned**:

- [Razer Core X - Thunderbolt™ 3 eGPU | Razer United Kingdom](https://www.razer.com/gb-en/gaming-egpus/razer-core-x): Now compatible with Mac and Windows laptops, featuring 3-slot PCI-Express desktop graphic cards, 650W power supply, and charges via USB-C.
- [PSU for NVIDIA GeForce RTX 3090 | Power Supply Calculator](https://www.whatpsu.com/psu/gpu/NVIDIA-GeForce-RTX-3090): See what power supply you need for your NVIDIA GeForce RTX 3090

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1215487173072126012) (4 messages):

- **Anticipation Builds Up**: User `@yagilb` hinted at an upcoming release with a brief message: *Coming soon.*
- **Are We There Yet?**: `@wolfspyre` inquired in a light-hearted fashion about the arrival of the expected update, asking, *are we there yet?*
- **The Countdown Reset**: Shortly after, `@wolfspyre` jokingly apologized for resetting the imaginary countdown for the awaited update, saying, *oops... I just reset the timer y'all... my bad... it's my fault it's gonna take a bit longer... sorry.*

---

### LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1215278554787618847) (22 messages🔥):

- **AMD's CEO Intervenes in Tiny Corp GPU Saga**: `@senecalouck` posted an article link highlighting how AMD's CEO Lisa Su stepped in to address [Tiny Corp's](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production) frustration with the Radeon RX 7900 XTX GPU firmware, following public complaints and a request for the firmware to be open sourced; `@berendbotje1` sees this as potentially eye-opening for AMD.
- **Boost Your Productivity with AMD and AI**: `@helloword` shared an [AMD Community Blog](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709) post detailing how to run a GPT based LLM-powered AI chatbot on AMD Ryzen™ AI PCs or Radeon™ 7000 series graphics cards to help increase productivity without needing an internet connection.
- **Troubleshooting LM Studio on AMD**: `@briansp2020` described issues with running models on LM Studio using the Radeon 7900XTX GPU, which worked when not using GPU acceleration; `@jello_pudding` suggested LM Studio might be trying to use an integrated GPU instead of the dedicated one.
- **GPU Selection for LM Studio**: `@jello_pudding` mentioned the need for a feature to select specific GPUs for LM Studio usage, hinting at difficulties caused by the software defaulting to integrated graphics; `@yagilb` acknowledged the suggestion as a valid point of concern.
- **VRAM Confusion Resolved**: Clarifications regarding VRAM estimations were discussed as `@beanz_y` questioned the VRAM capacity, with `@yagilb` correcting that the 47GB figure referred to regular RAM, while the program estimated `23.86GB` VRAM usage.

**Links mentioned**:

- [AMD’s Lisa Su steps in to fix driver issues with GPUs in new TinyBox AI servers — firm calls for AMD to make its GPU firmware open source, points to issues with Radeon 7900 XTX](https://www.tomshardware.com/pc-components/gpus/amds-lisa-su-steps-in-to-fix-driver-issues-with-new-tinybox-ai-servers-tiny-corp-calls-for-amd-to-make-its-radeon-7900-xtx-gpu-firmware-open-source): The intervention comes as Tiny Box publicly frets about Radeon-based platform bugs.
- [How to run a Large Language Model (LLM) on your AMD Ryzen™ AI PC or Radeon Graphics Card](https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709): Did you know that you can run your very own instance of a GPT based LLM-powered AI chatbot on your Ryzen™ AI PC or Radeon™ 7000 series graphics card? AI assistants are quickly becoming essential resou...
- [GitHub - amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW/.): Contribute to amd/RyzenAI-SW development by creating an account on GitHub.

---

### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1215334149397680168) (3 messages):

- **Seeking Solutions for Response Speed**: `@alluring_seahorse_04960` is in search of ways to increase response speed and encountered a `Connection to telemetry.crewai.com timed out` error.
- **Baseline for Local Operations**: `@wolfspyre` suggests establishing a baseline with a simple operation that runs locally as a potential starting point for dealing with response speed issues.
- **Building a Generalizable Framework**: `@pefortin` detailed their work on a more generalizable framework involving a front-facing agent to clarify user tasks, a project manager to delineate atomic tasks, HR recruitment expert agents to craft specialized agents for tasks, and an executor agent to launch configured python scripts. While the system is currently slow and performing poorly, refinements are underway.

---

### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1215284906796519454) (87 messages🔥🔥):

- **Confusion Over Interpreter Options**: User `@nxonxi` brought up running the interpreter with a **system message** option, while `@1sbefore` expressed confusion, noting that it wasn't mentioned in the docs they found. `@nxonxi` then clarified the option `-s` is short for `--system_message` as mentioned in the [documentation](https://docs.openinterpreter.com/settings/all-settings#system-message).
  
- **Seeking Python Script Help**: `@nxonxi` sought assistance on setting the **default system message** within a Python script, which `@1sbefore` admitted inability to help with. The issue revolved around using the command `interpreter.system_message = "Your message"` in the script but not getting the expected result.
  
- **Troubleshooting Profile Issues**: `@nxonxi` faced challenges trying to implement changes in **intent profiles**, ending in no observed changes on the language model server (LMS). `@1sbefore` suggested ensuring the modification path matched with the Python path provided by `which interpreter` in the user's environment.
  
- **Exploring Different Language Models**: There was a discussion about various language models such as **deepseek coder 6** and **openchat/mistral** and their responses to prompts. `@berendbotje1` and `@1sbefore` considered the potential and shared experiences with models like **LHK_DPO_v1** and **Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B**.
  
- **Exchanging Model Insights and Recommendations**: `@1sbefore` provided a link to the **GGUFs for HanNayeoniee/LHK_DPO_v1** hosted on Hugging Face, and expressed intention to update channel members on further tests. They also warned about potential limitations regarding context size, citing a discussion on the reliability of **FusionNet_7Bx2_MoE_14B** beyond 4000 tokens ([source](https://huggingface.co/TomGrc/FusionNet_7Bx2_MoE_14B/discussions/9#65bb8619a0c61d0c634e7d08)).
  

**Links mentioned**:

- [owao/LHK_DPO_v1_GGUF · Hugging Face](https://huggingface.co/owao/LHK_DPO_v1_GGUF): no description found
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#system-message): no description found
- [All Settings - Open Interpreter](https://docs.openinterpreter.com/settings/all-settings#custom-instructions): no description found
- [TomGrc/FusionNet_7Bx2_MoE_14B · Contextsize](https://huggingface.co/TomGrc/FusionNet_7Bx2_MoE_14B/discussions/9#65bb8619a0c61d0c634e7d08): no description found
- [GitHub - jondurbin/bagel: A bagel, with everything.](https://github.com/jondurbin/bagel?tab=readme-ov-file#prompt-formatting): A bagel, with everything. Contribute to jondurbin/bagel development by creating an account on GitHub.

---

### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1215391107030978631) (1 messages):

- **We Want Your Feedback!**: `@seldo_v` invites users to **complete a 3-minute user survey** to assist LlamaIndex in enhancing their offerings. The survey seeks to gather insights to improve documentation, demos, and tutorials; the link is [here](https://www.surveymonkey.com/r/PNSP3P9).

**Links mentioned**:

[LlamaIndex user survey](https://www.surveymonkey.com/r/PNSP3P9): Take this survey powered by surveymonkey.com. Create your own surveys for free.

---

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1215341139079462992) (4 messages):

- **Exploring Claude 3's Versatility**: A new video guide 🎞️ showcases a comprehensive cookbook for **Claude 3**, featuring various use cases with @llama_index's tools such as Vanilla RAG, Routing, Sub-question query planning, and more. The guide is accessible on [Twitter](https://twitter.com/llama_index/status/1765782262795448358).
- **LlamaIndex Seeks User Feedback**: LlamaIndex is conducting a quick 3-minute user survey to better understand users' experience levels and needs in order to improve documentation, demos, and tutorials. Interested participants can find the survey [here](https://twitter.com/llama_index/status/1765831945077084578).
- **Improving RAG with Jina's New Reranker**: A just-released reranker tool named `jina-reranker-v1-base-en` by @JinaAI_ promises to dramatically enhance **RAG applications** by providing quality improvements to vector search. Details are available via [Twitter](https://twitter.com/llama_index/status/1765858347583193432).
- **Novel Hierarchical Code Splitting Technique Unveiled**: The `CodeHierarchyNodeParser` is a new technique credited to ryanpeach, allowing for advanced RAG/agents for code understanding by converting large code files into a manageable hierarchy. Announcement and more information shared on [Twitter](https://twitter.com/llama_index/status/1766152269874266170).

**Links mentioned**:

[LlamaIndex user survey](https://t.co/cadlrPztJo): Take this survey powered by surveymonkey.com. Create your own surveys for free.

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1215247863781527572) (339 messages🔥🔥):

- **Missing Documentation for Older Versions**: Users `@torsten_13392` and `@nesgiv` expressed concerns about the unavailability of older version documentation for LlamaIndex, noting that they could no longer access them via Google or on the official site.
- **Integration Issues with Chat Engine and NodeWithScore**: `@cheesyfishes` clarified to `@nesgiv` that chat engines only take strings as input and are specifically meant for chatting, suggesting that customization outside of this may require a custom retriever.
- **Multi-modal Data Storage and Retrieval Confusion**: Users encountered difficulties determining how to store images with metadata in Weaviate using LlamaIndex. `@cheesyfishes` shared a workaround to store the node in the database and the image elsewhere, while `@whitefang_jr` referred users to Chroma.
- **Customizing Scoring Algorithms**: User `@cheesyfishes` provided insight on customizing the scoring algorithm in a retriever, indicating that the ability to configure scoring depends on the vector database exposing such an option.
- **Issues with Updating and Persisting Data**: Users including `@capn_stabn` discussed issues related to updating indexes and persistent storage. Capn_stabn specifically mentioned problems with Milvus deleting data after updating the index, which was later resolved by adjusting the `overwrite` setting.

**Links mentioned**:

- [no title found](https://llamahub.ai/l/tools/llama-index-tools-database?from=tools): no description found
- [no title found](https://llamahub.ai/l/readers/llama-index-readers-snowflake?from=): no description found
- [no title found](https://llamahub.ai/l/llama-packs/llama-index-packs-snowflake-query-engine?from=): no description found
- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [no title found](https://www.secinsights.ai/): no description found
- [Prefill Claude's response](https://docs.anthropic.com/claude/docs/prefill-claudes-response): no description found
- [Starter Tutorial - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html): no description found
- [LlamaIndex user survey](https://www.surveymonkey.com/r/PNSP3P9): Take this survey powered by surveymonkey.com. Create your own surveys for free.
- [gist:7f54b5ae756b5362b3ec0871b845eeac](https://gist.github.com/thoraxe/7f54b5ae756b5362b3ec0871b845eeac): GitHub Gist: instantly share code, notes, and snippets.
- [Building a Slack bot that learns with LlamaIndex, Qdrant and Render — LlamaIndex, Data Framework for LLM Applications](https://www.llamaindex.ai/blog/building-a-slack-bot-that-learns-with-llamaindex-qdrant-and-render-c88d4aa72840): LlamaIndex is a simple, flexible data framework for connecting custom data sources to large language models (LLMs).
- [Advanced Multi-Modal Retrieval using GPT4V and Multi-Modal Index/Retriever - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/multi_modal/gpt4v_multi_modal_retrieval.html): no description found
- [Usage Pattern - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts): no description found
- [HuggingFace LLM - StableLM - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.html): no description found
- [Ingestion Pipeline - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html): no description found
- [Chroma Multi-Modal Demo with LlamaIndex - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/multi_modal/ChromaMultiModalDemo.html): no description found
- [Multimodal Retrieval Augmented Generation(RAG) | Weaviate - Vector Database](https://weaviate.io/blog/multimodal-rag): A picture is worth a thousand words, so why just stop at retrieving textual context!? Learn how to perform multimodal RAG!
- [Ensemble Retrieval Guide - LlamaIndex 🦙 v0.10.18.post1](https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval.html): no description found
- [Custom Response - HTML, Stream, File, others - FastAPI](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse%3E)): FastAPI framework, high performance, easy to learn, fast to code, ready for production
- [llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-opensearch/llama_index/vector_stores/opensearch/base.py at 0ae69d46e3735a740214c22a5f72e05d46d92635 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/0ae69d46e3735a740214c22a5f72e05d46d92635/llama-index-integrations/vector_stores/llama-index-vector-stores-opensearch/llama_index/vector_stores/opensearch/base.py#L249): LlamaIndex is a data framework for your LLM applications - run-llama/llama_index

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1215245881713107004) (269 messages🔥🔥):

- **Claude 3 Opus Usage Limits Discussed**: Users expressed concern about the limited number of uses for Claude 3 Opus under Perplexity's plan, with users like `@hra42` noting disappointment about the 5-use limit. The conversation circled around the cost-efficiency of Perplexity AI subscription plans, with some users debating the sufficiency of the daily message limits.
  
- **Difficulty Accessing Features for Some Users**: Multiple users, including `@netrot.` and `@laurant3855`, encountered issues when attempting to use certain features such as photo generation, using Cloude 3 Opus, and the rewrite function. `@icelavaman` and `@icelavaman` provided assistance, including suggesting to reset browser data and contacting [support@perplexity.ai](mailto:support@perplexity.ai).
  
- **Comparisons of AI Performance and Efficiency**: Various AI models were compared throughout the discussion for their performance and efficiency. Models like **Inflection-2.5** were announced by `@codelicious`, and tools such as **Sonnet** were suggested as viable options for model comparison by `@deicoon` and supported by other users like `@akumaenjeru`.
  
- **Pros and Cons of Perplexity AI and Other AI Services**: Users like `@thaholylemon` and `@hra42` evaluated the value and capabilities of Perplexity AI, comparing its services and cost against other platforms like ChatGPT Plus. Discourse centered around the benefits of source-gathering functionality and overall value for researchers and students, while others discussed their personal preferences and experiences with different subscriptions.
  
- **Subscription Elements and User Experiences Exchange**: Users were seen exchanging experiences with different AI platforms and debating the features included with premium subscriptions like Pro Discord and access to models like **Claude 3 Opus**. Some users, such as `@toby1260`, reported ambiguous experiences with the AI’s responses, leading to a discussion of prompt engineering and model limitations.
  

**Links mentioned**:

[Inflection-2.5: meet the world's best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1215216398406651934) (6 messages):

- **Perplexity AI vs. Other Platforms**: User `@oen99un` shared a [comparison](https://www.perplexity.ai/search/compare-perplexityai-and-1_XF4y8tSoKbyZOg1Krk9g#0) between Perplexity AI and another platform, highlighting differences and similarities.
  
- **Learning on Perplexity AI**: `@bluesky1911` provided a [link](https://www.perplexity.ai/search/how-to-learn-FoGL4Ir3SHeUO.is1GLZkQ) detailing methods for learning using Perplexity AI's vast resources.
  
- **Historical Innovations and Progress**: `@vishrutkmr7` shared a [link](https://www.perplexity.ai/search/have-there-been-DqbDwspUTIGtYtQqFMmK_Q) related to past innovations and the progress of civilization.
  
- **Perplexity AI's Role Understanding**: User `@croak_plonk` posted a [link](https://www.perplexity.ai/search/You-are-a-VLuvGxsQQAGdwpOfv8EfCA) that examines the concept and functionality of Perplexity AI in chatbot form.
  
- **Questions on Authorship in AI**: `@pope9870` shared a [link](https://www.perplexity.ai/search/How-do-we-wrYmycCYQFGx6QJGKiYCww#1) delving into who holds the writing credit in AI-assisted creation.
  
- **Existence of Emotion in AI**: `@bodhibios` posed a question about AI and emotions, referencing a [Perplexity AI query](https://www.perplexity.ai/search/Is-there-a-RxP0FffWQdym_rMQpnTSpA#0) exploring this concept.
  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1215269211778322493) (14 messages🔥):

- **Perplexity Discovery Feature Inquiry**: `@yankovich` asked about **Perplexity Discover**'s functionality, to which `@bitsavage.` described it as a tool for exploring new content based on user interests. `@bitsavage.` suggested checking the **Perplexity API** documentation for potential implementation.
  
- **Channel Preservation Check-in**: `@leoesq` posted a message to keep the **pplx-api** channel active, while `@po.sh` provided a tip on how to view all channels in Discord to prevent losing access in the future.
  
- **Seeking Documentation on RAG Pipeline**: `@leoesq` inquired about the documentation regarding the RAG pipeline and specific text handling used by **Sonar**, showing interest in understanding the interaction between search text and the LLM.
  
- **API Inquiry for Answer Engine**: `@ruxorly` questioned the future availability of an API for using models like **Claude/GPT4/Mistral Large** with web search capability through Perplexity API.
  
- **Clarification on Model Output Limitations**: `@brknclock1215` and `@leoesq` discussed the maximum output in tokens for models, noting that it depends on the model's context window and finetuning behavior, which significantly affects the token output size.
  

---

### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1215326440371785790) (1 messages):

- **New Benchmarks for Korean Language Models**: `@gson_arlo` announced the creation of two new Korean language evaluation datasets: [Hae-Rae Bench](https://arxiv.org/abs/2309.02706) and [K-MMLU](https://arxiv.org/abs/2402.11548). Hae-Rae Bench is accepted at LREC-COLING 2024, and KMMLU, a Korean adaptation of MMLU, is under review at ACL, with both benchmarks designed to test language models' abilities to understand Korean-specific knowledge.
- **Call for Multilingual Model Evaluation**: `@gson_arlo` highlighted the limited tooling for evaluating multilingual models, particularly for languages other than English and Chinese, and invited community members to contribute to designing benchmarks for diverse languages and cultures in the `<#1208111628051152969>` channel. They also directed those interested in model evaluation to the `<#755950983669874798>` channel.

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1215211323152142346) (39 messages🔥):

- **TensorRT Code Integration Pending Approval**: `@abhishekvijeev` informed that they, along with another user, are in the process of integrating their TensorRT code, which requires approval due to being developed with company resources.
- **Context Length in Training LLMs Discussed**: `@sentialx` and `@thooton_` discussed training large language models (LLMs) with longer versus shorter contexts, with `@thooton_` explaining the benefits of starting with shorter context lengths for more focused training before moving to longer contexts.
- **EEVE-Korean-v1.0 Introduced**: `@seungduk` shared an [arXiv technical report](https://arxiv.org/abs/2402.14714) on efficiently adding more tokens to LLMs while maintaining the original model's performance, mentioning their work on \\texttt{EEVE-Korean-10.8B-v1.0}.
- **Open Invitation for ML/AI Research Collaboration**: `@andrew_f0874` offered his background in CS, PhD from Cornell, and experience as a Google research scientist to collaborate part-time on ML/AI research, stating a broad interest especially in RL, ML privacy, ML security, and applying ML to programming/compilers.
- **Simple AI Discussions and Questions**: New member `@shida3916` sought a suitable forum to discuss everyday AI uses and ask simple questions, while `@stellaathena` suggested looking at other servers listed in <#732688974337933322> for more beginner-friendly advice.

**Links mentioned**:

- [Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models](https://arxiv.org/abs/2402.14714): This report introduces \\texttt{EEVE-Korean-v1.0}, a Korean adaptation of large language models that exhibit remarkable capabilities across English and Korean text understanding. Building on recent hig...
- [eleutherai](https://wandb.ai/eleutherai/pythia)?): Weights & Biases, developer tools for machine learning

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1215240020475248681) (107 messages🔥🔥):

- **GaLore's Memory Efficiency Draws Attention**: Users `@xylthixlm`, `@main.ai`, and others discussed the potential of GaLore, a technique that claims better results than full-rank updates with less memory usage. Skepticism about the actual gradient savings and the practicalities of its implementation were expressed, particularly due to the way gradients are handled within the optimizer.
  
- **Anticipation for GaLore Replication**: Community members such as `@ai_waifu`, `@random_string_of_character`, and `@jckwind` showed interest in seeing replication of the GaLore results. A pull request for an optimizer by `@maximegmd` on GitHub suggests that replication attempts are imminent.
  
- **Exploring GaLore's Codebase Raises Questions**: `@xylthixlm` examined GaLore's code, noting that the optimizer runs after every parameter grad update during backprop, which suggests that all gradients don't need to be stored simultaneously. Users also discussed Python's capability to index a dictionary with a PyTorch parameter, with contributions from `@_inox` and `@tulkascodes`.
  
- **CAME Optimizer Piques Curiosity**: The CAME optimizer was mentioned by `@xylthixlm` as a lesser-known tool featured in PixArt-\\Sigma; it aims to provide the speed of adaptive methods with reduced memory usage. Interest in understanding CAME's performance and comparisons with other optimizers like Adafactor and Adam was sparked.
  
- **Instruction Tuning Dataset Discussions**: `@kublaikhan1` inquired about the best instruction tuning dataset, receiving a response from `@jstephencorey` who recommended OpenAssistant and others. The importance of fine-tuning order and dataset quality was discussed, referring to a recent paper that found high-quality SFT (single fine-tuning) on GPT-4 outputs can yield results as good as or better than more complex tuning methods.
  

**Links mentioned**:

- [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692): In this paper, we introduce PixArt-Σ, a Diffusion Transformer model~(DiT) capable of directly generating images at 4K resolution. PixArt-Σrepresents a significant advancement over its predecessor, Pix...
- [CAME: Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047): Adaptive gradient methods, such as Adam and LAMB, have demonstrated excellent performance in the training of large language models. Nevertheless, the need for adaptivity requires maintaining second-mo...
- [SOCIAL MEDIA TITLE TAG](https://byte-gpt.github.io/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Pretrained-Language-Model/CAME/came.py at master · huawei-noah/Pretrained-Language-Model](https://github.com/huawei-noah/Pretrained-Language-Model/blob/master/CAME/came.py): Pretrained language model and its related optimization techniques developed by Huawei Noah's Ark Lab. - huawei-noah/Pretrained-Language-Model
- [pytorch/torch/_tensor.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_tensor.py#L1059): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch
- [A direct comparison between llama.cpp, AutoGPTQ, ExLlama, and transformers perplexities - LLM blog](https://oobabooga.github.io/blog/posts/perplexities/): no description found
- [A detailed comparison between GPTQ, AWQ, EXL2, q4_K_M, q4_K_S, and load_in_4bit: perplexity, VRAM, speed, model size, and loading time. - LLM blog](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/): no description found
- [ToDo: Token Downsampling for Efficient Generation of High-Resolution Images](https://arxiv.org/abs/2402.13573): Attention mechanism has been crucial for image diffusion models, however, their quadratic computational complexity limits the sizes of images we can process within reasonable time and memory constrain...
- [Making Large Language Models Better Reasoners with Step-Aware Verifier](https://arxiv.org/abs/2206.02336): Few-shot learning is a challenging task that requires language models to generalize from limited examples. Large language models like GPT-3 and PaLM have made impressive progress in this area, but the...
- [GitHub - jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [GaLore/torchrun_main.py at master · jiaweizzhao/GaLore](https://github.com/jiaweizzhao/GaLore/blob/master/torchrun_main.py#L356): Contribute to jiaweizzhao/GaLore development by creating an account on GitHub.
- [WIP: galore optimizer by maximegmd · Pull Request #1370 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1370): Adds support for Galore optimizers Still a WIP, untested.

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1215222634841645066) (24 messages🔥):

- **Custom Output Format Implementation for Harness**: `@pminervini` inquired about customizing output format in the harness, suggesting a two-step generation process. `@baber_` proposed a modification of the `generate_until` method and shared a [GitHub link](https://github.com/EleutherAI/lm-evaluation-harness/blob/9e6e240229429d2214bc281bed7a4e288f5169a1/lm_eval/models/huggingface.py#L1186) as a potential starting point for implementation.
  
- **MCQA Evaluation Paper Discussion**: `@nish5989` shared their paper on multiple-choice questions (MCQA) evaluation and dataset artifacts. The subsequent discussion touched on empirical results of answer format validity in the appendix and the consideration of rerunning experiments with likelihood methods.
  
- **The Question of Language-Specific Evaluation**: `@seanbethard` questioned the preference for language-specific evaluation criteria over crosslingual criteria, referencing language nuances like evidentiality and animacy, but arguing for the sufficiency of syntax and lexicon for language evaluation.
  
- **Confidence Interval Clarification**: `@yamashi` sought clarification on calculating a 95% confidence interval using the standard error of the mean (SEM). `@hailey_schoelkopf` confirmed that multiplying 1.96 times the SEM is the correct approach.
  
- **BOS Token Usage Variances**: `@jwngx` asked about the standards for using the beginning-of-sentence (BOS) token in evaluations, noting a recent change in practice. `@stellaathena` clarified that usage depends on the model, but no consolidated information exists on which models perform better with it.
  

**Links mentioned**:

- [Multiple Choice Question Standard Deviation · Issue #1524 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1524): I saw that the multiple choice type evaluation would compute the metrics along with standard deviation. From my understanding, multiple choice answer is chosen from the choice with highest probabil...
- [lm-evaluation-harness/lm_eval/models/huggingface.py at 9e6e240229429d2214bc281bed7a4e288f5169a1 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/9e6e240229429d2214bc281bed7a4e288f5169a1/lm_eval/models/huggingface.py#L1186).): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [Do Prompt-Based Models Really Understand the Meaning of their Prompts?](https://arxiv.org/abs/2109.01247): Recently, a boom of papers has shown extraordinary progress in zero-shot and few-shot learning with various prompt-based models. It is commonly argued that prompts help models to learn faster in the s...
- [Are Language Models Worse than Humans at Following Prompts? It's Complicated](https://arxiv.org/abs/2301.07085): Prompts have been the center of progress in advancing language models' zero-shot and few-shot performance. However, recent work finds that models can perform surprisingly well when given intention...

---

### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1215686529695883364) (1 messages):

- **New Member Seeking AI Knowledge**: `@shida3916` expressed enthusiasm about joining the community, looking to discuss **everyday AI applications** and seek answers to simple questions. They inquired if this Discord server is the appropriate place for such discussions.

---

### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1215280118465822771) (102 messages🔥🔥):

- **Exploring Environment Setup Options**: `@biiter` and `@tastybucketofrice` discussed the intricacies of setting up environments for GPT-NeoX development, pondering over using Docker versus local system setups and acknowledging the complexity of dependency management. The idea of consolidating the environment setup was proposed by `@catboy_slim_` to ensure one correct way to prepare the development environment.
  
- **NGC Container Contemplations**: `@tfidia` introduced the use of NVIDIA NGC PyTorch container to ease the struggles with setting up Apex and CUDA dependencies, and offered details on dependencies pre-installed within these containers. `@catboy_slim_` acknowledged the benefits but also expressed caution over potential reproducibility issues when moving outside of the containerized environment.
  
- **Dependency Management Discussions**: The conversation moved towards managing dependencies more effectively, with `@catboy_slim_` suggesting a move to poetry for deterministic package management, while also considering the current dependency state and setup instructions. There was recognition of the usefulness of NGC containers, but also the challenges they might introduce due to pre-installed and pre-updated packages like Flash Attention.
  
- **Flash Attention Update Conundrum**: `@catboy_slim_` pointed out concerns about version inconsistencies with Flash Attention when provided in pre-built containers like those from NGC. `@tfidia` advised on how to manually update Flash Attention, and the ongoing discussion acknowledged pytorch version specifications and the potential need for precision in dependency management.
  
- **ProtoBuf Dependency Mystery Solved**: `@hailey_schoelkopf` and `@catboy_slim_` hashed out the need for the ProtoBuf dependency installation, deducing that it might be required for SentencePiece usage within Llama's tokenizer, illustrating the complexity in pinpointing dependency origins. The exchange highlights the importance of documenting dependency reasons in dynamic development environments.
  

**Links mentioned**:

- [PyTorch Release 24.02 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html): no description found
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.
- [Cleaner dockerfile: Remove already installed deps by tf-nv · Pull Request #1175 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1175/files): Cleaning up the Dockerfile after the ngc pytorch switch (#1170): Eliminate already installed apt packages sparse attn requirement lead to a triton downgrade flash attn is already part of the ngc c...
- [PyTorch Release 24.02 - NVIDIA Docs](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-02.html#rel-24-02): no description found

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1215222357912715344) (94 messages🔥🔥):

- **Sonnet vs ChatGPT**: `@aaron_speedy` highlighted that **Sonnet** is a stronger free model than **ChatGPT 3.5**, noting features like image upload and queries if it supports file uploads.
- **Anticipating GPT-4**: Both `@futuremachine` and `@drinkoblog.weebly.com` are looking forward to the release of **GPT-4**, especially since competitors like **Claude 3** seem to be outperforming current models.
- **Prompt Optimization Debate**: `@.vilden` mentioned that overly verbose prompts can limit model performance, advising users to learn effective prompting to see fewer limitations with **GPT 3.5**.
- **Seeking AI for Flashcard Creation**: `@khaledoo.` inquired about an AI tool that can convert lecture PDFs into flashcards, sparking interest from others like `@glamrat` and questions about content accuracy from `@dezuzel`.
- **Encountering GPT-4 Issues**: `@spikyd` reported that **GPT-4** has been repeating answers and providing responses in incorrect languages, voicing frustration about the service quality, which led to a discussion with `@dojan1` about potential workarounds and reasons for these anomalies.

**Links mentioned**:

[GitHub - Kiddu77/Train_Anything: A repo to get you cracking with Neural Nets .](https://github.com/Kiddu77/Train_Anything): A repo to get you cracking with Neural Nets . Contribute to Kiddu77/Train_Anything development by creating an account on GitHub.

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1215374129944068096) (38 messages🔥):

- **No GPTs over API for now**: `@solbus` clarified that **GPTs** are exclusive to ChatGPT and cannot be accessed via the OpenAI API in response to `@cliffsayshi`'s query about using custom GPTs like Human Writer-Humanizer-Paraphraser (Human GPT) via the API.
- **Understanding OpenAI's offering**: In the discussion with `@cliffsayshi`, `@solbus` explained that the entities available through the OpenAI API such as DALL-E, and the Babbage and Davinci models are not referred to as GPTs but as "models," with GPTs being a specific feature of ChatGPT.
- **ChatGPT Access Issues Addressed**: Users `@pteromaple`, `@bluesdante`, `@aialra`, and `@cypriang` found that changing the language settings to "Auto-detect" and refreshing (F5) resolved issues with ChatGPT not responding in browsers.
- **Language Settings Bug**: `@joachimpimiskern` and `@pteromaple` reported and confirmed an ongoing issue with language settings in ChatGPT, where using English resolved the problem, but switching to other languages could cause the interface to break again.
- **Localized ChatGPT Troubleshooting**: `@meteopx` mentioned that using a VPN allowed messages to send through ChatGPT, highlighting localized technical concerns regarding the accessibility of the service in different regions.

---

### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1215291894012186664) (54 messages🔥):

- **Roleplay Prompt Guidance Questioned**: `@loamy_` discusses the best way to instruct AI for roleplay while `@dezuzel` recommends providing positive instructions on actions the AI should perform, rather than mentioning what it shouldn't do. They suggest being explicit about AI reactions to achieve the desired roleplay effect.
- **Random Seeds in GPT**: `@interactiveadventureai` queries whether GPT can use a different seed for random number generation each iteration to enhance an interactive adventure. `@solbus` recommends using Python via data analysis features for generating randomness, clarifying that control over the underlying model's seed isn't available to users.
- **Combatting Narrative Overlays in Outputs**: `@interactiveadventureai` seeks advice to eliminate unwanted narrative summaries in the AI's responses, and `@eskcanta` suggests altering the writing style in the prompts as a possible solution. A touch of humor is added with a playful mention of drastic measures against servers.
- **New Member Intro**: `@thebornchampion` introduces themselves to the community, expressing their enthusiasm for prompt engineering and discussing their use of GPT for data analytics and various personal projects, like planning a trip and academic support.
- **GPT Classifier for Conversation Closure**: `@chemlox` discusses building a GPT classifier to decide if an agent-consumer conversation should be closed, contemplating between using a react-based agent or fine-tuning GPT with training data. `@eskcanta` recommends testing the base model first to save effort and resources.
- **Organic Dialogue and Custom Instructions**: `@feedonyourtearskappa` seeks advice on creating more organic dialogue without repetitive phrases, while `@openheroes` highlights the "Customize ChatGPT" feature to set instructions for a more natural writing style, including mimicry of specific text examples.
- **Professional Headshots with DALL-E**: `@elhadrami.oussama` expresses interest in generating professional headshots using DALL-E and seeks insights, but `@enkai3526` responds with a humorous comment related to gaming.

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1215291894012186664) (54 messages🔥):

- **Prompt Engineering for Roleplay**: User `@loamy_` discussed how to formulate prompts for roleplay scenarios, considering whether to instruct the AI never to claim it's an assistant. `@dezuzel` recommended focusing on what the AI should do rather than what it shouldn't.
- **Random Seed Selection Solved**: `@interactiveadventureai` sought advice on having GPT select a different seed for random number generation, considering using timestamps. `@solbus` suggested using Python's built-in random functions in the context of the data analysis feature.
- **Optimizing Narrative Responses**: `@interactiveadventureai` expressed frustration over the AI's tendency to provide narrative summaries and a certain style of dialogue. `@eskcanta` shared guidance on prompt engineering to steer GPT into different writing styles.
- **Building a GPT Classifier for Conversations**: User `@chemlox` asked for advice on creating a GPT classifier to assess if user-agent conversations are resolved. `@eskcanta` advised checking GPT's base model performance before deciding on further actions.
- **Crafting More Organic Dialogue Responses**: `@feedonyourtearskappa` inquired about prompting the AI to produce natural dialogue without repetition. `@openheroes` suggested using the "Customize ChatGPT" feature to guide the model toward the desired output.

---

### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1215416422071533658) (1 messages):

- **RAG Demo Live**: `@bishmoy` shared a RAG demo for searching Arxiv CS papers, accessible at [HuggingFace Spaces](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG).
- **Novel Protein Anomaly Detection**: `@403280164433297409` released a paper on detecting anomalous proteins using deep representations, announcement was accompanied by a [Twitter link](https://twitter.com/danofer/status/1763962202472484991).
- **Fine-tuning Gemma with ChatML**: `@817334594075623435` provided a finetuning demonstration of Google's Gemma LLM with a notebook now available at [HuggingFace Spaces](https://huggingface.co/Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101).
- **Illuminating LLM Insights**: `@1120804749273477242` authored a blog post discussing the need to move beyond conversation to meaningful AI actions, linked on [LinkedIn](https://www.linkedin.com/pulse/action-all-you-need-moving-beyond-conversation-ai-vishal-mysore-sukfc/?utm_source=share&utm_medium=member_android&utm_campaign=share_via).
- **Cutting-edge LLM Interface in Rust**: An interface using HuggingFace/Candle among others has been built entirely in Rust, showcased in a video by `@538229308678733851`, while `@282727276733399041` introduced a new 16k context pretrained encoder model available at [HuggingFace](https://huggingface.co/BEE-spoke-data/mega-encoder-small-16k-v1).

**Links mentioned**:

- [Arxiv CS RAG - a Hugging Face Space by bishmoy](https://huggingface.co/spaces/bishmoy/Arxiv-CS-RAG): no description found
- [Andyrasika/Gemma-ChatML · Hugging Face](https://huggingface.co/Andyrasika/Gemma-ChatML): no description found
- [Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101 · Hugging Face](https://huggingface.co/Andyrasika/vit-base-patch16-224-in21k-finetuned-lora-food101): no description found
- [Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz): no description found
- [UDOP DocVQA - a Hugging Face Space by RamAnanth1](https://huggingface.co/spaces/RamAnanth1/udop-vqa): no description found
- [Yi 9B - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/Yi-9B): no description found
- [BEE-spoke-data/mega-encoder-small-16k-v1 · Hugging Face](https://huggingface.co/BEE-spoke-data/mega-encoder-small-16k-v1): no description found
- [Andyrasika/lora_gemma · Hugging Face](https://huggingface.co/Andyrasika/lora_gemma): no description found
- [Locutusque/UltraTextbooks-2.0 · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/UltraTextbooks-2.0): no description found
- [Mistral-ChatBot-Arena - a Hugging Face Space by rwitz](https://huggingface.co/spaces/rwitz/Mistral-ChatBot-Arena): no description found
- [GitHub - treebeardtech/treebeard-kubeflow: 🪐 scale Jupyter in Kubernetes](https://github.com/treebeardtech/terraform-helm-kubeflow): 🪐 scale Jupyter in Kubernetes. Contribute to treebeardtech/treebeard-kubeflow development by creating an account on GitHub.
- [Large Language Models in Quest for Adventure](https://huggingface.co/blog/crazyjeannot/llms-mapping-adventure): no description found

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1215209812191739924) (142 messages🔥🔥):

- **Clash Over Proposed AI Safety Czar**: Concerns surfaced about the anticipated appointment of [Paul Christiano](https://paulfchristiano.com/) to the US AI Safety Institute, causing an internal crisis at the National Institute of Standards and Technology with staff threats of resignation and a revolt. The [article](https://venturebeat.com/ai/nist-staffers-revolt-against-potential-appointment-of-effective-altruist-ai-researcher-to-us-ai-safety-institute/) details the conflict and Christiano's controversial views on AI's potential for causing existential risks.
- **AI for Optimizing Code**: `@techintermezzo` sought advice on the best AI model for optimizing shader code, prompting discussions on models like DeepSeek-Coder instruct and resources like the [OpenCodeInterpreter paper](https://arxiv.org/abs/2402.14658). The [code processing review work](https://arxiv.org/abs/2311.07989) breaks down current advancements, helping those interested in understanding and utilizing AI for programming tasks.
- **Exploring AI-Enhanced Geopolitics**: In a lengthy discussion about the potential of AI in global strategies, `@acidgrim` and others debated the contrasting approaches of Western and Chinese AI, touching on topics from censorship to potential applications in asymmetric warfare. The debate covered implications of unrestricted AI, AI training data concerns, and potential regulations.
- **Prompt Engineering for RAG**: `@jeffry4754` inquired about the standard term for preprocessing a question into sub-questions for Retrieval-Augmented Generation (RAG), suggesting "multi-hop question-answering task" might be the title for such a technique. The conversation continued without a clear consensus or reference for a standard term.
- **Stable Diffusion Query**: User `@maycolrox` requested assistance with loading models in the diffusers library pertaining to stable diffusion, implying a problem with a model called loras. No direct solution was offered within the given messages.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [Inflection-2.5: meet the world's best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Repeat After Me: Transformers are Better than State Space Models at Copying](https://arxiv.org/abs/2402.01032): Transformers are the dominant architecture for sequence modeling, but there is growing interest in models that use a fixed-size latent state that does not depend on the sequence length, which we refer...
- [NIST staffers revolt against expected appointment of ‘effective altruist’ AI researcher to US AI Safety Institute](https://venturebeat.com/ai/nist-staffers-revolt-against-potential-appointment-of-effective-altruist-ai-researcher-to-us-ai-safety-institute/): NIST faces turmoil as staff consider quitting over Paul Christiano's expected appointment to a role at the US AI Safety Institute, sources say.
- [Deploying 🤗 Hub models in Vertex AI](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai): no description found
- [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/abs/2402.14658): The introduction of large language models has significantly advanced code generation. However, open-source models often lack the execution capabilities and iterative refinement of advanced systems lik...
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers): no description found
- [Haiper | Generative AI For Video Content Creation](https://haiper.ai/): Video creation AI products crafted to empower individuals in creatively expressing themselves.
- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989): In this work we systematically review the recent advancements in code processing with language models, covering 50+ models, 30+ evaluation tasks, 170+ datasets, and 700+ related works. We break down c...
- [Deploying 🤗 Hub models in Vertex AI](https://huggingface.co/blog/alvarobartt/deploy-from-hub-to-vertex-ai#model-upload): no description found
- [Federal Register :: Request Access](https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious): no description found
- [Regulations.gov](https://www.regulations.gov/document/NTIA-2023-0009-0001): no description found
- [My views on “doom” — LessWrong](https://www.lesswrong.com/posts/xWMqsvHapP3nwdSW8/my-views-on-doom): I’m often asked: “what’s the probability of a really bad outcome from AI?” …

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1215571025186521118) (4 messages):

- **Interest in AI for Data Analytics**: `@umbreenh` expressed a desire to learn about using **generative AI** for data analytics development and welcomed any assistance or pointers in this domain.
- **Collaborative Learning Spirit**: `@yasirali1149` responded to `@umbreenh` with an interest to **learn together** about generative AI applications in data analytics.
- **Ready to Join the Learning Venture**: `@kenngala` addressed `@Singhaditya4333` (who hasn't written in the provided messages), indicating their readiness to engage and collaborate in the learning process.

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1215271464782008400) (5 messages):

- **"Let's Build GPT" Educational Resource Shared**: `@kurtfehlhauer` recommended an introductory video to GPT construction - a thorough walkthrough titled ["Let's build GPT: from scratch, in code, spelled out."](https://www.youtube.com/watch?v=kCc8FmEb1nY) The video explains the creation of a Generative Pretrained Transformer following OpenAI's papers.
  
- **Spotlight on Hugging Face's Task Page**: `@andysingal` expressed enthusiasm for Hugging Face's [Machine Learning tasks portal](https://huggingface.co/tasks), which lists resources like demos, use cases, models, datasets across various tasks in computer vision and other domains.
  
- **A Gentle Note on Resource Familiarity**: In response to @andysingal's post about the tasks page, `@cakiki` pointed out that the resource isn't new and credited `@697163495170375891` for their longstanding efforts on the platform.
  
- **Discovery is Personal**: Continuing the dialogue, `@andysingal` clarified that the tasks page was new to him and hence his excitement.
  
- **Qwen-Agent Empowers AI Developers**: `@andysingal` highlighted the capabilities of Qwen-Agent, an AI framework that integrates instruction following, tool usage, planning, and memory in LLMs, in a detailed [Medium article](https://medium.com/ai-advances/unleashing-the-power-of-qwen-agent-revolutionizing-ai-assistance-with-rag-application-a19feecf38bb) titled "Unleashing the Power of Qwen-Agent: Revolutionizing AI Assistance with RAG Application."
  

**Links mentioned**:

- [Tasks - Hugging Face](https://huggingface.co/tasks): no description found
- [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections t...
- [Unleashing the Power of Qwen-Agent: Revolutionizing AI Assistance with RAG Application](https://medium.com/ai-advances/unleashing-the-power-of-qwen-agent-revolutionizing-ai-assistance-with-rag-application-a19feecf38bb): Ankush k Singal

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1215225192025030727) (13 messages🔥):

- **Gemma Takes on ChatML**: `@andysingal` shared their work on fine-tuning **Gemma**, Google's LLM, with ChatML, demonstrating this with a [model card](https://huggingface.co/Andyrasika/Gemma-ChatML) and acknowledging @philschmid’s tokenizer.
  
- **Recap of AI x Web3 at ETHDenver**: `@aerophilian` penned a recap of ETHDenver, highlighting Web3 and AI intersection and shared a [blog post](https://www.spatialawareness.net/p/ethdenver-recap-emerging-trends-in?utm_source=activity_item) with insights and YouTube links for conference talks.
  
- **Searching via CLIP Index Just Got Easier**: `@robbesneyders` introduced a CLIP index for the Datacomp-12.8M dataset, facilitating prompt-based searches, and pointed to their team's method and outputs on the [Hugging Face Hub](https://huggingface.co/datasets/fondant-ai/datacomp-small-clip) and a [blog post](https://fondant.ai/en/latest/blog/2024/03/05/building-a-datacomp-clip-index-with-fondant/#with-fondant) for more details.
  
- **Fine Dining with AI**: `@chongdashu` built a restaurant name and menu generator app in under 100 lines of Python, showcasing LangChainAI and Gradio, complete with a [Medium article](https://medium.com/@chongdashu/langchain-and-gradio-d23c5e9cee90), live [demo](https://huggingface.co/spaces/chongdashu/langchain-crash-course-gradio), and full [source code](https://github.com/chongdashu/langchain-crash-course/tree/lesson-1).
  
- **Legal Precedents at Your Fingertips**: `@conceptofmind` announced the release of over 6.6 million state and federal court decisions, a collaborative effort supported by the Caselaw Access Project and Harvard Library Innovation Lab, with datasets and embeddings available for use, as mentioned in an [update](https://x.com/EnricoShippole/status/1766157358672359862?s=20) by @EnricoShippole and acknowledged additional help from `<@274244546605613056>`.
  

**Links mentioned**:

- [Doodle Wars](https://doodlewars.netlify.app): no description found
- [Andyrasika/Gemma-ChatML · Hugging Face](https://huggingface.co/Andyrasika/Gemma-ChatML): no description found
- [ETHDenver Recap: Emerging Trends in web3 and AI](https://www.spatialawareness.net/p/ethdenver-recap-emerging-trends-in?utm_source=activity_item): Where we're at, where we're heading, and the return of Kevin.
- [Tweet from Enrico Shippole (@EnricoShippole)](https://x.com/EnricoShippole/status/1766157358672359862?s=20): @TeraflopAI is excited to help support the @caselawaccess and @HarvardLIL, in the release of over 6.6 million state and federal court decisions published throughout U.S. history.
- [Building a Datacomp CLIP index with Fondant - Fondant](https://fondant.ai/en/latest/blog/2024/03/05/building-a-datacomp-clip-index-with-fondant/#with-fondant).): no description found
- [Langchain Crash Course (Gradio) - a Hugging Face Space by chongdashu](https://huggingface.co/spaces/chongdashu/langchain-crash-course-gradio): no description found
- [GitHub - chongdashu/langchain-crash-course at lesson-1](https://github.com/chongdashu/langchain-crash-course/tree/lesson-1): Contribute to chongdashu/langchain-crash-course development by creating an account on GitHub.

---

### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1215302782731419688) (45 messages🔥):

- **Seeking Guidance for llamas2 Chatbot**: User `@neerajjulka1986` requested resources for an end-to-end chatbot project using the opensource model llamas2. `@chad_in_the_house` recommended checking out resources for finetuning and deployment on GitHub, including [PEFT](https://github.com/huggingface/peft) and [text-generation-inference](https://github.com/huggingface/text-generation-inference), and mentioned TRL for reinforcement learning at [TRL GitHub](https://github.com/huggingface/trl).
  
- **Gathering for Gemini 1.5 Pro Overview**: `@shashank.f1` announced a meeting on Sparse MOEs and Gemini 1.5 Pro, providing a Zoom link and indicating an overview would take place. They also shared [Jeremy Howard's tweet about Sparse MOEs](https://twitter.com/jeremyphoward/status/1765868543235805232?t=KY1CvJ2j3fyuAwEWGYtSNQ) as a resource.
  
- **Gemini 1.5 Pro Discussion Recording Shared**: `@shashank.f1` posted a YouTube video link to the earlier Gemini 1.5 Pro discussion and Sparse Mixture of Experts Model, which can be found [here on YouTube](https://youtu.be/IuehDA1M_Lw).
  
- **Understanding Mixture of Experts (MoEs)**: `@chad_in_the_house` recommended a blog post from Hugging Face to understand MoEs, accessible [here](https://huggingface.co/blog/moe). Additionally, `@shashank.f1` explained that the VRAM requirement for tuning MoEs with QLoRA goes up, making it impractical on a single GPU but viable with multiple GPUs, and shared a library for implementing this at [fsdp_qlora](https://t.co/qcyEa7EGGY).
  
- **Meeting times and recordings**: Users asked about meeting times and recording availability. `@chad_in_the_house` confirmed that meetings might be planned for the weekend and indicated that recordings should be posted by `@shashank.f1`.
  

**Links mentioned**:

- [Join our Cloud HD Video Meeting](https://us06web.zoom.us/j/82222903768?pwd=g9GXLBBgIad5CaXJm0qMJ2Zuc1KhHc.1): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
- [Join our Cloud HD Video Meeting](https://us06web.zoom.us/j/82222903768?pwd=g9GXLBBgIa): Zoom is the leader in modern enterprise video communications, with an easy, reliable cloud platform for video and audio conferencing, chat, and webinars across mobile, desktop, and room systems. Zoom ...
- [Gemini 1.5 Pro: Unlock reasoning and knowledge from entire books and movies in a single prompt](https://youtu.be/IuehDA1M_Lw): 🚀 Dive into the world of AI with Gemini 1.5! 🌟In this video, we unpack the magic behind Gemini's sparse mixture of experts architecture, perfect for unleas...
- [GitHub - huggingface/trl: Train transformer language models with reinforcement learning.](https://github.com/huggingface/trl): Train transformer language models with reinforcement learning. - huggingface/trl
- [GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP](https://t.co/qcyEa7EGGY): Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.
- [Mixture of Experts Explained](https://huggingface.co/blog/moe): no description found
- [GitHub - huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.](https://github.com/huggingface/peft): 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. - huggingface/peft
- [GitHub - huggingface/text-generation-inference: Large Language Model Text Generation Inference](https://github.com/huggingface/text-generation-inference): Large Language Model Text Generation Inference. Contribute to huggingface/text-generation-inference development by creating an account on GitHub.

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1215264290467094609) (1 messages):

- **Seeking Guides on Merging SDXL with LoRA**: `happy.j` is looking for resources or guides on how to merge **sdxl-lightning LoRA** with a standard SDXL model, pointing to a [discussion](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104) where more information on the procedure would be appreciated.
- **ByteDance Weighs in on SDXL-Lightning Techniques**: The `ByteDance org` suggests training a regular **SDXL model** on your dataset before applying SDXL-Lightning LoRA for acceleration, and for best compatibility, to train SDXL as LoRA from the start.
- **Advanced Training Tips from ByteDance**: For those seeking higher quality, `ByteDance org` recommends merging SDXL-Lightning LoRA onto your model and then training, while noting that using MSE loss may dilute acceleration benefits. The most advanced method involves merging and then using an adversarial objective, as described in the SDXL-Lightning paper.

**Links mentioned**:

[ByteDance/SDXL-Lightning · finetune](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104): no description found

---

### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1215277600851759164) (9 messages🔥):

- **Normalization Conundrum in Data Processing**: `@huzuni` raised a topic about the effectiveness of normalization, stating they have noticed little to no impact on their data metrics with various normalization methods like **imagenet norm**, **channel wise norm**, and **min-max norm**. They inquired whether there are studies on the actual effects of normalization or explanations for its potential lack of utility.
  
- **Seeking Ultralytics Alternatives for Commercial Use**: `@prod.dopamine` prompted discussion for alternatives to **ultralytics**, expressing discontent with the AGPL license for commercial applications. They are looking for options that offer ease of use like Ultralytics but are also suitable for commercial use.
  
- **Yolov4 Suggested as a Viable Alternative**: In response to @prod.dopamine, `@toni_alright` suggested **Yolov4** as an alternative due to its different license that is more suitable for **commercial use**. The implication is that Yolov4 could be used as an Ultralytics alternative that conforms to commercial license requirements.
  
- **Darknet Implementation of Yolov4 Clarification**: Following the suggestion, `@prod.dopamine` asked whether the recommended **Yolov4** was the **darknet implementation** or another one, indicating a need for clarity on the specific alternative being proposed.
  
- **Call for AI Co-Study Partners**: `@nobita_nobii_` put out a call for a **co-study partner** in AI, which led to an affirmative response from `@prod.dopamine`. This indicates community interest in collaborative learning within the channel.
  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1215210455769808916) (18 messages🔥):

- **DeBERTa Pre-Training Hurdles**: `@henkiespenkie22` is facing issues pre-training **DeBERTa** from Microsoft, with current implementations like camemdeberta not working for them. `@grimsqueaker` responded, highlighting that Electra pretraining is not supported in HuggingFace, presenting a challenge for users seeking to pre-train these models.
  
- **Retriever Puns Galore**: After `@.sgp` asked what a retriever is, `@cakiki` shared a humorous [Golden Retriever GIF](https://tenor.com/view/golden-retriever-dog-puppy-gif-26065357), and `@lucnzz` quipped about a retriever being "a dog that catches embeddings."
  
- **Choosing the Right Language Model for Colab**: `@iloveh8` inquired about recommendations for small to medium open-source language models suitable for Google Colab, leading to suggestions from `@cursorop` who mentioned any 2b model and flan T5, while `@lucnzz` proposed any small quantisized 4-bit models.
  
- **Herculean Task on a Raspberry Pi 4**: `@verdagon` humorously contemplated the idea of running a 70B model on a Raspberry Pi 4, even if it meant 40 minutes per token.
  
- **Mapping Attention Weights Challenge**: `@komorebi6466` sought advice on how to map attention weights to each word in a sentence using BertModel for sentiment analysis, wanting to convert the attention output to a list with a specific shape. `@darwinanim8or` requested to see their code, offering a code snippet that demonstrates a similar process for a classifier based on DeBERTa.
  

**Links mentioned**:

[Golden Retriever Dog GIF - Golden Retriever Dog Puppy - Discover & Share GIFs](https://tenor.com/view/golden-retriever-dog-puppy-gif-26065357): Click to view the GIF

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1215264290467094609) (1 messages):

- **Seeking Guidance on SDXL-Lightning LoRA Merging**: User `happy.j` is looking for assistance on how to merge **SDXL-Lightning LoRA** with a standard **SDXL** model, expressing difficulty in finding resources beyond a [HuggingFace discussion thread](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104).
- **Expert Recommendations for SDXL Variants**: A member from the **ByteDance organization** recommends first training a regular **SDXL** model and then applying **SDXL-Lightning LoRA** for acceleration. For compatibility, training the SDXL with LoRA from the outset is preferred.
- **Advanced Training Approaches for SDXL-Lightning LoRA**: For quality improvements, ByteDance suggests merging **SDXL-Lightning LoRA** with the user's model and training further, cautioning that using MSE loss could dilute acceleration benefits. Employing an adversarial objective during training is considered the most advanced strategy, following the SDXL-Lightning paper's approach.

**Links mentioned**:

[ByteDance/SDXL-Lightning · finetune](https://huggingface.co/ByteDance/SDXL-Lightning/discussions/11#65de29cdcb298523e70d5104): no description found

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1215218631198113813) (113 messages🔥🔥):

- **AI Image Generator Alternatives Buzz**: `@lunsei` asked about alternatives to Sora being developed. `@thejonasbrothers` humorously replied that numerous projects are in the works, with `@pseudoterminalx` adding that many are following MagViT2 as a base.
- **Marketing Spend Mayhem Uncovered!**: `@pseudoterminalx` revealed shocking marketing expenditure figures, with $7,099 spent per conversion for mere $100 sales, invoking criticism and disbelief among community members. The conversation touched on gross inefficiencies and the need for better campaign strategies.
- **Midjourney Users Alarmed by Scraping**: Some users in the discussion, such as `@pseudoterminalx`, chuckled over Midjourney members panicking about a 'security issue' regarding their AI-generated images being scraped. Meanwhile, `@mfcool` and `@chad_in_the_house` talked about the simplicity of accessing these images and a leaked artist list used by Midjourney.
- **SD3 Discussed, Diffusers Updates Anticipated**: `@thejonasbrothers` shared news about upcoming invites to use SD3 on Discord and hinted at contributions to the Diffusers project.
- **Ideogram AI Test Skepticism**: `@pseudoterminalx` voiced skepticism regarding the claimed superiority of Ideogram AI compared to SDXL, sharing disappointment in trying to generate decent images and raising questions about the credibility of blind test results.

**Links mentioned**:

- [What Luddites can teach us about resisting an automated future](https://www.technologyreview.com/2024/02/28/1088262/luddites-resisting-automated-future-technology/): Opposing technology isn’t antithetical to progress.
- [360° Panorama Viewer Online](https://renderstuff.com/tools/360-panorama-web-viewer/): Online Panorama 360 Viewer. An easy way to View & Share 360-degree pictures for free. VR ready. 360 image viewer instantly creates interactive full-screen immersive VR spherical 360 3d panoramas i...
- [Database of 16,000 Artists Used to Train Midjourney AI, Including 6-Year-Old Child, Garners Criticism](https://www.artnews.com/art-news/news/midjourney-ai-artists-database-1234691955/): Artists included Warhol, Picasso, Cezanne, van Gogh, Anish Kapoor, Yayoi Kusama, Gerhard Richter, Frida Kahlo, and Banksy.

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1215341233530994708) (83 messages🔥🔥):

- **SVD Update Slows Down Training**: User `@metal63` mentioned experiencing significant delays when performing SVD updates with **Stable Cascade**, leading to a 2-minute pause in the whole training process.
- **Inefficiency of LLMs Challenged**: `@mkaic` strongly critiqued the parameter inefficiency in large language models (LLMs) and opened a discussion on the potential for breakthroughs in training more efficient sparse/small networks, sparking a lively debate with `@recviking` and `@thejonasbrothers`.
- **LLMs and the Compression Challenge**: `@mkaic` posited that current LLMs do not optimally compress training data and suggested that there's significant room for improving the architectures and training methods to better utilize parameters.
- **PixArt Sigma Debuts**: `@thejonasbrothers` shared about a new 4K PixArt named PixArt Sigma, providing a [link to the project](https://pixart-alpha.github.io/PixArt-sigma-project/) along with several sample images, and noted that it still has issues with text due to using only 600m parameters.
- **Discussing the Nature of Pruning**: A series of exchanges between `@recviking`, `@thejonasbrothers`, and `@mkaic` explored the limits and implications of model pruning and generalizability, with commentary on the current state of model efficiency. `@thejonasbrothers` referred to a [new paper](https://arxiv.org/pdf/2403.04692.pdf) in their discussion.

**Links mentioned**:

- [PIXART-Σ:Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://pixart-alpha.github.io/PixArt-sigma-project/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Neverseenagain Yourleaving GIF - Neverseenagain Yourleaving Oh - Discover & Share GIFs](https://tenor.com/view/neverseenagain-yourleaving-oh-no-he-gif-10093833): Click to view the GIF

---

### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1215371537415151707) (3 messages):

- **Early Sneak Peek at 'Nitro' Models**: `@alexatallah` alerted users to the appearance of new "nitro" models which are safe to use and build with, despite the possibility of minor changes before an official announcement.
  
- **Introducing Nitro Models and Extended Contexts**: `@alexatallah` excitedly introduced **Nitro models**, including Mixtral, MythoMax, and Llama 70B, which feature a new Nitro variant button and are powered by **Groq** and other providers. Additionally, context-extended models are now available, with Mixtral expanding to 732,768 context ([OpenRouter Models](https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct:nitro)), and a dedicated [video demonstration](https://openrouter.ai) showcases the models' improved speed and cost-effectiveness.
  
- **Developer Features and Dynamic Routing**: New developer features are highlighted, including performance timelines, JSON mode, and dynamic routing. Early users are invited to check out the [documentation](https://openrouter.ai/docs#provider-routing) for detailed information.
  
- **OpenRouter's Path to Model Selection and Use**: `@alexatallah` explains that OpenRouter helps in selecting models based on price and performance metrics, standardized APIs for easy switching between models, and upcoming features that include usage-based comparison and OAuth capabilities for user-choice models. Details can be found in the [documentation and rankings](https://openrouter.ai/docs#provider-routing).
  
- **Mistral 7b 0.2 Goes Nitro**: `@alexatallah` reveals the latest Nitro model, **Mistral 7b 0.2**, noting its significant speed increase (up to 20x for long outputs), and an expanded context limit of 32k. A live demo is available on [Twitter](https://twitter.com/OpenRouterAI/status/1766147110443909184).
  

**Links mentioned**:

- [Mixtral 8x7B Instruct (nitro) by mistralai | OpenRouter](https://openrouter.ai/models/mistralai/mixtral-8x7b-instruct:nitro): A pretrained generative Sparse Mixture of Experts, by Mistral AI, for chat and instruction use. Incorporates 8 experts (feed-forward networks) for a total of 47 billion parameters. Instruct model fin...
- [OpenRouter](https://openrouter.ai/docs#provider-routing%60): Build model-agnostic AI apps

---

### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1215222542282002482) (109 messages🔥🔥):

- **Model Comparison and Performance**: `@filth2` highlighted that **Sonnet** offers an impressive price-performance ratio, with costs as low as .03 for "5k context and 1200 response length," making it a valuable option compared to other models. Meanwhile, `@phoshnk` and `@mka79` debated the subtle differences and cost-effectiveness between **Opus** and **Sonnet**, with a general consensus on **Sonnet** being more affordable.
  
- **Moderation Layer Confusion Clarified**: `@filth2`, `@spaceemotion`, and `@alexatallah` discussed the nuances of moderation in models offered by OpenAI, Anthropic, and OpenRouter. It was clarified that **OpenRouter** applies an additional layer of moderation, which could lead to more refusals compared to using the **OpenAI** or **Anthropic** API directly.
  
- **Data Retention and Training Practices Inquired**: `@mka79` raised questions about **Anthropic's** use of customer content in model training. `@spaceemotion` shared links to Anthropic's support articles, leading to the understanding that content from paid services may not be used for training.
  
- **Anthropic Endpoint Clarifications by Alex Atallah**: `@alexatallah` illuminated how **Anthropic** moderates content specifically for **OpenRouter** self-moderated requests, which includes a server-side classifier and transformer affecting the responses. Users engaging directly with Anthropic's API may not have an additional moderation layer, but risk facing repercussions without a proper moderation strategy in place.
  
- **Discussions on Nitro Models and Pricing Insights**: Users like `@starlord2629`,`@xiaoqianwx`, and `@louisgv` talked about **Nitro models**, particularly their higher throughput and different pricing, with Groq now powering **Mixtral 8x7b instruct nitro** at a cost of 0.27/1M tokens. Users expressed optimism and interest around these developments.
  

---

### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1215221224158797896) (6 messages):

- **Memes Incoming**: User `@iron_bound` expressed a strong desire to post memes, encouraged by `@marksaroufim` who directed them to post in a specific memes channel.
- **Flash Attention in CUDA Shared**: `@tspeterkim_89106` shared a project implementing Flash Attention using CUDA ([Flash Attention in ~100 lines of CUDA](https://github.com/tspeterkim/flash-attention-minimal)) and opened the floor for feedback and discussion about flash attention implementations.
- **CUDA Explained in a Flash**: `@iron_bound` shared a [YouTube video](https://www.youtube.com/watch?v=pPStdjuYzSI) titled "Nvidia CUDA in 100 Seconds", summarizing what CUDA is and its role in AI development.
- **Nvidia's Slick Marketing Move Noticed**: `@iron_bound` commented on the Nvidia's strategy of featuring a 4090 graphics card in a video mentioning Nvidia's GPU Technology Conference (GTC), with `@apaz` acknowledging the observation.

**Links mentioned**:

- [Nvidia CUDA in 100 Seconds](https://www.youtube.com/watch?v=pPStdjuYzSI): What is CUDA? And how does parallel computing on the GPU enable developers to unlock the full potential of AI? Learn the basics of Nvidia CUDA programming in...
- [GitHub - tspeterkim/flash-attention-minimal: Flash Attention in ~100 lines of CUDA (forward pass only)](https://github.com/tspeterkim/flash-attention-minimal): Flash Attention in ~100 lines of CUDA (forward pass only) - tspeterkim/flash-attention-minimal

---

### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/) (1 messages):

marksaroufim: do you see where the link to join that meetup is?

---

### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1215270303811371068) (78 messages🔥🔥):

- **Coarsening May Hit Throughput Ceiling**: `@cudawarped` suggested that coarsening might not improve performance on a workload already at 93% memory throughput, implying a performance ceiling had been reached.
- **CuTe DSL Studying for Better Understanding of FlashAttention**: `@ericauld` discussed studying the **CuTe DSL** as it is used in the NVIDIA **[FlashAttention repository](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)**, indicating that it is necessary for optimizing tensor core utilization.
- **Discovering Dequantization Speed**: `@zippika` shared a dequantize implementation using the `cuda::pipeline` API, which was improved upon realizing a bug, claiming that the dequantize is now faster than bnb dequant.
- **Vectorized Operations in CUDA**: `@uwu1468548483828484` inquired about vector loads for generic types in CUDA, to which `@zippika` shared an example implementation and suggested that vectorized addition and storage could improve performance.
- **Benchmarks in CUDA Indicate Coarsening Works on Large Data**: `@zippika` and `@cudawarped` had a detailed discussion on the effect of thread coarsening and the use of vectorized loads and storage, with benchmarks showing some benefits but also complexities related to using `int4`/`float4` types and vectorized operations like `__hadd2` on half precision arrays.

**Links mentioned**:

[cutlass/media/docs/cute at main · NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute): CUDA Templates for Linear Algebra Subroutines. Contribute to NVIDIA/cutlass development by creating an account on GitHub.

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1215206268935999498) (4 messages):

- **Clarifying CPU and CUDA Tensor Indexing**: `@mikkelisk` expressed confusion about why CPU tensors can be indexed by CUDA tensors if they are scalar, despite typically not being able to mix devices in operations. `@_t_vi_` explained that mixing is allowed due to compatibility with indexing using non-tensor CPU objects and historical reasons related to the special treatment of scalars.
- **Scalars Bridge the CPU-CUDA Gap in PyTorch**: Focusing on why this mix of devices works, `@_t_vi_` pointed out that scalars are treated specially and are automatically converted, a convenience and a legacy from the times when PyTorch treated scalars differently at the C/C++ level through `c10::Scalar`.
- **Beware of Hidden Inefficiencies**: `@_t_vi_` warned that while auto-transfer between CPU and GPU tensors for scalars is convenient, it can lead to hard-to-debug inefficiencies in code.

---

### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1215325325471256598) (5 messages):

- **Query on RelayAttention Mechanics**: `@lancerts` inquired about the differences between **RelayAttention** and ring/flash attention after coming across the GitHub repository [vLLM with RelayAttention](https://github.com/rayleizhu/vllm-ra).
  
- **Memory-Efficient Fine-Tuning Method**: `@iron_bound` referenced a method that significantly lowers memory requirements for fine-tuning, called **Gradient Low-Rank Projection (GaLore)**, as described in an [arXiv paper](https://arxiv.org/abs/2403.03507). They mentioned that even a single RTX 4090 GPU can be used for pre-training large models.
  
- **Efficient Training on Standard Gaming GPUs**: `@iron_bound` shared information about a technique enabling the fine-tuning of a 70b model on desktop computers with standard gaming GPUs. The method combines FSDP and QLoRA and has been detailed on [Answer.AI's blog post](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html), showcasing a collaboration with notable figures and organizations in the AI field.
  

**Links mentioned**:

- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507): Training Large Language Models (LLMs) presents significant memory challenges, predominantly due to the growing size of weights and optimizer states. Common memory-reduction approaches, such as low-ran...
- [Answer.AI - You can now train a 70b language model at home](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html): We’re releasing an open source system, based on FSDP and QLoRA, that can train a 70b model on two 24GB GPUs.
- [GitHub - rayleizhu/vllm-ra: vLLM with RelayAttention integration](https://github.com/rayleizhu/vllm-ra): vLLM with RelayAttention integration. Contribute to rayleizhu/vllm-ra development by creating an account on GitHub.

---

### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1215415993518522450) (4 messages):

- **Beginner Seeks CUDA Wisdom**: User `@violetmantis` sought advice on key resources within *cuda-mode/resource-stream* for learning about **cache-efficient parallel algorithms** and **kernel development/optimization**, hoping for direction on where to start given a plethora of available content.
- **Lectures for Starting Line**: `@marksaroufim` recommended beginning with the Discord's lectures designed for **complete beginners** to CUDA and parallel programming.
- **Concurrent Study Approach Suggested**: `@mertbozkir`, while also new to the field, suggested pairing video lectures with the accompanying book for a more informative learning experience.

---

### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1215329220998336552) (12 messages🔥):

- **Training at 16k leads to System Failure**: `@iron_bound` reported that training at 16k leads to the system going "brain dead" after 5 minutes with the GPUs at 100w. They speculated that the failure may occur at the end of the first epoch according to wandb logs.
- **Scheduling Sync-Up Discussions**: `@jamesmel` indicated an intention to join the next day's discussion, while `@iron_bound` mentioned that there wasn't much to discuss in the last sync as only Eric and they attended.
- **Inference Stuck Using Flash Attention**: `@jamesmel` encountered an issue with ring-llama inference on two GPUs, getting stuck at the `block_out` operation within `_flash_attn_forward` function, with both subprocesses pausing. They mentioned having installed flash-attn via pip before running ring-llama.

---

### CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1215398628928258098) (1 messages):

- **Mandelbrot Marvel**: User `@apaz` shared an image link showcasing a [Mandelbrot fractal](https://cdn.discordapp.com/attachments/1001261706762264709/1152786434420387922/mandelbrot2.jpg?ex=65f64f07&is=65e3da07&hm=eb2f8bf851ed742bc9d49fe9932f1d21f8c269ebbc681d1f65b75c6969c68081&). The image does not come with further context or discussion points.

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1215320232348880896) (31 messages🔥):

- **GaLore Breakthrough in Memory Efficiency**: User `@tiagoefreitas` shared a tweet from `@AnimaAnandkumar` showcasing that the Llama 7B LLM can now be trained on a single RTX 4090 GPU, significantly reducing memory costs for storing optimizer states. `@fx2y` points out the method, **Gradient Low-Rank Projection (GaLore)**, offers huge memory savings not just for pre-training but possibly for fine-tuning as well, potentially enabling further combination with techniques like 1-bit quantization for even greater efficiency.
  
- **Impressive Leap with Inflection-2.5**: `@stealthgnome` introduced a significant claim from `@inflectionAI` asserting that their **Inflection-2.5 model** approaches the performance of GPT-4 using only 40% of the compute for training. `@swyxio` pointed out that, while this claim is significant, Inflection didn't highlight it in their official blog post which gives a detailed introduction to Inflection-2.5.
  
- **FSDP/QLoRA Enables Home Training of Large Models**: `@fanahova` shared a tweet from `@jeremyphoward` announcing FSDP/QLoRA, a collaboration that allows training very large models on consumer-grade GPUs. `@fx2y` provided a link to the GitHub repo and noted the support for quantization methods like HQQ and bitsandbytes.
  
- **Yann LeCun Discusses AI Risks and Future on Lex** Podcast: Users `@stealthgnome`, `@swyxio`, and `@mr.osophy` discussed Yann LeCun's appearance on the Lex Fridman podcast, where he talked about Meta AI, the limits of LLMs, and his views on the future of AI, touching on the concept of Contrastive Learning.
  
- **Personal AI Stories with Life Story**: `@swyxio` shared their experience with **Life Story**, an AI that acts as a personal biographer, and provided feedback that while the call experience is good, he advised caution on the full experience and data security. The idea sparked interest, with `@tiagoefreitas` expressing a desire for more locally-hosted apps like this feature.
  
- **Controversy Surrounds OpenAI Leadership**: `@guardiang` pointed to brewing internal controversy at OpenAI with a New York Times article discussing the circumstances surrounding Sam Altman's departure. Further, `@aardvarkoncomputer` highlighted an ongoing dispute between `@inflectionAI` and claims regarding their Claude-3 model's wrapper.
  

**Links mentioned**:

- [Tweet from Prof. Anima Anandkumar (@AnimaAnandkumar)](https://x.com/animaanandkumar/status/1765613815146893348?s=46&t=PW8PiFwluc0tdmv2tOMdEg): For the first time, we show that the Llama 7B LLM can be trained on a single consumer-grade GPU (RTX 4090) with only 24GB memory. This represents more than 82.5% reduction in memory for storing optimi...
- [Inflection-2.5: meet the world's best personal AI](https://inflection.ai/inflection-2-5): We are an AI studio creating a personal AI for everyone. Our first AI is called Pi, for personal intelligence, a supportive and empathetic conversational AI.
- [Life Story](https://getlifestory.com/): Capture life, one story at a time.
- [Tweet from lmsys.org (@lmsysorg)](https://x.com/lmsysorg/status/1765774296000172289?s=46&t=90xQ8sGy63D2OtiaoGJuww): 🔥Exciting news from Arena @Anthropic's Claude-3 Ranking is here!📈 Claude-3 has ignited immense community interest, propelling Arena to unprecedented traffic with over 20,000 votes in just three...
- [Tweet from Jeremy Howard (@jeremyphoward)](https://x.com/jeremyphoward/status/1765868543235805232?s=20): Today, with @Tim_Dettmers, @huggingface, & @mobius_labs, we're releasing FSDP/QLoRA, a new project that lets you efficiently train very large (70b) models on a home computer with consumer gaming G...
- [Yann Lecun: Meta AI, Open Source, Limits of LLMs, AGI & the Future of AI | Lex Fridman Podcast #416](https://www.youtube.com/watch?v=5t1vTLU7s40): Yann LeCun is the Chief AI Scientist at Meta, professor at NYU, Turing Award winner, and one of the most influential researchers in the history of AI. Please...
- [Tweet from swyx (@swyx)](https://x.com/swyx/status/1765995892107317407?s=20): I've now had multiple >20min phone calls with AI therapists and it feels completely natural. Every AI Engineer should be building their own therapist rn, and voice is the right medium. forg...

---

### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1215596118767964201) (3 messages):

- **Epic GPT-2 Presentation Alert**: `@ivanleomk` announced that `@1123457263638683770` will be presenting on the GPT-2 paper in 20 minutes, urging the Asia `@paper-club` to attend what is promised to be an *EPIC sharing*.
  
- **Catching the Replay**: `@swyxio` responded to the announcement with excitement, indicating a desire to record the session.
  

**Links mentioned**:

[Join the Latent Space (née /dev/invest) Discord Server!](https://discord.gg/8sYsGc83): Check out the Latent Space (née /dev/invest) community on Discord - hang out with 3061 other members and enjoy free voice and text chat.

---

### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1215598769534533675) (30 messages🔥):

- **Preparation for GPT Paper Share**: `@ivanleomk` announced that the discussion will start soon and provided links to notes on the Generative Pre-trained Transformers, including [the concept](https://www.gaohongnan.com/transformer/decoder/concept.html) and [the implementation](https://www.gaohongnan.com/transformer/decoder/implementation.html), which `@1123457263638683770` would refer to during the sharing.
- **Ready to Start**: `@ivanleomk` gave a 5 minute heads-up before the start of the LLM paper club meeting.
- **A Newcomer's Enthusiasm**: `@healthymonkey` expressed being new to the NLP space and asked the more experienced participants like `<@1039021595089448990>` and `<@206404469263433728>` to correct any mistakes in their forthcoming discussion points.
- **Technical Clarification**: During the discussion, `@kishore.reddy` corrected `@ivanleomk`'s explanation of a decoder model by mentioning "causal attention," which refers to ensuring that the model predicts the next token without access to future token states.
- **Practical Demonstration of LLM Concepts**: `@fx2y` shared a [link to LLM Visualization](https://bbycroft.net/llm), a tool useful for visualizing the GPT family of models, and commended `@1123457263638683770`'s effort in the discussion.

**Links mentioned**:

- [LLM Visualization](https://bbycroft.net/llm): no description found
- [The Concept of Generative Pre-trained Transformers (GPT) — Omniverse](https://www.gaohongnan.com/transformer/decoder/concept.html): no description found
- [The Implementation of Generative Pre-trained Transformers (GPT) — Omniverse](https://www.gaohongnan.com/transformer/decoder/implementation.html): no description found

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1215300641686163516) (21 messages🔥):

- **LangChain JS Feature Query**: `@0x404blockchainnotfound` inquired about whether the **LangChain JS library** has achieved feature parity with the Python library, but no direct answer was provided within the chat.
- **AGI Claims Discussed**: `@sales_god` sought opinions on claims about agent/AGI discussed on Hacker News, but the discussion did not lead to a resolution and was sidetracked by the comment from `@baytaew` highlighting concerns about the LangChain tool and ReACT agents.
- **Finished Agent Event in Python**: `@cybersmiths` reported a delay issue with the Finished agent event in Python, causing a 1-2 second delay after the last character is streamed. However, the thread did not include a solution to the problem.
- **Handling PDF Loader Extractions**: `@yd4224` encountered formatting problems when using **langchain.document_loaders PyPDFLoader**, receiving guidance from `@travellingprog` to create a custom loader or contribute to the repository to handle `extraction_mode` arguments.
- **Push for JavaScript URL Loader**: `@mohitsakhiya077` expressed the need for functionality to load documents from multiple URLs in JavaScript, similar to what's available in the Python version with `UnstructuredURLLoader`, prompting discussion on parity between languages.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39623023): no description found
- [Ollama Functions | 🦜️🔗 Langchain](https://js.langchain.com/docs/integrations/chat/ollama_functions): LangChain offers an experimental wrapper around open source models run locally via Ollama
- [Extract Text from a PDF — pypdf 4.0.1 documentation](https://pypdf2.readthedocs.io/en/stable/user/extract-text.html): no description found
- [URL | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/document_loaders/url): This covers how to load HTML documents from a list of URLs into a
- [langchain/libs/community/langchain_community/document_loaders/parsers/pdf.py at v0.1.11 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/v0.1.11/libs/community/langchain_community/document_loaders/parsers/pdf.py#L97): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.

---

### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1215349964125769758) (9 messages🔥):

- **Redis Chat History Woes**: `@justanothergraphguy` is working to create a chat chain with Redis chat history and structured output parsing using a pydantic model. They encountered an issue where the latest "HumanMessage" appears in the `AIMessage` content incorrectly, suggesting potential problems with memory propagation.
  
- **Designing the Prompt and Model**: A system prompt for a "User Profile Builder" guides the assistant's interaction, aiming to extract user information and build a profile.
  
- **Technical Setup Unveiled**: `@justanothergraphguy` shared a snippet of Python code integrating various `langchain` modules such as `ChatOpenAI`, `RunnableWithMessageHistory`, and `PydanticOutputParser` to create the chat chain.
  
- **First Interaction Flawlessly Executed**: An initial example provided by `@justanothergraphguy` showed correct extraction of "Bob's" name, while the system prompted for more information to complete the profile.
  
- **Subsequent Interaction Confusion**: In the follow-up interaction, the output incorrectly included the 'HumanMessage' as part of the `AIMessage` content, highlighting the memory issue in their system.
  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1215256103240212530) (2 messages):

- **Visual AI in the Spotlight**: `@vru.shank` announced a workshop featuring [MultiOn](https://www.multion.ai/) and [Quizizz](https://quizizz.com) discussing their use of vision models in production. Interested individuals can RSVP for a session hosted by the [LLMs in Prod community](https://portkey.ai/community) via [this link](https://lu.ma/multimodal-llms).
  
- **Introducing Prompt Mixer - Your Prompt IDE**: `@tomatyss` is developing **Prompt Mixer**, a desktop tool for building, testing, and iterating AI prompts with version tracking. Feedback and feature suggestions are welcomed, and interested users can download it at [Prompt Mixer’s website](https://www.promptmixer.dev/).
  
- **How to Customize Your Connectors**: For advanced users of Prompt Mixer, `@tomatyss` shared a [documentation link](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector) detailing the steps to create a custom connector, thus enhancing their tool's flexibility and functionality.
  

**Links mentioned**:

- [no title found](https://quizizz.com)): no description found
- [Multi-Modal LLMs in Prod | Practitioners' Workshop · Luma](https://lu.ma/multimodal-llms): The LLMs in Prod community is hosting practitioners from top Gen AI companies to talk about how they are using multi-modal models (vision, audio, image gen, etc.) in...
- [Prompt Mixer — Prompt IDE and LLMOps tool](https://www.promptmixer.dev/): PromptMixer – the innovative Prompt IDE for crafting, testing, and deploying prompts with unparalleled ease.
- [Create a Custom Connector | Prompt Mixer Docs](https://docs.promptmixer.dev/tutorial-extras/create-a-custom-connector): Step 1: Copy the Sample Connector

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages):

pradeep1148: [https://www.youtube.com/watch?v=PtP8R8VjTGc](https://www.youtube.com/watch?v=PtP8R8VjTGc)

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1215337699179892736) (3 messages):

- **In search of German-finetuned Mixtral**: User `@johannhartmann` inquired about comparing models like **sauerkraut oder discolm mixtrals** for German language prompts and noted that **Nous Hermes Mixtral** has no German finetuning involved.
- **Introducing Evo: Biological Language Model**: `@rasdani` highlighted the new **Evo architecture** from TogetherAI named Striped Hyena, designed for DNA sequence modeling. [Read about Evo's capabilities](https://www.together.ai/blog/evo) in handling various biological sequences and its collaborative development with the [Arc Institute](https://arcinstitute.org/).

**Links mentioned**:

[Evo: Long-context modeling from molecular to genome scale](https://www.together.ai/blog/evo): no description found

---

### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1215221563574452264) (11 messages🔥):

- **Fine-Tuning Discourse with Hermes Mixtral DPO**: `@flozi00` is working on finetuning the **Nous Hermes Mixtral DPO model**, aiming for improvements before moving to train a classification model, but notes the process involves *sorting out much trash*.
- **Creating a Quality Translation Dataset**: In pursuit of quality translation estimations, `@flozi00` plans to set up an **Argilla space** to label translations from Google Translate, DeepL, and Azure Translate.
- **Targeting En-De Translation Pairs**: `@crispstrobe` recommends leveraging the EN-DE pairs from the OPUS 100 dataset to create a subset with reliable pairings suited for unspecific contexts, highlighting the dataset's utility in creating training subsets.
- **Dataset Licensing and Quality Concerns**: `@philipmay` shares that the mMARCO dataset now has an Apache 2.0 license but encountered issues with viewing the dataset on HuggingFace, indicating the need for assistance to make the **dataset viewer** work.
- **Public Collection for Translation Data Quality**: `@flozi00` mentions an update to their judge model and datasets, seeking additional tips for improvement, which is now part of a [HuggingFace collection aimed at measuring translation pair quality](https://huggingface.co/collections/flozi00/translation-data-quality-65e9d0cdd977e1e0aed2de9d).

**Links mentioned**:

- [Translation Data Quality - a flozi00 Collection](https://huggingface.co/collections/flozi00/translation-data-quality-65e9d0cdd977e1e0aed2de9d): no description found
- [unicamp-dl/mmarco · Datasets at Hugging Face](https://huggingface.co/datasets/unicamp-dl/mmarco#licensing-information): no description found
- [Data (Hint ID)](https://huggingface.co/data): no description found

---

### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1215570910300602368) (3 messages):

- **Merging German Translations with SPIN**: `@johannhartmann` shared that they use a German translation of the slim orca dataset for **Mistral merges**, applying a SPIN-like method across multiple steps. They create datasets with answers from multiple models for the same translated instruction/input-pair, which has led to observable drifts in the model's responses, sometimes becoming more verbose or degrading post-merge. They plan to clean up and upload the dataset soon.
  
- **Brezn3 Outshines Brezn-7b**: `@crispstrobe` expressed amazement as **Brezn3** scores 63.25 on EQ-Bench (v2) (de) without revision, outperforming Brezn-7b's score of 58.22. They inquired whether this was solely due to changing the base model to **LeoLM/leo-mistral-hessianai-7b-chat** and setting `tokenizer_source: base`, or if different DPO modifications were applied.
  
- **DPO Still Baking for Brezn3**: `@johannhartmann` responded that the DPO process for Brezn3 is still underway, with approximately 13 hours remaining before completion.
  

---

### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1215297494645481472) (6 messages):

- **Models and the Challenge with Names**: `@res6969` observed that **names** could pose a difficulty for models to handle correctly.
- **Hands-On with Claude Functionality**: `@res6969` shared their experience of experimenting with **function calling** on *Claude*, suggesting that progress is being made.
- **Accolades for Claude's Humorous Accuracy**: `@res6969` expressed amusement and approval for Claude's performance by calling it **"hilarious and correct"**.
- **Function Calling Works with a Catch**: `@res6969` confirmed that function calling on Claude is effective, but highlighted that **XML tags** are necessary for optimal results.
- **XML Complexity Raises Concern**: `@pantsforbirds` commented on the complexity of using **XML tags**, noting that it complicates the sharing of prompt generators.

---

### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1215370759464165386) (5 messages):

- **GPT-4 Flunks the Test**: `@dbreunig` expressed surprise at how poorly GPT-4 performed on an unspecified test.
- **Clickable Bookshelves Innovation**: `@xnimrodx` shared admiration for a script that automates the creation of clickable bookshelf images, leading users to a Google Books page for each book, found in a [blog post](https://jamesg.blog/2024/02/14/clickable-bookshelves/) and accompanied by a [demo](https://capjamesg.github.io/cv-book-svg/).
- **A Librarian's Dream Tool**: `@xnimrodx` noted personal interest in automated bookshelf management for a librarian, remarking it would greatly aid in shelf-reading tasks across large collections.
- **Impressive Library Management**: `@xnimrodx` shared that their librarian-wife manages the largest school library within a 35-school diocesan system, rivaling the size of some public library branches.
- **Little Library Cataloging App Idea**: `@dbreunig` mentioned an interest in creating a toy app to catalog the books in little libraries throughout their town.

**Links mentioned**:

[Making my bookshelves clickable | James' Coffee Blog](https://jamesg.blog/2024/02/14/clickable-bookshelves/): no description found

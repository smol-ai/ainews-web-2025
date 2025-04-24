---
id: d2ec8f17-55b8-4970-b09c-76b6c3739617
title: Evals-based AI Engineering
date: '2024-03-29T22:20:49.586743Z'
original_slug: ainews-evals-based-ai-engineering
description: >-
  AI news covering eval systems, model updates, voice cloning, and new
  architectures. Highlights include OpenAI's voice engine demo, new models like
  Jamba, Bamboo, Qwen1.5-MoE, Grok 1.5, and quantization advances such as 1-bit
  Llama2-7B and QLLM.
companies:
  - openai
  - hamel husain
  - langsmith
  - heygen
  - noam
  - dwarkesh
  - sholto
  - trenton
  - x.ai
models:
  - gpt-4
  - claude-3-opus
  - grok-1.5
  - qwen1.5-moe
  - bamboo-7b
  - jamba
  - llama2-7b
  - llama2-7b-quantized
topics:
  - ai-evaluation
  - model-architecture
  - voice-cloning
  - multimodal
  - quantization
  - llm-performance
  - ai-safety
---


<!-- buttondown-editor-mode: plaintext -->> AI News for 3/27/2024-3/29/2024. We checked 5 subreddits and [**364** Twitters](https://twitter.com/i/lists/1585430245762441216) and **25** Discords (**377** channels, and **5415** messages) for you. Estimated reading time saved (at 200wpm): **615 minutes**.

Evals are the "eat your vegetables" of AI engineering - everyone knows they should just do more of it:

 ![image.png](https://assets.buttondown.email/images/b87d3db1-c544-4a8c-ad8a-f1b65e7f9156.png?w=960&fit=max) 

**Hamel Husain** has yet another banger in his blog series: [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/):

> Like software engineering, success with AI hinges on how fast you can iterate. You must have processes and tools for:
> 
> 1. Evaluating quality (ex: tests).
> 1. Debugging issues (ex: logging & inspecting data).
> 1. Changing the behavior or the system (prompt eng, fine-tuning, writing code)
>
> **Many people focus exclusively on #3 above**, which prevents them from improving their LLM products beyond a demo. **Doing all three activities well creates a virtuous cycle** differentiating great from mediocre AI products (see the diagram below for a visualization of this cycle).

We are guilty of this at AINews - our loop is slow and hence the product improvement pace has also been much slower than we would want to see. Hamel proposes a mental model to center on evals:

 ![image.png](https://assets.buttondown.email/images/1ff2d80f-b60e-4067-94eb-8bf4e1470972.png?w=960&fit=max) 

Excerpts we liked:

- **You must remove all friction from the process of looking at data.**
- **Many vendors want to sell you tools that claim to eliminate the need for a human to look at the data.** but **You should track the correlation between model-based and human evaluation to decide how much you can rely on automatic evaluation**.
- **Eval Systems Unlock Superpowers For Free**. In addition to iterating fast, having good evals unlock the ability to finetune and synthesize data.

The post has a lot of practical advice on how to make these "sensible things" easy, like using spreadsheets for hand labeling or hooking up LangSmith (which doesn't require LangChain).

 ![image.png](https://assets.buttondown.email/images/4fea57f5-e2ff-4905-8b8d-5dfbd4fd3790.png?w=960&fit=max) 

---

**Obligatory AI Safety PSA**: OpenAI today released some samples of [their rumored Voice Engine](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) taking a 15s voice samples and successfully translating to different domains and languages. It's a nice demo and is great marketing for HeyGen, but more importantly they are trying to warn us that very very good voice cloning from small samples is here. Take [Noam's word](https://twitter.com/polynoamial/status/1773799870890918358) for it (who is at OpenAI but not on the voice team): 

![image.png](https://assets.buttondown.email/images/eb5488d6-160c-43da-b1c0-e3ca47d8c7cf.png?w=960&fit=max) 

Alec Radford does not miss. We also [enjoyed Dwarkesh's pod with Sholto and Trenton](https://twitter.com/dwarkesh_sp/status/1773381318266786180?t=90xQ8sGy63D2OtiaoGJuww).


---

**Table of Contents**

[TOC] 


---

# AI Reddit Recap

> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence. Comment crawling still not implemented but coming soon.


**New Models and Architectures**:

- **Jamba**: The first production-grade Mamba-based model using a hybrid Transformer-SSM architecture, outperforming models of similar size. ([Introducing Jamba - Hybrid Transformer Mamba with MoE](https://www.reddit.com/r/LocalLLaMA/comments/1bpx9sh/introducing_jamba_hybrid_transformer_mamba_with/))
- **Bamboo**: A new 7B LLM with 85% activation sparsity based on Mistral's weights, achieving up to 4.38x speedup using hybrid CPU+GPU. ([Introducing Bamboo: A New 7B Mistral-level Open LLM with High Sparsity](https://www.reddit.com/r/LocalLLaMA/comments/1bpqzck/introducing_bamboo_a_new_7b_mistrallevel_open_llm/))
- **Qwen1.5-MoE**: Matches 7B model performance with 1/3 activated parameters. ([Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters](https://qwenlm.github.io/blog/qwen-moe/))
- **Grok 1.5**: Beats GPT-4 (2023) in HumanEval code generation and has 128k context length. ([Grok 1.5 now beats GPT-4 (2023) in HumanEval (code generation capabilities), but it's behind Claude 3 Opus](https://i.redd.it/1jl3qaqzl6rc1.png), [X.ai announces grok 1.5 with 128k context length](https://x.ai/blog/grok-1.5))

**Quantization and Optimization**:

- **1-bit Llama2-7B**: Heavily quantized models outperforming smaller full-precision models, with a 2-bit model outperforming fp16 on specific tasks. ([1-bit Llama2-7B Model](https://www.reddit.com/r/LocalLLaMA/comments/1bptl1w/1bit_llama27b_model/))
- **QLLM**: A general 2-8 bit quantization toolbox with GPTQ/AWQ/HQQ, easily converting to ONNX. ([Share a LLM quantization REPO , (GPTQ/AWQ/HQQ ONNX ONNX-RUNTIME)](https://www.reddit.com/r/LocalLLaMA/comments/1bqeqr7/share_a_llm_quantization_repo_gptqawqhqq_onnx/))
- **Adaptive RAG**: A retrieval technique dynamically adapting the number of documents based on LLM feedback, reducing token cost by 4x. ([Tuning RAG retriever to reduce LLM token cost (4x in benchmarks)](https://www.reddit.com/r/LocalLLaMA/comments/1bq2g3e/tuning_rag_retriever_to_reduce_llm_token_cost_4x/), [Adaptive RAG: A retrieval technique to reduce LLM token cost for top-k Vector Index retrieval [R]](https://www.reddit.com/r/MachineLearning/comments/1bq3hwb/adaptive_rag_a_retrieval_technique_to_reduce_llm/))

**Stable Diffusion Enhancements**:

- **Hybrid Upscaler Workflow**: Combining SUPIR for highest quality with 4x/16x quick upscaler for speed. ([Hybrid Upscaler Workflow](https://www.reddit.com/gallery/1bpqo5z))
- **IPAdapter V2**: Update to adapt old workflows to the new version. ([IPAdapter V2 update old workflows](https://www.reddit.com/r/StableDiffusion/comments/1bpz0zn/ipadapter_v2_update_old_workflows/))
- **Krita AI Diffusion Plugin**: Fun to use for image generation. ([Krita AI Diffusion Plugin is so much fun (link in thread)](https://v.redd.it/1k6360roj7rc1))

**Humor and Memes**:

- **AI Lion Meme**: Humorous image of a lion. ([AI Lion Meme](https://i.redd.it/r4n37xjj66rc1.jpeg))
- **"No, captain! What you don't get is that these plants are Alive!!"**: Humorous text-to-video generation. (["No, captain! What you don't get is that these plants are Alive!! & they are overwhelming us... what's that? Yes! Marijuana Leaves! Hello Captain? Hello!!" (TEXT TO VIDEO- SDCN for A1111, no upscale)](https://v.redd.it/e33z5vu2t5rc1))
- **"Filming animals at the zoo"**: Humorous image of a person filming animals. (["Filming animals at the zoo"](https://i.redd.it/c8fev6j9a4rc1.jpeg))

---


# AI Twitter Recap

> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.

**AI Models and Architectures**

- **New open-source models released**: [@AI21Labs](https://twitter.com/AI21Labs/status/1773350888427438424) introduced Jamba, an open SSM-Transformer model based on Mamba architecture, achieving **3X throughput and fitting 140K context on a single GPU**. [@databricks](https://twitter.com/databricks/status/1773355487578280317) released DBRX, setting **new SOTA on benchmarks like MMLU, HumanEval, and GSM 8k**. [@AlibabaQwen](https://twitter.com/AlibabaQwen/status/1773370876496334962) released Qwen1.5-MoE-A2.7B, a **MoE model with 2.7B activated parameters that can achieve 7B model performance**.
- **Quantization advancements**: [@maximelabonne](https://twitter.com/maximelabonne/status/1773268783139881073) shared a Colab notebook comparing **FP16 vs. 1-bit LLama 2-7B models quantized using HQQ + LoRA**, with SFT greatly improving the quantized models, enabling fitting larger models into smaller memory footprints.
- **Mixture of Experts (MoE) architectures**: [@osanseviero](https://twitter.com/osanseviero/status/1773360705682411750) provided an overview of different types of MoE models, including **pre-trained MoEs, upcycled MoEs, and FrankenMoEs**, which are gaining popularity for their efficiency and performance benefits.

**AI Alignment and Factuality**

- **Evaluating long-form factuality**: [@quocleix](https://twitter.com/quocleix/status/1773406864589672521) introduced a new dataset, evaluation method, and aggregation metric for assessing the factuality of long-form LLM responses using **LLM agents as automated evaluators through Search-Augmented Factuality Evaluator (SAFE)**.
- **Stepwise Direct Preference Optimization (sDPO)**: [@_akhaliq](https://twitter.com/_akhaliq/status/1773571320627790178) shared a paper proposing sDPO, an extension of Direct Preference Optimization (DPO) for alignment tuning that **divides available preference datasets and utilizes them in a stepwise manner**, outperforming other popular LLMs with more parameters.

**AI Applications and Demos**

- **AI-powered GTM platform**: [@perplexity_ai](https://twitter.com/perplexity_ai/status/1773446960889938055) partnered with @copy_ai to create an AI-powered Go-To-Market platform offering **real-time market insights**, with Copy AI users receiving 6 months of Perplexity Pro for free.
- **Journaling app with long-term memory**: [@LangChainAI](https://twitter.com/LangChainAI/status/1773381734215958971) introduced LangFriend, a journaling app leveraging **memory capabilities for a personalized experience**, available to try with a developer-facing memory API in the works.
- **AI avatars and video generation**: [@BrivaelLp](https://twitter.com/BrivaelLp/status/1773442026966585484) showcased an experiment with Argil AI, creating **entirely AI-generated content featuring an AI Barack Obama teaching quantum mechanics**, demonstrating AI's potential to revolutionize user-generated and social content creation.

**AI Community and Events**

- **Knighthood for services to AI**: [@demishassabis](https://twitter.com/demishassabis/status/1773421805929202165), CEO and co-founder of @GoogleDeepMind, was awarded a **Knighthood by His Majesty for services to Artificial Intelligence** over the past 15 years.
- **AI Film Festival**: [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1773363405715321249) announced the **second annual AI Film Festival in Los Angeles on May 1** to showcase the best in AI cinema.

---

# PART 0: Summary of Summaries of Summaries



**1) New AI Model Releases and Architectures**:

- **[AI21 Labs unveils Jamba](https://www.ai21.com/blog/announcing-jamba)**, a hybrid **SSM-Transformer model** with **256K context window**, **12B active parameters**, and **3X throughput** for long contexts. Open weights under **Apache 2.0 license**.
- **[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)**, a **Transformer-based MoE model** matching 7B performance with only **2.7B activated parameters**.
- **[LISA algorithm](https://arxiv.org/abs/2403.17919)** enables **7B parameter fine-tuning on 24GB GPU**, outperforming LoRA and full training for instruction tasks.
- **SambaNova unveils Samba-1**, a **Composition of Experts (CoE) model** claiming reduced compute needs and higher performance, though transparency concerns exist.

**2) Open Source Collaboration and Community Projects**:

- **[Modular open-sources Mojo standard library](https://modul.ar/open-source-blog)** under Apache 2 license, with nightly builds and a [community livestream](https://www.youtube.com/watch?v=PL71FV2KKHE) covering updates.
- **[LlamaIndex blog](https://t.co/MCnmhQGS7g)** shares optimizations like **Int8 and Binary Embeddings for RAG pipelines** by Cohere to reduce memory and costs.
- Community showcases include **[Tinyllama](https://ollama.com/pacozaa/tinyllama-alpaca-lora)** and **[Mischat](https://ollama.com/pacozaa/mischat)** models leveraging Unsloth notebooks.
- **[OpenGPTs project](https://github.com/langchain-ai/opengpts)** gets a dedicated <#1222928565117517985> channel for collaboration on the open-source platform.

**3) Model Evaluation, Benchmarking and Datasets**:

- Discussions on **reliability of existing AI benchmarks** like [HuggingFace Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) which includes human evaluation.
- Sharing of **long context datasets** like [MLDR on HuggingFace](https://huggingface.co/datasets/Shitao/MLDR) for training models requiring extensive sequences.
- Plans for **translation model battle** between DiscoLM, Occiglot, Mixtral, GPT-4, DeepL, Azure Translate on Capybara dataset.
- **[Shure-dev mission](https://shure-dev.github.io/)** to curate high-quality **LLM research papers** as a resource for the rapidly evolving field.

**4) Local LLM Deployment and Hardware Optimization**:

- Guidance on **running LLMs locally** with LM Studio, Open Interpreter, and hardware like RTX 3090 GPUs.
- Recommendations for **power supply sizing (1200-1500W)** and efficient **GPU power limits (350-375W for RTX 4090)**.
- Discussions on **VRAM requirements** for training large models like LLaMa 7B with QLoRA and LoRA.
- **[Windows Subsystem for Linux (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install)** suggested for running CUDA/Torch on Windows gaming machines.


---



# PART 1: High level Discord summaries




## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Supercharging SD3 with Turbo Techniques**: Enthusiasm buzzes around [SD3 Turbo](https://arxiv.org/pdf/2403.12015.pdf) to provide a high-detail model similar to SDXL Lightning, with discussions hinting at promising performance on 12GB VRAM systems.

- **Balancing Power & Accessibility in AI Art**: Debate centers on potential SD3 feature downscaling for broader accessibility; suggestions include providing control nets and fine-tuning tools upon launch to empower users.

- **Training LoRA with SD3 Under the Microscope**: The likelihood of training LoRA models using SD3 with 24GB of VRAM is mulled over, with questions about the transition of these tools to SD3 after its release remaining unanswered.

- **Smaller GPUs Tackling Big Training Ambitions**: Success stories of training smaller LoRA models on 16GB VRAM GPUs emerge, highlighting optimisations such as Aggressive train rates and network dropouts.

- **Arc2Face Steals the Spotlight**: [Arc2Face](https://arc2face.github.io/) entices members with its face manipulation prowess amid jests about alien technology and dataset censorship controversies.




---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**DBRX Takes the Limelight**: **DBRX**, a newfound heavyweight in language models by Databricks, steals the show at Perplexity Labs, outshining GPT-3.5 and neck-to-neck with Gemini 1.0 Pro on tasks demystifying math and coding challenges. A green light is given to test-run DBRX for free at [Perplexity Labs](https://labs.pplx.ai), throwing the gauntlet down for AI connoisseurs.

**Mark Your Calendars for Copy AI and Perplexity Alliance**: The fusion of Copy AI's platform with Perplexity's state-of-the-art APIs points to an uprising in go-to-market strategies, lighting the path for real-time market acumen. For users leveraging Copy AI, a half-year access to Perplexity Pro cuts the ribbon, and it's all spelled out in their co-authored [blog post](https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams).

**In Search of Perfection with Perplexity**: Users are scratching their heads over the hit-or-miss performance of **academic focus mode** in Perplexity's search capabilities, puzzled by intermittent outages. Improvement in the **Pro Search** spaces and conflicting tales of file sources dominated discussions, with a spotlight on possibly employing **RAG** or **GPT-4-32k** technology for diverse file processing.

**Tuning into Enhanced Scratchpad Tactics**: The community exchanges notes on drawing out the best from Perplexity; one user gives a hands-on demo using `<scratchpad>` XML tags, and space enthusiasts fling questions at the AI about Starship and astronautics. Users also threw in finance-flavored queries, probing into Amazon's monetary moves and the FTX conundrum.

**API Adventures and Misadventures**: Queries abound regarding the Perplexity AI API's unpredictable behavior, where search results are sometimes lost in the web/API rift while hunting for answers, veering off from the steadiness promised on the web interface. For those thirsting for beta feature participation, including coveted URL citations, a beeline can be made to [Apply for beta features](https://perplexity.typeform.com/to/j50rnNiB), keeping API fanatics at the edge of their seats.



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Jamba Juices Up AI Debate**: The discussion was sparked by the announcement of [Jamba](https://www.ai21.com/blog/announcing-jamba) by AI21 Labs, a model that integrates Mamba SSM with Transformers. Engineers debated Jamba's capacity, tokenization, and lack of transparency in token counts.

- **Quantifying Model Behaviors**: Conversations arose around evaluating open-ended generation in models. To assist, various resources, including the paper *Are Emergent Abilities of Large Language Models a Mirage?* [available here](https://arxiv.org/abs/2304.15004), were shared to provide clarity on perplexity and pairwise ranking metrics.

- **Fine-Tuning Face-Off**: There was robust sharing of fine-tuning strategies on models like Mistral, involving SFTTrainer difficulties and optimal epochs and batch sizes discussion. Technical difficulties in multi-GPU support and dependencies like bitsandbytes were flagged for resolution.

- **Karpathy's Kernel of Wisdom**: Andrej Karpathy's [YouTube deep-dive](https://www.youtube.com/watch?v=c3b-JASoPi0) likened LLMs to operating systems, stirring talks on finetuning practices and the importance of using original training data. He voiced criticism of current RLHF techniques, contrasting them with AlphaGo, and suggested a fundamental shift in model training approaches.

- **Training Trials and Successes Shared**: Insights were exchanged regarding tokenizer errors, finetuning pitfalls, checkpointing methods, and best practices to avoid overfitting. Practical guidance was given on pushing local checkpoints to Hugging Face and the usage of `hub_token` and `save_strategy`.

- **Innovative Model Instances Introduced**: The community showcased new models, such as [Tinyllama](https://ollama.com/pacozaa/tinyllama-alpaca-lora) and Mischat, illustrating how Unsloth's notebooks and a Huggingface dataset can lead to novel LLMs. A member promoted 'AI Unplugged 5,' a blogpost covering diverse AI topics found at [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple).



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AI's Pricey Playground**: AI hobbyists are voicing concerns over the high cost of entry, with investments closing in on **$10k** for necessary hardware like cutting-edge GPUs or MacBooks to ensure optimal performance. Discussions reveal a "hard awakening" to the complex, jargon-heavy, and resource-intensive nature of the field.

- **GPU Talk – Power, Performance, and Particulars**: Engagements in hardware chat center on optimal **power supply sizing** with recommendations such as **1200-1500W** for high-performance builds and efficient power limit settings. There's also interest in the practical performance benefits of legacy hardware like Nvidia K80, alongside advanced cooling hacks using old iMac fans.

- **Model Performance and Compatibility Challenges**: Conversations pinpoint **VRAM choke points** and compatibility, noting the inadequacy of **32GB VRAM** for upcoming models and discussing model performance, with one mention of **Zephyr's** superiority in creative writing tasks over **Hermes2 Pro**. Users are also troubleshooting issues with GPUs proper recognition and utilization with **LM Studio's ROCm beta version** and grappling with the selection of coder models for constrained VRAM.

- **LM Studio Talk – Updates, Issues, and Integrations**: The latest LM Studio version **0.2.18** rollout, which introduces new features and bug fixes, is being dissected, with users highlighting discrepancies in VRAM display and version typings and requesting features like GPU and NPU monitoring tools. A member shared their successful integration of **LM Studio** into Nix, [available as a PR](https://github.com/NixOS/nixpkgs/pull/290399). Plus, a repository for a chatbot **OllamaPaperBot** designed to work with PDFs was shared [on GitHub](https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py).

- **Rethinking Abstraction with AI**: There is exploration into using AI to distill abstract concepts, with a reference to the paper on [Abstract Concept Inflection](https://arxiv.org/pdf/2304.01904.pdf). A request was made for guidance on choosing the right Agent program to plug into **LM Studio**, aiming for a seamless integration.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**Voice Cloning Sparks Heated Debate**: Discussions emerged around **OpenAI's Voice Engine**, with some excited by its potential to generate natural-sounding speech from a [15-second audio sample](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices), while others raised ethical concerns about the tech's misuse.

**Confused Consumers and Missing Models**: Confusion reigns among users regarding different versions of **GPT-4** implemented in various applications, with contradictory reports about model stability and cutoff dates. Meanwhile, anticipation for **GPT-5** is rife, yet no concrete information is available.

**Encounters with Errant Equations**: Users across multiple channels grappled with transferring **LaTeX equations** into Microsoft Word, proposing **MathML** as a potential solution. The intricacies of proper prompt structuring for specific AI tasks, like translations maintaining HTML tags, also took center stage.

**Meta-Prompting Under the Microscope**: AI enthusiasts debated the merits of **metaprompting** over direct instructions, with experiences suggesting inconsistent results. Precise prompts were underscored as pivotal for optimized AI performance.

**Roleplay Resistance in GPT**: A peculiar behavior was noted with the **gpt-4-0125-preview** model regarding roleplay prompts, with the AI refusing to role-play when an example format was given, yet complying when the example was omitted. Users shared workarounds and tactics to guide the AI's responses.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**New Fine-Tuning Frontiers**: LISA, a **new fine-tuning technique**, has outshined LoRA and full-parameter training in instruction following tasks, capable of tuning 7B parameter models on a 24GB GPU. LISA's details and applications can be explored through the [published paper](https://arxiv.org/abs/2403.17919) and [its code](https://github.com/OptimalScale/LMFlow).

**Chip Chat heats up**: AI21 Labs revealed **Jamba**, a model fusing the **Mamba** architecture and Transformers, with a claim of 12B active parameters from a 52B total. Meanwhile, SambaNova introduced **Samba-1**, a Composition of Experts (CoE) model alleging reduced compute needs and higher performance, though transparency concerns persist. Details about Jamba can be found on their [official release page](https://www.ai21.com/jamba), and scrutiny over Samba-1's performance is encouraged via [SambaNova's blog](https://sambanova.ai/blog/accurate-models-at-blazing-speed).

**Sensitive Data Safety Solutions Discussed**: Techniques for safeguarding sensitive data in training, including SILO and differential privacy methods, formed a topic of serious discussion. Researchers interested in these topics can examine the [SILO paper](https://arxiv.org/abs/2308.04430) and [differential privacy papers](https://arxiv.org/abs/1607.00133, https://arxiv.org/abs/2110.05679) for more insights.

**Discrepancy Detective Work in Model Weights**: Discordants untangled differences in model weight parameters between **Transformer Lens** (`tl`) and **Hugging Face** (`hf`). The debugging process involved leveraging `from_pretrained_no_processing` to avoid preset weight modifications by Transformer Lens, as elucidated in this [GitHub issue](https://github.com/neelnanda-io/TransformerLens/issues/346).

**MMLU Optimization Achieved**: Efficiency in MMLU tasks has been boosted, enabling extraction of multiple logprobs within a single forward call. A user reported memory allocation issues when attempting to load the **DBRX base model** on incorrect GPU configurations, corrected upon realizing the node configuration error. Further, a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571) aimed at improving context-based task handling in the **lm-evaluation-harness** awaits review and feedback after the CoLM deadline.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

**A Peek at AI21's Transformer Hybrid**: AI21 Labs has launched [Jamba](https://www.ai21.com/blog/announcing-jamba), a transformative SSM-Transformer model with a 256K context window and performance that challenges existing models, openly accessible under the Apache 2.0 license.

**LLMs Gearing Up with MoE**: The engineering community is charged up about [microqwen](https://twitter.com/JustinLin610/status/1773285084025475355), speculated to be a more compact version of Qwen, and the debut of [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B), a transformer-based MoE model that promises high performance with fewer active parameters.

**LLM Training Woes and Wins**: Engineers are troubleshooting issues with the **Deepseek-coder-33B**'s full-parameter fine-tuning, exploring *structured approaches* for a large book dataset, and peeking at Hermes 2 Pro's multi-turn agentic loops. Meanwhile, they're diving into the significance of 'hyperstition' in expanding AI capacities and clarifying heuristic versus inference engines in LLMs.

**RAG Pipelines and Data Structuring Strategies**: To boost performance and efficiency in retrieval tasks, AI engineers are exploring structured XML with metadata and discussing **RAG** models. A mention of a [ragas GitHub repository](https://github.com/explodinggradients/ragas) indicates ongoing enhancements to RAG systems.

**Worldsim, LaTeX, and AI's Cognitive Boundaries**: Tips and resources, like the [gist for LaTeX papers](https://gist.github.com/irl-dan/61e2f45eb1c9a879b3), are being exchanged on the Worldsim project. Engineers are considering the potential of AI to delve into alternate history scenarios, while carefully differentiating between large language model use-cases.

With these elements converged, engineers are evidently navigating the challenges and embracing the evolving landscape of AI with a focus on efficiency, structure, and the constant sharing of knowledge and resources.



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo Gets Juiced with Open Source and Performance Tweaks**: Modular has cracked open the **Mojo standard library** to the open-source community under the Apache 2 license, showcasing this in the MAX 24.2 release. Enhancements include implementations for _generalized complex types_, workshop sessions on **NVIDIA GPU support**, and a focus on stabilizing support for MLIR with the syntax set to evolve.

**Hype Train Gathers Steam for Modular's Upcoming Reveal**: Modular is stoking excitement through a series of cryptic tweets, signaling a new announcement with emojis and a ticking clock. Community members are keeping a keen eye on the official [Twitter](https://twitter.com/Modular) handle for details on the enigmatic event.

**MAX Engine's Leaps and Bounds**: With the **MAX Engine 24.2** update, Modular introduces support for TorchScript models with dynamic input shapes and other upgrades, as detailed in their [changelog](https://modul.ar/max-changelog). A vivid discussion unfolded around performance benchmarks using the BERT model and GLUE dataset, showcasing the advancements over static shapes.

**Ecosystem Flourishing with Community Contributions and Learning**: Community projects are syncing up with the latest Mojo version 24.2, with an expressed interest in creating deeper contributions through understanding MLIR dialects. Modular acknowledges this enthusiasm and plans to divulge more on internal dialects over time, adapting a progressive disclosure approach towards the complex MLIR syntax.

**Teasers and Livestreams Galore**: Modular is shedding light on their recent developments with a [livestream on YouTube](https://www.youtube.com/watch?v=PL71FV2KKHE) covering the open sourcing of Mojo's stdlib and MAX Engine support, whereas tantalizing teasers in the form of tweets [here](https://twitter.com/Modular) sustain high anticipation for impending announcements.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Quantum Leaps in Hugging Face Contributions**: New advancements have been made in AI research and applications: **HyperGraph Representation Learning** provides novel insights into data structures, **Perturbed-Attention Guidance (PAG)** boosts diffusion model performance, and the **Vision Transformer model** is adapted for medical imaging applications. The HyperGraph paper is discussed on [Hugging Face](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05), while PAG's project details are on [its project page](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) and the Vision Transformer details on [Hugging Face space](https://huggingface.co/spaces/emre570/google-vit-large-finetuned).

**Colab and Coding Mettle**: Engineers have been sharing tools and tips ranging from the use of **Colab Pro** to run large language models to the **HF professional coder assistant** for improving coding. Another shared their experience with AutoTrain, posting a [link to their model](https://huggingface.co/abhishek/autotrain-c71ux-tngfu).

**Model Generation Woes and Image Classifier Queries**: Some are facing challenges with models generating infinite text, prompting suggestions to use `repetition penalty` and `StopCriterion`. Others are seeking advice on fine-tuning a zero-shot image classifier, sharing issues and soliciting expertise in channels like #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1222883104981651517) and #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1222819278810779658).

**Community Learning Announcements**: The **reading-group** channel's next meeting has a confirmed date, strengthening community collaboration. Interested parties can find the [Discord invite link](https://discord.gg/hqrZjkjJaq?event=1222215283343622276) to participate in the group discussion. 

**Real-Time Diffusion Innovations**: Marigold's depth estimation pipeline for diffusion models now includes a [LCM function](https://huggingface.co/spaces/prs-eth/marigold-lcm), and an improvement allows real-time image transitions at 30fps for 800x800 resolution. Questions on the [labmlai diffusion repository](https://github.com/labmlai/annotated_deep_learning_paper_implementations) indicate ongoing interest in optimizing these models.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Voices for AI Training Debated**: Some participants expressed concern that professional voice actors may shy away from contributing to AI projects like [11labs](https://11labs.io/) due to negative sentiments towards AI, suggesting that amateur voices might suffice for training purposes where emotional depth isn't crucial.
  
- **Benchmarks Under Scrutiny**: There is criticism regarding the reliability of AI benchmarks, with the [HuggingFace Chatbot Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) being recommended for its human evaluation element. This conversational note calls into question the effectiveness of prevalent benchmarking methods in the AI field.

- **Jamba Ignites Hype**: AI21 Labs has introduced **Jamba**, a cutting-edge model fusing SSM-Transformer architecture, which has shown promise in excelling at benchmarks. The discussion revolved around this innovation and its implications on model performance ([announcement link](https://www.ai21.com/blog/announcing-jamba)).

- **Confusion and Clarification on Diffusion Models and Transformers**: A member pointed toward a YouTube video that they believe misinterprets how diffusion transformers work, particularly concerning the use of attention mechanisms ([YouTube video](https://www.youtube.com/watch?v=OPqFpm3wksY)). Calls were made for better explanations of transformers within diffusion models, highlighting the need for simpler "science speak" breakdowns.


---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Intelligent RAG Optimizations Emerge**: [Cohere](https://t.co/MCnmhQGS7g) is enhancing **Retrieval-Augmented Generation (RAG) pipelines** by introducing Int8 and Binary Embeddings, offering a significant reduction in memory usage and operational costs.
- **PDF Parsing Puzzles Addressed by Tech**: While parsing PDFs poses more complexity than Word documents, the community spotlighted **LlamaParse** and **Unstructured** as useful tools to tackle challenges, particularly with tables and images.
- **GenAI Gets a Data Boost from Fivetran**: Fivetran's integration for GenAI apps simplifies data management and streamlines engineering workflows, as detailed in their [blog post](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain).
- **Beyond Fine-Tuning: Breakthroughs in Model Alignment**: RLHF, DPO, and KTO alignment techniques are proving to be advanced methodologies for improved language generation in **Mistral and Zephyr 7B models**, per insights from a [blog post on the topic](https://blog.premai.io/model-alignment-process/).
- **LLM Repository Ripe for Research**: The mission of [Shure-dev](https://shure-dev.github.io/) is to compile a robust repository of **LLM-related papers**, providing a significant resource for researchers to access a breadth of high-quality information in a rapidly evolving field.



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**International Shipping Hacks for O1 Light**: Engineers explored workarounds for **international delivery of the O1 light**, including buying through US contacts. It was noted that O1 devices built by users are functional globally.

**Local LLMs Cut API Expenses**: There's active engagement around using **Open Interpreter in offline mode** to eliminate API costs. Contributions for running it with local models such as LM Studio were detailed, including running commands like *`interpreter --model local --api_base http://localhost:1234/v1 --api_key dummykey`*, and can be referenced in the [official documentation](https://docs.openinterpreter.com/guides/running-locally).

**Calls for Collaboration on Semantic Search**: A call to action was issued for improving **local semantic search** within the **[OpenInterpreter/aifs](https://github.com/OpenInterpreter/aifs)** GitHub repository. This highlights a community-driven approach to enhancing the project.

**Integrating O1 Light with Arduino's Extended Family**: Technical discussions looked at merging **O1 Light with Arduino** hardware for greater utility. While ESP32 is standard, there's eagerness to experiment with alternatives like Elegoo boards.

**O1 Dev Environment Installation Windows Woes**: Members reported and discussed issues with installing the **01 OS on Windows systems**. A [GitHub pull request](https://github.com/OpenInterpreter/01/pull/192) aims to provide solutions and streamline the setup process for Windows-based developers.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**Compilers Confront CUDA**: While the debate rages on the merits of using **compiler technology** like PyTorch/Triton versus manual CUDA code creation, members also sought guidance on **CUDA courses**, including recommendations for the [CUDA mode on GitHub](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses) and *Udacity's Intro to Parallel Programming* available on [YouTube](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2). A community-led **CUDA course** by Cohere titled **Beginners in Research-Driven Studies (BIRDS)** was announced, starting April 5th, [advertised on Twitter](https://x.com/CohereForAI/status/1773419415406809432).

**Windows Walks with WSL**: Several members provided ease-of-use solutions for running CUDA on Windows, emphasizing **Windows Subsystem for Linux (WSL)**, particularly **WSL2**, supported by a helpful [Microsoft guide](https://learn.microsoft.com/en-us/windows/wsl/install).

**Circling the Ring-Attention Revolution**: In the #[ring-attention] channel, a misalignment of fine-tuning experiments with ring-attention goals halted progress, but insights on resolving `modeling_llama.py` loss issues spearheaded advancements. The successful training of tinyllama models with extended context lengths up to 100k on substantial A40 VRAM was a hot topic, alongside a [Reddit discussion](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/) on the hefty VRAM needs for Llama 7B models with QLoRA and LoRA.

**Triton Tangle Untangled**: The #[triton-puzzles] channel was abuzz with a sync issue in `triton-viz` linked to a specific [pull request](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21), and an official fix was provided, though some still faced installation woes. The use of Triton on Windows was also clarified, pointing to alternative environments like Google Colab for running Triton-based computations.

**Zhihu Zeal Over Triton**: A member successfully pierced the language barrier on the Chinese platform [Zhihu](https://www.zhihu.com/signin?next=%2F) to unearth a trove of Triton materials, stimulating a wish for a glossary of technical terms to aid in navigating non-English content.



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**Model Mania: OpenRouter Introduces App Rankings**: OpenRouter launched an **App Rankings** feature for models, exemplified by [Claude 3 Opus](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps), to showcase utilization and popularity based on public app usage and tokens processed.

**Databricks and Gemini Pro Stir Excitement, But Bugs Buzz**: Engineers shared enthusiasm for Databricks' **DBRX** and **Gemini Pro 1.5**, although issues like **error 429** suggested rate-limit challenges, and downtime coupled with **error 502** and **error 524** signaled areas for reliability improvements in model availability.

**Claude's Capabilities and API Discuss**ed: The community clarified that **Claude in OpenRouter** doesn't support prefill features and explored error fixing for **Claude 2.1**. A side conversation praised **ClaudeAI via OpenRouter** for better handling roleplay and sensitive content with fewer false positives, noting standardized access and cost parity with official **ClaudeAI API**.

**APIs and Clients Get a Tune-Up**: OpenRouter has simplified their API to **/api/v1/completions** and shunned Groq for Nitro models due to rate limitations, alongside improvements in OpenAI's API client support.

**Easing Crypto Payments for OpenRouter Users**: OpenRouter is slashing gas costs of cryptocurrency transactions by harnessing Base chain, an Ethereum L2 solution, aiming for more economical user experiences.



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Joins the AI Fray**: AI21 Labs has launched [Jamba](https://www.ai21.com/jamba), an advanced language model integrating Mamba with Transformer elements, marked by a **256K context window** and single GPU hosting for 140K contexts. It's available under the Apache 2.0 license on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) and received coverage on [TechCrunch](https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/).

- **Groundbreaking Throughput Announced**: Jamba achieves a 3X throughput improvement in long-context tasks such as Q&A and summarization, setting a new performance benchmark in the generative AI space.

- **AI21 Labs Platform Ready for Jamba**: The SaaS platform of AI21 Labs will soon feature Jamba, complementing its current availability through Hugging Face, and staff have indicated a forthcoming deep-dive white paper detailing Jamba's training specifics.

- **Jamba, the Polyglot**: Trained on multiple languages including **German, Spanish, and Portuguese**, Jamba's multilingual prowess is confirmed, although its efficacy in Korean language tasks seems to be in question with no Hugging Face Space demonstration planned.

- **Pricing and Performance Tweaks**: AI21 Labs has removed the $29 monthly minimum charge for model use, and discussions regarding Jamba's high efficiency reveal that its mix of transformer and Mixture-of-Experts (MoE) layers enables sub-quadratic scaling. Community debates touched on quantization for memory reduction and clarified that Jamba's MoE consists of Mamba blocks, not transformers.



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Debugging Delight**: An engineer shared their *meme-inducing madness* while dealing with a **Conda metal compiler bug**, indicating high levels of frustration with debugging the issue.
- **Tinygrad's Memory Management**: The topic of implementing virtual memory in the tinygrad software stack was brought up with interest in adapting it for cache/MMU projects.
- **Comparing CPU Giants**: Users debated the responsiveness of AMD's versus Intel's customer service, highlighting Intel's better GPU price-performance and pointing out software issues with AMD.
- **Optimization Opportunities for Intel Arc**: Suggestions arose regarding potential performance enhancements for the Intel Arc by optimizing transformers/dot product attention within the IPEX library.
- **FPGA-Driven Tinygrad**: An inquiry about extending [tinygrad to leverage FPGA](https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md) was made, engaging users in potential hardware acceleration benefits.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI's Prepayment Explainer**: OpenAI's adoption of a prepayment model is to combat fraud and offer clearer pathways to increased rate limits, with [LoganK's tweet](https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) providing insider perspective. Despite its advantages, comparisons are drawn to issues faced by consumers with Starbucks' gift card model, where unspent balances contribute significantly to revenue.

- **AI21's Jamba Roars**: AI21's new "Jamba" model merges Structured State Space and Transformer architecture, boasting an MoE structure and currently evaluated on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1). However, the community noted the lack of straightforward mechanisms for testing the newly announced model.

- **xAI's Grok-1.5 Making Strides in AI Reasoning**: The newly launched Grok-1.5 model by xAI is recognized for outperforming its predecessor, especially in coding and mathematical tasks, with the community anticipating its release on the xAI platform. Benchmarks and performance details were shared via the [official blog post](https://x.ai/blog/grok-1.5).

- **Ternary LLMs Stir BitNet Debate**: The community engaged in a technical debate over the correct terminology for LLMs using three-valued logic, discussing the impact of BitNet and emphasizing that these are natively trained models rather than quantized versions.

- **Stability AI's Shakeup**: Emad Mostaque's exit from Stability AI sparked conversations about leadership changes in the generative AI space, fueled by insights from his interview on [Peter Diamandis's YouTube channel](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa) and a detailed backstory found on [archive.is](https://archive.is/8QkSl). These discussions reflect ongoing shifts and potential volatility in the industry.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Jamba Sets the Bar High**: **AI21 Labs** has introduced a new model, **Jamba**, featuring a **256k token context window**, optimized for performance with **12 billion parameters active**. Their accelerated progress is underlined by the quick training time, with knowledge cutoff on March 5, 2024, and details can be found in their [blog post](https://www.ai21.com/blog/announcing-jamba).

**Pushing the Boundaries of Optimization**: Discussions have highlighted the effectiveness of **bf16 precision** in **torchTune** leading to substantial memory savings over fp32, with these optimizations being applied to SGD and soon to the Adam optimizer. Skepticism remains over whether Axolotl provides the same level of training control as torchTune, particularly in the context of memory optimization.

**The Cost of Cutting-Edge**: Conversations around the **GB200-based server prices** revealed a steep cost of US$2-$3 million each, prompting a consideration of alternative hardware solutions by the community due to the high expenses.

**Size Matters for Datasets**: The hunt for **long-context datasets** prompted sharing of resources including one from Hugging Face's collections and the [MLDR dataset on Hugging Face](https://huggingface.co/datasets/Shitao/MLDR), which cater to models requiring extensive sequence training.

**Fine-Tuning Finesse and Repetition Debate**: The community has been engaging in detailed discussions about model training, with a focus on strategy sharing like `▀` and `▄` usage in prompts and debates over dataset repetition's utility, referencing a [paper on data ordering](https://arxiv.org/abs/2310.10638) to support repetition. New fine-tuning approaches for larger models like Galore are also being experimented with, despite some memory challenges.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**OpenGPTs: DIY Food Ordering System**: A resourceful engineer integrated a *custom food ordering API* with **OpenGPTs**, capturing the adaptability and potential of LangChain's open-source platform, showcased in a [demonstration video](https://youtu.be/V1SKJfE35D8). They encouraged peer reviews to refine the innovation.

**A Smarter SQL AI Chatbot**: Members explored methods to enable an **SQL AI Chatbot** to *remember* previous interactions, enhancing the bot’s context-retaining abilities for more effective and coherent dialogues.

**Gearing Up for Product Recommendations**: Engineers discussed the development of a bot that would suggest products using natural language queries, considering the use of vector databases for semantic search or employing an SQL agent to parse user intents like "planning to own a pet."

**Upgrade Your Code Reviews With AI**: A new AI pipeline builder designed to automate code review tasks including validation and security checks was introduced, coupled with [a demo](https://www.youtube.com/watch?v=6kfr1lqw2gg) and [a product link](https://gitgud.autonoma.app), poised to streamline the code review process.

**GalaxyAI Throws Down the Gauntlet**: GalaxyAI is providing *free access* to elite AI models such as **GPT-4** and **Gemini-PRO**, presented as an easy-to-adopt option for projects via their OpenAI-compatible [API service](https://galaxyapi.onrender.com).

**Nurturing Engineer Dialogues**: The creation of the <#1222928565117517985> channel fosters concentrated discussion on **OpenGPTs** and its growth, as evidenced by its [GitHub repository](https://github.com/langchain-ai/opengpts).



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Jamba Jumps Ahead**: AI21 Studios unveils [Jamba](https://www.ai21.com/blog/announcing-jamba), a new model merging Mamba's Structured State Space with Transformers, delivering a whopping **256K context window**. The model is not only touted for its performance but is also accessible, with open weights under Apache 2.0 license.

- **Jamba's Specs Spark Interest**: Jamba's hybrid design has prompted discussions over its 12B parameters focused on inference and its total 52B parameter size, leveraging a **MoE** framework with 16 experts and only 2 active per token, as found on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1).

- **Dissecting Jamba's DNA**: Conversations surfaced around Jamba's true architectural definition, debating whether it should be dubbed a "striped hyena" due to its hybrid nature and specific incorporation of attention mechanisms.

- **Qwen Redefines Efficiency**: The hybrid model `Qwen1.5-MoE-A2.7B` is recognized for matching the prowess of 7B models, with a much leaner 2.7B parameters, a feat of efficiency highlighted with resources including its [GitHub repo](https://github.com/QwenLM/Qwen1.5) and [Hugging Face space](https://huggingface.co/Qwen).

- **Microsoft Gains Genius, Databricks on Megablocks**: Liliang Ren hops to Microsoft GenAI as a Senior Researcher to construct scalable neural architectures evident from his [announcement](https://x.com/liliang_ren/status/1773118751413596588?s=46), while Megablocks gets a new berth at Databricks, enhancing its long-term developmental prospects mentioned on [Twitter](https://x.com/tgale96/status/1773342375806374307?s=46).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**AI21 Labs Cooks Up Jamba**: **AI21 Labs** has launched **Jamba**, a model blending Structured State Space models with Transformer architecture, promising high performance. Check out **Jamba** through its [Hugging Face deployment](https://huggingface.co/ai21labs/Jamba-v0.1), and read about its groundbreaking approach on the [AI21 website](https://www.ai21.com/jamba).

**Translation Titans Tussle**: Members are gearing up for a translation battle among **DiscoLM**, **Occiglot**, **Mixtral**, **GPT-4**, **DeepL**, and **Azure Translate**, using the first 100 lines from a dataset like *Capybara* to compare performance.

**Course to Conquer LLMs**: A [GitHub repository](https://github.com/mlabonne/llm-course) offering a course for Large Language Models with roadmaps and Colab notebooks was shared, aiming to educate on LLMs.

**Token Insertion Tangle Untangled**: A debugging success was shared regarding unexpected token insertions believed to be caused by either **quantization** or the **engine**; providing a `added_tokens.json` resolved the anomaly.

**Training Data Transparency Tremors**: The community has asked for more information on the training data used for a certain model, with specific interest in the definition and range of "English data" as stated in the model card or affiliated blog post.





---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Call for Python Prodigies**: A member inquired about the upcoming **onboarding schedule** for Python developers keen on joining the project, indicating that sessions tailored for experienced Python talent were previously in the pipeline.

- **YouTube Educational Content Shared**: Pradeep1148 shared a [YouTube video link](https://www.youtube.com/watch?v=LWz2QaSRl2Y), but no context was provided regarding the content or relevance to the ongoing discussions.




---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1222804942134185986)** (936 messages🔥🔥🔥): 

- **Turbo Boost for SD3**: The idea of finetuning SD3 lingers, with a hypothesis that the [SD3 Turbo](https://arxiv.org/pdf/2403.12015.pdf) might serve as a powerful, detailed model akin to SDXL Lightning. Members speculate on the performance benefits of running the upcoming SD3 on hardware with 12GB VRAM.

- **AI Art and Politics**: The discussion covers the balance between model power and public access, with concerns over potential downsizing of SD3 features for public use. There's hope for providing fine-tuning tools and control nets at launch, which could keep the user base productive.

- **LoRA's Future with SD3**: Conversations revolve around training LoRA (Local Rank Aware) models on SD3, with the likelihood that 24GB of VRAM should be sufficient. There are uncertainties about how quickly and effectively these tools will be adapted to SD3 post-launch.

- **Training on Hardware**: Users discuss the feasibility of training smaller sized LoRA models on GPUs that have 16GB VRAM, with some managing to achieve this on resolutions as high as 896 with success. Techniques and optimisations for training, such as Aggressive train rates, network dropouts, and ranking are shared.

- **Leveraging New Models for Growth**: Members showcase and recommend using Arc2Face for face manipulation, admiring its demo and capabilities. Discussions are sprinkled with irony as users mock the potential of technology being perceived as alien intervention and trivialize the controversy over dataset censorship.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://brianfitzgerald.xyz/prompt-augmentation/">SuperPrompt - Better SDXL prompts in 77M Parameters | Brian Fitzgerald</a>: Left SDXL output with SuperPrompt applied to the same input prompt.</li><li><a href="https://arc2face.github.io/">Arc2Face: A Foundation Model of Human Faces</a>: no description found</li><li><a href="https://svilentodorov.xyz/blog/gpt-15b-chat-finetune/">Talking to Myself or How I Trained GPT2-1.5b for Rubber Ducking using My Facebook Chat Data</a>: Previously in this series - finetuning 117M, finetuning 345M</li><li><a href="https://huggingface.co/spaces/FoivosPar/Arc2Face">Arc2Face - a Hugging Face Space by FoivosPar</a>: no description found</li><li><a href="https://tenor.com/view/so-sayweall-battlestart-galactica-william-adama-bsg-twelve-colonies-gif-18616973">So Sayweall Battlestart Galactica GIF - So Sayweall Battlestart Galactica William Adama - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://toyxyz.gumroad.com/l/ciojz">Character bones that look like Openpose for blender _ Ver_96  Depth+Canny+Landmark+MediaPipeFace+finger</a>: -Blender version 3.5 or higher is required.- Download — blender.orgGuide : https://youtu.be/f1Oc5JaeZiwCharacter bones that look like Openpose for blender Ver96 Depth+Canny+Landmark+MediaPipeFace+fing...</li><li><a href="https://leonardo.ai/">Home v2</a>: Transform your projects with our AI image generator. Generate high-quality, AI generated images with unparalleled speed and style to elevate your creative vision</li><li><a href="https://www.youtube.com/watch?v=QdRP9pO89MY">Stable Diffusion explained (in less than 10 minutes)</a>: Curious about how Generative AI models like Stable Diffusion work? Join me for a short whiteboard animation where we will explore the basics. In just under 1...</li><li><a href="https://www.youtube.com/watch?v=5_qKLZq6frg">Beast Wars: Transformers - Theme Song | Transformers Official</a>: Subscribe to the Transformers Channel: https://bit.ly/37mNTUzAutobots, roll out! Welcome to Transformers Official: The only place to see all your favorite Au...</li><li><a href="https://github.com/foivospar/Arc2Face">GitHub - foivospar/Arc2Face: Arc2Face: A Foundation Model of Human Faces</a>: Arc2Face: A Foundation Model of Human Faces. Contribute to foivospar/Arc2Face development by creating an account on GitHub.</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus: Focus on prompting and generating</a>: Focus on prompting and generating. Contribute to lllyasviel/Fooocus development by creating an account on GitHub.</li><li><a href="https://github.com/Nick088Official/SuperPrompt-v1">GitHub - Nick088Official/SuperPrompt-v1: SuperPrompt-v1 AI Model (Makes your prompts better) both Locally &amp; Online</a>: SuperPrompt-v1 AI Model (Makes your prompts better) both Locally &amp; Online - Nick088Official/SuperPrompt-v1</li><li><a href="https://civitai.com/models/9513/fnaf-multi-character-lora">FNAF - Multi-Character LoRA - v1.11 | Stable Diffusion LoRA | Civitai</a>: Multi-character FNAF LoRA. Earlier versions were for Yiffy-e18, current version is for Pony Diffusion V3. Latest version contains Classic Freddy, C...</li><li><a href="https://github.com/Vargol/StableDiffusionColabs">GitHub - Vargol/StableDiffusionColabs: Diffusers Stable Diffusion script that run on the Google Colabs&#39; free tier</a>: Diffusers Stable Diffusion script that run on the Google Colabs&#39; free tier - Vargol/StableDiffusionColabs</li><li><a href="https://github.com/Vargol/8GB_M1_Diffusers_Scripts">GitHub - Vargol/8GB_M1_Diffusers_Scripts: Scripts demonstrating how to run Stable Diffusion on a 8Gb M1 Mac</a>: Scripts demonstrating how to run Stable Diffusion on a 8Gb M1 Mac - Vargol/8GB_M1_Diffusers_Scripts
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1222985154033619026)** (2 messages): 

- **DBRX Debuts on Perplexity Labs**: The state-of-the-art language model **DBRX** from Databricks is now available on Perplexity Labs, outperforming **GPT-3.5** and rivaling **Gemini 1.0 Pro** especially in math and coding tasks. Users are invited to try DBRX for free at [Perplexity Labs](https://labs.pplx.ai).

- **Copy AI and Perplexity Forge a Powerful Partnership**: Copy AI integrates Perplexity's APIs to create an AI-powered go-to-market (GTM) platform, providing real-time market insights for improved decision-making. As an added perk, Copy AI users get 6 months of Perplexity **Pro** for free - details on the collaboration can be found in their [blog post](https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams).

**Link mentioned**: <a href="https://www.copy.ai/blog/copy-ai-perplexity-purpose-built-partners-for-gtm-teams">Copy.ai + Perplexity: Purpose-Built Partners for GTM Teams | Copy.ai</a>: Learn more about how Perplexity and Copy.ai&#x27;s recent partnership will fuel your GTM efforts!

  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1222807897013092364)** (728 messages🔥🔥🔥): 

- **Perplexity Search & Focus Modes**: Discussions reveal confusion among users whether **academic focus mode** is functioning properly on Perplexity, noting potential outages with [semantic search](https://www.perplexity.ai/search/I-am-a-XNNmFl83Rry_HEhBb3k8Tw). Users recommend turning off focus mode or trying **focus all** as potential workarounds.

- **Exploring Pro Search and "Pro" Features**: Conversation around **Pro Search** suggests some users find the follow-up questions redundant, preferring immediate answers. Pro users receive $5 credit for the API and access to multiple AI models, including the ability to attach sources like PDFs persistently in threads for referencing.

- **Understanding Perplexity's File Context and Sources**: Through testing, users found the context for files on Perplexity differs from direct uploads vs. text input, with speculation that **Retrieval Augmented Generation (RAG)** is utilized—Claude may rely on other models like **GPT-4-32k** for file processing and referencing.

- **Generative Capabilities & API Utilization**: Users discuss the costs and benefits of using the Perplexity API vs. subscriptions to services like **GPT Plus** and **Claude Pro**. Models like **Opus** and other subscription-based models are mentioned for specific tasks, such as generating content ideas, with users seeking guidance on when to use which service or feature.

- **Clarifying Claude Opus and GPT-4.5**: User inquire about the accuracy and authenticity of **Claude Opus** responses, noting it has incorrectly identified itself as an OpenAI development in outputs. Participants suggest caution when interpreting such AI self-descriptions and to refrain from querying AI about their origins.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://trysora.io/">Sora AI</a>: no description found</li><li><a href="https://x.com/perplexity_ai/status/1773418423726305529?s=20">Tweet from Perplexity (@perplexity_ai)</a>: DBRX, the state-of-the-art open LLM from @databricks is now available on Perplexity Labs. Outperforming GPT-3.5 and competitive with Gemini 1.0 Pro, DBRX excels at math and coding tasks and sets new b...</li><li><a href="https://x.ai/blog/grok-1.5">Announcing Grok-1.5</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/pricing">Pricing</a>: no description found</li><li><a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://tenor.com/view/rickroll-meme-internet-never-gonna-gif-26474110">Rickroll Meme GIF - Rickroll Meme Internet - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/lost-go-back-jack-we-have-to-go-back-gif-5469258">Lost Go Back GIF - Lost Go Back Jack - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://tenor.com/view/older-meme-checks-out-gif-14849207">Older Meme Checks Out GIF - Older Meme Checks Out - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bq5aw1/working_on_open">Reddit - Dive into anything</a>: no description found</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: no description found</li><li><a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">Introducing DBRX: A New State-of-the-Art Open LLM | Databricks</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=GanTUWLUUWQ">Answer Engine Tutorial: The Open-Source Perplexity Search Replacement</a>: Answer Engine install tutorial, which aims to be an open-source version of Perplexity, a new way to get answers to your questions, replacing traditional sear...</li><li><a href="https://youtu.be/57LqvutrOI8?si=x7hGU42L_z3KwmX_">Perplexity CEO: Disrupting Google Search with AI</a>: One of the most preeminent AI founders, Aravind Srinivas (CEO, Perplexity), believes we could see 100+ AI startups valued over $10B in our future. In the epi...</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bq5aw1/working_on_opensource_alternative_to_perplexityai/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics">GitHub - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics: Towards World&#39;s Most Comprehensive Curated List of LLM Related Papers &amp; Repositories</a>: Towards World&#39;s Most Comprehensive Curated List of LLM Related Papers &amp; Repositories - shure-dev/Awesome-LLM-related-Papers-Comprehensive-Topics</li><li><a href="https://docs.anthropic.com/claude/docs/let-claude-think">Let Claude think</a>: no description found
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1222809690124320798)** (22 messages🔥): 

- **Exploration of Perplexity's Capabilities**: Various members are exploring and sharing [Perplexity AI](https://www.perplexity.ai) search results on topics such as **Truth Social**, *Anthropic tag models*, Amazon's investment strategies, Exchange administration intricacies, and global warming research implications.
- **Delving into Detailed Analyses**: One member has been experimenting with `<scratchpad>` XML tags in Perplexity AI to demonstrate differences in search results, specifically illustrating the **effectiveness in structuring detailed outputs**.
- **Space Enthusiasm on Display**: Members shared their curiosity about space technologies and phenomena through Perplexity searches, inquiring into details about **SpaceX's Starship chunks** and how **space impacts the human body**.
- **Tech and Finance Merge in User Queries**: The topics of investment by Amazon and the story of FTX's co-founder, **Sam Bankman-Fried**, were brought into focus, reflecting users' interests in the intersection of technology and finance.
- **Scratchpad Strategy Sharpening**: By employing **improved scratchpad strategies**, a member claims to have significantly enhanced the content quality of final outputs, highlighting the potential advancements in extracting **more explorative and useful content** from AI-generated text.
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1222855722119991327)** (8 messages🔥): 

- **API Results Inconsistency**: A member reports that they often receive no results when using the **Perplexity AI API**, contrasting with the "plenty of results" on the web interface when searching for information, such as "Olivia Schough spouse."
- **Seeking API Parameter Guidance**: The same member inquires if there are more parameters to guide the API to improve results.
- **Comparison of API and Web Interface Strengths**: The member expresses an opinion that the web app seems "way better" than the API in terms of performance.
- **Sources Lacking in API Responses**: Another user highlights a discrepancy: the API's returned response lacks the variety of sources that are present when using Perplexity AI on the web.
- **Beta Features and Application Process**: In response, a user redirects members to a previous message about **URL citations still being in beta** and provides a link to apply for it: [Apply for beta features](https://perplexity.typeform.com/to/j50rnNiB).

**Link mentioned**: <a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.

  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1222793845616410735)** (351 messages🔥🔥): 

- **Jamba's Potential and Limitations Discussed**: Members discussed the newly announced [Jamba model by AI21 Labs](https://www.ai21.com/blog/announcing-jamba), noting its claim of blending Mamba SSM technology with traditional Transformers. Some were impressed with its performance and context window, while others were skeptical about its tokenization and absence of detailed token counts.

- **Concerns Over Various Models**:
  Members shared mixed experiences with AI tools like Copilot and claude, noting a drastic drop in Copilot quality, while some found claude low-key helpful. Concerns were raised regarding Model Merging tactics, with a mention of potentially discussing Argilla in the future.

- **Evaluation of Generative Models**:
  Discussion centered on how to evaluate open-ended generation quality, with links shared to resources that delve into perplexity, pairwise ranking, and metrics linked to apparent emergent abilities, such as [this emergence as a myth paper](https://arxiv.org/abs/2304.15004).

- **Model Training Challenges**:
  Users shared experiences and sought advice on training AI models, particularly issues when fine-tuning Mistral and using SFTTrainer. They exchanged configurations and debated the optimal number of training epochs and batch sizes.

- **multi-GPU Support and Other Technical Difficulties Addressed**:
  Concerns were raised about the lack of multi-GPU support with Unsloth AI, along with issues related to bitsandbytes dependency and error messages during fine-tuning. Advice and troubleshooting tips were shared among members.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://mobiusml.github.io/1bit_blog/">1-bit Quantization</a>: A support blog for the release of 1-bit Aana model.</li><li><a href="https://arxiv.org/abs/2304.15004">Are Emergent Abilities of Large Language Models a Mirage?</a>: Recent work claims that large language models display emergent abilities, abilities not present in smaller-scale models that are present in larger-scale models. What makes emergent abilities intriguin...</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit">unsloth/mistral-7b-v0.2-bnb-4bit · Hugging Face</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://huggingface.co/yanolja/EEVE-Korean-10.8B-v1.0#technical-deep-dive">yanolja/EEVE-Korean-10.8B-v1.0 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 is climbing on the AlpacaEval leaderboard, outperforming all of the latest open-source models. </li><li><a href="https://tenor.com/view/come-look-at-this-gif-21207051">Come Look At This GIF - Come Look At This - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://download.pytorch.org/whl/cu121">no title found</a>: no description found</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: no description found</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://gist.github.com/idncsk/8a05711b57fc0b9b2ee3186fcbb43c25">trainer-v1.py</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://sambanova.ai/blog/benchmarking-samba-1">Benchmarking Samba-1</a>: Benchmarking Samba-1 with the EGAI benchmark - a comprehensive collection of widely adapted benchmarks sourced from the open source community. </li><li><a href="https://apps.sambanova.ai/sambachat">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=2-SPH9hIKT8">A little guide to building Large Language Models in 2024</a>: A little guide through all you need to know to train a good performance large language model in 2024.This is an introduction talk with link to references for...</li><li><a href="https://huggingface.co/spaces/PingAndPasquale/MOE-LLM-GPU-Poor-Leaderboard">MOE-LLM-GPU-POOR_LEADERBOARD - a Hugging Face Space by PingAndPasquale</a>: no description found</li><li><a href="https://github.com/potsawee/selfcheckgpt">GitHub - potsawee/selfcheckgpt: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models</a>: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models - potsawee/selfcheckgpt</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/trainer">Trainer</a>: no description found
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1222881502019194961)** (34 messages🔥): 

- **Andrej Karpathy's AI Analogy**: A [YouTube video](https://www.youtube.com/watch?v=c3b-JASoPi0) featuring Andrej Karpathy discusses LLMs in analogy to operating systems, citing important aspects of **finetuning** and the misconception of labeling Mistral and Gemma as open source.
- **Finetuning's Fine Balance**: It's highlighted that for effective finetuning, a blend of the model’s original training data and new data is essential; otherwise, a model's proficiency might regress.
- **Missed Opportunity at AI Panel**: Dialogues revealed disappointment as a non-technical question about Elon Musk's management style took precedence over deeper AI topics during a public panel featuring AI expert Andrej Karpathy.
- **Reinforcement Learning Critique**: Andrej Karpathy criticizes current RLHF techniques, comparing them unfavorably to AlphaGo's training, suggesting that we need more fundamental methods for training models, akin to textbook exercises.
- **Snapdragon's New Chip in the Spotlight**: Discussion of the new Snapdragon X Elite chip's benchmarks and capabilities surfaced, with comparisons to existing technology and hopes for future advancements in processor performance shared alongside a [benchmark review video](https://youtu.be/dTCm6BupWEQ).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=c3b-JASoPi0">Making AI accessible with Andrej Karpathy and Stephanie Zhan</a>: Andrej Karpathy, founding member of OpenAI and former Sr. Director of AI at Tesla, speaks with Stephanie Zhan at Sequoia Capital&#39;s AI Ascent about the import...</li><li><a href="https://youtu.be/dTCm6BupWEQ">Now we know the SCORE | X Elite</a>: Qualcomm&#39;s new Snapdragon X Elite benchmarks are out! Dive into the evolving ARM-based processor landscape, the promising performance of the Snapdragon X Eli...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1222794356579369041)** (201 messages🔥🔥): 

- **Tokenizer Troubles**: User encountered a `list index out of range` error using the `get_chat_template` function from unsloth with the Starling 7B tokenizer. This suggests potential tokenization issues with Starling's implementation.
- **Finetuning Frustrations**: One member discussed difficulties with the finetuning process, describing various parameters tried without desired results. Suggestions from others included fixing the rank and experimenting with different alpha rates and learning rates.
- **Checkpoint Challenges**: Concerns about unsloth's need for internet access during training were raised, especially when internet instability occurs, potentially interfering with checkpoint saving. Unsloth doesn't inherently need internet connection to train, and issues may relate to Hugging Face model hosting or WandB reporting.
- **Pushing Checkpoints to Huggingface**: A member asked how to push local checkpoints to Hugging Face, with the response pointing towards using Hugging Face's `hub_token` for cloud checkpoints and `save_strategy` for local storage.
- **Overfitting Overhead**: A user sought advice for a model that overfits, with recommendations provided to reduce the learning rate, lower the LoRA rank and alpha, shorten the number of epochs, and include an evaluation loop to stop training if loss increases.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharin">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://ko-fi.com/unsloth">Support Unsloth AI on Ko-fi! ❤️. ko-fi.com/unsloth</a>: Support Unsloth AI On Ko-fi. Ko-fi lets you support the people and causes you love with small donations</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template-variables">ollama/docs/modelfile.md at main · ollama/ollama</a>: Get up and running with Llama 2, Mistral, Gemma, and other large language models. - ollama/ollama</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/toranb/sloth">GitHub - toranb/sloth: python sftune, qmerge and dpo scripts with unsloth</a>: python sftune, qmerge and dpo scripts with unsloth - toranb/sloth</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.hub_token">Trainer</a>: no description found</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L327),">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L540)">transformers/src/transformers/models/llama/modeling_llama.py at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers</li><li><a href="https://youtu.be/rANv5BVcR5k?si=g3VOwbGUFCWaLWd3">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: Discover the secrets to effortlessly fine-tuning Language Models (LLMs) with your own data in our latest tutorial video. We dive into a cost-effective and su...</li><li><a href="https://www.youtube.com/live/g68qlo9Izf0?si=X3dDHSeeqOCV6WN6">Efficient Fine-Tuning for Llama-v2-7b on a Single GPU</a>: The first problem you’re likely to encounter when fine-tuning an LLM is the “host out of memory” error. It’s more difficult for fine-tuning the 7B parameter ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1222911167232475256)** (6 messages): 

- **Tinyllama Leverages Unsloth Notebook**: The Lora Adapter from Unsloth's notebook got converted to a ggml adapter to be used in the Ollama model, resulting in the [Tinyllama model](https://ollama.com/pacozaa/tinyllama-alpaca-lora) with training emerged from the Unsloth Notebook and a dataset from Huggingface.
- **Mischat Model Unveiled**: Another model named Mischat, which leverages gguf, is now updated on Ollama. It involved fine-tuning with the [Mistral-ChatML Unsloth notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) and can be found at [Ollama link](https://ollama.com/pacozaa/mischat).
- **Showcasing Template Reflection in Models**: A user showcased how templates defined in notebooks can reflect in Ollama modelfiles, using the recent models as examples of this process. 
- **Blogpost for AI Enthusiasts**: A member introduced a new blog post titled 'AI Unplugged 5' covering a variety of AI topics ranging from Apple MM1 to DBRX to Yi 9B which offers insight into recent AI developments and available at [AI Unplugged](https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple).
- **Positive Reception for AI Summary Content**: The blog summarizing weekly AI activities received positive feedback from another user who appreciated the effort to provide insights on AI advancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://ollama.com/pacozaa/mischat">pacozaa/mischat</a>: Model from fine-tuning session of Unsloth notebook ChatML with Mistral Link to Note book: https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing</li><li><a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>: Tinyllama Train with Unsloth Notebook, Dataset https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple">AI Unplugged 5: DataBricks DBRX, Apple MM1, Yi 9B, DenseFormer, Open SORA, LlamaFactory paper, Model Merges.</a>: Previous Edition Table of Contents Databricks DBRX Apple MM1 DenseFormer Open SORA 1.0 LlaMaFactory finetuning ananlysis Yi 9B Evolutionary Model Merges Thanks for reading Datta’s Substack! Subscribe ...</li><li><a href="https://datta0.notion.site/AI-Unplugged-c2c577fe8af54534aec540fc4a4032dd?pvs=4">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 messages): 

starsupernova: ooo very cool!
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1222822120027852913)** (237 messages🔥🔥): 

- **AI Hobbies Aren't Cheap**: Members discuss the costs associated with AI as a hobby, with expenditures reaching around $10k due to the need for high-end GPUs or MacBooks for good performance.

- **Challenges for Newcomers in AI**: Newcomers to the world of AI face a steep learning curve, with discussions highlighting the "hard awakening" to the computationally demanding and jargon-laden field.

- **Local LLM Guidance and Troubleshooting**: Several inquiries and tips were exchanged on running local LLMs, from issues with LM Studio and GPU compatibility to recommendations on model selection based on VRAM and system specifications.

- **LM Studio Enhancements and Feature Requests**: Users express excitement for new features like the Branching system, and highlight the usefulness of having folders for branched chats; some are requesting a 'Delete All' feature for smoother user experience.

- **Community Support and Shared Resources**: Helpful conversations and shared resources like YouTube guides and GitHub links provide support among users, who collaboratively troubleshoot technical challenges and explore the capabilities of different models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat">Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face</a>: no description found</li><li><a href="https://useanything.com/">AnythingLLM | The ultimate AI business intelligence tool</a>: AnythingLLM is the ultimate enterprise-ready business intelligence tool made for your organization. With unlimited control for your LLM, multi-user support, internal and external facing tooling, and 1...</li><li><a href="https://www.youtube.com/watch?v=-bVc3i9hZJg">LM Studio Realtime STT/TTS Integration Please</a>: A message to the LM Studio dev team. Please give us Realtime Speech to Text and Text to Speech capabilities. Thank you!</li><li><a href="https://www.youtube.com/watch?v=kFC-OWw7G8k">Build a Next.JS Answer Engine with Vercel AI SDK, Groq, Mistral, Langchain,  OpenAI, Brave &amp; Serper</a>: Building a Perplexity Style LLM Answer Engine: Frontend to Backend TutorialThis tutorial guides viewers through the process of building a Perplexity style La...</li><li><a href="https://youtu.be/Z5_LvCwbgqg?si=D">LM Studio: Easiest Way To Run ANY Opensource LLMs Locally!</a>: Are you ready to dive into the incredible world of local Large Language Models (LLMs)? In this video, we&#39;re taking you on a journey to explore the amazing ca...</li><li><a href="https://youtu.be/Z5_LvCwbgqg?si=DesXy5T0dg5BHVkK">LM Studio: Easiest Way To Run ANY Opensource LLMs Locally!</a>: Are you ready to dive into the incredible world of local Large Language Models (LLMs)? In this video, we&#39;re taking you on a journey to explore the amazing ca...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6074">Add qwen2moe by simonJJJ · Pull Request #6074 · ggerganov/llama.cpp</a>: This PR adds the support of codes for the coming Qwen2 MoE models hf. I changed several macro values to support the 60 experts setting. @ggerganov
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1222837018728468491)** (39 messages🔥): 

- **VRAM Requirements for Future Models**: An issue has been raised about current GPU configurations lacking the VRAM for upcoming models, specifically pointing out that **32GB VRAM might fall short for Q2 or Q3 versions** of models. [A GitHub issue](https://github.com/ggerganov/llama.cpp/issues/6344) mentions that **llama.cpp** doesn't yet support newer, VRAM-intensive model formats like DBRX.
  
- **LM Studio Pairing with Open Interpreter**: A member asked if anyone has gotten Open Interpreter to work with LM Studio, and another member linked both the [official documentation](https://docs.openinterpreter.com/language-models/local-models/lm-studio) and a [YouTube tutorial](https://www.youtube.com/watch?v=8HIatLzCJDA) to assist with setup.
  
- **Leveraging Existing Server Setups for Large LLMs**: Some members discussed using older servers with Nvidia P40 GPUs for running large language models, noting that while they are not ideal for tasks like stable diffusion, they can handle large models with decent speed.
  
- **Assessing Creative Writing Capabilities**: A few members compared different language models, noting that **Zephyr may outperform Hermes2 Pro** in creative writing tasks, highlighting the nuances in how various language models handle specific functions.

- **Choosing the Right Coder LLM for Limited VRAM**: A query about the best coder language model to use on a 24GB RTX 3090 led to a mention that **DeepSeek Coder** might be the top choice, and that instruct models might function better when used as agents.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=F0KDFRbh5h0">DBRX: My First Performance TEST - Causal Reasoning</a>: On day one of the new release of DBRX by @Databricks I did performance tests on causal reasoning and light logic tasks. Here are some of my results after the...</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=8HIatLzCJDA">LM Studio + Open Interpreter to run an AI that can control your computer!</a>: This is a crappy video (idk how to get better resolution) that shows how easy it is to use ai these days! I run mistral instruct 7b in the client and as a se...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6344">Add support for DBRX models: dbrx-base and dbrx-instruct · Issue #6344 · ggerganov/llama.cpp</a>: Prerequisites Please answer the following questions for yourself before submitting an issue. I am running the latest code. Development is very rapid so there are no tagged versions as of now. I car...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1222969151560351755)** (2 messages): 

- **LM Studio 0.2.18 Update Rolls Out**: LM Studio has released version 0.2.18, which introduces stability improvements, a new 'Empty Preset' for Base/Completion models, default presets for various LMs, a 'monospace' chat style, along with a number of bug fixes. Download the new version from [LM Studio](https://lmstudio.ai) or use the 'Check for Updates' feature.

- **Comprehensive Bug Fixes Detailed**: The update addresses issues such as duplicated chat images, ambiguous API error messages without models, GPU offload bugs, and incorrect model names in the UI. Mac users get a Metal-related load bug fixed, and Windows users see a correction for app opening issues post-installation.

- **Documentation Site Launched**: A dedicated documentation site for LM Studio is now live at [lmstudio.ai/docs](https://lmstudio.ai/docs), with plans to expand the content available in the near future.

- **Config Presets Accessible on GitHub**: Users who can't find the new configs in their setup can access them at [openchat.preset.json](https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json) and [lm_studio_blank_preset.preset.json](https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json) on GitHub, although the latest downloads and updates should already include these presets.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://lmstudio.ai/docs.">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json">configs/openchat.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json">configs/lm_studio_blank_preset.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON configuration file format and a collection of example config files. - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1222993668508745769)** (13 messages🔥): 

- **Praise for AI Ease-of-Use**: A member expressed their appreciation for an AI tool, calling it the most user-friendly AI project they have ever used.
- **VRAM Display Inaccuracies Reported**: A member noted that the VRAM displayed is incorrect, showing only *68.79GB* instead of the expected *70644 GB*.
- **Seeking Stable Models Amidst Volume**: Another member disclosed that they are sifting through almost *2TB of models*, encountering issues with many models producing garbled or repetitive text.
- **Prompt Formatting Woes**: Members discussed the importance of using the correct prompt format for models to function properly and mentioned checking the model's Hugging Face page for clues.
- **VRAM Readings Fluctuate Across Versions**: It was observed that the displayed VRAM changed across different versions of the software, with one member speculating that version *2.17* possibly showed the most accurate value.
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1222868077964820590)** (111 messages🔥🔥): 

- **Power Supply Sizing Recommendations**: Discussions suggest a **1200-1500W power supply** for an Intel 14900KS and an RTX 3090 build, whereas **1000W** might suffice for AMD setups like a 7950x/3d, considering no GPU power limits. Gold/platinum rated PSUs were endorsed for better efficiency.

- **GPU Power Limit Tips**: It was shared that locking **RTX 4090** GPUs to a 350-375W range only results in a minimal performance loss, emphasizing that such a limit is more practical than **full power overclocking's minuscule gains**.

- **Performance Enthusiasm for Legacy Hardware**: Some members discuss leveraging older GPU hardware like the **K80**, while noting limitations like high heat output. It was suggested that **old iMac fans** can be modded to help with cooling on such antique tech.

- **Exploring the Utility of NVLink**: Members exchanged ideas on whether NVLink can improve performance for model inference, with some arguing for a **noticeable speed increase in token generation** with the link, while others questioned its utility for inference tasks.

- **LM Studio Compatibility Queries**: As LM Studio updates arrive, users typically inquire about support for various GPUs, highlighting limitations and workarounds, like using **OpenCL for GPU offloading or ZLUDA for older AMD cards**. Moreover, issues are sometimes encountered and solved, such as the **UI problem fixed** by running the app in fullscreen.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://tenor.com/view/grammar-police-drift-police-car-gif-17312786">Grammar Police Drift GIF - Grammar Police Drift Police Car - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.aliexpress.com/item/1005005943893305.html">no title found</a>: no description found
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1222799944990195734)** (80 messages🔥🔥): 

- **User Proposes GPU/NPU Monitoring**: A suggestion was made to add a monitoring feature for GPU and NPU in the usage sections along with utilization statistics for better system parameter optimization.
- **GPU Offloading Troubleshooting**: Users discussed issues with **ROCm beta** on Windows not loading models into GPU memory. The conversation led to identifying a possible issue with partial offloading when not all layers are offloaded to the GPU. Users shared logs and error messages to diagnose the problem.
- **LM Studio Windows Version Confusion**: Some users reported the **LM Studio** version appears as 0.2.17 in logs, even when 0.2.18 is in use. There's discussion about an issue with adjusting the GPU offload layers slider in the UI.
- **ROCm Beta Stability Questions**: One user encountered crashes while using the **0.2.18 ROCm Beta**; it works initially but fails upon asking questions. There is a call for users to test with a debug build and send verbose logs for further investigation.
- **LM Studio on Nix via GitHub PR**: A user successfully integrated **LM Studio** into Nix, shared the Pull Request link (https://github.com/NixOS/nixpkgs/pull/290399), and indicated plans to merge the update soon.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">no title found</a>: no description found</li><li><a href="https://lmstudio.ai/rocm">👾 LM Studio - Discover and run local LLMs</a>: Find, download, and experiment with local LLMs</li><li><a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe">no title found</a>: no description found</li><li><a href="https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information">Windows 11 - release information</a>: Learn release information for Windows 11 releases</li><li><a href="https://github.com/NixOS/nixpkgs/pull/290399">lmstudio: init at 0.2.18 by drupol · Pull Request #290399 · NixOS/nixpkgs</a>: New app: https://lmstudio.ai/  Description of changes  Things done   Built on platform(s)   x86_64-linux  aarch64-linux  x86_64-darwin  aarch64-darwin   For non-Linux: Is sandboxing enabled in nix....
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1223039440469885099)** (2 messages): 

- **Check Out OllamaPaperBot's Code**: A member shared a [GitHub repository](https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py) for **OllamaPaperBot**, a chatbot designed to interact with PDF documents using open-source LLM models, inviting others to review their code.
- **Inquiring about LMStudio's JSON Output**: A member inquired if anyone has tried using the new JSON output format from LMStudio in conjunction with the shared OllamaPaperBot repository.

**Link mentioned**: <a href="https://github.com/eltechno/OllamaPaperBot/blob/main/simplechat.py">OllamaPaperBot/simplechat.py at main · eltechno/OllamaPaperBot</a>: chatbot designed to interact with PDF documents based on OpenSource LLM Models - eltechno/OllamaPaperBot

  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1222819107192307784)** (54 messages🔥): 

- **Understanding GPU Utilization with LM Studio**: Users reported issues with LM Studio where in version 0.2.17, GPUs were properly utilized, but after updating to 0.2.18, there was a noticeable drop in GPU usage under 10%. Instructions were provided for exporting app logs using **[a debug version](https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe)** to investigate the problem.

- **Geared Up for Debugging**: In response to low GPU usage, a user attempted troubleshooting with a verbose debug build and noted that GPU usage improved when LM Studio was used as a chatbot rather than a server.

- **Ejecting Models causes errors**: A user discovered a potential bug where after ejecting a model in LM Studio, they were unable to load any new models without restarting the application. Other users were unable to reproduce this failure.

- **Driver Issues May Affect Performance**: A conversation about the proper application of ROCm instead of AMD OpenCL suggested that driver updates or installing the correct build could resolve some issues. A user confirmed improvement after updating AMD drivers and installing the correct build of LM Studio.

- **Select GPUs Unrecognized by New Update**: Users observed that while LM Studio version 2.16 recognized secondary GPUs, version 2.18 no longer did. A user clarified that models started working properly after ensuring they were using the ROCm build, not the regular LM Studio build.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">no title found</a>: no description found</li><li><a href="https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709">How to run a Large Language Model (LLM) on your AMD Ryzen™ AI PC or Radeon Graphics Card</a>: Did you know that you can run your very own instance of a GPT based LLM-powered AI chatbot on your Ryzen™ AI PC or Radeon™ 7000 series graphics card? AI assistants are quickly becoming essential resou...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1223044088232808448)** (2 messages): 

- **Deepening Abstract Concepts with AI**: A member shared their interest in a paper ([Abstract Concept Inflection](https://arxiv.org/pdf/2304.01904.pdf)) which discusses using AI to expand abstract concepts into detailed ones, similar to breaking down coding tasks. They are currently experimenting with using different models as a critic or generator for a proof of concept with autogen.

- **Seeking Agent Program Recommendations for LM Studio**: A member inquired about which Agent program to use and if there are any that easily plug into LM Studio. They are looking for guidance on integrating with LM Studio.
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1223319785140387853)** (1 messages): 

- **Voice Engine Unveiled**: OpenAI has shared insights on a new model called **Voice Engine**, which requires only text and a [15-second audio sample](https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices) to generate natural-sounding speech, aiming to produce emotive and realistic voices.
- **Voice Engine Powers Text-to-Speech API**: Initially developed in late 2022, **Voice Engine** is currently used for preset voices in OpenAI's [text-to-speech API](https://platform.openai.com/docs/guides/text-to-speech) and also enhances [ChatGPT Voice and Read Aloud](https://openai.com/blog/chatgpt-can-now-see-hear-and-speak) features.
- **AI Ethics in Focus**: OpenAI emphasizes a careful approach to releasing **Voice Engine** more broadly, acknowledging potential misuse and stressing the importance of responsible synthetic voice deployment as outlined in their [AI Charter](https://openai.com/charter).

**Link mentioned**: <a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">Navigating the Challenges and Opportunities of Synthetic Voices</a>: We’re sharing lessons from a small scale preview of Voice Engine, a model for creating custom voices.

  

---


**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1222814546272587836)** (75 messages🔥🔥): 

- **Gemini Advanced Leaves Users Wanting**: Members express disappointment with **Gemini Advanced**, noting longer wait times for responses compared to other products and limited availability in Europe.
- **OpenAI Products Version Confusion**: Discord users discuss the versions of GPT-4 used in different OpenAI applications, with instances of **ChatGPT** reporting varying cutoff dates, creating confusion about model stability.
- **VoiceCraft Sparks Enthusiasm and Worry**: A link to **VoiceCraft**, a *state-of-the-art* tool for speech editing and TTS with impressive demos, attracts attention, but equally raises concerns about potential misuse for scams.
- **Legal and Ethical Discussions on Voice Cloning**: Users debate the legality of voice cloning, with some emphasizing the difficulty of preventing misuse preemptively and others pointing out the distinction between technology availability and illicit activities.
- **OpenAI's Cautious Approach to Innovation**: A conversation about OpenAI’s cautious release strategy for emerging AI technologies elicits user frustration over perceived slow progress, yet also garners understanding for the methodical approach to manage risk.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://jasonppy.github.io/VoiceCraft_web/">VoiceCraft</a>: no description found</li><li><a href="https://github.com/jasonppy/VoiceCraft">GitHub - jasonppy/VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild</a>: Zero-Shot Speech Editing and Text-to-Speech in the Wild - jasonppy/VoiceCraft
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1222855761248653393)** (14 messages🔥): 

- **Bot Development Enthusiasm**: A member expressed their intention to create a **Telegram or Discord bot**.
- **Inquisitive Minds for Future GPT**: A query was posed about whether **GPT-5** will outperform *Gemini 1.5 Pro*. Answers varied, jokingly predicting it to be better than GPT-4 but worse than a hypothetical GPT-6.
- **Anticipation for GPT-5**: Users discussed the possible release and access to **GPT-5**, although no specific details were provided about its availability.
- **Assistance Offered for GPT-4 Code Execution**: A user needed help with running code in GPT-4 as GPT-3.5 wasn't providing solutions, and another member volunteered to assist via direct message.
- **Channel Usage Reminder**: A reminder was issued for users to keep discussions about **ChatGPT** and related applications in a dedicated channel, emphasizing that **gpt-4-discussions** is meant for GPI AI models' discussions.
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1222843909466751046)** (153 messages🔥🔥): 

- **Prompt Precision Makes Perfect**: Members discussed the importance of phrasing prompts correctly to achieve desired outcomes. Examples such as `Respond **only** with {xyz}. Limit yourself to {asdf}.` were highlighted to ensure concise responses from models.

- **Roleplay Puzzler with GPT-4.0125**: One user shared an oddity with GPT-4.0125 model's responses regarding roleplay prompts, where providing an example of how to respond can cause the model to refuse roleplaying. The user also found workarounds, like pre-crafting a first message and removing "comply with the following directives", and discussed troubleshooting steps involving subtle leading phrases and excluding mention of what not to do.

- **Equations in MS Word Troubles**: A user sought advice on transferring LaTeX equations from ChatGPT to Microsoft Word. Despite following a YouTube tutorial, issues persisted due to differences in their MS 365 version, leading to suggestions about using MathML as an intermediary.

- **Meta-Prompting Merits Discussion**: The efficacy of metaprompting versus traditional prompting was debated, with members examining whether metaprompting consistently results in higher quality outputs, despite evidence suggesting models can unpredictably provide both correct and incorrect responses. Another user discouraged linking to external papers due to platform rules but suggested search terms for finding related studies.

- **Presentation Prompting 101**: A dialogue about creating prompts for presentations was initiated by a user seeking a fill-in-the-blanks approach. The conversation highlighted the need to provide detailed information to AI for it to effectively assist with tasks like generating presentations.
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1222843909466751046)** (153 messages🔥🔥): 

- **Prompt Precision Leads to Better Performance**: A discussion highlighted the significance of **prompt wording**, suggesting specific phrasing such as *"Respond **only** with {xyz}. Limit yourself to {asdf}"* to improve AI performance; avoiding vague terms is crucial.
- **Presenting Proper Translation Prompts**: One user sought advice on improving their prompt for translations, aiming to keep HTML tags and certain elements in English. Another member suggested focusing on what **to do** in the **prompt**, for better output, such as *"Provide a skilled translation from English to {$lang_target}"* while maintaining specific formatting and content requirements.
- **Strategic Prompting for Role-Play Responses**: Users exchanged strategies to make GPT-4 adhere to role-play instructions, addressing oddities encountered with **gpt-4-0125-preview** that refused to role-play when given an example format but complied when the example was omitted.
- **Formatting Frustration with Equations in Word**: Members tried to help a user who faced difficulties copying **LaTeX equations** into Microsoft Word. Suggestions like using **MathML** as an intermediate format and verifying **Microsoft Word version compatibilities** were given.
- **Is Metaprompting the Way to Go?**: There was a debate on whether **metaprompting** consistently achieves better results, with various users sharing skepticism and personal experiences that challenge the effectiveness of metaprompting vs. traditional direct instructions.
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1222882406109937684)** (272 messages🔥🔥): 

- **AI Predictions Trickle Into Chat**: The conversation revolved around the potential of AI to surpass human-level intelligence, with concerns about needing radically different architectures and hardware to achieve such advancements. The cost and eco-economics of scaling up current systems, and the death of Moore's law, were hotly debated.

- **Leaning Into Latin and PR Speak**: A member shared a disdain for the word "testament" as a telltale sign of AI-generated content, while others highlighted how the corporate PR-like demeanor of ChatGPT makes it easily identifiable.

- **Skepticism Straddles Startup Speculation**: A YouTube-sourced claim about a startup developing more efficient AI chips led to discussions about the difficulties in shifting hardware paradigms and the intricacies of semiconductor manufacturing.

- **Grammar and Language Peculiarities Plucked**: Chat participants chuckled over the mistaken singular forms of plural Latin-derived words, sharing personal anecdotes and clarifying correct usage in various languages, including English and Portuguese.

- **Fears of Fast AI Change Fostered**: Some expressed the opinion that society might not cope with a rapid acceleration of AI capabilities, reflecting on how society has dealt with the impacts of the digital revolution and the need for mindful progress in AI.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://summerofcode.withgoogle.com/">Google Summer of Code</a>: Google Summer of Code is a global program focused on bringing more developers into open source software development.</li><li><a href="https://www.grammar-monster.com/plurals/plural_of_axis.htm">The Plural of Axis</a>: no description found</li><li><a href="https://www.semianalysis.com/p/the-ai-brick-wall-a-practical-limit">The AI Brick Wall – A Practical Limit For Scaling Dense Transformer Models, and How GPT 4 Will Break Past It</a>: Large generative AI models unlock massive value for the world, but the picture isn’t only roses. Costs for training for these civilization-redefining models have been ballooning at an incredible pace....</li><li><a href="https://blog.eleuther.ai/rotary-embeddings/">Rotary Embeddings: A Relative Revolution</a>: Rotary Positional Embedding (RoPE) is a new type of position encoding that unifies absolute and relative approaches. We put it to the test.</li><li><a href="https://www.eleuther.ai/releases">Releases &mdash; EleutherAI</a>: no description found</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...</li><li><a href="https://github.com/davisyoshida/haiku-mup">GitHub - davisyoshida/haiku-mup: A port of muP to JAX/Haiku</a>: A port of muP to JAX/Haiku. Contribute to davisyoshida/haiku-mup development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1222852101764546560)** (46 messages🔥): 

- **LISA: A New Fine-Tuning Technique**: LISA, a new method for fine-tuning large language models, outperforms LoRA and full-parameter training in instruction following tasks and can tune 7B parameter models on a 24GB GPU. The simple yet powerful algorithm activates embedding and linear head layers while randomly sampling intermediate layers for unfreezing. [Paper](https://arxiv.org/abs/2403.17919) and [Code](https://github.com/OptimalScale/LMFlow) are available for review.
  
- **Mamba Joins Forces with Transformer**: AI21 announces Jamba, a model combining the Mamba architecture with Transformers, leading to a hybrid that has 12B active parameters out of a total of 52B. It's claimed to offer the benefits of both worlds, including a pseudo working memory and potentially more data-efficient operation. [AI21 Jamba Release](https://www.ai21.com/jamba)

- **Protecting Sensitive Training Data**: Various approaches for training with sensitive data are discussed, including SILO which decouples sensitive examples ([paper](https://arxiv.org/abs/2308.04430)) and differential privacy methods ([paper 1](https://arxiv.org/abs/1607.00133), [paper 2](https://arxiv.org/abs/2110.05679)) to prevent data extraction from models.

- **SambaNova Debuts Samba-1**: SambaNova released Samba-1, a Composition of Experts (CoE) model which integrates over 1 trillion parameters from 50+ expert models pilfered from the open source community, claiming superior performance on enterprise tasks with reduced compute needs. Skepticism abounds regarding their actual in-use performance due to lack of transparent details. [SambaNova Blog](https://sambanova.ai/blog/accurate-models-at-blazing-speed)

- **Understanding CoE at SambaNova**: Composition of Experts (CoE) at SambaNova could involve a heterogeneous mixture of expert modules, such as multiple models with different sizes combined into one system. Discussion indicates this may resemble an ensemble of fine-tuned models originating from other creators. [Benchmarking Samba-1 Blog](https://sambanova.ai/blog/benchmarking-samba-1)
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sambanova.ai/blog/accurate-models-at-blazing-speed">SambaNova Delivers Accurate Models At Blazing Speed</a>: Samba-CoE v0.2 is climbing on the AlpacaEval leaderboard, outperforming all of the latest open-source models. </li><li><a href="https://www.ai21.com/jamba">Introducing Jamba</a>: A groundbreaking SSM-Transformer Open Model</li><li><a href="https://sambanova.ai/blog/benchmarking-samba-1">Benchmarking Samba-1</a>: Benchmarking Samba-1 with the EGAI benchmark - a comprehensive collection of widely adapted benchmarks sourced from the open source community. </li><li><a href="https://arxiv.org/abs/2308.04430">SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore</a>: The legality of training language models (LMs) on copyrighted or otherwise restricted data is under intense debate. However, as we show, model performance significantly degrades if trained only on low...</li><li><a href="https://arxiv.org/abs/2110.05679">Large Language Models Can Be Strong Differentially Private Learners</a>: Differentially Private (DP) learning has seen limited success for building large deep learning models of text, and straightforward attempts at applying Differentially Private Stochastic Gradient Desce...</li><li><a href="https://www.frontiersin.org/articles/10.3389/fncir.2016.00023/full">Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex</a>: Pyramidal neurons represent the majority of excitatory neurons in the neocortex. Each pyramidal neuron receives input from thousands of excitatory synapses that are segregated onto dendritic branches....</li><li><a href="https://fixupx.com/Rui45898440/status/1772996453557997924">Tweet from Rui (@Rui45898440)</a>: Excited to share LISA, which enables - 7B tuning on a 24GB GPU - 70B tuning on 4x80GB GPUs  and obtains better performance than LoRA in ~50% less time 🚀</li><li><a href="https://fixupx.com/Rui45898440/status/1772996456422805606">Tweet from Rui (@Rui45898440)</a>: - Paper: https://arxiv.org/abs/2403.17919 - Code: https://github.com/OptimalScale/LMFlow  LISA outperforms LoRA and even full-parameter training in instruction following tasks</li><li><a href="https://fixupx.com/Rui45898440/status/1772996458893246939">Tweet from Rui (@Rui45898440)</a>: LISA algorithm in two lines:   - always activate embedding and linear head layer   - randomly sample intermediate layers to unfreeze</li><li><a href="https://arxiv.org/abs/2403.17919">LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning</a>: The machine learning community has witnessed impressive advancements since the first appearance of large language models (LLMs), yet their huge memory consumption has become a major roadblock to large...</li><li><a href="https://github.com/athms/mad-lab">GitHub - athms/mad-lab: A MAD laboratory to improve AI architecture designs 🧪</a>: A MAD laboratory to improve AI architecture designs 🧪 - athms/mad-lab</li><li><a href="https://arxiv.org/abs/2403.17844">Mechanistic Design and Scaling of Hybrid Architectures</a>: The development of deep learning architectures is a resource-demanding process, due to a vast design space, long prototyping times, and high compute costs associated with at-scale model training and e...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1222915547465912340)** (40 messages🔥): 

- **Weight Wonders or Woes?**: Discrepancies were identified when comparing weight parameters between **Transformer Lens** (`tl`) and **Hugging Face** (`hf`) versions of models like GPT-2 and Pythia 70m. The examination of these discrepancies involved comparing shapes and computing the maximum absolute differences between the models' embedding parameters.

- **A Feature, Not a Bug**: The variation in model weights was initially thought to be a bug but was confirmed to be a featured processing step by Transformer Lens. The solution advised was to use `from_pretrained_no_processing` when initializing models to avoid such pre-processing. References were made to a [GitHub issue](https://github.com/neelnanda-io/TransformerLens/issues/346) and [further comments on weight processing](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing).

- **Logit Labyrinths**: Despite same-shaped logits showing significant absolute differences between `nn_model` and `tl_nn_model`, a comparison after applying softmax indicated negligible differences—the implication being that while raw logit values differ, the relative order and softmax outputs are consistent.

- **A Plotting Pathway to Clarification**: The process of identifying and understanding the weight disparities involved plotting various matrices, including the embedding matrices and their absolute differences, and analyzing their characteristics.

- **Sparse Autoencoders Under Scrutiny**: A separate mention highlighted a research post discussing the impact of reconstruction errors in Sparse Autoencoders (SAEs), suggesting that these errors could significantly alter model predictions more than random errors of equal magnitude. The post is accessible via a [LessWrong link](https://www.lesswrong.com/posts/rZPiuFxESMxCDHe4B/sae-reconstruction-errors-are-empirically-pathological).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/wesg52/status/1773756298531918268">Tweet from Wes Gurnee (@wesg52)</a>: Short research post on a potential issue arising in Sparse Autoencoders (SAEs): the reconstruction errors change model predictions much more than a random error of the same magnitude! https://www.less...</li><li><a href="https://colab.research.google.com/drive/1juAJrTb3Z9hkVFJnbrj1OYmnGJ0MlH_G?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing">TransformerLens/further_comments.md at main · neelnanda-io/TransformerLens</a>: A library for mechanistic interpretability of GPT-style language models - neelnanda-io/TransformerLens</li><li><a href="https://github.com/neelnanda-io/TransformerLens/issues/346">[Bug Report] hook_resid_pre doesn&#39;t match hidden_states · Issue #346 · neelnanda-io/TransformerLens</a>: Describe the bug cache[f&quot;blocks.{x}.hook_resid_pre&quot;] doesn&#39;t match hidden states (or only up to a set decimal place). Hidden states is from transformer&#39;s model(tokens, output_hidden_...</li><li><a href="https://fixupx.com/jxmnop/status/1773377787153248638">Tweet from jack morris (@jxmnop)</a>: Diffusion Lens is a pretty neat new paper, you can see a text-to-image encoder&#39;s representation of a giraffe getting less and less abstract with every layer 🦒</li><li><a href="https://arxiv.org/abs/2403.05846">Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines</a>: Text-to-image diffusion models (T2I) use a latent representation of a text prompt to guide the image generation process. However, the process by which the encoder produces the text representation is u...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1222866433772617778)** (8 messages🔥): 

- **Optimizing MMLU Performance**: [Efficiency improvements for MMLU](https://discord.com) have been implemented, allowing one forward call to extract multiple logprobs simultaneously, which can be disabled by setting `logits_cache` to `False`.
- **Troubleshooting Large Model Support**: A user encountered **memory allocation issues** when loading the **DBRX base model** using **lm-eval** harness. They discovered their user error of operating on a node with only 4 GPUs instead of the 8 with 64GB VRAM they anticipated having.
- **Enhanced Context-Based Task Handling**: A [pull request was submitted](https://github.com/EleutherAI/lm-evaluation-harness/pull/1571) proposing a new approach for handling context-based tasks in **lm-evaluation-harness** which better supports tasks relying on prior request answers. The update includes methods to refine requests before model ingestion and manage an external log for crucial contextual information.
- **Feedback Requested on Pull Request**: User *hailey_schoelkopf* promises to review and provide feedback on the mentioned pull request after the CoLM deadline, apologizing for the delay.

**Link mentioned**: <a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1571).">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1222916017303457894)** (6 messages): 

- **Quick Exit**: A user simply posted that they had been **banned and kicked** without providing further context or details.
- **Simple Gratitude**: A short message expressing **thanks** was posted, with no additional information included.
- **Cohere's Efficiency in Embeddings**: A YouTube video titled "Cohere int8 & binary Embeddings" was shared, which discusses how to scale vector databases for large datasets. The video is relevant for those interested in AI, LLMs, ML, deep learning, and neural networks.
- **Styling with StyleGAN2**: A user queried about training a complex directory structure of various fashion datasets in **StyleGAN2-ADA** and whether script modification is necessary for this process.
- **Aerospace Student Eying AI**: An aerospace student expressed a desire to delve into ML/AI and contribute to open source projects, seeking advice on whether starting with **fast.ai** courses is a suitable approach given their math and coding background.

**Link mentioned**: <a href="https://www.youtube.com/watch?v=LWz2QaSRl2Y">Cohere int8 &amp; binary Embeddings</a>: Cohere int8 &amp; binary Embeddings - Scale Your Vector Database to Large Datasets#ai #llm #ml #deeplearning #neuralnetworks #largelanguagemodels #artificialinte...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1222918777596018722)** (6 messages): 

- **MicroQwen on the Horizon**: A tweet hints at an upcoming release for what is speculated to be a smaller model variant of Qwen named [microqwen](https://twitter.com/JustinLin610/status/1773285084025475355).

- **AI21 Labs Jams with Jamba**: AI21 Labs announces [Jamba](https://www.ai21.com/blog/announcing-jamba), a novel Mamba-based model with Transformer elements, boasting a 256K context window and unmatched throughput and efficiency. Jamba comes with open weights under the Apache 2.0 license for community innovation.

- **Qwen Embraces Mixture of Experts**: Introducing [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B), a transformer-based MoE model with 14.3B parameters yet only 2.7B activated at runtime. This model claims to achieve comparable performance to much larger models with significantly less training resources and faster inference speeds.

- **LISA Amplifies LoRA's Fine-Tuning Impact**: A new study identifies a layerwise skewness in weight norms when using LoRA and proposes a simple yet more efficient training strategy called Layerwise Importance Sampled AdamW (LISA), which shows improvements over LoRA and full parameter training with low memory costs. Details available on [arXiv](https://arxiv.org/abs/2403.17919).

- **BLLaMa Project Forks on GitHub**: A GitHub repository named [bllama](https://github.com/rafacelente/bllama) has surfaced, promoting a 1.58-bit LLaMa model which seems to be an efficient variant of the original LLaMa model, aiming for contributions from the community.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B">Qwen/Qwen1.5-MoE-A2.7B · Hugging Face</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://arxiv.org/abs/2403.17919">LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning</a>: The machine learning community has witnessed impressive advancements since the first appearance of large language models (LLMs), yet their huge memory consumption has become a major roadblock to large...</li><li><a href="https://github.com/rafacelente/bllama">GitHub - rafacelente/bllama: 1.58-bit LLaMa model</a>: 1.58-bit LLaMa model. Contribute to rafacelente/bllama development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1222844959346917429)** (145 messages🔥🔥): 

- **Speculation on Future NOUS Tokens**: A member inquired about the potential for NOUS to introduce a cryptographic token, sparking a light-hearted response dismissing the notion.
- **NOUS Subnet Connection in Place of Tokens**: In response to a query on NOUS tokens, it was clarified that while there's no crypto token, NOUS has a subnet on bittensor/Tao, which has its own coin.
- **Sneak-Peek at Qwen1.5-MoE-A2.7B**: A member teased the imminent release of a compact MoE model dubbed Qwen1.5-MoE-A2.7B, prompting anticipation and discussions about its upcycling of experts from an existing 1.8B parameter model.
- **Jamba's Entry Brings SSM-Transformer Hybrid to Forefront**: AI21 announced **Jamba**, a hybrid SSM-Transformer LLM boasting a 256K context window and potential for high throughput and efficiency, stirring excitement among community members about its capabilities.
- **Discussions of Support for New Model Architectures in Open Source Libraries**: Conversations revolved around the anticipation for popular open source libraries like *vllm* and *transformers* to embrace new architectures like RWKV, SSMs, and the freshly announced Jamba, highlighting the industry's quick evolution and the desire for timely support of innovative models.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Experiments with Bitnet 1.5 (ngmi)</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction Since the surge in interest sparked by Mixtral, research on mixture-of-expert (MoE) models has gained significant momentum. Both researchers an...</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://x.com/justinlin610/status/1773285084025475355?s=46&t=H75DmkDKk9Sgmp8kjT8f_A">Tweet from Junyang Lin (@JustinLin610)</a>: Hours later, you will find our little gift. Spoiler alert: a small MoE model that u can run easily🦦</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1970zhf/merging_mistral_with_whisper_to_make_a_multimodal/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://github.com/databricks/dbrx/issues/5">Loading over multiple gpus in 8bit and 4bit with transformers loader · Issue #5 · databricks/dbrx</a>: I can load the instruct model using the transformers loader and 8bit bits and bytes, I can get it to load evenly among multiple gpus. However, I cannot seem to load the model with 4bit precion over...</li><li><a href="https://fxtwitter.com/elonmusk/status/1773655245769330757">Tweet from Elon Musk (@elonmusk)</a>: Should be available on 𝕏 next week.   Grok 2 should exceed current AI on all metrics. In training now.  ↘️ Quoting xAI (@xai)   https://x.ai/blog/grok-1.5
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1222860177024221204)** (26 messages🔥): 

- **Fine-Tuning Frustrations**: A member shared a full-parameter fine-tuning case of **Deepseek-coder-33B** where the training loss decreases at the start of each epoch while the evaluation loss increases. It was speculated to be a case of textbook overfitting, with suggestions to shuffle the dataset to address possible data inconsistencies.

- **Hardware Hunt for Local LLMs**: Queries were made about where to find information on hardware requirements for running large language models locally. An alternate preference for **ERP** systems was mentioned, but no specific resources were provided.

- **Pretraining Prep for Prolific Prose**: A challenge was raised on how to prepare a **1,500 book dataset** for pretraining when some books have up to 200k tokens. It was recommended to use clustering and hierarchical approaches to handle repetition and maintain thematic consistency in the data.

- **Retrieving the Right RAG Strategy**: Discussion on whether to use one big **Retrieval Augmented Generation (RAG)** model or several domain-focused RAGs for different documents took place. Suggestions included using structured approaches with domains and metadata filtering, with the ultimate decision noted to be use-case specific.

- **Hermes Model Missing Tokens Mystery**: A concern was raised about the **Hermes-2-Pro-Mistral-7B** model configuration on **HuggingFace** where a mismatch in the defined vocabulary size and extra defined tokens might cause errors during generation. The issue was addressed as an artifact from model padding processes, with a suggestion to create extra unused tokens to resolve it.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/config.json#L25">config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/added_tokens.json">added_tokens.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B/blob/main/tokenizer_config.json#L30">tokenizer_config.json · NousResearch/Hermes-2-Pro-Mistral-7B at main</a>: no description found
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 messages): 

night_w0lf: Did it work?
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1222799124689063936)** (57 messages🔥🔥): 

- **Agentic Data Sets for Hermes 2 Pro**: Hermes 2 Pro is trained on agentic datasets with all popular frameworks, and work is in progress to develop multi-turn agentic loops.

- **CoT Revisions and the Value of Metadata**: The Chain of Thought (CoT) process is revised to enhance answer retrieval, with structured formats like XML found to be effective in enhancing Claude's performance as evidenced by [Anthropic's use of XML tags](https://docs.anthropic.com/claude/docs/use-xml-tags).

- **Structured Inputs for RAG**: It's proposed to utilize structured formats such as XML for input delineation in modeling and to create a dedicated category for pydantic related implementations, indicating a trend towards more structured and metadata-rich inputs for AI models.

- **Incorporating Temporal Awareness in Models**: There's a curiosity about increasing a model's temporal awareness, highlighting the importance of attributing proper context and metadata to data being processed by the model.

- **RAG Evaluation Framework Introduction**: The chatbot referred to a GitHub repository for an evaluation framework called [ragas](https://github.com/explodinggradients/ragas), indicating ongoing efforts to improve Retrieval Augmented Generation (RAG) pipelines.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.useinstructor.com/examples/exact_citations/">Citing Sources (RAG) - Instructor</a>: no description found</li><li><a href="https://docs.anthropic.com/claude/docs/use-xml-tags">Use XML tags</a>: no description found</li><li><a href="https://github.com/explodinggradients/ragas/tree/main">GitHub - explodinggradients/ragas: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines</a>: Evaluation framework for your Retrieval Augmented Generation (RAG) pipelines - explodinggradients/ragas
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1222805877698990111)** (88 messages🔥🔥): 

- **Prompt Assistance and Sharing Resources**: Members are sharing tips, tutorials, and code snippets for various projects like generating LaTeX papers for arXiv using Worldsim. The [gist for LaTeX papers](https://gist.github.com/irl-dan/61e2f45eb1c9a879b3) was specifically mentioned as helpful. 

- **Hyperstition Concept Discussions**: The term *hyperstition* has been analyzed in-depth, with explanations about how it affects the cognitive domains of LLMs like Claude. The term is associated with expanding the generative range of AI models and is linked to **Mischievous Instability (MI)**.

- **Philosophy Open Mic Interest**: There's an expressed interest in hosting philosophy discussions or open mic nights to explore language as a complex adaptive system amongst other topics. This could potentially be coupled with new steamcasting technology for wider distribution.

- **Exploring Historical 'What-ifs'**: Members discussed using Worldsim to experiment with alternate history scenarios, like Julius Caesar surviving or JFK avoiding assassination, and expressed curiosity about the potential outcomes on world history.

- **Clarifying the Usage of Large Language Models**: A member explained the difference between heuristic engines and inference engines, highlighting how terms are defined based on relationships and associations in the latter, leading to what's known as "Cognitive Domains."
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/openrouterai/status/1773738942350712907?s=46">Tweet from OpenRouter (@OpenRouterAI)</a>: Ever wondered which apps are using an LLM? Now you can find out yourself with the new Apps tab.  @NousResearch has the top Claude 3 Opus apps this week 👀</li><li><a href="https://drive.google.com/file/d/18iaWyZ0MNp5BTHkTTAWj4bmpnc9F-A5Y/view?usp=sharing">2024-03-25 21-12-56-Nous Hermes world sim night.mkv</a>: no description found</li><li><a href="https://tenor.com/view/chess-checkmate-its-it-over-gif-17459417336418427755">Chess Checkmate GIF - Chess Checkmate Its - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://www.reddit.com/r/teslore/comments/ppk9y2/comment/hd53ngs/?utm_source=share&utm_medium=web2x&context=3">Reddit - Dive into anything</a>: no description found</li><li><a href="https://tenor.com/view/dark-knight-joker-its-not-about-the-money-its-about-sending-a-message-gif-15254722">Dark Knight Joker GIF - Dark Knight Joker Its Not About The Money - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://old.reddit.com/r/lexfridman/comments/1boyj88/an_instance_of_a_claude_3_opus_model_claims/">An instance of a Claude 3 Opus model claims consciousness </a>: I’ve been interacting with an instance (I suppose several instances) of the newest Opus model in a self reflective and philosophical dialogue....</li><li><a href="https://gist.github.com/irl-dan/2be642f22c28bdacd92d1a2ac0172d8e">self-system-prompt</a>: self-system-prompt. GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://youtu.be/65pOHKuvNGU?si=vI5RJdfOFL4V9oxc">Introducing WebSim: Hallucinate an Alternate Internet with Claude 3</a>: Go to websim.ai to imagine alternate internets.WebSim was inspired by world_sim, an &quot;amorphous application&quot; built by Nous Research that simulates a world wit...</li><li><a href="https://gist.github.com/irl-dan/595f74f17fc5b269c96e9f9f9079595b">strange-loop+claude3-self-modeling</a>: GitHub Gist: instantly share code, notes, and snippets.</li><li><a href="https://gist.github.com/irl-dan/61e2f45eb1c9a879b39d480694a4c4a3">claude-world-modelling</a>: claude-world-modelling. GitHub Gist: instantly share code, notes, and snippets.
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1222832511827312756)** (127 messages🔥🔥): 

- **Understanding Mojo's Performance Boost for RL**: Members discussed the _potential_ benefits of using **Mojo** for speeding up reinforcement learning environments. A member pointed out, despite Mojo's speed, that it will take time and effort to adapt existing Python-based tools and environments to work effectively with Mojo.

- **Dynamic Shape Capabilities in MAX Engine**: A [blog post](https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities) unveiled improvements in MAX Engine's 24.2 release, focusing on _dynamic shape_ support in machine learning models and comparing dynamic shapes' latency to static ones.

- **Mojo Running Locally**: Members discussed running Mojo locally on different platforms. Mojo has native support for Ubuntu and M-series Macs, can run through WSL on Windows, and Docker on Intel Mac, with **native support for other platforms in the works**.

- **Mojo and Interoperability with Python**: Clarification was provided on how Mojo works with Python, highlighting that Python interop runs through the CPython interpreter and uses reference counting for memory management. Code not interfacing with Python avoids garbage collection and instead uses compiler-driven deallocation similar to C++ RAII or Rust.

- **Package Management and Cross-Compiling in Mojo**: There is currently **no official package manager** for Mojo, although there's ongoing discussion about the project manifest format, which will precede a package manager. Cross-compiling capabilities for Windows aren't confirmed, with community members looking forward to potential Windows support as a sign of Mojo's maturity for further engagement.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com">Modular Docs</a>: no description found</li><li><a href="https://docs.modular.com/mojo/manual/get-started/#system-requirements">Get started with Mojo🔥 | Modular Docs</a>: Get the Mojo SDK or try coding in the Mojo Playground.</li><li><a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</li><li><a href="https://pastebin.com/P3iwmJq9">curl -s https://get.modular.com | sh -modular authExecuting the  setup scrip - Pastebin.com</a>: Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md#signing-your-work">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/mojopaa/mojopi">GitHub - mojopaa/mojopi: Mojo Package Index, inspired from PyPI.</a>: Mojo Package Index, inspired from PyPI. Contribute to mojopaa/mojopi development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1222981979046154322)** (6 messages): 

- **Teasing New Frontiers**: Modular teased an upcoming announcement in a tweet, hinting at something new on the horizon. The tweet can be explored [here](https://twitter.com/Modular/status/1773418812915978284).
- **Dropping Hints**: Another Modular tweet dropped a hint with a single emoji, sparking curiosity among followers. Check the tweet [here](https://twitter.com/Modular/status/1773418820184764572).
- **Countdown to Reveal**: A tweet by Modular included a clock emoji, suggesting a countdown to an upcoming reveal or event. The tweet is accessible [here](https://twitter.com/Modular/status/1773418823707955529).
- **Sneak Peek Shared**: Modular released a sneak peek of what's to come in a recent tweet, which can be seen [here](https://twitter.com/Modular/status/1773440424205783455).
- **Event Announcement**: Modular announced a specific event, potentially related to the previous teasers, in their latest tweet. Full details can be found [here](https://twitter.com/Modular/status/1773762747098124491).
- **Event Reminder**: A follow-up tweet from Modular served as a reminder about the recently announced event. Visit the tweet for more information [here](https://twitter.com/Modular/status/1773767659278250242).
  

---


**Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1223313935202123957)** (1 messages): 

- **Modular Community Livestream Release**: Modular announced a new video on YouTube titled **"Modular Community Livestream - New in MAX 24.2"**. The video covers the latest updates in MAX 24.2, including the **open sourcing of Mojo standard library** and **MAX Engine support**. [Watch the video here](https://www.youtube.com/watch?v=PL71FV2KKHE).

**Link mentioned**: <a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular Community Livestream - New in MAX 24.2</a>: MAX 24.2 is now available! Join us on our upcoming livestream as we discuss everything new in MAX - open sourcing Mojo standard library, MAX Engine support f...

  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1222976873974468850)** (3 messages): 

- **Mojo🔥 Standard Library Now Open Source**: Modular announced the general availability of **MAX 24.2**, featuring the open sourcing of the **Mojo standard library**. Developers are invited to contribute to its development, with the library now available on [GitHub](https://github.com/modularml/mojo/tree/nightly/stdlib), and **nightly builds** released for the latest language features.

- **Mojo🔥Open Source Contributions Welcomed**: Modular has released core modules of the **Mojo standard library under the Apache 2 license**, as part of their belief in open-source development. The ongoing improvements since May 2023 can be tracked through the [changelog](https://docs.modular.com/mojo/changelog), with the company encouraging collaboration from developers worldwide.

- **Leveraging MAX Engine 24.2 for Dynamic Shapes**: The **MAX Engine 24.2** release includes support for dynamic shapes in machine learning, crucial for handling real-world data variability. The impact of dynamic shapes versus static shapes is demonstrated through a latency comparison using the [BERT model](https://huggingface.co/docs/transformers/model_doc/bert) on the [GLUE dataset](https://gluebenchmark.com/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</li><li><a href="https://www.modular.com/blog/max-24-2-is-here-whats-new">Modular: MAX 24.2 is Here! What’s New?</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: MAX 24.2 is Here! What’s New?
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1222977000646508635)** (1 messages): 

- **Mojo Standard Library Goes Open Source**: Modular has officially open-sourced the **core modules of the Mojo standard library**, releasing them under the Apache 2 license. This move is aligned with their belief that open-source collaboration will lead to a better product, further outlined in their [blog post](https://modul.ar/open-source-blog).

- **Nightly Builds for the Standard Library**: Alongside the open-sourcing, Modular has introduced nightly builds for developers to stay updated with the latest improvements. Open-sourcing is seen as a step towards closer collaboration with the global Mojo developer community.

- **MAX Platform v24.2 Update**: The latest **MAX platform update**, version 24.2, includes support for TorchScript models with dynamic input shapes and other enhancements. Detailed changes are listed in the [MAX changelog](https://modul.ar/max-changelog).

- **Latest Mojo Language Tools and Features**: The **Mojo language update** to v24.2 has the standard library going open source and brings advancements such as implicit conformance of structs to traits. The running list of significant changes can be found in the [Mojo changelog](https://modul.ar/mojo-changelog).

- **Deep Dive into MAX's Dynamic Shapes Support**: A dedicated [blog post](https://modul.ar/max-dynamic-shapes) explains the new dynamic shapes support in MAX Engine’s **24.2** release, its use cases, and the performance impact, particularly showcasing improvements on the BERT model on the GLUE dataset.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://modul.ar/open-source-blog">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://modul.ar/max-changelog">MAX changelog | Modular Docs</a>: Release notes for each version of the MAX platform.</li><li><a href="https://modul.ar/mojo-changelog">Mojo🔥 changelog | Modular Docs</a>: A history of significant Mojo changes.</li><li><a href="https://modul.ar/max-dynamic-shapes">Modular: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: Leveraging MAX Engine&#x27;s Dynamic Shape Capabilities
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1222846623948865566)** (91 messages🔥🔥): 

- **Generalized Complex Types Introduced**: Moplex, offering fully featured generalized complex types, has been released on GitHub by helehex. Detailed information and code repository is available at [helehex/moplex on GitHub](https://github.com/helehex/moplex).
- **NVIDIA GPU Support Gears Up for Summer Release**: Modular is prioritizing NVIDIA GPU support with a target release around summer, which includes split compilation similar to CUDA with Python-like syntax that transitions from CPU to GPU. A [MAX session about Mojo's GPU support](https://www.youtube.com/watch?v=QD-svwZistc) shares insights on the development.
- **Modular AI Engine and Mojo Changelog for 24.2.0**: An early peek into the Mojo 24.2 changelog shows significant updates including open-sourcing of the Mojo standard library. Changelog details can be found [here](https://docs.modular.com/mojo/changelog#v242-2024-03-28).
- **Featuring Simplified References Across Collections**: Structs and nominal types can now implicitly conform to traits in the fresh updates, hinting at more intuitive conformance patterns. A member spotlighted the feature and its potential, directing others to Modular's contributing guidelines at [modularml/mojo on GitHub](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md).
- **Matrix Multiplication Example Revision Needed**: A user identified a bug in the matrix multiplication example, proposing a correction for cases where the columns of C are not a multiple of `nelts` to avoid a crash. The user suggests altering the range loop to rectify the problem.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>: no description found</li><li><a href="https://docs.modular.com/mojo/stdlib/memory/memory#memcpy`">memory | Modular Docs</a>: Defines functions for memory manipulations.</li><li><a href="https://www.geeksforgeeks.org/system-call-in-c/amp/">system() in C/C++ - GeeksforGeeks</a>: no description found</li><li><a href="https://realpython.com/python-assert-statement/">Python&#x27;s assert: Debug and Test Your Code Like a Pro – Real Python</a>: In this tutorial, you&#x27;ll learn how to use Python&#x27;s assert statement to document, debug, and test code in development. You&#x27;ll learn how assertions might be disabled in production code, s...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=QD-svwZistc">ModCon 2023 Breakout Session: MAX Heterogenous Compute: CPU + GPU</a>: In this session, Modular engineers Abdul Dakkak and Ian Tramble discuss how Mojo and the Modular AI Engine were designed to support systems with heterogeneou...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md#import-statements">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/development.md">mojo/stdlib/docs/development.md at main · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/issues/1958),">Issues · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/helehex/moplex">GitHub - helehex/moplex: Generalized complex numbers for Mojo🔥</a>: Generalized complex numbers for Mojo🔥. Contribute to helehex/moplex development by creating an account on GitHub.</li><li><a href="https://github.com/carlca/ca_mojo.git">GitHub - carlca/ca_mojo</a>: Contribute to carlca/ca_mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/commits/main/stdlib/src/builtin/builtin_list.mojo">History for stdlib/src/builtin/builtin_list.mojo - modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://docs.modular.com/mojo/changelog#v242-2024-03-28">Mojo🔥 changelog | Modular Docs</a>: A history of significant Mojo changes.</li><li><a href="https://github.com/networkx/networkx/blob/f0b5a6d884ac7e303f2e2092d3a0a48723815239/networkx/classes/graph.py#L66)">networkx/networkx/classes/graph.py at f0b5a6d884ac7e303f2e2092d3a0a48723815239 · networkx/networkx</a>: Network Analysis in Python. Contribute to networkx/networkx development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/max/blob/65c88db34f50beea2f403205ac9f91d3bf28ce8c/examples/graph-api/llama2/tokenizer/ball.%F0%9F%94%A5#L27)">max/examples/graph-api/llama2/tokenizer/ball.🔥 at 65c88db34f50beea2f403205ac9f91d3bf28ce8c · modularml/max</a>: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform - modularml/max</li><li><a href="https://github.com/modularml/mojo/blob/6d2a7b552769358ec68a94b8fd1f6c2126d59ad9/stdlib/src/collections/list.mojo#L42)">mojo/stdlib/src/collections/list.mojo at 6d2a7b552769358ec68a94b8fd1f6c2126d59ad9 · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1222892211864342578)** (13 messages🔥): 

- **Mojo Standard Library on Fast Track**: Updates have been made to several Mojo packages such as `mojo-prefix-sum`, `mojo-flx`, `mojo-fast-base64`, and `mojo-csv` to version 24.2, while `mojo-hash` and `compact-dict` are partially updated with some ongoing issues.
- **Call for Understanding MLIR Dialects**: A community member expressed a desire to learn about Mojo's underlying MLIR dialects, suggesting that such knowledge would enable more direct contributions to Mojo's standard library.
- **Patience for MLIR Syntax Documentation**: In response to queries about MLIR details, it was pointed out that the syntax for using MLIR in Mojo is set to undergo significant changes and is not yet documented, except for a notebook provided in the link: [Mojo with MLIR notebook](https://docs.modular.com/mojo/notebooks/BoolMLIR).
- **Mojo Language Development in Progress**: The Modular development team acknowledges the curiosity in the community regarding internal dialects and assures that more information will be made available over time, emphasizing that the current focus is on stabilizing MLIR syntax.
- **Reference Feature is Evolving**: The `Reference` feature in the Mojo language was highlighted as being in an early and rapidly changing state, with expectations set for it to continue evolving and improving.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/notebooks/BoolMLIR">Low-level IR in Mojo | Modular Docs</a>: Learn how to use low-level primitives to define your own boolean type in Mojo.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/os/atomic.mojo#L80">mojo/stdlib/src/os/atomic.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/dict.mojo#L65">mojo/stdlib/src/collections/dict.mojo at nightly · modularml/mojo</a>: The Mojo Programming Language. Contribute to modularml/mojo development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-blogs-vids](https://discord.com/channels/1087530497313357884/1151418796993683477/1223153258298871880)** (1 messages): 

- **Mojo Standard Library Goes Open Source**: The [Mojo standard library (stdlib)](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) has been open-sourced, allowing the community to contribute to its codebase. A new [blog post](https://vmois.dev/mojo-local-stdlib-build/) provides a guide on how to build the stdlib locally on macOS (and possibly Linux), detailing steps to make and test changes in the repository.

**Link mentioned**: <a href="https://vmois.dev/mojo-local-stdlib-build/">Use locally built standard library in Mojo</a>: Mojo standard library (stdlib) was open-sourced yesterday. It is exciting that the community can now contribute directly to the codebase. After spending some time with the stdlib repository, I want to...

  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1222902298393575494)** (4 messages): 

- **Clarification on TensorSpec Usage**: A member questioned the use of TensorSpec in the "Run Inference with Mojo" example, suggesting that documentation might need an update if it's not part of Mojo. The response clarified that **TensorSpec is indeed part of the Mojo API**, Python API, and C API, with minor differences set to be smoothed out soon.
- **PyTorch Example Requested**: A member sought an example utilizing `add_input_spec` and TensorSpec objects for PyTorch in Mojo. It was acknowledged that such an example is currently missing and will be provided in the future.
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1223011855899955290)** (1 messages): 

- **HyperGraph Paper by SauravMaheshkar**: A new paper on **HyperGraph Representation Learning** has been published and discussed, providing insights into advanced data representations. The paper is accessible via [Hugging Face](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05).

- **Professional Tips for Coders**: Hugging Face introduces the **HF professional coder assistant** using Hugging Chat, inspired by an open-source GPT and open for exploration at [Hugging Face's Chat](https://hf.co/chat/assistant/6603733512f69f8b440448b4).

- **Visualizing LLM Leadership**: **Open LLM Leaderboard Viz** has received an update featuring new functionalities such as data type filters and model details, showcased at [Hugging Face Spaces](https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz).

- **OpenCerebrum Datasets Unveiled**: Introducing **OpenCerebrum**, an open-source initiative aimed at replicating the proprietary Cerebrum dataset, now available on Hugging Face with approximately 21,000 examples. Detailed information can be found at the [OpenCerebrum dataset page](https://huggingface.co/datasets/Locutusque/OpenCerebrum-dpo).

- **Launch of Fluently XL v3**: Explore the newest version of **Fluently XL**, a space dedicated to unleashing creativity and language prowess, currently live on Hugging Face at [Fluently Playground](https://huggingface.co/spaces/fluently/Fluently-Playground).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - a SauravMaheshkar Collection</a>: no description found</li><li><a href="https://hf.co/chat/assistant/6603733512f69f8b440448b4">Koder (Professional Coder) - HuggingChat</a>: Use the Koder (Professional Coder) assistant inside of HuggingChat</li><li><a href="https://huggingface.co/spaces/dimbyTa/open-llm-leaderboard-viz">Open Llm Leaderboard Viz - a Hugging Face Space by dimbyTa</a>: no description found</li><li><a href="https://huggingface.co/datasets/Locutusque/OpenCerebrum-dpo">Locutusque/OpenCerebrum-dpo · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/karpathy/minbpe/issues/60">Implementation of LlamaTokenizer (without sentencepiece) · Issue #60 · karpathy/minbpe</a>: @karpathy Thanks for the great lecture and implementation! As always, it was a pleasure. I have tried to implement LlamaTokenizer (without using sentencepiece backend) staying as close to minbpe im...</li><li><a href="https://github.com/not-lain/loadimg">GitHub - not-lain/loadimg: a python package for loading images</a>: a python package for loading images. Contribute to not-lain/loadimg development by creating an account on GitHub.</li><li><a href="https://huggingface.co/spaces/Tonic/Command-R">Command-R - a Hugging Face Space by Tonic</a>: no description found</li><li><a href="https://huggingface.co/spaces/fluently/Fluently-Playground">Fluently Playground v0.25 - a Hugging Face Space by fluently</a>: no description found</li><li><a href="https://huggingface.co/blog/Andyrasika/mongodb-llamaindex-rag">Elevate Responses: RAG with LlamaIndex &amp; MongoDB</a>: no description found</li><li><a href="https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance">hyoungwoncho/sd_perturbed_attention_guidance · Hugging Face</a>: no description found</li><li><a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance/">Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance</a>: no description found</li><li><a href="https://huggingface.co/blog/monsoon-nlp/proteins-matryoshka-embeddings">Protein similarity and Matryoshka embeddings</a>: no description found</li><li><a href="https://huggingface.co/blog/mayank-mishra/aurora">Aurora-M: The First Open Source Biden-Harris Executive Order Red teamed Multilingual Language Model</a>: no description found</li><li><a href="https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz">LLM Jargons Explained</a>: Welcome to the &quot;LLM Jargons Explained&quot; series, where I demystify the complex world of language models and decoding techniques. Whether you&#39;re a language mode...</li><li><a href="https://huggingface.co/blog/mlabonne/frankenmoe">Create Mixtures of Experts with MergeKit</a>: no description found</li><li><a href="https://huggingface.co/blog/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes">Samantha Mistral Instruct 7b - Comprehensive Bulleted Notes</a>: no description found</li><li><a href="https://huggingface.co/blog/kgourgou/a-first-look-at-automerger-data">A brief analysis of automerger data, feat. SLERP and DARE-TIES LLM merging</a>: no description found</li><li><a href="https://huggingface.co/posts/BramVanroy/231419617229308">@BramVanroy on Hugging Face: &quot; 🎈 LLM Benchmarks Update!

**tl;dr: do not depend on benchmark leaderboards…&quot;</a>: no description found</li><li><a href="https://huggingface.co/posts/xiaotianhan/875756613096016">@xiaotianhan on Hugging Face: &quot;🎉 🎉 🎉 Happy to share our recent work. We noticed that image resolution…&quot;</a>: no description found</li><li><a href="https://huggingface.co/posts/banghua/252914089485275">@banghua on Hugging Face: &quot;Have we really squeezed out the capacity of a compact chat model? Thrilled to…&quot;</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1222877312358547517)** (73 messages🔥🔥): 

- **Diffusion Repositories Explored**: A user inquired about the `labmlai diffusion` repository. No further details or links were provided about the inquiry or responses.

- **AutoTrain Model Demonstrated**: A participant shared a public model they trained using **AutoTrain**, providing a direct [link to the model](https://huggingface.co/abhishek/autotrain-c71ux-tngfu) and accompanying code snippet for its use.

- **Guidance for Data Science Students on Big Models**: A member seeking to learn about big models in **NLP** and **CV** received advice to get involved in open-source software (*OSS*) projects to gain practical experience.

- **Requirements for Running LLMs on Colab**: A user asked how to run **grok-1** on Colab and learnt that **Colab Pro** is likely necessary due to the high resources required for such large models.

- **Hugging Face Onboarding**: A newcomer to Hugging Face sought a zero-to-hero guide. They were referred to a guide titled "A Total Noob’s Introduction to Hugging Face Transformers" found on the [Hugging Face blog](https://huggingface.co/blog/noob_intro_transformers).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/abhishek/autotrain-c71ux-tngfu">abhishek/autotrain-c71ux-tngfu · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/blog/noob_intro_transformers">Total noob’s intro to Hugging Face Transformers</a>: no description found</li><li><a href="https://github.com/developersdigest/llm-answer-engine">GitHub - developersdigest/llm-answer-engine: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper</a>: Build a Perplexity-Inspired Answer Engine Using Next.js, Groq, Mixtral, Langchain, OpenAI, Brave &amp; Serper - developersdigest/llm-answer-engine</li><li><a href="https://github.com/huggingface/autotrain-advanced">GitHub - huggingface/autotrain-advanced: 🤗 AutoTrain Advanced</a>: 🤗 AutoTrain Advanced. Contribute to huggingface/autotrain-advanced development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1223061225294794794)** (4 messages): 

- **Quick on the Draw**: A member expressed enthusiasm about the performance of a tool after personally testing it, highlighting **quantization** as an intriguing aspect in efficient knowledge representation.
- **Greeting Newcomers**: Another member is looking forward to engaging with the large community here.
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1222897735829819524)** (26 messages🔥): 

- **Introducing PAG for Diffusion Models**: A new **guidance technique** called **Perturbed-Attention Guidance (PAG)** has been introduced for diffusion models, enhancing sample quality without requiring external conditions or additional training. The project is detailed on its [project page](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) and a demo is available on [Hugging Face](https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance).

- **HyperGraph Datasets Now on the Hub**: The Hub now contains preprocessed hypergraph datasets, previously used in a paper for HyperGraph Representation Learning, and they can be found in this [collection](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05). An open PR to PyTorch Geometric will allow for direct use within the PyG ecosystem.

- **Vision Transformer Model Fine-tuned for Medical Imaging**: A fine-tuned **Vision Transformer model** targeting breast cancer image classification has been uploaded to Hugging Face. Model details and a demo are available on the uploader's [Hugging Face space](https://huggingface.co/spaces/emre570/google-vit-large-finetuned).

- **CFG vs. PAG Testing for Prompts**: Users are experimenting with different settings of CFG (Classifier-Free Guidance) and PAG to optimize prompt-following while maintaining image quality. It’s suggested to start with CFG=4.5 and PAG=5.0 and adjust accordingly, and the demo will be updated to correct the output as per the feedback.

- **Seeking Suggestions for PII Detection**: A user requests recommendations for models or approaches for a PII detection project, mentioning that they have already utilized Text Mining models and BERT.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/emre570/google-vit-large-finetuned">emre570/google-vit-large-finetuned · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/emre570/google-vit-large-finetuned">Emre570 Test Model - a Hugging Face Space by emre570</a>: no description found</li><li><a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance/">Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance</a>: no description found</li><li><a href="https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance">hyoungwoncho/sd_perturbed_attention_guidance · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph Datasets - a SauravMaheshkar Collection</a>: no description found
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1223275212687278230)** (3 messages): 

- **Next Meet-Up Date Confirmed**: A member inquired about the date of the next meeting within the **reading-group**. Another member provided the details, sharing a [Discord invite link](https://discord.gg/hqrZjkjJaq?event=1222215283343622276) to the event.

**Link mentioned**: <a href="https://discord.gg/hqrZjkjJaq?event=1222215283343622276">Discord - A New Way to Chat with Friends &amp; Communities</a>: Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1222819278810779658)** (16 messages🔥): 

- **Roboflow Gaining Traction**: A participant highlighted the growing popularity of [Roboflow](https://roboflow.com/), a tool for computer vision tasks, and mentioned other domain-specific tools such as 3d Slicer and ITK-SNAP for medical images. For those looking to fine-tune models, they cautioned that apps like SAM might not be ideal.

- **SAM Fine-Tuning Troubles**: One user experienced an issue while fine-tuning the SAM model, encountering a `TypeError` related to a 'multimask_output' argument. They shared a [Towards AI blog post](https://pub.towardsai.net/fine-tune-meta-sam-19f7cd4331dd) but didn't specify if the error was resolved.

- **YOLO Conversion Script Shared**: In a discussion about converting YOLO models to safetensors, a member shared a script used in the conversion process, pointing to a [Hugging Face discussion thread](https://huggingface.co/lmz/candle-yolo-v8/discussions/1) for further details. The script included functions for renaming layers to match expected format by safetensors.

- **Inquiry into Zero-Shot Image Classifier Fine-Tuning**: A self-proclaimed beginner sought advice on fine-tuning a zero-shot image classifier with a custom dataset. While they were unsure of their system's capabilities, they confirmed using an NVIDIA GeForce GTX 1650 GPU.

- **Pix2Pix Prompt Testing for a Demo**: A user was in search of a method to test instruct pix2pix edit prompts for a demo, expressing that while they had success using the prompts in the model's space, they lacked a `gradio_client` API for further testing. They were open to alternatives as long as they could generate edited images for their demo pipeline.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/lmz/candle-yolo-v8/discussions/1">lmz/candle-yolo-v8 · Can you provide scripts that convert to safetensors format?</a>: no description found</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?tab=readme-ov-file">GitHub - bhpfelix/segment-anything-finetuner: Simple Finetuning Starter Code for Segment Anything</a>: Simple Finetuning Starter Code for Segment Anything - bhpfelix/segment-anything-finetuner</li><li><a href="https://github.com/bhpfelix/segment-anything-finetuner?t">GitHub - bhpfelix/segment-anything-finetuner: Simple Finetuning Starter Code for Segment Anything</a>: Simple Finetuning Starter Code for Segment Anything - bhpfelix/segment-anything-finetuner
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1222883104981651517)** (21 messages🔥): 

- **Battling the Infinite Generation**: A member shared an issue with their custom LLM (based on decilm7b) persistently generating text without stopping. Fellow members suggested looking into *Supervised Fine Tuning (SFT)* and tuning generation behavior with [`repetition penalty`](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin) or employing a `StopCriterion` in the configuration.

- **Making Sense of Summarization Errors**: A problem was reported where the BART CNN summarization model began to output an error message during use. The issue highlights the potential pitfalls of model dependencies or API changes.

- **RAG Enthusiasts Assemble**: One user sought advice on setting up a Retrieval Augmented Generation (RAG) system, considering to use `faiss` for the vectorDB and `llama.cpp` for model effectiveness with limited GPU resources. Suggestions directed the user towards resources including a video explaining RAG system evaluation and a simple installation process for ollama.

- **Seeking Research Collaborators**: An interaction was spotted where one member reached out to another for a potential research project collaboration, illustrating the friendly and cooperative nature of the NLP community channel.

- **Discovering Assistant Models via Tokenizer**: A newcomer inquired about identifying assistant models compatible with a main model's tokenizer (specifically for the `model.generate` function), proving that the quest for model compatibility prevails as a common concern in NLP applications.

**Link mentioned**: <a href="https://youtu.be/r0_O0IogbKo?si=lNon-ytkDjw9x1-3">Evaluate Retrieval Augmented Generation (RAG) Systems</a>: Retrieval Augmented Generation is a powerful framework which improves the quality of responses that you get from LLMs. But if you want to create RAG systems ...

  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1222877278602526844)** (4 messages): 

- **Exploring Labmlai Diffusion**: Inquiry about experiences with [labmlai's diffusion repository](https://github.com/labmlai/annotated_deep_learning_paper_implementations) was raised in the discussion.
- **Marigold Project Expands**: The **Marigold depth estimation pipeline** has been contributed to the Hugging Face Diffusers, and the team is now working on integrating the [LCM function](https://huggingface.co/spaces/prs-eth/marigold-lcm) with plans for more modalities.
- **Real-Time img2img Evolution**: An optimisation allows img2img to run at **30fps at 800x800 resolution** using sdxl-turbo, enabling captivating **real-time** visual transitions.
- **Possible img2img Bug Detected**: An "off by 1 bug" in the img2img model is believed to cause images to drift to the right; a workaround involving manipulating image edges every few frames has been experimented with. Further investigation into **conv2d padding** and possible deterministic patterns in the jitter is planned.
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1222793504493801554)** (108 messages🔥🔥): 

- **Voice Actors' Data for AI Training**: It was discussed that professional voice actors might not be willing to contribute their voice data to AI projects like 11labs, due to anti-AI sentiments, but non-professional voices might suffice for training on natural speech, especially when emotional range is not a priority.
- **Benchmarks Called Into Question**: Multiple messages indicated that existing AI benchmarks may not be reliable indicators of model performance, with claims that benchmarking is currently very flawed. A particular [chatbot arena benchmark on HuggingFace](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) was suggested as a more sensible option due to human evaluation.
- **AI21's New Model Jamba Breaks Ground**: [AI21 Labs announced Jamba](https://www.ai21.com/blog/announcing-jamba), a new state-of-the-art, hybrid SSM-Transformer LLM which combines the strengths of both architectures, allowing it to match or outpace cutting-edge models in benchmark performance.
- **Questions on Mamba vs. Flash Attention Advantages**: A discussion was held about the advantages of Mamba, a memory-based architecture, over models using flash attention, with skepticism about Mamba's comparative real-world performance and costs.
- **Debate Over Audio Quality in AI-Generated Music**: Some participants critiqued the audio quality of AI music generators like Suno, pointing out issues like noise artifacts and suggesting the need for better neural codecs. There were mixed opinions on whether the latest versions of these tools showed any improvement over earlier iterations.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://huggingface.co/Undi95/dbrx-base">Undi95/dbrx-base · Hugging Face</a>: no description found</li><li><a href="https://arstechnica.com/security/2024/03/thousands-of-servers-hacked-in-ongoing-attack-targeting-ray-ai-framework/">Thousands of servers hacked in ongoing attack targeting Ray AI framework</a>: Researchers say it&#39;s the first known in-the-wild attack targeting AI workloads.</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1/tree/main">ai21labs/Jamba-v0.1 at main</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=9K2tkWjOTDU&list=PLMzBiaqOoxQWBs7UpS2WW_Qf3zSuWOux_">mockingbird - Neural hacker</a>: = NEURAL HACKER =___________________________Ahahah!Yo, newb, you ready for a trip down my twisted lane?Dive into my digital chaos, escape the mundane.With a ...</li><li><a href="https://github.com/PKU-YuanGroup/Open-Sora-Plan/">GitHub - PKU-YuanGroup/Open-Sora-Plan: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project.</a>: This project aim to reproduce Sora (Open AI T2V model), but we only have limited resource. We deeply wish the all open source community can contribute to this project. - PKU-YuanGroup/Open-Sora-Plan
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1222823441481334804)** (31 messages🔥): 

- **Misconceptions Cleared on DiT Models**: According to a discussion, a [YouTube video](https://www.youtube.com/watch?v=OPqFpm3wksY) on **DiT (Diffusion Transformers)** fails to adequately explain the technology and misrepresents the role of attention in diffusion models. A member clarified that, unlike the video's implications, normal diffusion models also utilize attention mechanisms.

- **Seeking Clarity on Transformer Diffusion Models**: Amidst confusion, there's a call for clarity regarding the implementation and function of transformers within diffusion models. A member highlighted the prevalent lack of understanding by stating that even the basics of U-nets and transformers are opaque to many people without a "science speak" breakdown.

- **Understanding U-nets at a High Level**: There was an attempt to explain **U-nets** at a high level, describing them as a structure that encodes an image into a lower-dimensional space followed by upsampling. The explanation emphasized the model's ability to discard superfluous information during encoding to simplify subsequent predictive decoding.

- **Request for Resources on Aesthetics and Preference Ranking**: One member announced their experimentation with **RAIG** that yielded promising image outputs, akin to the style of Midjourney, and they solicited for resources related to aesthetics ranking and human preference selection.

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul><li><a href="https://www.youtube.com/watch?v=OPqFpm3wksY">DiT: The Secret Sauce of OpenAI&#39;s Sora &amp; Stable Diffusion 3</a>: Don&#39;t miss out on these exciting upgrades designed to elevate your content creation experience with DomoAI! Go try out: discord.gg/sPEqFUTn7nDiffusion Transf...
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1222961258731667616)** (5 messages): 

- **Optimizing RAG with Int8 and Binary Embeddings**: To tackle memory and cost challenges, [@Cohere](https://t.co/MCnmhQGS7g) introduces the use of Int8 and Binary Embeddings for **RAG pipelines**. This can save both memory and money when working with large datasets.
- **LLMxLaw Hackathon Involvement**: Mark your calendars for an intriguing event where [@hexapode](https://twitter.com/hexapode) and [@yi_ding](https://twitter.com/yi_ding) will speak at the **LLMxLaw Hackathon** at Stanford on April 7th. Interested participants can [get on the list](https://t.co/7lXZBX5APy) to see the location and join the initiative to integrate LLMs in the legal sector.
- **Enhanced Data Representation for RAG**: Take a look at [@palsujit](https://t.co/OKe6goTiJp)'s blog post on improving **RAG/LLM data representation** by using semantic chunking combined with hierarchical clustering and indexing for better results.
- **Building Self-RAG from Scratch**: Discover how to construct a dynamic **Retrieval-Augmented Generation (RAG) model** with built-in reflection, guided by Florian June's blog post featuring a two-step retrieval process triggered by a special token. The full explanation is available in the shared [blog post](https://t.co/JeW2p294Bw).
- **LlamaParse for Enhanced RAG Queries**: @seldo demonstrates in a quick video how **LlamaParse**, powered by LLM, can transform complex insurance policies into simple queries, significantly improving the quality of **RAG queries** against intricate documents. Watch the instructional [video here](https://t.co/jOvvarw1n6).

**Link mentioned**: <a href="https://t.co/7lXZBX5APy">RSVP to LLM x Law Hackathon @Stanford #3 | Partiful</a>: As artificial intelligence (AI) continues to revolutionize industries across the globe, the legal sector is no exception. LLMs, a foundation model capable of understanding and generating natural langu...

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1222807131011420231)** (107 messages🔥🔥): 

- **Understanding Parsing Complexity**: A user inquired about the complexity of **PDF parsing** compared to Word documents. Another user suggested that while parsing tables and images can be challenging, libraries like **LlamaParse** and **Unstructured** could be useful.

- **Visualization of Embeddings with Qdrant**: One user was curious about how to visualize embeddings with **Qdrant**. Another user indicated that Qdrant has a UI accessible at `http://localhost:6333/dashboard`, but the address may differ if not on localhost.

- **GenAI Apps Data Integration**: An article from **Fivetran** was shared that demonstrates how data integration can simplify building GenAI applications and save engineering time, offering a [direct link to the article](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain).

- **Request for Streamlit Integration Examples with LlamaIndex**: A new user inquired for examples of LlamaIndex with Streamlit, and a helpful response included a [link to a related blog post on Streamlit](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/).

- **Challenges with RAG Chatbot Retrieval Accuracy**: A user working on a RAG chatbot faced issues with the retrieval of correct source documents. They were guided to consider metadata fields, prefix issues, and metadata queries, with a suggestion to delete and recreate the index if issues persist.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/">Build a chatbot with custom data sources, powered by LlamaIndex</a>: Augment any LLM with your own data in 43 lines of code!</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/">Indexing - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#text-splitters">Node Parser Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=#keywordnodepostprocessor">Node Postprocessor Modules - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/monsterapi#rag-approach-to-import-external-knowledge-into-llm-as-context>)">no title found</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/production_rag#key-techniques_1>).">Building Performant RAG Applications for Production - LlamaIndex</a>: no description found</li><li><a href="https://docs.llamaindex.ai/en/latest/optimizing/basic_strategies/basic_strategies#metadata-filters>).">Basic Strategies - LlamaIndex</a>: no description found
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1223223563981684747)** (2 messages): 

- **Exploring Model Alignment in LLMs**: A new [blog post](https://blog.premai.io/model-alignment-process/) examines alignment methods like **RLHF**, **DPO**, and **KTO** for **Mistral and Zephyr 7B models**, highlighting the superiority of these methods over standard supervised fine-tuning. It suggests that such alignment techniques have greatly enhanced language generation tasks.
- **Mission to Curate LLM Research**: The [Shure-dev's mission](https://shure-dev.github.io/) focuses on providing a curated selection of high-quality papers on **LLM (Large Language Models)**, aiming to serve as a pivotal resource for researchers keeping pace with the field's rapid advancements.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://shure-dev.github.io/">Awesome-LLM-related-Papers-Comprehensive-Topics</a>: World's Most Comprehensive Curated List of LLM Papers & Repositories</li><li><a href="https://blog.premai.io/model-alignment-process/">Model Alignment Process</a>: The alignment of generative models with human feedback has significantly improved the performance of natural language generation tasks. For large language models (LLMs), alignment methods like reinfor...
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1222812198431096903)** (59 messages🔥🔥): 

- **Looking for O1 Light International Delivery Options**: Members have discussed the possibility of pre-ordering the **01 light** for delivery to countries outside the US. It seems viable to order through a US friend and then have it shipped internationally, as there is no requirement for US citizenship to activate and use the device.

- **Running Open Interpreter in Offline Mode**: Users explored using Open Interpreter in offline mode to avoid API costs, leading to discussions of local LLM providers like Jan, Ollama, and LM Studio. Detailed steps for running with local models, including *`interpreter --model local --api_base http://localhost:1234/v1 --api_key dummykey`*, have been shared, including a [documentation link](https://docs.openinterpreter.com/guides/running-locally) and troubleshooting related to LM Studio usage on Windows.

- **Contribution Invites to OpenInterpreter Projects**: There's an open call for contributions to the **[OpenInterpreter/aifs](https://github.com/OpenInterpreter/aifs)** GitHub repository, with a focus on local semantic search enhancements.

- **AI in Industrial Design**: A user reminisced about early concepts of voice-activated AI companions in industrial design, citing examples like Jerome Olivet's ALO concept phone, which features fully vocalized UX.

- **Using Open Interpreter with Debuggers in Coding IDEs**: A conversation took place about debugging O1 projects using PyCharm and Visual Studio Code, highlighting the potential to review conversations between Open Interpreter and local model servers like LM Studio.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>: no description found</li><li><a href="https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504">Bane No GIF - Bane No Banned - Discover &amp; Share GIFs</a>: Click to view the GIF</li><li><a href="https://github.com/OpenInterpreter/aifs">GitHub - OpenInterpreter/aifs: Local semantic search. Stupidly simple.</a>: Local semantic search. Stupidly simple. Contribute to OpenInterpreter/aifs development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1222798061978255362)** (54 messages🔥): 

- **Open Source LLMs Lack OS Control**: Discussions revealed that open source LLMs are not pre-tuned to control operating systems like Windows or macOS. A user expressed interest in fine-tuning Mistral with synthetic data for this purpose.
- **Shipping O1 Light Internationally**: One user inquired about pre-ordering the O1 light and shipping it internationally via a friend in the US. It was confirmed that self-built O1s would operate anywhere.
- **01 Hardware Compatibility with Arduino**: Users discussed the technical possibility of integrating 01 Light with Arduino. While an ESP32 was the default, there was interest in using it with other boards like Elegoo during the wait.
- **01 Software and Vision Capabilities**: The conversation touched on 01 software's capability to handle vision tasks, mentioning that it can automatically switch to GPT-4-vision-preview when an image is added as a message.
- **Windows Installation for 01 Dev Environment**: Users reported difficulties installing the 01 OS on Windows, with one member actively working to resolve these issues and planning to share their progress.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://app.suno.ai/song/cbf3c6a9-dc4e-4663-b7c5-b7f1aebe687a/">Electric Echoes | Suno</a>: alternative vaper wave ska song. Listen and make your own with Suno.</li><li><a href="https://app.suno.ai/song/6fb4e5a3-fa8e-4a7b-8362-3c5b355d9936">01100001 01101101 00100000 01001001 0010 | Suno</a>: 8bit,chiptune,speedup,Arpeggio,fatbass,Hardbass,FemaleVocals,synthesizer，Electronic，speedup song. Listen and make your own with Suno.</li><li><a href="https://github.com/OpenInterpreter/01/pull/192">[WIP] Fix setup and running on Windows by dheavy · Pull Request #192 · OpenInterpreter/01</a>: Attempts to bridge the gap and facilitate onboarding for windows users by adding missing parts and fixing Win-specific issues. More details to come.
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1222876738049146911)** (8 messages🔥): 

- **Debating the Merits of Compiler Technology**: An individual highlighted the potential advantages of using **compiler technology** like PyTorch/Triton to generate efficient GPU code over manually writing CUDA, despite having no prior experience with compilers.
- **Searching for Benchmark Suggestions**: A user asked for recommendations on benchmarks to test with cutting-edge hardware they're gaining access to, indicating a current focus on *flash attention* and *triton examples*.
- **Sharing CUDA Knowledge Resources**: A link to a lecture series was shared for understanding how **compilers** aid in optimizing memory bandwidth-bound kernels, but not so much for compute-bound kernels.
- **Discussing Distributed Data Parallel (DDP)**: A message mentioned results involving DDP, suggesting that disabled peer-to-peer (p2p) might specifically impact FSDP (Fully Sharded Data Parallel) training setups.
- **CUDA Development IDE Preferences**: A question was raised about preferred IDEs for CUDA development, with a personal affinity for **VSCode** and some experimentation with **CLion** being mentioned.
- **CUDA Programming Cohort Announcement**: A link was shared about Cohere's **CUDA course**, with an announcement of a community-led cohort titled **Beginners in Research-Driven Studies (BIRDS)**, set to begin on April 5th. [Cohere CUDA course announcement](https://x.com/CohereForAI/status/1773419415406809432).

**Link mentioned**: <a href="https://x.com/CohereForAI/status/1773419415406809432">Tweet from Cohere For AI (@CohereForAI)</a>: Our community-led Beginners in Research-Driven Studies (BIRDS) group is kicking off it’s first mini-cohort learning group focused on CUDA Programming for Beginners, beginning on Friday, April 5th 🎉

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1222844463232061541)** (9 messages🔥): 

- **CUDA Coursework Queries**: A user inquired about **CUDA online courses** and received a recommendation to peruse the CUDA related news and materials on the [GitHub - cuda-mode/resource-stream](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses).
- **Classic CUDA Course for Beginners**: Another course suggestion was the **Udacity's Intro to Parallel Programming**, which is also available as a playlist on [YouTube](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2).
- **Inquiry on Setting Up CUDA on MacBook**: A user sought assistance for setting up **CUDA with C++** on an older MacBook (big or sur early 2015).
- **Potential Solution for CUDA on Mac**: The suggestion given was to run a **virtual machine (VM) with Linux** to use CUDA on a MacBook.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://a.co/d/0MImxSS">no title found</a>: no description found</li><li><a href="https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses">GitHub - cuda-mode/resource-stream: CUDA related news and material links</a>: CUDA related news and material links. Contribute to cuda-mode/resource-stream development by creating an account on GitHub.</li><li><a href="https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2">Intro to the Class - Intro to Parallel Programming</a>: This video is part of an online course, Intro to Parallel Programming. Check out the course here: https://www.udacity.com/course/cs344.
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1222965591594369145)** (8 messages🔥): 

- **Ease into CUDA Without the Hassle**: A chat member expressed a desire for a guide to using a **3090 GPU** on a Windows gaming machine without having to install Ubuntu. They are contemplating whether to set up a dual boot as a simpler solution.
- **WSL to the Rescue**: Another suggested using **Windows Subsystem for Linux (WSL)** for CUDA/Torch as it works well and there is no need to replace the Windows installation. They provided a [guide to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install).
- **WSL Version Matters**: It was clarified that when using WSL for CUDA, one must ensure the system is set to **WSL2**, because WSL1 is not sufficient for the task.

**Link mentioned**: <a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: Install Windows Subsystem for Linux with the command, wsl --install. Use a Bash terminal on your Windows machine run by your preferred Linux distribution - Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin,...

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1222918447067959406)** (57 messages🔥🔥): 

- **Fine-tuning Experiments Stop Due to Scope**: Fine-tuning using [FSDP+QLoRA](https://github.com/AnswerDotAI/fsdp_qlora) on `meta-llama/Llama-2-13b-chat-hf` was halted as it was not aligning with the intended experiments related to ring-attention. The repo did have Flash-Attention-2 but incorporating Ring-Attention into it was uncertain.

- **Troubleshooting Loss Issues with Ring-Attention**:
  - Debugging was performed on the loss issue in the `modeling_llama.py` for `LlamaForCausalLM` using CrossEntropyLoss.
  - The issue was traced to incorrect broadcast handling of the labels in multi-GPU environments and was subsequently patched.

- **Exploration of Long Context Training**: Rigorous testing showcased successful training of the tinyllama model with context lengths from 32k to 100k on A40 GPUs, each having 48GB VRAM. It was observed that llama-2 7B could run on 2x A100, utilizing 54GB with 4k sequence length.

- **VRAM Requirements Discussed for Llama 7B Training**: A [Reddit link](https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/) discussing the VRAM requirements for training models like Llama 7B with QLoRA and LoRA was shared, highlighting the necessity of ample VRAM for large model training. 

- **Dataset Search for Long-Context Models**: Members canvassed for long-context datasets, suggesting resources like the `booksum` dataset from Hugging Face’s Long-Data-Collections and others for potentially fine-tuning models suitable for lengthy text inputs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/18o5u0k/helpful_vram_requirement_table_for_qlora_lora_and/">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.together.ai/blog/llama-2-7b-32k">Preparing for the era of 32K context: Early learnings and explorations</a>: no description found</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>: Training LLMs with QLoRA + FSDP. Contribute to AnswerDotAI/fsdp_qlora development by creating an account on GitHub.</li><li><a href="https://wandb.ai/cataluna84/fsdp_qlora/runs/o59wbxpr/workspace?nw=nwusercataluna84">cataluna84</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1222990325698789376)** (5 messages): 

- **Navigating Zhihu for Triton Tutorials**: A member sought assistance in Mandarin to log in to [Zhihu](https://www.zhihu.com/signin?next=%2F), which boasts excellent Triton tutorials. The member managed to create an account on iOS but needed help locating the app's scan feature for login verification.

- **Scan Button Found**: Another member provided a solution for locating the "Scan" button within the Zhihu app, allowing for successful login.

- **Triton Content Galore**: After successfully logging in, the member confirmed finding a wealth of Triton-related content on Zhihu.

- **Wish for Chinese Search Glossary**: A mention was made regarding the difficulty in searching due to language barriers, noting that a Chinese glossary of search terms would be helpful.

**Link mentioned**: <a href="https://www.zhihu.com/signin?next=%2F">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1222828137361051720)** (24 messages🔥): 

- **Sync Issue in Triton-Viz Identified**: A contributor identified an error caused by an update in `triton-viz`, particularly due to changes in a [recent pull request](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21), which led to compatibility issues with outdated versions of triton installed in the notebook.
- **Workaround for Triton-Viz Update**: Members shared solutions to overcome the error by installing an older version of `triton-viz` using the command `pip install git+https://github.com/Deep-Learning-Profiling-Tools/triton-viz@fb92a98952a1e8c0e6b18d19423471dcf76f4b36`, and for some the workaround was effective.
- **Persistent Installation Errors**: Despite applying the suggested fixes and restarting the runtime, some users continued to face import errors with `triton-viz`. Suggestions to restart the runtime and re-execute the installation command were given.
- **Official Fix and Installation Procedure**: Srush1301, presumably a maintainer, provided an official fix with a detailed installation procedure that works with the latest triton. This should resolve the issues discussed, as confirmed by a user who mentioned it worked fine.
- **Clarity on Triton Usage on Windows**: A member asked if Triton can be used on Windows, to which another replied Triton itself is not Windows-compatible but Triton-based puzzles can be run on Colab or other non-GPU hosts.

**Link mentioned**: <a href="https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21.">[TRITON] Sync with triton upstream by Jokeren · Pull Request #19 · Deep-Learning-Profiling-Tools/triton-viz</a>: no description found

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1223140446587457618)** (1 messages): 

- **Peek Into Model Popularity**: OpenRouter introduced **App Rankings for Models**, displayed on a new **Apps** tab on model pages. The tab ranks apps based on their usage of a specific model, showing top public apps and tokens processed, as seen for [Claude 3 Opus](https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps).

- **Benchmarking Claude 3 Opus**: [Claude 3 Opus](https://www.anthropic.com/news/claude-3-family) is hailed as Anthropic's most capable model, excelling in complex tasks with high-level intelligence and fluency, and the benchmark results are available in the launch announcement.

- **Community Projects On Spotlight**: Notable community projects include a Discord bot, [Sora](https://github.com/mintsuku/sora), that leverages the Open Router API for enhanced server conversations, and a platform, [nonfinito.xyz](https://nonfinito.xyz/), allowing users to create and share model evaluations.

- **New API and Improved Clients**: There's a simpler /api/v1/completions API mirroring the chat API functionality with a prompt parameter, alongside improved OpenAI API client support, and OpenRouter has stopped using Groq for Nitro models due to excessive rate limiting.

- **Optimized Crypto Payments**: OpenRouter is improving cryptocurrency payments by reducing gas costs, utilizing Base chain, an Ethereum L2, to make transactions more affordable for users.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/anthropic/claude-3-opus?tab=apps">Claude 3 Opus by anthropic | OpenRouter</a>: Claude 3 Opus is Anthropic&#x27;s most powerful model for highly complex tasks. It boasts top-level performance, intelligence, fluency, and understanding.  See the launch announcement and benchmark re...</li><li><a href="https://github.com/mintsuku/sora">GitHub - mintsuku/sora: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers.</a>: Sora is a Discord bot that integrates with the Open Router API to facilitate conversation in Discord servers. - mintsuku/sora</li><li><a href="https://nonfinito.xyz/">Evaluations - Non finito</a>: no description found
</li>
</ul>

</div>
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1222809408849973308)** (107 messages🔥🔥): 

- **Image Support Clarification**: Users discussed whether image input is supported, with a clear statement that for models accepting image input, "you just upload it."
- **No Prefill for Claude in OpenRouter**: There was confusion about **Claude in OpenRouter** supporting a prefill feature, but it was clarified that there is no prefill in **Sillytavern**, and one must create a prompt with absolute depth 0, which acts similarly to a prefill.
- **Databricks DBRX and Gemini Pro Excitement**: The release of the new model from Databricks, **DBRX**, sparked interest, and positive sentiments were shared about **Gemini Pro 1.5**'s performance, although there were reports of **error 429** (rate limit) and recommendations to try after 00:00 UTC.
- **Challenges with Model Availability and Error Codes**: Multiple users reported issues with models being down or encountering error codes such as **error 502 for Claude 2.1** and **error 524** related to Cloudflare CDN timeouts. The chat discussed potential fixes and the possibility of setting up failover models.
- **Discussion on Self-Moderated ClaudeAI and Price Comparisons**: There was a discussion about the benefits of using **ClaudeAI through OpenRouter**, especially for roleplay and sensitive content with less false positive detection. **OpenRouter's API** standardizes access and pricing was confirmed to be at cost, the same as directly using the official **ClaudeAI API**.


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://olympia.chat">Olympia | Better Than ChatGPT</a>: Grow your business with affordable AI-powered consultants that are experts in business strategy, content development, marketing, programming, legal strategy and more.</li><li><a href="https://github.com/OlympiaAI/open_router">GitHub - OlympiaAI/open_router: Ruby library for OpenRouter API</a>: Ruby library for OpenRouter API. Contribute to OlympiaAI/open_router development by creating an account on GitHub.</li><li><a href="https://openrouter.ai/models?o=pricing-low-to-high">OpenRouter</a>: Browse models on OpenRouter</li><li><a href="https://openrouter.ai/docs#limits">OpenRouter</a>: Build model-agnostic AI apps</li><li><a href="https://openrouter.ai/docs#model-routing">OpenRouter</a>: Build model-agnostic AI apps</li><li><a href="https://openrouter.ai/models/anthropic/claude-3-opus:beta">Claude 3 Opus by anthropic | OpenRouter</a>: This is a lower-latency version of [Claude 3 Opus](/models/anthropic/claude-3-opus), made available in collaboration with Anthropic, that is self-moderated: response moderation happens on the model&#x...
</li>
</ul>

</div>
  

---



**AI21 Labs (Jamba) ▷ #[announcements](https://discord.com/channels/874538902696914944/874538945168408606/1222913979249328190)** (1 messages): 

- **AI21 Labs Unveils Jamba**: AI21 Labs proudly announces [Jamba](https://www.ai21.com/jamba), a **state-of-the-art model** with a novel architecture that merges Mamba and elements of the Transformer. It's the first open-sourced model from AI21, with key features such as a 256K context window and the ability to fit up to 140K context on a single GPU.
  
- **Jamba Elevates Long Context Performance**: Jamba delivers an **unprecedented 3X throughput** for long context use cases like question/answering and summarization, addressing critical enterprise challenges and setting a new standard in GenAI efficiency.

- **Open Access to Next-Gen AI**: This groundbreaking model is now available with **open weights under the Apache 2.0** license, and users can access it on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) as well as the upcoming inclusion in the NVIDIA API catalog.

- **Harness GenAI with NVIDIA NIM**: The message details an expansive offering of production-ready APIs that run anywhere with NVIDIA NIM, touching on *models*, *integrations*, *run anywhere*, *how to buy*, *use cases*, *ecosystem*, *resources*, and *docs*. However, the sections beyond "Harness Generative" are incompletely rendered and thus specifics cannot be provided.

- **Industry Buzz Around Jamba**: A [TechCrunch Exclusive](https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/) discusses the movement towards generative AI models with longer contexts and highlights that AI21 Labs' Jamba model competes with models like OpenAI's ChatGPT and Google's Gemini, by offering large context windows that are less compute-intensive.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="http://ai.nvidia.com/)">Production-Ready APIs That Run Anywhere</a>: Experience the leading models to build enterprise generative AI apps now.</li><li><a href="https://techcrunch.com/2024/03/28/ai21-labs-new-text-generating-ai-model-is-more-efficient-than-most/)">AI21 Labs&#039; new AI model can handle more context than most | TechCrunch</a>: Increasingly, the AI industry is moving toward generative AI models with longer contexts. But models with large context windows tend to be
</li>
</ul>

</div>
  

---


**AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1222940200183795842)** (40 messages🔥): 

- **Jamba Soon on AI21 Labs Platform**: AI21 Labs' staff member confirmed that Jamba is currently available via **Hugging Face**, and an aligned version will soon be available through their **SaaS platform**.
- **Multilingual Capabilities Shine**: Users have confirmed with AI21 staff that Jamba has been trained on multiple languages, including **German**, **Spanish**, and **Portuguese**, although Korean performance may not be as strong.
- **Awaiting Jamba's Technical Deep Dive**: AI21 plans to **release a white paper** with details about Jamba's training, including the number of tokens, languages, and hyperparameters.
- **No Hugging Face Space Planned for Jamba**: An AI21 staff member indicated that there are no current plans for a **Hugging Face Space** where Jamba could be tested.
- **Open-sourced Jamba Stirring Excitement**: The Jamba model's open-sourcing generated a lot of excitement among users, leading to questions about comparisons with other models like Mamba, potential for smaller versions, batch inference engines, and performance on coding tasks.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Jamba_(language_model)">Jamba (language model) - Wikipedia</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.
</li>
</ul>

</div>
  

---


**AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1222915429341855744)** (56 messages🔥🔥): 

- **Minimum Pricing Policy Vanishes**: AI21 Labs has removed the **$29/mo** minimum charge for their models. However, the pricing page tooltip still shows the minimum fee, leading to a brief confusion.
- **Fine-Tuning on the Horizon**: There's an ongoing consideration about reintroducing fine-tuning at AI21 Labs, but **no firm date** has been set for its comeback.
- **Exploring Jamba's Efficiency**: A discussion took place regarding how **Jamba** achieves high throughput despite having transformer blocks that scale with sequence length squared. **Mamba** and **Mixture-of-Experts (MoE)** layers contribute to a different, sub-quadratic scaling behavior, making Jamba faster.
- **Quantization Clarification**: Members debated whether **Jamba** models could be quantized for reduced memory use on GPUs. Links to a Hugging Face quantization guide and comments on creating quantized models even with mid-range hardware were shared.
- **Transformation Confusion**: There was a clarification that the **MoE** consists of *Mamba* blocks, not transformers, and that within Jamba, there's a ratio of one transformer per Jamba and seven other types of layers.

**Link mentioned**: <a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://huggingface.co/docs/accelerate/en/usage_guides/quantization&ved=2ahUKEwjQ-4eopJmFAxWaXmwGHZPhBN8QFnoECBMQAQ&usg=AOvVaw2RxBEXoJMjtWqDScwaFZqc">no title found</a>: no description found

  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1222829670811172865)** (56 messages🔥🔥): 

- **Conda Compiler-Induced Madness**: A member expressed frustration while trying to debug a **Conda metal compiler bug**, noting the ordeal was driving them insane enough to require breaks for making memes.
- **Tinygrad and Virtual Memory**: A discussion was raised about the importance of virtual memory in the tinygrads software stack, with a member considering whether to support virtual memory in their cache/MMU project.
- **AMD Access Via Runpod**: It was mentioned that AMD now appears on Runpod, but users are required to schedule a meeting to gain access, with a hint of sarcasm regarding AMD's perception of customer complaints as PR issues rather than technical ones.
- **Intel’s Favorable Customer Service Compare to AMD**: Members compared Intel's helpfulness and better price-performance of their GPUs to AMD's buggy software, while discussing fluctuating market availability and regional pricing differences.
- **Seeking Optimizations for Intel Arc**: A member described inefficiencies with the transformers/dot product attention implementation for Intel Arc in the IPEX library and suggested that significant performance improvements could be gained with simple optimizations, based on their experience with enhancing stable diffusion performance.
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1222844163901362206)** (39 messages🔥): 

- **FPGA Acceleration Inquiry**: A member inquired about porting [tinygrad to custom hardware accelerators](https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md) like FPGA, and was directed to the tinygrad documentation for adding new accelerators.

- **Deep Dive into Dimension Merging**: [Merging dimensions in tinygrad is explored](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/mergedim.md) with a need for further insights on multiple views interaction, specifically why certain operations create multiple views.

- **Exploring In-Place Operations and Their Errors**: A member questioned potential in-place modifications during evaluation in tinygrad, supported by an "AssertionError" they encountered, and discussed the use of `+=` in Python.

- **Tensor View Simplification Effort**: There was significant engagement in simplifying tensor views within tinygrad, leading to discussions about the necessity and accuracy of retaining multiple views, symbolic representations, and a [proposed change](https://github.com/tinygrad/tinygrad/pull/39888) to address this issue.

- **Combining std+mean into One Kernel**: Efforts to simplify the standard deviation and mean calculations into a single kernel were mentioned, with a provided [write-up on kernel fusion](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/scheduleitem.md) to facilitate solving the associated bounty challenge.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nEN9q_PK8SHqrRcBIC6LnrJQE9FVvJE6?usp=sharing">Google Colaboratory</a>: no description found</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3988.">Build software better, together</a>: GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md">tinygrad/docs/adding_new_accelerators.md at master · tinygrad/tinygrad</a>: You like pytorch? You like micrograd? You love tinygrad! ❤️  - tinygrad/tinygrad
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1222807309600952381)** (93 messages🔥🔥): 

- **OpenAI Prepaid Model Insights**: An explanation for OpenAI's prepaid model surfaced through an active discussion featuring [a response from LoganK](https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) highlighting fraud prevention and a pathway to higher rate limits as key drivers. Concerns about opaque pricing due to unspent credits was paralleled with [Starbucks' revenue from unused balances](https://arstechnica.com/tech-policy/2024/01/hard-to-spend-card-balances-net-starbucks-200m-per-year-says-consumer-group/).

- **AI21 Unveils Transformer Mamba Hybrid "Jamba"**: AI21 announced a new model called "Jamba," which combines Structured State Space and Transformer architectures. The model was shared on [Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1) and discussed for its notable MoE structure, though it was pointed out that there was no easy way to test the model at the time of the announcement.

- **Grok-1.5 Model's Impressive Reasoning Abilities**: xAI introduced Grok-1.5, boasting significant performance enhancements over its predecessor, especially in coding and math-related tasks, with notable scores on prominent benchmarks. Users eagerly awaited access to Grok-1.5 on the xAI platform, with discussions emphasizing Grok's capabilities [in the official blog post](https://x.ai/blog/grok-1.5).

- **Discussion on 1-bit (Ternary) LLMs**: A debate about the accuracy of referring to three-valued (-1, 0, 1) models as "1-bit LLMs" led to discussions around BitNet's influence and training methodology for such models. The conversation included a reference to the BitNet paper and clarification that these models are trained from scratch, not quantized.

- **Major Moves in AI Leadership**: Emad Mostaque's departure from Stability AI incited discussion about the causes and implications, illuminated by an interview on [Peter Diamandis's YouTube channel](https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa) and a detailed [backstory on archive.is](https://archive.is/8QkSl). The shifting leadership underscores emergent dynamics and potential instability within the generative AI sector.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.ai/blog/grok-1.5">Announcing Grok-1.5</a>: no description found</li><li><a href="https://www.ai21.com/jamba">Introducing Jamba</a>: A groundbreaking SSM-Transformer Open Model</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: We are building a next-generation AI developer platform for the world. Check out our latest post: The Next Big Step in Mojo🔥 Open Source</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>: no description found</li><li><a href="https://x.com/yampeleg/status/1773401745269379409?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Yam Peleg (@Yampeleg)</a>: Performance:  - 3x Throughput in comparison to transformer. - 140K fits in a single GPU (!) - 256K context in general.</li><li><a href="https://x.com/dwarkesh_sp/status/1773381318266786180?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Dwarkesh Patel (@dwarkesh_sp)</a>: Had so much fun chatting with my friends @TrentonBricken and @_sholtodouglas.  No way to summarize it, except:  This is the best context dump out there on how LLMs are trained, what capabilities they&...</li><li><a href="https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: @atroyn @OpenAIDevs This is driven by two main factors:   - wanting to prevent fraud and make sure tokens are used my real people - wanting to give devs a clearer path to higher rate limits (by allowi...</li><li><a href="https://openai.com/blog/navigating-the-challenges-and-opportunities-of-synthetic-voices">Navigating the Challenges and Opportunities of Synthetic Voices</a>: We’re sharing lessons from a small scale preview of Voice Engine, a model for creating custom voices.</li><li><a href="https://x.com/clementdelangue/status/1771395468959813922?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from clem 🤗 (@ClementDelangue)</a>: Should we acquire Stability and open-source SD3?</li><li><a href="https://www.youtube.com/watch?v=57LqvutrOI8">Perplexity CEO: Disrupting Google Search with AI</a>: One of the most preeminent AI founders, Aravind Srinivas (CEO, Perplexity), believes we could see 100+ AI startups valued over $10B in our future. In the epi...</li><li><a href="https://x.com/GoogleDeepMind/status/1773422123869974878?s=20">Tweet from Google DeepMind (@GoogleDeepMind)</a>: Congratulations to our CEO and co-founder @demishassabis who has been awarded a Knighthood by His Majesty for services to Artificial Intelligence.  ↘️ Quoting Demis Hassabis (@demishassabis)   Delight...</li><li><a href="https://arstechnica.com/tech-policy/2024/01/hard-to-spend-card-balances-net-starbucks-200m-per-year-says-consumer-group/">Consumer group wants to end $255M “gift card loophole” for Starbucks and others</a>: Changes to Washington&#39;s gift card laws could affect cardholders nationwide.</li><li><a href="https://x.com/anissagardizy8/status/1773759144425930962?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Anissa Gardizy (@anissagardizy8)</a>: breaking: Microsoft and OpenAI are drawing up plans for a $100 billion AI supercomputer  The supercomputer, codenamed &#34;Stargate,&#34; would contain *millions* of GPUs and require several gigawatts...</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction Since the surge in interest sparked by Mixtral, research on mixture-of-expert (MoE) models has gained significant momentum. Both researchers an...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B">Qwen/Qwen1.5-MoE-A2.7B · Hugging Face</a>: no description found</li><li><a href="https://youtu.be/e1UgzSTicuY?si=rF7LX1X6Kt7N2YRa">Why I&#39;m Leaving My Company Immediately (Stability AI) w/ Emad Mostaque | EP #93</a>: In this episode, Peter and Emad discuss Emad&#39;s stepping down as CEO of StabilityAI, his next steps into decentralized AI, and why there is so much urgency to...</li><li><a href="https://www.neelnanda.io/mechanistic-interpretability/glossary">A Comprehensive Mechanistic Interpretability Explainer &amp; Glossary &mdash; Neel Nanda</a>: no description found</li><li><a href="https://youtu.be/KV5gbOmHbjU?si=a0r05EWMctXB_l7v">A Walkthrough of A Mathematical Framework for Transformer Circuits</a>: A Walkthrough of A Mathematical Framework for Transformer Circuits - I read through the paper, and give a bunch of thoughts, hot takes, and clarifications on...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/cqJMkfO2xm">Reddit - Dive into anything</a>: no description found</li><li><a href="https://www.securityfrontiers.ai/">Security Frontiers | Where GenAI Meets Security Innovation</a>: Learn from industry experts and GenAI innovators on integrating Generative AI to optimize security operations, automate tasks, and strengthen cyber defenses.</li><li><a href="https://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a>: no description found</li><li><a href="https://archive.is/8QkSl">Inside Stability AI's bad breakup with Coatue and Lightspeed Venture &#x2026;</a>: no description found
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1222954808143511592)** (45 messages🔥): 

- **Moe Moe or Noe Noe?**: Conflicted first impressions of **Qwen MOE's** performance sparked a brief discussion, with one member initially enthusiastic only to later describe it as "kind of ass."
- **Jamba's Juicy Details Revealed**: **AI21 Labs** announced their new model, **Jamba**, standing out with a **256k token context window** and **MoE architecture with 12b parameters active**. Their release includes a [blog post](https://www.ai21.com/blog/announcing-jamba) and an apparent training time that seems remarkably brief, with knowledge cutoff on March 5, 2024.
- **High Cost for High Performance**: A statement was made highlighting **GB200-based server prices** as ranging between **US$2-$3 million each**, leading to discussions on alternative hardware choices due to the hefty price tags.
- **GPT4's Beast Mode**: A comparison was drawn to **GPT-4's 32k version**, which seems to be recognized as a substantial upgrade, potentially altering the performance dynamics in AI applications.
- **Datasets for Ring-Attention**: In pursuit of suitable **long-context datasets**, a member shares one from Hugging Face's collections and another provides an alternate link, [MLDR dataset on Hugging Face](https://huggingface.co/datasets/Shitao/MLDR), both geared for long sequence training needs.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/Shitao/MLDR">Shitao/MLDR · Datasets at Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/datasets/togethercomputer/Long-Data-Collections">togethercomputer/Long-Data-Collections · Datasets at Hugging Face</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1222943820828905515)** (16 messages🔥): 

- **Request for Assistance with LISA and Jamba**: A member is seeking help for integrating **LISA** and testing the **Jamba** (mamba moe) to see if it works out-of-the-box.

- **TorchTune Optimizations Revealed**: After a conversation with the **torchTune** team, a participant shares that using **bf16 for everything, including the optimizer**, leads to dramatic memory usage reductions and matches **fp32** or mixed precision training in terms of stability.

- **bf16 Optimization Real-World Applications**: The team successfully applied **bf16** to SGD and plans to add support for Adam optimizer soon. This methodology is highlighted for its significant memory efficiency.

- **PagedAdamW vs. 8bit Adam**: A member shares results from testing **PagedAdamW**, which results in more memory savings as compared to 8bit Adam. The difference is substantial, cutting down peak memory usage by nearly half.

- **Training Control Differences Affect Memory Savings**: Another member points out **axolotl** does not offer the same level of training control as **torchTune**, which may affect memory saving outcomes. The conversation suggests that memory savings are attributed to a combination of **PagedAdamW** and optimizing the backward pass.

**Link mentioned**: <a href="https://github.com/pytorch/torchtune/blob/main/recipes/configs/llama2/7B_full_single_device_low_memory.yaml">torchtune/recipes/configs/llama2/7B_full_single_device_low_memory.yaml at main · pytorch/torchtune</a>: A Native-PyTorch Library for LLM Fine-tuning. Contribute to pytorch/torchtune development by creating an account on GitHub.

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1222887793328656415)** (5 messages): 

- **Quest for Fine-Tuning Knowledge**: A member inquired about the best resources to learn how to fine-tune/train open source models, expressing a desire to understand the foundations before utilizing **axolotl** for their tasks.
- **Choosing the Right Model for Text Classification**: Another member questioned which base model would be optimal for fine-tuning on a text classification task with around 1000 data points, using only a T4 GPU and specifying the need for English language support. The member asked if **qwen with Lora/qlora** was a suitable choice given their GPU constraints.
- **Mistral Recommended for English Text Classification**: A member recommended using **Mistral** for English text tasks, suggesting that **qlora** with a batch size of 1 could be adequate for small VRAM GPUs like the mentioned 4070ti with 12GB VRAM.
- **No Experience with Qwen**: The same member who recommended Mistral noted they had no experience working with **qwen**.
  

---


**OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1223024934075236444)** (25 messages🔥): 

- **Suikamelon shares conversation modeling strategy**: Detailed a formatting method using special block characters `▀` and `▄` as start and end tokens for model prompts, allowing for seamless multi-character dialogue generation without distinguishing between "user" or "model".
- **Jaredquek discusses dataset structure for nuanced conversation training**: Revealed unique dataset construction mixing original text with conversations in chatml form, which includes repetitions to boost information accuracy without overfitting, even proving effective on smaller models like Hermes 7B.
- **Repetition in Training Data questioned and defended**: While some were unsure about the benefits of verbatim repetition in training data, others supported the technique, linking a research [paper on data ordering](https://arxiv.org/abs/2310.10638) to argue in favor of the practice.
- **Skepticism about bland repetition techniques**: Jaredquek and suikamelon engaged in a conversation debating the effectiveness of bland repetition, where data is cloned multiple times to focus on the same topic, versus varied repetition.
- **New approaches and ongoing experiments**: Jaredquek plans to test full fine-tuning methods on larger models like Galore, after achieving satisfactory results with current techniques, despite facing some out-of-memory challenges.

**Link mentioned**: <a href="https://arxiv.org/abs/2310.10638">In-Context Pretraining: Language Modeling Beyond Document Boundaries</a>: Large language models (LMs) are currently trained to predict tokens given document prefixes, enabling them to directly perform long-form generation and prompting-style tasks which can be reduced to do...

  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1222928694398418946)** (1 messages): 

- **New Discussion Channel for OpenGPTs**: A new Discord channel, <#1222928565117517985>, has been created specifically for discussions related to the [OpenGPTs GitHub repository](https://github.com/langchain-ai/opengpts). This platform offers a space for community contributions and collaboration on the project.

**Link mentioned**: <a href="https://github.com/langchain-ai/opengpts">GitHub - langchain-ai/opengpts</a>: Contribute to langchain-ai/opengpts development by creating an account on GitHub.

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1222843915405885522)** (57 messages🔥🔥): 

- **SQL AI Chatbot Memory Challenge**: A member is developing an SQL AI Chatbot but is facing difficulties as the chatbot doesn't remember previous messages and responses. They seek guidance on implementing history-aware responses.
- **Natural Language Query for Product Recommendations**: Discussion about building a bot to recommend products based on natural language queries like "planning to own a pet" using either a vector database for semantic search or extracting attributes and using an SQL agent.
- **Storing Vectored Data for RAG Applications**: A conversation about how to store vectored data of documents when building a Retrieval-Augmented Generation (RAG) AI application using a PostgreSQL database, and handling document differentiation on a user basis.
- **How to Use on_agent_finish in StreamingStdOutCallbackHandler**: Dialog around the correct usage of `on_agent_finish` in `StreamingStdOutCallbackHandler` within the LangChain framework, with members seeking and providing clarification, even though not always successfully.
- **Pythia: The Oracle of AI Hallucination Detection**: An announcement about **Pythia**, a Proof of Concept application for AI hallucination detection, and a request for guidance on integrating it into the langchain ecosystem.
- **Langchain Skepticism, Seeking Rebuttals**: A member called for a point-by-point rebuttal of the "Don't use LangChain" sentiment that is spreading as fear, uncertainty, and doubt (FUD) on the internet, suggesting it could be used to improve the roadmap of LangChain.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/templates/rag-supabase#setup-supabase-database>)">rag_supabase | 🦜️🔗 Langchain</a>: This template performs RAG with Supabase.</li><li><a href="https://python.langchain.com/docs/templates/rag-lantern#setup-lantern-database>)">rag_lantern | 🦜️🔗 Langchain</a>: This template performs RAG with Lantern.</li><li><a href="https://js.langchain.com/docs/integrations/vectorstores/pgvector#usage>)">PGVector | 🦜️🔗 Langchain</a>: To enable vector search in a generic PostgreSQL database, LangChain.js supports using the pgvector Postgres extension.
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1222930753562611743)** (7 messages): 

- **OpenGPTs Custom Food Ordering Hack**: A member has demonstrated an innovative integration of custom food ordering API with OpenGPTs, showcasing its adaptability as a platform. They encourage feedback and shared a [YouTube tutorial](https://youtu.be/V1SKJfE35D8) on how to "Hack OpenGPT to Automate Anything".
  
- **Automate Your Code Review Process**: An announcement about a builder that creates AI pipelines to automate code validation, security, and eliminate tedious tasks was shared, along with [a link to the tool](https://gitgud.autonoma.app) and an accompanying explanatory [YouTube video](https://www.youtube.com/watch?v=6kfr1lqw2gg).
  
- **Bringing Company Data to GenAI Apps**: A recent [blog post](https://www.fivetran.com/blog/building-a-chatbot-with-fivetran-and-langchain) was shared, discussing how Fivetran assists in integrating company data with RAG apps, along with a [survey link](https://www.surveymonkey.com/r/6X9MDJF) for more information.

- **Galaxy AI Unlocks Access to Premier AI Models**: GalaxyAI is offering a **free** API service that provides access to high-end AI models such as **GPT-4** and **Gemini-PRO**, with OpenAI format compatibility for easy project integration. Interested users can [Try Now](https://galaxyapi.onrender.com).

- **Dive into Model Alignment in LLMs**: A blog post examining model alignment, focusing on RLHF, DPO, and KTO methods and their practical application on Mistral and Zephyr 7B models was highlighted with [a link to the full article](https://blog.premai.io/model-alignment-process/).
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://blog.premai.io/model-alignment-process/">Model Alignment Process</a>: The alignment of generative models with human feedback has significantly improved the performance of natural language generation tasks. For large language models (LLMs), alignment methods like reinfor...</li><li><a href="https://producthunt.com/posts/sciphi?"> SciPhi - One-click RAG deployment for developers | Product Hunt</a>: SciPhi is a cloud platform for developers that simplifies building and deploying serverless RAG pipelines. Built with the open source R2R framework, it enables developers to focus on innovative AI app...</li><li><a href="https://youtu.be/V1SKJfE35D8">Hack OpenGPT to Automate Anything</a>: Welcome to the future of custom AI applications! This demo showcases the incredible flexibility and power of OpenGPTs, an open source project by LangChain. W...</li><li><a href="https://gitgud.autonoma.app">GitGud</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=6kfr1lqw2gg">GitGud Demo</a>: no description found
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1223049799926939730)** (2 messages): 

- **Data Integration Meets AI**: The article shared discusses integrating company data with RAG app using **Fivetran** to efficiently build General AI applications, reducing engineering work on data management. For more personalized information, there's an invitation to reach out via a [SurveyMonkey link](https://www.surveymonkey.com/r/6X9MDJF).
- **Custom API Merges with OpenGPTs**: A community member showcased the adaptability of **OpenGPTs** by integrating a *custom food ordering API*, demonstrating the platform's potential beyond basic usage. The shared [YouTube video](https://youtu.be/V1SKJfE35D8) provides a visual demo with the member inviting feedback on this innovative application.

**Link mentioned**: <a href="https://youtu.be/V1SKJfE35D8">Hack OpenGPT to Automate Anything</a>: Welcome to the future of custom AI applications! This demo showcases the incredible flexibility and power of OpenGPTs, an open source project by LangChain. W...

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1222911986984288380)** (28 messages🔥): 

- **Introducing Jamba**: AI21 Studios announces [Jamba](https://www.ai21.com/blog/announcing-jamba), a hybrid of Mamba's Structured State Space model and the traditional Transformer architecture, boasting a 256K context window and remarkable throughput and efficiency. The model outperforms or matches others in its class and is released with open weights under the Apache 2.0 license.
  
- **Jamba's Technical Specs Revealed**: Discussion in the channel points out the specifications of Jamba, highlighting it as a [Mixture of Experts (MoE)](https://huggingface.co/ai21labs/Jamba-v0.1), featuring 12B parameters for inference purposes, with a total of 52B, and activating 16 experts, with only 2 active per token.

- **Some Confusion Around Model Identity**: A member points out confusion regarding the actual architecture of Jamba, questioning if it could be more accurately described as a larger "striped hyena" rather than Mamba, given its hybrid nature and inclusion of attention mechanisms.

- **Qwen MoE Model Competency Discussed**: Another hybrid model `Qwen1.5-MoE-A2.7B` is brought into the discussion, which supposedly matches 7B models' performance with only 2.7B activated parameters, indicating significant efficiency in training expense. The model information and resources can be found in various places, with links provided including a [GitHub repo](https://github.com/QwenLM/Qwen1.5) and [Hugging Face space](https://huggingface.co/Qwen).

- **Scaling Paper for Hybrid Architectures Shared**: A member shares a [scaling paper](https://arxiv.org/abs/2403.17844) that conducts extensive research on beyond Transformer architectures. It is noted that 'striped architectures' perform better due to the specialization of layer types.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters</a>: GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD Introduction Since the surge in interest sparked by Mixtral, research on mixture-of-expert (MoE) models has gained significant momentum. Both researchers an...</li><li><a href="https://x.ai/blog/grok-1.5">Announcing Grok-1.5</a>: no description found</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21&#x27;s Groundbreaking SSM-Transformer Model</a>: Debuting the first production-grade Mamba-based model delivering best-in-class quality and performance.</li><li><a href="https://fxtwitter.com/MichaelPoli6/status/1773370168929825073?s=20">Tweet from Michael Poli (@MichaelPoli6)</a>: 📢New research on mechanistic architecture design and scaling laws.  - We perform the largest scaling laws analysis (500+ models, up to 7B) of beyond Transformer architectures to date  - For the first...</li><li><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_moe">transformers/src/transformers/models/qwen2_moe at main · huggingface/transformers</a>: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1222895903157911585)** (25 messages🔥): 

- **Microsoft GenAI Snags New Senior Researcher**: [Liliang Ren](https://x.com/liliang_ren/status/1773118751413596588?s=46) announces his move to Microsoft GenAI as a Senior Researcher to focus on creating **neural architectures** that are efficient and scalable, using small language models with less than 100B parameters.
- **Databricks Inherits Megablocks**: The creator of Megablocks has handed over the project to Databricks, as announced on [Twitter](https://x.com/tgale96/status/1773342375806374307?s=46). This move is expected to provide a stable long-term home for the development and use of **Megablocks**.
- **Debating the Scale of "Small"** in AI Language Models: Discussion about what constitutes a "small language model" suggests that under 100B parameters might be the threshold, though the term "small" is seen as a bit of a misnomer.
- **Prospects of AI Open Data from OMB Guidance**: The new **OMB guidance** on AI is being sifted through, with speculation about potential interesting open datasets that might emerge from the government.
- **Elon's Attention and AI Releases**: Commentary suggests that AI releases, such as those from OpenAI, might be timed for periods when Elon Musk is likely to be paying attention.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/tgale96/status/1773342375806374307?s=46">Tweet from Trevor Gale (@Tgale96)</a>: Some of you noticed that megablocks is now databricks/megablocks. I gave the project to them this week and I couldn’t think of a better long-term home for it. I’m looking forward to watching it grow a...</li><li><a href="https://x.com/liliang_ren/status/1773118751413596588?s=46">Tweet from Liliang Ren (@liliang_ren)</a>: Personal Update: I will join Microsoft GenAI as a Senior Researcher starting from this summer, focusing on the next generation of neural architectures that are both efficient and extrapolatable. We ar...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1222899275919396894)** (5 messages): 

- **Clarification on Model Fine-Tuning**: A member clarified their previous statement, mentioning they were referring to the fine-tuning and training of open source models, not other topics.
- **LLM Course on Github**: A link to a [GitHub repository](https://github.com/mlabonne/llm-course) was shared, offering a course with roadmaps and Colab notebooks to help individuals get into Large Language Models.
- **Introduction of Jamba, an Open Model**: AI21 Labs has introduced **Jamba**, a Mamba-based model combining Structured State Space models (SSM) with Transformer architecture for high quality and performance. You can [try Jamba on Hugging Face](https://huggingface.co/ai21labs/Jamba-v0.1).
- **Inquiry about Model Training Data**: A member requested more information regarding the training data for a certain model, seeking a direct link for detailed insights.
- **Model's Language Capabilities Discussed**: It was mentioned that the model in question is reportedly trained on English data, according to its model card or affiliated blog post, but noted that the specifics of what "English data" entails weren't clearly defined.
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.ai21.com/jamba">Introducing Jamba</a>: A groundbreaking SSM-Transformer Open Model</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks.</a>: Course to get into Large Language Models (LLMs) with roadmaps and Colab notebooks. - mlabonne/llm-course
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1222841255432814744)** (3 messages): 

- **Token Insertion Troubleshooting**: A member noted encountering unexpected token insertions like `<dummy00002>|im_end|>`, which may be attributed to **quantization** or the **engine** used. The issue was resolved by supplying an `added_tokens.json` file with specified tokens.

- **Translation Model Showdown**: There is interest in a comparison of translation outputs from models like **DiscoLM**, **Occiglot**, **Mixtral**, **GPT-4**, **DeepL**, and **Azure Translate**. The proposed idea involves translating the first 100 lines of a dataset like *Capybara* through each service.
  
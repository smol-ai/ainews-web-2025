---
id: 5daa80d8-1d76-461d-9b2e-facedc00251c
title: 'GPT4Turbo A/B Test: gpt-4-1106-preview'
date: '2024-01-26T22:07:42.174546Z'
original_slug: ainews-gpt4turbo-ab-test-gpt-4-1106-preview
description: >-
  OpenAI released GPT-4 Turbo, with discussions on model speed, prompt
  engineering, and issues with DALL-E text typos. The community explores large
  language models running on unconventional hardware, fine-tuning challenges,
  and external NLP tools for large document processing. Topics include AI model
  deployment, infrastructure, and societal impact.
companies:
  - openai
  - huggingface
models:
  - gpt-4-turbo
  - gpt-4
  - gpt-3.5
  - dall-e
topics:
  - large-language-models
  - prompt-engineering
  - model-fine-tuning
  - nlp-tools
  - multimodal
  - ai-infrastructure
  - ai-societal-impact
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 1/25/2024. We checked **20** guilds, **297** channels, and **5898** messages for you. Estimated reading time saved (at 200wpm): **557 minutes**.

OpenAI released a new [GPT4 Turbo version](https://openai.com/blog/new-embedding-models-and-api-updates) yesterday ([our notes here](https://twitter.com/swyx/status/1750620187114787243)). We're using this opportunity to conduct a natural experiment for summarization. This version is generated with the "old" GPT4T from Nov 2023 (Dev Day), stay tuned for the next email with the 2024 Jan 25th version for comparison and commentary.


--

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Troubleshooting Model Loading Error**: User `@sco7116` experienced an error when attempting to load *OpenHermes-2.5-Mistral-7B-4.0bpw* from Huggingface and was advised by `@itsme9316` to ensure they're using *exllamav2* for correct version compatibility. The error message was "Could not locate pytorch_model-00001-of-00002.bin."

- **RHEL Debate in ML Contexts**: There was an active discussion on the pros and cons of using Red Hat Enterprise Linux (RHEL) in machine learning settings. User `@.dontriskit` shared perspectives on infrastructure preferences and challenges encountered with RHEL.

- **Highlighting Dataset Generation for GPT Flaw Understanding**: `@kaltcit` proposes generating a dataset aimed at fingerprinting common GPT shortcomings like looping and topic drift, positing that it could be pivotal in understanding and addressing these issues systematically.

- **LLMs Running on Consoles**: The guild was intrigued by demonstrations of large language models like Llama and Mistral running on unconventional hardware such as the Nintendo Switch, showing the novelty and potential wide applicability of these models in various platforms.

- **LangChain Fine-Tuning Stumbles**: New user `@nandavikas` is seeking assistance to apply LangChain for fine-tuning the Llama2 model to extract information from PDFs, having previously achieved the task using PyTorch, and is lacking relevant guidance from LangChain documentation.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT-4's Need for Speed**: Users reported **GPT-4** having speed inconsistencies, with the API being slower compared to web usage, especially during **peak times**. Dedicated capacity was mentioned as a potential solution for enterprise-scale consumers, yet problems persisted when API loads were high.

- **Api-discussions & Prompt-engineering Channel Crossover**: The challenge of *prompt engineering* with **GPT-3.5** involved efforts to chunk large text for grammar correction, highlighting token limits and memory constraints. The use of OpenAPI, Python scripting, and *Custom Actions* was advised, and **GPT-4 Turbo** was suggested as a superior alternative for processing large documents.

- **DALL-E's Typo Trouble**: Users noted that **DALL-E** tends to include misspellings in its text generation within images, with community discussions suggesting shorter text inputs to possibly mitigate the issue. A [community link](https://community.openai.com/t/does-anyone-experience-issues-with-dall-e3-generating-typos-in-text-within-images/472966) provided further insights into this ongoing problem.

- **NLP Tools Touted as Text Transformers**: With GPT models struggling with large document processing, users like `@doublefelix` looked towards external NLP tools, such as *semantic-text-splitter* on **PyPI**, for a potential remedy. The conversation underscored the importance of maintaining historical context across API calls and the possibility of leveraging **ChatGPT Plus** for cost-effective solutions.

- **Conceptualizing Collaboration and Compliance**: Queries about deploying **GPT-4 Vision** on **Azure** and team account collaboration were met with concerns about account issues linked to unusual activity, possibly due to VPN/proxy usage or flagged prompts. There was also mention of problems saving/updating GPT bots, specifically related to policies on mimicking living artists.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Extended Context Capabilities in Focus**: The best current method to extend context capabilities is being discussed, with finetuning suggested as a viable solution. **Mistral instruct v0.2** disabled sliding windows to scale `rope_theta`, and configurations for **MistralLite** and main branch models are aligning in context window management. An impressive feat by **LLaMA-2-7B-Chat** extended its context window to 16,384 with minimal samples and training steps, while **SelfExtend** offers a non-finetuning alternative for context extension. 

- **AI's Societal Impact and Technological Puzzles**: Technology's role in societal polarization is being considered, with the observed fluctuating activity on Twitter prompting theories of AI slowdown. A resource for quantizing LLMs to GGUF format using a shared notebook is shared, aiding in the process of converting models.

- **Everyone Coder and Hermes Mixtral Benchmarks Sought**: A *human eval* benchmark has been requested for the quantized **Everyone Coder 33B**, using the **GGUF** format, available via [TheBloke on Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF), supported by [Massed Compute](https://massedcompute.com/). There's interest in seeing benchmarks for a combined Hermes and Mistral model, humorously referred to as **Hermes Mixtral**.

- **Launch of OpenAI Embedding Models & Synthetic Data Innovations**: OpenAI has released a new generation of embedding models, notable for data privacy and cost reductions, detailed in their [announcement post](https://openai.com/blog/new-embedding-models-and-api-updates). Additionally, the **Genie** method for generating high-quality content-grounded data suggests potential advancements in Long-Form Question-Answering and summarization, as documented in the [arXiv paper](https://arxiv.org/abs/2401.14367).

- **Enhancing AI Operations with GPU Acceleration**: Machine learning computations are being executed using WebGL for GPU acceleration, with discussions involving systems for real-time word embedding adjustments. Model training requires high-end motherboards for multi-GPU setups, with the **Mixtral instruct** outperforming several fine-tuned models and prototypes of new GPU-enhanced models making headway.

- **LLMs in the Limelight for Code and Cybersecurity**: For fine-tuning **CodeLlama 34B**, 4xA6000 GPUs might only suffice for a *qlora*, not a full fine-tune. **T5** faces fine-tuning challenges with stability issues, while LLMs like **WhiteRabbitNeo-13B** and **WhiteRabbitNeo-33B-v1** are recommended for offensive cyber tasks. The [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html) is a resource for evaluating AI coders, and there's dialogue on hyperparameter essentials for fine-tuning with **Llama Factory**.

- **Project Obsidian Scripts and Refactors**: A Python script for the **3b Obsidian** model is sought for remote execution, and efforts are underway to refactor code for compatibility with the latest **llava repo**.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Base Over Instruct for Norwegian Fine-tuning**: `@le_mess` advised using the **base model** instead of the Instruct variant for fine-tuning in languages with scarce training data, informed by `@henriklied`'s experience with a Norwegian dataset.

- **Epoch Caution to Avoid Overfitting**: To prevent overfitting during fine-tuning, `@le_mess` recommended stopping at **4 epochs**, especially in light of `@henriklied`'s observation of a constant eval loss after 10 epochs.

- **Block Characters Potential in Training**: Unique block characters like `▀` and `▄` were debated by `@suikamelon` and `@c.gato` regarding their efficacy and tokenization in training models like Mistral over ChatML tags, with only a small dataset.

- **Quantization and Config Optimizer Discussions**: Amidst quantization talks, `@dangfutures` preferred AWQ while `@bjoernp` and `@noobmaster29` discussed the importance of using the config optimizer over the default deepspeed optimizer, referenced to a [deepspeed config PR #1208](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208).

- **Grant Celebration and Dataset Introduction**: `@gahdnah` retracted a previous concern upon observing **active development**, and there was a celebratory note for a grant received. Meanwhile, `@dangfutures` highlighted a new dataset on Hugging Face for the Snorkel model, based on Mistral improvements.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LM Studio in Educational Settings**: `@josemanu72` looked for ways to run LM Studio as a server for connecting from students' desktops, with a solution involving a [frontend UI](https://github.com/JShollaj/Awesome-LLM-Web-UI) suggested by `@fabguy`.
  
- **Hardware Headaches and Help**: Discussions about utilizing GPUs ranged from `@xmiruso` solving GPU utilization issues by reloading models to `@gitanos` seeking advice on GPU value, with a recommendation to choose a used **Nvidia 3090** over the **4060ti 16GB RAM**. Meanwhile, `@mudf00t` reported VRAM detection problems on Nobara with an **RTX 3090**, with no immediate solution.

- **Open Source Model Musings**: `@vbwyrde` discussed the release of **"Intern" (InternLM)** and its supposed 200k context window and function calling capabilities. There were also strategic discussions around Meta releasing open-source models such as **Llama2** and using models like **Solar-10b for function calling**.

- **Troubleshooting and Tips**: Various users reported issues from a **bug switching MoE models** to errors in recent LM Studio updates leading to models failing to load, such as `@pdg` needing to downgrade to **version 0.2.10** to address the issue. A **broken link for empty strings** in Autogen was reported by `@sunglasses.emoji`, and suggestive solutions for better model performance were shared.

- **Software Snafus**: Conversations highlighted struggles with frameworks for open models and eccentric behaviors exhibited by models such as **Mistral** hallucinating nonexistent directories. There was an amusing note about **OpenAI's models** not recalling the current API, affecting context during training.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

**GPU Rental to Mistral Integration**: GPU rental options including **runpod, vast, and lambda** were discussed, with **Kaggle** also mentioned as offering free access up to 30 hours weekly. **Mistral 7B** use cases and integration challenges were shared, seeking insights for effective implementations, referencing **[Hugging Face's Mistral 7B model](https://huggingface.co/intfloat/e5-mistral-7b-instruct)**.

**Memory Matters in Model Finetuning**: Discourse around **Mixtral's** large memory appetite for inference highlighted that 26GB is required across four T4 GPUs, with actual usage potentially higher than expected. Efficiency debates compared **exllamav2** and **bnb 4 bit** for quantization, with a nod to [exllamav2 GitHub](https://github.com/turboderp/exllamav2) for running LLMs efficiently.

**Evaluating LLMs Beyond Traditional Metrics**: Emphasis was placed on the inadequacy of **BLEU and ROUGE** metrics for LLMs, suggesting **elo rankings** ([arena.lmsys.org](https://arena.lmsys.org/)) and benchmarks like **MMLU** and **Alpaca eval** for better performance measurements. The introduction of a normalized Alpaca eval market version was mentioned without further details.

**Creative Showcases and Random RAG Tips**: A tool named *SoContextual.com* that integrates AI for browser searches including DOM references was showcased, working with **MistralAI** and spotted on [Hacker News](https://news.ycombinator.com/item?id=39128480). Meanwhile, **prompt optimization** for RAG applications was touched upon, recommending **DSPy** and sharing a [prompting guide](https://www.promptingguide.ai/models/mistral-7b).

**Platform Puzzles and API Anomalies**: A billing page bug causing the monthly limit to reset to €150 was reported, while **API bugs** concerning the 'max_tokens' parameter and early stopping issues were discussed, including a posted [GitHub issue](https://github.com/mistralai/mistral-src/issues/122). Hosting queries affirmed Mistral's API is located on Azure in Sweden.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Quest for SAM Model Fine-tuning Code**: `@the_alt_man` sought a codebase for fine-tuning **Meta's SAM model**, only to learn that it wasn't included in the original release. **`AutoGluon`** was discussed as a tool integrating with **Lightning** but limitations on GPU usage were noted.

- **Exploration of Federated Learning**: A collaborative effort was made to discuss the practicality of federated learning, especially multinode training without **infiniband** and specifics of model merging. A reference to the DiLoCo study was provided, which can be found on [arXiv](https://arxiv.org/abs/2311.08105).

- **Deep Dive into Proxy Tuning LLMs**: `@digthatdata` introduced a method for tuning **Large Language Models (LLMs)** via proxies, potentially streamlining the tuning process. This alternative approach is detailed in a [paper available on arXiv](https://arxiv.org/abs/2401.08565).

- **GPT-NeoX QLoRA Tuning Troubles**: `@kenakafrosty` asked for assistance with tuning **GPT-NeoX 20B** using **QLoRA**, facing a non-descending loss issue. It was clarified that NeoX does not currently support QLoRA, and help was redirected to GitHub for solutions with `trl`, `transformers`, and `peft`.

- **Testing Woes and Collaboration in GPT-NeoX Development**: Updates to **Python, PyTorch, and CUDA** brought about issues with running tests, setting off a discussion on the necessity of a functional testing suite for the project. Efforts to fix testing processes, track forked test failures, and provide compute access to project collaborators are active, as exemplified by [this GitHub issue](https://github.com/EleutherAI/gpt-neox/issues/1132).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Dataset Disappearance Disrupts Developers**: The **LAION dataset**, specific to **laion2b-en aesthetics scores**, is currently inaccessible as the dataset author has requested a temporary disablement. Engineers are advised to stay tuned to official announcements for updates on dataset access.

- **Voice Chat on the Verge**: A new **voice chat interface** that integrates Whisper and WhisperSpeech with an LLM has been demoed, promising reduced latency and more natural conversation. Collaborators are being sought to further improve the system; details can be found on the [Hacker News announcement](https://news.ycombinator.com/item?id=39130140).

- **Image Is Everything**: Methods for **image captioning** using AI have sparked discussion around the importance of clear prompts to minimize hallucinations, centering on accurate descriptions of visible content only.

- **Competition Calls for Creative Coders**: The AI Tech Sprint invites developers to contribute to a project focused on clinical encounter notes with a chance to win prize money. Interested parties should [register their interest](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data).

- **AI's Big Expense Report**: Discussions have taken place around the high costs involved in training AI models like **Stable Diffusion**, while acknowledging expected cost reductions due to technological advancements over time.

- **Google's Missed SaaS Boat?**: Google's heavy reliance on advertising revenue was critiqued, and an alternative path focusing on **SaaS models**, like that by OpenAI, was suggested to potentially lead to deeper financial impacts.

- **Byte-Level Transformers Nearly There**: Interest is rising around byte-level transformers with an expectation of significant progress soon, demonstrated by recent related [arXiv research](https://arxiv.org/pdf/2401.13660.pdf).

- **Restoration and Recognition Reimagined**: Technological strides in **text-to-image diffusion** and **identity preservation** were highlighted through [text-to-image diffusion tech](https://github.com/YangLing0818/RPG-DiffusionMaster) and [ID-preserving generation systems](https://github.com/InstantID/InstantID), promising new capabilities for AI-generated imagery.

- **Scaling the Summit with SUPIR**: A paper on **SUPIR**, an image restoration method leveraging generative priors and model scaling, gained attention due to its innovative approach and mention among the top papers on Hacker News, detailed in the [arXiv submission](https://arxiv.org/abs/2401.13627).



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Perplexity Pro’s Power Unveiled**: Enthusiasts discussed [Perplexity Pro features](https://blog.perplexity.ai/faq/what-is-perplexity-pro), such as unlimited Copilot queries and access to AI models like GPT-4 and Claude 2.1, shedding light on enhanced capabilities beyond the standard offering.

- **Privacy Policies Under Scrutiny**: Concerns about **Perplexity's data retention policies** led to clarifications that deleted threads are purged after 30 days; however, ambiguities in privacy policies about account and search data provoked calls for clearer communication for user reassurance.

- **API Queries and Billing Grievances**: Technical discussions unveiled discrepancies between Perplexity AI’s website and API, with the latter producing inferior code outputs, and users, including `@aiagileguy`, confronted billing issues, such as double charges, without quick resolutions.

- **Perplexity in Tutorials and Education**: Users shared success stories and practical uses of Perplexity AI, like smoothing out the transition from Excel to Smartsheet and aiding in explaining complex astronomy concepts in educational settings.
  
- **A Vote for Perplexity over Giants**: A YouTube video titled "[I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)" depicted Perplexity AI as a superior choice over mainstream options such as Google and ChatGPT for tasks like content creation.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **HuggingFace Introduces Social Post Explorers**: `@lunarflu` invited the **HuggingFace community** to join for [early access to the "Posts" feature](https://huggingface.co/social-post-explorers) which aims to provide a focused space for AI & ML discussions away from the noise of platforms like Twitter or LinkedIn.

- **Pretraining Predicaments and Cost-Effective Tactics**: GPU-intensive model pretraining, as in the [Llama-2-7B model](https://huggingface.co/meta-llama/Llama-2-7b), has the community contemplating less resource-heavy alternatives like fine-tuning or employing [LoRA/QLoRA adapters](https://huggingface.co/docs/peft/conceptual_guides/lora).

- **Desperate for Data Set Evaluation Standards**: `@rosebei3ngan3g` highlighted the lack of frameworks to evaluate data sets for large language models, which contrasts sharply with numerous model evaluation frameworks.

- **Insightful Innovations in Dataset Analysis and Demo Displays**: A [GitHub project on dataset cards analysis](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis) and a multi-language text-to-speech HuggingFace demo, [WhisperSpeech](https://huggingface.co/spaces/Tonic/whisperspeech), showcase the dynamic range of work within the HuggingFace ecosystem.

- **Recognition for Revolutionary Models and Metrics**: Google's **Lumiere** model, combining a Space-Time UNET for fluid video generation capabilities, is turning heads in the community alongside interest in a new release of `gradio 4.16`, which includes features such as support for **Polars Dataframe** and a new Gallery component, detailed in the [changelog](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **LlamaIndex Enthralls with LLM Webinar**: A **webinar** on the **LLMCompiler** featuring a discussion on **parallel function calls in agents** was announced by `@jerryjliu0`, with resources such as **LlamaPack** and a **Notebook** being available. The related paper can be found [here](https://arxiv.org/pdf/2312.04511.pdf), and the webinar details [here](https://lu.ma/lf9iroox).

- **Stack of Innovations Unveiled**: The **Slack Bot Tutorial** by `@seldo` instructs on integrating organizational learning into bots; **Zilliz Cloud Pipeline** is newly linked with LlamaIndex, as covered in a [guest blog post](https://t.co/luDjSgiokt). Version 0.9.38 of LlamaIndex now supports **OpenAI's latest embedding models** with further details in their [release notes](https://t.co/kyIoTUaeuD), while TypeScript users get **LlamaIndex.TS** in a 0.1.0 version supporting the same.

- **Discourse in #general Heats up Around Retrieval and Customization**: LlamaIndex lacks its own **LLM for TextGenerationInference** yet is compatible with Langchain's. **Complex retrieval scenarios** and incorporation of OpenAI's updated **embedding models** were also debated. In response to a query about extracting answers without sufficient context, a **link to modifying default prompts** was shared: [Usage Pattern](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts).

- **Zep's Multifaceted Chat Features Scrutinized**: **Zep’s** ability to remember conversations and perform entity extraction sparked interest with `@yoursaviorjesus` sharing [Zep Documentation](https://docs.getzep.com/). Clarifying LlamaIndex’s functionality, `@cheesyfishes` described it as akin to Amazon Kendra with adaptability across **any vector store or language model**.

- **Innovations in Knowledge Graphs Shared**: `@chiajy` demonstrated a self-learning knowledge graph with **recursive retrieval and multi-hop reasoning**, through a **Harry Potter book demo**. For a detailed exploration of this work, consult [Harry Potter and the Self-Learning Knowledge Graph](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning).



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **LLM Paper Club maintains no-recording policy**: The Latent Space's LLM Paper Club sessions **will not be recorded** to encourage open sharing. No option for replays will be provided.
- **Dreams Meet AI in Morpheus-1**: A tweet launched the news about **Morpheus-1**, a multi-modal generative ultrasonic transformer designed for lucid dream induction, set for beta release in Spring 2024. The excitement revolves around its novel approach.
- **GPT-4 Turbo and New Embeddings Roll Out**: OpenAI has released an updated **GPT-4 Turbo** model and new embedding models. Detailed notes and announcements were shared, highlighting improvements and potential impacts on AI applications.
- **Martian's LLM Benchmarks Go Live**: Martian has debuted a Model Router at **[Martian's LLM Leaderboard](https://leaderboard.withmartian.com/)** to evaluate different LLM inference products, backed by open-source documentation and tools.
- **Asia Joins the Fold in LLM Paper Club**: The LLM Paper Club has expanded to Asia, offering discussions on seminal papers like "Attention Is All You Need". The club is soliciting suggestions for future papers and feedback to enhance the **beta** experience.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Mixtral and Merging Models Under the Lens**: *mergekit*'s author provided insights in a [GitHub comment](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289), influencing **DiscoResearch mixtral training** approaches. Importance is placed on the correct application of *auxiliary loss* for Mixture of Experts (MoE) training.

- **Rethinking Data Filtering and Model Training Approaches**: A new paper challenges the efficacy of quality filtering for pretraining data, pointing towards data selection aligned with model performance on target tasks, mentioned [here](https://arxiv.org/abs/2401.12926). Conversations revolve around adopting new training methods like Direct Preference Optimization (DPO) and Key Term Optimization (KTO), with insights on using the DPO Trainer detailed in [Hugging Face's TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer).

- **Advancements in Embedding Development and Usage**: The German Jina embeddings model is soon to be released, promising enhancements for ranking applications. OpenAI's new embedding models, featuring improved multilinguality, signify a leap forward, as detailed [here](https://openai.com/blog/new-embedding-models-and-api-updates).

- **Translation and Language Model Fine-tuning Feats**: **DiscoLM German 7B v1** has been successfully finetuned with a custom dataset intended to translate Middle High German to modern German. This fine-tuning process is eagerly anticipating versions based on **Mixtral-Instruct**.

- **Impending Efficiencies in Embedding Technologies**: Upcoming embeddings are set to outrun OpenAI's on the MIRACL benchmark, delivering a **12x saving on vector database costs** with only 256 dimensions, as teased by `@Nils_Reimers` in [this tweet](https://x.com/nils_reimers/status/1750631888094380268?s=46&t=-TRJUfVdW8KeDqen1HJU1Q).



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **OpenAI Unleashes Embedding Innovation**: [OpenAI's latest announcement](https://openai.com/blog/new-embedding-models-and-api-updates) details the launch of **GPT-4 Turbo** and new **embedding models**, tools for better API control, and imminent lower pricing for **GPT-3.5 Turbo**. Engineers are spotlighting **shortened embeddings** as a leap in efficiency, looking forward to integrating these into their systems.
  
- **Updated OpenAI APIs Ease Developer Journey**: OpenAI's commitment to enhancing the **API experience** includes new moderation models and **API usage management** tools aimed at refining developers' oversight. Developers now have an [updated documentation guide](https://platform.openai.com/docs/guides/embeddings/) to navigate the latest models and features.
  
- **Ease Over Open Source**: The debate between OpenAI and open-source models unfolds, with professionals like `@res6969` pointing out the speed of feature implementation with OpenAI, while others advocate for the customizability of open-source alternatives.
  
- **Convenience Can Trump Customization**: Despite the availability of open-source models for personal fine-tuning, members like `@potrock` stress the straightforward, out-of-the-box convenience offered by OpenAI's embedding models.
  
- **Striking a Cost-Effectiveness Balance**: The economic conversation shifts to **cost benefits** of OpenAI's new larger embedding models, as discussed by `@shacrw` and `@michelcarroll`, presenting a balancing act between **storage savings** and **API costs** in the wake of these updates.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **Welcome Aboard to LangChain Universe**: `@quarknova`, currently interning at INRIA, showcased their interest in applying LangChain in their projects, prompting them to consider the advantages of the GitHub version against its commercial counterpart.

- **Custom AI Personalities Now Tailored**: `@jstansbe` examined the possibility of creating customized AI entities like an "Elon Musk AI," and `@ksolo__` contributed a resource, introducing the concept of finetuning and sharing a [deep learning course link](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/).

- **LangChain Applause for Chatbot Creation**: `@johnnucleus` recognized the LangChain community for its effective assistance in rapidly developing a chatbot integrated with web search functionalities using LangChain and Streamlit.

- **LLMs Turn Data Synthesizers**: Discussion involved using Large Language Models (LLMs) for synthetic data generation to feed traditional ML models, with special mentions of employing LLMs for RAG generation to create SQL queries based on context and schema.

- **Manipulating PARQUET in LangChain**: `@benjaminbascary` and `@johnny2x2` exchanged insights on handling PARQUET files within LangChain, with code examples via `pandas` and the `DataFrameLoader` feature.

- **Dive into LangServe's Capabilities**: `@veryboldbagel` shared examples and resources on creating custom tools and agents using LangServe and LCEL, stressing the utility of [LangGraph](https://python.langchain.com/docs/langgraph#agentexecutor) to construct agents with enhanced expressive power.

- **The Mysterious Case of the Missing Stream Response**: `@hiranga.g` faced challenges with stream responses while experimenting with LangServe's [agent_with_history](https://github.com/langchain-ai/langserve/blob/main/examples/agent_with_history/server.py), highlighting a potential bug when incorporating Agents via LangServe with `chain.streamLog()`.

- **SQL Chain’s Battle with Database Giants**: `@johnny2x2` recounted experiences of SQL Chain's difficulty in handling large databases and found that crafting **curated views** with descriptive names within the databases amplified performance.

- **Improved SQL Query Management Through Refinement**: `@johnny2x2` described the shift from utilizing a local AI to relying on OpenAI for SQL query processing in order to maintain data privacy, leading to a more efficient querying process within LangChain.

- **Task Processing Elevated with Chain Workflow**: Introducing a new methodology, `@johnny2x2` describes the transition to using each SQL query as an individual tool in their **task processing chain**, leading to significant improvements in workflow management.

Please note that any direct references to usernames were included as they were considered contextually relevant based on the information provided.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Heads-Up for an LLM Library Update**: @SimonW announced an upcoming update to the **openai library** within the LLM project, with detailed information for testers in a [GitHub comment](https://github.com/simonw/llm/issues/325#issuecomment-1911533536).

- **LLM Strides Towards 0.13**: The up-and-coming **0.13 Milestone** of the LLM release, aimed at enhancing command-line accessibility to large language models, is documented on the [GitHub milestone page](https://github.com/simonw/llm/milestone/8).

- **Call for Coders to Tackle Readline Bug**: There's an open call for assistance regarding a readline issue in LLM where arrow keys display ANSI codes instead of navigating the cursor, as detailed in this [GitHub issue](https://github.com/simonw/llm/issues/376).



# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199991901110145084) (1212 messages🔥🔥🔥): 

- **Trying Out Hermes 2.5**: User `@sco7116` asks for help testing a model named *OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2* from Huggingface, but struggles with the error "Could not locate pytorch_model-00001-of-00002.bin." `@itsme9316` suggests they are not loading it with *exllamav2* and should switch to the correct version.
- **Pros and Cons of Using RHEL for ML**: `@kquant` reports successfully running Linux and moves to Ubuntu for synthetic dataset generation. The discussion includes various opinions on using Red Hat Enterprise Linux for machine learning, with `@.dontriskit` sharing insights on preferred infrastructure and challenges with RHEL in an ML development context.
- **DPO Scripts and Techniques Shared**: `@kaltcit` offers information on their approach to dataset pruning optimization (DPO) for models, focusing on generating a dataset that captures common GPT failure patterns such as looping and topic drifting. They suggest such a dataset could be the most comprehensive collection of GPT flaws.
- **Fascination with LLM Predictions**: Users discuss various finetunes and merges with models like Lama and ChatGPT. `@itsme9316` finds that a 7B 20 million token discord message finetune surpasses several larger merges in quality and even suggests they might attempt a 500 million token finetune.
- **AI Running on Nintendo Switch**: Users shared videos and comments on running LLMs on surprising hardware. `@kquant` expresses amazement at the possibility of running models like Llama and Mistral on a Nintendo Switch, while `@kalomaze` contributes to the theme with media showing LLMs in action on unconventional platforms.

**Links mentioned**:

- [DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence](https://arxiv.org/abs/2401.14196): The rapid development of large language models has revolutionized code intelligence in software development. However, the predominance of closed-source models has restricted extensive research and dev...
- [God helmet - Wikipedia](https://en.wikipedia.org/wiki/God_helmet): no description found
- [Marvelous Dc Gotham GIF - Marvelous Dc Gotham Gotham Tv - Discover &amp; Share GIFs](https://tenor.com/view/marvelous-dc-gotham-gotham-tv-hugo-strange-dr-strange-gif-17601265): Click to view the GIF
- [Tails - Home](https://tails.net/): no description found
- [LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/OpenHermes-2.5-Mistral-7B-4.0bpw-h6-exl2): no description found
- [no title found](https://neurosity.co/): no description found
- [import sysimport osfrom tqdm import tqdmsys.path.append(os.path.dirname(os - Pastebin.com](https://pastebin.com/wgD8Q5Qs): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [How to Install NVIDIA Drivers on Rocky Linux 9 or 8 - LinuxCapable](https://www.linuxcapable.com/how-to-install-nvidia-drivers-on-rocky-linux/): Learn to install NVIDIA Drivers on Rocky Linux 9 or 8 using the command line terminal and Nvidia Cuda REPO for the latest version.
- [How I Won Singapore’s GPT-4 Prompt Engineering Competition](https://towardsdatascience.com/how-i-won-singapores-gpt-4-prompt-engineering-competition-34c195a93d41): A deep dive into the strategies I learned for harnessing the power of Large Language Models (LLMs)
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind#screenshots): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [GitHub - facebookresearch/audio2photoreal: Code and dataset for photorealistic Codec Avatars driven from audio](https://github.com/facebookresearch/audio2photoreal): Code and dataset for photorealistic Codec Avatars driven from audio - GitHub - facebookresearch/audio2photoreal: Code and dataset for photorealistic Codec Avatars driven from audio
- [GitHub - facebookresearch/Qinco: Residual Quantization with Implicit Neural Codebooks](https://github.com/facebookresearch/Qinco?tab=readme-ov-file): Residual Quantization with Implicit Neural Codebooks - GitHub - facebookresearch/Qinco: Residual Quantization with Implicit Neural Codebooks
- [Stanford Hypnosis Integrated with Functional Connectivity-targeted Transcranial Stimulation (SHIFT): a preregistered randomized controlled trial - Nature Mental Health](https://www.nature.com/articles/s44220-023-00184-z): Investigators present findings from a double-blind randomized controlled trial of personalized stimulation of the left dorsolateral prefrontal cortex using transcranial magnetic stimulation to increas...

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1200016786880471090) (74 messages🔥🔥): 

- **ExLLamaV2 Loader Support Question**: `@ks_c` initially questioned whether the exllamav2 loader in oobabooga supported `min_p`, despite not being an hf loader, but then confirmed that it was merged into exllama.
- **CPU Mode Confusion Cleared**: `@neriss` informed `@keyboardking` that **exl2** cannot run on CPU after `@keyboardking` asked about its utility compared to **gguf** in CPU-only mode.
- **Model Configuration Comparison**: `@dreamgen` inquired about differences in `rope_theta` and `sliding_window` configurations among various models, sharing links to the [bagel](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json), [Mistral instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json), and [dolphin](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json) config files. `@jondurbin` replied, explaining inheritance from the base model and the potential for future changes.
- **Role-Play Models Discussion**: `@shadowplague` requested recommendations for models adept at generating content for *disrespectful, abusive, racist, and dirty-talking scenarios* for role-play, with `@c.gato` and `@kalomaze` pointing out that existing models can be prompted for such content and suggesting **Kunoichi DPO v2** or **Fett-uccine** for role-play purposes.
- **7B Parameter Models for RP**: Discussing the best role-play models with 7B parameters, members provided various suggestions such as **HamSter-0.2**, **Kunoichi DPO v2**, and **Fett-uccine**, while touching on both 6 and 8-bit quantization depending on VRAM capacity.

**Links mentioned**:

- [Epiculous/Fett-uccine-7B-GGUF at main](https://huggingface.co/Epiculous/Fett-uccine-7B-GGUF/tree/main): no description found
- [Release Quadratic Sampling Test Build (koboldcpp) · kalomaze/koboldcpp](https://github.com/kalomaze/koboldcpp/releases/tag/quad-sampling-v1): Replacement for the last idea (Smooth Sampling) with a different scaling mechanism.  The idea behind it is to simplify sampling as much as possible and remove as many extra variables as is reasonab...
- [config.json · mistralai/Mistral-7B-v0.1 at main](https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json): no description found
- [config.json · jondurbin/bagel-dpo-7b-v0.1 at main](https://huggingface.co/jondurbin/bagel-dpo-7b-v0.1/blob/main/config.json): no description found
- [config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json): no description found
- [Kronk Its All Coming Together GIF - Kronk Its All Coming Together - Discover &amp; Share GIFs](https://tenor.com/view/kronk-its-all-coming-together-gif-15058130): Click to view the GIF
- [config.json · cognitivecomputations/dolphin-2.6-mistral-7b at main](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b/blob/main/config.json): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1200022344538800189) (20 messages🔥): 

- **Chatbots Learning Names and Styles**: `@lordofthegoons` is looking to train a chatbot to have a consistent conversation style and remember its own name, much like the Samantha model. `@amogus2432` suggests using 10 to 100 examples for style transfer, but `@dirtytigerx` recommends more, citing that the Samantha model used around 6,000 multiturn conversations.
- **The Quest for Financial Advisor Chatbot**: `@VR` is seeking advice on creating a financial investment advisor chatbot that can run on a 24GB GPU and utilize RAG for up-to-date stock price information, trends, and expert analysis. They are considering prompt tuning versus fine-tuning on financial documents.
- **Building a Unique Chatbot Persona**: `@lordofthegoons` aims to create a chatbot with a specific persona by producing a custom dataset. They note difficulties in achieving variation when using ChatGPT to generate examples and are considering manually creating the dataset due to challenges with rate limiting.
- **Financial Constraints Influence Dataset Building**: `@dirtytigerx` points out the high cost associated with using the GPT-4 API for dataset generation and the inefficiency of waiting out ChatGPT's rate limits. They suggest experimenting with local large language models (LLMs) as a more cost-effective option.
- **Rate Limits Prompt Creative Solutions**: `@lordofthegoons` expresses intent to manually build a chatbot dataset using ChatGPT while dealing with rate limitations. `@dirtytigerx` further advises that utilizing services like runpod to run large LLMs locally might be cheaper and more efficient than facing rate limits with OpenAI's API.
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199997735223447642) (19 messages🔥): 

- **Optimizing Weight in Model Merging**: `@sanjiwatsuki` proposed a hypothesis that setting the model weight **slightly above 1.0** could be optimal due to the **TIES resolution process** potentially causing some effective weight to drop out.
- **Exploring Negative Weights in Script**: `@kquant` inquired whether **negative numbers** could break the merging script. `@sanjiwatsuki` expressed uncertainty but speculated that the code might handle negative weights without issues.
- **Selective Model Assimilation**: `@kquant` discussed the possibility of selectively merging models to assimilate desired characteristics, mentioning methods like **DARE** and **SLERP** that could potentially combine two models with high evaluation scores on different benchmarks.
- **SLERP and Overfit Models' Performance**: `@kquant` noted an unexpected result where two overfit models were merged using **SLERP** and managed to maintain their test positions, which raised questions about the impacts of overfitting in model merging contexts.
- **Merging Methods Clarification Needed**: `@kquant` mentioned a need to better understand how **DARE** and **SLERP** differ in the context of model merging, expressing a desire to conduct more research and testing.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1200086950397366292) (1 messages): 

- **New User Seeking LangChain Guidance**: User `@nandavikas` expressed difficulty in replicating a fine-tuning process with **LangChain** that they previously accomplished using **PyTorch**. They are seeking assistance to fine-tune **Llama2** for extracting specific information from PDFs and couldn't find relevant documentation in **LangChain** docs.
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1200055671194927236) (35 messages🔥): 

- **Speed Concerns with GPT-4 Document Analysis**: `@romansh2302` highlighted a significant speed discrepancy between GPT-4 on the web and the `gpt-4-1106-preview` model via the API, with the latter being slower. `@lugui` responded that **varying API speeds** can occur during peak usage and mentioned the possibility of **requesting dedicated capacity**, which tends to cater to company-scale use.
  
- **Looking for GPT-4 Performance Solutions**: While seeking a speed remedy, `@romansh2302` was informed by `@lugui` that the speed differential is tied to **peak times** and API loads, a situation not easily fixed. `@og_tort` suggested considering **GPT-3.5** as an alternative; however, `@romansh2302` found it less effective for document analysis.

- **Users Faced with Account Issues**: `@hellomyfriend1576` reported receiving a warning of **unusual activity from their system** when using GPT-4. Answers from community members like `@lugui` and `@og_tort` entertained the possibility of VPN or proxy usage and potentially **flagged prompts** as reasons for the issue.

- **Typos in DALL-E's Text Generation** Prompted a Query:  `@alwayspercipient` noted DALL-E frequently includes **misspellings** in its image creation. This issue has been discussed in the community, as pointed out by `@muyfashionista`, who also offered a [community link](https://community.openai.com/t/does-anyone-experience-issues-with-dall-e3-generating-typos-in-text-within-images/472966) on the subject and mentioned that using shorter text might reduce errors.

- **Confusion Over AI Services and Team Accounts**: Users like `@paras4887` and `@leithwhitley` posed questions about specific use cases, such as deploying GPT-4 Vision on **Azure** and issues about collaboration using a shared **paid team GPT account**. Solutions or clear guidance were not offered in the provided message chain.

**Links mentioned**:

[TuringsSolutions/PFAF750 · Datasets at Hugging Face](https://huggingface.co/datasets/TuringsSolutions/PFAF750): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1200050868888801362) (105 messages🔥🔥): 

- **Troubleshooting 'Always expand code output' Feature**: `@darthgustav.` clarified for `@angry_coder` that "Always expand code output" means wrapped code blocks will always be expanded for easy reading, after testing the feature themselves.
- **Unpacking Libraries in GPT**: `@bambooshoots` suggested uploading a library as a zip file with a `.py` file that unzips it and adds the /mnt/data/ folder to the system path, a method supported in the past. `@darthgustav.` expressed concern about potential security issues and halted testing of this environment expansion.
- **CustomGPT Edits Require New Conversations**: For `@elegante94`, `@darthgustav.` confirmed that to see the effects of a CustomGPT edit, a new conversation is required as ongoing conversations won't update with the new changes.
- **Image Attachments in GPT Prompt Instructions**: `@elegante94` queried about the effectiveness of attaching images to prompt instructions, and `@darthgustav.` responded that using concise language is better than attaching images because DALL-E will improvise creative elements.
- **Saving/Updating GPT Bots Issues**: `@rodney.leonardo` encountered errors when saving/updating a GPT bot and sought assistance. `@darthgustav.` suggested removing the knowledge, saving as private, then reattaching files one by one, noting a possible block due to mimicking a living artist which is not permitted.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **Engaging in Prompt Engineering**: `@darthgustav.` advised `@novumclassicum` on creating API configurations for custom workflows, including the use of OpenAPI for Python scripting and Custom Actions in GPT. They also discussed strategies for ensuring standardized output despite potential deviation.
- **Document Chunking Dilemma**: `@doublefelix` sought a way to chunk a large text into paragraphs for grammar correction via AI, while `@eskcanta` recommended a method using Python for smaller sections. Avi and Felix debated the best practice for this task, with Avi suggesting the use of semantic text splitting via AI.
- **AI-Powered Workflow Challenges**: `@doublefelix` tested various approaches to directing GPT-3.5 to add paragraph markings and address grammatical issues in a transcribed sermon, encountering issues with compliance and hallucinations due to token limits and memory holes in the AI context.
- **Exploring GPT-4 Turbo as a Solution**: `@darthgustav.` proposed utilizing GPT-4 Turbo's Python Tool to semantically analyze text and generate paragraphs, bypassing the chunking limitations `@doublefelix` faced with GPT-3.5. Darthgustav also highlighted issues with excessive token usage and lost context in large document processing.
- **Considering NLP Tools for Text Splitting**: Frustrated with the complexity of managing GPT-3.5's limitations, `@doublefelix` decided to explore third-party NLP tools like semantic-text-splitter for potentially automating the text chunking process for large documents while acknowledging the higher capabilities of GPT-4 Turbo for such tasks.

**Links mentioned**:

[How do you maintain historical context in repeat API calls?](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395): Each time I make a call to the API it starts off with no prior context, unlike the chat.openai.com scenario. Is there a way to maintain state of the model during a session?  response = openai.Completi...

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1200065926914117652) (558 messages🔥🔥🔥): 

- **Prompt Engineering Proves Challenging**: User `@doublefelix` has been working to prompt GPT 3.5 to process large amounts of text and add paragraph breaks. Despite various strategies, including breaking the task into smaller chunks, the AI has difficulty managing context and often ignores parts of the input text or hallucinates.

- **Finding the Right Approach**: The conversation highlighted the complexity of managing AI context and the notion of 'document retrieval curves'. `@darthgustav.` suggested that GPT-4, particularly with ChatGPT Plus, might be able to better manage the task due to its larger context window and ability to process attached files using retrieval-augmented generation (RAG).

- **API vs. ChatGPT for Cost-Effective Solutions**: `@doublefelix` is exploring options to keep costs minimal while automating the processing of transcribed sermons. `@darthgustav.` pointed out that using the ChatGPT interface with custom instructions could avoid token costs associated with the API method.

- **The Potential of Custom Prompts**: `@darthgustav.` noted the importance of structuring prompts with explicit instructions and "open variables" that encode instructions, which might allow for more nuanced control over the AI's output and help with the task of breaking text into sections.

- **Exploring Alternatives and Considering Next Steps**: The conversation indicated a fallback plan involving Custom GPTs and traditional NLP methods with a Python tool as alternatives. `@doublefelix` plans to research NLP packages, such as a semantic-text-splitter found on PyPI, to find a workable solution.

**Links mentioned**:

[How do you maintain historical context in repeat API calls?](https://community.openai.com/t/how-do-you-maintain-historical-context-in-repeat-api-calls/34395): Each time I make a call to the API it starts off with no prior context, unlike the chat.openai.com scenario. Is there a way to maintain state of the model during a session?  response = openai.Completi...

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1200095821480341595) (7 messages): 

- **Seeking the Best Context Extension Solution**: `@cryptossssun` inquired about the best current method to extend context capabilities. `@_cherrry` responded positively to a paper, suggesting finetuning is a viable solution.

- **Mistral Instruct Moves Away from Sliding Window**: `@dreamgen` discussed the implications of **Mistral instruct v0.2** disabling sliding windows in favor of scaling `rope_theta`, questioning the effectiveness of the sliding window approach. They shared a [configuration file](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json) showing these changes.

- **MistralLite Mimics Main Branch Configurations**: `@dreamgen` also noted that **amazon/MistralLite** follows the same configuration strategy as its main branch counterpart concerning context window management.

- **Remarkable Efficiency in LLaMA-2-7B-Chat Context Extension**: `@stellaathena` highlighted an impressive feat where the LLaMA-2-7B-Chat model's context window was extended to 16,384 with only 100 samples and 6 training steps.

- **SelfExtend as a Non-Finetuning Alternative**: In the discussion on context extension, `@leontello` mentioned **SelfExtend** as an intriguing option for those who prefer not to fine-tune their models.

**Links mentioned**:

[config.json · mistralai/Mistral-7B-Instruct-v0.2 at main](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json): no description found

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1200111343458599076) (5 messages): 

- **Technological Impact on Society's Extremes**: `@ldj` contemplates that technology might lead to further polarization, where the least capable individuals become more entrenched in degeneracy, while the most capable are propelled towards greater self-improvement.
- **Twitter's Fluctuating Activity Puzzle**: `@fullstack6209` observes a radical change in the frequency of new Twitter posts, questioning if anyone else noticed this shift from 2-3 posts every 10 minutes to about 70 posts in a minute.
- **Twitter's AI Slow Down Theory**: `@fullstack6209` brings up a suggestion that Twitter might have deliberately slowed down the AI to explain the changes in the frequency of posts observed.
- **Quantizing LLMs Made Easy**: `@pradeep1148` shares a [YouTube video](https://www.youtube.com/watch?v=wlPxEq_Mtkc) titled "AutoGGUF Quantize LLMs in GGUF format in one click," providing a resource for converting large language models to GGUF format using a shared notebook.

**Links mentioned**:

- [Cat Swimming GIF - Cat Swimming Poopsie - Discover &amp; Share GIFs](https://tenor.com/view/cat-swimming-poopsie-silly-gif-14546589990767279660): Click to view the GIF
- [AutoGGUF Quantize LLMs in GGUF format in one click.](https://www.youtube.com/watch?v=wlPxEq_Mtkc): Quantize any hf LLMS to GGUF format using the notebook provided by Maxim Labonne#llms #ml #ai #neuralnetworks #deeplearning #ggufhttps://colab.research.googl...

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1200251006743744512) (2 messages): 

- **Benchmark Request for Everyone Coder 33B**: `@benxh` requested a *human eval* benchmark for a quantized version of **Everyone Coder 33B**, which uses the new **GGUF** format introduced by llama.cpp. The model was made available by [TheBloke](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF) on Hugging Face, and the [quantisation was supported by Massed Compute](https://massedcompute.com/).
- **Call for Hermes Mixtral Evaluation**: User `@teknium` expressed a desire to see a benchmark on a combined Hermes and Mistral model, referring to it as **Hermes Mixtral**, and requested it with a hopeful 🙏 emoji.

**Links mentioned**:

[TheBloke/Everyone-Coder-33B-Base-GGUF · Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF): no description found

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1200148361370669126) (2 messages): 

- **New OpenAI Embedding Models Launch**: `@tsunemoto` shared [OpenAI's announcement](https://openai.com/blog/new-embedding-models-and-api-updates) unveiling a new generation of embedding models, GPT-4 Turbo, updated moderation models, and cost reductions for GPT-3.5 Turbo. **Data privacy** by default and improved API management tools were highlighted, as well as **lower pricing for new embedding models.**

- **Genie: A Method for High-Quality Synthetic Data**: `@metaldragon01` introduced a paper about **Genie**, a novel method for creating high-quality content-grounded data, detailed in the [published work on arXiv](https://arxiv.org/abs/2401.14367). Genie is claimed to produce data so refined that in human evaluations, it was found to be natural and high-quality, with implications for improving Long-Form Question-Answering and summarization models.

**Links mentioned**:

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367): The lack of high-quality data for content-grounded generation tasks has been identified as a major obstacle to advancing these tasks. To address this gap, we propose Genie, a novel method for automati...
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1200033366272122910) (362 messages🔥🔥): 

- **Introducing GPU-Accelerated AI**: User `@n8programs` discussed performing **machine learning computations** using WebGL, typically used for texture processing in video games, by packing data into textures because WebGL doesn't support raw buffers (as `@everyoneisgross` and `@n8programs` exchanged ideas). This method enables GPU acceleration, as practiced by TensorFlow.js, despite current limitations such as a maximum vector size of 16,000 elements.
- **Wise Words on Model Training**: `@intervitens` advises that a server or high-end desktop (HEDT) motherboard is necessary for effective multi-GPU setups due to the requirement of ample PCI-e lanes, suggesting second-hand Gen2 EPYC for a balance of performance and economy. The discussion includes options like mining-style chassis, two-slot spacing motherboards for 4x 2-slot graphics cards, and bespoke water cooling setups.
- **Real-Time Word Embedding Adjustments**: User `@everyoneisgross` described a system for quick on-the-fly adjustments to word embeddings in a word2vec model, allowing real-time feedback by increasing or decreasing weights based on user input. This process is functionally quick on small corpuses, making it practical for expansion or refinement based on fresh data fetched by an LLM, which in this case was **Mistral instruct**.
- **Mixtral Instruct Surprisingly Strong**: `@gabriel_syme` asked why **Mixtral instruct** performs better than many fine-tuned models. `@intervitens` responded that Mixtral instruct is particularly adept at following instructions and there might be unresolved issues with MoE model fine-tuning.
- **Exploring GPU Enhancements for Models**: `@carsonpoole` shared progress on adapting a variant of the phi2 model with modified fine-tuning, planning to publish weights on Hugging Face and potentially develop a LLaMA-based model variant. The model has also been fine-tuned with chatml data and tokens, integrating them into the model's knowledge.

**Links mentioned**:

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://openreview.net/forum?id=AL1fq05o7H): Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many...
- [Human vs. Machine: Intelligence per Watt](https://meditations.metavert.io/p/human-vs-machine-intelligence-per): Contemplating the possibility that machines won&#x27;t win everywhere all at once
- [Google Colaboratory](https://colab.research.google.com/drive/1-D6ZGE3SZZbIkqhWfxun8CwQWBD5YC2d?usp=sharing): no description found
- [🤗 Transformers](https://huggingface.co/docs/transformers/): no description found
- [Recommendations on new 2 x RTX 3090 setup](https://forums.fast.ai/t/recommendations-on-new-2-x-rtx-3090-setup/78202): Hi,  I’m selling my old GTX 1080 and upgrading my deep learning server with a new RTX 3090. I’m also contemplating adding one more RTX 3090 later next year.  I’ve read from multiple sources blower-sty...
- [Growing Living Rat Neurons To Play... DOOM?](https://www.youtube.com/watch?v=bEXefdbQDjw): Head to https://squarespace.com/thethoughtemporium to save 10% off your first purchase of a website or domain using code: thethoughtemporium_________________...
- [Mixtral](https://huggingface.co/docs/transformers/model_doc/mixtral#transformers.MixtralConfig): no description found
- [GitHub - cg123/mergekit at mixtral](https://github.com/cg123/mergekit/tree/mixtral): Tools for merging pretrained large language models. - GitHub - cg123/mergekit at mixtral
- [EVGA SuperNOVA 1600 P2, 80+ PLATINUM 1600W, Fully Modular, EVGA ECO Mode, 10 Year Warranty, Includes FREE Power On Self Tester Power Supply 220-P2-1600-X1](https://www.evga.com/products/product.aspx?pn=220-P2-1600-X1): Ready for 4th Generation Intel Core Processors (C6/C7 Idle Mode)  Introducing the EVGA SuperNOVA 1600 P2 power supply. This power supply raises the bar with 1600W of continuous power delivery and 92% ...
- [Designs Beyond The Reticle Limit](https://semiengineering.com/designs-beyond-the-reticle-limit/): Chips are hitting technical and economic obstacles, but that is barely slowing the rate of advancement in design size and complexity.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1200051857771475034) (35 messages🔥): 

- **Seeking Specs for CodeLlama Fine-Tuning**: `@ganymede123` inquired about workstation specifications for fine-tuning **CodeLlama 34B** and considered 4xA6000 GPUs. `@teknium` responded that it would only suffice for a *qlora* and stated that a full fine-tune would require nearly a full DGX setup.
  
- **T5 Fine-Tuning Difficulties**: `@maxpappa` is facing challenges with aligning a fine-tuned version of **T5**, noticing deterministic output and reward-accuracies steady at 0.5. Despite tweaking optimizers and schedulers, suggestions such as avoiding paged 8bit Adam by `@locutusque`, and clamping infs particularly in the encoder from `@carsonpoole` were offered to handle the apparent numerical instability.

- **LLMs for Offensive Cyber and CTFs**: `@useewhynot` sought recommendations for LLMs suitable for offensive cyber or CTF challenges. `@kenakafrosty` and `@georgejrjrjr` recommended [WhiteRabbitNeo-13B](https://huggingface.co/whiterabbitneo/WhiteRabbitNeo-13B) and [WhiteRabbitNeo-33B-v1](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1), with options available on HuggingFace.

- **Evaluating AI Coders**: `@findmyke` inquired about the best coding LLM currently available. `@.ben.com` linked to the [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html), which evaluates AI coders with rigorous tests, as a resource for making an informed decision.

- **Fine-Tuning with Llama Factory**: `@moconna` expressed an intent to fine-tune **Mixtral** using Llama Factory and asked for advice on necessary hyperparameters. No specific hyperparameters or templates were provided in the discussion that followed.

**Links mentioned**:

- [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html): no description found
- [WhiteRabbitNeo/WhiteRabbitNeo-33B-v1 · Hugging Face](https://huggingface.co/WhiteRabbitNeo/WhiteRabbitNeo-33B-v1): no description found
- [WhiteRabbitNeo/WhiteRabbitNeo-13B-v1 · Hugging Face](https://huggingface.co/whiterabbitneo/WhiteRabbitNeo-13B): no description found
- [WhiteRabbitNeo - A co-pilot for your cybersecurity journey](https://www.whiterabbitneo.com/): no description found

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1200102006929506434) (3 messages): 

- **Seeking Python Script for 3B Obsidian**: `@vic49.` is in search of a simple Python script that utilizes the transformers library to work with the **3b Obsidian** model. They specify that the code should allow for remote execution (`remote code = true`).
- **Code Refactor in Progress**: `@qnguyen3` confirms they are working on refactoring code to be compatible with the latest **llava repo** for enhanced functionality with **3b Obsidian**.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1200042055980822649) (219 messages🔥🔥): 

- **Base Model Preferred for Fine-tuning in Specific Cases**: `@le_mess` recommended using the base model over the Instruct variant when fine-tuning on languages with limited foundational training data. The suggestion came after `@henriklied` shared his fine-tuning approach with a dataset of 100k Norwegian articles.
  
- **Training Length and Overfitting Concerns**: `@le_mess` advised to stop training at 4 epochs to prevent overfitting, responding to `@henriklied`, who observed flatlining eval loss during his 10-epoch fine-tuning. `@henriklied` also shared a [link](https://gist.githubusercontent.com/henriklied/3dd25bf3090ddb792ec3b1e702fe321d/raw/a155986e117ea69c384ddd87e0580ec18c1c0cef/gistfile1.txt) to the output from a debug flag (`prepare_dataset`) to diagnose the training setup.
  
- **Effective Chat Format for Training Models**: `@suikamelon` and `@c.gato` discussed the effectiveness of different chat formats for training language models, with `@suikamelon` introducing "BlockML" using unique block characters to potentially improve token efficiency. There was also a mention of the challenges related to integrating ChatML tokens in training due to their rare occurrence.
  
- **Discussions on Model Training with Uncommon Tokens**: `@suikamelon` reported tentative success in using unique block characters in place of ChatML tags, noting `▀` and `▄` offered reliable tokenization with Mistral, despite a limited dataset of about 100 examples.
  
- **Regarding Model Quantization and Optimizer Settings**: `@dangfutures` expressed preference for quantization using AWQ and `@bjoernp` sought clarification on whether setting the optimizer in their config would override the default deepspeed optimizer, leading to `@noobmaster29` confirming that the config optimizer should be used due to removal from default deepspeed config. A relevant deepspeed config PR by `@winglian` was also mentioned ([PR #1208](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208)).

**Links mentioned**:

- [axolotl/deepspeed_configs/zero3.json at main · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/deepspeed_configs/zero3.json): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.
- [Pullrequest GIF - Pullrequest - Discover &amp; Share GIFs](https://tenor.com/view/pullrequest-gif-20256291): Click to view the GIF
- [more checks and fixes for deepspeed and fsdp by winglian · Pull Request #1208 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1208): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1200172001835352115) (1 messages): 

- **Quick Retraction from gahdnah**: `@gahdnah` retracted a message after noticing the **active development** in the latest commits, indicating a well-monitored and rapidly evolving project area.
- **Community Celebrates a Grant**: `@gahdnah` expressed excitement and congratulations for a grant received, celebrating the positive news with emoji flair. 🎉🎉🎉
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1200033196105023572) (47 messages🔥): 

- **Loading Models without BitsandBytes**: `@matanvetzler` encountered a `ValueError` when attempting to load a qlora-trained model into vLLM due to it not supporting bitsandbytes quantization. They were advised that vLLM can load the model in fp16, or use AutoAWQ for quantization to fit within VRAM constraints.
- **Merging QLoRA-trained Models**: `@stefangliga` provided a link to a [merge script](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base) on GitHub for combining a qlora-trained model back into the base model, a process separate from the model serving mechanism. They further suggested that Tim Dettmers recommends merging a quantized model, linking to a [twitter post](https://twitter.com/Tim_Dettmers/status/1694654191325573456) for elaboration.
- **SQL Dataset Confusion**: `@sadaisystems` expressed concern about an extraordinarily low loss after a few training steps on a SQL dataset, wondering if this indicates a lack of diversity in the dataset or the model’s proficiency. `@caseus_` reasoned that SQL's deterministic nature might explain the low loss and suggested halting training unless there are complex cases in the data.
- **Benchmark Evaluations During Training with Axolotl**: `@caseus_` informed `@sadaisystems` about the `do_bench_eval: true` option in axolotl to run mini evaluations during training, pointing out that they use datasets from [dharma-1](https://huggingface.co/datasets/pharaouk/dharma-1/tree/main) as benchmarks, which are useful for relative improvement checks.
- **Continuous Pretraining Inquiry**: `@nickbro0355` asked for assistance on how to conduct continuous pretraining on Mistral, with `@dangfutures` indicating they have been attempting that, and `@caseus_` inquiring about the specific dataset to provide further help.

**Links mentioned**:

- [qlora/qmerge.py at main · jondurbin/qlora](https://github.com/jondurbin/qlora/blob/main/qmerge.py): QLoRA: Efficient Finetuning of Quantized LLMs. Contribute to jondurbin/qlora development by creating an account on GitHub.
- [pharaouk/dharma-1 at main](https://huggingface.co/datasets/pharaouk/dharma-1/tree/main): no description found
- [GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions](https://github.com/OpenAccess-AI-Collective/axolotl?tab=readme-ov-file#merge-lora-to-base): Go ahead and axolotl questions. Contribute to OpenAccess-AI-Collective/axolotl development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1200086950594498580) (7 messages): 

- **New Dataset Dropped**: User `@dangfutures` highlighted a new dataset on [Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset) which was used to train the **Snorkel model**. The dataset leverages only the prompts from **UltraFeedback**, with no external LLM responses.

- **Snorkel-Mistral Training Methodology Explained**: The dataset's creation involved generating 5 response variations from **Mistral-7B-Instruct-v0.2** for each prompt, reranking them with **PairRM**, and then applying **Direct Preference Optimization (DPO)** for LLM updates across three iterations.

- **Mistral Gets an Upgrade**: The user `@dangfutures` expressed that **Mistral 7** has been finetuned, which is likely a reference to the improvements mentioned in the dataset methodology.

- **ALPACA's Evaluation Metric Mentioned**: The user `@dangfutures` mentioned a number associated with ALPACA, although the specific context or meaning of "34 percent" was not clarified.

- **Impressive Performance Noted**: A follow-up by `@dangfutures` indicated that despite the initially perceived low percentage, the performance was noted to be **better than an older version of GPT-4**.

- **A Playful Response**: User `_dampf` shared a gif from [Tenor](https://tenor.com/wSUt.gif) in what could be interpreted as a reaction to the preceding messages, though the context of the gif's use was not elucidated in the conversation.

**Links mentioned**:

- [snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset · Datasets at Hugging Face](https://huggingface.co/datasets/snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset): no description found
- [Sure Jennifer Lawrence GIF - Sure Jennifer Lawrence The Mocking Jay - Discover &amp; Share GIFs](https://tenor.com/wSUt.gif): Click to view the GIF

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1200279388797804555) (2 messages): 

- **Seeking Wisdom on DPO Training Plots**: `@noobmaster29` asked if there are any resources available to better understand the **dpo training plots**. However, there were no responses or resources provided in the chat history.
- **Dataset Dilemma for DPO Training**: `@noobmaster29` inquired about the necessary components for a **dpo dataset**, mentioning they have included *prompt/input* and *chosen rejected pair* columns but are experiencing issues with the dataset processing. There was no further clarification or troubleshooting advice provided in the chat history.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=wlPxEq_Mtkc
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199991535681417248) (115 messages🔥🔥): 

- **Seeking Classroom Connection**: `@josemanu72` inquired about running LM Studio as a server in a classroom and connecting from students' desktops. `@fabguy` suggested a [frontend](https://github.com/JShollaj/Awesome-LLM-Web-UI) and a reverse proxy setup; later mentioned solving the issue in another channel.
- **GPU Woes on Ubuntu**: `@xmiruso` faced difficulties with LM Studio not utilizing the GPU on their Ubuntu setup with a Geforce Nvidia 3090. After discussing with `@fabguy` and others, the problem was resolved by ejecting and reloading the model, leading to increased processing speed.
- **Proxy Issues Hinder Model Search**: User `@laooopooo_02864` faced challenges using the model search function due to proxy issues, and `@heyitsyorkie` deduced they might be in a country where huggingface is blocked.
- **Local Model Access from Mobile**: `@cloakedman` sought a way to access their LLM model from a phone, and `@wildcat_aurora` provided a solution with a GitHub link to LM_Chat_TTS_FrontEnd ([front-end](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html)) that permits interaction with LM Studio models.
- **Discussing Best Practices and Error Resolution**: Different users discussed the best coding AI in LM Studio, parallel model running, contacting support for errors, as well as recounting issues and fixes they've experienced, including GPU driver warnings shared by `@fate4real` regarding AMD graphics cards.

**Links mentioned**:

- [GitHub - FriendofAI/LM_Chat_TTS_FrontEnd.html: LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, making it suitable for a wide range of users interested in exploring voice interactions with AI models.](https://github.com/FriendofAI/LM_Chat_TTS_FrontEnd.html): LM_Chat_TTS_FrontEnd is a simple yet powerful interface for interacting with LM Studio models using text-to-speech functionality. This project is designed to be lightweight and user-friendly, makin...
- [GitHub - JShollaj/awesome-llm-web-ui: A curated list of awesome Large Language Model (LLM) Web User Interfaces.](https://github.com/JShollaj/Awesome-LLM-Web-UI): A curated list of awesome Large Language Model (LLM) Web User Interfaces. - GitHub - JShollaj/awesome-llm-web-ui: A curated list of awesome Large Language Model (LLM) Web User Interfaces.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199997195924033607) (54 messages🔥): 

- **C++ Redist Resolution to Model Loading Error**: `@rparada` encountered a model loading error with several models including **Stable Code**, **Deepseek**, **Codellama**, but was able to resolve the issue by updating the C++ redistributables on the suggestion of `@heyitsyorkie`.

- **Assessment of Model Capabilities**: `@heyitsyorkie` commented that the **Magicoder DS 6.7b** and **GPT4** are closely matched in performance, while also elaborating that there isn't a single local multimodal open-source model available to rival GPT4.

- **Azure Cloud Usage for GPT4**: `@mickael6102` shared that their company is running **GPT4 locally on Azure cloud**. This sparked a conversation with `@vbwyrde` about data privacy concerns, costs, and the relation between Microsoft and OpenAI regarding usage and control of proprietary data.

- **Open Source Model Options**: `@vbwyrde` discussed a new model called **"Intern" (InternLM)** and provided a link to it with claims about its exceptional abilities such as a 200k context window and function calling. `@mickael6102` responded with interest and mentioned using **Solar-10b for function calling**.

- **Debating the Strategy of Meta's Llama2**: In response to `@vbwyrde`, `@.gumdro` and `@ptable` speculated on Meta's rationale for providing **open-source models like Llama2**, suggesting reasons such as setting a standard to benefit from downstream product development and taking market space from competing services like OpenAI.

**Links mentioned**:

- [internlm (InternLM)](https://huggingface.co/internlm): no description found
- [Mark Zuckerberg Adjust GIF - Mark Zuckerberg Adjust Facebook - Discover &amp; Share GIFs](https://tenor.com/view/mark-zuckerberg-adjust-facebook-smile-on-trial-gif-11618142): Click to view the GIF

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1200212033254215700) (4 messages): 

- **Bug Alert in Switching MoE Models**: `@msz_mgs` reported a **bug** where changing from a **4X MoE** model to a **2X MoE** model caused an error that wouldn't allow changes, necessitating an app restart.
- **Thanks and Request for Model Details**: `@yagilb` acknowledged `@msz_mgs`'s bug report and asked to share details about the **2x moe** and **4x moe** models being used for further investigation.
- **Insight on MoE Model Configuration**: `@dagbs` offered a tip regarding setting the **num_experts_used** config, suggesting that for a **4x model**, the correct setting should be 2 experts.
- **Performance Issues with Latest Update**: `@golangorgohome` expressed concern about **version 0.2.11** performing poorly on Windows 11 with 32GB RAM, citing slow search icon response and long search times despite having a fast internet connection.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1200033522530914395) (11 messages🔥): 

- **GPU Preferences for Value**: `@gitanos` inquired about the value of the **4060ti 16GB RAM**, prompting a response from `@heyitsyorkie` who recommends investing in a used **3090** for better value, costing only slightly more in the UK.
- **Compatibility Concerns with e0.211**: User `@madan.pandit` reported issues with models ceasing to function, specifically when using version e0.211. Another user, `@heyitsyorkie`, indicated no issues on their end but noted *deprecations of GGML models in favor of llama.cpp*.
- **Memory Error with gguf Models**: `@madan.pandit` mentioned receiving an error about insufficient memory when attempting to utilize **gguf models**.
- **M2 Mac Studio Endorsed for LLMs**: `@heyitsyorkie` advised that buying a **maxxed out M2 Mac Studio** is the perfect choice for running Large Language Models, noting its small form factor and aesthetic appeal.
- **Mixed Opinions on Older GPUs**: A conversation between `@docorange88`, `@wildcat_aurora`, and `@rugg0064` covered the viability of using **P40 or M40 GPUs** for machine learning. The consensus appears to favor P40 GPUs while M40 GPUs are generally not recommended.
  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1200006286159401030) (46 messages🔥): 

- **VRAM Detection Issues on Nobara**: `@mudf00t` reported that LM Studio wasn't recognizing the VRAM on an RTX 3090 on Nobara. `@yagilb` provided a workaround, but noted that the solution was for Nvidia on different setups, not applicable for `@pdg`'s issue on a Mac M2.

- **Models Fail to Load in Final Version**: `@pdg` encountered issues with all models after upgrading to version 0.2.11, which previously worked in an older version. Downgrading to version 0.2.10 via [this link](https://releases.lmstudio.ai/mac/arm64/0.2.10/latest/LM+Studio-0.2.10-arm64.dmg) provided by `@yagilb` led to a new set of errors, with requests for a link to an even older version, 0.2.9.

- **App Download Stalling Issue**: `@khalifa007` faced a problem where the app download would get stuck. `@yagilb` suggested that the issue might be related to the user's internet connection or firewall, and considering the use of a VPN might help.

- **Unusual RAM Error and Interim Fix**: `@pdg` reported an error indicating insufficient RAM, despite having 16 GB available. They discovered that starting with a short sentence and allowing the model to respond first avoids errors when submitting longer text.

- **Insight on Context Length Settings**: `@mattjcly_55150` suggested that the error experienced by `@pdg` could be due to the initial context length setting and recommended adjusting it or the context overflow policy to avoid errors with longer input texts.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1200325029498458112) (1 messages): 

- **Broken Link for Empty Strings in Autogen**: `@sunglasses.emoji` reported that the pinned link regarding empty strings is **broken** and is seeking assistance on creating a custom agent class in autogen studio. No further details or a resolution were provided.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1200104377801773109) (10 messages🔥): 

- **Struggles with Frameworks for Open Models**: `@pefortin` expressed frustration that open models like **memGPT**, **crewai**, and **Open Interpreter** are failing to properly use tools and elements they have access to, despite running medium-sized models such as mixtral8x7B and deepseek coder 33B.
- **mudf00t's Model Exploration**: `@mudf00t` is testing various models and highlighted that having an **RTX 3090** allows loading significantly larger models than some others might be able to.
- **API Amnesia**: `@mudf00t` humorously pointed out that **OpenAI's models**, including those accessed via API, don't seem to recall the current API itself, causing context issues during training.
- **Fine-Tuning Focus**: `@222gate` mentioned discontinuing integration with memGPT and is looking into fine-tuning a **mistral model** for specific function calls, similar to efforts seen with **memgpt datasets**.
- **Hallucinating Directories with Mistral**: `@mudf00t` shared an amusing instance where **Mistral** created an imaginary directory structure complete with a node app and showed code for a non-existent file.
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1200008042968789032) (163 messages🔥🔥): 

- **GPU Rental Options Discussed**: `@mrdragonfox` recommended services like **runpod, vast, and lambda** for renting GPUs by the hour, and later mentioned that **Kaggle** offers free access to GPUs for up to 30 hours per week.
- **Mistral Spending Limits and Support**: `@glorfsf` raised an issue with changing the spending limit in the subscription options, which `@mrdragonfox` clarified defaults to €150. `@mrdragonfox` also suggested contacting **support@mistral.ai** for assistance with changing spending limits.
- **BART Model Limitations and LLM Suggestions**: `@miraimech` expressed dissatisfaction with the BART model from Hugging Face for production use, to which `@mrdragonfox` responded by suggesting the use of open-source models with higher context windows.
- **Model Discussion and API Issues**: `@ethux` and `@i_am_dom` discussed the application of Mistral models and the intricacies behind model versions used in GitHub Copilot, with `@mrdragonfox` clarifying its current backend and use of GPT-3.5.
- **Mistral 7B's Integration and Use Cases Inquiry**: `@sophiamyang` asked for interesting use cases of Mistral models, while `@ethux` and `@f127467` shared their experiences and challenges with model integration, seeking community insights into effective implementations. 



**Links mentioned**:

- [Reddit - Dive into anything](https://www.reddit.com/r/OpenAI/comments/19emcxp/stanford_and_openai_just_released_a_research/): no description found
- [intfloat/e5-mistral-7b-instruct · Hugging Face](https://huggingface.co/intfloat/e5-mistral-7b-instruct): no description found
- [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2): A fast inference library for running LLMs locally on modern consumer-class GPUs - GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs
- [TuringsSolutions/PFAF750 · Datasets at Hugging Face](https://huggingface.co/datasets/TuringsSolutions/PFAF750): no description found

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1200043535878074391) (9 messages🔥): 

- **Finetuning vs. Inference Memory Requirements**: User `@l0gr1thm1k` clarified for `@ethux` that the memory capacity in question is for finetuning rather than training, emphasizing that the concern is about loading the model into memory.
- **Mixtral's Memory Appetite**: In response to `@ethux`, `@l0gr1thm1k` confirmed having adequate memory across four T4 GPUs to handle Mixtral's necessity of at least 26GB for 4-bit inference.
- **On-the-Ground Memory Usage Reports**: `@l0gr1thm1k` reports that the GPU memory usage exceeds expectations just for loading the model, suggesting that actual usage may be higher than the anticipatory figures shared.
- **Quantization Efficiency Debate**: `@mrdragonfox` recommends using exllamav2 for quantization over bnb 4 bit, questioning the use of accelerate in the context of memory efficiency.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1200202437080916009) (3 messages): 

- **Traditional metrics fall short for LLMs**: `@adrienbufort` emphasized that **BLEU and ROUGE** are not useful for evaluating Language Large Models (LLMs) or instruction-tuned LLMs, as these metrics are traditionally used for translation performance assessment.

- **"Elo" for human-like LLM evaluation**: `@adrienbufort` highlighted **"elo"**, a system mimicking chess rankings as being very close to human preference for LLM evaluation, available at [arena.lmsys.org](https://arena.lmsys.org/), although it requires human involvement.

- **Structured evaluations via MMLU and Alpaca**: `@adrienbufort` pointed to multiple-choice questions, like the **Massive Multitask Language Understanding (MMLU)** benchmark ([MMLU paper](https://arxiv.org/pdf/2009.03300.pdf)) for clear LLM performance measurement, and **Alpaca eval** ([Alpaca GitHub](https://github.com/tatsu-lab/alpaca_eval)) for using another LLM to evaluate the responses.

- **Normalized Alpaca Eval Announcement**: `@akshay_1` announced that a **normalized version of Alpaca eval** is now available on the market.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1200043856981405820) (1 messages): 

- **AI browser queries with a twist**: User `@sublimatorniq` showcased *SoContextual.com*, a tool for AI browser queries that include DOM node references. This works with **MistralAI** and was also featured on [Hacker News](https://news.ycombinator.com/item?id=39128480).

**Links mentioned**:

[no title found](https://news.ycombinator.com/item?id=39128480): no description found

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1200185918250819814) (8 messages🔥): 

- **Newcomer seeking RAG guidance**: User `@xrdg` joined the chat asking for advice on how to structure prompts for RAG applications. Not much detail was provided about their specific use case.
- **DSPy for prompt optimization**: `@akshay_1` recommended using **DSPy** to optimize prompt structures, sparking a brief interaction with `@xrdg`.
- **Shoutout from Guatemala**: In a follow-up message, `@xrdg` sends cheers from 🇬🇹, but doesn't provide any further discussion points.
- **Mistral prompt examples explored**: `@xrdg` shared that they have been using **langchain, chroma, and Mistral 7B** and referred to a [prompting guide](https://www.promptingguide.ai/models/mistral-7b). They provided a link that includes an overview and various resources related to Mistral 7B.
- **Optimizing RAG Stacks**: `@akshay_1` suggested that `@xrdg`'s current RAG stack can be further optimized and inquired whether the project was a hobby or in production, but no additional context was provided by `@xrdg`.

**Links mentioned**:

[Prompt Engineering Guide](https://www.promptingguide.ai/models/mistral-7b): A Comprehensive Overview of Prompt Engineering

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1200021798826278983) (35 messages🔥): 

- **Early Stopping Conundrum Continues**: User `@digitalphotographer` is still facing issues with early stopping in their prompts, despite not using control tokens or special characters. They had previously provided notebooks with reproducible examples to Mistral's team but have not received a response.
  
- **Monthly Usage Limit Bug Reported**: Users `@ewanhc`, `@ethux`, and `@fersingb` reported a bug where the monthly usage limit on the billing page resets to 150 euros after an attempt to change it, even if the intention is to lower the limit. They have reported this issue to Mistral support via email.

- **API Hosting Inquiry Cleared Up**: `@loicboutet` inquired about the hosting location of Mistral's API and learnt that it's hosted on Azure in Sweden, information which was found on the privacy page.

- **API's "max_tokens" Bug Surfaced**: `@mrxavierx` discovered and reported a bug where setting "max_tokens" to 1 causes a 500 internal server error instead of returning a single token response or a proper validation error. The issue was documented on Mistral's GitHub repository ([Issue #122](https://github.com/mistralai/mistral-src/issues/122)).

**Links mentioned**:

- [BUG: API /completion endpoint returns 500 (server error) when sending &quot;max_token&quot; = 1 · Issue #122 · mistralai/mistral-src](https://github.com/mistralai/mistral-src/issues/122): While I was playing with the API endpoint /completion I found out a bug with the &quot;max_tokens&quot; body field when it&#39;s set to 1. Instead of returning 1 token response or a validation error, ...
- [no title found](https://console.mistral.ai/billing/): no description found

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1200057056078594111) (29 messages🔥): 

- **Searching for SAM Finetuning**: `@the_alt_man` inquired about a codebase to fine-tune Meta's SAM model, discovering it's not included in the original code and mentioning the use of `AutoGluon` toolbox that employs Lightning but limits to GPU usage.

- **Federated Learning Feasibility Discussed**: `@elyxlz` wondered about the feasibility of multinode training without infiniband and model merging steps. `@stellaathena` indicated experiments on island-like device training while `@smerkyg` pointed to a potential recent study, which @_r0n12 identified as the DiLoCo paper from an [arXiv.org link](https://arxiv.org/abs/2311.08105).

- **Accessing the Pile Dataset**: `@sk5544` sought information on how to access the Pile dataset, getting directions from `@stellaathena` and a direct message offer from `@elyxlz`.

- **Finance Lawyers Described Analogously**: `@catboy_slim_` offered an analogy likening the role of lawyers in finance to combat medics, conveying their reactive position in fast-paced financial events.

- **Project Contributions and Dataset Creation Appeal**: `@pinconefish` offered ML expertise, notably in CV, to contribute to existing projects, while `@stellaathena` and `@wonkothesensible` sparked an idea for a dataset focused on analog clocks displaying 10:10 to study out-of-domain generalization, flagging potential model collapses and active learning cases.

**Links mentioned**:

[DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105): Large language models (LLM) have become a critical component in many applications of machine learning. However, standard approaches to training LLM require a large number of tightly interconnected acc...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1200008066314285086) (125 messages🔥🔥): 

- **Byt5's Efficiency Questioned**: `@main.ai` mentioned that Byt5 shows byte-level transformers to be less efficient, sparking debate with `@the_random_lurker` around the fairness of comparing token and byte sequence lengths.
- **Trouble in Paper Acceptance Land**: `@stellaathena` expressed confusion over the mysterious rejection of a paper with seemingly high review scores, implicating a meta-reviewer's mistake. The discussion highlights the difficulty and lack of transparency in the paper appeal process within academic conferences.
- **Proxy Tuning Large Language Models**: `@digthatdata` shared a [link](https://arxiv.org/abs/2401.08565) to a paper on proxy-tuning LLMs, an efficient alternative to traditional tuning which uses predictions of smaller models to direct larger base models, demonstrating significant performance gains.
- **Self-Rewarding LM Paper Critique**: `@thatspysaspy` critiqued a paper on self-rewarding LMs for using stronger models like Claude 2 and Llama-2-chat during training, suggesting it diminishes the paper's claims and could lead to misguided future research efforts.
- **Chess.com's Fiction of a Chess AI Rival**: `@clockrelativity2003` shared a [Chess.com article](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024) predicting the future of AI in chess in 2024. However, `@alexanderrgriffing` suggested that the article is written by GPT and casts doubt on its seriousness.

**Links mentioned**:

- [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565): Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become inc...
- [Transformers and Cortical Waves: Encoders for Pulling In Context Across Time](https://arxiv.org/abs/2401.14267): The capabilities of transformer networks such as ChatGPT and other Large Language Models (LLMs) have captured the world&#39;s attention. The crucial computational mechanism underlying their performanc...
- [The Quantum Leap of Checkmate: Chess in the Age of AI The year is 2024](https://www.chess.com/blog/IM_practical01/the-quantum-leap-of-checkmate-chess-in-the-age-of-ai-the-year-is-2024): The year is 2024. Robots roam streets, holograms flicker in living rooms, and self-driving cars navigate rush hour with the grace of a seasoned taxi driver. Yet, on a humble wooden board, an ancient d...
- [MoE-Infinity: Activation-Aware Expert Offloading for Efficient MoE Serving](https://arxiv.org/abs/2401.14361): This paper presents MoE-Infinity, a cost-efficient mixture-of-expert (MoE) serving system that realizes activation-aware expert offloading. MoE-Infinity features sequence-level expert activation traci...
- [Evaluating the Medical Knowledge of Open LLMs - Part 1 &mdash; MedARC](https://www.medarc.ai/blog/medarc-llms-eval-part-1): In this MedARC blog post, we compare generalist and medical domain-specific Large Language Models (LLMs) like GPT-4, Mistral, and Llama, and we evaluate their performance on MultiMedQA tasks for medic...
- [MVDream: Multi-view Diffusion for 3D Generation](https://arxiv.org/abs/2308.16512): We introduce MVDream, a multi-view diffusion model that is able to generate consistent multi-view images from a given text prompt. Learning from both 2D and 3D data, a multi-view diffusion model can a...
- [| bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.20.517210v3)): no description found
- [CLARA: Multilingual Contrastive Learning for Audio Representation Acquisition](https://arxiv.org/abs/2310.11830): Multilingual speech processing requires understanding emotions, a task made difficult by limited labelled data. CLARA, minimizes reliance on labelled data, enhancing generalization across languages. I...
- [Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom](https://www.biorxiv.org/content/10.1101/2023.10.09.561603v1): Although AlphaFold2 (AF2) and RoseTTAFold (RF) have transformed structural biology by enabling high-accuracy protein structure modeling, they are unable to model covalent modifications or interactions...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1200153068268961913) (2 messages): 

- **Fix Integration Issue Acknowledged**: `@hailey_schoelkopf` expressed readiness to merge a fix if necessary for an unspecified integration issue, highlighting surprise at the behavior and a desire to test it personally.
- **Adding Weights and Biases Support**: `@hailey_schoelkopf` shared a [GitHub pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339) by `@ayulockin` which adds support for **Weights and Biases** to the `lm-evaluation-harness`. They are considering the optimal placement for the newly created `wandb.py` file within the project structure.

**Links mentioned**:

[feat: Add Weights and Biases support by ayulockin · Pull Request #1339 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1339): In #359 @parambharat did proposed to add support for W&amp;B logging. However it was done before the big refactor that got in. As a user of both lm-evaluation-harness and wandb, I have opened this PR ...

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1200163709352427631) (16 messages🔥): 

- **Seeking Guidance on QLoRA Tuning**: `@kenakafrosty` inquired about resources or info related to tuning GPT-NeoX 20B with QLoRA and mentioned having issues with loss not decreasing during training. `@stellaathena` clarified that the NeoX library doesn’t support QLoRA and suggested reaching out on GitHub for help with `trl`, `transformers`, and `peft` which `@kenakafrosty` is using.

- **pytest Issues with GPT-NeoX**: `@catboy_slim_` mentioned removing `--forked` from pytest and highlighted the need for a separate effort to get pytest running cleanly again for the project.

- **Tests Failing When Forked**: `@catboy_slim_` reported major updates to Python, PyTorch, and CUDA, and while able to run a basic model, expressed concern over the inability to manually validate every possible branch, indicating that tests need to work and created an [issue on GitHub](https://github.com/EleutherAI/gpt-neox/issues/1132).

- **Testing Framework Discussions for Torch**: `@catboy_slim_` expressed doubts about existing testing frameworks adequately handling PyTorch code due to infrequent testing of such code by developers.

- **Project Collaborators on Validation and Compute Access**: `@tastybucketofrice` is arranging compute access for collaborators, including `@337128969059172353`, to further test their changes to the project and extended an offer to `@catboy_slim_` for compute access to assist in testing.

**Links mentioned**:

[Tests fail when run with pytest --forked · Issue #1132 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/issues/1132): Describe the bug When tests are run with pytest --forked per the instructions in /test/README.md, a large number of tests fail with the error: RuntimeError: Cannot re-initialize CUDA in forked subp...

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199994496906166282) (59 messages🔥🔥): 

- **LAION dataset seeking**: User `@ppwwyyxx` inquired about the **laion2b-en aesthetics scores**, as the initial dataset link provided was disabled. A response stated that the dataset access has been temporarily disabled on the dataset author's request, with a recommendation to check the announcements for updates.
- **Voice Chat Interface Demo Unveiled**: `@jpcl_` announced a new demo of a complete **voice chat interface**, combining Whisper and WhisperSpeech with an open-source LLM, touting reduced latency for more natural conversations, and inviting collaboration to improve the system. They shared a link to the [Hacker News announcement](https://news.ycombinator.com/item?id=39130140).
- **Image Captioning Strategies Discussed**: Users `@pseudoterminalx`, `@thejonasbrothers`, and `@limiteinductive` shared approaches for image captioning with AI, with an emphasis on giving clear prompts to avoid hallucination and focusing on describing visible content only.
- **AI Tech Sprint Recruitment**: `@ninjaa2377` is looking for developers to join a team for the VA's AI Tech Sprint to work on an ambitious project involving clinical encounter notes, offering potential prize money and prestige. Developers interested were directed to reach out via DM and visit the [official challenge website](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data).
- **Pirated US Channels Operate Freely?**: `@pseudoterminalx` mentioned that local cable companies use pirated channels from the US without repercussions, claiming the government in their unspecified country isn't influenced by foreign companies or bribery.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39130140): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/DefendingAIArt/comments/19djc0a/): no description found
- [laion/laion2B-en-aesthetic at main](https://huggingface.co/datasets/laion/laion2B-en-aesthetic/tree/main): no description found
- [ European Commission &#x1f1ea;&#x1f1fa; on Instagram: &quot;Nice try Fluffy, but indeed you got the news right! Today we presented measures to allow European AI start-ups and SMEs to train their model using our High-Performance Computing&#x2019;s capacity.](https://www.instagram.com/reel/C2e0qkNqqu7/?igsh=MXc2MzB2ZGo3dXUwOQ==): 87K likes, 666 comments - europeancommission on January 24, 2024: &quot;Nice try Fluffy, but indeed you got the news right!   Today we presented measures to allow Europe...&quot;
- [Challenge.Gov](https://www.challenge.gov/?challenge=ai-tech-sprint-for-documenting-va-clinical-encounters-and-integrating-community-care-data): Challenge.Gov is the official GSA government website supporting prize challenges and prize competitions that are sponsored by the US federal government.  Here federal agencies provide prize awards to ...

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1200069735388287056) (71 messages🔥🔥): 

- **Google's Monetization Trouble**: User `@max_voltage` discussed Google's struggle to find financially significant business models beyond advertising over the past 15 years, pointing out that perhaps they ought to have taken a SaaS approach similar to OpenAI instead of fully integrating DeepMind into Google.
- **Cautious Optimism for Byte-Level Transformers**: `@marianbasti` shared cautious optimism about byte-level transformers, referencing an [arXiv paper](https://arxiv.org/pdf/2401.13660.pdf), and `@thejonasbrothers` humorously noted that progress always seems "one month away."
- **Text-to-Image Diffusion and Identity Preservation Advances**: `@vrus0188` shared two GitHub links describing the latest advances: [RPG-DiffusionMaster for text-to-image diffusion](https://github.com/YangLing0818/RPG-DiffusionMaster) and [InstantID for ID-preserving generation](https://github.com/InstantID/InstantID), and `@chad_in_the_house` confirmed its coolness.
- **Scaling-Up Image Restoration (SUPIR)**: `@thejonasbrothers` linked to an [arXiv submission](https://arxiv.org/abs/2401.13627) that introduces SUPIR and its capabilities for image restoration guided by textual prompts, highlighting the paper's presence among top papers on Hacker News.
- **High Costs of AI Model Training Discussed**: Users `@vrus0188`, `@chad_in_the_house`, `@thejonasbrothers`, and `@limiteinductive` engaged in a conversation about the substantial costs of training AI models like Stable Diffusion, though acknowledging that costs are expected to decrease with time and technological advancements.


**Links mentioned**:

- [The Architecture of a Biologically Plausible Language Organ](https://arxiv.org/abs/2306.15364): We present a simulated biologically plausible language organ, made up of stylized but realistic neurons, synapses, brain areas, plasticity, and a simplified model of sensory perception. We show throug...
- [Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild](https://arxiv.org/abs/2401.13627): We introduce SUPIR (Scaling-UP Image Restoration), a groundbreaking image restoration method that harnesses generative prior and the power of model scaling up. Leveraging multi-modal techniques and ad...
- [PALP: Prompt Aligned Personalization of Text-to-Image Models](https://arxiv.org/abs/2401.06105): Content creators often aim to create personalized images using personal subjects that go beyond the capabilities of conventional text-to-image models. Additionally, they may want the resulting image t...
- [GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥](https://github.com/InstantID/InstantID): InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥 - GitHub - InstantID/InstantID: InstantID : Zero-shot Identity-Preserving Generation in Seconds 🔥
- [GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (PRG)](https://github.com/YangLing0818/RPG-DiffusionMaster): Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs (PRG) - GitHub - YangLing0818/RPG-DiffusionMaster: Mastering Text-to-Image Diffusion: Recaptioning, Pl...
- [Whole brain functional recordings at cellular resolution in zebrafish larvae with 3D scanning multiphoton microscopy - Scientific Reports](https://www.nature.com/articles/s41598-021-90335-y): no description found

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1200030911715090472) (87 messages🔥🔥): 

- **Perplexity Pro Features Clarified**: In response to `@winnie.zhao`'s inquiry, `@icelavaman` and `@mares1317` provided a [link](https://blog.perplexity.ai/faq/what-is-perplexity-pro) detailing Perplexity Pro's features like unlimited Copilot queries, the ability to upload files for content exploration, and access to powerful AI models including GPT-4 and Claude 2.1.
- **Data Retention Concerns Voiced**: Several users, led by `@emisaurus_hex` and `@firesonwires`, expressed confusion and concern over Perplexity's data retention policy. Clarifications by `@icelavaman`, a presumed Perplexity expert, indicated that deleted threads are removed from servers after 30 days.
- **Questions on Using Perplexity's Models**: Users like `@divyanshu0500`, `@odobostudio`, `@lukas8a`, and others asked technical questions regarding JSON output from models, file upload limits, and the efficiency of models like Claude and GPT-4 for summarizing PDFs and academic work.
- **Understanding Account and Search Data Policies**: The discussions about Perplexity's privacy policy highlighted some ambiguity, prompting suggestions for clearer policy wording to avoid misinterpretation and to confirm if search data is indeed retained for the lifetime of an account.
- **Community Interaction and Technical Support**: Members like `@danielagmz888` and `@icelavaman` offered assistance on issues ranging from applying credit codes to addressing concerns. There was also a lighthearted exchange between `@reflext` and `@sedierta` about pro subscription costs and the performance of various models.

**Links mentioned**:

- [What data does Perplexity collect about me?](https://blog.perplexity.ai/faq/what-data-does-perplexity-collect-about-me): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [What is Perplexity Pro?](https://blog.perplexity.ai/faq/what-is-perplexity-pro): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Perplexity - AI Companion](https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo?utm_content=chrome-link&ref=chrome&utm_medium=google&utm_source=chrome+store&utm_campaign=chrome-extension>)): Ask anything while you browse
- [Perplexity Blog](https://blog.perplexity.ai/faq/how-does-file-upload-work.): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1200104475394850866) (4 messages): 

- **Intersection of Search and AI**: `@jsudaniel` highlighted the CEO's connection to Google Search and OpenAI, noting that Perplexity AI serves as an intersection of these technologies. They shared a YouTube video titled "[I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D)" discussing the benefits of using Perplexity AI.

- **Perplexity Eases Smartsheet Learning Curve**: `@nicknalbach` found that Perplexity AI provided efficient answers to problems encountered while transitioning from Excel to Smartsheet. Perplexity helped him overcome the steep learning curve where other resources provided scattered solutions.

- **Conceptual Aid for Astronomy Education**: `@coloradocomplex` mentioned using Perplexity to help explain concepts in their astronomy class, showing the usefulness of Perplexity AI in education.

- **No additional info**: A link was shared by `@coloradocomplex`, but no context or additional information regarding the content or purpose of the link was provided.

**Links mentioned**:

[I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q&t=589s&pp=ygU6QUkgU2VhcmNoIFdhcnM_ISBQZXJwbGV4aXR5LmFpIChDaGF0R1BUICsgQmluZykgdnMuIEdvb2dsZQ%3D%3D): Main Takaways From this Video: &quot;I use Perplexity more than ChatGPT, BARD, and Microsoft Copilots for five main reasons, including its use in content creation...

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1200017158302863361) (5 messages): 

- **Web vs API Discrepancy in Code Output**: `@benhirap` expressed that the website version of Perplexity AI produces much better code than the API version.
- **Seeking API and Labs Parity**: `@stijntratsaert_01927` inquired about the default parameters used by Perplexity AI labs as they are experiencing difficulty replicating lab results via the API.
- **Billing Issues Need Resolution**: `@aiagileguy` reported an issue with being double charged and reached out to support@perplexity.ai for a refund of credits but has not received a resolution after more than 1-2 business days.
- **Assistance Requested for Support Concern**: Following the billing issue, `@aiagileguy` is seeking help or pointers to expedite the refund process from Perplexity AI.
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1200176286639870022) (3 messages): 

- **Community Highlights - Social Exploration**: `@lunarflu` praises the **HuggingFace community** for its focus on ML content and invites members to join the organization for [early access to the "Posts" feature](https://huggingface.co/social-post-explorers). It's highlighted as a less noisy alternative compared to Twitter or LinkedIn for people interested in AI & ML.


**Links mentioned**:

- [social-post-explorers (Social Post Explorers)](https://huggingface.co/social-post-explorers): no description found
- [Cosmos Arena](https://thenameless.net/cosmos-arena): no description found
- [@gsarti on Hugging Face: &quot;🔍 Today&#39;s pick in Interpretability &amp; Analysis of LMs: From Understanding to…&quot;](https://huggingface.co/posts/gsarti/888341627040205): no description found
- [CheXRay - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/CheXRay): no description found
- [GitHub - TonyAssi/HF-Embed-Images: Generates image embeddings for 🤗 Datasets](https://github.com/TonyAssi/HF-Embed-Images): Generates image embeddings for 🤗 Datasets. Contribute to TonyAssi/HF-Embed-Images development by creating an account on GitHub.
- [@Tonic on Hugging Face: &quot;hey there folks , work in progress, but basically celebrating the release of…&quot;](https://huggingface.co/posts/Tonic/220992701457145): no description found
- [not-lain/TunBERT · Hugging Face](https://huggingface.co/not-lain/TunBERT): no description found
- [@mehd-io on Hugging Face: &quot;We just released the first Text2SQL model for DuckDB 🦆🧠
You can try it out…&quot;](https://huggingface.co/posts/mehd-io/779023528910338): no description found
- [@Tonic on Hugging Face: &quot;👋 Hi there folks,

I launched my first competition ! 

Goal : Use AI to…&quot;](https://huggingface.co/posts/Tonic/783827682062088): no description found
- [@gsarti on Hugging Face: &quot;🔍 Today&#39;s pick in Interpretability &amp; Analysis of LMs: Model Editing Can Hurt…&quot;](https://huggingface.co/posts/gsarti/256926950283134): no description found
- [ClovenDoug/small_128_all-MiniLM-L6-v2 · Hugging Face](https://huggingface.co/ClovenDoug/small_128_all-MiniLM-L6-v2): no description found
- [Deepfake Detection - a Hugging Face Space by not-lain](https://huggingface.co/spaces/not-lain/deepfake-detection): no description found
- [@vicgalle on Hugging Face: &quot;Can you merge models of different sizes? ⚗️

Well, yes, if the models are…&quot;](https://huggingface.co/posts/vicgalle/320544784279721): no description found
- [tenyx/TenyxChat-8x7B-v1 · Hugging Face](https://huggingface.co/tenyx/TenyxChat-8x7B-v1): no description found
- [AI Lineage Explorer: A Step Towards AI Integrity.](https://huggingface.co/blog/backnotprop/integrity-explorer): no description found

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199992656353304586) (40 messages🔥): 

- **Decoding the Training Dilemma**: `@asprtnl_50418` highlighted the sheer scale of resources needed for pretraining models, referencing the [Llama-2-7b model](https://huggingface.co/meta-llama/Llama-2-7b), which required about 184k GPU hours on A100 GPUs. They also mentioned alternative cost-effective methods like *fine-tuning* and using [LoRA/QLoRA adapters](https://huggingface.co/docs/peft/conceptual_guides/lora) to lessen hardware demands.
  
- **Strategies for Training and Evaluation Split**: Users `@enka55` and `@the_aureo` discussed the challenge of splitting data into training and evaluation sets for LLM training, with suggestions including using pandas `train_test_split` with the `stratify` parameter, and supplementing with knowledge bases like RAG for topics not covered in training data.

- **Feature Extraction Fundamentals Explained**: `@vipitis` clarified that feature extraction refers to sequence embedding via encoder-only models like BERT, with uses in tasks like clustering. They also directed users to the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for relevant models and metrics.

- **The Quest for Model Evaluation Insights**: `@green_eye` expressed frustration over the lack of accessible qualitative assessments of models beyond benchmarks, seeking more human-readable reviews that detail where models excel or fall short.

- **Troubleshooting Model Loading Issues**: `@newincoding` faced difficulties loading a model which `@sebastian3079` diagnosed as potentially being due to hardware limitations, recommending at least 32GB of RAM for handling models with 40 billion parameters.

**Links mentioned**:

- [Google Colaboratory](https://colab.research.google.com/drive/1tPiTnqk2tMwYLhehS9qVPkcQ9J0gTVv2?usp=sharing): no description found
- [Supported models and hardware](https://huggingface.co/docs/text-embeddings-inference/supported_models): no description found
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): no description found
- [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b#hardware-and-software)): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1200218914693582849) (1 messages): 

- **In Quest of a Data Set Evaluation Framework**: User `@rosebei3ngan3g` expressed a need for frameworks to evaluate data sets specifically for large language models, highlighting the absence of such tools despite the availability of many frameworks for evaluating the models themselves. They questioned how data set evaluation should be approached without established frameworks.
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1200037182090530897) (7 messages): 

- **HuggingFace Datasets Under the Microscope**: User `@andysingal` shared a [GitHub project](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis) focusing on a large-scale analysis of dataset cards on **HuggingFace**. This project could be particularly insightful for anyone diving into dataset documentation in AI.
- **From Zero to ML Hero**: User `@pacificvoltage` is exploring the basics of machine learning by reading the first chapter of "Understanding Deep Learning" ([udlbook](https://udlbook.github.io/udlbook/)), and marveled at the use of deepfake technology on a recent **Machine Learning Street Talk** interview with Noam Chomsky, which can be watched [here on YouTube](https://www.youtube.com/watch?v=axuGfh4UR9Q&t=8412s).
- **Binocular Vision on AI-Generated Text**: `@tea3200` introduced a [paper from arXiv](https://arxiv.org/abs/2401.12070) that presents ***Binoculars***, a novel detector that claims to distinguish human from machine-generated text with over 90% accuracy, without requiring any training data or model-specific modifications.
- **SemEval2024 Shared Task Spotlight**: User `@vipitis` mentioned a [GitHub shared task](https://github.com/mbzuai-nlp/SemEval2024-task8) for the **SemEval2024-task8** competition, focused on multidomain, multimodel, and multilingual machine-generated text detection, potentially related to the "Binoculars" approach just shared.
- **On the Flutter Wing with AI**: `@akindelemichael` sparked interest with a [new package](https://github.com/gtbluesky/onnxruntime_flutter) for integrating ONNX models in Flutter apps, coinciding with a growing trend noted by `@osanseviero` for AI capabilities in Flutter, including a [Flutter SDK for HuggingFace Inference APIs](https://huggingface.co/posts/shivance/676533662914249).

**Links mentioned**:

- [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070): Detecting text generated by modern large language models is thought to be hard, as both LLMs and humans can exhibit a wide range of complex behaviors. However, we find that a score based on contrastin...
- [@shivance on Hugging Face: &quot;Hi Community! 
I&#39;m ecstatic to announce Flutter SDK for HuggingFace Inference…&quot;](https://huggingface.co/posts/shivance/676533662914249): no description found
- [GitHub - mbzuai-nlp/SemEval2024-task8: SemEval2024-task8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection](https://github.com/mbzuai-nlp/SemEval2024-task8): SemEval2024-task8: Multidomain, Multimodel and Multilingual Machine-Generated Text Detection - GitHub - mbzuai-nlp/SemEval2024-task8: SemEval2024-task8: Multidomain, Multimodel and Multilingual Mac...
- [NOAM CHOMSKY - THE GHOST IN THE MACHINE](https://www.youtube.com/watch?v=axuGfh4UR9Q&t=8412s): Patreon: https://www.patreon.com/mlstDiscord: https://discord.gg/ESrGqhf5CBIn this special edition episode, we&#39;re elated to unveil the Professor Noam Chomsky...
- [GitHub - YoungXinyu1802/HuggingFace-Dataset-Card-Analysis: Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on HuggingFace (ICLR 2024)](https://github.com/YoungXinyu1802/HuggingFace-Dataset-Card-Analysis/tree/master): Navigating Dataset Documentations in AI: A Large-Scale Analysis of Dataset Cards on HuggingFace (ICLR 2024) - GitHub - YoungXinyu1802/HuggingFace-Dataset-Card-Analysis: Navigating Dataset Documenta...
- [GitHub - gtbluesky/onnxruntime_flutter: A flutter plugin for OnnxRuntime provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile and desktop platforms.](https://github.com/gtbluesky/onnxruntime_flutter): A flutter plugin for OnnxRuntime provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile and desktop platforms. - GitHub - gtbluesky/onnxruntime_flutter...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1200059722879991848) (12 messages🔥): 

- **A Launchpad for Nemo Project on HuggingFace**: `@tonic_1` expressed enthusiasm for launching a **Nemo model project** and the idea of writing a detailed **blog post** on HuggingFace resonated well. `@not_lain` agreed, responding with a commitment to write a post *as soon as possible*.

- **WhisperSpeech Demo Hosted on HuggingFace**: `@tonic_1` introduced a [**WhisperSpeech** demo on HuggingFace](https://huggingface.co/spaces/Tonic/whisperspeech), which allows for multi-language text-to-speech and the creation of a voice print with a minimal audio input.

- **CheXRay Analysis in Development**: `@tonic_1` shared a [link to CheXRay](https://huggingface.co/spaces/Tonic/CheXRay), a work-in-progress tool for analyzing Chest X-Rays, indicating active projects and development in medical imaging AI.

- **Community Blogpost Outreach by @lunarflu**: `@lunarflu` reached out to `@mateomd_dev` suggesting that a community blog post could help increase reach for `@mateomd_dev`'s work, and provided a [link to HuggingFace's blog section](https://huggingface.co/blog/community). `@mateomd_dev` showed interest in refining their article for the HuggingFace audience.

- **Upcoming wav2vec2-bert Model Announcement**: `@yehors` announced the pending publication of a **wav2vec2-bert model** based on the Common Voice 10 dataset, indicating the model is in the final stages of preparation.

**Links mentioned**:

- [WhisperSpeech - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/whisperspeech): no description found
- [CheXRay - a Hugging Face Space by Tonic](https://huggingface.co/spaces/Tonic/CheXRay): no description found
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers): no description found
- [Hugging Face – Community Blogs](https://huggingface.co/blog/community): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1200020728574132275) (3 messages): 

- **Encouragement for Isamu**: `@lunarflu` expressed support for a user named Isamu, telling them to take their time and including a heart emoji for emphasis.

- **Text-to-Video Model Lumiere Raises the Bar**: `@fishie22` discussed Google's new **Lumiere** model, explaining its innovative use of a **Space-Time UNET** that maintains temporal consistency and can generate video at a notable 16fps for 80 frames. They provided a link to the research paper: [Google's Lumiere Research](https://arxiv.org/abs/2401.12945).

- **Positive Feedback on Medium Article Benchmarking**: `@starsupernova` tweeted about a Medium article, praising its benchmarking as "Super great" and adding a smiley face emoji to emphasize their positive feedback.

**Links mentioned**:

[Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945): We introduce Lumiere -- a text-to-video diffusion model designed for synthesizing videos that portray realistic, diverse and coherent motion -- a pivotal challenge in video synthesis. To this end, we ...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 

spikespiegel5112: How to load LoRA model in local?
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1200164215164518431) (5 messages): 

- **Brief Inquiry on LMM Architecture**: `besiktas` asked about the design choices behind using **idefics/flamingo resampler/cross-attention** in the LMM currently in training instead of a simpler approach like linear projection or a pretrained vision encoder.
  
- **Gemini Pro Vision AI Introduced**: `ahmed3ibrahim` discussed trying out the Swift API's [Gemini Pro Vision AI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/), mentioning its key features like handling multiple images in one request and providing a comprehensive **API health report**.

- **Curiosity About CVPR2024 Papers**: `iloveh8` was looking for a way to see all papers, both rejected and accepted, for **CVPR2024** but did not receive a response.

**Links mentioned**:

[Gemini Pro Vision AI API Documentation (swift-api-swift-api-default) | RapidAPI](https://rapidapi.com/swift-api-swift-api-default/api/gemini-pro-vision-ai1/): no description found

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1200003453414559744) (15 messages🔥): 

- **TorToiSe Pronunciation Engine Acknowledged**: `@mr_nilq` mentioned that **TorToiSe** offers the best quality in TTS but is slow, sharing a [link](https://github.com/152334H/tortoise-tts-fast) to a modified version that is 5x faster.
- **Seeking Advice on Training AI for Q&A**: User `@ysk.dev` is considering options for training AI on 10k Q&A pairs, debating between Amazon Lex and training VDS, and inquiring about the hardware specs needed for running a server with long answers.
- **Help Requested for Transformer ImportError**: User `@srovnbh` faced an error importing `TFTrainer` from the `transformers` package and received suggestions to ensure the correct version is installed.
- **Talk on Trusting 'Black Box' Models**: `@vipitis` shared a [link to a talk](https://talks.cam.ac.uk/talk/index/211336) about evaluating "black box" models, questioning the trust in models when users can't see behind the API.
- **Windows Compatibility Issue for Bits and Bytes**: `@kingpoki` realized the reason for their issue was the lack of Windows support for an unnamed application or feature they referred to as bits and bytes.

**Links mentioned**:

[talks.cam : Replicating and auditing black-box Language Models.](https://talks.cam.ac.uk/talk/index/211336): no description found

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (1 messages): 

spikespiegel5112: How to load LoRA model in local?
  

---


### HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1200238081865941012) (1 messages): 

- **Gradio Hits 4.16 with Robust Features**: `@abidlabs` announced the release of **`gradio 4.16`** boasting major features such as native support for **Polars Dataframe**, a new Gallery component usable as an input, improved low-latency streaming for chatbots, and automatic documentation for custom components. This "HUGE release" is detailed in their comprehensive changelog, available [here](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md).

**Links mentioned**:

[gradio/CHANGELOG.md at main · gradio-app/gradio](https://github.com/gradio-app/gradio/blob/main/CHANGELOG.md): Build and share delightful machine learning apps, all in Python. 🌟 Star to support our work! - gradio-app/gradio

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1200121061669343343) (1 messages): 

- **Webinar on LLMCompiler Incoming**: `@jerryjliu0` reminded everyone about a **webinar** happening in 10 minutes featuring the authors of the **LLMCompiler** paper, which details a framework for **parallel function calls in agents**. The framework, envisioned to boost performance and efficiency, can be explored in their paper ([LLMCompiler Paper](https://arxiv.org/pdf/2312.04511.pdf)) and further resources like **LlamaPack** and a **Notebook** are available at their dedicated links.

**Links mentioned**:

[LlamaIndex Webinar: Efficient Parallel Function Calling Agents with LLMCompiler · Zoom · Luma](https://lu.ma/lf9iroox): LLMs are great at reasoning and taking actions. But previous frameworks for agentic reasoning (e.g. ReAct) were primarily focused on sequential reasoning, leading to higher...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1200128299385499748) (7 messages): 

- **Slack Bot Tutorial Shared**: IFTTT announced a new OSS repository with a **step-by-step guide** to build a Slack bot that can learn from conversations and answer organizational questions, written by `@seldo`. The bot is built on the @SlackHQ platform. [Build your Slack bot](https://t.co/E8KJNeoXfr).

- **Zilliz Cloud Pipeline Integrated with LlamaIndex**: LlamaIndex highlighted their collaboration with `@zilliz_universe` on integrating the Zilliz Cloud Pipeline into LlamaIndex, enhancing retrieval services and multi-tenancy support. Check out the [guest blog post](https://t.co/luDjSgiokt).

- **LlamaIndex Supports New OpenAI Embedding Models**: The LlamaIndex team has released version 0.9.38, which includes **day 0 support** for @OpenAI's latest embedding models. For more details, see the [release notes](https://t.co/kyIoTUaeuD).

- **Good Prompting Out of the Box with LlamaIndex**: IFTTT mentioned an often overlooked feature of LlamaIndex, emphasizing that it **creates effective prompts** by default, which can be customized if desired. Further insights available [here](https://t.co/GUJxx6TO0a).

- **LlamaIndex Now Available in TypeScript**: Announcement from IFTTT that LlamaIndex.TS version 0.1.0 has been released, extending support for @OpenAI's latest embeddings to TypeScript thanks to a quick contribution from `@yi_ding`. For TypeScript enthusiasts, visit [LlamaIndex.TS 0.1.0 release](https://t.co/lVVsWAXcdl).

- **Qdrant Engine Included in LlamaIndex.TS Release**: The 0.1.0 version of LlamaIndex.TS also comes with added support for `@qdrant_engine`. The update was highlighted as a bonus feature in the TypeScript release. [Check this feature out](https://twitter.com/llama_index/status/1750673214840394198).

**Links mentioned**:

- [llama-index](https://t.co/kyIoTUaeuD): Interface between LLMs and your data
- [Building Scalable RAG Applications with LlamaIndex and Zilliz Cloud Pipelines](https://t.co/luDjSgiokt): Introduction

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1200040486535176283) (38 messages🔥): 

- **LLM not available for LlamaIndex Text Inference Server**: `@cheesyfishes` confirmed that LlamaIndex does not currently have a large language model (LLM) for the TextGenerationInference server but mentioned that the Langchain one works with a wrapper.
- **Configuring Chat Engine with `similarity_top_k`**: In response to `@richard1861`, `@whitefang_jr` provided a Python code snippet to configure the similarity retrieval count of the chat engine in LlamaIndex, using `similarity_top_k=5`.
- **Retrieval Challenges in Domain-Specific Use Cases**: `@lancerninja` and `@cheesyfishes` discussed a more complex retrieval scenario involving rephrasing questions using an LLM before executing another retrieval, aiming for improved performance but concerned about increased response times due to multiple steps.
- **Anticipating Integration with New OpenAI Embedding Models**: `@ayfri` shared a link to OpenAI's announcement about new embedding models and API updates. `@cheesyfishes` responded, hinting at upcoming support for these new features in LlamaIndex.
- **Customizing Prompts for Contextualized Responses in LlamaIndex**: `@shri_j` asked about obtaining answers from OpenAI when the query information isn't in the provided context. `@cheesyfishes` directed toward modifying default prompts to allow for such functionality, sharing a link to documentation.

**Links mentioned**:

- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.
- [Usage Pattern - LlamaIndex 🦙 0.9.38](https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern.html#getting-and-setting-custom-prompts): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1200109478939480164) (5 messages): 

- **Exploring Zep's Capabilities**: User `@yoursaviorjesus` inquired if anyone has experience with **Zep**, pointing out its features like chat history memory and entity extraction. They provided a link to Zep’s documentation and various quick start guides: [Zep Documentation](https://docs.getzep.com/).

- **Inquiring LlamaIndex's Nature**: `@zeekg_46676` asked if **LlamaIndex** is a vector store or operates like Amazon Kendra which uses natural language search. `@cheesyfishes` clarified that LlamaIndex is more akin to Kendra and is versatile, capable of using any vector store or language model for various operations.

- **Demonstrating Self-Learning Knowledge Graph**: `@chiajy` shared their work on a self-learning knowledge graph RAG workflow that features recursive retrieval, automated creation, and multi-hop reasoning, exemplified through a Harry Potter book demo. A detailed explanation and the ramifications of this knowledge graph can be found in their write-up: [Harry Potter and the Self-Learning Knowledge Graph](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning).

**Links mentioned**:

- [Zep - Fast, scalable building blocks for LLM apps](https://docs.getzep.com/): no description found
- [Harry Potter and the Self-Learning Knowledge Graph RAG](https://messyproblems.substack.com/p/harry-potter-and-the-self-learning): WhyHow.AI&#x27;s self-learning RAG with knowledge graphs to bring accuracy and rules to Vertical AI - demonstrating recursive retrieval, memory, automated context-aware knowledge graph construction.

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1200087852042698923) (36 messages🔥): 

- **No LLM Paper Club Recording**: `@kbal11` responded to `@farrealdori` with the information that sessions of the LLM Paper Club are not recorded to allow participants to share details about their work more freely, thus no replay is available.
- **Introducing Morpheus-1**: `@shivdinho` shared a link to a tweet announcing Morpheus-1, described as the world's first multi-modal generative ultrasonic transformer designed to induce and stabilize lucid dreams, and noted its innovative nature.
- **Go-Go-Labs Coding Sprint**: `@slono` provided a link to a GitHub repo, showcasing that 5k lines of code were written in 4 days for `yaml-custom-tags` experiments, indicating swift progress towards project completion.
- **GPT-4 Turbo & Embedding Models Update**: `@dimfeld` shared OpenAI's announcement on the release of an updated GPT-4 Turbo preview model and new embedding models, while `@swyxio` linked notes on the matter from Twitter.
- **Martian's LLM Leaderboard Launch**: `@cute_hamster_07119` announced the launch of Martian's Model Router at `https://leaderboard.withmartian.com/`, a tool helping to evaluate various LLM inference products, with `@coffeebean6887` and `@fanahova` discussing the documentation and open-source aspect of the project.

**Links mentioned**:

- [🚨🚨 That's a lot of YAML 🚨🚨](https://noyaml.com/): no description found
- [Tweet from talrid23 (@talrid23)](https://x.com/talrid23/status/1750463847226388574?s=46&t=XV1VJkM4nCYVU6fROoKkfw): JSON format is becoming the de facto standard for output generation with LLM.  However, is it the optimal format ? 🤔 We claim that not - YAML outputs are shorter and simpler, leading to faster infere...
- [LLM Inference Provider Leaderboard](https://leaderboard.withmartian.com/): A live, unbiased benchmark on LLM inference APIs made by Martian
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/): no description found
- [KREA is building the next frontier of human creativity ⚡️](https://cerebralvalley.beehiiv.com/p/krea-building-next-frontier-human-creativity): Plus: Co-founder Diego on embracing curiosity and chaos...
- [go-go-labs/cmd/experiments/yaml-custom-tags at main · go-go-golems/go-go-labs](https://github.com/go-go-golems/go-go-labs/tree/main/cmd/experiments/yaml-custom-tags): GO GO EXPERIMENTAL LAB. Contribute to go-go-golems/go-go-labs development by creating an account on GitHub.
- [Tweet from Prophetic (@PropheticAI)](https://x.com/propheticai/status/1750534355242418300?s=46&t=JE84TqLviekDnEt8MAT-Eg): INTRODUCING MORPHEUS-1  The world’s first multi-modal generative ultrasonic transformer designed to induce and stabilize lucid dreams.   Available for beta users Spring 2024
- [Evaluation Methodology - Provider Leaderboard](https://docs.withmartian.com/provider-leaderboard/evaluation-methodology): no description found
- [Reproducibility - Provider Leaderboard](https://docs.withmartian.com/provider-leaderboard/reproducibility): no description found
- [GitHub - withmartian/provider-dashboard: Open sourced backend for Martian&#39;s LLM Inference Provider Leaderboard](https://github.com/withmartian/provider-dashboard): Open sourced backend for Martian&#39;s LLM Inference Provider Leaderboard - GitHub - withmartian/provider-dashboard: Open sourced backend for Martian&#39;s LLM Inference Provider Leaderboard

  

---


### Latent Space ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1200014492210319413) (1 messages): 

- **LLM Paper Club Asia Launches**: `@ivanleomk` announced the kickoff of the **LLM Paper Club in Asia**, starting with a discussion on the "Attention Is All You Need" paper. Interested individuals can [sign up for future notifications](https://lu.ma/llm-paper-asia) and access the event [here](https://discord.gg/tPnG5qMu).

**Links mentioned**:

- [LLM Paper Club (Asia Edition!) · Luma](https://lu.ma/llm-paper-asia): UPDATE: Updated with a link to the discord stage that we&#x27;ll be using Asia-timezone friendly version of the Latent.Space x EugeneYan.com LLM Paper Club! This week we&#x27;ll be covering the...
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/tPnG5qMu): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1200030161123414027) (8 messages🔥): 

- **Asia Paper Club Timing**: `@ivanleomk` thanked everyone for participating in today's paper club and mentioned that next week's discussion may cover *Self-Rewarding Language Models*. They are open to other paper suggestions and note that `@796917146000424970` or they will cover it if there are no volunteers.
- **Beta Test Feedback Request**: `@aimuggle` expresses gratitude for participation and is requesting feedback to improve the paper club, which is still in a `beta` phase.
- **Clarification on Self-Instruction**: `@stealthgnome` inquired whether "self-instruct" is the input for "self-reward," suggesting an interest in discussing the interplay between these concepts.
- **Upcoming US Paper Club Schedule**: `@ivanleomk` asked about the scheduled paper for next week's US paper club, and `@eugeneyan` provided the [Pythia paper](https://arxiv.org/abs/2304.01373) as the topic of discussion, listing the authors and their arXiv links.
- **Appreciation for Pythia Paper Info**: `@ivanleomk` showed appreciation for the details `@eugeneyan` provided for the forthcoming discussion on the Pythia paper.

**Links mentioned**:

[Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373): How do large language models (LLMs) develop and evolve over the course of training? How do these patterns change as models scale? To answer these questions, we introduce \textit{Pythia}, a suite of 16...

  

---



### DiscoResearch ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/1199993210039177247) (2 messages): 

- **Mergekit Guidance for Mixtral Training**: `@philipmay` shared a [GitHub issue comment](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289) from the author of **mergekit** that may inform the **DiscoResearch mixtral training**, questioning the finetuning process post-merging models with options like "hidden" or "random."
- **Auxiliary Loss Key for MoE Training**: `@bjoernp` acknowledged the potential helpfulness of the shared **mergekit** information, stressing that getting the auxiliary loss right is crucial for **MoE (Mixture of Experts) training**.

**Links mentioned**:

[Mixtral branch: What option should I choose when I want to do some finetuning after the merge? · Issue #116 · cg123/mergekit](https://github.com/cg123/mergekit/issues/116#issuecomment-1909429289): The parameter description of &quot;hidden&quot; and &quot;random&quot; does not exactly explain what to do when I want to finetune later. Is it even useful (possible) to finetune after merging with &q...

  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1200000600281194536) (23 messages🔥): 

- **Quality Data Filtering May Not Be King**: `@bjoernp` shared a fascinating paper from arXiv which challenges the standard practice of filtering pretraining data for quality, suggesting that "quality" filtering doesn't always correlate with improved model performance. The study proposes selecting data to maximize model performance on target tasks, avoiding biases of handpicked data quality notions. [Read the abstract here](https://arxiv.org/abs/2401.12926).

- **Experimenting with Preference Signals for LLMs**: User `@hammadkhan` suggested an experiment involving Supervised Fine-Tuning (SFT) where a prompt's completion is changed from positive to negative, potentially influencing the learning of language models.

- **KTO: A Different Approach to Training Models**: `@bjoernp` mentioned that Key Term Optimization (KTO) could be utilized for training models. It is likened to Direct Preference Optimization (DPO) but with binary signals, relating completions to being either desirable or undesirable.

- **Guidance on Using KTO with Datasets**: In a detailed explanation, `@hammadkhan` outlined how the KTO loss can be maximized for model generation utility, contrasting it with DPO which requires preference-based paired data. Hugging Face's TRL documentation and the paper by Rafailov et al., 2023, provide further context on the DPO Trainer and expected dataset formats. [See the TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer).

- **Binary Labels for Continual Model Updates**: `@hammadkhan` brought up ContextualAI's suggestion of Kahneman-Tversky Optimisation (KTO), which uses binary good or bad labels for model updates, simplifying the labelling process in production environments.

- **OpenAI Launches GPT4-Turbo and Reduces GPT3.5 Prices**: `@rasdani` highlighted an announcement from @OfficialLoganK about OpenAI's launching of GPT-4 Turbo, updates to GPT-3.5 Turbo, including significant price reductions, and new API features like scoped API keys and embedding dimension specifications. [More details on OpenAI's Blog](https://openai.com/blog/new-embedding-models-and-api-updates).

**Links mentioned**:

- [DsDm: Model-Aware Dataset Selection with Datamodels](https://arxiv.org/abs/2401.12926): When selecting data for training large-scale models, standard practice is to filter for examples that match human notions of data quality. Such filtering yields qualitatively clean datapoints that int...
- [DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer): no description found
- [Tweet from Logan.GPT (@OfficialLoganK)](https://x.com/officiallogank/status/1750589278709780780?s=46&t=1jtkL4JPu-DUOdo8JC668g): Great news for @OpenAIDevs, we are launching:  - Embedding V3 models (small & large) - Updated GPT-4 Turbo preview - Updated GPT-3.5 Turbo (*next week + with 50% price cut on Input tokens / 25% price ...

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1200004770333741117) (12 messages🔥): 

- **Neue deutsche Jina Modellankündigung**: `@sebastian.bodza` informierte über die bevorstehende Veröffentlichung des **deutschen Jina Modells** "jinaai/jina-embeddings-v2-base-de" auf *hf*. Dieses könnte für Ranking-Zwecke hilfreich sein.

- **Exploring Question Generation with Mixtral**: `@sebastian.bodza` teilte [Beispiele für Fragegenerierung](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md) auf GitHub und erwähnte die Nutzung von **Mixtral** in 4 bit **gptq** with **vllm** für diese Aufgabe.

- **Community Collaborative Efforts**: `@bjoernp` zeigte Interesse an `@sebastian.bodza`s Arbeit und bot Unterstützung an, insbesondere beim Generieren von positiven und schwierigen negativen Beispielen.

- **Neue OpenAI-Embedding-Modelle veröffentlicht**: `@bjoernp` wies auf die Veröffentlichung neuer OpenAI-Embedding-Modelle hin, die auch eine verbesserte Mehrsprachigkeit aufweisen. Der Beitrag enthält einen Link mit weiterführenden Informationen: [Read more about it here](https://openai.com/blog/new-embedding-models-and-api-updates).

- **Automatische Generierung von Qualitätsdaten mit Genie**: `@bjoernp` teilte einen [Link zu einer Studie](https://arxiv.org/abs/2401.14367) über das Genie-Verfahren zur automatischen Erstellung hochwertiger datenbasierte Inhalte, das möglicherweise nützliche Filtermechanismen enthält.

**Links mentioned**:

- [Genie: Achieving Human Parity in Content-Grounded Datasets Generation](https://arxiv.org/abs/2401.14367): The lack of high-quality data for content-grounded generation tasks has been identified as a major obstacle to advancing these tasks. To address this gap, we propose Genie, a novel method for automati...
- [New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.
- [GitHub: Let’s build from here](https://github.com): GitHub is where over 100 million developers shape the future of software, together. Contribute to the open source community, manage your Git repositories, review code like a pro, track bugs and fea...
- [Embedding_Training/README.md at main · SebastianBodza/Embedding_Training](https://github.com/SebastianBodza/Embedding_Training/blob/main/README.md): Contribute to SebastianBodza/Embedding_Training development by creating an account on GitHub.

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199995853339902003) (6 messages): 

- **Finetuning Success with DiscoLM German**: `@thomasrenkert` reported success in **finetuning DiscoLM German 7B v1** using *unsloth*, and is looking forward to **DiscoLM German** versions based on **Mixtral-Instruct**. 
- **Middle High German Translation Data**: In response to `@hammadkhan`'s inquiry, `@thomasrenkert` clarified that the finetuning was done on a **custom dataset** for translating Middle High German to Modern German.
- **Bjoernp Acknowledges DiscoLM Update**: `@bjoernp` commended `@thomasrenkert`'s finetuning achievement with a brief message of approval.
- **Impressive Embeddings Efficiency Announced**: `@hammadkhan` shared a tweet from `@Nils_Reimers` about upcoming **embeddings** that significantly outperform OpenAI’s on the MIRACL benchmark with only 256 dimensions, offering a potential **12x saving on vector database costs**.

**Links mentioned**:

[Tweet from Nils Reimers (@Nils_Reimers)](https://x.com/nils_reimers/status/1750631888094380268?s=46&t=-TRJUfVdW8KeDqen1HJU1Q): @OttoZastrow @jerryjliu0 Yes, embeddings is a massive focus for us, with amazing launches upcoming.  E.g. OpenAI 54.3 on MIRACL with 3072 dimensions versus our upcoming  256 dimensional-like model wit...

  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/1200152287843197129) (2 messages): 

- **OpenAI Unveils Next-Gen Embedding Models**: User `@potrock` shared [OpenAI's announcement](https://openai.com/blog/new-embedding-models-and-api-updates) about new embedding models launch, GPT-4 Turbo and moderation models, tools for API usage management and upcoming reduced pricing on GPT-3.5 Turbo. The enhancements aim to refine developers' control over API keys and provide insights into API usage.
- **Documentation for New Features Available**: Accompanying the announcement, OpenAI has updated its [documentation](https://platform.openai.com/docs/guides/embeddings/) to guide users through the new embedding models and the updated GPT and moderation models. The documentation is a key resource for developers using these APIs.
- **Navigational Misstep in Message Posting**: `@shacrw` noted a misdirection in the message posting, suggesting that the announcement should have been shared in a different channel, likely intended for a focused discussion on the new updates. The correct channel was indicated with a link.

**Links mentioned**:

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.

  

---


### LLM Perf Enthusiasts AI ▷ #[announcements](https://discord.com/channels/1168579740391710851/1168760950803931136/) (1 messages): 

mat_mto: Thanks Jeff! love all the work you're doing so far
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1200152253890301993) (16 messages🔥): 

- **OpenAI Unveils New Models and Lower Prices**: `@potrock` shared a [blog post](https://openai.com/blog/new-embedding-models-and-api-updates) announcing new embedding models, updates to GPT-4 Turbo and moderation models, addition of API management tools, and soon-to-come lower pricing on GPT-3.5 Turbo.
- **A Win for Embedding Efficiency**: `@potrock` highlighted the benefits of the new shortened embeddings, while `@res6969` expressed eagerness to upgrade their system to include the updated models, citing the unnecessary move to open-source embeddings given these improvements.
- **OpenAI: The Simple Solution for Shipping Features**: `@res6969` reflected on the ease of using OpenAI for quickly implementing features, compared to managing independent open-source models.
- **The Dilemma of Convenience vs. Community Models**: While `@potrock` acknowledged the convenience of OpenAI's solutions, he also pointed out the availability of many great open-source embedding models that allow for personal fine-tuning.
- **Economic Trade-Offs in Model Selection**: `@shacrw` and `@michelcarroll` discussed the cost benefits of using OpenAI's newer, larger embedding models with dimension shortening, mentioning storage savings and comparable API costs that could lead to overall reduced expenditure.

**Links mentioned**:

[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates): We are launching a new generation of embedding models, new GPT-4 Turbo and moderation models, new API usage management tools, and soon, lower pricing on GPT-3.5 Turbo.

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1200045647257153567) (12 messages🔥): 

- **Welcome to the AI Galaxy**: `@quarknova`, a newcomer from ENS interning at INRIA, expressed interest in using LangChain for their projects and queried the community for tips, contemplating the use of the GitHub version over the commercial one.

- **Crafting AI Personalities**: `@jstansbe` inquired about the possibility of creating custom AI personalities like an "Elon Musk AI" without relying on external AI APIs. `@ksolo__` responded with a resource, suggesting that the process is known as finetuning, and provided a [link to a course on deep learning](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/).

- **Shoutout to LangChain's Efficiency**: `@johnnucleus` applauded the LangChain community for enabling the swift creation of a chatbot with web search capabilities using LangChain and Streamlit, expressing amazement at the efficiency and simplicity.

- **Generating Synthetic Data with LLMs**: `@rajib2189` is exploring the use of Large Language Models (LLMs) for synthesizing data to train traditional machine learning models, while `@johnny2x2` shared how he employs LLMs for RAG generation to produce SQL queries from a context and schema.

- **Working with PARQUET in LangChain**: `@benjaminbascary` sought assistance for manipulating PARQUET files in LangChain, leading to `@johnny2x2` providing a code snippet showing how to import and use PARQUET files as document sources, using `pandas` for loading and the `DataFrameLoader` from LangChain.

**Links mentioned**:

[Finetuning Large Language Models](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/): no description found

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1200090590503780512) (3 messages): 

- **LangServe Agent Examples Promoted**: User `@veryboldbagel` shared links to **LangServe** agent examples, including one not listed in the main examples section at the [LangServe main examples](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples) and a specific example for a [configurable agent executor](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor).
- **Custom Agent Construction with LCEL**: `@veryboldbagel` clarified that for defining custom tools, an off-the-shelf **OpenAI tools agent** suffices, and further instructed on constructing a custom agent with **LCEL**, recommending the [LangGraph](https://python.langchain.com/docs/langgraph#agentexecutor) for defining custom agent runtime with more expressive power.
- **Stream Response Issues in LangServe**: `@hiranga.g` reported an issue with not receiving a **stream response** while using the example [agent_with_history](https://github.com/langchain-ai/langserve/blob/main/examples/agent_with_history/server.py) and experimenting with `RemoteRunnable` from **langchain.js**; There was also a mention of a bug when using Agents with LangServe, suggesting that `chain.streamLog()` might be a workaround, which did not yield the expected results.

**Links mentioned**:

- [🦜🕸️LangGraph | 🦜️🔗 Langchain](https://python.langchain.com/docs/langgraph#agentexecutor.): ⚡ Building language agents as graphs ⚡
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve?tab=readme-ov-file#examples): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/configurable_agent_executor at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/configurable_agent_executor): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1200076122575085578) (2 messages): 

- **Exploring SQL chain limitations**: `@johnny2x2` shared insights on handling **SQL queries** with **LangChain** for a manufacturing company's order delays. They found that **SQL Chain** struggles with large databases, but creating **curated views** within the database with descriptive names improves performance.
- **Refinements lead to better query management**: By embedding questions that would return a query within a custom multi-vector retriever, `@johnny2x2` initially found the local AI ran examples too often—a challenge that was mitigated by using OpenAI to process SQL queries while keeping the **data private** with local LLM.
- **Enhanced chain workflow with tool-oriented queries**: Now abandoning local AI for SQL code generation, `@johnny2x2` adopts a new strategy where each query acts as a tool in their **task processing chain**, leading to improved results in their workflow which involves a sequence of generating tasks, processing tasks with SQL tools, and evaluating information for task loops.
  

---



### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1200325637450235934) (3 messages): 

- **Impending LLM Release Upgrade Preview**: `@simonw` announced plans to release an update for the LLM, which involves a significant upgrade to the **openai library**. Testers are invited with details provided in a [GitHub comment](https://github.com/simonw/llm/issues/325#issuecomment-1911533536).

- **Anticipating the 0.13 Milestone**: More information about the forthcoming LLM release, labeled as **0.13 Milestone**, can be found in the dedicated [GitHub milestone page](https://github.com/simonw/llm/milestone/8).

- **Request for Readline Issue Resolution**: `@simonw` is seeking assistance for a readline issue within LLM, where arrow keys yield ANSI codes instead of cursor navigation, as described in this [GitHub issue](https://github.com/simonw/llm/issues/376).

**Links mentioned**:

- [0.13 Milestone · simonw/llm](https://github.com/simonw/llm/milestone/8): Access large language models from the command-line - 0.13 Milestone · simonw/llm
- [llm chat - readline problems still present · Issue #376 · simonw/llm](https://github.com/simonw/llm/issues/376): When I open llm chat, I expect that using the left and right arrow keys will navigate the cursor but instead I get nasty ANSI codes printed to the screen. $ llm chat Chatting with gpt-4 Type &#39;exit...
- [Upgrade for compatibility with OpenAI 1.0 library · Issue #325 · simonw/llm](https://github.com/simonw/llm/issues/325#issuecomment-1911533536): Currently: Successfully installed openai-1.0.1 $ llm -m gpt-4-turbo &#39;hi&#39; Error: module &#39;openai&#39; has no attribute &#39;ChatCompletion&#39;

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=wlPxEq_Mtkc
  

---


### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 

arielnlee: Anyone working on bakklava-2?!
  
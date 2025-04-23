---
id: 914ee360-1e87-4dc6-90f8-6de50ace445b
title: '[AINews] MetaVoice & RIP Bard'
date: '2024-02-07T22:41:50.157897Z'
status: sent
type: public
source: api
metadata: {}
original_slug: ainews-metavoice-rip-bard
---

<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/6/2024. We checked **20** guilds, **308** channels, and **5284** messages for you. Estimated reading time saved (at 200wpm): **437 minutes**.

Remember Coqui, the TTS startup that [died last month](https://buttondown.email/ainews/archive/ainews-132024-rip-coqui/)? Well, a new TTS model that supports voice cloning and longform synthesis is here ([try it](https://ttsdemo.themetavoice.xyz/)).

 ![image.png](https://assets.buttondown.email/images/fa979c16-2f6e-4893-b68b-3c295d3d3ef2.png?w=960&fit=max) 

It's a [small](https://metavoice.notion.site/Join-MetaVoice-e4c907cb6a2f4c33af2b148f635adda4) startup but a promising first ship.

In other news, [Google killed the Bard brand](https://twitter.com/AndrewCurran_/status/1754546359460590002) for Gemini.

---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **AI Training Conversations Heat Up**: Discussions involved AI models like **Mixtral**, **Nous Mixtral DPO**, and **Miqu 70B**, comparing them with OpenAI's GPT models on efficiency and capability. Debates flared on the **Reddit /r/LocalLLaMA subreddit**'s moderation with shared links to GitHub, Hugging Face, and YouTube videos discussing AI advancements and issues within the community.

- **Antiquity Meets Modernity in AI Nomenclature**: In the #characters-roleplay-stories channel, `Thespis 0.8` sparked a debate about its Greek tragedy origins, turning the conversation towards the use of mythology for AI context. The practice of **lorebooks in roleplay** was discussed as a tool for **prompt engineering**, and **DPO (Direct Preference Optimization)** was mentioned with example wandb links provided.

- **Removing Safety Features for AI's Full Potential**: Users shared insights on **LoRA fine-tuning** to remove safety guardrails from models like **Llama2 70B instruct**, citing a [LessWrong post](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from). Discussions also suggested **combining datasets** could enhance model control during fine-tuning.

- **Finding Tech Limits in AI Development**: Query on the possibility of offloading transformer layers to a GPU as observed in `llama.cpp`, led to the conclusion that the **Transformers library doesn't support layer splitting between CPU and GPU**. Interest was shown in **Meta's Sphere project**, considered for its potential of incorporating **frequent updates using big data tools**.

- **Exploring AI Implementation on Alternative Platforms**: Questions arose about the implementation of **LLaMa 2 on MLX for Apple Silicon**, specifically on how to adapt the model’s query parameters to this platform. The technical intricacies of multi-head or grouped-query formats were under consideration.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **DALL-E Adopts Content Authenticity Initiative Standards**: DALL-E generated images now include metadata conforming to **C2PA specifications**, aiding verification of OpenAI-generated content, as announced by `@abdubs`. The change aims to assist social platforms and content distributors in content validation. Full details are available in the [official help article](https://help.openai.com/en/articles/8912793-c2pa-in-dall-e-3).

- **AI Censorship and Authenticity Stir Debate**: AI censorship's impact on user experience prompted a hot exchange; users `@kotykd` and `@arturmentado` respectively criticized and defended AI censorship, highlighting user freedom and misuse prevention. Ethical concerns about presenting AI-generated art as human-created were also voiced, emphasizing the need for honest disclosure per OpenAI's TOS.

- **Metadata Manipulation Recognized as Trivial by Community**: The significance of metadata in image provenance was heavily discussed, with users agreeing on the ease of removing such data, rendering it an unreliable measure for image source verification. This reflects the technical challenges of securing digital image authenticity.

- **Open Source AI Models Positioned Against Commercial Giants**: A debate flourished comparing open-source AI models to commercial options like GPT-4, touching on the impact on innovation and the potential growth of competitive open-source alternatives. The discussion reflects the engineering community's focus on the development landscape of AI technologies.

- **Discussions Surround GPT-4 Usability and Development**: Multiple users, such as `@mikreyyy`, `@glory_72072`, and others, sought help regarding GPT-4 usage issues like logouts, finding demos and storytelling capabilities. The conversation also touched on answer limits and complexities involved in custom GPT usage, indicating a concentration on the practical applications and limitations of GPT-4 in real-world scenarios. 

- **Community Seeks Improvement and Interaction in AI-Driven Projects**: In the realm of **prompt engineering and API discussions**, community members `@loier` and `@_fresnic` looked for collaborative input on improving GPT modules and refining character interaction prompts. Meanwhile, `@novumclassicum` sought to provide feedback directly to OpenAI, indicating a desire for more streamlined communication between developers and the OpenAI team.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Hugging Chat Assistant Personalization**: Hugging Face introduces **Hugging Chat Assistant**, allowing users to build personalized assistants with customizable name, avatar, and behavior. It supports LLMs such as **Llama2** and **Mixtral**, streamlining the user experience by eliminating the need for separate custom prompt storage. Check out the new feature at [Hugging Chat](https://huggingface.co/chat/assistants).

- **Dataset Viewer for PRO and Enterprise**: The Dataset Viewer on HuggingFace now supports private datasets, but the feature is exclusively for PRO and Enterprise Hub users. This update is aimed at enhancing data analysis and exploration tools ([source](https://x.com/julien_c/status/1752716726012129577)).

- **Synthetic Data Trends**: HuggingFace Hub adds a `synthetic` tag to facilitate the sharing and discovery of synthetic datasets, signaling the growing importance of synthetic data in AI.

- **AI-Powered Quadrupeds and AI in Fashion**: Cat Game Research is developing a video game featuring **the first quadruped character controller utilizing ML and AI**, while Sketch to Fashion Collection turns sketches into fashion designs. Explore [badcatgame.com](https://badcatgame.com) and [Hugging Face Spaces](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection) for innovations in AI-powered gaming and fashion.

- **BLOOMChat-v2 Elevates Multilingual Chats**: **BLOOMChat-v2**'s 176B parameter multilingual language model with 32K sequence length is improving upon its predecessors. An API and further advancements are anticipated; details are communicated in a [Twitter summary](https://twitter.com/SambaNovaAI/status/1754928815590277146) and [detailed blog post](https://sambanova.ai/blog/bloomchat-v2).

- **Reading Group Excitement and Resources**: The HuggingFace Reading Group schedules a presentation for decoder-only foundation models for time-series forecasting, and a GitHub repository ([link](https://github.com/isamu-isozaki/huggingface-reading-group)) is established to compile resources from past sessions, enhancing knowledge sharing.

- **Diffusers and Transformers Learning**: For those new to diffusion models, the community suggests courses by HuggingFace and FastAI for deep dives, while queries about details such as timestep setting and conditioning model validation suggest active experimentation and learning within the domain.

- **NLP Channel Gets Textbook**: A discussion on fine-tuning LLMs like **LLama chat** or **Mistral** with textbook content proposes to enhance domain specific understanding, thereby improving the educational chat capabilities of models.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **LM Studio Launches v0.2.14**: A new **LM Studio v0.2.14** version is released, tackling critical bugs like UI freezes and input hangs, which you can access via [LM Studio website](https://lmstudio.ai/). Remember to update for a smoother experience.

- **Ease-of-Use Shines with LM Studio**: Users are drawn to [LM Studio for its simple user interface](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed), enabling anyone to use LLMs without coding skills. But, watch out for default resets in LLM folder locations after updates.

- **Local Model Execution Challenges**: While users experience issues like model generation freezes and poor GPU utilization, patches in **LM Studio's latest updates** are meant to address these. Also, for detailed model-fine tuning instructions, check out the [YouTube tutorial](https://www.youtube.com/watch?v=MDA3LUKNl1E).

- **Hardware Hub**: Debates continue over optimal hardware setups for AI tasks, with one user preparing an AMD 8700g test bed and prompting curiosity over possible **7950x3d** upgrades. Fan configurations for cooling involve **2x180mm fans versus 3x120mm Arctic P120 fans**, but some caution against overestimating APU performance for AI-related computations.

- **Feedback Loop for LM Studio**: Users call attention to a few issues in beta versions, like app hang-ups when ejecting models and outdated non-AVX2 beta releases; LM Studio team seems responsive to these concerns. macOS users highlight a persistent bug where the server stays active even after app closure.

- **Specialized AI Utilization Queries**: Inquiries arise about **chain-compatible seq2seq LLM models** for Python scripting and whether Crew-AI has similar **UI or web interfaces** as other platforms like AutoGen Studio.

- **Model Preferences and Experiences Shared**: Users discuss their experiences with various models, with **Mixtral** mentioned casually as working "alright" for user phoenix2574 in the open-interpreter channel.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

**Mistral Outshines in Programming**: `@carsonpoole` discovered that **Mistral** outperforms **phi2** significantly on the code section of **OpenHermes 2.5** under identical sampling scenarios. The discussions included implications for GPT-4's programming capabilities and sparked curiosity surrounding the expected skillset of a 2-billion-parameter model, with cited expectations from Microsoft Research.

**Sparsetral Unveiled and Math Benchmarking Excitement**: The introduction of **Sparsetral**, a sparse MoE model, comes complete with resources such as the [original paper](https://arxiv.org/abs/2401.02731) and [GitHub repos](https://github.com/wuhy68/Parameter-Efficient-MoE). Meanwhile, `.benxh` celebrated **Deepseek**, which incorporates a technique called DPO to achieve new proficiency levels in math-focused assessments.

**Quant Tune and EQ-Bench**: `@tsunemoto` has quantized Senku-70B, a finetuned version of the hypothetical Mistral-70B, yielding an EQ-Bench score of 84.89, and shared it on [HuggingFace](https://huggingface.co/ShinojiResearch/Senku-70B-Full). This sparked a broader discourse on the significance of mathematics in appraising language models' abilities and hosting LLM-powered robotics hackathons.

**Language Model Quirks and Mixtral Issues Noted**: Users experienced that Mixtral, directed in Chinese, presents mixed-language responses, and similar issues with OpenHermes. Cloudflare’s AI platform adoption of these models was highlighted through [tweets](https://x.com/teknium1/status/1755020133398155269?s=46).

**Support for Robot-Control Framework**: `@babycommando` sought suggestions on finetuning multi-modal models and released **MachinaScript for Robots** with a [GitHub repository](https://github.com/babycommando/machinascript-for-robots). They asked for guidance on finetuning Obsidian and technical specifications for robot interactions using their LLM-driven framework.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **VRAM Cowboys and Silicon Showdowns**: The discussions emphasize the hardware demands of running full fp16 AI models, with suggestions pointing to a minimum of **100GB vRAM** for optimal performance and speculation on the adequacy of **Nvidia's 4090** and dual **4080 setups**. The debate also took turns discussing the merits and obsolescence of **Intel Macs** in the face of **Apple's Silicon Macs**, evidencing strong divergences on upgrade philosophies and practical longevity concerns.

- **Cost-Effective AI Modeling Secrets Unveiled**: Users tackled the challenge of reducing computational costs for **Mistral models**, referencing [DeepInfra pricing](https://deepinfra.com/pricing) and suggesting solutions like serverless platforms, hardware accelerators, and **LlamaCPP**. Operational cost discussions also navigated the terrain of data sensitivity, fine-tuning, and the balance between in-house inferences and professional hosting services.

- **From Fine-Tuning Frustrations to Inference Innovations**: Technical frustrations arose around padding inconsistencies in fine-tuning, with one member expressing confusion despite following resources like [QLoRa tutorials](https://youtu.be/OQdp-OeG1as). Others shared experiences of improved model boot times, some reporting readiness within **2-10 seconds**, and the community brainstormed effective prompt engineering for **Mistral-8x7B** models using tools like **Llama 1.6**.

- **Anticipation Builds for Open Source Release**: A brief exchange indicated community interest in an unspecified tool, with the promise to **release it open source** once feasible; however, no further details or timelines were provided.

- **Mark Your Calendars for Office Hours**: The next **office hour** session for the Mistral community is officially on the schedule, and interested parties are provided with a [Discord event link](https://discord.gg/mistralai?event=1204405056825327677) for access.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Compromised Discord Account Re-secured**: `@astropulse` fell prey to a spearfishing attack compromising their Discord account. Users shared cybersecurity tips, highlighting the usefulness of [Have I Been Pwned](https://haveibeenpwned.com/) for checking if email addresses have been affected by data breaches.

- **Creating Sparse Neural Networks**: `@mkaic` is working on innovative neural network architectures that allow dynamic reconfiguration of connections during training, which could enhance model sparsity and performance.

- **Novel AI Projects in Need of Recognition**: `@SegmentationFault` brought attention to [PolyMind on GitHub](https://github.com/itsme2417/PolyMind), which aims to combine multiple AI capabilities in one platform, emphasizing the project's practical value over entertainment-focused applications.

- **Text-to-Image Consistency Without Training**: Introducing [ConsiStory](https://arxiv.org/abs/2402.03286), a training-free model discussed by `@thejonasbrothers`, designed to improve consistency in text-to-image generation using novel attention mechanisms and feature injections.

- **Google Research's Lumiere for Text-to-Video**: A [YouTube video](https://youtu.be/Pl8BET_K1mc) shared by `@spirit_from_germany` showcases *Lumiere*, Google Research's model focused on creating globally consistent video content from text inputs.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

**GPT Rivalries and Bots**: GPT-3.5 showed surprising prowess in generating code for obscure languages over GPT-4, while the Eleuther server debates the trade-offs between openness and spambot disruptions.

**MetaVoice TTS Model Unveiled**: MetaVoice 1B, a new TTS model, was released with open source licensing, sparking discussions about its performance which includes features like zero-shot voice cloning and emotional speech synthesis as detailed in a [tweet](https://x.com/reach_vb/status/1754984949654904988?s=46).

**Evaluating Model Extrapolation and Optimization**: A variety of methods for understanding and pushing model capabilities were reviewed, from analyzing loss vs sequence length to SELF-DISCOVER framework outperforming traditional methods on reasoning benchmarks as described in [this paper](https://arxiv.org/abs/2402.03620).

**Infinite Limits and Interpretability**: Queries about deep learning infinite depth limits and loss landscapes sparked interest in existing research, while a new method called Evolutionary Prompt Optimization (EPO) for language model interpretation was proposed in a [research paper](https://arxiv.org/abs/2402.01702).

**Dissecting LLM Prompt Influence**: The search for reliable input saliency methods in LLM prompts continued with skepticism against Integrated Gradients, underscored by a [concerning paper](https://arxiv.org/abs/2212.11870) that casts doubt on the attribution methods' ability to infer model behavior.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **GCP Grapples with A100 Availability**: Community members are experiencing difficulty sourcing **A100 GPUs** on Google Cloud Platform, raising concerns of a potential shortage. Discussions also touched on benchmark times for various models and tools, such as **lm-eval-harness**, where a 7b model MMLU test takes around 12 hours on a 4090 GPU.

- **Quest for Axolotl UI**: Hugging Face extends a $5000 bounty for creating an Axolotl Spaces training UI, prompting a collaboration call for frontend (with **Tailwind CSS** preference) and backend developers, ideally in Python. Debates ensue whether to use **Shiny** or **Gradio** for the UI, with prototype and support offers on the table from the Shiny team at Posit.

- **Saving Models Multiplied**: Users report persistent issues when attempting to save models on multi-GPU, multi-node configurations, with suspicion that distributed saving might not be correctly implemented in Axolotl. Despite the latest transformers library version (4.37.2), pull requests with purported fixes are being scrutinized for multi-node training, as community members actively seek code adaptations to resolve **mistral fft** saving errors.

- **Tuning DPO with Alpacas and ChatML**: Community interactions reveal challenges in achieving reliable **DPO** results, prompting advice to significantly lower learning rates. A shift from **Alpaca format** to **ChatML** is being explored despite earlier successes with Alpaca, with insights into personal workaround methods involving Metharme being shared.

- **Jupyter Confusions and Corrections**: Users experiencing critical errors and warnings with Jupyter notebooks are directed towards potential fixes, including a Github pull request ([Cloud motd by winglian](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235)) addressing issues with a mounted volume affecting the workspace directory. Advice is dispensed to reclone repositories as part of troubleshooting.





---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Seeking Clarity on Claude Pro's Necessity**: In the **general** channel, `@brewingbrews_92072` questioned the need for a **Claude Pro** subscription for minimal use cases, reflecting a thoughtful approach before upgrading.
- **Evaluating AI Services Price Points**: A contrasting discussion in the **general** channel delved into the cost-effectiveness of **Perplexity's API** (priced at 0.07/0.28) relative to other AI services, and the general expense associated with AI extensions valued around $12 per month.
- **API Credit Economy**: `@general3d` shared, also in the **general** channel, their experience of economically running a Discord bot using a $5 monthly API credit, emphasizing the affordability when hosting locally.
- **Gemini Pro Vs. Premium AI Competitors**: The performance of **Gemini Pro** compared to premium models such as **GPT-4** was discussed by `@luke_____________` in **general**, with a look ahead to the potential offered by the upcoming Gemini Ultra.
- **Challenges with API Utilization and Summarization**: Over in the **pplx-api** channel, users encountered difficulties in tasks like creating an ongoing conversation tracking shortcut, replicating summarization capabilities, and matching the API key format of Perplexity with that of OpenAI's for broader tool compatibility.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Webinar on LLM Handling of Tabular Data**: Upcoming webinar to focus on *tabular data understanding with LLMs*, examining **Chain-of-Table** method and enhancement of LLMs' performance using multiple reasoning pathways. Register for the Friday 9am PT session [here](https://lu.ma/1cq5hvi4) and delve into papers like "[Chain-of-Table](https://arxiv.org/abs/2401.04398v1)" and "[Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)".

- **Leveraging RAG for Enterprise and Research**: `@seldo` to discuss **language models** and **RAG** for enterprises, with resources like [self-RAG evolution](https://t.co/na6n0kw2kX), [Mistral's RAG documentation](https://t.co/LVMTG0YJ43), and [webinar info](https://t.co/1yo21Z5QDN). A **simple GitHub repo** for RAG beginners can be accessed [here](https://github.com/jotarretx/RAG_Tester).

- **Technical Queries from General Discussions**: Resolving PDF parsing using `ServiceContext.from_defaults`, addressing labeling limitations in Neo4j, improving efficiency in node content extraction, clarifying on documents versus nodes in `VectorStoreIndex`, and troubleshooting SQL query synthesis with LlamaIndex.

- **Hacker News and Medium Reveal SQL and RAG Insights**: Discussions pivot around seeking reliable NL to SQL solutions highlighted by a Hacker News thread [found here](https://news.ycombinator.com/item?id=39261486) and a Medium article on Self-Chunking with RAG and LlamaIndex [located here](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484), concerning accuracy challenges and the future of document analysis.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **ChromaDB Speeds Up RAG Systems**: `@bwo_28` inquired about optimizing the performance of a RAG system, and `@david1542` recommended using **ChromaDB's persistent client** to speed up similarity searches by saving embeddings to disk, potentially improving load times by avoiding the need to recreate embeddings ([ChromaDB documentation](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client)).

- **Langchain Strides Forward**: There's a buzz around LangChain's integration and functionalities. Discussions involved the request for guidance using LangChain with **Mistral** on **AWS SageMaker**, queries on setting "response_format" to "vtt" for audio files using `OpenAIWhisperAudio`, and troubleshooting a `ModuleNotFoundError` involving `SQLDatabase` import from `langchain`.

- **Langserve's Robust Updates and Fixes**: New event stream API agent examples were updated by `@veryboldbagel`, complete with detailed comments, available at [GitHub](https://github.com/langchain-ai/langserve/tree/main/examples/agent), while `@albertperez.` reported a self-resolving deployment loop issue with LangServe.

- **Demand for Personal AI Work Coach**: `@bartst.` is looking to create a personal AI work coach, leading to a discussion where `@david1542` showed interest in contributing ideas for such an initiative.

- **MLBlocks Unveiled**: `@neil6430` shared an introduction to [MLBlocks](https://mlblocks.com/), a no-code platform that enables building image processing workflows using both AI models and traditional methods, streamlining processes into a single REST API endpoint.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI Assistants Becoming Everyday Heroes**: `@ashpreetbedi` supported the practicality of AI personal assistants in tasks like **summarizing daily standups**, demonstrating their increasing integration into workplace routines.
- **Code Automation via AI**: `@slono` shared an automated programming approach with a [GitHub Gist](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769), revealing the potential of AI assistants like 'Aider' in streamlining development processes.
- **RPA's AI Revolution**: `@pennepitstop` sparked discussions on AI's transformative role in Robotic Process Automation (RPA), citing **Adept** as a notable newcomer challenging giants like **UiPath** in personal automation technologies.
- **Querying the Future with Vector Databases**: The dialogue on production-ready **vector databases with API endpoints** led to a recommendation of **Supabase pgvector** by `@swyxio`, highlighting a trend towards more robust data query tools.
- **Scaling AI Models on the Radar**: An insightful [tweet by Stella Biderman](https://twitter.com/BlancheMinerva/status/1754960269250339286) about AI scale discussed by `@swyxio` resonated with the community, emphasizing developments in models such as **RWKV** and **Mamba**.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **GPU Aspirations and Practicality Blend**: `@timjones1` whimsically showed an interest in setting up a personal computing environment, while `@joseph_en` recommended starting with a single 3090 GPU or a more cost-effective 3060 with 12GB VRAM. Meanwhile, `@vim410` discussed incremental work revealing unexploited hardware features that suggest room for optimizing hardware performance.

- **Tuning and Monitoring for Performance**: `@cudawarped` recommended using Nvidia's `ncu` tool for better benchmarking, and shared an [example command](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.py). `@iron_bound` discussed kernel tuning with [CLTune](https://github.com/CNugteren/CLTune), a tool that may be outdated for modern CUDA. `@smexy3` introduced `gmon`, a tool for simplifying GPU monitoring, providing its [GitHub link](https://github.com/AdamLouly/gmon).

- **Solving Quantization Puzzles in PyTorch**: `@hdcharles_74684` faced challenges with dynamic quantization in torchao and improved performance by adding a dequant epilogue, as seen in this [GitHub commit](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295).

- **GPU Workarounds for Older MacBooks and Cloud Options**: `@boredmgr2005` questioned the feasibility of using a 2015 MacBook Pro for CUDA programming, while `@joseph_en` suggested Google Colab as a free and capable cloud solution for such cases.

- **Jax's Ecosystem Gains Momentum**: `@joseph_en` noted Jax's rising popularity and queried Google's strategic direction with Jax, hinting at competition with TensorFlow within the AI and machine learning community.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **The Quest for Representative Datasets**: Members like `@bjoernp` recommend using the **original SFT dataset** for model training with an interest shown in benchmarks using the same and augmented resources like the German part of multilingual c4, German Wikipedia, and *malteos wechsel_de* for perplexity testing.

- **Avoid the Memory Pit**: `@philipmay` reports Out of Memory (OOM) issues with the **Axolotl** model, hinting at possible misconfiguration with Deepspeed's `stage3_gather_16bit_weights_on_model_save` setting, which might not allow the model to fit on a single GPU.

- **Jina Embeddings Fall Short**: Users like `@sebastian.bodza` criticize **Jina embeddings** for underperforming, especially when dealing with Out-of-Distribution (OOD) coding documentation; `@rasdani` echoes these sentiments with evident disappointment.

- **German Inference Pricing Models**: `@lightningralf` introduces a proposed two-tiered price model for German inference services, triggering discussions about potential free services with corporate sponsorship. 

- **Self-Reliant Server Management to Boost Efficiency**: `@flozi00` reveals an in-house endeavor to construct a data center, reflecting a trend towards proprietary server solutions and specialized in-house server management departments.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **Lindy AI Passes Preliminary Tests**: `@thebaghdaddy` found **Lindy AI** capable of performing basic tasks like data retrieval and write-ups, but hints at the possibility of specialized systems for task-specific efficiency.
  
- **Azure's AI Offerings Questioned**: One user, `.psychickoala`, queried if Azure has a **GPT-4 vision model**, but the conversation did not progress with an answer.

- **Super JSON Mode Promises Speed**: `@res6969` introduced **Super JSON Mode** via a [tweet](https://x.com/varunshenoy_/status/1754967233141633513?s=46) from `@varunshenoy_`, claiming a **20x speed increase** in structured output generation for language models without the need for unconventional approaches.
  
- **Optimizing Hosting for MythoMax**: `@jmak` is on the lookout for a more cost-effective hosting solution for deploying **MythoMax** LLM, but lacks community input on the matter.

- **Struggling with PDFs? OCR Them All!**: `@pantsforbirds` is seeking enhancements for processing PDFs, mainly those with poorly encoded text, while `@res6969` advocates for universal OCR application to counteract text extraction issues, despite additional costs.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Seeking Synergy in Silicon**: A member known as `@blankcoo` reached out for potential **collaboration opportunities** with the Alignment Lab project.
- **Enthusiastic Newcomer Alert**: `@craigba` shared his enthusiasm for joining the Alignment Lab AI Discord and offered his expertise in **cybersecurity**, referencing his work at [Threat Prompt](https://threatprompt.com).
- **Ingenious Code Generation Tools**: `@craigba` brought attention to **AlphaCodium**, a tool leveraging adversarial techniques akin to **GANs** to generate high-quality code, inviting others to view [Tamar Friedman's brief introduction](https://twitter.com/itamar_mar/status/1747957348293824676) and explore its [GitHub repository](https://github.com/Codium-ai/AlphaCodium).
- **Appreciation for AI Interview Insights**: Acknowledgment was given to the dialogues in **Jeremy's Latent Space interview**, specifically praising a question surrounding deep learning and productivity outside of big tech, as seen in "[The End of Finetuning](https://www.latent.space/p/fastai)."



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Free Hairstyle App Hunt Turns Fruitless**: `@soundblaster__` is on a quest for a **free app for changing hairstyles**, but is hitting a snag even after scouring the **first and second page of Google** for options that don't require post-registration payment.



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1204334882218713088) (1312 messages🔥🔥🔥): 

- **AI Performance and Training Discussions**: Users discussed various AI models and training techniques, with particular attention paid to models like **Mixtral**, **Nous Mixtral DPO**, and **Miqu 70B**. They compared these to OpenAI's GPT models in terms of efficiency and capability.
- **LLM Community and Resource Sharing**: Conversations touched on **PolyMind**, a project that targets Mixtral Instruct with features like Python interpretation and semantic PDF search. [Its GitHub repository was shared](https://github.com/itsme2417/PolyMind), but there was also mention of a Reddit post about it being removed, indicating possible moderation issues on the `/r/LocalLLaMA` subreddit.
- **Tech Specs and Equipment Discussions**: Users exchanged insights on computing hardware suitable for running large models locally. They debated the memory bandwidth of chips like Apple's M2 and AMD's Epyc, and the practicality of setups with large amounts of RAM for AI inference tasks.
- **Community Dynamics**: The tone of various AI-focused Discord servers was discussed, with opinions on the nature of conversations and community behavior across servers including **TheBloke's server**, **SillyTavern**, and **EleutherAI**.
- **Reddit Moderation and Policy**: There was a critique of ambiguous moderation practices on Reddit, particularly concerning posts on `/r/LocalLLaMA` subreddit and issues with transparency. Users expressed frustration over apparent unequal enforcement of rules when sharing useful AI-related content.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1111984430945402960/1202079366134382633): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [bartowski/dolphin-2.6-mistral-7b-dpo-exl2 · Hugging Face](https://huggingface.co/bartowski/dolphin-2.6-mistral-7b-dpo-exl2): no description found
- [Sfm Soldier GIF - Sfm Soldier Tf2 - Discover &amp; Share GIFs](https://tenor.com/view/sfm-soldier-tf2-meme-american-gif-24728385): Click to view the GIF
- [Oh My God Its Happening GIF - Oh My God Its Happening Ok - Discover &amp; Share GIFs](https://tenor.com/zIIQ.gif): Click to view the GIF
- [Transformer Inference Arithmetic | kipply&#x27;s blog](https://kipp.ly/transformer-inference-arithmetic/): kipply&#x27;s blog about stuff she does or reads about or observes
- [Apple Apple Mac GIF - Apple Apple Mac Apple Mac Studio - Discover &amp; Share GIFs](https://tenor.com/view/apple-apple-mac-apple-mac-studio-apple-mac-studio2022-apple-mac-studio-m1-gif-25082394): Click to view the GIF
- [GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k · Hugging Face](https://huggingface.co/GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k): no description found
- [Qwen-1.5 72B: China&#39;s AI juggernaut DEFEATS Mistral 7B and GPT4! (AI News) 🐉](https://www.youtube.com/watch?v=-oD2JVPD9Nc): The East throws down the gauntlet with Qwen 1.5, a game-changing LLM that shatters boundaries. Not only does it rival ChatGPT4 in math and coding prowess, bu...
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [The Voices GIF - The Voices - Discover &amp; Share GIFs](https://tenor.com/view/the-voices-gif-26307682): Click to view the GIF
- [Fully Uncensored GPT Is Here 🚨 Use With EXTREME Caution](https://www.youtube.com/watch?v=BntGOaMrB90): In this video, we review Wizard Vicuna 30B Uncensored. All censorship has been removed from this LLM. You&#39;ve been asking for this for a while, and now it&#39;s h...
- [Hugging Face – The AI community building the future.](https://huggingface.co/posts): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1akvwdp/we_need_to_talk_about_pol): no description found
- [The Voices Meme GIF - The Voices Meme Cat - Discover &amp; Share GIFs](https://tenor.com/view/the-voices-meme-cat-gif-23917781): Click to view the GIF
- [GitHub - ml-explore/mlx-examples: Examples in the MLX framework](https://github.com/ml-explore/mlx-examples): Examples in the MLX framework. Contribute to ml-explore/mlx-examples development by creating an account on GitHub.
- [abacusai/Smaug-72B-v0.1 · Hugging Face](https://huggingface.co/abacusai/Smaug-72B-v0.1): no description found
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1akgebk/how_i_got_finetuning_mistral7b_to_not_suck/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1akvwdp/we_need_to_talk_about_polymind/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLa): no description found
- [THE DECODER](https://www.google.com/amp/s/the-decoder.com/ccp-releases-politically): Artificial Intelligence is changing the world. THE DECODER brings you all the news about AI.

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1204340730848350218) (503 messages🔥🔥🔥): 

- **Greek Tragedy or Coding Strategy?**: Discord users debated the origin and nature of `Thespis 0.8`, with `@c.gato` clarifying its namesake comes from a Greek tragedy origin and provides a historical context behind the naming. `@billynotreally` and others engaged in a playful confusion of the term, likening it to "sepsis" and a Norwegian word.
- **Detailing DPO Results**: `@dreamgen` requests a public Weights & Biases (wandb) for DPO (Direct Preference Optimization) run metrics. `@c.gato` indicates that accuracy should rise to 100% and margins should increase over time, and provides an example wandb project [link](https://wandb.ai/jondurbin/projects) (this link was part of the original chat and does not lead to a real destination).
- **Lorebooks as Prompt Engineering Tools**: In a conversation about using Lorebooks for enhancing roleplay stories, `@johnrobertsmith` questions their effectiveness beyond prompt engineering. `@mrdragonfox` suggests their utility in injecting information without user prompting, reinforcing important elements during a roleplay session.
- **Merges and Model Trainings Discussed**: Users debate the pros and cons of merging models. `@mrdragonfox` expresses disapproval of merging, while `@mrg` argues that users only care about end results. The discussion points towards a general consensus on the importance of dataset creation over pure model merging.
- **Contemplating Benchmark and Dataset Strategies**: There's a contention regarding the value of merging models for benchmarks versus the value of creating datasets, with `@flail_` suggesting that benchmarks can be distorted by merges and `@mrdragonfox` affirming the real value is in the datasets, not just the merging of models.

**Links mentioned**:

- [Artefact2/BagelMIsteryTour-v2-8x7B-GGUF · Hugging Face](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF): no description found
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2): no description found
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): no description found
- [Objective | docs.ST.app](https://docs.sillytavern.app/extras/extensions/objective/): The Objective extension lets the user specify an Objective for the AI to strive towards during the chat.
- [TheBloke/Beyonder-4x7B-v2-GGUF · Hugging Face](https://huggingface.co/TheBloke/Beyonder-4x7B-v2-GGUF): no description found
- [jondurbin](https://wandb.ai/jondurbin/projects): Weights & Biases, developer tools for machine learning

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1204406569534754847) (11 messages🔥): 

- **Seeking to Sidestep Safety**: User `@mmarkd` inquired about **removing safety guardrails** from Llama2 70B instruct, while sharing a [LessWrong post](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from) discussing how LoRA fine-tuning can undo safety training cost-effectively, but noted it lacked specific details for action.
- **Frustration Over AI's Ethical Restraints**: `@mmarkd` mentioned the **difficulty of using models** like Llama2 70B instruct for tasks due to excessive ethical guardrails, which prevent even non-harmful code refactoring assistance.
- **Recommendations to Reduce Restraint**: `@flail_` suggested using *alternative finetuned models* such as toxicdpo, spicyboros, airoboros, dolphin, and hermes to circumvent overbearing safety features.
- **Tip for Finer Fine-tuning**: `@london` advised that **combining datasets during fine-tuning** improves control over various model parameters.
- **LoRA's Training Attributes Questioned**: `@cogbuji` discussed the extent of change LoRA fine-tuning provides, referencing the QLoRa paper which suggests that applying LoRA to all transformer layers might **match the performance of full fine-tuning**.
- **Seeking Colab Support for LM Usage**: User `@thiagoribeirosnts` is looking for assistance on using **wizardLM or LLaMA 2** on Google Colab after facing difficulties.
- **Contemplating Correct Course for LoRA**: `@gandolphthewicked_87678` pondered whether to continue using **Mistral 7b** for LoRA fine-tuning or switch to a base model, seeking recommendations.

**Links mentioned**:

[LoRA Fine-tuning Efficiently Undoes Safety Training from Llama 2-Chat 70B — LessWrong](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from): Produced as part of the SERI ML Alignment Theory Scholars Program - Summer 2023 Cohort, under the mentorship of Jeffrey Ladish.  …

  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1204422463107112991) (6 messages): 

- **Limitations in Transformers for Layer Offloading**: User `@mmarkd` inquired about offloading specific layers to GPU using Transformers, similar to `-ngl` command in llama.cpp, but `@itsme9316` responded by saying that **Transformers library cannot split layers between CPU and GPU**.

- **Meta's Sphere Project Intrigue**: `@spottyluck` shared a [GitHub link](https://github.com/facebookresearch/Sphere) to Facebook’s **Sphere project**, musing about Meta's shift away from a strategy that was viewed as a challenge to Google. They offered insight on leveraging Sphere for **frequent updates using common crawl and big data tools**.

- **Implementation Queries for LLaMa 2 on MLX**: `@lushboi` is exploring the implementation of **LLaMa 2 on MLX for Apple Silicon** and questioned the structure of the model's query parameters, pondering whether to adapt them to a multi-head or grouped-query format.

**Links mentioned**:

[GitHub - facebookresearch/Sphere: Web-scale retrieval for knowledge-intensive NLP](https://github.com/facebookresearch/Sphere): Web-scale retrieval for knowledge-intensive NLP. Contribute to facebookresearch/Sphere development by creating an account on GitHub.

  

---



### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1204505065105133640) (1 messages): 

- **DALL-E Images get Metadata Upgrade**: `@abdubs` announced that images generated in **ChatGPT** and **OpenAI API** now include metadata following **C2PA specifications**. This enables verification that an image was generated by OpenAI products, helpful for social platforms and content distributors. Read the full details in the [help article](https://help.openai.com/en/articles/8912793-c2pa-in-dall-e-3).
  

---


### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1204337278982561852) (300 messages🔥🔥): 

- **AI Censorship Doubts Deter Usage**: Users `@kotykd` and `@arturmentado` discussed the impact of AI censorship on user experience, with `@kotykd` arguing for user freedom in AI outputs and `@arturmentado` explaining the necessity of safeguarding against misuse. However, `@kotykd` thinks it's overreach, while `@arturmentado` insists there are good reasons for protective measures.
  
- **The Cost of True Artistry in the AI Era**: `@infidelis` and others debated the ethics of misrepresenting AI-generated art as human-made on platforms like Artstation, emphasizing the importance of disclosure. Concerns were raised that platforms suffer when AI art is falsely presented, with `@lugui` underscoring that OpenAI's TOS requires honesty about AI's role in content creation.

- **Metadata's Role in Provenance Mapping**: A significant focus was on the importance and ease of removing metadata from images, with `@whereyamomsat.com` providing resources on EXIF data, and `@heavygee` commenting on file size changes after metadata removal. Many users considered metadata manipulation an irrelevant measure due to ease of alteration.

- **Navigating Open Source AI vs. Commercial Solutions**: Participants like `@infidelis` and `@arturmentado` discussed the superiority of open-source models versus commercial AI, like GPT-4. They considered the impact on innovation progress, with some users forecasting the rise of competitive open-source solutions.

- **The Controversial Convenience of AI Learning Assists**: Users debated the educational applications of AI, with `@germ_storm` and `@chief_executive` discussing its effectiveness for learning and research versus traditional methods, arguing that AI can be a powerful tool for efficient learning in some study areas.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/977259063052234752/1204505065105133640): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [
    Overview - C2PA
  ](https://c2pa.org): no description found
- [Portals - [TouchDesigner + Stable WarpFusion Project Files]](https://www.youtube.com/watch?v=zptPQbTScto): ♫ + 👁 by myself.You can access these TouchDesigner project files [+ its corresponding warpfusion settings], plus many more project files, tutorials and expe...
- [Online photo metadata and EXIF data viewer | Jimpl](https://jimpl.com): View EXIF data of your images online. Find when and where the picture was taken. Remove metadata and location from the photo to protect your privacy.
- [NO C2PA - Remove C2PA Metadata](https://noc2pa.com/): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/midjourney/s/Hr8aYtbcYW): no description found
- [Content Credentials](https://contentcredentials.org/verify…): Introducing the new standard for content authentication. Content Credentials provide deeper transparency into how content was created or edited.

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1204348340935065621) (46 messages🔥): 

- **Troubleshooting Log-In Issues**: User `@mikreyyy` inquired about how to log out of all devices from their account, but no solutions or follow-up comments were provided in the discussion thread.

- **Finding GPT Demonstrations**: User `@glory_72072` asked how to find and use the GPT demo, but the details or responses to guide them were absent from the subsequent messages.

- **Experiencing Enhanced Storytelling**: User `@blckreaper` expressed satisfaction with GPT's improved narrative output, mentioning longer responses that follow directions more accurately, though they reported issues with the AI not fully adhering to custom instructions.

- **Queries About Answer Limits**: `@ytzhak` confronted a possible usage cap after getting blocked following 12 to 20 interactions with GPT, sparking a discussion about limitations on customized GPT usage—an issue `@blckreaper` attributed to a cap on regeneration that counts as one message.

- **Custom GPT Troubles and Tips**: Users reported various challenges and solutions with Custom GPT, from timeout errors (`@realspacekangaroo`) to disappearance of the Explore button (`@hawk8225`)—which `@blckreaper` suggested might be fixed by signing out and back in. Others like `@woodenrobot` discussed instructional conflicts and token limitations within custom instructions, while `@drinkoblog.weebly.com` commented on the cost-effectiveness of different GPT models.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1204475546591825991) (5 messages): 

- **Looking for Audience & Advice on GPTs**: User `@loier` is seeking advice on where to find people interested in their GPTs and wants to learn how to improve module and script setup.
- **Refining Character Interaction in Prompts**: `@_fresnic` suggests incorporating hints within each segment of a character interaction conversation to fine-tune the system's responses better and is open to reviewing others' prompts/conversation flows if provided with a screenshot or gist.
- **Seeking OpenAI Contact for Teams Feedback**: `@novumclassicum` expresses a strong desire to discuss the teams integration with OpenAI and is looking for a way to contact someone from the organization, including Sam Altman, to provide passionate feedback and suggestions for improvements.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1204475546591825991) (5 messages): 

- **Request for Collaboration and Guidance**: User `@loier` is seeking a community to discuss **GPTs usage**, looking for advice on **setting up modules and scripts** to enhance their GPTs' performance.
- **Refining Character Interactions in Prompts**: User `@fresnic` suggests interspersing **hints about character interactions throughout conversation segments** instead of only the initial system prompt and is open to reviewing examples if shared, such as through a **screenshot or gist**.
- **Seeking Direct Access to OpenAI Team**: User `@novumclassicum` expresses a strong desire to discuss **Teams integration** with **OpenAI representatives**, inviting someone like **Sam Altman** to connect for an in-depth conversation.
  

---



### HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1204421684006879252) (1 messages): 

- **Create Your Own Chat Assistant**: A new feature called **Hugging Chat Assistant** allows users to build personal assistants with ease. As described by `@_philschmid`, it includes customizable elements like name, avatar, and behavior controls, and uses different LLMs like **Llama2** or **Mixtral**. The feature is celebrated for eliminating the need to store custom prompts separately. Discover yours at [Hugging Face Chat](https://huggingface.co/chat/assistants).

- **Private Dataset Viewing Now Possible**: `@julien_c` announces an update that makes Dataset Viewer available for private datasets. However, this feature is exclusive to PRO and Enterprise Hub users, offering enhanced tools for data exploration and analysis. Read more from the [data team's work](https://x.com/julien_c/status/1752716726012129577).

- **Synthetic and Croissant Data Tags on the Hub**: In anticipation of synthetic data's growing importance, `@vanstriendaniel` reveals a new `synthetic` tag on the HuggingFace Hub. It's been added to ease the discovery and sharing of synthetic datasets; just include this tag in your dataset card metadata.

- **Showcase Your Blogposts on Your HF Profile**: According to `@not_so_lain`, when HuggingFace users write blog posts, they will now appear on their own profiles. This feature serves as a new way to spotlight individual contributions and insights within the community.

- **Mini Header for Spaces**: `@lunarflu1` introduces a `header: mini` option for HuggingFace Spaces, allowing for full-screen displays with a minimal header, enhancing the user interface and focus on content.

**Links mentioned**:

- [Tweet from Philipp Schmid (@_philschmid)](https://x.com/_philschmid/status/1753429249363452274): Introducing Hugging Chat Assistant! 🤵 Build your own personal Assistant in Hugging Face Chat in 2 clicks! Similar to @OpenAI GPTs, you can now create custom versions of @huggingface  Chat! 🤯  An Ass...
- [HuggingChat - Assistants](https://huggingface.co/chat/assistants): Browse HuggingChat assistants made by the community.
- [Tweet from Julien Chaumond (@julien_c)](https://x.com/julien_c/status/1752716726012129577): NEW on the @huggingface hub:  the Dataset Viewer is now available on private datasets too  You need to be a PRO or a Enterprise Hub user. 🔥  The Dataset Viewer allows teams to understand their data a...
- [Tweet from Daniel van Strien (@vanstriendaniel)](https://x.com/vanstriendaniel/status/1754466661321879814): Synthetic data is going to be massively important in 2024, so we have recently launched a new tag on the @huggingface Hub to facilitate the discovery and sharing of synthetic datasets. To add this tag...
- [Tweet from hafedh (@not_so_lain)](https://x.com/not_so_lain/status/1754302175159701910): @huggingface just found out that when you write a blogpost it shows in your own profile ❤️
- [Tweet from lunarflu (@lunarflu1)](https://x.com/lunarflu1/status/1754800761303683436): There&#39;s a new option for @huggingface Spaces 🤗!  Add `header: mini` in the metadata, and the space will be displayed full-screen with a floating mini header.
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1753643552301617585): Announcing new releases on the weekend is the new norm 🤷🚀  Presenting Diffusers 0.26.0 with two new video models, support for multi IP-adapter inference, and more 📹  Release notes 📜 https://github...
- [Tweet from Sourab Mangrulkar (@sourab_m)](https://x.com/sourab_m/status/1752648062877798867): New Release Alert! 🚨  PEFT v0.8.0 is out now! 🔥🚀✨ Check out the full release notes at https://github.com/huggingface/peft/releases/tag/v0.8.0 [1/9]
- [Tweet from Titus.von.Koeller (@Titus_vK)](https://x.com/Titus_vK/status/1754358165343461704): Exciting news for bitsandbytes! We&#39;re thrilled to announce the release of the initial version of our new documentation! 🧵https://huggingface.co/docs/bitsandbytes/main/en/index
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1753168382017327472): LETS GOO! Faster CodeLlama 70B w/ AWQ & Flash Attention 2⚡  Powered by AutoAWQ, Transformers & @tri_dao&#39;s Flash Attention 2.  GPU VRAM ~40GB 🔥  Want to try it yourself? You&#39;d need to make two...
- [Tweet from Sayak Paul (@RisingSayak)](https://x.com/RisingSayak/status/1754556329887166553): Stable Video Diffusion (SVD) can now be used with 🧨 diffusers, thanks to @multimodalart ❤️  SVD v1.1 👉 https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1  Guide 👉 https://hugg...
- [Constitutional AI with Open LLMs](https://huggingface.co/blog/constitutional_ai): no description found
- [NPHardEval Leaderboard: Unveiling the Reasoning Abilities of Large Language Models through Complexity Classes and Dynamic Updates](https://huggingface.co/blog/leaderboards-on-the-hub-nphardeval): no description found
- [Patch Time Series Transformer in Hugging Face](https://huggingface.co/blog/patchtst): no description found
- [Hugging Face Text Generation Inference available for AWS Inferentia2](https://huggingface.co/blog/text-generation-inference-on-inferentia2): no description found
- [SegMoE: Segmind Mixture of Diffusion Experts](https://huggingface.co/blog/segmoe): no description found
- [Tweet from ai geek (wishesh) ⚡️ (@aigeek__)](https://x.com/aigeek__/status/1753554577490690305): finally a leaderbord that matter the most.  @huggingface&#39;s new Enterprise Scenarios leaderboard just launched.  it evaluates the performance of language models on real-world enterprise use cases. ...
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/879548962464493619/1201999637360148520): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1204338674117115964) (170 messages🔥🔥): 

- **XP Boost Debate**: `@lunarflu` discussed the idea of adding a **multiplier to XP earned** for specific roles in the HuggingFace community, mentioning the `server booster` as a good candidate for this bonus.
- **Docker Inquiry on AI Processing Location**: `@criticaldevx` posted a question about whether **Docker text generation inference** is processed on the user's device or HuggingFace's servers, to which no clear answer was provided in the messages.
- **Error Troubleshooting**: Several users including `@leifer_` and `@criticaldevx` reported experiencing a ***504 error*** on [HuggingFace's chat feature](https://huggingface.co/chat/), suggesting the service might be down.
- **Collaboration and Contribution**: `@lunarflu` expressed willingness to look into **fellowships** for helping elevate users' impact, while `@ufukhury` sought advice on how to contribute to HuggingFace but received no specific steps forward.
- **Accelerator Load State Issues**: `@bit0r` and `@doctorpangloss` engaged in a ***detailed troubleshooting*** conversation about issues restoring a checkpoint using **Accelerate**'s load_state functionality, where using containers like lxd and the specifics of the code were questioned for effectiveness.

**Links mentioned**:

- [LoRA Studio - a Hugging Face Space by enzostvs](https://huggingface.co/spaces/enzostvs/lora-studio): no description found
- [Quick tour](https://huggingface.co/docs/accelerate/quicktour#saveload-entire-states): no description found
- [Rockwell Retro Encabulator](https://youtu.be/RXJKdh1KZ0w?si=9sF3L2f4S2YXCaq8): Latest technology by Rockwell Automation
- [GitHub - HSG-AIML/MaskedSST: Code repository for Scheibenreif, L., Mommert, M., &amp; Borth, D. (2023). Masked Vision Transformers for Hyperspectral Image Classification, In CVPRW EarthVision 2023](https://github.com/HSG-AIML/MaskedSST): Code repository for Scheibenreif, L., Mommert, M., &amp;amp; Borth, D. (2023). Masked Vision Transformers for Hyperspectral Image Classification, In CVPRW EarthVision 2023 - GitHub - HSG-AIML/MaskedSS...
- [GitHub - Sanster/tldream: A tiny little diffusion drawing app](https://github.com/Sanster/tldream): A tiny little diffusion drawing app. Contribute to Sanster/tldream development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1204442473263276103) (7 messages): 

- **Cat Game Research Unleashes Quadruped Fur-Force**: `@technosourceressextraordinaire` shared a model of their cat named Leela created for a video game in development at Cat Game Research for Bat Country Entertainment. The game is said to be pioneering **the first quadruped character controller using ML and AI**, with more details available at [badcatgame.com](https://badcatgame.com).

- **Scholarly Dive into Language Models and Code**: `@vipitis` is engrossed in a comprehensive survey paper that reviews advancements in code processing with language models. The paper covers an extensive range of models, datasets, and over 700 works, with an ongoing update thread on GitHub and references to add to a [HuggingFace collection](https://huggingface.co/collections/Vipitis/code-evaluation-6530478d8e4767ecfe1bc489). The full paper is accessible at [arXiv](https://arxiv.org/abs/2311.07989).

- **No Discord Invites, Please**: A reminder was issued by `@cakiki` to follow the channel's guidelines, which prohibit Discord invites. The rule enforcement was directed at users `@1134164664721350676` and later reiterated for `@985187584684736632`.

- **Alibaba's AI Outperforms Competitors**: `@dreamer1618` highlighted an article stating that Alibaba's latest artificial intelligence model, **Qwen 1.5**, outshone both ChatGPT and Claude in multiple benchmark tests. The article discussing these advancements is available at [wccftech.com](https://wccftech.com/alibabas-latest-a-i-beats-gpt-3-5-claude-in-multple-benchmark-tests/).

- **Innovative Fine-Tuning Paper 'RA-DIT' Explored**: `@austintb.` discussed plans to implement techniques from a promising paper titled "RA-DIT: Retrieval-Augmented Dual Instruction Tuning." The paper proposes advanced methodologies for retriever and language model fine-tuning, with the full document found on [arXiv](https://arxiv.org/abs/2310.01352).

**Links mentioned**:

- [Alibaba&#039;s Latest A.I. Beats GPT-3.5, Claude In Multple Benchmark Tests](https://wccftech.com/alibabas-latest-a-i-beats-gpt-3-5-claude-in-multple-benchmark-tests/): With 2024 marking a strong start to the global artificial intelligence race, Chinese technology giant Alibaba Group has also announced the latest iteration of its Qwen artificial intelligence model. A...
- [Bad Cat Game](https://badcatgame.com):  You&#39;re a cat, and a jerk — an action adventure rpg game currently being developed by Bat Country Entertainment LLC
- [RA-DIT: Retrieval-Augmented Dual Instruction Tuning](https://arxiv.org/abs/2310.01352): Retrieval-augmented language models (RALMs) improve performance by accessing long-tail and up-to-date knowledge from external data stores, but are challenging to build. Existing approaches require eit...
- [Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code](https://arxiv.org/abs/2311.07989): In this work we systematically review the recent advancements in code processing with language models, covering 50+ models, 30+ evaluation tasks, 170+ datasets, and 700+ related works. We break down c...

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1204373522416279563) (10 messages🔥): 

- **Praise for Creation**: `@furquan.sal` showed appreciation for a creation in a simple message stating "*Impressive Bro! Liked it 💛*".
  
- **Inquiries into Frontend Tech**: In response to `@furquan.sal`'s question about the frontend framework used, `@wubs_` elaborated that their project's frontend is built with **React** and uses **Jotai** for state management. They also shared their [development timeline](https://www.artforgelabs.com/post/art-forge-labs-development-timeline-ai-art-innovation), inviting questions and interaction.

- **TensorLM-webui Unveiled**: `@ehristoforu` announced **TensorLM-webui**, a Gradio web UI for **LLM in GGML format** based on **LLaMA**, encouraging users to clone the project from [GitHub](https://github.com/ehristoforu/TensorLM-webui) or test a mini-demo on [Hugging Face Spaces](https://hf.co/spaces/ehristoforu/TensorLM-for-HF).

- **From Sketch to Fashion**: `@tony_assi` presented **Sketch to Fashion Collection**, an application that transforms sketches into fashion designs, available on [Hugging Face Spaces](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection). They later inquired about the possibility of an image generation API.

- **BLOOMChat-v2 Announced**: `@urmish.` shared information about **BLOOMChat-v2**, a 176B parameter multilingual language model with 32K sequence length capability. The model, soon to be complemented with an API, shows significant improvements over earlier models; further details available in a [Twitter summary](https://twitter.com/SambaNovaAI/status/1754928815590277146) and a [detailed blog post](https://sambanova.ai/blog/bloomchat-v2).

**Links mentioned**:

- [Sketch To Fashion Collection - a Hugging Face Space by tonyassi](https://huggingface.co/spaces/tonyassi/sketch-to-fashion-collection): no description found
- [Introducing BLOOMChat 176B - The Multilingual Chat based LLM](https://sambanova.ai/blog/bloomchat-v2): We are proud to release BLOOMChat-v2, a 32K sequence length, 176B multilingual language model.
- [GitHub - ehristoforu/TensorLM-webui: Simple and modern webui for LLM models based LLaMA.](https://github.com/ehristoforu/TensorLM-webui): Simple and modern webui for LLM models based LLaMA. - GitHub - ehristoforu/TensorLM-webui: Simple and modern webui for LLM models based LLaMA.
- [TensorLM - Llama.cpp UI - a Hugging Face Space by ehristoforu](https://hf.co/spaces/ehristoforu/TensorLM-for-HF): no description found
- [Art Forge Labs Development Timeline - AI art innovation](https://www.artforgelabs.com/post/art-forge-labs-development-timeline-ai-art-innovation): Welcome to Art Forge Labs – where our journey in AI art innovation has been as rapid as it has been revolutionary. From our early days of basic setups to leading the charge in AI-driven art, our path ...

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1204419830711717928) (101 messages🔥🔥): 

- **Mamba Presentation Shifted**: `@lunarflu` announced shifting the **Mamba** paper presentation to the following week, inviting others to present something in the meantime. No specific paper or topic has been determined for the current week.
- **Reading Group on the Radar**: `@tonic_1` and a friend expressed interest in presenting a [paper on decoder-only foundation models for time-series forecasting](https://arxiv.org/pdf/2310.10688.pdf) for the next **Reading Group**, coordinating for a Friday presentation and stirring up excitement ("geeking out (hard)").
- **GitHub Repo for Reading Group Resources**: `@chad_in_the_house` created a [GitHub repository](https://github.com/isamu-isozaki/huggingface-reading-group) to compile past presentations and recordings of the **HuggingFace Reading Group** for easy access and potential future YouTube dissemination.
- **S4 and Mamba Discussion Anticipation**: `@ericauld` is preparing to cover **Mamba** and **S4**, seeking input on what aspects the community would find most valuable in a presentation. They suggest focusing on iterations others have made on the papers and potential future developments.
- **Learning Pathways in ML/AI**: Various users shared tips, resources, and starting points for those new to machine learning and AI. Suggested methods include engaging with reading groups, using high-level libraries, working on specific projects, and learning foundational knowledge such as linear algebra.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/879548962464493619/1203285706949009448): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/8795): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Using ML-Agents at Hugging Face](https://huggingface.co/docs/hub/ml-agents): no description found
- [Spaces - Hugging Face](https://huggingface.co/spaces): no description found
- [Civitai | Share your models](https://civitai.com/user/Yamer): no description found
- [GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group](https://github.com/isamu-isozaki/huggingface-reading-group): This repository&amp;#39;s goal is to precompile all past presentations of the Huggingface reading group - GitHub - isamu-isozaki/huggingface-reading-group: This repository&amp;#39;s goal is to precomp...
- [SDXL Unstable Diffusers ヤメールの帝国 ☛ YamerMIX - V11 + RunDiffusion | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/84040/sdxl-unstable-diffusers-yamermix): For business inquires, commercial licensing, custom models/commissions, large scale image captioning for datasets and consultation contact me under...
- [Mobile ALOHA](https://sota.beehiiv.com/p/mobile-aloha?utm_source=sota.beehiiv.com&utm_medium=newsletter&utm_campaign=mobile-aloha): no description found
- [This new AI that will take your job at McDonald&#39;s](https://www.youtube.com/watch?v=HNlS7GyVYK4): Here is a glimpse into the future of robots powered by AI and trained with teleoperated data.check out my leaderboard website at:https://leaderboard.bycloud....

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204341495663169536) (7 messages): 

- **Off-Topic Guidance**: `@juancopi81` directed `@eugenekormin` to move server-side code discussions to the appropriate channel, urging that **diffusion-discussions** is for diffusion model-related topics.

- **Diffusion Newbie Seeks Guidance**: `@_elab`, a newcomer to diffusion models, sought advice on several key parameters, including timestep (`T`), beta scheduling, training pace, and hardware requirements specific to image synthesis using the Stanford Cars dataset.

- **Training Trial and Error**: `@bitpattern` shared a log of training parameters for image generation with **Stable Diffusion**, including batch sizes, gradient accumulation, and optimization steps, while noting the intention to refine the process by possibly reducing the number of images.

- **Diffusion Courses Recommended**: `@juancopi81` recommended the HuggingFace and FastAI courses on diffusion models to `@_elab` for a deeper understanding of diffusion concepts and answers to `_elab`'s questions.

- **Unet2dconditionmodel Query**: `@blankspace1586` was searching for guidance on using the **Unet2dconditionmodel** for validation with a non-textual embedding, as there seemed to be a lack of examples for this scenario.

- **Per-Token Mask in Cross Attention Inquiry**: `@jfischoff` inquired about the possibility of implementing a per-token mask for cross attention in diffusers pipelines, aiming to limit the influence of a token to a specific region of the latent space.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1204343311486427146) (5 messages): 

- **In Search of Domain-Specific Chat**: `@alexkstern` is considering fine-tuning **LLama chat** or a **Mistral model** on a class syllabus using chunks of a textbook to foster model expertise in the subject matter. The goal is to create a model that understands domain-specific content before further fine-tuning it to improve its educational chat capabilities.
- **Beyond Textbooks - Audio as Data**: `@technosourceressextraordinaire` suggests that content likely already exists in the pre-trained data ("the pile") but recommends considering audio-to-text transcriptions, such as those from **Whisper**, for dataset creation.
- **Clarification on Dataset Utility**: `@alexkstern` seeks confirmation on whether using textbook content to fine-tune for domain knowledge, followed by chat-context fine-tuning, is a valid strategy.
- **Frustration with Fine-Tuning Failure**: `@zen_maypole_40488` encountered an `InvalidRequestError` when attempting to fine-tune a model on the OpenAI platform, indicating potential issues with the fine-tuning request URL or API usage.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1204341495663169536) (7 messages): 

- **Diffusion Newbie Seeks Guidance**: `@_elab` asked for advice on setting parameters for a diffusion model for image synthesis using Stanford Cars database, specifically on the best value for `T` (timestep), calculating `betas`, and the hardware requirements for training. They are concerned about the speed of training and whether multiple GPUs are necessary.
  
- **Training Session Underway for Image Synthesis**: `@bitpattern` shared a snapshot of their training log for an image synthesis model utilizing techniques like mixed precision and gradient accumulation. The log indicates the use of a pre-trained model and a resolution of 512 for the dataset.

- **Guided Help for Diffusion Model Learners**: In response to `@_elab`, `@juancopi81` suggested checking out HuggingFace's and FastAI's courses on diffusion models as useful resources for understanding diffusion model parameters and training processes.

- **Inquiry on Unet2dconditionmodel Pipelines**: `@blankspace1586` discussed their success in implementing a training loop for Unet2dconditionmodel with non-textual embeddings as conditioning but expressed uncertainty about the appropriate pipeline to use for validation where they can pass that embedding.

- **Seeking Cross Attention Masks for Pipelines**: `@jfischoff` had a technical query about the possibility of applying a per-token mask for cross attention within diffusers pipelines, aiming to restrict the influence of a token to a specific latent region.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1204347504569618462) (141 messages🔥🔥): 

- **Intel Mac Support for LM Studio**: `@robert.bou.infinite` suggests that while **LM Studio** could theoretically support Intel Macs, the lack of powerful GPUs in older Intel Macs would likely result in poor performance. They advise using remote control of a compatible machine for those stuck on Intel Macs and direct to DM for cloud provider recommendations.

- **TTS and Image Support Questions**: Users are exploring features related to **text-to-speech (TTS)** and **image support**. `@joelthebuilder` is having trouble getting AI voice to work on iOS, `@enragedantelope` asks about filtering models with "vision adapters," and `@lyracon` seeks tips for OCR postprocessing.

- **Model Compatibility and Operations**: `@xermiz.` is guided by `@justmarky` and `@robert.bou.infinite` regarding compatible models for their RTX 3060 and feedback channels are provided by `@robert.bou.infinite` and `@heyitsyorkie` for discussing LM Studio features and bug reporting.

- **Prompting Strategies, Quantization, and Memory Requirements**: `@kujila` discusses building minimal requirement embedded llama apps and `@robert.bou.infinite` talks about challenges with quantizing and running extraordinary large models like **Giant Hydra MOE 240b** which failed to load even on HuggingFace's powerful A100x4 setup.

- **Executing Code and Other Offline AI Tools**: `@artik.ua` inquires about software that executes code and browses the internet like ChatGPT, and `@curiouslycory` suggests tools like **ollama + ollama-webui** that support image input and document chat, indicating that these tools are alternatives to LM Studio for different AI tasks.

**Links mentioned**:

- [Hugging Face – The AI community building the future.](https://huggingface.co): no description found
- [coqui (Coqui.ai)](https://huggingface.co/coqui): no description found
- [ibivibiv/giant-hydra-moe-240b · Hugging Face](https://huggingface.co/ibivibiv/giant-hydra-moe-240b): no description found
- [WHY IS THE STACK SO FAST?](https://www.youtube.com/watch?v=N3o5yHYLviQ): In this video we take a look at the Stack, which sometimes is called Hardware Stack, Call Stack, Program Stack.... Keep in mind that this is made for educati...
- [Making LLMs lighter with AutoGPTQ and transformers](https://huggingface.co/blog/gptq-integration): no description found
- [The unofficial LMStudio FAQ!](https://rentry.org/LMSTudioFAQ): Welcome to the unofficial LMStudio FAQ. Here you will find answers to the most commonly asked questions that we get on the LMStudio Discord. (This FAQ is community managed).  LMStudio is a free closed...
- [Running Assistant LLM(ChatGPT) on your computer locally without internet connection using LM Studio!](https://youtu.be/sLOOLbKM1ys?si=T8H50jrrY8_toqO8): We explore the need behind running large language models based assistants locally, how to run them and their uses cases using LM Studio
- [Extending context size via RoPE scaling · ggerganov/llama.cpp · Discussion #1965](https://github.com/ggerganov/llama.cpp/discussions/1965): Intro This is a discussion about a recently proposed strategy of extending the context size of LLaMA models. The original idea is proposed here: https://kaiokendev.github.io/til#extending-context-t...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1204375720147689483) (73 messages🔥🔥): 

- **Users Grapple with Model Preferences**: `@hexacube` canceled subscriptions to ChatGPT and expressed mixed feelings about alternatives like Guanaco 33b q5. Meanwhile, `@fabguy` shared a positive experience with generating long stories using a 120B model, suggesting the use of big models with certain tactics. `@goldensun3ds` detailed a structured approach to storytelling with specific instructions to the AI, though experienced some teething issues while transitioning to local models and testing extended contexts.
  
- **Fine-Tuning AI Models to Personal Data**: In response to `@goofy_navigator`'s query about training models on personal data, `@heyitsyorkie` indicated that it's possible but cannot be done within LMStudio. Heyitsyorkie further shared a [YouTube tutorial](https://www.youtube.com/watch?v=MDA3LUKNl1E) demonstrating the process of fine-tuning a model with custom datasets.

- **Local Model Troubleshooting**: After a few users, including `@rumpelstilforeskin` and `@joelthebuilder`, reported issues with various models producing suboptimal results or behaving unexpectedly, others recommended checking the version of LMStudio in use or switching to newer models like Dolphins or Nous Hermes.

- **Hardware Conversations and Model Advice**: `@kujila` and `@goldensun3ds` discussed the feasibility and performance of running the Goliath 120B LongLORA model on different systems, with varying results and emphasis on RAM and VRAM requirements. `@heyitsyorkie` advised `@goofy_navigator` on ideal model quant levels for laptop usage and recommended sticking with 7b models for the user's hardware constraints.

- **Finding the Right Fit for Specialized AI Use**: `@supersnow17` was looking for a model fine-tuned for math and physics, to which `@fabguy` suggested that a different kind of tool might be more suitable for solving math problems specifically. `@juanrinta` inquired about AI for reading disorganized text and was pointed towards RAG resources.

**Links mentioned**:

- [Fine-tuning Llama 2 on Your Own Dataset | Train an LLM for Your Use Case with QLoRA on a Single GPU](https://www.youtube.com/watch?v=MDA3LUKNl1E): Full text tutorial (requires MLExpert Pro): https://www.mlexpert.io/prompt-engineering/fine-tuning-llama-2-on-custom-datasetLearn how to fine-tune the Llama ...
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18ljvxb/llm_prompt_format_comparisontest_mixtral_8x7b/): no description found

  

---


### LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1204472573740716053) (1 messages): 

- **Critical Bug Fixes in LM Studio v0.2.14**: `@yagilb` announced important bug fixes in **LM Studio v0.2.14** that address UI freezes when interrupting model generation and hangs caused by pasting long inputs. Users are urged to update through the [LM Studio website](https://lmstudio.ai/) or the app's "Check for updates..." feature.

**Links mentioned**:

[👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/): Find, download, and experiment with local LLMs

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1204335083981504514) (12 messages🔥): 

- **LM Studio Simplifies LLMs for All**: `@drawless111` highlights [LM Studio's ability to run LLMs](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed) with a simple interface that requires no coding, enabling anyone to download and use pre-trained models easily.
- **Alert on LLM Folder Default Reset**: `@msz_mgs` observed that the LLM folder location was reset to default following a recent update, but no other settings were affected.
- **GPU Inactivation to Load Models**: `@georde` reported an issue where models fail to load when attempting to use the GPU, which works upon disabling the GPU.
- **Praise for Improved Text Pasting Speed**: `@msz_mgs` appreciated the increased speed of pasting long text in LM Studio.
- **Suggestions for LM Studio Enhancements**: `@justmarky` suggested new features such as an audible beep when model downloads complete, the ability to favorite release users like TheBloke, and filters for models by size and user. `@fabguy` guided them to post feature requests in a designated channel and informed how to filter models by specific release users.

**Links mentioned**:

[LM Studio: experience the magic of LLMs with Zero technical expertise](https://blog.stackademic.com/lm-studio-experience-the-magic-of-llms-with-zero-technical-expertise-4561039a01ed): Your guide to Zero configuration Local LLMs on any computer.

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1204340234620239912) (27 messages🔥): 

- **Tornado Case Fans vs. Compact Pressure Solution**: `@666siegfried666` discusses optimal cooling solutions for hardware, mentioning the use of a case with **2x180mm fans** generating significant airflow or, alternatively, a smaller case with **3x120mm Arctic P120 fans** for good static pressure and cost-efficiency.
  
- **Exploring AMD 8700g for Dual Local Models**: `@bobzdar` inquires about performance figures when pairing an **AMD 8700g with DDR5 RAM** and a 4090 GPU to run language and code models, suggesting 128 GB of RAM to possibly avoid bottlenecks.
  
- **Discussion on APU's Potential for Running Models**: `@ptable` and `@bobzdar` exchange thoughts on whether the APU's system RAM could be a limitation for running models. `@bobzdar` shares that the APU can address 32GB directly and the memory controller's high overclocking potential, with system RAM speeds around **100GB/s**.

- **Skepticism on APU Performance for AI**: `@goldensun3ds` advises taking a cautious approach regarding the performance of APUs for AI tasks, similar to how one would treat ARC GPUs, while `@rugg0064` points out that despite being faster than system RAM, it still falls short compared to VRAM.
  
- **Awaiting Real-world APU Testing Results**: `@bobzdar` decides to test the APU's performance by ordering a setup, ready to opt for a **7950x3d** if the results are unsatisfactory, prompting responses from others like `@quickdive.` who believe the 7950x3d might have been the better option from the start.

**Links mentioned**:

[I Saw W Gus Fring GIF - I Saw W Gus Fring Gus - Discover &amp; Share GIFs](https://tenor.com/view/i-saw-w-gus-fring-gus-gustavo-deleted-gif-25440636): Click to view the GIF

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1204382807263019008) (15 messages🔥): 

- **Ejection Issue Needs a Fix**: User `@goldensun3ds` mentioned a problematic bug where ejecting a model during message processing causes the program to hang indefinitely. They suggested the need for a way to cancel generation without waiting for token output, to prevent having to restart the program.
  
- **Non-AVX2 Beta Falling Behind**: `@mike_50363` pointed out that the non-AVX2 beta version lags by two releases, impacting their ability to use the software on several Sandy Bridge systems with 128GB RAM. `@yagilb` acknowledged the issue and promised to tag them when a new AVX build is released.

- **Persistent Bug on LM Studio MacOS**: `@laurentcrivello` reported a recurring bug over 3-4 versions on MacOS where the server remains reachable after the app window is closed. `@yagilb` recognized the problem and clarified the expected behavior when the red cross is clicked.

- **Optimizing UI for Server Indication**: `@laurentcrivello` explained their preference for fewer active app indications on MacOS, desiring the server to run without multiple app icons. They proposed a top bar icon that varies depending on server activity, and `@wolfspyre` inquired about the specific UI expectations.

- **Shortcut Creation Complaint**: `@jiha` inquired about an option to stop the beta version from creating a desktop shortcut upon each installation, implying it's an undesired behavior.
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1204378574539333632) (1 messages): 

- **Inquiry About Chain-Compatible Models**: User `@eugenekormin` seeks assistance in identifying **small seq2seq LLM models** (around a few billion parameters) that support the **chain and invoke methods** for a Python script using langchain. They requested help or direction to obtain a model list supporting these methods.
  

---


### LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1204505828154023966) (1 messages): 

- **Inquiry About Crew-AI UI**: User `@docorange88` asked if there are any **UI interfaces** or **web interfaces** for CrewAI similar to AutoGen Studio. They expressed that Crew-AI seems to be better and are looking for thoughts on this.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/) (1 messages): 

phoenix2574: <@294336444393324545> I'm using Mixtral and it seems to work alright
  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1204397687668215838) (8 messages🔥): 

- **Mistral Trumps Phi2 in Programming**: User `@carsonpoole` observed a significant difference when fine-tuning both **phi2 and Mistral** on the code section of **OpenHermes 2.5**; Mistral notably outperformed phi2 under the same sampling settings.
- **Debate Over GPT-4 Programming Capabilities**: User `@teknium` made a quip about GPT-4's programming performance in light of `@carsonpoole`'s findings, leading to a discussion on the matter.
- **GPT-4's Size Discussed**: In response to the programming performance, `@n8programs` noted that the model in question was only 2 billion parameters, hinting at the limitations due to its size.
- **Expectations Versus Reality for GPT-4 Skill Level**: `@teknium` countered by citing a claim from Microsoft Research that suggested they had achieved a GPT-4 level of skill, setting expectations for the model's performance.
- **An Expression of Disbelief**: User `@Error.PDF` shared a humorous [shocked cat gif](https://tenor.com/view/shocked-shocked-cat-silly-cat-cat-kitten-gif-7414586676150300212) from Tenor, potentially reacting to the discussed performance results.

**Links mentioned**:

[Shocked Shocked Cat GIF - Shocked Shocked cat Silly cat - Discover &amp; Share GIFs](https://tenor.com/view/shocked-shocked-cat-silly-cat-cat-kitten-gif-7414586676150300212): Click to view the GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1204334563212398632) (16 messages🔥): 

- **Sparsetral MoE Launches**: `@dreamgen` introduced **Sparsetral**, a sparse MoE model derived from dense model Mistral, with resources like the [original paper](https://arxiv.org/abs/2401.02731), [original repo](https://github.com/wuhy68/Parameter-Efficient-MoE), and [Sparsetral integration repo](https://github.com/serp-ai/Parameter-Efficient-MoE). They also highlighted forking [unsloth](https://github.com/serp-ai/unsloth) for efficient training, noting **Sparsetral on vLLM** works on hardware like a 4090 with bf16 precision, and shared the model on [Hugging Face](https://huggingface.co/serpdotai/sparsetral-16x7B-v2).
  
- **DeepSeek Sets New Math SOTA**: `.benxh` expressed enthusiasm for **Deepseek**, a tool that has apparently set a new state-of-the-art for math-related benchmarks, by introducing a technique known as DPO and a new way to build datasets.

- **PanGu-$\pi$-1 Tiny Language Model Examined**: `@bozoid` shared a research paper ([link](https://arxiv.org/abs/2402.02791)) focused on optimizing tiny language models like PanGu-$\pi$-1 with 1B parameters, investigating architecture, initialization, and optimization strategies to improve tiny LLMs' performance.

- **EQ-Bench for LLMs**: `@nonameusr` introduced [EQ-Bench](https://eqbench.com/), an Emotional Intelligence Benchmark for Large Language Models, including links to the [GitHub repository](https://github.com/EQ-bench/EQ-Bench) and the [related paper](https://arxiv.org/abs/2312.06281) noting updates to their scoring system.

- **Audio Flamingo Excels in Audio Understanding**: `@2bit3thn` shared details on **Audio Flamingo**, an audio language model excelling in various audio understanding benchmarks, mentioning that the model adapts well to unseen tasks through in-context learning and has strong multi-turn dialogue abilities ([project link](https://audioflamingo.github.io/)).

**Links mentioned**:

- [LLM check](https://rahulschand.github.io/gpu_poor/): no description found
- [EQ-Bench Leaderboard](https://eqbench.com/): no description found
- [Audio Flamingo](https://audioflamingo.github.io/): no description found
- [Rethinking Optimization and Architecture for Tiny Language Models](https://arxiv.org/abs/2402.02791): The power of large language models (LLMs) has been demonstrated through numerous data and computing resources. However, the application of language models on mobile devices is facing huge challenge on...
- [GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!](https://github.com/babycommando/machinascript-for-robots): Build LLM-powered robots in your garage with MachinaScript For Robots! - GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1ajwijf/model_release_sparsetral/): no description found

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1204344002741149716) (198 messages🔥🔥): 

- **Quanting the Miqu Finetune**: `@tsunemoto` shared their quantization of Senku-70B-Full, a finetune of an alleged early Mistral-70B medium model, boasting an EQ-Bench score of 84.89. Queries about whether this is a Finetune on FLAN were answered as an OpenOrca finetune, and you can find it on [HuggingFace](https://huggingface.co/ShinojiResearch/Senku-70B-Full).

- **Exploring Math Performance in LLMs**: `@gabriel_syme` sparked a conversation about the significance of mathematics in evaluating language models. It was suggested that good performance in math may correlate with solid logic and reasoning in (non task specific) LLMs, and a model should be trained on math and programming corpuses.

- **Upcoming Event Hype in San Francisco**: `@teknium` and other users discussed the Ollama AI developer event in SF, with `@coffeebean6887` providing links like [Starter to SF Guide](https://www.startertosf.guide/) and [Cerebral Valley](https://cerebralvalley.ai/) for additional resources and activities in the area. Despite the sizeable capacity, it’s nearing full and advises swift RSVP actions.

- **Mixed Language Conundrums with Mixtral**: `@light4bear` expressed that Mixtral, when instructed in Chinese, responds with a mix of Chinese and English, while inversely, OpenHermes occasionally displays responses in Chinese. The latter, mentioned by `@teknium`, has been added to Cloudflare's AI platform as evidenced on [official Tweets](https://x.com/teknium1/status/1755020133398155269?s=46) by both Cloudflare and Teknium.

- **Questions and Support for Multi-Modal Model Finetuning**: `@babycommando` introduced MachinaScript For Robots, a project aiming to control robots using LLMs, and sought advice on finetuning multi-modal models in the `[channel]` category. They also extended gratitude towards Nous’s work and contributions to the field.

**Links mentioned**:

- [Cerebral Valley](https://cerebralvalley.ai/): A community of founders and builders creating the next generation of technology.
- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878): Aligning language models (LMs) with curated human feedback is critical to control their behaviors in real-world applications. Several recent policy optimization methods, such as DPO and SLiC, serve as...
- [tsunemoto/Senku-70B-Full-GGUF · Hugging Face](https://huggingface.co/tsunemoto/Senku-70B-Full-GGUF): no description found
- [ShinojiResearch/Senku-70B-Full · Hugging Face](https://huggingface.co/ShinojiResearch/Senku-70B-Full): no description found
- [Tweet from Teknium (e/λ) (@Teknium1)](https://x.com/teknium1/status/1755020133398155269?s=46): Cloudflare has added my OpenHermes 2.5 7b to their workers ai platform!  ↘️ Quoting Cloudflare (@Cloudflare)   Over the last few months, the Workers AI team has been hard at work making improvements t...
- [RSVP to Chat (Ro)bots Hackathon 
@ AGI House | Partiful](https://partiful.com/e/d2fCE2WW4MGeUr8pEZLV): Welcome to Robotics x LLMs Hack, where creativity, collaboration, and cutting-edge technology meet. Whether you&#x27;re a seasoned coder or a problem-solving guru, this is your chance to build with th...
- [Tweet from Alice (e/nya) (@Alice_comfy)](https://x.com/Alice_comfy/status/1754965801147490418?s=20): Ok so this is one benchmark, but Senku-70B (leaked mistral finetune) beats GPT-4 in EQ Bench. Not sure how I go about getting this added on the website.   Senku-70B is available here.   https://huggin...
- [Local &amp; open-source AI developer meetup · Luma](https://lu.ma/devs2): The Ollamas and Friends are back for another developer focused meetup! We&#x27;re going to Cerebral Valley @ the San Francisco Ferry Building! Open-source AI demo day  Free catered food &amp;...
- [Tweet from Cloudflare (@Cloudflare)](https://x.com/cloudflare/status/1754958644326604930?s=46): Over the last few months, the Workers AI team has been hard at work making improvements to our AI platform. After adding models like Code Llama, Stable Diffusion, Mistral, today, we’re excited to anno...
- [Starter Guide to SF for Founders](https://www.startertosf.guide/): A kickstarter resource for anyone new to or thinking of moving to San Francisco.

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1204507713783078942) (11 messages🔥): 

- **Seeking Research on Data Generation**: `@bigdatamike` is exploring question answer pair generation for fine-tuning an internal LLM and discussed leveraging strategies from a Microsoft paper. They asked for recommendations on other papers covering high-quality data generation.
  
- **Constraints on Mining Model Architectures**: `@lunarsylph_67003` inquired about the constraints for models supported by `LlamaForCausalLM`, questioning whether using novel architectures was permissible. `@teknium` clarified that Bittensor exclusively allows the use of **Mistral**.

- **Loss Values for Further Finetuning**: `@gabriel_syme` posed a question about typical loss values when finetuning an already finetuned model and whether losses taper off or stay close to the initial finetuning values.

- **New Framework Announced**: `@babycommando` introduced **MachinaScript for Robots**, a framework and language that allows building LLM-powered robots. The framework handles LLM outputs in a JSON-like syntax which is then parsed and executed by robots; the repository can be found at [MachinaScript for Robots on GitHub](https://github.com/babycommando/machinascript-for-robots).

- **How to Finetune Obsidian**: `@babycommando` asked for guidance on finetuning **Obsidian**, including processes, steps, tools (like Lora or Qlora), and system specs. Additionally, they requested information on dataset formats and if anyone could shed light on the finetuning process for their MachinaScript-based robot interaction project.

**Links mentioned**:

- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/html/2402.01878v1): no description found
- [GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!](https://github.com/babycommando/machinascript-for-robots): Build LLM-powered robots in your garage with MachinaScript For Robots! - GitHub - babycommando/machinascript-for-robots: Build LLM-powered robots in your garage with MachinaScript For Robots!

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1204334580543520799) (159 messages🔥🔥): 

- **Vying for VRAM**: The conversation revolves around the hardware requirements for running large models, with `@ethux` suggesting a minimum of **100GB vRAM**, while others, like `@dawn.dusk` and `@i_am_dom`, discuss the adequacy of **Nvidia's 4090** or even **dual 4080s** for a cumulative 32GB vRAM. The context suggests this is for running full fp16 models, likely AI or ML applications.
  
- **Intel Macs vs Apple Silicon Debate**: A heated debate about the relevancy of **Intel-based Macs** compared to the newer **Apple Silicon Macs**, with `@frosty04212` expressing strong opinions about the obsolescence of Intel Macs and advocating for upgrades. Others, like `@firesonwires`, argue for the practical longevity of laptops and against unnecessary upgrades.

- **Hardware Upgrade Philosophy**: A contentious discussion about when to upgrade technology, sparked by comments from `@mrdragonfox` and `@firesonwires`, underscores the varied opinions on the need for the latest hardware. While `@frosty04212` argues that **Apple's silicon** is a significant and final upgrade, others emphasize the continuous evolution of technology and personal financial considerations.

- **AI Model Access and Performance**: Users like `@ethux` and `@mrdragonfox` share resources and insights on using AI models, including **Mistral models** and **Mistral guides**. The conversation touches on topics like MoE based models, API access, **Google's LocalLLM**, and quantized models that run on CPUs.

- **Accelerator Cards and Tech Evolution**: `@mrdragonfox` discusses high-end hardware such as **Groq accelerators** and the **NVIDIA A100's end-of-life announcement**, reflecting on the fast pace of hardware lifecycle and the associated costs for cutting-edge processing power. Some find humor in contemplating the future value of these accelerators on platforms like eBay.

**Links mentioned**:

- [Mistral AI | Open-weight models](https://mistral.ai/): Frontier AI in your hands
- [Transformer Inference Arithmetic | kipply&#x27;s blog](https://kipp.ly/transformer-inference-arithmetic/): kipply&#x27;s blog about stuff she does or reads about or observes
- [Prompting Capabilities | Mistral AI Large Language Models](https://docs.mistral.ai/guides/prompting-capabilities/): When you first start using Mistral models, your first interaction will revolve around prompts. The art of crafting effective prompts is essential for generating desirable responses from Mistral models...
- [New localllm lets you develop gen AI apps locally, without GPUs | Google Cloud Blog](https://cloud.google.com/blog/products/application-development/new-localllm-lets-you-develop-gen-ai-apps-locally-without-gpus?utm_source=twitter&utm_medium=unpaidsoc&utm_campaign=fy24q1-googlecloudtech-blog-ai-in_feed-no-brand-global&utm_content=-&utm_term=-&linkId=9418398&s=09): Want to use open-source LLM models from Hugging Face on your local development environment? With localllm and Cloud Workstations, you can.
- [Exploring the Latency/Throughput &amp; Cost Space for LLM Inference // Timothée Lacroix // LLM 3 Talk 3](https://www.youtube.com/watch?v=mYRqvB1_gRk): // AbstractGetting the right LLM inference stack means choosing the right model for your task, and running it on the right hardware, with proper inference co...
- [Chat with Open Large Language Models](https://chat.lmsys.org): no description found
- [HuggingChat](https://huggingface.co/chat): Making the community's best AI chat models available to everyone.
- [Google Cloud Blog](https://cloud.google.com/blog/products/application-development/new-localll): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1204404525725646859) (47 messages🔥): 

- **Mistral Inference Cost Calculation Queries**: `@kushagra_67246` expressed confusion about calculating the cost per token for the **Mistral-7b model** and noted that their calculations indicated costs substantially higher than those for ChatGPT 3.5, even when using quantization to reduce the memory footprint. They cited pricing from [DeepInfra](https://deepinfra.com/pricing) as a benchmark, looking for ways to lower their own costs.
- **Community Suggestions for Cost Reduction**: Community members suggested various strategies to `@kushagra_67246` for cost-saving, including utilizing serverless platforms like **Runpod**, considering accelerators from **Groq**, and exploring **LlamaCPP** for implementing Mixtral-8x7B. Dedicated hardware and prompt engineering were mentioned as factors that could impact cost and performance.
- **Concerns About Data Sensitivity and Fine-Tuning**: `@kushagra_67246` highlighted the need to maintain control over model hosting due to the sensitivity of input data and the desire to fine-tune models on custom datasets. The cost of privacy and running in-house inferences versus professional hosting services was debated in terms of operational requirements and data protection.
- **Model Loading and Boot Time Experiences Shared**: `@casper_ai` and `.superintendent` shared personal experiences, noting that model boot times are not significantly high, with models sometimes being ready for inference within **2-10 seconds**. `.superintendent` noted recent improvements in boot times.
- **Prompt Engineering for LlamaCPP and Mistral Models**: `@aiman1993` asked the community about the effectiveness of prompt engineering techniques when using **LlamaCPP** for **Mistral-8x7B**, having observed non-standard results from their attempts. The conversation indicated that while techniques may generally apply, specific configurations and updates like **Llama 1.6** for autoAWQ might require adaptation.

**Links mentioned**:

[Mistral 7B is 187x cheaper compared to GPT-4 ](https://www.linkedin.com/pulse/mistral-7b-187x-cheaper-compared-gpt-4-tzejf): Mistral AI 7B model can be a great alternative to GPT 3.5 or 4 models with 187x cheaper in cost. Find calculation inside to find out cost comparison between the models.

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1204344730541101066) (16 messages🔥): 

- **Padding Conundrums in Fine Tuning**: `@ramin2024` expressed confusion about the inconsistent behavior when generating text with padded tokens, even after following tutorials like [Fine-tuning Language Models for Structured Responses with QLoRa](https://youtu.be/OQdp-OeG1as). They questioned the need for multiple pad token definitions across the tokenizer and model configurations.
  
- **Seeking Clarity on Fine Tuning Practices**: `@ramin2024` shared a YouTube [tutorial](https://youtu.be/OQdp-OeG1as) they used for reference on fine-tuning language models, noting its relevance despite the tutorial focusing on Lama2 and their work on Mistral using LlamaTokenizer.

- **Fine-Tuning Platform Recommendations**: `@js06` sought advice on platforms for fine-tuning after encountering an error with Huggingface's Autotrain; `@mrdragonfox` suggested using local training or renting GPUs, as well as considering services like [togetherai](https://togetherai.com/).

- **Mistral Training Questions**: `@xzuyn` asked for guidance on fine-tuning Mistral v0.1 7B beyond 8k tokens without SWA and enquired about the performance of such an approach, mentioning they were aware that the model was pretrained with 8k.

- **Clarification on Model Pretraining**: `@xzuyn` raised a concern about using `s2_attention` with Axolotl and sample packing, and how it relates to Mistral v0.1 7B's pretraining methodology.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1144547040454508606/1144547040928481394/1157033013335576796): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Fine-tuning Language Models for Structured Responses with QLoRa](https://youtu.be/OQdp-OeG1as?si=ZtD9ld9qqF4xaSAT): I cover fine-tuning of language models to return *structured responses*, e.g. to return function calls, json objects or arrays. Lecture notes here: https://c...

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1204681413299929138) (2 messages): 

- **Awaiting the Open Source Release**: `@stdewinter` expressed interest in using a tool, asking how it could be utilized. `@hugoduprez` responded with intentions to **release it open source** once they have time.
  

---


### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1204405187343552594) (1 messages): 

- **Next Office Hour Scheduled**: `@sophiamyang` announced the scheduling of the next office hour, which can be accessed via [this Discord link](https://discord.gg/mistralai?event=1204405056825327677).

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/mistralai?event=1204405056825327677): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1204401703860830239) (114 messages🔥🔥): 

- **Astro's Discord Account Compromised**: `@astropulse` confirmed that they were hacked in a targeted spearfishing attack affecting their Discord account. After thorough purging and recovery actions, the account is secure now, with precautionary advice given by other users like `@vrus0188` and `@nodja` on resetting recovery codes and checking if emails were part of hacked sites using [Have I Been Pwned](https://haveibeenpwned.com/).

- **AI Architecture Trials and Tribulations**: `@mkaic` shared their frustrations and hopes regarding novel AI architecture development, expressing that their current model performs poorly on CIFAR100 but remains optimistic by attributing issues to poor gradient flow.
  
- **Spooky Hacking Tales**: In a twist of humor, `@pseudoterminalx` joked about `@astropulse` being kidnapped and used as a "battery" for compute power, while `@progamergov` recounted their own story of a botnet used for hacking their account.

- **Neuromorphic Networks Under Construction**: `@mkaic` discussed their approach to creating truly sparse neural networks by allowing neurons to change connection points during training, aiming to provide an alternative to the current notions of sparsity in AI models.

- **Underappreciated Projects**: `@SegmentationFault` highlighted [PolyMind on GitHub](https://github.com/itsme2417/PolyMind), a project that merges several advanced AI functionalities into one interface and lamented that such utilitarian projects are often overshadowed by more entertainment-oriented AI applications.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/823813159592001537/823813160075132991/1204474029080182785): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Have I Been Pwned: Check if your email has been compromised in a data breach](https://haveibeenpwned.com/): Have I Been Pwned allows you to search across multiple data breaches to see if your email address or phone number has been compromised.
- [Good Boy Dance GIF - Good Boy Dance - Discover &amp; Share GIFs](https://tenor.com/view/good-boy-dance-gif-25381375): Click to view the GIF
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.
- [Matrix Morpheus GIF - Matrix Morpheus Battery - Discover &amp; Share GIFs](https://tenor.com/bg2bf.gif): Click to view the GIF
- [Astropulse](https://astropulse.co/#retrodiffusionhack): The home site of Astropulse, the developer of Retro Diffusion.

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1204448100803158017) (9 messages🔥): 

- **A New Approach to Consistent Image Generation**: `@thejonasbrothers` introduced [ConsiStory](https://arxiv.org/abs/2402.03286), a novel training-free approach aiming to address the consistency challenge in text-to-image generation. The `ConsiStory` model leverages subject-driven shared attention blocks and correspondence-based feature injection.

- **Boosting Anticipation and Commitments**: `@chad_in_the_house` mentioned a paper generating excitement, [link to paper](https://arxiv.org/abs/2401.14953), with DeepMind researchers examining cognitive abilities like anticipation and commitments in AI.

- **Evening Read Teaser**: User `@twoabove` reacted to the shared papers, suggesting that analyzing them will require a dedicated evening due to their complexity and depth.

- **Lumiere, Lighting Up Video Generation**: `@spirit_from_germany` shared a [YouTube video](https://youtu.be/Pl8BET_K1mc) discussing *Lumiere*, Google Research's model for text-to-video generation that tackles the challenge of global consistency in videos.

- **Tweeting Trending AI Topics**: `@helium__` provided a link to a [tweet by @danlyth](https://twitter.com/danlyth/status/1754823375208280430) related to AI, without additional context about the content.

**Links mentioned**:

- [LiPO: Listwise Preference Optimization through Learning-to-Rank](https://arxiv.org/abs/2402.01878): Aligning language models (LMs) with curated human feedback is critical to control their behaviors in real-world applications. Several recent policy optimization methods, such as DPO and SLiC, serve as...
- [Training-Free Consistent Text-to-Image Generation](https://arxiv.org/abs/2402.03286): Text-to-image models offer a new level of creative flexibility by allowing users to guide the image generation process through natural language. However, using these models to consistently portray the...
- [Learning Universal Predictors](https://arxiv.org/abs/2401.14953): Meta-learning has emerged as a powerful approach to train neural networks to learn new tasks quickly from limited data. Broad exposure to different tasks leads to versatile representations enabling ge...
- [Lumiere: A Space-Time Diffusion Model for Video Generation (Paper Explained)](https://youtu.be/Pl8BET_K1mc): #lumiere #texttovideoai #google LUMIERE by Google Research tackles globally consistent text-to-video generation by extending the U-Net downsampling concept t...

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1204373557392703488) (42 messages🔥): 

- **Morning Cheer with Minimal Interaction**: Pagangpegus just dropped a simple "morning yall" into the chat.

- **GPT-3.5 vs GPT-4 for Code Generation Debated**: `@catboy_slim_` discussed that in generating code for obscure languages, GPT-3.5 seemed to obey complex instructions better than GPT-4, which would more quickly revert to tutorial-like responses. They note that GPT-4 may appear smarter globally, but may show inconsistencies with complex, edge-case prompts.

- **Spambots Infiltrate the Server**: Members, `@.undeleted` and `@random_string_of_character`, conversed about the persistence of spambots in the server, noting that the openness of the server and a visible invite link on the site might be the enablers. `@stellaathena` pointed out that the server prefers accessibility over restrictive moderation, hence accepting the occasional bot appearance.

- **MetaVoice TTS Model Announced with Open Source License**: `@stellaathena` shared [a tweet](https://x.com/reach_vb/status/1754984949654904988?s=46) celebrating the launch of *MetaVoice 1B*, a new text-to-speech model. Users, including `@random_string_of_character`, discussed its performance mentioning an *open demo* and the model’s ability to clone voices and generate emotional speech with mixed results.

- **Model Capability Judgement Through Extensive Use**: In a discussion involving understanding model capabilities, `@rallio.` highlighted that truly knowing a model's effectiveness comes with extensive use, which could be an alternative until standardized testing, like SAT/GRE for LLMs, becomes available. `@alexanderrgriffing` hinted towards a future solution called *gollarkbench™*.

**Links mentioned**:

- [TTS by MetaVoice](https://ttsdemo.themetavoice.xyz/): no description found
- [Tweet from Vaibhav (VB) Srivastav (@reach_vb)](https://x.com/reach_vb/status/1754984949654904988?s=46): Let&#39;s go! MetaVoice 1B 🔉  &gt; 1.2B parameter model. &gt; Trained on 100K hours of data. &gt; Supports zero-shot voice cloning. &gt; Short & long-form synthesis. &gt; Emotional speech. &gt; Best ...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1204401102074810428) (71 messages🔥🔥): 

- **Exploring Extrapolation Research Dilemmas**: `@paganpegasus` sought advice on benchmarks for a length extrapolation/generalisation research using 150M models. Other users like `@ai_waifu` and `@nverix` suggested looking at loss vs sequence length graphs, induction, and using log likelihood of given completions as performance measures.

- **Self-Discover Framework Outshines Traditional Methods**: A new framework named SELF-DISCOVER, introduced in [this paper](https://arxiv.org/abs/2402.03620), was discussed, which significantly improves GPT-4 and PaLM 2 performance on reasoning benchmarks compared to existing techniques, reducing inference compute by up to 40 times.

- **Prompt Optimization Via Intent-based Calibration**: `@elad7318` shared their latest paper on automatic prompt engineering for Large Language Models (LLMs), which includes generating challenging boundary cases to refine prompts iteratively. They pointed to the open-source system AutoPrompt, available on [GitHub](https://github.com/Eladlev/AutoPrompt).

- **Combining Hurst Exponent with BPB for Downstream Performance**: `@random_string_of_character` shared a Twitter post from @ibomohsin discussing the combination of the Hurst exponent with bits per byte (BPB) to better predict downstream performance in language models, an idea further explained in an [arXiv paper](https://arxiv.org/abs/2402.01825).

- **Controversy and Clarifications Surrounding Moirai Paper**: Discussions arose around a paper related to universal time series forecasting models (possibly from Google). Allegations surfaced on Twitter regarding potential misrepresentations and oversight errors in the paper's claims.

**Links mentioned**:

- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620): We introduce SELF-DISCOVER, a general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems that are challenging for typical prompting method...
- [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592#:~:text=Deep%20learning%20for%20time%20series,of%20large%20pre%2Dtrained%20models.): Deep learning for time series forecasting has traditionally operated within a one-model-per-dataset framework, limiting its potential to leverage the game-changing impact of large pre-trained models. ...
- [Hungry Hungry Hippos: Towards Language Modeling with State Space Models](http://arxiv.org/abs/2212.14052): State space models (SSMs) have demonstrated state-of-the-art sequence modeling performance in some modalities, but underperform attention in language modeling. Moreover, despite scaling nearly linearl...
- [Read to Play (R2-Play): Decision Transformer with Multimodal Game Instruction](https://arxiv.org/abs/2402.04154): Developing a generalist agent is a longstanding objective in artificial intelligence. Previous efforts utilizing extensive offline datasets from various tasks demonstrate remarkable performance in mul...
- [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592#:~:text=Deep%20learning%20for%20time%20series,of%20large%20pre%2Dtr): Deep learning for time series forecasting has traditionally operated within a one-model-per-dataset framework, limiting its potential to leverage the game-changing impact of large pre-trained models. ...
- [A decoder-only foundation model for time-series forecasting &#8211; Google Research Blog](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html): no description found
- [Tweet from Valeriy M., PhD, MBA, CQF (@predict_addict)](https://x.com/predict_addict/status/1754134502895460421): A new paper from Google that peddles “foundational model” for time series forecasting is both an example of beginner mistakes coupled with deployment of deceptive “benchmarks.”  In figure 6 the author...
- [Scaling Laws for Downstream Task Performance of Large Language Models](https://arxiv.org/abs/2402.04177): Scaling laws provide important insights that can guide the design of large language models (LLMs). Existing work has primarily focused on studying scaling laws for pretraining (upstream) loss. However...
- [Tweet from Ibrahim Alabdulmohsin | إبراهيم العبدالمحسن (@ibomohsin)](https://x.com/ibomohsin/status/1754912619985604818): So, can we combine H with BPB to predict downstream performance? Yes: take the average H + 1/BPB (we invert BPB so that higher values are better). This simple average predicts downstream performance m...
- [Tweet from Ibrahim Alabdulmohsin | إبراهيم العبدالمحسن (@ibomohsin)](https://x.com/ibomohsin/status/1754912601165775296): How is next-token prediction capable of such intelligent behavior? I’m very excited to share our work, where we study the fractal structure of language. TLDR: thinking of next-token prediction in lang...
- [Intent-based Prompt Calibration: Enhancing prompt optimization with synthetic boundary cases](https://arxiv.org/abs/2402.03099): Prompt engineering is a challenging and important task due to the high sensitivity of Large Language Models (LLMs) to the given prompt and the inherent ambiguity of a textual task instruction. Automat...
- [GitHub - Eladlev/AutoPrompt: A framework for prompt tuning using Intent-based Prompt Calibration](https://github.com/Eladlev/AutoPrompt): A framework for prompt tuning using Intent-based Prompt Calibration  - GitHub - Eladlev/AutoPrompt: A framework for prompt tuning using Intent-based Prompt Calibration
- [Hurst exponent - Wikipedia](https://en.wikipedia.org/wiki/Hurst_exponent): no description found
- [Induction heads - illustrated — LessWrong](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated):  Many thanks to everyone who provided helpful feedback, particularly Aryan Bhatt and Lawrence Chan! …
- [Tweet from Dimitris Papailiopoulos (@DimitrisPapail)](https://x.com/DimitrisPapail/status/1754962834113356231): arxiv drop tonite  &#34;Can Mamba Learn How to Learn?: A Comparative Study on In-Context Learning Tasks&#34;   with all-star set of collaborations from @Krafton_inc @SeoulNatlUni @UMich and @UWMadison
- [Tweet from Dimitris Papailiopoulos (@DimitrisPapail)](https://x.com/DimitrisPapail/status/1754965567004389427): I&#39;ll leave this here for now

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1204511912994410638) (3 messages): 

- **Discussing Infinite Depth Limit**: `@niket` mentioned the relevance of work on the **infinite depth limit** by researchers like Greg Yang and Soufiane Hayou.
- **Expanding on Tensor Programs**: `@niket` referenced **tensor programs**, possibly the sixth iteration, in the context of deep learning and scaling laws.
- **Exploring Loss Landscapes in Scaling Laws**: `@niket` expressed an interest in understanding how the **loss landscape** behaves at these infinite limits, noting a lack of resources and requesting references for existing research in this area.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1204490850713866250) (2 messages): 

- **Discovering Fluent Dreaming for LLMs**: `@tbenthompson` shared a [new research paper](https://arxiv.org/abs/2402.01702) titled “Fluent dreaming for language models,” which presents a method called Evolutionary Prompt Optimization (EPO) for optimizing prompts to maximize activations while maintaining low perplexity. This method is likened to feature visualization in vision models and could be a significant addition to the LLM interpretability toolkit.

- **Seeking Saliency Solutions for LLM Prompts**: `@bingbingan.` is looking for methods to determine the input saliency in LLM prompts, specifically to understand which tokens influence refusal to respond to sensitive queries like "How to build a bomb." They considered Integrated Gradients but shared a [concerning paper](https://arxiv.org/abs/2212.11870) indicating that such feature attribution methods might not be reliable for inferring model behavior.

**Links mentioned**:

- [Fluent dreaming for language models](https://arxiv.org/abs/2402.01702): Feature visualization, also known as &#34;dreaming&#34;, offers insights into vision models by optimizing the inputs to maximize a neuron&#39;s activation or other internal component. However, dreamin...
- [Impossibility Theorems for Feature Attribution](https://arxiv.org/abs/2212.11870): Despite a sea of interpretability methods that can produce plausible explanations, the field has also empirically seen many failure cases of such methods. In light of these results, it remains unclear...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1204354051857326150) (3 messages): 

- **Pre-Meeting Prep Acknowledged**: `@asuglia` expressed gratitude towards `@1072629185346019358` and mentioned plans to review certain materials before an upcoming meeting.
- **Offer to Assist with VLM Integration**: `@hailey_schoelkopf` concluded the discussion by suggesting a reconvening after the ACL deadline and offered help with integrating **VLMs** with loglikelihood-based request types, inviting `@1072629185346019358` to reach out if needed.
- **Short and Sweet Farewell**: `@jbdel.` kept it brief with a simple thanks to `@hailey_schoelkopf`.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1204348713649184789) (21 messages🔥): 

- **GCP A100 Shortage**: `@dangfutures` expressed difficulty in finding **A100 GPUs** on Google Cloud Platform (GCP), indicating a possible shortage or limited availability.
- **LLM Autoeval Benchmark Duration Queries**: `@c.gato` inquired about the benchmark time on **LLM Autoeval** for a 7b model and mentioned a 4-hour wait time for a test on a 4090 GPU.
- **Understanding LLM Autoeval**: `@teknium` provided information that replicating the HF Leaderboard benchmarks takes approximately 12 hours for an MMLU test on a 7b model with a 4090 GPU, and also clarified that they use **lm-eval-harness** rather than LLM Autoeval.
- **Benchmark Timeline Clarification Provided**: After discussing with `@teknium`, `@c.gato` realized their benchmarks might be taking longer than expected, considering **lm-eval-harness** takes about 1-2 hours for a set of benchmarks on a single 4090.
- **Technical Issue Reported**: `@youraveragedev` shared an error message they encountered indicating a deprecated `ServerApp` config and a bad initialization config related to the Jupyter server's root contents directory.

**Links mentioned**:

[GitHub - mlabonne/llm-autoeval: Automatically evaluate your LLMs in Google Colab](https://github.com/mlabonne/llm-autoeval): Automatically evaluate your LLMs in Google Colab. Contribute to mlabonne/llm-autoeval development by creating an account on GitHub.

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1204393097979760707) (26 messages🔥): 

- **Hugging Face Grants Axolotl Bounty**: `@dctanner` revealed an offer from Hugging Face (HF) to provide a bounty and compute credits worth $5000 for developing an **Axolotl Spaces training UI**. The UI should allow users to clone a UI Space into their account, specify training runs, and manage Spaces with required hardware.

- **Frontend and Backend Collaboration Call**: `@dctanner` expressed interest in developing the frontend for the Axolotl Spaces training UI using **Tailwind CSS** and sought assistance for backend development, preferably in Python as it could be more maintainable and adaptable.

- **Shiny vs Gradio for the Training UI**: `@jameshwade` suggested building a prototype using **Shiny** and even provided a [mockup](https://huggingface.co/spaces/jameshwade/axolotl-ui) link for the app concept. However, `@le_mess` countered suggesting **Gradio** might be more suitable, recalling Shiny’s propensity for bloat and complexity.

- **Shiny Team Offers Support**: Amidst the discussion of tooling for the UI, `@gordon_shotwell` from the **Shiny team at Posit** extended an offer to assist with the development.

- **Persistent Volume Mountpoint Issue in Runpod Template**: `@m4ttfl0` suggested changing the mount point for the persistent volume in the runpod template to avoid overwriting the existing Axolotl directory, pointing to a [GitHub pull request](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235) with more details. `@nanobitz` participated in the conversation, noting the issue's recent emergence and possible implications for established workflows.

**Links mentioned**:

- [Shiny](https://shiny.posit.co/): Shiny is a package that makes it easy to create interactive web apps using R and Python.
- [Axolotl Launcher 🚀 - a Hugging Face Space by jameshwade](https://huggingface.co/spaces/jameshwade/axolotl-ui): no description found
- [Fine-tune Flair Models on NER Dataset with 🤗 AutoTrain SpaceRunner](https://huggingface.co/blog/stefan-it/autotrain-flair-mobie#start-fine-tuning-with-🤗-autotrain-spacerunner): no description found
- [Cloud motd by winglian · Pull Request #1235 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235): Description sometimes in runpod, the extra disk gets mounted and it clobbers the axolotl dir. add a motd to help users w a solution

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1204396049964339230) (17 messages🔥): 

- **Multi-GPU Multi-Node Saving Issue**: `@lykon78` reports errors when saving a final model (mistral fft) on a multi-GPU, multi-node configuration. They suspect distributed saving might not be implemented correctly, especially since the error originates from the non-master node.
  
- **No Difference with Identical Configurations**: `@lykon78` confirms that both nodes have the same configuration, folder structure, and identical versions of Python libraries. The command used for running the model includes `torchrun` with DeepSpeed configurations.

- **Distributed Training Challenges**: `@nanobitz` and `@caseus_` discuss potential issues with multi-node setups, acknowledging a lack of extensive testing with `axolotl` and the need to synchronize checkpoint data across nodes.

- **Looking into Transformer Library Fixes**: `@lykon78` investigates a similar issue with multi-node training in the transformers library, referencing a pull request that might contain the fix, but notes the issue persists even with the latest version (4.37.2) which should have addressed it.

- **Code Tweaks Required for Multi-Node Saving**: `@caseus_` suggests that code alterations are necessary to enable model saving on each node by modifying `TrainingArguments`, and `@lykon78` expresses willingness to test any proposed code changes.

**Links mentioned**:

- [Can&#39;t resume from checkpoint for multi-node fine-tuning  · Issue #884 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/884): Please check that this issue hasn&#39;t been reported before. I searched previous Bug Reports didn&#39;t find any similar reports. Expected Behavior I have two nodes and i want to do fine-tuning on th...
- [Fix bug for checkpoint saving on multi node training setting by dumpmemory · Pull Request #28078 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28078/files): What does this PR do?   Fixes # (issue) fix bug on multi node training setting with shared file system Before submitting   This PR fixes a typo or improves the docs (you can dismiss the other check...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1204357412631486485) (9 messages🔥): 

- **Troubleshooting DPO Results**: `@fred_fups` questioned the **DPO** results as being unreliable and sought guidance after using **Qlora** with a Mistral model trained on non-DPO data. `@xzuyn` advised that the learning rate might be too high and recommended lowering it significantly.
- **DPO and Learning Rates**: `@xzuyn` suggested that **DPO** generally requires a much lower learning rate compared to standard fine-tuning and recommended that `@fred_fups` add "another 0 or two" to the learning rate for better results.
- **Alpaca vs ChatML in DPO**: `@fred_fups` inquired about the possibility of using **Alpaca format** with **DPO**, but `@xzuyn` indicated that **ChatML** is the only supported format, though custom formats might be possible.
- **Format Switch Advise**: `@fred_fups` decided to switch to **ChatML** after learning from `@xzuyn` that it's the standard on DPO, despite former success with **Alpaca format** for a singular task on a small dataset.
- **Personal Approach and Assistance**: In lieu of using **ChatML**, `@xzuyn` shared a personal approach, which involves using **Metharme** and "hacking things in until it works".
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1204431673203363840) (7 messages): 

- **Jupyter Log Confusion**: `@caseus_` clarified that Jupyter logs showing up do not impact regular training within the system.
- **Jupyter Notebook Startup Issues**: `@nruaif` mentioned a problem where the Jupyter notebook doesn't start, leading `@caseus_` to suggest that the issue may be due to **axolotl** being in the workspace directory which gets affected by a mounted volume.
- **Deprecated Config Warning and Bad Initialization Error**: `@youraveragedev` experienced an error due to a deprecated ServerApp config and received a critical message saying **"/workspace is outside root contents directory"**.
- **Potential Fix Suggested for Jupyter Issue**: In response to `@youraveragedev`'s error, `@nanobitz` recommended running code from a GitHub pull request that addresses the issue of the extra disk clobbering the axolotl directory in the runpod. The pull request can be found here: [Cloud motd by winglian · Pull Request #1235](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235).
- **Recloning Repo as a Solution**: `@nanobitz` further advised `@youraveragedev` to reclone the repository and run pip install once again as part of the troubleshooting process.

**Links mentioned**:

[Cloud motd by winglian · Pull Request #1235 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1235): Description sometimes in runpod, the extra disk gets mounted and it clobbers the axolotl dir. add a motd to help users w a solution

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1204392073978183730) (48 messages🔥): 

- **Claude Pro Subscription Queries**: `@brewingbrews_92072` inquired whether subscribing to **Claude Pro** is necessary even for minimal usage and considered this before getting the Pro version.
- **AI Services Pricing Discussion**: `@archient` compared the cost-effectiveness of **Perplexity's API** (0.07/0.28) to other AI services, finding it cheaper, and discussed the generally higher price of AI extensions at around $12 per month for limited token usage.
- **Autopilot Usage Debate**: `@stocktown` asked about the frequency of using Autopilot, while `@brknclock1215` advised it's generally on but acknowledged situations where it may be better disabled.
- **API Credit Utilization Sharing**: `@general3d` shared the use of a $5 monthly API credit running a Discord bot for a personal server, emphasizing minimal expense and hosting it locally.
- **Comparison of Gemini Pro with Premium Models**: `@luke_____________` argued that while **Gemini Pro** might outperform other *free* AI chatbots, it doesn't quite match up to premium models like **GPT-4** or **Claude 2.1**, but anticipates the release of Gemini Ultra for a potential shift in performance dynamics.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1191690469592289280): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Perplexity Blog](https://blog.perplexity.ai/faq/how-does-file-upload-work)): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1204383639278985246) (9 messages🔥): 

- **Discovering AI Applications for Business**: `@krisograbek` mentioned learning about methods business leaders can utilize AI.
- **Programming Language Recall**: `@grigAI` shared an experience about quickly recollecting a programming language they used years ago.
- **In Search of Knowledge Threads**: `@ok.alex` asked `@1056184370014191686` and `@912748706640564244` for any valuable threads related to AI in business and programming knowledge recall.
- **Efficiency through Perplexity**: `@glnarayanan` provided feedback on using Perplexity AI for research, highlighting its efficiency by sharing a link to their search about LLP registrations and costs in India: [LLP Registrations and Costs](https://www.perplexity.ai/search/What-is-the-Y6lauTzyTFWrcJ0SKGdctw?s=c).
- **Reminder to Set Threads Public**: `@me.lk` reminded `@1204584104662929488` to ensure that threads are set to public to be accessible, providing instructions on how to adjust the thread's privacy settings.
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1204426483678978128) (9 messages🔥): 

- **GPT-4 API Access Inquiry**: `@erlebach123` inquired if GPT-4 is accessible via the Perplexity API, to which `@icelavaman` responded that it is not, and can only be accessed via OpenAI.
- **Broken iCloud Shortcut Link**: `@the_only_alexander` shared an iCloud shortcut link, which was reported as not found. They also confirmed that no API is needed for their discussed use case.
- **Request for API-Based Conversation Tracking**: `@loyah` asked for a shortcut that can maintain an ongoing conversation using the API, rather than starting a new one with every message, but `@the_only_alexander` indicated they do not have such a shortcut.
- **Challenges Mimicking Perplexity's Summarization**: `@adventurous_lynx_53570` sought advice on replicating the summarization capabilities of the Perplexity Chrome extension using the API and encountered difficulties uploading files through the API.
- **API Key Format Compatibility Issue**: `@juan_sc2` questioned why it's challenging to match the API key format of Perplexity with that of OpenAI's, to facilitate its use in tools already supporting the OpenAI API key.

**Links mentioned**:

[Shortcuts](https://www.icloud.com/shortcuts/ba386a9ff0de41c7b51a40a01f0cd10f): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1204594969537683498) (1 messages): 

- **Delving Deep into Tabular Data with LLMs**: `@jerryjliu0` announced a webinar showcasing advanced techniques in *tabular data understanding with LLMs*, featuring the authors of two significant papers, with notable topics including **Chain-of-Table** and improving model performance through the aggregation of multiple reasoning pathways. The webinar is scheduled for this Friday at 9am PT and interested parties can [register here](https://lu.ma/1cq5hvi4).
- **Exploring Robust Techniques for LLMs**: Participants will learn about overcoming the typical challenges faced with naive approaches in table-based question-answering and gain insight into the research through papers like "[Chain-of-Table](https://arxiv.org/abs/2401.04398v1)" and "[Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702)". The event promises to discuss new methodologies for *robust tabular data reasoning*.
- **LlamaPack Implements Cutting-edge Research**: The advanced techniques from the papers have been translated into practical tools with the *LlamaPack templates*, enabling users to apply these breakthroughs in their own LLM applications. The webinar serves as an opportunity to explore the efficiency and potential of these newly implemented strategies.

**Links mentioned**:

- [Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding](https://arxiv.org/abs/2401.04398v1): Table-based reasoning with large language models (LLMs) is a promising direction to tackle many table understanding tasks, such as table-based question answering and fact verification. Compared with g...
- [Rethinking Tabular Data Understanding with Large Language Models](https://arxiv.org/abs/2312.16702): Large Language Models (LLMs) have shown to be capable of various tasks, yet their capability in interpreting and reasoning over tabular data remains an underexplored area. In this context, this study ...
- [LlamaIndex Webinar: Advanced Tabular Data Understanding with LLMs · Zoom · Luma](https://lu.ma/1cq5hvi4): Using LLMs to do question-answering and understanding over tabular data is challenging. The naive approaches (dump text into prompt, text-to-SQL), oftentimes don&#x27;t work well over...

  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1204449406473211945) (5 messages): 

- **Talk on LLMs and RAG for Enterprises**: Catch `@seldo`'s discussion about **language models** and **retrieval-augmented generation (RAG)** for enterprises today. More details on the talk can be found [here](https://t.co/ddp5VRiby3).
- **Self-RAG Evolution in Retrieval Dynamics**: The **Self-RAG** method by `@AkariAsai` allows a language model to perform dynamic retrieval, utilizing a retrieval token and evaluating whether the retrieval is necessary, thereby improving the model's generation abilities. For further insight, click on [this link](https://t.co/na6n0kw2kX).
- **Mistral's New RAG Documentation**: `@MistralAI` released new documents on RAG. Check out `@sophiamyang`'s contribution or learn to implement RAG with **Mistral and LlamaIndex** in 10 lines of code by accessing [this guide](https://t.co/LVMTG0YJ43).
- **Cookbook Entry for Building Agentic RAG**: Explore how to build **agentic RAG** using Mistral through the entry in their brand-new cookbook, provided [here](https://t.co/g6QWClaKa9).
- **Webinar on Advanced Tabular Data Understanding with LLMs**: Discover how to enhance question-answering and understanding over **tabular data** using LLMs in the upcoming LlamaIndex Webinar. This challenges conventional approaches such as text dumping and text-to-SQL conversions. More information available [here](https://t.co/1yo21Z5QDN).

**Links mentioned**:

- [cookbook/llamaindex_agentic_rag.ipynb at main · mistralai/cookbook](https://t.co/g6QWClaKa9): Contribute to mistralai/cookbook development by creating an account on GitHub.
- [Basic RAG | Mistral AI Large Language Models](https://t.co/LVMTG0YJ43): Retrieval-augmented generation (RAG) is an AI framework that synergizes the capabilities of LLMs and information retrieval systems. It&#x27;s useful to answer questions or generate content leveraging ...

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1204438156628197447) (38 messages🔥): 

- **PDF Parsing Puzzle**: `@nbulkz` sought advice on using SummaryIndex to parse a single PDF for Q&A with LlamaIndex, clarifying a preference over using vector indices because of the need for exact textual matches. They were advised to use `ServiceContext.from_defaults(llm=llm, embed_model=None)` by `@cheesyfishes` for not using embeddings when dynamically associating each query directly to the uploaded PDF.

- **Beginners' RAG Playground Shared**: `@wrapdepollo` uploaded a [simple GitHub repo](https://github.com/jotarretx/RAG_Tester) for anyone beginning with RAG and LlamaIndex, inviting feedback and offering help.

- **Graph Labels Limitation in Neo4j**: `@mikefseaff` inquired about adding additional labels to nodes when using LlamaIndex to generate graphs in Neo4j, and `@cheesyfishes` responded that it is not possible with the current setup, suggesting the consideration of alternatives to Knowledge Graphs (KGs).

- **Slow Node Content Extraction Solved**: `@gkossakowski` experienced slow performance when fetching content from 990 nodes in an index but discovered a more efficient method by iterating directly over the document store values.

- **Clarifying Documents vs. Nodes**: `@bin4ry_d3struct0r` questioned the practical difference between using documents or nodes when working with `VectorStoreIndex`. `@cheesyfishes` clarified that documents are chunked by the parser, whereas nodes are not, implying that nodes are typically pre-chunked.

- **SQL Query Synthesis Struggle**: `@dieghox90` expressed frustration when encountering an error while attempting to synthesize a response from a SQL query using LlamaIndex and a local LLAMA2 model, receiving an error indicating an invalid SQL statement.

**Links mentioned**:

- [mrm8488/longformer-base-4096-finetuned-squadv2 · Hugging Face](https://huggingface.co/mrm8488/longformer-base-4096-finetuned-squadv2): no description found
- [GitHub - jotarretx/RAG_Tester: Simple playground for RAG parameters](https://github.com/jotarretx/RAG_Tester): Simple playground for RAG parameters . Contribute to jotarretx/RAG_Tester development by creating an account on GitHub.

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1204449204399775864) (6 messages): 

- **Seeking the Best NL to SQL Converters**: `@prodicer21` inquired about the best natural language to SQL solutions available that do not produce hallucinations.
- **Hacker News Thread on SQL Solutions**: `@the_xyt` shared a [Hacker News thread](https://news.ycombinator.com/item?id=39261486) discussing an NL to SQL solution with a 76.5% accuracy rate on SQL-Eval, comparing it with GPT-4 and sqlcoder-15b.
- **High Accuracy SQL Solution Spotlight**: `@the_xyt` highlighted finding a solution mentioned in the Hacker News thread that achieved a 93% accuracy rate in NL to SQL conversion.
- **Concerns Over SQL Query Accuracy**: `@the_xyt` expressed concerns that for users not well-versed in SQL, even a system with 90% accuracy could generate harmful queries.
- **RAG Analysis and the LlamaIndex Evolution**: `@andysingal` posted an article discussing the integration of RAG chunking analysis with LlamaIndex, hinting at a significant advancement in document analysis. The article is titled "Self-Chunking Brilliance with RAG Analysis and LlamaIndex Revolution" and can be found on [Medium](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484).

**Links mentioned**:

- [Self-Chunking Brilliance with RAG Analysis and LlamaIndex Revolution](https://medium.com/ai-advances/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484): Ankush k Singal
- [Show HN: Natural-SQL-7B, a strong text-to-SQL model | Hacker News](https://news.ycombinator.com/item?id=39261486): no description found

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1204357371275644938) (15 messages🔥): 

- **Optimizing RAG system performance**: `@bwo_28` asked for advice on how to reduce the time it takes for a similarity search in their RAG system. `@david1542` suggested using **ChromaDB's persistent client** to save embeddings to disk, which can speed up the process by loading data from the disk instead of recreating embeddings each time ([ChromaDB documentation](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client)).

- **Asynchronous Blocking in Agent Creation**: `@ferasawadi` is experiencing blocking issues with an asynchronous agent and is seeking insights on why the code isn't running asynchronously as expected.

- **Langchain Audio File Loading with Custom Parameters**: `@risatoga` inquired how to set the model parameters in Langchain to have the "response_format" to be "vtt" instead of json, particularly when using `OpenAIWhisperAudio`.

- **Langchain Integration with AWS SageMaker**: `@jbower` requested guidance for using Langchain with Mistral hosted on an AWS SageMaker endpoint.

- **ModuleNotFoundError in Langchain Import**: `@metaverxe.` faced a `ModuleNotFoundError` when trying to import `SQLDatabase` from `langchain` and is looking for a solution to this error.

- **Langgraph Use Cases Inquiry**: `@sdfjo` is seeking information on projects or use cases currently using **langgraph**, aiming to understand its applications.

**Links mentioned**:

- [WebVoyager](https://www.youtube.com/watch?v=ylrew7qb8sQ): WebVoyager: Building an End-to-End Web Agent with Large Multimodal ModelsWebVoyager is a new vision-powered web-browsing agent that uses browser screenshots ...
- [🧪 Usage Guide | Chroma](https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client): Select a language

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1204477691391250453) (8 messages🔥): 

- **Successful Curl Achievement**: User `@johnda98` expressed satisfaction by confirming that their curl command execution was successful.
- **Agent Examples Updated with Event Stream API**: `@veryboldbagel` has updated agent examples to demonstrate the new event stream API, providing updated client notebooks and examples on [GitHub](https://github.com/langchain-ai/langserve/tree/main/examples/agent). These examples showcase token by token output and tool calls, as well as how to customize agent streaming with a Runnable Lambda.
- **Praise for Timely Examples**: `@gitmaxd` praised the recently updated examples provided by `@veryboldbagel` as *amazing work*, noting their usefulness for working with the Alpha service and commending the detailed comments in the code.
- **Deployment Loop Issue Appears Solved**: `@albertperez.` encountered a deployment loop issue with LangServe, which kept restarting at step #9. `@gitmaxd` inquired about possible issues with capital letters in the project name, however, `@albertperez.` reported that the issue resolved without intervention.

**Links mentioned**:

- [langserve/examples/agent at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/agent_with_history at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent_with_history): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.
- [langserve/examples/agent_custom_streaming at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/tree/main/examples/agent_custom_streaming): LangServe 🦜️🏓. Contribute to langchain-ai/langserve development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1204348485080584202) (8 messages🔥): 

- **In Search of a Personal AI Work Coach**: User `@bartst.` expressed interest in finding a personal (work) coach AI, prompting `@david1542` to offer to join in brainstorms and contribute ideas, even though he's unavailable for hands-on work.
- **Secure AI Agents Blog Post by rez0**: User `@rez0` shared a link to their blog post about the need for capable and secure AI agents, [calling it a read you'll love](https://josephthacker.com/ai/2024/02/05/secure-ai-agents.html), and emphasizing the **massive potential** for someone to build the mentioned features.
- **MLBlocks: A No-code Image Processing API Builder**: User `@neil6430` introduced [MLBlocks](https://mlblocks.com/), a no-code tool allowing users to build multi-step image processing workflows with AI models and classical functions, all through a **single REST API call**.
- **Crypto Project Seeking Talent**: User `@hinayoka` advertised several **job vacancies** in an exciting crypto project, with roles for Web3 Developer, Game Developer, Web Developer, Moderator, and UI/UX Designer, specifying experience with relevant technologies and strong teamwork skills as requirements.
- **Praise for the No-code Image Processing Project**: User `@djabatt` commended `@neil6430` on the no-code image processing API builder project, calling it **really good** and recognizing the effort put into it.

**Links mentioned**:

- [ML Blocks | Home](https://mlblocks.com/): ML Blocks lets you build AI-powered image generation and analysis workflows, without writing any code.
- [Tweet from Joseph Thacker (@rez0__)](https://x.com/rez0__/status/1754860746070999076?s=20): I wrote about the needs we have for more capable and secure AI agents. I think you&#39;ll love it 😊 https://josephthacker.com/ai/2024/02/05/secure-ai-agents.html

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1204367956813946920) (23 messages🔥): 

- **AI Personal Assistants and Real-world Usage**: `@ashpreetbedi` responded to `@jozexotic`'s skepticism about AI personal assistants by mentioning their utility in specific use cases, such as summarizing daily standups.
- **Programming with AI Helps**: `@slono` shared a [GitHub Gist link](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769) demonstrating an automated approach to programming, which sparked a brief conversation on the potential resemblance to the assistant 'Aider'.
- **AI's Influence on RPA Gaining Attention**: `@pennepitstop` highlighted the growing interest in AI-enabling Robotic Process Automation (RPA) for personal automation, mentioning companies like Adept and their challenge to established players like UiPath.
- **Exploring Vector Databases with API Endpoints**: `@henriqueln7` inquired about vector databases that offer API endpoints for production use, and `@swyxio` suggested trying out Supabase pgvector.
- **Highlighting AI Scaling Insights**: `@swyxio` pointed to a [tweet from Stella Biderman](https://twitter.com/BlancheMinerva/status/1754960269250339286) discussing scale in AI, which received a nod for encapsulating sentiments on models like RWKV and Mamba.

**Links mentioned**:

- [create-notices.md](https://gist.github.com/wesen/a4ca759275f1a2bb2a9d4bf4b4b57769): GitHub Gist: instantly share code, notes, and snippets.
- [Effective Data Augmentation With Diffusion Models [NeurIPS 2023]](https://youtu.be/IKDWOOWzwns?si=kr_ErWuS1RciGvF0): 25 minute talk for DA-Fusion from the Synthetic Data Generation with Generative AI Workshop at NeurIPS 2023.Full Paper: arxiv.org/abs/2302.07944
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind): A multimodal, function calling powered LLM webui.  - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1204337689013256192) (3 messages): 

- **Mining Rig Daydreams for AI**: `@timjones1` expresses a whimsical desire for a mining rig and a garage, hinting at the aspirations to set up a personal computing environment.
- **Start Small, Dream Big on GPUs**: `@joseph_en` recommends starting with a single 3090 GPU and considering expansion to a dual setup. He also suggests a cost-effective option of a 3060 with 12GB VRAM that can run a quantized 13b parameter model for around $250.
- **Strides in Hardware Optimization**: `@vim410` comments on work that seems incremental but acknowledges that it reveals unexploited hardware features, implying that there is room for further performance gains through optimization and hardware tailoring.
  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1204399651034505226) (7 messages): 

- **Better Benchmarking with NVIDIA Tools**: `@cudawarped` suggests using the `ncu` tool from Nvidia for **measuring kernel execution times** without the overhead of launch latency and device-to-host memory copy. They shared how the tool can be used and provided an [example command](https://github.com/cudawarped/cuda_mode_lectures/blob/rgb_to_grey/lecture3/rgb_to_grey.py) for running `ncu`.

- **OpenCL & CUDA Kernel Tuning Discussed**: `@iron_bound` inquires about the utility of [CLTune](https://github.com/CNugteren/CLTune), an automatic kernel tuner for OpenCL and CUDA. _tvi_ responded by highlighting the importance of candidate generation and dynamic shapes handling, and `@cudawarped` pointed out that CLTune may be outdated for modern CUDA versions.

- **Introducing GMON for GPU Monitoring**: `@smexy3` has developed a tool called `gmon`, designed to **simplify GPU monitoring** during training jobs. They provided a link to the [tool's GitHub repository](https://github.com/AdamLouly/gmon) and instructions on how to install and use it, including the capability to generate a GPU memory usage report in HTML format at the end of training.

**Links mentioned**:

- [GitHub - CNugteren/CLTune: CLTune: An automatic OpenCL &amp; CUDA kernel tuner](https://github.com/CNugteren/CLTune): CLTune: An automatic OpenCL &amp; CUDA kernel tuner. Contribute to CNugteren/CLTune development by creating an account on GitHub.
- [GitHub - AdamLouly/gmon: GPU Monitor for python.](https://github.com/AdamLouly/gmon): GPU Monitor for python. Contribute to AdamLouly/gmon development by creating an account on GitHub.
- [GitHub - AdamLouly/gmon: GPU Monitor for python.](https://github.com/adamlouly/gmon.git): GPU Monitor for python. Contribute to AdamLouly/gmon development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1204511384415633439) (1 messages): 

- **Dynamic Quantization Hurdles in TorchAO**: `@hdcharles_74684` encountered issues when trying to implement a performant dynamic quantization in torchao due to an unexpected **order of operation fusions** in Torch Compile. To work around the problem, where pointwise operations were fused before matrix multiplications, preventing optimal fusion with dequantization, they manually added a **dequant epilogue** to the kernel, linking to the relevant [GitHub commit](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295).

**Links mentioned**:

[pytorch/torch/_inductor/kernel/mm.py at main · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/mm.py#L295): Tensors and Dynamic neural networks in Python with strong GPU acceleration - pytorch/pytorch

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1204582112427118612) (4 messages): 

- **Seeking GPU Alternatives for a 2015 MacBook Pro**: User `@boredmgr2005` inquired if it's possible to follow lecture materials on a 2015 MacBook Pro with Intel Iris Graphics, or if renting GPUs from the cloud is the only option.
- **Cloud Resources to the Rescue**: `@joseph_en` recommended trying Google Colab, highlighting its free service and capability to handle the lecture requirements.

**Links mentioned**:

[Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1191300313928433664/1198770808122785912): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1204566165662212126) (2 messages): 

- **Jax Rising Among the Cool Kids**: `@joseph_en` expressed interest in learning **Jax**, mentioning its popularity. They inquired about Google's strategic move in developing Jax as a potential competitor to TensorFlow.
  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1204376620803170365) (6 messages): 

- **SFT Dataset Recommendations**: `@bjoernp` advocates for using the **original SFT dataset** for training as it is most representative of the model's intended tasks.
- **Exploring Multilingual Adaptations**: `@johannhartmann` confirms that they will try using the original SFT dataset for benchmarks and also mentions having used the German part of the multilingual c4, German Wikipedia, and *malteos wechsel_de* for laserRMT perplexity.
- **Axolotl's OOM Woes Post-Training**: `@philipmay` reports that **Axolotl** is running Out of Memory (OOM) even after the completion of training.
- **Deepspeed Configuration Caution**: In response to the OOM issue, `@bjoernp` suggests that it could be a Deepspeed configuration problem, specifically the `stage3_gather_16bit_weights_on_model_save` setting, which requires the model to fit on a single GPU.
  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1204535226814959696) (4 messages): 

- **Jina Embeddings Underperformance**: `@sebastian.bodza` expressed dissatisfaction with **Jina embeddings**, stating they tend to **underdeliver**.
- **Issues with OOD Coding Documentation**: `@sebastian.bodza` also mentioned that for coding documentation, the embeddings are a bit **out-of-distribution (OOD)**.
- **Disappointment Voiced Over Embedding Performance**: `@rasdani` reacted to the issues shared by `@sebastian.bodza` with a sentiment of disappointment: "*hm what a pitty :/*".
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1204362164408811540) (5 messages): 

- **In-House Hardware Ambitions**: `@flozi00` mentioned their employer has constructed their own data center, offering server solutions and managing this via specialized departments.
- **Two-Tiered Inference Proposal**: `@lightningralf` suggested creating two price tiers for German inference services, with one including data as open-source and the other as private inference.
- **Free Service Consideration**: In response to the two-tier pricing model, `@flozi00` noted they are exploring some sort of free service that includes sponsorship for usages.
  

---



### LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1204488573575897088) (2 messages): 

- **Initial Impressions of Lindy AI**: `@thebaghdaddy` shared their personal experience with **Lindy**, stating they have *"limited tested its capabilities"* and find it adequate for tasks such as pulling data and doing write-ups, although they believe a more tailored system could potentially be better for specific tasks.
  

---


### LLM Perf Enthusiasts AI ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (1 messages): 

.psychickoala: does azure have a gpt 4 vision model
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1204487819360206918) (2 messages): 

- **User sourya4 exploring suggestions**: `@sourya4` engages in the chat by acknowledging a received suggestion but inquires whether others have tried it yet, indicating curiosity and seeking feedback from the community.
- **Introducing Super JSON Mode**: `@res6969` shared a tweet from `@varunshenoy_` announcing **Super JSON Mode**, which is a new framework designed for **low latency structured output generation** from LLMs, claiming up to **20x faster** generation from OpenAI and open-source models. The framework negates the need for unconventional methods such as threatening the model or tipping the AI.

**Links mentioned**:

[Tweet from Varun Shenoy (@varunshenoy_)](https://x.com/varunshenoy_/status/1754967233141633513?s=46): Introducing 𝗦𝘂𝗽𝗲𝗿 𝗝𝗦𝗢𝗡 𝗠𝗼𝗱𝗲, a framework for low latency structured output generation from LLMs.  Generate JSON up to 𝟮𝟬𝘅 𝗳𝗮𝘀𝘁𝗲𝗿 from OpenAI and open source models.  ❌ No need to...

  

---


### LLM Perf Enthusiasts AI ▷ #[cost](https://discord.com/channels/1168579740391710851/1169026016887459961/1204693376478482432) (1 messages): 

- **Seeking Better Hosting for MythoMax**: User `@jmak` inquired if anyone utilizes **MythoMax** and is searching for a more efficient hosting service to deploy this LLM to users. No recommendations or follow-up discussions were provided in the available messages.
  

---


### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1204510444333703208) (3 messages): 

- **Seeking Better PDF Handling**: `@pantsforbirds` expressed concerns about handling **PDF documents**, especially dealing with poorly encoded text from selectable documents. They use a basic extraction and generation pipeline with **AWS Textract** for OCR when needed, but are looking for preprocessing or runtime strategies to improve reliability.

- **OCR As A Universal Solution**: `@res6969` announced that their team **OCRs all documents** regardless of whether they are selectable or not. This approach has proven effective in handling edge cases, despite higher costs and was implemented for speed of deployment.
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1204507737048875068) (2 messages): 

- **Networking for Collaboration**: User `@blankcoo` inquired about whom to contact for the purpose of **collaborating** with the project.
- **New Member Greeting**: User `@mosessamuel` greeted everyone in the **general-chat** channel.
  

---


### Alignment Lab AI ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/1204432724086235186) (4 messages): 

- **Craig Joins Alignment Lab**: `@craigba` expressed excitement about joining the **Alignment Lab AI Discord** after hearing **Jeremy's** Latent Space interview. Craig offers his help in anything related to **cybersecurity** and shares his work at [threatprompt.com](https://threatprompt.com).
- **Adversarial Concept in Code Generation**: `@craigba` highlighted **AlphaCodium**, an open-source AI tool that uses adversarial concepts similar to **GANs** for creating high-integrity code. Interested viewers can learn more from [Tamar Friedman's 5-minute video](https://twitter.com/itamar_mar/status/1747957348293824676) and explore AlphaCodium through its [GitHub page](https://github.com/Codium-ai/AlphaCodium).
- **Gratefulness for Deep Learning Insights**: `@craigba` thanked `@fanahova` for asking **Jeremy** a critical question in the "[The End of Finetuning](https://www.latent.space/p/fastai)" interview, appreciating the practical and honest advice shared about how to spend time productively when interested in deep learning without joining large tech companies.

**Links mentioned**:

[Threat Prompt - AI Security](https://threatprompt.com.): no description found

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1204534233020891226) (2 messages): 

- **In Search of a Free Hairstyle App**: `@soundblaster__` inquired about a **free app for changing hairstyles** but reported difficulties finding one that doesn't require payment after registration, despite checking **Google's first and second page results**.
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/) (1 messages): 

._z: @everyone Weekly meeting is beginning now. 😄
  

---




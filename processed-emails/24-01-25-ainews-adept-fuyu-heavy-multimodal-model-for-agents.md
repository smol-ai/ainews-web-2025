---
id: 58f3574c-ce94-4006-b73f-944f6a76e291
title: 'Adept Fuyu-Heavy: Multimodal model for Agents'
date: '2024-01-25T21:30:23.929279Z'
original_slug: ainews-adept-fuyu-heavy-multimodal-model-for
description: >-
  AI Discord summaries highlight recent developments including Adept's
  Fuyu-Heavy multimodal model, model deployment strategies, model merging,
  quantization issues, and discussions on large language models like GPT-4,
  Claude 2, and Gemini Ultra. The content also covers model training insights,
  heterogenous AI architectures, and chatbot potential.
companies:
  - adept
  - huggingface
  - lesswrong
models:
  - fuyu-heavy
  - gpt-4
  - claude-2
  - gemini-ultra
  - mistral-7b
  - yi-34b-200k
  - goliath-120b
  - mistral
topics:
  - multimodal
  - large-language-models
  - model-deployment
  - model-merging
  - quantization
  - fine-tuning
  - instruct-tuning
  - heterogeneous-ai
  - transformers
  - llm-training
  - api-integration
  - digital-agents
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 1/24/2024. We checked **20** guilds, **297** channels, and **3025** messages for you. Estimated reading time saved (at 200wpm): **295 minutes**.

Adept's turn for a [splashy launch](https://www.adept.ai/blog/adept-fuyu-heavy):

 ![image.png](https://assets.buttondown.email/images/db04109d-dc48-4921-bc96-012766913818.png?w=960&fit=max) 

The emphasis seems to be UI understanding, which given Adept's business makes sense as a focus. [The demo video](https://vimeo.com/906055649) shows very good and precise visual QA on 7 screenshots of UIs, but revealed no other part of the Adept product because it was on a gradio interface. Fuyu also  uses DPO, which has suddenly become the presumptive winner of the brief [DPO vs IPO vs KTO wars](https://huggingface.co/blog/pref-tuning). Fuyu-Heavy beats Gemini Pro on the new MMMU benchmark, but it's unclear where GPT4V registers on this (someone run it?)

A couple people [called out](https://twitter.com/teortaxestex/status/1750353889499459746) the side comments on the size of Fuyu-Heavy vs Claude 2 and GPT4-V and Gemini Ultra given those details aren't public, and Adept itself didn't actually even mention their own model size (it's bigger than [Fuyu-8B](https://www.adept.ai/blog/fuyu-8b), that's all we really know). Assuming those frontier models are in the rumored [400B](https://www.lesswrong.com/posts/iQx2eeHKLwgBYdWPZ/retrospective-on-gpt-4-predictions-after-the-release-of-gpt) to [1.7T](https://twitter.com/swyx/status/1671272883379908608?lang=en) param range, being 10-20x smaller puts **Fuyu-Heavy around the 20B-170B lower-upper bounds**.

**In other news**, Mamba was [rejected for ICLR as "not good enough"](https://twitter.com/srush_nlp/status/1750526956452577486). Lol?


---

**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Coder Showdown: User Model vs GPT-4**: A user named `@rombodawg` made the claim that their new coding model outperformed **GPT-4** in coding challenge tests, specifying that parts of **DeepSeek Coder 33B** were integrated into their merge.

- **Optimizing LLM Deployment**: There was significant discussion around deploying large models like **Yi-34B-200K** and **Goliath-120B** with users like `@super.deap` and `@aikitoria` seeking strategies for low cost, fast inference, and fitting into **80GB VRAM setups**.

- **Emerging from Quantization Quandary**: Chatter about quantization revealed mixed feelings; `@keyboardking` reported that **Q8** made models worthless, but `@kquant` presented a case for their effective use and shared an endorsement for **EXL2 quants**.

- **Merging Models Mastery**: Conversations included insights on model merging tactics, with `@alphaatlas1` indicating optimal results when the merged model weights sum up to **1.0 total weight** and noting deficiencies when surpassing a **1.4 threshold**.

- **Questions on Fine-tuning**: Users like `@nigelt11` sought clarity on fine-tuning **Mistral 7B Instruct v0.2** regarding tags and prompt formatting, and a shared **Medium article** provided additional guidance on these practices.



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Exploring AI Chatbot Potential**: There was a discussion about the possibilities of turning AIs into chatbots capable of interacting with APIs. However, no specific products were identified.

- **Mistral Model Discussions**: The utility of longer sequences and larger batch sizes was debated as potentially beneficial for models like **Mistral**. Clarifications on the differences and applications of fine-tuning versus instruct-tuning for models were sought, with instruct tuning typically following fine-tuning. The correct data format for fine-tuning Mistral was also queried, indicating confusion over the instruction-format associated with instruct-tuning.

- **LLM Training Insights**: Using RMS optimization was noted to improve loss results over layernorm in recent work. The anticipation of open-sourcing an implementation if successful was expressed, as well as the intent to evaluate a model's performance soon. 

- **Heterogeneous AI and Model Advancements**: A [LessWrong post](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and) discussed **heterogeneous AI architectures** combining **Transformers** and **Selective SSM (Mamba)**. Capabilities of *DoraemonGPT* in dynamic video understanding, the potential of recurrent LLMs like RWKV for tracking character states, and **Contrastive Preference Optimization (CPO)** in translation for moderate-sized LLMs were mentioned. Also, **Adept Fuyu-Heavy** was introduced as a new multimodal model designed for digital agents, which despite its size, outperforms larger models in certain benchmarks as per an [Adept.ai blog post](https://www.adept.ai/blog/adept-fuyu-heavy).

- **Steering LLMs and Hardware Talk**: Interest in activation steering for language models was shown, as well as debates on the efficiencies of different prompt structuring methods using dynamic templates. GPU capability discussions highlighted the AI model running potential on various hardware like the 1080 ti and 3060.

- **Project Obsidian Gears Up for v2**: An upgrade for **Project Obsidian** to v2 was announced, selecting **stableLM 2 1.6B** as the model of choice. The community responded positively, and a resource for zero-shot generalization in visual document understanding, the **InstructDoc dataset**, was shared ([GitHub - nttmdlab-nlp/InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc)).



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **ChatGPT Creativity Call**: `@abdubs` invites users to share distinctive applications of **ChatGPT** in creative and practical scenarios in a bid to understand the broad benefits the technology provides.

- **GPT-4 Image Prompt Insights & Alternatives**: `@lugui` clarifies that **GPT-4** builds prompts based on image descriptions, not generating new images. For image manipulation, `@xenowhiz` suggests the use of **Code Interpreter** and its modules.

- **AI Podcast Recommendation**: `@fran9000` endorses "[The AI Breakdown](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614)", offering daily news and debates on multiple AI facets.

- **Preferences for Older GPT Versions**: `@lzgodhook13` inquires about reverting to **GPT-3 versions from May to June 2022**, citing better performance on straightforward tasks.

- **GPT-4's Updated Context Window Issues**: Users report a decrease in context window after an update, affecting custom model performance. A bug report is created but no direct link is provided.

- **Prompt Engineering Exchange**: Frustrations with default list outputs in ChatGPT are discussed with strategies to evade them, including negative prompting, while `@brayheart` proposes forming a team for a prompt engineering hackathon.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Pro Perks and Promo Code Pandemonium**: Users discussed the advantages of upgrading to **Perplexity Pro**, highlighting its **relevancy checks** and suggesting use with **GPT-4** for an improved experience. However, some encountered issues with applying discount codes, with solutions ranging from following steps by Redditor **u/ArakDemonBlade** to reaching out to support at **pro@perplexity.ai**. 

- **Comparing Perplexity AI Companion's Capability**: Debate ensued over the **Perplexity AI Companion**'s ability to retain sources for **GPT-4** follow-up questions, with contrasting views on its functionality between the extension and the iOS app.

- **Limits and Quirks with Online LLMs and Google Drive**: User **@mantas_82008** raised a query about increasing the **10 per minute** limit for online **LLM models**, while others shared fixes for accessing large **Google Drive** documents, including download and conversion suggestions, and a discussion of how Perplexity's prompt box could be strategically used.

- **Diving into Perplexity's Design Insights**: A **YouTube video** featuring **Henry Modisett**, Head of Design at Perplexity AI, was shared, outlining the nuances of AI design and job acquisition in the field. Additionally, kudos were given for the utility of Perplexity's **Copilot** feature, complemented with links on how it informs users about global trends.

- **API Insights and Credit Incentives Uncovered**: Enthusiasm about Perplexity's API was noted when **PPLX 70B** was identified as the model equivalent to the **"Experiment" feature** with Copilot, and an explanation about differing responses between browser and API suggested variations in system prompts/sources. A $5.00 credit offer was also mentioned, activated after autoloading at least $2 to one's account.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Switch Hits: From CPU to GPU in AI Ops**: Engineers discussed the process of changing from CPU to GPU usage for AI operations, with a recommendation to check the chat page settings. Linux users, particularly those with older processors lacking AVX2 support, sought alternatives to LM Studio, with the suggestion to compile [llama.cpp](https://github.com/ggerganov/llama.cpp), which supports language model loading.

- **Chasing Tailored Models on HuggingFace**: The members expressed confusion regarding the selection of models on HuggingFace, emphasizing the lack of clear Unique Selling Points (USPs) due to minimal documentation. [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) was recommended to compare model performance.

- **Gearing Up GPU Layers for Enhanced Model Performance**: Discussions in the hardware channel focused on configuring GPU layers to optimize model performance. They evaluated using `n_gpu_layers` and `num_layers` settings for improved processing, despite facing issues like underutilized system RAM and compatibility with non-AVX2 instruction support.

- **RAG and Embeddings API in the AI Limelight**: There was a focus on using Retrieval-Augmented Generation (RAG) and possible workarounds for not having an NVIDIA GPU for using the OpenAI Embeddings API. A code snippet for reading from PDF using RAG and a relevant GitHub repository were shared. Conversations also extended to exploring RAG through a [HuggingFace model](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp) and an explanation of RAG from a [Databricks glossary entry](https://www.databricks.com/glossary/retrieval-augmented-generation-rag).

- **Bridging LM Studio and Open Interpreter**: An attempt to integrate LM Studio inference with memgpt and Open Interpreter was discussed, with a focus on whether memgpt's server can emulate OpenAI's chat and completion call functionalities. This indicates ongoing exploration into interoperable systems within the AI community.

- **Prompting Puzzles and Integration Trials**: Members requested ideas for improved prompting without giving a specific context, and shared on-going challenges in integrating LM Studio memGPT with OpenAI, reflecting a broader interest in cross-compatibility and effective prompting strategies in model development.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **EleutherAI Boosts Open AI Research**: EleutherAI has partnered with the National Science Foundation to launch the National AI Research Resource, aiming to provide increased access to AI resources. They have also made strides in AI research with contributions like GPU grants and the development of the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox), known for its scalability on various high-performance computing platforms.

- **Licensing Landscapes and Legalities for LMs**: Within the community, there are discussions indicating ambiguity surrounding the licensing for GitHub repositories as it pertains to model training, with advice ranging from consulting lawyers to exploring local copyright laws. Alongside this, concerns about CoPilot litigation were met with the acknowledgment that legal proceedings can extend over long durations.

- **The Algorithmic Almanac**: Debates and examinations thrived around topics like the importance of data quality over size evidenced by references to Wavenet data, potential and pitfalls in newer frameworks like [Burn](https://burn.dev/) for Rust, and advanced techniques such as MambaByte [token-free Language Modeling](https://arxiv.org/abs/2401.13660) and **Elastic Weight Consolidation** in continual updates for models in production.

- **Deciphering Deep Learning Directives**: Interpretability discussions highlighted the plotting of **decoder weights in Sparse Autoencoder space** and connected research updates from the Anthropic team, which focused on discoveries like attention superposition and **dictionary learning on MNIST**. A noted typo in a key research update reveals the close attention paid to detail within the community.

- **GPT-NeoX Development Dives Deep**: Conversations circled around technical aspects such as tensor+expert parallelism in model training with confirmations of similarity to DeepSpeed's implementation. An ongoing [DeepSpeed issue](https://github.com/microsoft/DeepSpeed/issues/1987) related to CUDA initialization is also subject to further investigation by community members.

- **Scaling and Special Channels**: A mention of overcoming limitations in the original scaling laws paper alluded to successfully training models at the **1.3B parameter scale**, and a pointer was given to discussions regarding 1b parameter models using a specific channel `<#1129489948710539334>` for in-depth analysis.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **Axolotl v0.4.0 Ready for Deployment**: The OpenAccess AI Collective announced the release of [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/releases/tag/v0.4.0), which introduces support for new models, numerous bug fixes, and a note of appreciation for the 56 contributors and the A16Z grant.

- **Model Training Mysteries and Maladies**: Users discussed challenges and best practices in model training; from ensuring **Mamba** compatibility with **Lora** to strategies for efficiently saving model training steps. One user is having trouble uploading to **Hugging Face**, while another seeks guidance on pretraining **CLIP** models specifically for domain adaptation.

- **Shifting Shader Discussions**: Conversations about **GPU Purchasing Decisions** became a focal point, where users compared the merits of 8 **H100** versus 16 **A100** GPUs, considering factors like VRAM for their hardware setups.

- **Medical Dataset Goldmine and Machine Learning Woes**: A [GitHub repository](https://github.com/abachaa/Existing-Medical-QA-Datasets) with multimodal QA datasets was shared, while users grappled with issues from securing **funding for compute resources** to technicalities of the **alpaca format prompt** for model inference.

- **Curiosity Chill Caused by Serverless**: `@dreamgen` expressed concerns about the reality of cold-start times in serverless deployments for **large models**, especially in light of past challenges with providers **not caching models or docker images**. This highlights a pressing performance issue for practical AI deployment.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **Lumiere's Space-Time Magic**: The **Lumiere model** by Google was introduced as a space-time diffusion model that can generate video from text. It notably features impressive inpainting capabilities, as outlined in [Google's detailed writeup](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/).

- **Transparency Tussle Over Google's AI Code**: Concerns were voiced about Google's hesitation to release AI-related code, which makes replicating their research a challenge for the community.

- **Self-Instructing for Smarter AI**: An innovative approach called **Self-Instruct** was shared, aimed at enhancing large language models through self-generated instructions, possibly improving AI's ability to bootstrap knowledge ([Self-Instruct paper](https://arxiv.org/abs/2212.10560)).

- **Discord Welcomes AI Chatbot Contenders**: An invitation was extended to implement a Discord chatbot leveraging large language models, with code available on [GitHub](https://github.com/jakobdylanc/Discord-LLM-Chatbot).

- **A Stage for AI Scholars**: The **Latent Space** guild is using Discord's new Stage feature to facilitate paper discussions, with a successful turnout of 40 participants for a recent session and plans to discuss the [Pythia paper](https://arxiv.org/abs/2304.01373) next, along with insight from a related [Twitter thread](https://twitter.com/rasbt/status/1734920232173539796).

- **Never Miss an AI Beat**: Members were invited to stay informed of future Latent Space events by signing up [here](https://lu.ma/ls) and subscribing to the calendar.

- **RestGPT: LLMs as RESTful Controllers**: A spotlight was cast on *RestGPT*, a project that explores **LLM-based autonomous agents** controlling real-world applications through **RESTful APIs**, hosted on [GitHub](https://github.com/Yifan-Song793/RestGPT).



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **CUDA Role Unrequited**: `@ziper_rom1` didn't get a reply for a CUDA developer position after a month; community insights imply a likely rejection. The position may already be filled, as implied by a [Twitter post](https://twitter.com/sandeep1337/status/1744399578269352293?t=6Vm_qBKAAgBHLHyVPwK13g&s=19) announcing a new Nvidia hire.

- **Mistral Speed Bumps**: `@duck` shared **Mistral** RAG timing benchmarks that showed 17.05 ms for sample times and 175910.47 ms for eval times using llama.cpp, which are considered slow for the intended use case.

- **Mixtral's Mammoth Memory Usage**: `@l0gr1thm1k` encountered **CUDA memory errors** when deploying Mixtral on NVIDIA T4s, where the 4bit quantized version surpassed the anticipated 24GB memory requirement.

- **Finetuning: Beyond BLEU and ROUGE**: `@bishwa3819` is finetuning Mistral-7B on a Dolly dataset and queried whether **BLEU and ROUGE metrics** are adequate for evaluating language model performance on such specific data.

- **Reddit Copilot Bot Powered by Mistral**: `@hugoduprez` created a **Reddit copilot bot** with Mistral which is notable for its operational speed. A new approach using **A-JEPA neural model** for audio understanding was shared by `@shashank.f1`, showcasing a [YouTube video](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF) on semantic extraction from audio files using the model.

- **Mistral API's Summarization Challenge**: `@nico2412_` faced issues with summarizing web content via URL using the Mistral API, a task that's not directly feasible due to LLMs' lack of internet access. `@duck` suggested an alternative approached detailed in a [GitHub notebook](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb) for summarizing articles.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

**AI Study Courts VFX Artists**: A [survey](https://forms.gle/vexRuFUVHoojnfah7) seeking insights from VFX artists and producers is underway as part of an AI study, soliciting valuable industry input.

**Greener Alexa Alternatives on Your Own Terms**: `@mattbcool` addresses electronic waste by retrofitting Alexa hardware, detailing efforts to build a local, [open-source personal assistant](https://mattcool.tech/posts/you-can-have-an-open-source-personal-assistant-in-2024) using Raspberry Pis.

**CircleCI Powers LLM Automated Testing Course**: **Deep Learning.ai** and **CircleCI** have teamed up to offer a [course](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) on using continuous integration tools to assess LLM applications effectively.

**Breathing Life Into Text with 3DTopia**: Discovered by `@meatfucker`, [3DTopia's GitHub repository](https://github.com/3DTopia/3DTopia) promises to transform text into 3D models promptly with downloadable model weights and code.

**Python Module Marries Steering Vectors with Hugging Face's Transformers**: `@mihai4256` created a Python module that integrates steering vectors with transformers, hinting at more details in a [tweet](https://twitter.com/m_chirculescu/status/1750149970026479720).



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **GoogleAI's **Lumiere** Lights Up AI Chat**: The introduction of **GoogleAI's Lumiere**, a robust video diffusion model capable of text-to-video, image-to-video, stylization, and more, sparks discussions among engineers. However, the lack of open sourcing leads to a mixture of excitement and skepticism, with some highlighting the simplicity behind its methods, while others express concerns about realism in AI-generated videos. ([Read Paper on Lumière](https://arxiv.org/abs/2401.12945)).

- **Data Dominance Drives Debate**: **Google** may have an unmatched data edge for training text-to-video (T2V) models considering their YouTube holdings. It's suspected that Lumiere leverages YouTube's extensive video repository, which includes auto-generated captions and a vast quantity of user comments, providing a considerable dataset for training video multimodal language models.

- **Video Model Showdown**: Discussions compare Google's **Lumiere** with Meta's EMU Video models, emphasizing their capabilities and questioning the naturalness of their AI-generated content. Critiques center on occasional inconsistencies in the generated videos, causing some community members to seek out the best open-source equivalents for video stylization.

- **AI Repository Access Woes**: Technical difficulties are evident as users encounter issues downloading LAION-en-aesthetics captions, with Huggingface disabling downloads becoming a point of concern and debate in the community.

- **AI Rivalries Heat Up**: A Reddit link is circulated that discusses the performance of **RWKV 7B**, which may be rivaling **Mistral 7B** in terms of multilingual support while also being noted for efficient CPU usage and linear runtime. The comparison intrigues enthusiasts who are following the advancements in language model capabilities. ([RWKV vs. Mistral Discussion](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/)).

Please note that while certain usernames were initially cited, they have been omitted from this summary as their direct relevance to the topics is not clarified to be of importance for an AI Engineer audience.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Vanna AI Dazzles with RAG for SQL**: **Vanna AI**, a project by `@zain_hoda`, is gaining attention for its use of **Retrieval Augmented Generation (RAG)** to improve SQL query generation, capable of indexing DDL/table schemas and text. The project has generated buzz on social media. [LlamaIndex Tweet](https://twitter.com/llama_index/status/1750196064660127848)

- **AI Response Optimization Sought**: There's a community interest in refining the response mechanics of AI bots, specifically in making openaiagent's reply process more iterative like the openai assistant, although a consensus on the method was not reached.

- **Efficiency Hunt in AI Tools**: Engineers are seeking efficiency improvements in tools like pandas, weighing options such as a context-aware response synthesizer and pondering the thread-safety of multi-threading the query engine.

- **Host Hunting for LLM Chatbots**: The conversation included tips on using open-source models like LLMs without local hosting, with hints about utilizing services like HuggingFace and Replicate for APIs and fine-tuning.

- **Enriching RAG's Performance**: Discussions about enhancing RAG focused on the potential of a BGE similarity reranker and the suggestion of reranking after RRF (Reciprocal Rank Fusion) for better outcomes.

- **Conversational Memory for Chatbots**: The need for tools to track conversational history, similar to the memory buffer of langchain, was expressed with the possibility of integrating chat memory buffer in chat engines to create conversational bots with memory.

- **Vector Storage Preferences Queried**: An individual voiced a request for community opinions on the best vector store companies, prompting a discussion on preferences but no clear favorite emerged.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

**X/Twitter Account Unblocked**: The **X/Twitter account** has been recovered, and users previously affected have been unblocked. Those still experiencing issues can seek help by posting in the thread.

**Streamline Your Apps with LangChain Streaming API**: LangChain reveals a new **streaming API** to support real-time responsiveness in user applications. Resources include [API documentation](https://python.langchain.com/docs/expression_language/streaming), specific modules for **AgentExecutor** and **LangGraph** ([AgentExecutor docs](https://python.langchain.com/docs/modules/agents/how_to/streaming), [LangGraph Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb)), and a [YouTube tutorial](https://youtube.com/watch?v=ZcEMLz27sL4) on `stream_events`. Feedback and discussions on the feature are welcomed on [GitHub](https://github.com/langchain-ai/langchain/discussions/16175).

**Database Dilemmas & Discussions**: Queries range from determining if LangChain is open source to how to best integrate vector embeddings in a Postgres Database schema, alongside a call to share preferred vector storage solutions. Helpful references include the [PostgreSQL Schemas documentation](https://www.postgresql.org/docs/current/ddl-schemas.html).

**LangServe Learning Curve**: Users in the **LangServe channel** grapple with utilizing **agent_executor** and understanding the capabilities of **LCELs**, some wishing for direct guidance from more experienced members in setting up and expanding tool usage.

**Innovations and Connections in Shared Work**: The launch of [AgentHub](https://www.agenthub.dev/), a platform aimed at combining RPA with AI, is announced along with a blog post on potential productivity gains ([AI and RPA: The Future of Work](https://www.agenthub.dev/blog/robotic_process_automation_with_ai)). Meanwhile, a user calls for collaboration without providing specific context.

**Educate with AI-Oriented Courses**: A **free 9-part AI series** including "Building Multimodal AI Applications with LangChain & the OpenAI API" is available at [DataCamp](https://www.datacamp.com/code-along), and a new **free course** on automated testing of AI applications is offered by CircleCI and Deeplearning.ai ([Automated Testing with LLMOPS](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)).



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **GPT-3.5 Faces Judgment Issues**: `@calytrix` experimented with **GPT-3.5** to assess the importance and sentiment in news stories, finding that it fares better in judging sentiment but has difficulty with importance and is biased. To confront these challenges, it was suggested that a specific evaluation system tailored to the model’s abilities might be more effective.

- **Quality Control in AI Training Data**: When fine-tuning language models, data quality was a concern raised by `@thewindmom`; `@hammadkhan` proposed methods such as eye-balling and heuristic filtering for quality checks. Meanwhile, `@bjoernp` recommended synthetic data creation as a strategy to diminish the need for extensive evaluation.

- **Reference to Synthetic Data Insights Shared**: `@_jp1_` mentioned **jon durbins' Airoboros repo** as a valuable resource for DiscoResearch's synthetic data generation techniques and linked to [jon durbins' Airoboros GitHub](https://github.com/jondurbin/airoboros). The repo provides a customizable implementation related to the self-instruct paper.

- **Trials and Tribulations with DPR**: `@philipmay` reported suboptimal outcomes from testing DPR models and investigated question specificity by summation of top result distances. In addition, `@sebastian.bodza` acknowledged the challenges involved with data referencing in embeddings and indicated that the next phase would involve question generation.

- **DiscoLM German 7B v1 Receives Applause and an Update**: User `@alex_22398` praised the **DiscoLM German 7B v1** for its high-quality language outputs and flagged an issue with extra blank lines. Subsequently, `@_chromix_` confirmed a resolution and considered an update to **GGUF quantization**.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1199620732741746738) (1239 messages🔥🔥🔥): 

- **GPT-4 vs Coder Models**: Users engaged in a discussion where one participant, `@rombodawg`, claimed their new coding model is outperforming GPT-4 in tests, providing comparative results of coding challenges. `@righthandofdoom` asked if the model tested was a new "deepseek 33b one," to which `@rombodawg` replied that the model tested is part of their merge and includes components of DeepSeek Coder 33B.
  
- **Yi-34B-200K Deployment Queries**: `@super.deap` sought advice on deploying the Yi-34B-200K model optimally, aiming for low cost and fast inference. Various suggestions were posed, including the use of vLLM, reducing bits per word (bpw), and utilizing one A100 GPU for inference of reduced context sizes.

- **Exploring Large LLM Deployment Options**: Conversation pivoted around deploying large language models like Goliath-120B, with `@aikitoria` looking for advice to fit the model into an 80GB VRAM setup. Participants discussed different bit precision weights (bpw) and cache sizes to optimize for VRAM limits.

- **Exchanging Modeling Techniques**: The discourse carried on with users exchanging insights on model compression strategies at inference time, with mentions of techniques like LoRA adapters for potentially reducing VRAM requirements.

- **Girlfriend GPT and Its Feasibility**: Started by `@bubblegum.btc`, a discussion highlighted skepticism about how Girlfriend GPT could afford to offer a high volume of messages to its users. Some proposed cost-reducing measures such as caching frequent queries or batching responses.

**Links mentioned**:

- [Oracle Embeds Generative AI Across the Technology Stack to Enable Enterprise AI Adoption at Scale](https://www.oracle.com/news/announcement/oracle-announces-availability-oci-generative-ai-service-2024-01-23/): OCI Generative AI service now generally available with choice of models from Cohere and Meta in the cloud and on-premises.
- [the tiny corp raised $5.1M](https://geohot.github.io/blog/jekyll/update/2023/05/24/the-tiny-corp-raised-5M.html): Here we go again. I started another company. The money is in the bank.
- [deepseek-ai/deepseek-coder-33b-instruct · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct): no description found
- [AGoodChoice.yml](https://drive.google.com/file/d/1Kus0s9fjOApFX1ESKrXeAQNAIeDhJ8-D/view?usp=drive_link): no description found
- [deepseek-ai/deepseek-coder-33b-base · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-33b-base): no description found
- [Cirno Cirno Fumo GIF - Cirno Cirno Fumo Fumo Touhou - Discover &amp; Share GIFs](https://tenor.com/view/cirno-cirno-fumo-fumo-touhou-fumo-plush-fumo-gif-25572816): Click to view the GIF
- [20x Faster as the Beginning: Introducing pgvecto.rs extension written in Rust](https://modelz.ai/blog/pgvecto-rs): A new Postgres vector search extension in Rust, with HNSW algorithm for 20x faster than pgvector. But speed is just the start - pgvecto.rs is architected to easily add new algorithms. We look forward ...
- [meta-llama/LlamaGuard-7b · Hugging Face](https://huggingface.co/meta-llama/LlamaGuard-7b): no description found
- [Meet Your Perfect Match: Discover the AI Girlfriend Experience - GPTGirlfriend](https://www.gptgirlfriend.online/blog/post/the-ai-girlfriend-experience): no description found
- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/): Our AI system surpasses the state-of-the-art approach for geometry problems, advancing AI reasoning in mathematics
- [TypeScript/src/compiler/checker.ts at main · microsoft/TypeScript](https://github.com/microsoft/TypeScript/blob/main/src/compiler/checker.ts): TypeScript is a superset of JavaScript that compiles to clean JavaScript output. - microsoft/TypeScript
- [Tweet from NVIDIA 550.40.07 Beta driver released with fixes for VRR and Wayland](https://www.gamingonlinux.com/2024/01/nvidia-550-40-07-beta-driver-released-with-fixes-for-vrr-and-wayland/): NVIDIA has today launched the 550.40.07 Beta driver for Linux which includes numerous important fixes, so you might want to jump in and give this one a thorough gaming test.
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer#train-on-completions-only): no description found
- [The Best GPUs for Deep Learning in 2023 — An In-depth Analysis](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/): Here, I provide an in-depth analysis of GPUs for deep learning/machine learning and explain what is the best GPU for your use-case and budget.
- [LLMLingua/llmlingua/prompt_compressor.py at main · microsoft/LLMLingua](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py): To speed up LLMs&amp;#39; inference and enhance LLM&amp;#39;s perceive of key information, compress the prompt and KV-Cache, which achieves up to 20x compression with minimal performance loss.  - micr...
- [(Long)LLMLingua | Designing a Language for LLMs via Prompt Compression](https://llmlingua.com/): no description found
- [rombodawg/Everyone-Coder-33b-Base · Hugging Face](https://huggingface.co/rombodawg/Everyone-Coder-33b-Base): no description found
- [TheBloke/Everyone-Coder-33B-Base-GGUF · Hugging Face](https://huggingface.co/TheBloke/Everyone-Coder-33B-Base-GGUF): no description found
- [Build a Retrieval-Augmented Generation Chatbot in 5 Minutes](https://www.youtube.com/watch?v=N_OOfkEWcOk>): In under 5 minutes and with only 100 lines of Python code, Rohan Rao, senior solutions architect at NVIDIA, demos how large language models (LLMs) can be dev...
- [rombo (Luis Jalabert)](https://huggingface.co/rombo): no description found
- [&quot;Legalize Nuclear Bombs&quot; Sound Effect](https://www.youtube.com/watch?v=575jfgf8hXU): no description found
- [llm_steer/demo/llm_steer_demo.ipynb at main · Mihaiii/llm_steer](https://github.com/Mihaiii/llm_steer/blob/main/demo/llm_steer_demo.ipynb): Steer LLM outputs towards a certain topic/subject and enhance response capabilities using activation engineering by adding steering vectors - Mihaiii/llm_steer
- [GitHub - Mihaiii/llm_steer: Steer LLM outputs towards a certain topic/subject and enhance response capabilities using activation engineering by adding steering vectors](https://github.com/Mihaiii/llm_steer): Steer LLM outputs towards a certain topic/subject and enhance response capabilities using activation engineering by adding steering vectors - GitHub - Mihaiii/llm_steer: Steer LLM outputs towards a...
- [GitHub - CHIP-SPV/chipStar: chipStar is a tool for compiling and running HIP/CUDA on SPIR-V via OpenCL or Level Zero APIs.](https://github.com/CHIP-SPV/chipStar): chipStar is a tool for compiling and running HIP/CUDA on SPIR-V via OpenCL or Level Zero APIs. - GitHub - CHIP-SPV/chipStar: chipStar is a tool for compiling and running HIP/CUDA on SPIR-V via Open...
- [AWQ: Up to 2.66x higher throughput by casper-hansen · Pull Request #2566 · vllm-project/vllm](https://github.com/vllm-project/vllm/pull/2566): The strategy is to dequantize and run FP16 matmul for longer sequences. This could probably be faster if we just used cublas instead of torch.matmul. EDIT: It seems throughput can be over 2x in vLL...
- [GitHub - asg017/sqlite-vss: A SQLite extension for efficient vector search, based on Faiss!](https://github.com/asg017/sqlite-vss): A SQLite extension for efficient vector search, based on Faiss! - GitHub - asg017/sqlite-vss: A SQLite extension for efficient vector search, based on Faiss!
- [Perfecting Merge-kit MoE&#39;s](https://docs.google.com/document/d/1_vOftBnrk9NRk5h10UqrfJ5CDih9KBKL61yvrZtVWPE/edit?usp=sharing): no description found
- [MoE Merge Misconceptions - Pastebin.com](https://pastebin.com/Mzfev83p): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector): Summary: We demonstrate a new scalable way of interacting with language models: adding certain activation vectors into forward passes.[2] Essentially, we add together combinations of forward passes in...
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vecto): Summary: We demonstrate a new scalable way of interacting with language models: adding certain activation vectors into forward passes.[2] Essentially, we add together combinations of forward passes in...

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1199624878333100124) (116 messages🔥🔥): 

- **Model Size Concerns Emerge**: `@keyboardking` expressed difficulties in understanding how people utilize models under 10GB, leading to a discussion where `@kquant` clarified that a 7B model is not under 10GB and provided insights about running models at full precision.
- **Thoughts on Model Quantization**: Different opinions surfaced about quantized models with `@keyboardking` mentioning that Q8 ggufs seemed to render the model worthless, while `@kquant` explained the actual sizes of q8 models and their effectiveness.
- **Character Creation Techniques Discussed**: `.justinobserver` shared a prompt designed to help users create character cards for roleplay, leading to questions from community members like `@givan_002` about how to apply the prompt to multi-round dialogues.
- **Feedback on Model Performance**: Users such as `@ks_c` tested models like bagelmistertour and reported mixed outcomes, indicating a need for more context, and `@kquant` endorsed using a `chat-instruct` setup for better stability.
- **Quantization Follow-up by Users**: Community members like `@kquant` suggested LoneStriker’s EXL2 quants for a better experience with models like frankendpo, and `@ks_c` talked about a positive experience when using higher quantization settings on frankendpo 4x7b, mentioning it was stable without tokenizer issues.

**Links mentioned**:

- [FrankenDPO-4x7B-EXL2 - a Kquant03 Collection](https://huggingface.co/collections/Kquant03/frankendpo-4x7b-exl2-65a74855e211a95509e459b7): no description found
- [Kquant03/Raiden-16x3.43B · Hugging Face](https://huggingface.co/Kquant03/Raiden-16x3.43B): no description found
- [Another LLM Roleplay Rankings](https://rentry.co/ALLMRR): (Feel free to send feedback to AliCat (.alicat) and Trappu (.trappu) on Discord) We love roleplay and LLMs and wanted to create a ranking. Both, because benchmarks aren't really geared towards rolepla...
- [GitHub - OFA-Sys/Ditto: A self-ailgnment method for role-play](https://github.com/OFA-Sys/Ditto): A self-ailgnment method for role-play. Contribute to OFA-Sys/Ditto development by creating an account on GitHub.

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1199632544405127188) (67 messages🔥🔥): 

- **Clarifying Fine-tuning Instructions**: `@nigelt11` asked about an issue where the `[INST]` tag isn't appearing in the output when fine-tuning **Mistral 7B Instruct v0.2**, even though their function includes it. `@gt9393` and `@flail_.` clarified that users are not required to type these tags; instead, the tags should be a part of the prompt formatting during fine-tuning.
- **Confusion Over Tokenizer Settings and Dataset Tags**: `@gt9393` shared that there's uncertainty and different practices concerning adding `add_eos_token/add_bos_token` in the tokenizer settings, while also including `<s>/</s>` in the dataset. `@nigelt11` acknowledged possibly missing the `add_eos_token` detail in their setup.
- **Understanding Instruction Tags and Output**: `@gt9393` discussed the appropriateness of placing system prompts inside the `[INST]` tags versus outside in the fine-tuning dataset. The importance of differentiating the instruction template `[INST]` tags, which should be used for inference, from other system components like `eos` or `bos` tokens was a point of dialogue.
- **Adding Identity to Model Chat Behavior**: `@lordofthegoons` inquired about training "identity" into a model's chat behavior, referencing models like Samantha, without follow-up discussion providing a direct answer.
- **Medium and DataCamp Articles as Guides**: `@gt9393` shared links to a Medium article ([Mistral 7B Fine-Tuning: A Step by Step Guide](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8)) and DataCamp tutorial ([Introduction to ChatGPT](https://www.datacamp.com/tutorial/mistral-7b-tutorial)), potentially for further reading on fine-tuning practices. These serve as resources to understand the use of instruction tags and tokenizers in fine-tuning language models.

**Links mentioned**:

- [Mistral-7B Fine-Tuning: A Step-by-Step Guide](https://gathnex.medium.com/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8): Introducing Mistral 7B: The Powerhouse of Language Models
- [Mistral 7B Tutorial: A Step-by-Step Guide to Using and Fine-Tuning Mistral 7B](https://www.datacamp.com/tutorial/mistral-7b-tutorial): The tutorial covers accessing, quantizing, fine-tuning, merging, and saving this powerful 7.3 billion parameter open-source language model. 

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1199962371284803644) (3 messages): 

- **Model Merging Optimal at 1.0 Total Weight**: `@alphaatlas1` mentioned that, although theoretically possible, merges with weights not summing up to 1.0 don’t perform as well in reality compared to those that do sum up to 1.0.
- **Clarification on DARE TIES Method**: `@alphaatlas1` clarified that the original DARE TIES paper suggested using very low densities with total weights above 1.0 for merging, which would combine "outlier" weights from two models without conflict; however, they noted this approach doesn't seem to yield the best results in practice.
- **Merging Models Beyond a Threshold Breaks Down**: `@sanjiwatsuki` observed that the process of merging models tends to break down when the total weight exceeds 1.4.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1199736389042970644) (4 messages): 

- **Inquiry about Hugging Face Pro for Large Models**: User `@keihakari` asked if **Hugging Face Pro** is suitable for running Deployment Inference with a 70b model.
- **SigLIP as a CLIP Alternative**: `@jeremy.london` mentioned that **CLIP** can be replaced with **SigLIP** now.
- **Modifying gguf File's Internal Model Info**: `@lordofthegoons` inquired about a method to manually change the **data of a gguf file's internal model info card**.
- **Parameter Count Issues with Frankenmerges**: `@lordofthegoons` also noted that **frankenmerges** tend to result in incorrect parameter counts.
  ,

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1199786275641753682) (10 messages🔥): 

- **In Search of API-Savvy Chatbots**: `@allanyield` inquired about companies creating products that use API documentation to turn AIs into chatbots capable of interacting with APIs on behalf of a user. No specific products or companies were mentioned in subsequent chat.
- **Random GIF Interjection**: `@Error.PDF` shared a GIF from Tenor with a language setting notification, which seems off-topic and unrelated to the surrounding technical discussion.
- **Mistral's Competitive Edge with Longer Sequences**: `@carsonpoole` mentioned that using longer sequences and larger batch sizes seems beneficial and expressed that such a model could be compatible with a **Mistral** implementation.
- **Training Run Curiosities**: `@everyoneisgross` was curious about the duration of training runs using **phi 2**, suggesting they have a desire to leverage their own 3060 GPU for overnight training.
- **Open Source Intentions Raise Anticipation**: In response to `@gabriel_syme`'s query, `@carsonpoole` indicated an interest in open-sourcing their implementation should it prove successful, generating interest in its potential release.
- **Loss Improvements with RMS**: `@carsonpoole` noted that utilizing RMS instead of layernorm is yielding favorable loss results, suggesting an optimized approach in their current work.
- **Mistral Compatibility Challenges**: `@carsonpoole` acknowledged that direct plug-and-play with a Mistral implementation isn't technically feasible, hinting at complexities in model integration.
- **Evaluating the Model's Performance**: `@carsonpoole` signaled an intent to conduct evaluations on the model soon, indicating an upcoming phase of performance assessment.

**Links mentioned**:

[Jordan Batter Looksmaxxing GIF - Jordan batter Looksmaxxing No - Discover &amp; Share GIFs](https://tenor.com/view/jordan-batter-looksmaxxing-no-sigma-me-gif-15250311043011508045): Click to view the GIF

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1199668001729953852) (27 messages🔥): 

- **Era of Heterogeneous AI Architectures**: `@burnytech` shared a [LessWrong post](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and) cross-posted from the [AI Alignment Forum](https://alignmentforum.org/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and), discussing the emergence of **heterogeneous AI architectures** combining **Transformers** and **Selective SSM (Mamba)**. `@mikahdang` remarked on the influential nature of this principle for future AI development.

- **DoraemonGPT for Dynamic Video Tasks**: `@DoraemonGPT` highlighted the capabilities of *DoraemonGPT* in dynamic video understanding by converting videos into a symbolic memory for spatial-temporal querying, using tools for external knowledge assessment, and a novel LLM-driven planner. This work is detailed in an [arXiv paper](https://arxiv.org/abs/2401.08392) and aims to handle complex tasks by leveraging LLMs for video scene interpretation.

- **RNNs Showing Promising Potential**: `@_3sphere` commented on the potential of recurrent LLMs like RWKV and Mamba to be better at tracking character states in sequences, spotlighting RWKV 7B's mission to outpace Mistral with only 1T tokens, as seen in a [teaser link](https://fxtwitter.com/picocreator/status/1750245003690201363/photo/1) shared by `@euclaise`.

- **Advancements in Moderate-Sized LLMs for Translation**: `@mister_poodle` shared a paper introducing **Contrastive Preference Optimization (CPO)**, an approach improving the translation capabilities of moderate-sized LLMs, presenting potential quality improvements over supervised fine-tuning. The detailed study is available on [arXiv](https://arxiv.org/abs/2401.08417).

- **Adept Introduces Fuyu-Heavy Multimodal Model**: `.benxh` presented **Adept Fuyu-Heavy**, a new multimodal model designed for digital agents, boasting strong multimodal reasoning capabilities, evident in the [Adept.ai blog post](https://www.adept.ai/blog/adept-fuyu-heavy). Despite its size, Fuyu-Heavy outperforms larger models in certain benchmarks, with further details and examples provided in the announcement.

**Links mentioned**:

- [MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660): Token-free language models learn directly from raw bytes and remove the bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences, and standard autoregressiv...
- [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://arxiv.org/abs/2401.08417): Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation mod...
- [Adept Fuyu-Heavy: A new multimodal model](https://www.adept.ai/blog/adept-fuyu-heavy): Adept Fuyu-Heavy is a new multimodal model designed specifically for digital agents.
- [DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392): The field of AI agents is advancing at an unprecedented rate due to the capabilities of large language models (LLMs). However, LLM-driven visual agents mainly focus on solving tasks for the image moda...
- [Tweet from PicoCreator (🌉 in/arena) (@picocreator)](https://fxtwitter.com/picocreator/status/1750245003690201363/photo/1): RWKV 7B teaser (not final) ....  🤞that we will pass mistral with only 1T tokens  https://huggingface.co/BlinkDL/temp/blob/30ac41f863dcf0600a71e4182600022df989dfa1/rwkv-x052-7b-world-v2-86%25trained-2...
- [A-JEPA neural model: Unlocking semantic knowledge from .wav / .mp3 audio file or audio spectrograms](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...
- [AGI will be made of heterogeneous components, Transformer and Selective SSM blocks will be among them — LessWrong](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made-of-heterogeneous-components-transformer-and): This post is prompted by two recent pieces: …
- [AGI will be made of heterogeneous components, Transformer and Selective SSM blocks will be among them — LessWrong](https://www.lesswrong.com/posts/Btom6dX5swTuteKce/agi-will-be-made): This post is prompted by two recent pieces: …

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1199633382678732800) (231 messages🔥🔥): 

- **Consciousness Conundrum Continues**: `@nonameusr` queried the group on what they believe consciousness is, leading to a discussion with `@_3sphere` about the "easy" and "hard" problems of consciousness and `@giftedgummybee` jokingly referencing a fictional court case over AI sentience.
- **Towards Steerable AI**: Discussing `@mihai4256`'s work, `@teknium` expressed interest in the methods behind activation steering for language models. The conversation progressed to `@mihai4256` explaining the use of text prompts to influence model behavior, which inspired further inquiry from community members.
- **Prompt Formatting Affects LLMs**: Members like `@euclaise` and `@stellaathena` debated the efficiency of various methods for structuring prompts in language models, with links to academic papers discussing prompt sensitivity and potential layout issues when using dynamic templates.
- **GPU and AI Hardware Discourse**: Chats like `@theluckynick` and `@sirri69` discussed the capabilities and limitations of various GPUs, such as the 1080 ti and the 3060, in the context of running AI models, along with anticipation for future software optimizations.
- **Miscellaneous Tech Banter**: `@teknium` and `@carsonpoole` exchanged opinions on the merits of different operating systems for running AI models and general computing, while `@n8programs` shared some technical achievements involving webGL2 and quantization methods for vector representations.

**Links mentioned**:

- [Quantifying Language Models&#39; Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting](https://arxiv.org/abs/2310.11324): As large language models (LLMs) are adopted as a fundamental component of language technologies, it is crucial to accurately characterize their performance. Because choices in prompt design can strong...
- [Skull Explode GIF - Skull Explode - Discover &amp; Share GIFs](https://tenor.com/view/skull-explode-gif-25528415): Click to view the GIF
- [Skeleto Skeleton GIF - Skeleto Skeleton Fire - Discover &amp; Share GIFs](https://tenor.com/view/skeleto-skeleton-fire-hell-burn-gif-26129219): Click to view the GIF
- [Sunglitters Glittery GIF - Sunglitters Glittery - Discover &amp; Share GIFs](https://tenor.com/view/sunglitters-glittery-gif-24179798): Click to view the GIF
- [Surface Form Competition: Why the Highest Probability Answer Isn&#39;t Always Right](https://arxiv.org/abs/2104.08315): Large language models have shown promising results in zero-shot settings (Brown et al.,2020; Radford et al., 2019). For example, they can perform multiple choice tasks simply by conditioning on a ques...
- [LM-exp/steering at main · nrimsky/LM-exp](https://github.com/nrimsky/LM-exp/tree/main/steering): LLM experiments done during SERI MATS - focusing on activation steering / interpreting activation spaces - nrimsky/LM-exp
- [Adding topic steering on layers  · Issue #5119 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/5119): Prerequisites Please answer the following questions for yourself before submitting an issue. [ x ] I am running the latest code. Development is very rapid so there are no tagged versions as of now....
- [GitHub - VikParuchuri/texify: Math OCR model that outputs LaTeX and markdown](https://github.com/VikParuchuri/texify): Math OCR model that outputs LaTeX and markdown. Contribute to VikParuchuri/texify development by creating an account on GitHub.
- [Steering GPT-2-XL by adding an activation vector](https://www.greaterwrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector): Summary: We demonstrate a new scalable way of interacting with language models: adding certain activation vectors into forward passes.[2] Essentially, we add together combinations of forward passes in...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1199900982402891796) (20 messages🔥): 

- **Clarification on Fine-tuning Versus Instruct-tuning**: `@moconna` inquired about the differences between fine-tuning and instruct-tuning and how to apply them with their own dataset. `@besiktas` clarified that instruct tuning typically comes after fine-tuning, and fine-tuning would involve causal language modeling on a dataset before potentially applying instruct tuning with high-quality specialized data.

- **Confusion Over Fine-tuning Data Format for Mixtral**: `@moconna` sought clarification on the correct format for fine-tuning Mistral, as tutorials seem to suggest instruction-based formatting, which they associated with instruct-tuning. `@besiktas` affirmed that the terminologies are vague and the provided format is often used for both fine-tuning and instruct tuning.

- **Direct Approach for Fine-tuning Tasks**: When `@moconna` asked how to fine-tune Mistral for specific tasks in new domains and languages, `@besiktas` suggested using cleaned data for the new domain and cautioned that success in a completely new language might be uncertain without further research.

- **Continual Model Updates in Production**: `@kenakafrosty` queried the community whether there is consensus on updating fine-tuned models with live usage data. The discussion appeared inconclusive, with varying potential methods mentioned but no definitive strategy.

- **Query and Banter about Optimal Dataset for Mistral**: `@locutusque` sought recommendations for the best dataset to fine-tune Mistral, to which `@besiktas` humorously suggested MNIST, leading to a playful exchange about visual capabilities of language models like Mistral.
  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1199669217855811615) (5 messages): 

- **Project Obsidian Upgrade Announced**: `@qnguyen3` announced plans to **upgrade Obsidian to v2**.
- **StableLM 2 1.6B Selected for Upgrade**: In the upgrade process, `@qnguyen3` will be **using stableLM 2 1.6B**, opting for an even smaller model this time.
- **Community Response to Upgrade**: `@giftedgummybee` simply responded with "Nice" to the news of using **stableLM 2 1.6B** for Obsidian's upgrade.
- **InstructDoc Dataset Shared**: `@gabriel_syme` shared a link to the **InstructDoc dataset** on GitHub, a resource for zero-shot generalization in visual document understanding ([GitHub - nttmdlab-nlp/InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc)).

**Links mentioned**:

[GitHub - nttmdlab-nlp/InstructDoc: InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions (AAAI2024)](https://github.com/nttmdlab-nlp/InstructDoc): InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions (AAAI2024) - GitHub - nttmdlab-nlp/InstructDoc: InstructDoc: A Dataset for Zero-Shot Generaliz...

  ,

### OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1199830782458474527) (1 messages): 

- **ChatGPT in Action**: `@abdubs` is calling out to **@everyone** to share their unique and creative uses of **ChatGPT** in the <#1155775326253756456> channel. They are enthusiastic to learn how the technology benefits users’ lives in various ways, from aiding students to storytelling and enhancing communication.
  

---


### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1199684939046723704) (18 messages🔥): 

- **AI Image Prompting Clarification**: `@lugui` explained that GPT-4 models describe an image to build a prompt rather than generating a new image, which leads to very different outcomes.
- **Image Manipulation Alternative Suggested**: `@xenowhiz` responded to a concern about DALL-E's image recreation capability by suggesting the use of **Code Interpreter** with its image manipulation modules to achieve desired results.
- **AI Podcast Endorsement**: `@fran9000` recommended "[The AI Breakdown](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614)" podcast for daily news and analysis on artificial intelligence, covering a wide range of topics from creativity to ethical considerations.
- **Using Older GPT Versions for Tasks**: `@lzgodhook13` expressed frustration with newer versions of GPT and inquired about using the **version 3 from May to June 2022**, citing difficulty with simple tasks such as ordering numbers.
- **Understanding GPT-4 Message Limits**: `@pope0004` questioned why GPT-4's usage is limited for premium users, with `@jaicraft` suggesting API or Copilot Pro for extended access, and `@satanhashtag` confirming that message restrictions apply to everyone.

**Links mentioned**:

- [Smol Talk](https://buttondown.email/ainews): We summarize AI discords, and send you a roundup each day!
- [‎The AI Breakdown: Daily Artificial Intelligence News and Discussions on Apple Podcasts](https://podcasts.apple.com/us/podcast/the-ai-breakdown-daily-artificial-intelligence-news/id1680633614): ‎Technology · 2024

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1199639704774115380) (66 messages🔥🔥): 

- **Getting GPT to Spit Out a Novel**: `@tetsujin2295` expresses frustration that while GPT is willing to produce a lengthy 1500-word summary within a chat, it fails to accomplish the same when directed to summarize in a downloadable document. Suggestions by `@loschess` include managing the output by breaking down the task into sections due to the token limitation in GPT's responses.

- **Custom GPT Gone AWOL**: `@sstrader29` encountered their custom GPT model vanishing from search. `@loschess` advises checking for any emails regarding the incident, implying potential communication from OpenAI about model issues.

- **Granting GPT Email Diplomacy Powers**: `@greg_nyc` is exploring ways to enable GPT to draft email responses through Gmail, with `@darthgustav.` suggesting the need to create a custom action with Google's API and to carefully follow Google documentation.

- **File Upload Frustrations in GPT Builder**: `@alexandre.f.s` brings up issues with uploading files for training custom GPTs, having troubles with the GPT Builder. `@darthgustav.` recommends steps like clearing the browser cache and methodically attaching files one-by-one to prevent corruption.

- **Context Confusion After Update**:
  - `@kickiniteasy` and `@ajkuba` report a severe reduction in the context window for custom GPTs post-update, drastically impacting the models’ performance and continuity.
  - `@nefas` mentions issues with GPT chains in the builder, as they seem to forget previous interactions, a problem that does not occur in the preview window.
  - `@cairpli` has created a bug-report for the said issues, linking to a Discord channel. (The link appears to be a placeholder '<<<null>>>' and thus is not included.)

- **GPT's Advertising Ambitions**: `@9rld` is interested in creating a custom GPT that mimics the style of print advertisements, looking to use a web crawler to collect data from ad websites. They are querying whether GPT can extract text from the image-heavy advert content available on these sites.
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1199685871780241418) (71 messages🔥🔥): 

- **Avoiding Lists Becomes a Hot Topic**: `@dbugger` expressed frustration with ChatGPT providing lists in responses, regardless of the input. `@darthgustav.` and others suggested providing explicit instructions on desired format, using examples such as an emotion-based prompt or character-named prompts to discourage list-style output ("EmotionPrompt").

- **Prompts to Avoid Lists**: `@eskcanta` recommended using 'negative prompting' combined with motivation to avoid the model defaulting to lists, sharing a detailed, character-driven prompt example designed to elicit a more narrative response.

- **Tackling Instagram Caption Challenges**: `@semir9526` sought advice on crafting prompts for creating Instagram captions in a specific format. Multiple suggestions were offered, including using variable-based structured output and avoiding mentioning character limits which ChatGPT cannot count.

- **Improving Keyword Searches in Documents**: `@novumclassicum` queried how to achieve better results when having GPT scan documents for keywords. `@darthgustav.` advised understanding stochastic inference and considering task chunking, whereas `@eskcanta` mentioned possible Python tool integration for string treatment, and both discussed semantic matching for better search accuracy.

- **Prompt Engineering Hackathon Team Formation**: `@brayheart` invited members to team up for a prompt engineering hackathon, opening the door for collaboration, while `@eskcanta` showed potential interest, asking for specific goals.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1199685871780241418) (71 messages🔥🔥): 

- **Breaking the List Habit**: `@dbugger` expressed frustration with ChatGPT consistently providing answers in list format, regardless of the prompt. `@darthgustav.` and `@eskcanta` proposed various solutions, including providing explicit output templates and style instructions, using EmotionPrompt, and crafting prompts to focus on single items at a time.
  
- **Marketing Magic Crafting**: `@semir9526` sought advice for improving an AI-generated marketing prompt for authentic Instagram captions. `@darthgustav.` advised on prompt structure, using variables, and avoiding specifying Instagram's character limits, instead suggesting to keep captions "brief and exciting."

- **Search Smarter, Not Harder**: `@novumclassicum` questioned the consistency of GPT's keyword searches within documents. `@darthgustav.` explained the stochastic nature of the model and suggested using semantic matching for keywords, while also recommending splitting the task into chunks for better results.

- **Prompt Engineering Challenge**: `@brayheart` inquired about interest in a prompt engineering hackathon. `@eskcanta` showed interest but asked for more details on the hackathon's specific goals.
  ,

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1199624218640400415) (160 messages🔥🔥): 

- **Perplexity AI Companion Extension Suggestion**: User `@mares1317` advocated using the Perplexity AI Companion extension, even suggesting it can retain sources for follow-up questions with GPT-4, despite a contrary point made by `@icelavaman` who noted that the iOS app cannot keep sources for follow-up queries.
- **Subscribing to Pro Brings Benefits**: `@thejuicyy` subscribed to Pro and was impressed with its relevancy checks. `@mares1317` recommended trying Perplexity AI companion with GPT-4 for a significantly different experience compared to the free version.
- **Troubleshooting Discount Code Application**: Multiple users discussed issues with applying discount codes for Perplexity Pro. `@toothpick4339` shared successful steps from Reddit user u/ArakDemonBlade, `@bennsiee` struggled to find where to enter a promo code, but `@ok.alex` and `@speedturkey` provided assistance and pointed to contacting support at pro@perplexity.ai.
- **Query on Increasing Limit for Online LLMs**: User `@mantas_82008` inquired about the possibility of increasing the limit beyond 10/minute for online LLM models. `@icelavaman` replied with a link to a related Discord channel for further information.
- **Challenges with Large Google Drive Documents**: `@jaybob32` faced issues with accessing large Google Drive documents. Users `@gentlefoxssbm`, `@deicoon`, and `@me.lk` suggested downloading or converting the document, with `@mares1317` providing a search link for converters, and the sharing of a solution involving copy-pasting the text directly to Perplexity's prompt box.

**Links mentioned**:

- [Pricing](https://docs.perplexity.ai/docs/pricing): no description found
- [More than an OpenAI Wrapper: Perplexity Pivots to Open Source](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/): Perplexity CEO Aravind Srinivas is a big Larry Page fan. However, he thinks he&#039;s found a way to compete not only with Google search, but with OpenAI&#039;s GPT too.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1199665672897179798) (4 messages): 

- **Perplexity AI Research Queried**: User `@nocode7` shared a [Perplexity AI search link](https://www.perplexity.ai/search/Arc-wont-let-usDclkxuSCehLZy3v6RMFA?s=c), although the context or content of the research was not specified.
- **Business Chat Recap Mastery**: `@miamiseipazzesco777` was recognized for effectively recapping some business chats.
- **Insights from Perplexity's Head of Design**: `@mares1317` shared a [YouTube video](https://www.youtube.com/watch?v=4BH454Kw-90&t=60s) featuring **Henry Modisett**, Head of Design at Perplexity AI, discussing the challenges of designing for AI and how to land a job in the field.
- **Praise for Perplexity's Team and Copilot**: `@zenrobot.eth` praised the aforementioned interview with Henry Modisett and also highlighted the utility of Perplexity’s Copilot feature, providing a link to a summary of global news trends ([Perplexity AI search link](https://www.perplexity.ai/search/Trends-in-Global-nU4umKTNQhywNTQFVhmGGg)) and explaining Copilot's conversational search capability detailed in a [blog post](https://blog.perplexity.ai/faq/what-is-copilot).

**Links mentioned**:

- [Designer at $525M AI Startup &#39;Perplexity&#39; reveals challenges of designing for AI](https://www.youtube.com/watch?v=4BH454Kw-90&t=60s): 🧠 WHAT WILL YOU LEARN IN THIS INTERVIEW?1. Challenges of designing for AI2. How to get a job as an AI Designer📢  WHY IS HENRY MODISETT A ROCKSTAR?Henry Mod...
- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1199858991895937094) (5 messages): 

- **Credit Unlocked After Auto Top Off**: User `@stevemac_90623` mentioned that a **$5.00 credit** was granted only after completing the auto top off process by entering **$2** as the amount.

- **Intrigued by API vs Browser Responses**: `@obicho` expressed curiosity about why responses differ when using the API versus the browser.

- **API Model Equivalent Enquiry**: `@benhirap` inquired about which API model corresponds to the "Experiment" feature with Copilot on the chat website.

- **Unraveling the Model Mystery**: `@noremac258` identified **PPLX 70B** as the API model equivalent to the "Experiment" (with Copilot) from the chat website.

- **Browser vs. API Behavior Explained**: `@icelavaman` explained the differing responses on browser versus API by pointing to the different system prompt/sources that are differently searched.
  ,

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1199683093158699008) (71 messages🔥🔥): 

- **GPU Over CPU in AI Operations**: `@laszlo01` inquired about how to switch from using CPU to GPU in AI and was advised by `@heyitsyorkie` to search for this setting in the chat page's settings panel on the right-hand side.
- **Request for Open-Source Chatbot Creation Tools**: `@jan_naj` sought a repository for building customizable GPT-like chatbots with options for local hosting and incorporating memory threads. `@fabguy` responded with a somewhat facetious link to GitHub's search page, leading to `@jan_naj`'s dissatisfaction and `@dagbs` suggesting a more specific inquiry.
- **Linux Build Lack for Older Processors**: `@d0mper` mentioned using an old computer without AVX2 support and asked about alternatives to LM Studio for loading language models. `@heyitsyorkie` clarified that the Linux build only supports AVX2 CPUs.
- **Crypto Bot Warning and Ban Requests**: `@heyitsyorkie` alerted users to report any crypto bots spamming Discord invites for prompt banning.
- **Compiling Llama.cpp as LM Studio Alternative**: For users like `@d0mper` seeking an alternative to LM Studio on platforms like Ubuntu, `@heyitsyorkie` suggested compiling llama.cpp, which supports loading language models.
- **Discussion on Incorporating Stable Diffusion into LMStudio**: After some confusion regarding support for StableLM models in LM Studio, `@heyitsyorkie` mentioned Stable Diffusion's separate C/C++ port and the potential of integrating it into LM Studio. `@altryne` noted a recent update that made a specific engine compatible.

**Links mentioned**:

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-release): Find, download, and experiment with local LLMs
- [Build software better, together](https://github.com/search): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [GitHub - ggerganov/llama.cpp: Port of Facebook&#39;s LLaMA model in C/C++](https://github.com/ggerganov/llama.cpp): Port of Facebook&#39;s LLaMA model in C/C++. Contribute to ggerganov/llama.cpp development by creating an account on GitHub.
- [GitHub - leejet/stable-diffusion.cpp: Stable Diffusion in pure C/C++](https://github.com/leejet/stable-diffusion.cpp): Stable Diffusion in pure C/C++. Contribute to leejet/stable-diffusion.cpp development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1199623974137647196) (20 messages🔥): 

- **Model Recommendations for a Newcomer**: New user `@rparada` asked for a **model recommendation** for code conversion and neural network architecture modification. User `@fabguy` directed them to check the entries in channel **#1185646847721742336**.

- **Code Snippet for PDF Reading with RAG**: `@ui.mz` shared a code snippet using **PyMuPDFLoader** to read from a PDF file with RAG and asked for tips since they were encountering a **ValidationError** related to the **OpenAI Embeddings API**.

- **Need for an OpenAI Embeddings API Substitute**: Following the error discussion, `@ui.mz` revealed a lack of an **NVIDIA GPU**, and `@fabguy` clarified that the intent was to help them figure out their own solution by providing a [GitHub repository](https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio/) as a reference.

- **Inquiry About Image/File Reading Models**: User `@pandora_box_open` inquired about a model capable of reading images/files effectively. `@heyitsyorkie` responded with a Discord link for a model limited to describing images and stated that there's no RAG system for **document reading** yet.

- **Explanation of RAG and Potential Integration**: User `@heyitsyorkie` explained the concept of **Retrieval-Augmented Generation (RAG)** after a prompt from `@pandora_box_open`, who afterwards expressed hope for its integration. The explanation included a [link](https://www.databricks.com/glossary/retrieval-augmented-generation-rag) to Databricks' glossary entry on RAG.

- **Exploring RAG with a HuggingFace Model**: `@pandora_box_open` shared a [HuggingFace model link](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp) that might showcase an integration resembling RAG.

**Links mentioned**:

- [Retrieval Augmented Generation](https://www.databricks.com/glossary/retrieval-augmented-generation-rag): Retrieval augmented generation or RAG is an architectural approach that pulls your data as context for large language models (LLMs) to improve relevancy.
- [MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp · Hugging Face](https://huggingface.co/MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp): no description found

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1199623814053634138) (60 messages🔥🔥): 

- **GPU Layer Configurations for Various Models**: `@aswarp` and `@cloakedman` discussed the optimal number of GPU layers for different sizes of models and graphics cards, using values like `-1` to offload all layers to GPU or manually adjusting to avoid crashes. `@heyitsyorkie` suggested playing with layer numbers if `-1` causes an error, recommending to inspect model details in LM Studio for guidance.
  
- **Performance Tied to Hardware Specifications**: `@smallshinyant` sought advice on benchmarking new hardware additions by focusing on metrics like tok/sec, whereas `@bobzdar` pointed out that extra VRAM allows for running larger or less compressed models, suggesting that performance is not just about speed.

- **Utilizing Maximum Hardware Potential**: `@aswarp` inquired about increasing system RAM usage when it remains largely unused while running models. `@.ben.com` advised that focusing solely on GPU use is preferable for performance, unless one intentionally wants to use a large model that doesn't fit in the GPU, which would, however, result in slower processing.

- **Software Settings Affect Model Performance**: Users like `@aswarp` and `@dylpickle300` encountered issues with model performance and discussed various settings in LM Studio, like `n_gpu_layers`, `num_layers` from the model's GGUF JSON, suggesting adjustments to potentially increase performance despite hitches such as unused system RAM and models stalling.

- **Hardware Support and Model Compatibility**: `@luthorheim` questioned the possibility of running models without AVX2 instruction support, to which `@heyitsyorkie` provided a link to beta releases that might support older processors. A reminder from `@cloakedman` highlighted that model files larger than the available VRAM would result in slow performance or the need to downscale quality.

**Links mentioned**:

[LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1199832714572996760) (2 messages): 

- **Navigating the Sea of Models**: `@greg0403` expressed confusion about choosing between various models on HuggingFace, questioning their Unique Selling Points (USPs) due to the minimal documentation. They sought guidance for understanding how to differentiate and select one model over another.
- **A Compass for Model Comparison**: In response to `@greg0403`, `@kadeshar` recommended using leaderboards like the one available at [HuggingFace open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) as a good starting point for comparing model performance.

**Links mentioned**:

[Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): no description found

  

---


### LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1199791991727333446) (1 messages): 

- **Exploring Interoperability Between LM Studio and Open Interpreter**: User `@222gate` is attempting to integrate **LM Studio inference with memgpt**, and then into **Open Interpreter**, discussing with `@cpacker` the similarities between memgpt server and the OpenAI Assistant API. They are investigating if memgpt's server can mimic OpenAI's in terms of the chat and completion call functionalities.
  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1199800276610207834) (2 messages): 

- **Seeking Prompting Tips**: User `@222gate` asked the community for **ideas for improved prompting in general**. No specific context or details were provided in the inquiry.
- **Integration Challenge**: `@222gate` mentioned difficulties in integrating **LM Studio memGPT and OpenAI** and is seeking assistance with the process. The complexities of the integration were not elaborated upon.
  ,

### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1199824460295970906) (1 messages): 

- **EleutherAI Partners with NSF for AI Research**: `@tastybucketofrice` announced EleutherAI's partnership with the **National Science Foundation (NSF)** to launch the **National AI Research Resource (NAIRR)**, which aims to provide access to critical resources for AI research. The official announcement can be read [here](https://new.nsf.gov/news/democratizing-future-ai-rd-nsf-launch-national-ai).

- **EleutherAI's Commitment to Open AI Research**: Starting from the release of Language Models (LMs) in 2020, EleutherAI has advocated for open research and has now contributed GPU grants to enhance academic AI research capabilities. The history of their commitment can be found in their blog post [“Why release a large language model?”](https://blog.eleuther.ai/why-release-a-large-language-model/).

- **Addressing Compute Resources for Researchers**: Despite broader access to pretrained models today, `@tastybucketofrice` points out the persistent issue of limited compute resources as a barrier for researchers. EleutherAI is working to ensure researchers can control how their models behave and the values they encode.

- **Empowering AI Research with GPT-NeoX**: To combat HPC challenges, `@tastybucketofrice` highlights the [GPT-NeoX library](https://github.com/EleutherAI/gpt-neox), which facilitates AI research by running at scale on various platforms, including Oak Ridge National Lab's Summit and Frontier, LUMI, AWS, and CoreWeave.
  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1199796929874374736) (11 messages🔥): 

- **License Confusion in the Code Wild West**: `@xa9ax` inquired about specific licenses on GitHub code repositories that may restrict training models for academic research. `@stellaathena` noted that virtually **no licenses directly address model training**, while `@avi.ai` highlighted that legal outcomes may **vary by jurisdiction**.

- **Searching for Legal Advice**: In response to further queries from `@xa9ax`, `@stellaathena` recommended consulting a lawyer for clarification on licensing issues. Meanwhile, `@avi.ai` suggested conducting initial research into copyright and intellectual property law relevant to one's own locale.

- **Goodbye to Discord?**: `@hostiq` simply stated, "It is time to leave this discord," suggesting they were leaving the community chat.

- **GPU Grant Collaboration Offer**: `@yikesawjeez` reached out to `@stellaathena` looking for a tie-in with an existing GPU grant, offering a **link to a server** and a **google form** for compute access listed in their profile.

- **CoPilot Case Crawl**: `@clockrelativity2003` inquired about the current status of the GitHub CoPilot litigation. `@.undeleted` humorously remarked that **legal cases take an incredibly long time to resolve**, implying the case is still unresolved.
  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1199630853467283556) (80 messages🔥🔥): 

- **Data Quality vs. Size in ML Training**: `@catboy_slim_` discussed data quality, remarking that curating diverse, high-quality data is challenging but doesn't require enormous volume. They mentioned memory of **Wavenet data** being relatively small yet well-curated.

- **GPU/TPU Hardware & Framework Compatibility Discussed**: `@alofty` shared an [article](https://arxiv.org/abs/2309.07181) about the portability of ML frameworks across different hardware, leading to a debate on the performance of JAX on TPUs and GPUs by `.the_alt_man`, who advised against PyTorch due to poor XLA support.

- **'MambaByte' Makes a Splash with Token-Free Language Modeling**: An [academic paper](https://arxiv.org/abs/2401.13660) shared by `@pizza_joe` introduced MambaByte, a state space model for token-free language modeling, sparking skepticism from `_inox` until `@thatspysaspy` noted the involvement of notable researcher Sasha Rush.

- **Burn: A New Rust Deep Learning Framework**: `@kenakafrosty` shared a [link](https://burn.dev/) to Burn, a Rust-based DL framework, and discussed its potential but noted it must have robust parallelism support to be competitive. `@canadagoose1` brought focus to the need for multinode support, while `.the_alt_man` favored more minimalistic frameworks like JAX for performance.

- **Continual Updates to Fine-Tuned Models in Production Explored**: `@kenakafrosty` inquired about best practices for updating fine-tuned models, leading to a discussion with `@fern.bear` about methods like **Elastic Weight Consolidation (EWC)** and the use of loss functions to balance between restricting changes too hard and allowing significant deviations.

**Links mentioned**:

- [MambaByte: Token-free Selective State Space Model](https://arxiv.org/abs/2401.13660): Token-free language models learn directly from raw bytes and remove the bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences, and standard autoregressiv...
- [DsDm: Model-Aware Dataset Selection with Datamodels](https://arxiv.org/abs/2401.12926): When selecting data for training large-scale models, standard practice is to filter for examples that match human notions of data quality. Such filtering yields qualitatively clean datapoints that int...
- [no title found](https://burn.dev/): no description found
- [The low-rank hypothesis of complex systems](https://arxiv.org/abs/2208.04848): Complex systems are high-dimensional nonlinear dynamical systems with intricate interactions among their constituents. To make interpretable predictions about their large-scale behavior, it is typical...
- [The Grand Illusion: The Myth of Software Portability and Implications for ML Progress](https://arxiv.org/abs/2309.07181): Pushing the boundaries of machine learning often requires exploring different hardware and software combinations. However, the freedom to experiment across different tooling stacks can be at odds with...
- [GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/](https://github.com/patrick-kidger/equinox): Elegant easy-to-use neural networks + scientific computing in JAX. https://docs.kidger.site/equinox/ - GitHub - patrick-kidger/equinox: Elegant easy-to-use neural networks + scientific computing in...

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1199741281753108558) (2 messages): 

- **Scaling Up to 1.3B Parameters**: `@stellaathena` mentioned that though the original paper had limitations, they provided the compute to train models at the **1.3B parameter scale**.
- **Reference to Channel**: `@random_string_of_character` pointed users to discussion results for 1b parameter models in another channel, specifically linked as `<#1129489948710539334>`.
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1199712859505045544) (14 messages🔥): 

- **Decoding the Neural Code**: User `@g_w1` queried about plotting points in space concerning the monosemanticity concept. `@woog` clarified that they're plotting **decoder weights**, which are interpreted as feature directions recovered in the **Sparse Autoencoder (SAE) space** to neuron direction.
  
- **Dictionary Elements as Neuron Directions**: In further clarification, `@woog` confirmed to `@g_w1` that each dictionary element in the SAE is a **direction in neuron space**, suggesting a connection between SAE space and neuron space that is visualized through plotting. `@g_w1` acknowledged the explanation.

- **An Update on Interpretability Research**: User `@loganriggs` shared a [January update](https://transformer-circuits.pub/2024/jan-update/index.html) from the Anthropic interpretability team outlining a set of preliminary research notes and developments in the field. Themes included **attention superposition, dictionary learning on MNIST**, and **features in multilayer models**.

- **Observation on Report Typo**: User `@ishitatsuyuki` pointed out a typo in the shared Anthropic interpretability team update with the title "Joint Superposition Between MLP Neurons and Residual Stream" being **duplicated twice** in the "Counterexamples in Superposition" section.

**Links mentioned**:

[Circuits Updates - January 2024](https://transformer-circuits.pub/2024/jan-update/index.html): no description found

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1199648537051406366) (7 messages): 

- **Parallelism Paradigms in Focus**: `@groggyrhombus` inquired if `@337128969059172353`'s approach was akin to DeepSpeed's tensor+expert parallelism, to which `xyzzyrz` simply affirmed with **"Yeah"**.
- **Commendable Progress on Compute**: `@tastybucketofrice` expressed appreciation for `@337128969059172353`'s efforts and mentioned plans to test the compute over the upcoming week.
- **Deep Dive into CUDA Initialization Issue**: `@tastybucketofrice` linked to an [issue in DeepSpeed](https://github.com/microsoft/DeepSpeed/issues/1987) concerning CUDA initialization before forking and showed interest in investigating this further.
- **Suspicions About Test Errors**: `@catboy_slim_` suspects recent changes in how CUDA or pytest handles forking, which might be causing test errors, but noted a lack of time to explore the issue.
- **Open to Assistance on Pull Request Alterations**: `@catboy_slim_` hinted at excluding certain parts from a pull request due to present constraints and welcomed anyone else to take on the task.

**Links mentioned**:

[Issues · microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed/issues/1987)): DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective. - Issues · microsoft/DeepSpeed

  ,

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1199626308687241236) (59 messages🔥🔥): 

- **Translator Troubles**: `@le_mess` shared a link to a **Tenor GIF**, which seems to be language-translated based on browser settings.
- **Seeking for Assistance**: `@nanobitz` recommended that someone (possibly `<@525830737627185170>`) could assist with an issue, while `@dangfutures` requested help with replicate configurations and `@mistobaan` asked for guidance on finetuning `phi-2`.
- **Explaining 'High Context'**: In response to `@hamelh` asking about the meaning of "high context," `@jsancs_` clarified it as an "8-10k context window".
- **Finetuning Frustrations and Successes**: `@dangfutures` casually noted another day of finetuning, while `@nafnlaus00` discussed the differences in learning rates needed when switching from `float16` to `bfloat16`.
- **GPU Purchasing Decisions**: Hardware choices were debated; `@yamashi` pondered between 8 `H100` or 16 `A100` GPUs, and `@casper_ai` suggests opting for more VRAM. `@dangfutures` supports the choice for 16 `A100` for increased VRAM, while `@c.gato` looks forward to Caseus’ announcement in the announcements channel.

**Links mentioned**:

- [Join the Replicate Discord Server!](https://discord.gg/replicate): Check out the Replicate community on Discord - hang out with 22243 other members and enjoy free voice and text chat.
- [Magic GIF - Magic - Discover &amp; Share GIFs](https://tenor.com/view/magic-gif-26166638): Click to view the GIF
- [[BOUNTY] update ReLoRA implementation · Issue #1198 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1198): ⚠️ Please check that this feature request hasn&#39;t been suggested before. I searched previous Ideas in Discussions didn&#39;t find any similar feature requests. I searched previous Issues didn&#39;t...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1199623649376870440) (13 messages🔥): 

- **Mamba and Lora Compatibility Unclear**: `@tank02.` asked if **Mamba** works with **Lora**, but `@le_mess` responded with uncertainty, stating they have **never trained Mamba**.
- **Tough Time Uploading Models to HF**: `@colejhunter` is encountering issues with pushing a trained model to **Hugging Face** (HF); the repo is created but not populated with model files. They provided their config and queried about missing settings beyond `hub_model_id`.
- **Save Step Strategies in Training**: `@caseus_` and `@noobmaster29` suggest using `save_steps` or `saves_per_epoch` with variations like `saves_per_epoch: 1` or `saves_per_epoch: 10` for controlling model save frequency during training.
- **Saving Models Without Specific Steps**: `@c.gato` mentions leaving save settings blank, which defaults to saving models at the end of each epoch.
- **Seeking Guidance on Pretraining CLIP**: `@emperor` is looking for papers, studies, or blog posts that discuss best practices for further pretraining of **CLIP**, specifically focused on domain adaptation through a contrastive objective on 50M images.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1199628958296190986) (15 messages🔥): 

- **GitHub Treasure Trove Unearthed**: `@dangfutures` shared a valuable [GitHub repository](https://github.com/abachaa/Existing-Medical-QA-Datasets) containing multimodal question-answering datasets in the medical domain with `@nanobitz`.
- **Funding Blues**: `@yamashi` lamented about being preoccupied with administrative tasks for securing **funding for compute resources**, expressing this in response to `@dangfutures`'s inquiry about their absence.
- **In Search of the Perfect Prompt**: `@builderx` inquired about the correct **alpaca format prompt** for model inference, leading to a collaborative clarification with `@c.gato`.
- **Echoes in the Alpaca Pen**: `@builderx` encountered an issue with the **alpaca prompt occasionally being repeated** in model outputs after training on it through Mistral, prompting a suggestion by `@c.gato` to seek help.
- **Format Matters for QA Training**: In a discussion initiated by `@neko.huh` on whether raw text should be converted to **alpaca QA format** for training, `@noobmaster29` remarked it might be beneficial if the training focus is on question answering.

**Links mentioned**:

[GitHub - abachaa/Existing-Medical-QA-Datasets: Multimodal Question Answering in the Medical Domain: A summary of Existing Datasets and Systems](https://github.com/abachaa/Existing-Medical-QA-Datasets): Multimodal Question Answering in the Medical Domain: A summary of Existing Datasets and Systems - GitHub - abachaa/Existing-Medical-QA-Datasets: Multimodal Question Answering in the Medical Domain:...

  

---


### OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1199812920255729745) (1 messages): 

- **Axolotl v0.4.0 Takes Flight**: The OpenAccess AI Collective announced the release of [axolotl v0.4.0](https://github.com/OpenAccess-AI-Collective/axolotl/releases/tag/v0.4.0), featuring support for new models, fixes for numerous bugs, and contributions from 56 individuals. `@caseus_` extended appreciation to everyone and gave a special shout-out to A16Z for their grant, promising to add discord contributor roles in the following week.

- **Acknowledgment of Community Contributions**: `@caseus_` thanked individual contributors by tagging them in the announcement and invited those not mentioned to DM with their GitHub and Twitter handles for recognition. Contributors listed include `@213644857309134849`, `@244959984352231425`, and many others.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1199636289234935819) (1 messages): 

- **Curiosity About Replicate Serverless Cold-Start**: User `@dreamgen` inquired about current cold-start times on serverless services for **large models**, noting past issues with providers **not caching models or docker images**. There's a concern that despite advances, load times should be a few seconds, but experiences have varied.
  ,

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1199654289170243594) (49 messages🔥): 

- **Lumiere Leads the Way**: `@swyxio` highlighted the [Lumiere model](https://lumiere-video.github.io/) by Google, a space-time diffusion model that generates video from text. They also shared their [writeup](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/) on the model's features including its impressive inpainting capabilities.
- **Challenging Google's Code Conservatism**: `@guardiang` commented on Google's reluctance to share AI-related code, supported by `@shivdinho` who agreed and noted the difficulty in replicating Google's research without code release.
- **Exploring Self-Instruct for AI**: `@youngphlo` provided a [link](https://arxiv.org/abs/2212.10560) to a paper discussing Self-Instruct, a method to enhance large language models by bootstrapping off their own generations.
- **Discussing Proper AI Language Model Architecture**: `@bathientran` sought advice on collaboration between infra/backend developers and those experienced with language models, prompting a response from `@philltornroth` expressing a willingness to help bridge communication gaps.
- **Potential Project for Discord LLM Chatbot**: `@swyxio` shared a link to a [GitHub repository](https://github.com/jakobdylanc/Discord-LLM-Chatbot) for a Discord chatbot powered by large language models and asked if anyone was interested in implementing it for the channel.

**Links mentioned**:

- [Weak-to-strong generalization](https://openai.com/research/weak-to-strong-generalization): We present a new research direction for superalignment, together with promising initial results: can we leverage the generalization properties of deep learning to control strong models with weak super...
- [Lumiere - Google Research](https://lumiere-video.github.io/): Space-Time Text-to-Video diffusion model by Google Research.
- [ai-notes/Resources/BENCHMARKS.md at main · swyxio/ai-notes](https://github.com/swyxio/ai-notes/blob/main/Resources/BENCHMARKS.md): notes for software engineers getting up to speed on new AI developments. Serves as datastore for https://latent.space writing, and product brainstorming, but has cleaned up canonical references und...
- [Tweet from ChatGPT (@ChatGPTapp)](https://x.com/chatgptapp/status/1750316948444086697?s=46&t=90xQ8sGy63D2OtiaoGJuww): we launched the GPT Store two weeks ago and wanted to share some of our featured GPTs!
- [GitHub - jakobdylanc/Discord-LLM-Chatbot: Choose a LLM from OpenAI / Mistral API or run a local model with LM Studio • Multi-user chat • Streamed responses • 200 lines of code](https://github.com/jakobdylanc/Discord-LLM-Chatbot): Choose a LLM from OpenAI / Mistral API or run a local model with LM Studio • Multi-user chat • Streamed responses • 200 lines of code - GitHub - jakobdylanc/Discord-LLM-Chatbot: Choose a LLM from O...
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560): Large &#34;instruction-tuned&#34; language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend he...
- [[AINews] Google Solves Text to Video](https://buttondown.email/ainews/archive/ainews-google-solves-text-to-video/): AI Discords for 1/23/2024. We checked 19 guilds, 291 channels, and 4199 messages for you. Estimated reading time saved (at 200wpm): 348 minutes. Lumiere -...

  

---


### Latent Space ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1199802649428623381) (2 messages): 

- **Self-Instruct Paper Walkthrough**: User `@swyxio` announced a session led by `<@556359685306056721>` to guide through the Self-Instruct paper on the new [`ai-event-announcements`](https://discord.com/channels/822583790773862470/1197350122112168006) Stage. The community was invited to join and is reminded to sign up [here](https://lu.ma/ls) for notifications of future events.

- **Stay Updated with Latent Space Events**: The [Latent.Space events](http://Latent.Space) calendar is available for subscription to stay notified of new events by clicking the RSS logo above the calendar on the right-hand side.

- **Final Frontiers Celebration in SF**: `@fanahova` shared excitement about the upcoming Latent Space demo day anniversary and an event titled **Final Frontiers** to celebrate it in San Francisco. More details can be found on [Twitter](https://twitter.com/FanaHOVA/status/1750288311594418632).

**Links mentioned**:

[Latent Space (Paper Club &amp; Other Events) · Luma](https://lu.ma/ls): View and subscribe to events from Latent Space (Paper Club &amp; Other Events) on Luma. Latent.Space events. PLEASE CLICK THE RSS LOGO JUST ABOVE THE CALENDAR ON THE RIGHT TO ADD TO YOUR CAL. &quot;Ad...

  

---


### Latent Space ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/1199772640349519912) (19 messages🔥): 

- **New Stage Feature for Paper Discussions**: `@swyxio` informed members about utilizing Discord's new Stage feature for the paper discussion with `@556359685306056721` and provided a [link to the stage](https://discord.com/channels/822583790773862470/1197350122112168006).
- **Elevator Music Adds Ambiance**: `@picocreator` joked about the addition of elevator music while waiting for the new Discord Stage feature to start.
- **Emoji Reactions Missed in Discord Stage**: `@420gunna` expressed disappointment over the inability to use emoji reactions during the Discord Stage session.
- **High Attendance at Paper Discussion**: `@swyxio` highlighted that 40 people attended the first paper session after the platform switch, while `@youngphlo` speculated that more might have tried to join during the initial week of the Luma reset.
- **Prospective Next Paper Teased**: `@swyxio` shared that, pending confirmation, the next paper for discussion could be [Pythia](https://arxiv.org/abs/2304.01373), potentially featuring several authors and the possibility of Quentin Anthony joining the latter half of the session. The Twitter thread by `@rasbt` is also recommended for further insight ([Twitter source](https://twitter.com/rasbt/status/1734920232173539796)).



**Links mentioned**:

- [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373): How do large language models (LLMs) develop and evolve over the course of training? How do these patterns change as models scale? To answer these questions, we introduce \textit{Pythia}, a suite of 16...
- [Tweet from varepsilon (@var_epsilon)](https://x.com/var_epsilon/status/1750034151065948379?s=20): this paper explaining how twitter&#39;s community notes feature works is very cool  https://arxiv.org/abs/2210.15723
- [Birdwatch: Crowd Wisdom and Bridging Algorithms can Inform Understanding and Reduce the Spread of Misinformation](https://arxiv.org/abs/2210.15723): We present an approach for selecting objectively informative and subjectively helpful annotations to social media posts. We draw on data from on an online environment where contributors annotate misin...

  

---


### Latent Space ▷ #[llm-paper-club-chat](https://discord.com/channels/822583790773862470/822583791217934366/1199855024868708432) (1 messages): 

- **Autonomous Agents using RESTful APIs**: `@swyxio` shared an interesting older project/paper titled *RestGPT*, which focuses on an **LLM-based autonomous agent** that controls real-world applications via **RESTful APIs**. The project can be found on GitHub [here](https://github.com/Yifan-Song793/RestGPT).

**Links mentioned**:

[GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs](https://github.com/Yifan-Song793/RestGPT): An LLM-based autonomous agent controlling real-world applications via RESTful APIs - GitHub - Yifan-Song793/RestGPT: An LLM-based autonomous agent controlling real-world applications via RESTful APIs

  ,

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1199635794449682522) (62 messages🔥🔥): 

- **Job Application Woes and Hints**: `@ziper_rom1` inquired about a CUDA developer position they applied for, expressing uncertainty after not hearing back for a month. Community members including `@mrdragonfox` and `@frosty04212` shared insights, suggesting that no response generally indicates rejection, and that personal connections often influence hiring decisions.
- **CUDA Developer Role Likely Filled**: `@kim_tech` provided an update stating that the CUDA developer position that `@ziper_rom1` applied for at Mistral might be filled, citing a [Twitter post](https://twitter.com/sandeep1337/status/1744399578269352293?t=6Vm_qBKAAgBHLHyVPwK13g&s=19) of a new hire from Nvidia.
- **Running Mistral on CPU-only Setups**: `@tominix356` asked about running Mistral 7M on a 16GB RAM computer without a GPU, receiving suggestions like trying LM Studio and feedback from `@kim_tech` about their experiences running similar models on a RAM-heavy laptop.
- **Finding the Right Tool for the Job**: A discussion initiated by `@xeglion` on what's best for an RTX 3080 evolved into advice on choosing tools like Github Copilot and Google Bard based on one's specific needs, with `@kerunix`, `@mrdragonfox`, and `@enerv` chiming in about different options and constraints.
- **Local Options for AI Code Assistance**: The demand for local AI code completion was discussed, with `@enerv` mentioning the potential of using Mistral models with local APIs and plugins. The conversation touched on options like Codeium and Sourcegraph's Cody, highlighting the variety of tools available for developers.

**Links mentioned**:

[Cody | AI coding assistant](https://sourcegraph.com/cody): Cody is the most powerful and accurate AI coding assistant for writing, fixing, and maintaining code.

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1199830072966791249) (1 messages): 

- **Mistral RAG Performance Checkpoint**: User `@duck` reported timings for Mistral 8x7b on a 3090 when performing a sort of RAG with Langchain and using llama.cpp for inference. They noted **sample times** of 17.05 ms for 102 runs and **eval times** as high as 175910.47 ms for 101 runs, considering these timings to be slow for the use case.

  

---


### Mistral ▷ #[ref-implem](https://discord.com/channels/1144547040454508606/1156609509674975262/1199847209408139345) (1 messages): 

- **Mixtral Memory Mayhem**: User `@l0gr1thm1k` is experiencing **CUDA memory errors** while trying to load Mixtral into memory. Despite using four NVIDIA T4s with 16GB of memory each, memory usage exceeds the expected 24GB for the 4bit quantized version and results in an error.
  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1199728095410794577) (1 messages): 

- **Metric Confusion for Mistral-7B on Dolly Dataset**: User `@bishwa3819` is attempting to finetune **Mistral-7B on the Dolly dataset** but expressed confusion about the adequacy of **BLEU and ROUGE metrics** for evaluating Language Model performance. They questioned if these metrics are sufficient for evaluating a Language Model trained on specific datasets like Dolly.
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1199750203679445013) (2 messages): 

- **Speedy Reddit Assistant with a Touch of Mistral**: `@hugoduprez` highlighted the creation of **Reddit copilot buddy**, a bot made with **Mistral** which operates so quickly that it appears to be offline.
- **New Insights into Audio Understanding**: `@shashank.f1` discussed a new approach to audio understanding, featuring a [YouTube video](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF) that delves into the **A-JEPA neural model** which can unlock semantic knowledge from .wav or .mp3 audio files.

**Links mentioned**:

[A-JEPA neural model: Unlocking semantic knowledge from .wav / .mp3 audio file or audio spectrograms](https://youtu.be/FgcN62LFzIU?si=AgXF48kylHmNZdmF): 🌟 Unlock the Power of AI Learning from Audio ! 🔊 Watch a deep dive discussion on the A-JEPA approach with Oliver, Nevil, Ojasvita, Shashank, Srikanth and N...

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1199754578716008498) (3 messages): 

- **Mistral API Summarization Limitations**: User `@nico2412_` inquired about using the **Mistral API** to summarize an article on the web via its URL, expressing difficulties in achieving this task.
- **LLMs Lack Internet Access**: `@mrdragonfox` clarified that large language models (LLMs), like Mistral, do not have direct internet access, which is why they can't call functions using web URLs for summarization.
- **Alternate Solution for Article Summarization**: User `@duck` proposed an alternative method by providing a link to a [GitHub notebook](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb) that outlines a process for summarizing web content with language models.

**Links mentioned**:

[LLM/llama-cpp-rag - final.ipynb at main · Quad-AI/LLM](https://github.com/Quad-AI/LLM/blob/main/llama-cpp-rag%20-%20final.ipynb): Contribute to Quad-AI/LLM development by creating an account on GitHub.

  ,

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1199631890978717767) (39 messages🔥): 

- **AI Study Survey for VFX Artists**: `@jordibares` shared a [survey link](https://forms.gle/vexRuFUVHoojnfah7) looking for insights from VFX artists and producers to be included in an AI study.
- **Quota Reset Request for `createSpace`**: `@troymurs` requested a `createSpace` quota reset due to canister crashes by reaching out to `<@907238188978950215>`. `@osanseviero` responded advising to send an email to website @ huggingface.co.
- **Fine-tuning Text Generation Models**: `@sookeyy` sought resources for fine-tuning text generation models and was recommended to use `remove_unused_columns=False` by `@robolicious` after encountering an error.
- **Interest in Collaborative Projects**: Users expressed interest in collaborative projects, with `@wondeys` looking for partners to create a Texas Hold'em Poker AI or an automated trading algorithm, and `@dsiegel` recruiting for building an Augmented Reality headset from scratch.
- **Organizing a Portuguese Model and Dataset Sprint**: `@namayra` is organizing a sprint/hackathon for Portuguese models and datasets and was directed by `@osanseviero` to contact Hugging Face at website@huggingface.co after not being able to reach Omar Espejel.

**Links mentioned**:

- [mathathon - a Hugging Face Space by Tonic1](https://huggingface.co/spaces/Tonic1/mathathon): no description found
- [app.py · Tonic/StableMed_Chat at main](https://huggingface.co/spaces/Tonic/StableMed_Chat/blob/main/app.py): no description found
- [Survey on the use of Artificial Intelligence in the visual effects industry](https://forms.gle/vexRuFUVHoojnfah7): no description found

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1199787478211641425) (1 messages): 

- **Alexa's Greener Alternative**: `@mattbcool` is exploring how to create a local, personal assistant using Alexa hardware, aiming to repurpose speakers and mics to minimize waste. They have been researching recent projects with raspberry pis and documented their progress in a [personal blog post](https://mattcool.tech/posts/you-can-have-an-open-source-personal-assistant-in-2024).
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1199833369211584552) (2 messages): 

- **Deep Learning.ai Launches Automated LLM Testing Course**: `@manialgie` shared a [link to a short course](https://www.deeplearning.ai/short-courses/automated-testing-llmops/) in collaboration with **CircleCI**. The course aims to teach how to use **continuous integration** tools to evaluate **LLM applications** more efficiently.
  
- **Text-to-3D Made Easy with 3DTopia**: `@meatfucker` discovered [3DTopia on GitHub](https://github.com/3DTopia/3DTopia), featuring **model weights** and **inference code** that promises **Text-to-3D Generation** within 5 minutes. They noted that they haven't tried it yet, but it looks promising for easier 3D generation from text.

**Links mentioned**:

- [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/): no description found
- [GitHub - 3DTopia/3DTopia: Text-to-3D Generation within 5 Minutes](https://github.com/3DTopia/3DTopia): Text-to-3D Generation within 5 Minutes. Contribute to 3DTopia/3DTopia development by creating an account on GitHub.

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1199650676033196152) (3 messages): 

- **Tackling Live Transcription Challenges**: `@apocalypse3917` raised concerns around the problem of **background noise** in live transcription, questioning whether there's a client-side solution or an auto-calibration feature to handle this. `@ggabe_2` responded with appreciation and explained that their **Proof of Concept (PoC) for Whisper** didn't specifically address ambient noise, but suggested that Voice Activity Detection (VAD) filters could partly mitigate the issue.
  
- **New Python Module for Steering Vectors**: `@mihai4256` announced the creation of a **Python module** that works with **Hugging Face's transformers** to add steering vectors. They shared their accomplishment with a link to a tweet, which likely contains more details about the module: [View Tweet](https://twitter.com/m_chirculescu/status/1750149970026479720).
  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1199646847199887411) (3 messages): 

- **Vector Search Simplified**: `@united_dove_38339` shared an informative [blog post](https://blog.kusho.ai/a-primer-on-vector-search-using-pinecone-serverless/) detailing how to implement vector search using **Pinecone Serverless**, which is essential for modern applications leveraging LLMs and RAG. Pinecone's serverless solution aims to ease and reduce the cost of vector search implementation.

- **Presentation Prep Update**: `@chad_in_the_house` informed the group of a delay in the presentation preparation, noting the complexity of the papers involved. The presentation is expected to be ready by the end of the week or by next Friday.

- **Unsloth Accelerates Fine-tuning**: `@zigglewomp` introduced an article on [Medium](https://medium.com/@drishtisharma96505/accelerate-llama-2-7b-fine-tuning-how-unsloth-outpaced-standard-and-flash-attention-ef2ba8a1131d) discussing the **Unsloth** technique, which has shown promise in improving memory efficiency and training performance for the Llama2–7b model.

**Links mentioned**:

- [A primer on vector search using Pinecone Serverless](https://blog.kusho.ai/a-primer-on-vector-search-using-pinecone-serverless/): Pinecone recently launched its serverless offering with the goal of making vector search implementation easier and cost-efficient. With LLMs rapidly becoming a core part of applications and most of th...
- [Accelerate Llama-2–7b Fine-tuning: Unsloth Outpaces Flash Attention-2](https://medium.com/@drishtisharma96505/accelerate-llama-2-7b-fine-tuning-how-unsloth-outpaced-standard-and-flash-attention-ef2ba8a1131d): Objective of this Study

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199742668188680214) (5 messages): 

- **Quest for the Most Challenging Dataset**: User `@idkman2021` queried about the hardest dataset to work with. However, no specific answers or further discussion followed this question.
  
- **Video Synthesis Challenge Shared**: User `@archer_cs` sought ideas for a project on video generation using a target audio and a reference video. They shared a [GitHub discussion](https://github.com/huggingface/diffusers/discussions/6696) outlining the details of their ambitious project.

- **Improving Stream Structures in GPT-4**: User `@kiraultra` expressed difficulties with the streaming structure while using the **GPT-4 turbo API**, mentioning the issue of getting no bullet points until the very end of the stream. No specific solutions or follow-up questions appeared in the chat.

- **Discrepancy in Logit Outputs**: User `@sherlockzoozoo` posted a code block showing their implementation of using `meta-llama/Llama-2-7b-hf` model and tokenizer, and asked why there’s a difference between values in `out_toks['scores']` and `just_out['logits']`. The question stands without a response, leaving the nature of the logit discrepancies unexplained.

**Links mentioned**:

[Video generation using target audio and reference video. · huggingface/diffusers · Discussion #6696](https://github.com/huggingface/diffusers/discussions/6696): I am working on a personal project which involves : Input a reference video and a target audio, synthesise a target video (lip synced talking head video generation driven by the target audio). I wo...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1199668017030778901) (4 messages): 

- **Enhancing Custom PyTorch Models**: User `@nielsr_` provided a valuable tip for custom PyTorch models, recommending the use of **[mixins from the `huggingface_hub` library](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)** to add `from_pretrained` and `push_to_hub` functionalities with ease.
- **Seeking Idefics Project Insights**: `@besiktas` inquired about where to ask questions to the team behind the **Idefics** project; `@osanseviero` directed them to the project's **[discussion tab on the HuggingFace repo](https://huggingface.co/ideas/discussions)**.

**Links mentioned**:

[Mixins &amp; serialization methods](https://huggingface.co/docs/huggingface_hub/v0.20.3/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin): no description found

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1199651088865972296) (5 messages): 

- **Rate Limit Warnings in Azure API**: User `@kyko6969` is facing issues with rate limit warnings when using `embed_with_retry` from Azure's OpenAI API in conjunction with `langchain`. They are looking to catch the warning and implement a `time.sleep()` function to handle the API's rate limitation.
  
- **Training BPE Tokenizer for Pashto**: `@imranullah` inquired about training the BPE tokenizer for low-resource languages like Pashto, as he is encountering garbage text output. 

- **Efficient Fine-Tuning with LoRA**: `@gugaime` mentioned that it's possible to perform quantization during fine-tuning by utilizing LoRA, where the base model remains quantized but is frozen.

- **Open Source Text-to-Speech Recommendation**: `@mattbcool` introduced the **[Coqui AI TTS](https://github.com/coqui-ai/TTS)** toolkit as a potential open-source, local Text-to-Speech solution.

- **Mysterious Weights in mBARTforCausalLM**: `@vikas.p` asked about the lm head weight being set as tied in the `mbartforcausallm` model on the Hugging Face's transformers repository, seeking clarity on whether it is tied to the input embedding and if some method is hidden that does so.

**Links mentioned**:

- [
            Dynamics 365 Customer Voice
        ](https://aka.ms/oai/quotaincrease): no description found
- [transformers/src/transformers/models/mbart/modeling_mbart.py at main · huggingface/transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mbart/modeling_mbart.py#L1926): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - huggingface/transformers
- [GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production](https://github.com/coqui-ai/TTS): 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production - GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and pr...

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1199742668188680214) (5 messages): 

- **Seeking Dataset Challenge Opinions**: User `@idkman2021` inquired about ***the most challenging dataset*** but did not provide context for the term "challenging" nor specify the field or application.

- **Innovating Video Generation Techniques**: User `@archer_cs` asked for ideas on a project involving video generation from a reference video and target audio, linking to a [GitHub discussion #6696](https://github.com/huggingface/diffusers/discussions/6696) for more details. The project aims to create a lip-synced talking head video driven by the target audio.

- **Improving Streaming Structure in GPT-4 Turbo API**: User `@kiraultra` brought up an issue with the **streaming structure** when using the gpt-4 turbo API, mentioning that bullet points and other structural elements only appear at the end of the stream, seeking advice on enhancing this aspect.

- **Understanding Discrepancies in Model Outputs**: User `@sherlockzoozoo` shared a code snippet querying two outputs from the **LLama-2-7b model** and asked why there are different values in `out_toks['scores']` and `just_out['logits']`. They are seeking clarification on which of these represents the "real logits".

**Links mentioned**:

[Video generation using target audio and reference video. · huggingface/diffusers · Discussion #6696](https://github.com/huggingface/diffusers/discussions/6696): I am working on a personal project which involves : Input a reference video and a target audio, synthesise a target video (lip synced talking head video generation driven by the target audio). I wo...

  ,

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1199659240265097256) (41 messages🔥): 

- **Google Unleashes Lumière**: `@spirit_from_germany` shared a tweet by `@omerbartal` introducing **Lumiere**, a new video diffusion model by GoogleAI that supports text-to-video, image-to-video, stylization, and more. There's a buzz about the model's capabilities, but `@mkaic` points out there's no open sourcing, leading to mixed reactions in the group regarding its potential versus realism ([Read Paper on Lumière](https://arxiv.org/abs/2401.12945)).
  
- **LAION Database Access Concerns**: Users `@_chenwang` and `@djdhdjjdjdjdj` raised issues regarding downloading LAION-en-aesthetics captions, due to Huggingface disabling downloads.

- **Comparison Between Lumière and EMU Video**: There's an ongoing debate between `@mkaic` and `@thejonasbrothers` about whether Google's Lumière or Meta's EMU video models seem more realistic, with criticisms pointing to occasional inconsistencies and unnatural appearances in the AI-generated content.

- **Enthusiasm Met with Skepticism**: `@kilgore.trout` inquires about the best open-source models for video stylization against the backdrop of Google's new model Lumière, while others like `@.undeleted` comment on the still uncanny nature of AI-generated videos.

- **RWKV and Mistral Rivalry Discussed**: `@SegmentationFault` shared a Reddit link discussing RWKV 7B's performance, possibly reaching the levels of Mistral 7B in multilingual support with additional benefits like linear runtime and efficient CPU usage.



**Links mentioned**:

- [Tweet from Omer Bar Tal (@omerbartal)](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19): Introducing Lumiere 📽️  The new video diffusion model we&#39;ve been working on @GoogleAI  * Text-to-Video * Image-to-Video * Stylized Generation * Inpainting * Cinemagraphs and more 🎨  W/ amazing t...
- [Democratizing the future of AI R&amp;D: NSF to launch National AI Research Resource pilot](https://new.nsf.gov/news/democratizing-future-ai-rd-nsf-launch-national-ai): Alexandria, Virginia: Today, the U.S. National Science Foundation and collaborating agencies launched the National Artificial Intelligence Research Resource…
- [Lumiere - Google Research](https://lumiere-video.github.io/#section_image_to_video): Space-Time Text-to-Video diffusion model by Google Research.
- [Lumiere - Google Research](https://lumiere-video.github.io/): Space-Time Text-to-Video diffusion model by Google Research.
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/19essc5/rwkv_7b_is_appears_to_be_approaching_mistral_7b/): no description found

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1199659282065535057) (15 messages🔥): 

- **GoogleAI Unveils Lumiere - The Video Diffusion Model**: `@spirit_from_germany` shared a tweet from [Omer Bar-Tal](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19) announcing GoogleAI's new video diffusion model, **Lumiere**, which features *Text-to-Video, Image-to-Video, Stylized Generation, Inpainting, Cinemagraphs,* and @mkaic followed up with the [accompanying research paper](https://arxiv.org/abs/2401.12945).

- **GoogleAI's Lumiere Surprises with Simplicity**: `@mkaic` comments on how surprisingly simple the method behind **GoogleAI's Lumiere** appears upon initial skimming of the research paper, though they hadn't delved into details yet.

- **Lumiere's Potential Training Ground - YouTube**: `@mkaic` speculates that **GoogleAI's Lumiere** must be training on YouTube, given that it's the largest video repository, and later confirms it by quoting from the paper: "*We train our T2V model on a dataset containing 30M videos*".

- **Google’s Unrivaled Video Data for T2V Models**: `@mkaic` points out **Google's** massive advantage in training text-to-video models due to their ownership of YouTube, which includes auto-generated captions and billions of comments, indicating a substantial dataset for training video multimodal language models.

- **DeepMind Considered a Powerhouse for Project Gemini**: `@thejonasbrothers` underscores **DeepMind's** notable efforts by reflecting on the significant number of researchers involved in project **Gemini**, implying their massive resources and capacity in AI research.


**Links mentioned**:

- [Tweet from Omer Bar Tal (@omerbartal)](https://fxtwitter.com/omerbartal/status/1749971963403997252?s=19): Introducing Lumiere 📽️  The new video diffusion model we&#39;ve been working on @GoogleAI  * Text-to-Video * Image-to-Video * Stylized Generation * Inpainting * Cinemagraphs and more 🎨  W/ amazing t...
- [Lumiere: A Space-Time Diffusion Model for Video Generation](https://arxiv.org/abs/2401.12945): We introduce Lumiere -- a text-to-video diffusion model designed for synthesizing videos that portray realistic, diverse and coherent motion -- a pivotal challenge in video synthesis. To this end, we ...

  ,

### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1199755988388032582) (1 messages): 

- **Vanna AI's overnight sensation in SQL generation**: The `@zain_hoda` project **Vanna AI** is turning heads with its straightforward yet potent interface that leverages **RAG** (Retrieval Augmented Generation) for enhanced SQL query creation. The bot features abilities to store and index DDL/table schemas and text for its operations. [LlamaIndex Tweet](https://twitter.com/llama_index/status/1750196064660127848)
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1199631950646870056) (49 messages🔥): 

- **Quest for Refined AI Response Mechanics**: User `@viky6453` inquired if there's a way for the openaiagent to behave more like the openai assistant, which applies a tool call, message, and tool call repeatedly until the response is deemed good enough, instead of the LlamaIndex openaiagent style where it sends a single message as a response after multiple tool calls. No definitive solution was provided in the discussion.
  
- **Tech Explorer Seeking Efficiency**: In the quest for optimizing pandas query engine with response synthesizer to be context-aware, `@techexplorer0` expressed a desire for a chat_engine equivalent, while `@pk2594` wondered about threading the query engine to speed things up, questioning its thread-safety.

- **Seeking the Perfect LLM Chatbot Host**: User `@basil11111` pondered whether it's possible to use open-source models without local hosting and discovered from `@nerdai`'s response that services like HuggingFace and Replicate can host LLMs, offering APIs and fine-tuning capabilities.

- **Fine-Tuning RAG's Effectiveness**: Discussion touched upon the enhancement of RAG applications with `@0tarumi` exploring the implementation of BGE similarity reranker and `@cheesyfishes` suggesting that reranking after RRF might yield the best results.

- **Memory Matters**: `@techexplorer0` sought a tool for tracking conversational history akin to langchain's memory buffer, leading to `@cheesyfishes` confirming the use of a chat memory buffer in every chat engine/agent, which could be paired with llama-index in langchain for those seeking a conversational chatbot with memory.


**Links mentioned**:

- [Home - Phoenix](https://phoenix.arize.com/): no description found
- [LlamaIndexTS/examples/vectorIndex.ts at main · run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS/blob/main/examples/vectorIndex.ts): LlamaIndex is a data framework for your LLM applications - run-llama/LlamaIndexTS
- [LlamaIndexTS/packages/core/src/llm/azure.ts at main · run-llama/LlamaIndexTS](https://github.com/run-llama/LlamaIndexTS/blob/main/packages/core/src/llm/azure.ts): LlamaIndex is a data framework for your LLM applications - run-llama/LlamaIndexTS
- [Replicate](https://replicate.com/): Run open-source machine learning models with a cloud API
- [meta/llama-2-70b-chat – Run with an API on Replicate](https://replicate.com/meta/llama-2-70b-chat): no description found
- [meta-llama/Llama-2-70b-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-70b-hf): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (1 messages): 

rawwerks: 👋 community question❓
what is your favorite vector store company and why?
  ,

### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1199770976922112070) (2 messages): 

- **X/Twitter Account Recovery**: `@.bagatur` announced that the **X/Twitter account** has been recovered and will unblock everyone affected. If someone remains blocked, they are instructed to post in the thread for assistance.

- **LangChain Introduces Streaming API**: `@veryboldbagel` shared links to new API documentation for **streaming events** in LangChain, highlighting the importance for responsive end-user applications. Detailed examples and instructions are provided in the [General Docs](https://python.langchain.com/docs/expression_language/streaming) and for **AgentExecutor** and **LangGraph** ([AgentExecutor Docs](https://python.langchain.com/docs/modules/agents/how_to/streaming), [LangGraph Notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb)).

- **Watch and Learn About Streaming Events**: A [YouTube video](https://youtube.com/watch?v=ZcEMLz27sL4) titled "Streaming Events: Introducing a new `stream_events` method" was shared, which explains the significance of streaming in LLM apps.

- **Feedback Request for Streaming Feature**: Users are encouraged to provide feedback and report issues regarding the new streaming feature on LangChain's GitHub discussion page found [here](https://github.com/langchain-ai/langchain/discussions/16175).

**Links mentioned**:

- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/expression_language/streaming#using-stream-events): streaming-with-langchain}
- [Streaming | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/agents/how_to/streaming): Streaming is an important UX consideration for LLM apps, and agents are
- [langgraph/examples/streaming-tokens.ipynb at main · langchain-ai/langgraph](https://github.com/langchain-ai/langgraph/blob/main/examples/streaming-tokens.ipynb): Contribute to langchain-ai/langgraph development by creating an account on GitHub.
- [Streaming Events: Introducing a new `stream_events` method](https://youtube.com/watch?v=ZcEMLz27sL4): Streaming is an important part of most LLM apps. Both streaming of individual tokens, as well as streaming of events that happen along the way.We recently in...
- [🛸 Streaming: RFC Adding astream_event to all Runnable objects to help with streaming use cases · langchain-ai/langchain · Discussion #16175](https://github.com/langchain-ai/langchain/discussions/16175): Hi everyone! We want to improve the streaming experience in LangChain. We&#39;re considering adding a astream_event method to the Runnable interface. The code below is from the following PR and has no...

  

---


### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1199629518474858496) (21 messages🔥): 

- **In Search of Open Source Clarity**: `@pirlog` inquired whether the project was open source or free, noting the lack of pricing or explanatory information on the website.
- **Request for Assistance**: `@irfansyah5572` and `@shanumas` indicated a need for help with a non-specific issue, showing interest in collaborative problem-solving.
- **Service Downtime and Recovery Noted**: `@adorable_quokka_56531` mentioned that the services appeared down but later noted they were back up, also suggesting the addition of a status page.
- **Assistance with Database Schemas**: `@mavrik_55410` sought guidance on storing vector embeddings in a specific schema in a Postgres Database using pgvector and langchain, which led to a clarification discussion with `@__ksolo__` about Postgres schemas and configurations.
- **Community Feedback on Vector Storage**: `@rawwerks` opened a discussion on favorite vector storage companies, inviting community opinions.

**Links mentioned**:

[5.9. Schemas](https://www.postgresql.org/docs/current/ddl-schemas.html): 5.9.&amp;nbsp;Schemas # 5.9.1. Creating a Schema 5.9.2. The Public Schema 5.9.3. The Schema Search Path 5.9.4. Schemas and Privileges 5.9.5. …

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1199963765467263037) (1 messages): 

- **Seeking Guidance for LangServe Agent Executors**: `@hiranga.g` faced difficulties getting an **agent_executor** to run in LangServe and sought assistance from the community. They posted a direct query to `@1033432389516546158` regarding the setup process. (Q1)

- **Clarification on LCELs and Tool Selection**: `@hiranga.g` achieved getting a **LCEL** to work but was uncertain if it is possible to add multiple Tools for the LCEL to utilize. They expressed a belief that LCELs don't allow for creating an "agent" and inquired for confirmation or guidance on the matter. (Q2)
  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 

sideways1: Has anyone built a Q&A chatbot that interacts with a database of JSON files?
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1199859575508172891) (4 messages): 

- **AgentHub Reveal**: `@maxbrodeururbas` announced the launch of [AgentHub](https://www.agenthub.dev/), a platform built with a friend, and invites feedback from the community. They added that they've written a blog post elaborating on the synergy between Robotic Process Automation (RPA) and AI which can be found [here](https://www.agenthub.dev/blog/robotic_process_automation_with_ai).
- **Call for Collaboration**: `@truethinker` reached out to `@939753620423991296` to express interest in connecting, although no context or detail was provided regarding the purpose of the connection.

**Links mentioned**:

[AI and RPA: The Future of Work](https://www.agenthub.dev/blog/robotic_process_automation_with_ai): The marriage of RPA tooling and AI is going to cause a monumental explosion in productivity in the next few years.

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1199750400849485894) (2 messages): 

- **Dive into Multimodal AI with DataCamp and LangChain**: `@datarhys` shared a link to a **free 9-part series on AI**, including a session on "Building Multimodal AI Applications with LangChain & the OpenAI API". This session teaches participants to transcribe YouTube videos using Whisper and then pose questions to GPT about the transcribed content. Check out the [entire code-along series](https://www.datacamp.com/code-along) and start the specific LangChain code-along with this [code-along link](https://www.datacamp.com/code-along/multimodal-ai-applications-langchain-openai-api).

- **CircleCI and Deeplearning.ai Present AI Testing Course**: `@manialgie` announced the release of a **free course** in collaboration with Deeplearning.ai on how to test and ship AI-powered applications. The course explores testing LLM-based applications, model-graded evaluations, and automating these processes with CircleCI to enhance application development. Find the course at [Automated Testing with LLMOPS](https://www.deeplearning.ai/short-courses/automated-testing-llmops/).

**Links mentioned**:

- [Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/): no description found
- [Building Multimodal AI Applications with LangChain &amp; the OpenAI API](https://www.datacamp.com/code-along/multimodal-ai-applications-langchain-openai-api): Combine the power of text and audio AI models to build a bot that answers questions about YouTube videos. 
- [Data Science &amp; AI Code Alongs](https://www.datacamp.com/code-along): From data visualization to AI, code along with experts as they solve real-world problems. Work your way through an entire project with the help of a screencast, so you never get stuck.

  ,

### DiscoResearch ▷ #[disco_judge](https://discord.com/channels/1178995845727785010/1178996063537991752/1199824286257516604) (1 messages): 

- **GPT-3.5 Struggles with Importance and Sentiment**: `@calytrix` shared that their pipeline using **GPT-3.5** to rate news stories is better at judging sentiment than importance but still has *strong biases*. They noted that adding explanations and descriptive scores did little to improve the system's discriminative power.
- **Tackling Model Bias and Rating Challenges**: `@calytrix` observed very strong preferences in the numbers the model assigns, suggesting that fine-tuning could mitigate these biases. They emphasized the complexity of the tasks and suggested developing a specific evaluation and scoring system tailored to the model's capabilities.
- **Complexity in Assessment Recognized**: The task of rating importance is particularly difficult for GPT-3.5 due to the **implied complexity** `@calytrix` mentioned, something humans find easy.
- **Rating System Recommendations**: `@calytrix` recommended breaking down complex questions, like assessing importance, into more specific ratings that GPT-3.5 can handle more effectively. They also suggested a simplified rating system with limited options, such as **"poor, mixed, good, very good"**.
  

---


### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1199631228857499669) (9 messages🔥): 

- **Gauging Data Quality for Language Model Fine-Tuning**: User `@thewindmom` questioned how to evaluate data quality when fine-tuning language models, emphasizing the avoidance of poor input data. `@hammadkhan` suggested techniques like **eye-balling, deduplication, and heuristic-based filtering** for maintaining data quality.

- **DIY Synthetics Reduce the Need for Evaluation**: In response to `@thewindmom`, `@bjoernp` proposed that creating your own synthetic data can lower the need for external evaluation, hinting at the use of **built-in guides** and **information-rich** models to produce such data.

- **Discussing Synthetic Data Generation for Fine-Tuning**: `@thewindmom` asked about the use of synthetic data, particularly referencing **NeMo Guardrails**, to which `@bjoernp` mentioned the proactive creation of synthetic data with guardrails that a strong model can utilize effectively for training.

- **Sharing Resources on Synthetic Data and Self-Instruct Methods**: `@_jp1_` shared that DiscoResearch is still refining their synthetic data generation process and highlighted **jon durbins' Airoboros repo** as a key reference, while discussing customized implementations for the self-instruct paper. They provided a [GitHub link to Airoboros](https://github.com/jondurbin/airoboros) for further insights and inspirations.

- **Training Dilemmas: Using Instruction+Answer Pairs**: In a discussion led by `@philipmay`, the query raised concerns about using **instruction+answer pairs**. They inquired if something useful could be done with just pairs of instruction and bad answer for model training, given the absence of a good answer to create **triplets** for DPO or other training efforts.

**Links mentioned**:

[GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros): Customizable implementation of the self-instruct paper. - GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.

  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1199665763955507270) (11 messages🔥): 

- **DPR Models Disappoint**: `@philipmay` expressed that testing with two different DPR models did not yield the expected results, calling the situation "very strange".
- **Efforts on Enhancing Data Positioning**: `@sebastian.bodza` plans to add a new column showing the position data was found at, while also acknowledging the complexity of the task mentioning, "Quite tricky".
- **Summing Distances for Question Specificity**: `@philipmay` discussed a strategy for determining the generic nature of questions by summing distances of the top results, noting that generic and specific questions don't get effectively distinguished this way.
- **Challenges in Discerning Patterns**: `@sebastian.bodza` commented on the difficulty of finding a pattern in the similarity of text and question or between top vectors, specifically highlighting issues related to questions about "Pferdemarkt" due to its multiple instances.
- **Brainstorming Completion and Next Steps**: `@sebastian.bodza` announced the completion of brainstorming with 82k search ideas/questions, emphasizing the next phase will be question generation.
  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1199796648063283200) (5 messages): 

- **Praising DiscoLM German 7B v1**: `@alex_22398` expressed gratitude for the high-quality language outputs of the **DiscoLM German 7B v1**, even on an older laptop with the help of TheBloke's GGUF quantization, while pointing out an issue with the generation of blank lines after output.
- **Blank Line Bug Squashed**: `@_chromix_` informed about the fix for excess blank lines after output and discussed options for generating an updated **GGUF quantization** or setting the stop token manually to `"

**Links mentioned**:

[DiscoResearch/DiscoLM-70b · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM-70b): no description found


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1199870824451162253) (2 messages): 

- **Catching up with the team**: `@teknium` reached out to the channel members, expressing a desire to reconnect and encouraging discussion on recent activities amongst colleagues by tagging `@748528982034612226`. No specific topics or links were mentioned.
  ,

### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1199630613586641006) (2 messages): 

- **Open Inquiry on ML Field Jobs**: User `@forsaken_ninja` reached out to the community with queries regarding job opportunities in the machine learning field, inviting members to engage in the discussion and offer insights. They encouraged others to ping them for a direct conversation.
  ,,,,



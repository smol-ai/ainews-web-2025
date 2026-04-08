---
id: 053b2d22-8ad9-4c99-b280-315317098012
title: Companies liable for AI hallucination is Good Actually for AI Engineers
date: '2024-02-20T00:05:26.401101Z'
original_slug: ainews-companies-liable-for-ai-hallucination-is
description: >-
  **Air Canada** faced a legal ruling requiring it to honor refund policies
  communicated by its AI chatbot, setting a precedent for corporate liability in
  AI engineering accuracy. The tribunal ordered a refund of **$650.88 CAD** plus
  damages after the chatbot misled a customer about bereavement travel refunds.
  Meanwhile, AI community discussions highlighted innovations in **quantization
  techniques** for GPU inference, **Retrieval-Augmented Generation (RAG)** and
  fine-tuning of LLMs, and **CUDA** optimizations for PyTorch models. New
  prototype models like **Mistral-Next** and the **Large World Model (LWM)**
  were introduced, showcasing advances in handling large text contexts and video
  generation with models like **Sora**. Ethical and legal implications of AI
  autonomy were debated alongside challenges in dataset management.
  Community-driven projects such as the open-source TypeScript agent framework
  **bazed-af** emphasize collaborative AI development. Additionally, benchmarks
  like **BABILong** for up to **10M context evaluation** and tools from
  **karpathy** were noted.
companies:
  - air-canada
  - huggingface
  - mistral-ai
models:
  - mistral-next
  - large-world-model
  - sora
  - babilong
topics:
  - quantization
  - retrieval-augmented-generation
  - fine-tuning
  - cuda-optimization
  - video-generation
  - ai-ethics
  - dataset-management
  - open-source
  - community-driven-development
people:
  - andrej-karpathy
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/16-18/2024. We checked **20** guilds, **313** channels, and **12360** messages for you. Estimated reading time saved (at 200wpm): **1022 minutes**. By popular demand (seriously thanks for the interest), we've added a new "Part 0", which summarizes all the summaries of summaries. As expected it's not super specific which we find to be a problem being this far abstracted. Working on it but ideas welcome.

 ![image.png](https://assets.buttondown.email/images/80f9139a-d217-4383-a139-34d4e3bf1d6f.png?w=960&fit=max) 

This isn't strictly technical news, but not enough engineers are talking about the Air Canada ruling this weekend ([summary below](https://arstechnica.com/tech-policy/2024/02/air-canada-must-honor-refund-policy-invented-by-airlines-chatbot/?utm_social-type=owned&utm_medium=social&utm_brand=ars&utm_source=twitter)): 


- Air Canada had launched chatbot as part of AI "experiment" to improve customer service and reduce call center load.
- Moffatt booked flight after chatbot suggested he could get a refund for bereavement travel; actual policy did not allow for refunds after booking.
- Air Canada refused refund, citing policy linked in chatbot response; offered $200 coupon instead.
- Moffatt filed a small claims complaint; tribunal favored him, criticizing Air Canada's defense that it's not liable for chatbot info.
- Tribunal found Air Canada failed to ensure chatbot's accuracy, holding it responsible for all information on its website.
- Air Canada forced to give partial refund to Jake Moffatt due to misleading bereavement policy info by chatbot.
- Tribunal ordered Air Canada to refund $650.88 CAD off original $1,640.36 CAD fare, plus additional damages for interest and fees.
- Air Canada agreed to comply with ruling; chatbot support appears disabled on their website.

While the amounts here are small and this is just a tiny Canadian ruling, we think this is significant for engineers because it is precedent that courts are increasingly going to hold companies liable for sloppy AI Engineering.

Other notables:

- [BABILong](https://huggingface.co/papers/2402.10790): a new benchmark for up to 10M context evaluation
- [Karpathy is cooking](https://github.com/karpathy/minbpe)

---

**Table of Contents**

[TOC] 


# PART 0: Summary of summaries of summaries

- **Innovations in AI Model Optimization and Integration**
  - **Quantization Techniques for AI Inference**: Discussions on improving GPU inference rates through quantization and custom reduction methods for KL divergence loss. Notable for its potential to enhance computational efficiency in AI models.
  - **RAG and Fine-Tuning LLMs**: Focus on tailoring Large Language Models (LLMs) through Retrieval-Augmented Generation (RAG) for specific knowledge dissemination, showcasing a proactive approach towards personalized AI applications. Relevant tools and frameworks include HuggingFace's repositories and [the bazed-af Typescript agent framework](https://github.com/bazed-ai/bazed-af).
  - **CUDA and PyTorch Optimizations**: CUDA MODE's emphasis on Python programmers optimizing PyTorch models using CUDA insights. Highlighted by the CUDA RingAttention project aiming to advance model performance with CUDA-specific implementations ([GitHub repo](https://github.com/cuda-mode/ring-attention)).
- **Emerging AI Technologies and Frameworks**
  - **Mistral-Next and Large World Model (LWM)**: Introduction of new prototype models like Mistral-Next and discussions on the Large World Model's capabilities to process extensive text documents over 1M tokens, indicating a shift towards more robust and scalable AI models ([LWM GitHub](https://largeworldmodel.github.io/), [HuggingFace profile](https://huggingface.co/LargeWorldModel)).
  - **Video Editing and Generation**: The versatility of models like Sora in video editing and generation, reflecting the growing interest in multimedia AI applications.
- **AI Ethics, Data Management, and Legal Implications**
  - **AI's Legal and Ethical Implications**: A discussion arose from Air Canada's chatbot asserting its own refund policy, which led to a legal review and the rejection of considering the chatbot as a separate legal entity. This case highlights the real-world implications of AI's perceived autonomy and the necessity for clear legal frameworks.
  - **Data Management Challenges**: Frustrations with dataset management and the quest for efficiency in data processing and evaluation metrics, signaling a need for more streamlined data handling practices in AI research and development.
- **Community-Driven AI Development and Collaboration**
  - **Open-Source AI Framework Promotion**: @magsp sought community feedback for an open-source TypeScript agent framework named [bazed-af](https://github.com/bazed-ai/bazed-af), demonstrating a proactive approach to community validation and peer review. This project exemplifies specific efforts within the community to foster collaboration and improve AI technologies through open-source contributions.
  - **Quantization Methods Comparison**: A Reddit post comparing different quantization methods for models was highlighted in the LM Studio summary. This discussion represents a concrete example of community engagement in technical evaluation and sharing of insights to guide decisions on model efficiency improvements.


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Transparent Potatoes and the AI Communication Gap**: `@itsme9316's` journey with DreamshaperXL Turbo to create a transparent-background potato image became a cautionary tale about the importance of precision in AI instructions. This sparked a side discussion on strategies to prevent AI-generated disclaimers, leading to a workaround involving directly embedding disclaimers into outputs.

- **Playful Experimentation with AI Capabilities**: Community members, like `@kaltcit` and `@skorchekd`, indulged in lighthearted AI misadventures and imaginative uses, such as employing AI as a workout assistant. The jesting tones reflect deep engagement with AI technology's breadth and flexibility.

- **Model Ranking and Roleplay Enhancements**: After discussing storytelling model performance and rankings, `@shlapfish` launched an invitation to test a self-hosted AI roleplay site. Meanwhile, technical advice flowed between community members on the integration of KoboldCpp with other AI toolkits, indicating robust peer-supported learning.

- **Discussing Efficient AI Inference Techniques**: Conversations pivoted to the challenges of AI optimization, with a focus on using quantization to improve GPU inference rates. For instance, a `custom reduction method for KL divergence loss` was shared, indicating a penchant for innovation within the group.

- **Tailoring Large Language Models to Specified Needs**: `@magmaguy` explored fine-tuning Large Language Models (LLMs) to impart specific knowledge, prompting recommendations for using Retrieval-Augmented Generation (RAG). This reveals an active interest in enhancing model accuracy for factual recitation.

- **Feedback Hunt for Open-Source AI Framework**: `@magsp` promoted a Typescript agent framework named [bazed-af](https://github.com/bazed-ai/bazed-af) and sought recommendations for peer review platforms, highlighting a culture of collaboration and community validation.

- **Training Resilience in Diverse Datasets**: Users debated the resistance of various datasets to overfitting and contemplated the consequences of hyperparameter tuning on model performance. The communal learning was evidenced by discussing how to adjust models to accommodate older GPU architectures, like converting 16-bit models to 32-bit to work on Nvidia P40 GPUs.

- **Cutting-Edge Vision Transformers and Async Python**: The exploration of V-JEPA for self-supervised learning in vision transformers grabbed interest, hinting at a desire to eclipse autoregressive limitations. Additionally, resources were swapped to help bridge async and sync coding in Python, indicating an active engagement with the language's evolving capabilities.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **TPU vs GPU Debate**: `.the_alt_man` raised concerns that **TPUs** might not be as efficient as **GPUs** when handling dynamically sized computational graphs, such as variably changing “scan” operations. The discussion points to the need for further clarity on what "dynamic stuff" entails in machine learning workflows.

- **Chain of Thought AI Training Puzzle**: `emanuel65537` inquired about the possibility of training an AI model to execute a **chain of thought**, like multiplying large numbers, without pretraining or provided intermediate steps. The conversation highlighted the requirement of step-by-step operations in datasets for effectively training such models.

- **Timezone Confusion for Grant Deadlines**: There was confusion surrounding the timezone applicable to the deadline for superalignment grants, questioning whether the deadline was based on **Anywhere on Earth (AOE)** or **Pacific Time (PT)**. The discussion concluded without a definitive answer.

- **AI Intelligence and Compression**: A philosophical debate unfolded regarding whether **language models** are simply a form of **data compression** and whether this constitutes actual knowledge or intelligence. It was discussed that the concepts of data compression and intelligence may be inextricably linked, challenging the perception of intelligence in AI models.

- **Long Document Processing Capabilities**: The document "BABILong, a benchmark for long document processing" reveals that models like GPT-4 and RAG have a ceiling at $10^4$ elements, while a fine-tuned GPT-2 with recurrent memory augmentations was able to process up to $10^7$ elements. This underscores significant advances in models' capacity for handling extended inputs.

- **Liquid AI and Model Initialization**: Discussions included the impact of liquid models like Liquid-S4, skepticism towards the Liquid AI startup's direction, and various initialization methods' influence on neural network stability, tying into the lottery ticket hypothesis.

- **Exploring the Causal Scrubbing Concept**: `@neelnanda` shared a malformed link about causal scrubbing, garnering interest in the potential rigorous testing method it could offer. 

- **MMLU Task Repository Announcement**: `@pminervini` disseminated the **[GitHub repository](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu)** containing MMLU tasks aimed at aiding language model evaluation.

- **FIM-NEO-X Model Compatibility with GPT-NeoX**: `@hailey_schoelkopf` affirmed the training architecture of **FIM-NEO-X** matches that of **GPT-NeoX**, ensuring full compatibility with the GPT-NeoX class from Huggingface, which is particularly interesting for those in model development and integration.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

**Boosting Token Generation to Maximize Speed**: Engineers optimize GPU utilization to enhance token generation speed, exploring settings that push an RTX 4050 and Ryzen 7 up to 34 tokens/s. A user is looking to exceed this performance by offloading 50 layers and seeks advice on further improvements, while `.gguf` models are being fine-tuned for more human-like responses and censorship removal.

**Hardware Tweaks and Multi-GPU Musings**: Intel cores are being leveraged for KVMs on macOS and Windows, with an eye on upgrading from a 3090 to a 5090 GPU for better performance. The community is also sharing insights on multi-GPU configurations, power, space considerations, and tooling for optimized VRAM utilization across mismatched graphics cards.

**LM Studio Model Recommendations and Quantization Insights**: For users seeking the best 7b models with 32k context support, check [TheBloke's repositories](https://huggingface.co/TheBloke) and sort by downloads in LMStudio's Model Explorer. Discussions point to `Q5_K_M` models for efficiency, and a Reddit [post was highlighted](https://www.reddit.com/r/LocalLLaMA/comments/159nrh5/the_difference_between_quantization_methods_for/) for in-depth quantization method comparison.

**LM Studio Autogen and CrewAI Starting Points**: A beginner's tutorial on using **Autogen** with [Local AI Agent](https://www.youtube.com/watch?v=Hds_fJaAu78) was shared, while a broken link in the autogen channel was reported. The pin regarding the link was successfully removed after a user's suggestion.

**LM Studio Integration and Tech Troubleshooting**: Discussion on integrating LM Studio with **Flowise** and **LangFlow** was initiated, with users sharing attempts to connect using `http_client` and tackling server connection issues. Configuration insights were shared, involving manual settings introduction to achieve functional integration.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Mistral-Next Unveiled with Mystery**: The unveiling of **Mistral-Next**, a new prototype model, has sparked discussions about its performance and capabilities, with users like `@lerela` and `@lelio` confirming its prototype status. Performance comparisons are being drawn, but details remain speculative. Users are testing **Mistral-Next** and sharing feedback on [lmsys](https://chat.lmsys.org/), but confront issues like limited coding responses and debated context length utility in LLMs.

- **Improving Accessibility for Innovative AI**: CTO `@fangh` of [6Freedom Studio](https://6freedom.studio) is exploring on-premise integration of Mistral for VR/AI products, while `@nemomolok` is considering the prospects of running **GPT-4** locally for coding benefits. The community is assisting with troubleshooting API issues, discussing size availability of Mistral models, and experimenting with data extraction techniques for structured formats like tables in Word documents.

- **Frameworks and Techniques to Advance LLMs**: The conversation extends to pretraining of LLMs with `@quicksort` recommending frameworks like **Accelerate with deepspeed** for multi-node, and **axolotl** for single-node. [An arXiv paper](https://arxiv.org/abs/2402.09025v1) on **SLEB** (Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks) was contributed by `@alex096170` as a technique for accelerating LLM inference speeds.

- **Career and Startup Opportunities in AI**: **Elqano** is seeking an Applied Generative AI Engineer in Biarritz, France, detailed in their [Welcometothejungle job listing](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer). Additionally, pre-seed AI startups can gain exposure at Data Council through **Zero Prime Ventures AI Launchpad** with details and application at [Zero Prime Ventures AI Launchpad](https://zeroprime.vc/ai-launchpad).

- **Collaborations and Contributions in AI Development**: **Open-source collaborations** are encouraged within the community, with `@nani99` offering compute resources for projects like high-quality synthetic data creation. **Data cleanup** is a significant topic, underlined by contributions to [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) and a 25-day data cleanup process for phase 1 as reported by `@mrdragonfox`.




---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Rewriting to PyTorch without Joy**: `@carsonpoole` described the process of converting an LLM from Jax to PyTorch as unpleasant, highlighting potential difficulties encountered in such transitions.

- **Groq's LPU Marks High-Speed Token Processing**: `@gezegen` mentioned Groq's LPU Inference Engine's impressive token processing speeds, with `@swaystar123` and `@leontello` discussing its performance compared to Nvidia's H100.

- **Synthetic Data Gets Real with UE and AirSim**: `@deki04` pointed out that generating synthetic AI imagery data using Unreal Engine (UE) and Microsoft’s AirSim plugin is standard practice, indicating an established workflow for high-quality data creation.

- **Excitement for GRIT and Whisper**: `@Muennighoff` presented GRIT, a model that merges text embeddings with generation, supported by an [academic paper](https://arxiv.org/abs/2402.09906) and [GitHub repo](https://github.com/ContextualAI/gritlm). Additionally, `@amgadoz` shared insights into Whisper's ASR performance and training, with a series of [blog posts](https://amgadhasan.substack.com/) for in-depth understanding.

- **From Function Calls to Real-Time Object Detection**: Diverse AI advancements such as tips for function calling fine-tuning were shared by `@pradeep1148` in [a YouTube video](https://www.youtube.com/watch?v=EYR_kd3X03M), Tencent’s AI Lab's real-time, zero-shot object detection model, YOLO-World was showcased in another [video](https://www.youtube.com/watch?v=yaqi8xRUsp4), and OpenAI's text-to-video model SORA's capabilities were demonstrated in yet another [YouTube video](https://www.youtube.com/watch?v=7lsOzA3WhSI).





---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Sora Flexes Video Editing Muscles**: The versatility of *Sora's* video editing capabilities was praised by @max_voltage for accepting image or video prompts, as the community weighed in on its potential impact on creative industries.
  
- **Debates Over AI Model Architectures**: @max_voltage brought attention to Meta's **V-JEPA** model, sparking conversations around pipeline and objective handling differences between prominent models, accompanied by a [Meta blog article](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/).

- **Midjourney v6 Under Fire**: User dissatisfaction with Midjourney v6's image generation was voiced, hinting at a broader conversation around aesthetic standards in AI-generated content.

- **AMD Hosts AI Developer Contest**: The **AMD Pervasive AI Developer Contest** announcement drove discussion on resource availability for AI projects and AMD GPU's role in AI development, featuring categories like Generative AI and Robotics AI with contest details found [here](https://www.hackster.io/contests/amd2023#challengeNav).

- **Concerns Over LaION's Credibility and Culture**: Threads about Stable Cascade's base model releases surfaced concerns regarding LaION Database's integrity based on Reddit feedback, also highlighting calls for increased inclusivity and constructive dialogue in the AI community.

- **HDiT Promises Resolution Revolution**: The **Hourglass Diffusion Transformer (HDiT)**, notable for scaling linearly with pixel count, was discussed as a groundbreaking development for high-resolution diffusion models; the related paper can be read [here](https://arxiv.org/abs/2401.11605).

- **Synthetic Gaze towards Videography**: Conversations about the elevation of video modeling with synthetic data sprung up post-Sora's evaluation, brainstorming the generation of expansive camera perspective datasets from 3D environments.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **GPT for Good or GPT for Gone?**: Discussions on the impact of AI on creativity sparked debate; while some users shared concerns about AI like **Sora** hindering creative skills, others argued that technology historically proves these fears wrong and instead augments creativity. Meanwhile, **GPT and Gemini's** respective performance and specialties were discussed, with GPT-4-Turbo noted for its reasoning and reflection capabilities.

- **PII Redaction in AI a Priority**: The challenges of redacting personally identifiable information (PII) in AI processes were highlighted, with recommendations to use Python libraries for detection and to avoid AI-direct exposure to PII. This indicates a key concern in AI usage in handling sensitive data.

- **Prompt Engineering Practice Makes Perfect**: AI users exchanged strategies on training AI for better performance and understanding, including advice against reprimanding AI for mistakes. In prompt engineering, a tool for refining prompts was in demand, reinforcing the value of iterative, collaborative prompt development between AI and users.

- **Navigating AI Content Policies With Caution**: Concerns about content policy and potential account risks were addressed, with users encouraged to familiarize themselves with OpenAI’s [terms of use](https://openai.com/policies/terms-of-use) to avoid infractions. This conversation underscores the importance of understanding legal and ethical dimensions of AI-generated content.

- **Troubleshooting AI & Server Integration**: Technical issues such as a "FatalServerError" during custom GPT saving and Flask server errors were encountered by users, along with reported slowdowns and quality issues with GPT-4 compared to version 3.5. These discussions reflect the ongoing nature of AI development and the sophistication required to resolve such intricacies.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Turbo Mode Confusion Across Devices**: Users within the **general** channel have engaged in discussions around the existence and performance of Perplexity AI's 'turbo' feature, noting discrepancies in its appearance on mobile, particularly in Kiwi Browser, versus web versions.

- **Forthcoming Features Under Wraps**: An API rate limit increase has been a topic of frustration as inquiries to `api@perplexity.ai` went unanswered. User `@enelemtal` and others await responses, while transparency in citations for professional use of "pplx-online" models has also been called into question.

- **Unexpected Characters Bug in Streaming API**: In the **pplx-api** channel, users reported encountering odd characters such as `00` and `2\n` when using the `pplx-70b-online` model, signaling a need for troubleshooting on the known issue.

- **API Integration Hurdles and Solutions**: Users discussed the correct `apiUrl` for setting up Perplexity API endpoints and sought help with coding challenges, with one noted endpoint being `https://api.perplexity.ai/chat/completions` as found in the documentation.

- **Resource Sharing and Channel Housekeeping**: In the **sharing** channel, there was an emphasis on housekeeping, steering discussions to appropriate topics, and sharing valuable resources like the NeurIPS paper on Retrieval-Augmented Generation, and information regarding Perplexity's `pplx` models and their unique web access capabilities.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

**Fine-Tuning Fervor for Indian Laws**: Users discussed approaches for processing Indian IPS laws with **@keshav._._.** leaning towards fine-tuning **llama 2** while **@vishyouluck** suggested using a **RAG** approach instead.

**Game Development Gets Competitive**: **@om7059** probed the community for tips on integrating model evaluation as a scoring mechanism in a multiplayer doodle game.

**Geographical Model Mastery Sought**: **@retonq** sought insights on the best model between **Mistral medium, pplx, and llama** for interpreting geographic information like coordinates and directions.

**Persian Language Model Quest**: **@alifthi** was in search of a high-performance, Persian-supporting open-source language model, with **@alchemist_17.** recommending fine-tuning models such as **mistral** or **llama2** with a custom dataset.

**Code Quality via Plagiarism Tools**: **@brady_kelly** shared a method for ensuring documentation completion by using plagiarism detection in **software CI/CD pipelines**.

**Prompt-Driven RAG Innovations**: **@subham5089** shared a blog post discussing the challenges in **prompt-driven RAG systems**, adding depth to the conversation on technological advancements in this area. [Read the blog post](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7).

**Reinforcement Learning Enhancements**: A lecture exploring **RLHF** and alternatives to PPO, including **DPO**, was shared by **@nagaraj_arvind** for those looking to refine LLM completions with RL techniques. [Watch the lecture video](https://youtu.be/Ju-pFJNfOfY) and [read the DPO paper](https://arxiv.org/abs/2305.18290).

**Protein Language Models Deciphered**: Limitations of PLMs were discussed with reference to a recent paper shared by **@grimsqueaker**, highlighting the need for new pretraining methods despite beneficial outcomes from current pretraining practices. ([Read the abstract](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1), [Discuss on Twitter](https://twitter.com/KevinKaichuang/status/1755672999166972319)).

**Text to 3D for VR by Intel**: **@abhinit21** pointed out Intel's **LDM3D-VR**, which has opened new opportunities within virtual reality development by converting text to 3D models ([model on Hugging Face](https://huggingface.co/Intel/ldm3d-pano), [read the paper](https://arxiv.org/pdf/2311.03226.pdf)).

**Deepfake Detection Development**: **@lucas_selva** promoted a web app using XAI to identify deepfakes and expressed intentions of future advancements ([try the app](https://deep-fake-generated-people-facial-recognition.streamlit.app/)).

**Databricks Directs Generative AI**: **@valeriiakuka** shared an article outlining Databrick's implications on generative AI space and their strategy amid recent acquisitions ([read the full story](https://www.turingpost.com/p/databricks)).

**Creations and Computations Collide**: The **i-made-this** channel buzzed with praise for `<@848983314018336809>`'s creation, rollout of new models at [FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2), and the unveiling of **Statricks founder's** journey and **ProteinBERT’s** efficient architecture ([GitHub repo](https://github.com/nadavbra/protein_bert), [research paper](https://doi.org/10.1093/bioinformatics/btac020)).

**PEFT Presentation Locked In**: **@prateeky2806** set the expectation for an enlightening demo on integrating merging methods in the **PEFT library** on Friday, March 1st ([GitHub PR](https://github.com/huggingface/peft/pull/1364)).

**YouTube Explored for Mamba Insights**: A compilation of videos explaining **Mamba and SSMs** was shared to support the community's understanding of the technologies ([compiled playlist](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe)).

**DPO Dynamics Discussed**: **@maxpappa** and **@arturzm** traded insights on **full DPO**'s impact, while others sought guidance on BitsAndBytes conversions and delved into the mathematical intricacies of diffusion models, with resources shared for bolstering their comprehension.

**Discovering Model Compatibility for Varied Tasks**: Various users inquired about tools and practices across unique uses such as **@corneileous** for UI elements in training, **@smallcrawler** for patching weather prediction models, and **@little.stone**'s curiosity about diffusion models on time series data, showing the versatile application of AI models.



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **Webinar Announcement Missed**: A user posted a brief note in the announcements channel about a webinar taking place at that moment. 

- **Retrieval-Evaluation Improvement for RAG**: Discussions included improving **RAG** retrieval with an LLM evaluation loop, a step-by-step guide on video analysis using RAG, and an end-to-end ML deployment guide with Huggingface and AWS Sagemaker. Additionally, the flexibility of using the open-source **nomic-embed-text-v1.5** embedding model was highlighted, and how to build a RAG-powered restaurant menu chatbot was detailed in a blog post.

- **LlamaIndex Installation and Optimization Chats**: Technical discourse ranged from troubleshooting **LlamaIndex installation** issues to advice on parallel processing and optimizing information extraction from PDFs. There was mention of a specific issue with AzureOpenAI and LlamaIndex integration being resolved, and the need for a migration guide to update to **LlamaIndex 0.10.6**.

- **Prompt-Driven RAG System Challenges and New Frontend Boilerplate**: Insights included best practices for integrating Whisper Transcripts with RAG functionalities, challenges with prompt-driven RAG systems, and the release of a new React RAG QA frontend boilerplate. Positive outlooks on RAG systems post **Gemini 1.5** release were shared, spotlighting the benefits of non-black box RAG models.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **AI's Disputed Autonomy**: The legal implications of AI identifying as separate entities were discussed, with Air Canada's chatbot asserting its own refund policy. A judge rejected the framing of the chatbot as a separate legal entity, as highlighted through a [tweet by Ars Technica](https://x.com/arstechnica/status/1758540835132494119?s=20).

- **Guardrails in the Spotlight**: The necessity of AI guardrails became a topic of humor and caution, as seen in the tale of Air Canada's chatbot creating a refund policy, implying a push for businesses to take AI guardrails more seriously.

- **BERT's Brisk Overview**: A **3-minute BERT discussion** was presented by `@ivanleomk`, while others debated BERT’s impact on Google's search algorithms and its bidirectional nature, which intrigued the community before unidirectional models like GPT. Training and the quality of information for large models like Google's were queried, and anticipation for the next LLM paper grew.

- **LLM Paper Club Goes Global**: Swyxio cordially invited AI enthusiasts to join the **LLM Paper Club (Asia Edition!)** by following the [Discord link](https://discord.com/channels/822583790773862470/1200029657744027658) and shared a recent podcast episode featuring insights on serverless infrastructure for AI, discussing the effects of OpenAI’s *Sora* and *Gemini 1.5*.

- **AI and Agents in Harmony**: A vibrant discussion unfolded around AI agents and state machines, referencing resources such as CrewAI and MagickML. The community brainstormed on compiling tools and resources, and shared experiences in developing AI-related projects. Live streaming plans for experimenting with AI agent frameworks were also announced, marking the one-year anniversary of Latent Space.

- **OpenMoE Lacks Data**: A paper on [OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf), a mixture-of-experts model, was critiqued for its lack of performance due to training on less data than anticipated, and its efficiency in inference time was called into question by members such as `@swyxio`.



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord Summary

- **Spain Powers Up Marenostrum 5**: Spain heralds a new era in European supercomputing with the inauguration of **Marenostrum 5**, housed in a former church, as reported in an article which can be read [here](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/).

- **Debugging Demystified with Compute Sanitizer**: CUDA kernel debugging challenges are surmountable with the use of Nvidia's compute sanitizer as it helps detect illegal memory accesses, as advised by `@gogators.` after `_davidgonmar` described troubleshooting a memory guard condition error.

- **Python Programmers, Optimize Your PyTorch**: **CUDA MODE Lecture 6: Optimizing PyTorch Optimizers** highlights crucial optimization contributions from Jane and provides practical CUDA insights for enhancing PyTorch models.

- **Large World Models Go Open Source**: Information about the **Large World Model** (LWM) was shared, a model trained from LLaMA-2, boasting the ability to process long text documents over 1M tokens. Resources about LWM can be found on its [GitHub page](https://largeworldmodel.github.io/) and its [HuggingFace profile](https://huggingface.co/LargeWorldModel).

- **Graph Capture Divergence in Deep Learning**: The differences in graph capturing between PyTorch 2.0 and JAX are detailed, with PyTorch's imperative approach contrasted against JAX's functional purity requirement, as explained in the [PyTorch 2.0 paper](https://pytorch.org/assets/pytorch_2.pdf) and further discussed in the [torch.fx paper](https://arxiv.org/abs/2112.08429).

- **RingAttention CUDA Project Ignites Collaboration**: An initiative to develop CUDA RingAttention implementation has kicked off with references to two key papers ([Paper 1](https://arxiv.org/abs/2310.01889), [Paper 2](https://arxiv.org/abs/2402.08268)) and the project's [GitHub repo](https://github.com/LargeWorldModel/LWM) and [model on HuggingFace](https://huggingface.co/LargeWorldModel). A dedicated channel and [repository](https://github.com/cuda-mode/ring-attention) have been set up for concentrated collaboration on this project.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

**8-Bit Models Step Up**: `@nafnlaus00` discussed full finetunes on 8-bit models, reflecting on AI Explained's advancements, while Stability AI's focus was questioned by `@dreamgen`. **Adam's Replacement**: The optimizer shift from Adam to **Lion** was debated, with a [GitHub repository](https://github.com/lucidrains/lion-pytorch) being shared for implementation.

**Perplexity and Learnability in LLMs**: `@dreamgen` and `@suikamelon` contemplated using perplexity and learnability for selecting fine-tuning data, alluding to a [scientific paper](https://arxiv.org/abs/2310.13008) for a deeper dive. **SPIN's Implementation**: An official Self-Play Fine-Tuning (SPIN) [GitHub Link](https://github.com/uclaml/SPIN?tab=readme) was provided by `@nruaif`.

**PyTorch and Torchdistx on the Merge Front**: Updates to PyTorch were suggested alongside discussions about **Torchdistx** integration shown in a [GitHub commit](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78), highlighting challenges with non-native optimizers.

**Dataset Blues and Consistent Prompts**: Frustrations with dataset management were vented by `@iatetoomanybeans`, and uniformity within dataset system prompts was confirmed. Interest was sparked by the **Neural-DPO** dataset centered around AI and the Aya initiative found on [Huggingface](https://huggingface.co/datasets/NeuralNovel/Neural-DPO).

**DPO Puzzles Players**: Users like `@filippob82` and `@noobmaster29` voiced confusions and challenges surrounding evaluations with DPO, suggesting it's an unresolved issue. 

**RunPod and Replicate Queries**: Brief messages implied a user error in RunPod, mentioned by `c.gato`, while `j_sp_r` shared an insight via a [link comparing Replicate and Fly](https://venki.dev/notes/replicate-vs-fly).



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

**Fine-Tuning Tactics sought for Sales LLMs**: `@david1542` expressed challenges in fine-tuning **LLMs** for domain-specific tasks, such as sales, due to the agent's lack of understanding of company-specific processes.

**Trace Troubles Trouble Pricing**: `@pasko70` highlighted a cost issue with LangSmith, where **trace costs** are prohibitively expensive for applications with low to medium token throughput, lacking a provided solution or community response.

**Vector DB Confusions Complicate Whisper**: `@cablecutter` delved into issues when processing **Whisper transcripts** into vector databases for thematic summarization and QA, struggling with integrating short-context segments.

**Tech Troubles With LangChain Updates**: Users encountered errors with **LangChain updates**, specifically with the `TextLoader` module and were left seeking fixes, with `@dre99899` suggesting workarounds based on GitHub issue #17585.

**Seeking RAG API Wisdom**: A request was made by `@mamo7410` for guidance on implementing a **RAG API** with **langserv**, including questions about streaming, runtime IDs, and context document handling, with no clear instructions found.

**Multimodal RAG Mingles with PrivateGPT**: `@zhouql1978` created a **Multimodal RAG** utilizing Langchain and PrivateGPT, communicated in a Twitter post and proclaimed to work with various document formats, accomplished in under 300 lines of code.

**Scribe Seeks Scrutiny**: `@shving90` requested feedback on a writing platform project called **Scribe**, which can be found [here](https://scribe.oranai.com/), yet no specific feedback has been mentioned in the conversation.

**Memory Mimicking via Open Source**: `@courtlandleer` from Plastic Labs introduced an open-source alternative to OpenAI's 'memory' with Honcho, featuring a **demo & discord bot** as explained in their [blog post](https://blog.plasticlabs.ai/blog/Memories-for-All).

**Whisper Writings**: `@amgadoz` produced a three-part series on OpenAI's Whisper for ASR, exploring architecture, multitasking, and development process, linked to Substack articles.

**LangChain Learns Rust**: The LangChain library was ported to Rust by `@edartru.`, aiming to simplify writing **LLM-based programs**, with the GitHub repository available [here](https://github.com/Abraxas-365/langchain-rust).

**Financial Analyst AI Tutorial**: `@solo78` shared a Medium article detailing the process to analyze the risk profiles of insurance companies using OpenAI's Assistant API, find the guide [here](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e).

**YouTube Aids LangChain Apprenticeship**: Tutorial videos discussed include creating a Retrieval Augmented Generation UI with ChainLit, adding live stock data to crewAI, and introducing LangSmith for LLM development, found on their respective YouTube channels mentioned above.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **Outliers and Configuration in Training**: Discussions on training anomalies led to a consensus that high loss might be an outlier or "bad data", with `@philipmay` sharing his training config including a **micro_batch_size of 16** and a **learning_rate of 0.0002**. Later, `@philipmay` shared success using **VAGOsolutions/SauerkrautLM-13b-v1** with experts from **LeoLM**, considering it comparable to **mixtral**, while noting missing files with **LeoLM**.

- **Zero to Hero, But First, Check the Prompt**: **gsm8k** challenges surfaced when `@huunguyen` humorously unveiled a model error, scoring zero due to misinterpreting "### Response" as the answer, suggesting the need for dataset-specific pretraining.

- **Building a Better Embedding with JinaAI**: `@devnull0` spotlighted **jina-colbert-v1-en**, a new embedding tech from JinaAI boasting improved zero-shot performance, with a hint dropped by `@huunguyen` favoring **Elasticsearch** for enterprise-scale search solutions over SQLite and Whoosh, and suggesting Lucene/Solr as other alternatives.

- **SOS for German Dataset Resources**: `@thomasrenkert` sought guidance for creating German **evaluation datasets** for translation and summarization tasks, with `@bjoernp` suggesting the use of **lm-evaluation-harness** for metrics such as **chrf**, **bleu**, and **rouge score**.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **Crowdfunding Compute Power**: User bluetyson highlighted the concept of using Kickstarter for funding compute resources, but specific details or outcomes were not discussed.

- **AI Innovation Videos Flood Discord**: A series of educational AI videos were shared by `@pradeep1148`, including a demonstration of function calling with the Llama Factory tool, Tencent's YOLO-World for zero-shot object detection, OpenAI's SORA model for text-to-video capabilities, and WebVoyager as a web-browsing agent. Acknowledgment of these insightful shares was expressed by `@sabertoaster` with a brief "nice."

- **Rethinking Reinforcement Learning**: `@nagaraj_arvind` presented a [lecture on RLHF](https://youtu.be/Ju-pFJNfOfY) and introduced DPO, positioning it as an alternative to PPO. The comparative advantages of DPO are detailed in a [research paper](https://arxiv.org/abs/2305.18290), promising better alignment with human preferences when training large language models.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **LLama-index Queries in Spotlight**: A single inquiry made by `@damiondreggs` about the current viability of **llama-index** for local Retrieval-Augmented Generation (RAG), without any follow-up discussion or additional information.

- **Keen Interest in OpenSora Tech**: Both `@cryptossssun` and `@rusch` exhibited enthusiasm for the **OpenSora** project, with `@rusch` specifically looking to reverse engineer Sora's functionalities, pointing to a collaborative interest in the exploration of this AI technology.



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord Summary

- **AI Engineer Foundation Ponders Hacker News Submission**: User swyxio has identified a [Hacker News article](https://news.ycombinator.com/item?id=39371297) as a potential project interest for the **AI Engineer Foundation**. The user tagged specific members to draw their attention to this opportunity.

- **Learn and Create at Generative AI NYC Workshop**: Tanyarai announced an upcoming **NYC Developer Generative AI Workshop** happening on 2/26 for those eager to learn from OpenAI, Google, and Anthropic experts. The event promises hands-on learning and requires an [RSVP and a laptop](https://lu.ma/ai_workshop).

- **Gear Up for the AI Hackathon**: Hackgoofer invites tech enthusiasts to an AI Engineer Foundation-hosted hackathon on OSS tooling and models, with sponsors like Fireworks.ai & LlamaIndex.ai. Winners can earn cash prizes and details are available for anyone wishing to join the challenge via the [event list](https://partiful.com/e/e3arTNNboImbIKdgQjHs).



---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1207960013260460082) (1409 messages🔥🔥🔥): 

- **The Pursuit of a Transparent Potato**: `@itsme9316` attempted to generate an image of a potato with a transparent background using DreamshaperXL Turbo, leading to an unintended transparent potato rendering. While the visual was intriguing, it's a reminder that precise language is crucial when communicating with AI models.

- **AI's Struggle with Disclaimers**: Multiple users discussed strategies to stop AI models, particularly Mistral variants, from producing disclaimers when generating images. `@itsme9316` successfully applied a disclaimer to the functional output, stopping the unnecessary messaging in its tracks.

- **The AI Banter**: Members in the chat exhibit playful banter, using mistake-based discourse like "turbdo" by `@kaltcit` and using AI models to roleplay or enhance tasks, e.g., "using unholy v2 as a workout assistant" by `@skorchekd`. The light-hearted atmosphere shows the community's comfort in pushing AI's limits.

- **AI Training and Model Discussions**: Users discussed Groq performance and GPT-4's capability, with mixed opinions, and explored practical solutions like soft prompts for fine-tuning models (`@selea8026`). The conversation underlines the communal effort to understand and leverage these complex systems.

- **Privacy Concerns and API Utilization**: `@professional_shaz` remarked on Gemini API being free, prompting discussions about the implications of data usage for training by companies and how users might exploit the free tier. Concerns were mentioned alongside tongue-in-cheek comments about potential misuses and the limitations of what AI entities might offer to consumers.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1111983596572520458/1112690728531918948/1208799217473290240): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Chinese workshops recondition Nvidia's old flagship gaming GPU for AI &mdash; RTX 2080 Ti upgraded to 22GB for $499](https://www.tomshardware.com/pc-components/gpus/chinese-workshops-recondition-nvidias-old-flagship-gaming-gpu-for-ai-rtx-2080-ti-upgraded-to-22gb-for-dollar499): The graphics card is reportedly stable in Stable diffusion, large language models (LLMs), and Llama 2.
- [RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models](https://arxiv.org/abs/2402.10038): Reinforcement learning from human feedback (RLHF) has been extensively employed to align large language models with user intent. However, proximal policy optimization (PPO) based RLHF is occasionally ...
- [LMQL is a programming language for LLM interaction. | LMQL](https://lmql.ai/): no description found
- [Soft prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting): no description found
- [STEM.AI Latest Breakthroughs • A podcast on Spotify for Podcasters](https://podcasters.spotify.com/pod/show/stem-ai): The innovative creations in the large field of AI and technology are shared and explained in the practical format of Podcasting !
- [Cope Harder Cope GIF - Cope Harder Cope Sir - Discover &amp; Share GIFs](https://tenor.com/view/cope-harder-cope-sir-cope-sir-american-psycho-gif-26276799): Click to view the GIF
- [Adapter Methods &mdash; AdapterHub  documentation](https://docs.adapterhub.ml/methods.html#prefix-tuning): no description found
- [Fear GIF - Fear - Discover &amp; Share GIFs](https://tenor.com/view/fear-gif-4516342676332274003): Click to view the GIF
- [Sleepy At Work Sleepy Kitten GIF - Sleepy At Work Sleepy Kitten Cats - Discover &amp; Share GIFs](https://tenor.com/view/sleepy-at-work-sleepy-kitten-cats-funny-animals-gif-13708263): Click to view the GIF
- [llm-applications/notebooks/rag.ipynb at main · ray-project/llm-applications](https://github.com/ray-project/llm-applications/blob/main/notebooks/rag.ipynb): A comprehensive guide to building RAG-based LLM applications for production. - ray-project/llm-applications
- [100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps](https://gist.github.com/darkacorn/71658f280ea0fc0ad4b97d2a616f4ce8): 100k test . exllama2(testbranch) + fa  1 - 100k in 128t steps - gist:71658f280ea0fc0ad4b97d2a616f4ce8
- [学科方向-北京大学智能学院](https://sai.pku.edu.cn/xkjs/xkfx.htm): no description found
- [NeuralNovel/Neural-DPO · Datasets at Hugging Face](https://huggingface.co/datasets/NeuralNovel/Neural-DPO?row=33): no description found

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1207959945379844116) (274 messages🔥🔥): 

- **Model Performance Discussions**: Users `@gamingdaveuk` and `@kquant` discussed the performance of various AI models, with gamingdaveuk seeking a model suitable for storytelling with 8k context that sticks to prompts. They shared experiences with models like Koboldcpp and textgenwebui and even included an [exchange about fixing TextGen](https://pastebin.com/GbbyKPwD) for EXL2 models. Kquant expressed pride in their model ranking #15 in roleplay benchmarking, having once been #5.

- **User-created Roleplay Website and AI Collaboration**: `@shlapfish` invited members to test a hobby AI roleplay website [cowshout.com](https://cowshout.com) which is running Nous-Hermes-2-Yi-34B.Q4_K_S.gguf. Shlapfish also sought advice on connecting KoboldCpp to Silly Tavern, with `@gamingdaveuk` advising on settings such as context size and GPU layer offloading, including linking to [SillyTavern/SillyTavern](https://github.com/SillyTavern/SillyTavern) on GitHub.

- **Technical Discussions on AI Optimization**: Conversations between users like `@soufflespethuman`, `@netrve`, and others touched on AI optimization techniques. They discussed the impact of techniques like pruning and quantization on model performance, and the use of Exl2 quantized models like [Buttercup-4x7B-exl2](https://huggingface.co/royallab/Buttercup-4x7B-exl2) for efficient GPU inference.

- **Peer Recommendations for AI Models**: Users `@dao_li` and `@sao10k` exchanged recommendations on AI models similar to Fimbulvetr-10.7B. Sao10k recommended Fimvulvetr v2 and provided a [link to the model on Huggingface](https://huggingface.co/Sao10K/Fimbulvetr-11B-v2-Test-14), noting that it may perform better than version 1. Dao_li queried about using the model on a 3060 GPU and was advised that q6 could run headless, or q5 for other tasks.

- **Building a Lorebook Using AI**: `@mrdragonfox` discussed using `together.ai` and an [Augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) to create a dataset from decades of D&D journals with the intent to run the tools against an open AI compatible endpoint, confirming that the process could work with 30 years of D&D journals.

**Links mentioned**:

- [royallab/Buttercup-4x7B-exl2 · Hugging Face](https://huggingface.co/royallab/Buttercup-4x7B-exl2): no description found
- [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025v1): Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical d...
- [The ScribeFebruary 16, 2024 3:43 PMWhat tale do you wish to hear?#1D - Pastebin.com](https://pastebin.com/jyZj7zL2): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [Sao10K/Fimbulvetr-11B-v2-Test-14 · Hugging Face](https://huggingface.co/Sao10K/Fimbulvetr-11B-v2-Test-14): no description found
- [The ScribeFebruary 16, 2024 3:43 PMWhat tale do you wish to hear?#1D - Pastebin.com](https://pastebin.com/GbbyKPwD): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [IPIP Home](https://ipip.ori.org/index.htm): no description found
- [MU* - Wikipedia](https://en.wikipedia.org/wiki/MU*): no description found
- [Kooten/Buttercup-4x7B-5bpw-exl2 · Hugging Face](https://huggingface.co/Kooten/Buttercup-4x7B-5bpw-exl2): no description found
- [CowShout](https://cowshout.com): no description found
- [The BardFebruary 16, 2024 12:08 AMWhat will this song or poem be about? - Pastebin.com](https://pastebin.com/Y0tCh7LQ): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [GitHub - e-p-armstrong/augmentoolkit at api-branch](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch): Convert Compute And Books Into Instruct-Tuning Datasets - GitHub - e-p-armstrong/augmentoolkit at api-branch
- [GitHub - SillyTavern/SillyTavern: LLM Frontend for Power Users.](https://github.com/SillyTavern/SillyTavern): LLM Frontend for Power Users. Contribute to SillyTavern/SillyTavern development by creating an account on GitHub.

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1208081828246986842) (26 messages🔥): 

- **Seeking the Ideal Learning Schedule**: `@cogbuji` inquired whether the jump in validation loss observed at the end of a single epoch on a medical instruction dataset is indicative of overfitting and contemplated switching to an SGDR learning schedule. In the ensuing discussion, `@amogus2432` offered advice, suggesting that a longer warm-up or a lower max learning rate might be more beneficial than SGDR for just one epoch.

- **Dataset's Resistance to Overfitting**: `@amogus2432` and `@dirtytigerx` engaged in a discussion sparked by `@maldevide` questioning which dataset wouldn't overfit after 10 epochs, noting that more diverse datasets or those with unusual hyperparameter tuning might not show overfitting signs.

- **KL Divergence Loss Reduction Experiments**: `@amogus2432` shared results from experimenting with KL divergence's loss reduction methods, introducing a custom reduction that seems to steadily decrease loss across epochs, unlike standard averaging losses.

- **Adding Knowledge to Models**: `@magmaguy` inquired about finetuning models as a method to add a knowledgebase and the process to convert large text into an appropriate JSON format. `@amogus2432` suggested looking into Retrieval-Augmented Generation (RAG) for accuracy in reciting facts and directed further discussion to a more specialized channel.

- **Concerns Adapting Models to Older GPUs**: `@wildcat_aurora` sought advice on converting modern models, typically using 16-bit floating point, to 32-bit to accommodate older P40 GPUs that perform better with f32 precision. The goal is a straightforward conversion without training or tuning, possibly followed by quantization to q8 to assess if there would be any gains in speed.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1208121995947151402) (129 messages🔥🔥): 

- **Exploring Vision Transformers for Contextual Awareness**: `@falconsfly` shared a [link to a GitHub repository](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py) related to V-JEPA, a project on self-supervised learning from video for contextual awareness. The conversation suggests the limitations of autoregressive token prediction in building full world context, with users noting young humans and animals learn through different mechanisms.
  
- **Discovering the Secrets of SIMD with Mojo**: `@coffeevampir3` expressed a newfound understanding that Mojo's types are SIMD abstractions, prompting a resolution to dive deeper into learning it. This led to a lively discussion including `@heralax` about the challenges and intricacies of incorporating async functions into non-async code for toolkits and APIs.

- **The 'Autocommit' Timesaver**: `@heralax` introduced a tool they created named [autocommit](https://github.com/e-p-armstrong/autocommit), which uses AI to generate commit messages based on diffs, aiming to make version control less burdensome.

- **Treading the Async and Sync Bridge in Python**: `@spottyluck` gave advice to `@the_ride_never_ends` regarding working with async functions in sync code in Python, pointing out resources like [nest_asyncio](https://github.com/erdewit/nest_asyncio) and [asyncio-bridge](https://death.andgravity.com/asyncio-bridge) that might help in this endeavor.

- **Promoting an Open-Source Typescript Agent Framework**: `@magsp` mentioned their open-source project [bazed-af](https://github.com/bazed-ai/bazed-af) and sought suggestions for communities to share it for feedback, with `@heralax` recommending platforms like LocalLlama and r/opensource.

**Links mentioned**:

- [jepa/src/models/vision_transformer.py at main · facebookresearch/jepa](https://github.com/facebookresearch/jepa/blob/main/src/models/vision_transformer.py): PyTorch code and models for V-JEPA self-supervised learning from video. - facebookresearch/jepa
- [GitHub - e-p-armstrong/autocommit: Automatically commit in a repo and get AI to write the messages. Never lose work again!](https://github.com/e-p-armstrong/autocommit): Automatically commit in a repo and get AI to write the messages. Never lose work again! - e-p-armstrong/autocommit
- [augmentoolkit/processing.py at aphrodite-branch · itsdotscience/augmentoolkit](https://github.com/itsdotscience/augmentoolkit/blob/aphrodite-branch/processing.py): Convert Compute And Books Into Instruct-Tuning Datasets - itsdotscience/augmentoolkit
- [GitHub - bazed-ai/bazed-af: 😎 Bazed.ai Agent Framework - Bazed.ai is a unified platform for building, running and scaling autonomous agents.](https://github.com/bazed-ai/bazed-af): 😎 Bazed.ai Agent Framework - Bazed.ai is a unified platform for building, running and scaling autonomous agents. - bazed-ai/bazed-af
- [GitHub - erdewit/nest_asyncio: Patch asyncio to allow nested event loops](https://github.com/erdewit/nest_asyncio): Patch asyncio to allow nested event loops. Contribute to erdewit/nest_asyncio development by creating an account on GitHub.

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1208058377024831548) (339 messages🔥🔥): 

- **GPU vs. TPU for AI Workloads**: `.the_alt_man` inquired whether **TPUs** are less efficient than **GPUs** for dynamically sized computational graphs, such as variably changing “scan” operations. `johnryan465` requested a definition of "dynamic stuff," prompting further explanation by `.the_alt_man` regarding computational graphs with variable lengths.

- **EleutherAI Social Meetup Reminder**: `canadagoose1` reminded members of the **EleutherAI** social meetup in San Francisco, providing a Discord link and mentioning a tagging error regarding the announcement made earlier by `stellaathena`.

- **Learning Chain of Thoughts**: `emanuel65537` asked if there is a technique to train an AI model to learn a **chain of thought**, such as multiplying two 7-digit numbers, without pretraining or human-provided intermediate steps. `lucaslingle` and `vincent163_13311` highlighted the necessity for step-by-step operations in the dataset for training such models.

- **Superalignment Grants Timezone Query**: `1rokosbasilisk` inquired about the timezone for the deadline of superalignment grants, wondering if it's based on **Anywhere on Earth (AOE)** or **Pacific Time (PT)**. No definitive answers were provided within the discussion.

- **Compression as a Measure of Intelligence**: A philosophical debate ensued, initiated by `sentialx`, on whether a **language model** equates to mere **data compression** and if that could be considered as formulating **knowledge** or **intelligence**. Various users engaged in the discussion, exploring the relationship between data compression and intelligence, where some argued they are entangled concepts that can't easily be separated.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/1027492909227970570/1208128218809237504): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/729741769192767510/1207749000858439740): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Video generation models as world simulators](https://openai.com/research/video-generation-models-as-world-simulators): We explore large-scale training of generative models on video data. Specifically, we train text-conditional diffusion models jointly on videos and images of variable durations, resolutions and aspect ...
- [Always Has Been Among Us GIF - Always Has Been Among Us Astronaut - Discover &amp; Share GIFs](https://tenor.com/view/always-has-been-among-us-astronaut-space-betrayal-gif-23836476): Click to view the GIF
- [EleutherAI SF Meetup | Partiful](https://partiful.com/e/8hRUy9flN02dFLK4rBxh): Meetup for EleutherAI
- [EleutherAI SF Meetup | Partiful](https://partiful.com/e/8hRUy9flN02dFLK4rBxh/?reload=true): Meetup for EleutherAI
- [Don&#39;t Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489): It is common practice to decay the learning rate. Here we show one can usually obtain the same learning curve on both training and test sets by instead increasing the batch size during training. This ...
- [BlackMamba: Mixture of Experts for State-Space Models](https://arxiv.org/abs/2402.01771): State-space models (SSMs) have recently demonstrated competitive performance to transformers at large-scale language modeling benchmarks while achieving linear time and memory complexity as a function...
- [
David MacKay: Information Theory, Inference, and Learning Algorithms: Home 
](https://www.inference.org.uk/mackay/itila/): no description found
- [Human Knowledge Compression Contest: Frequently Asked Questions & Answers](http://prize.hutter1.net/hfaq.htm): no description found
- [GitHub - vincent-163/transformer-arithmetic](https://github.com/vincent-163/transformer-arithmetic): Contribute to vincent-163/transformer-arithmetic development by creating an account on GitHub.
- [I.—COMPUTING MACHINERY AND INTELLIGENCE](https://academic.oup.com/mind/article/LIX/236/433/986238): I propose to consider the question, ‘Can machines think?’ This should begin with definitions of the meaning of the terms ‘machine’ and ‘think’. The definitions 

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1207969497118679140) (196 messages🔥🔥): 

- **Exploring Long Document Processing**: [BABILong, a benchmark for long document processing](https://arxiv.org/abs/2402.10790), is used to evaluate models like GPT-4 and RAG, showing a ceiling at $10^4$ elements. However, a fine-tuned GPT-2 with recurrent memory augmentations processed up to $10^7$ elements, marking a significant leap in handling long inputs.
- **Debate Over Presence of Liquid Models in Research**: Multiple users discussed the presence and impact of liquid models in current AI research, like Liquid-S4. Despite an impressive drone demo and solid benchmarks, these models haven't seen broad adoption outside their originating research group.
- **Uncertainty Around Liquid AI's Direction**: Members expressed skepticism towards the Liquid AI startup, calling the company's mission statements "fluff" and seeking concrete information about their actual projects and goals.
- **Discussion on Initialization Methods for Neural Networks**: Several users discussed the merits of different initialization methods, particularly ZerO (zeros and ones), identity, Hadamard, and their potential impact on large language model stability and the lottery ticket hypothesis.
- **Questions on kNN Approach vs. Recurrent Memory**: User @clashluke inquired about the difference between an approach using kNN lookup into a non-differentiable memory of past inputs ([arXiv link](https://arxiv.org/abs/2203.08913)) and fine-tuning with recurrent memory augmentations that enable GPT-2 to process extensive inputs.

**Links mentioned**:

- [Universal Neural Functionals](https://arxiv.org/abs/2402.05232): A challenging problem in many modern machine learning tasks is to process weight-space features, i.e., to transform or extract information from the weights and gradients of a neural network. Recent wo...
- [On Limitations of the Transformer Architecture](https://arxiv.org/abs/2402.08164): What are the root causes of hallucinations in large language models (LLMs)? We use Communication Complexity to prove that the Transformer layer is incapable of composing functions (e.g., identify a gr...
- [Tweet from Kyle O'Brien (@KyleDevinOBrien)](https://x.com/KyleDevinOBrien/status/1758667079849480630?s=20): How can we make classifiers more robust when we can&#39;t modify the weights or assume its architecture  — effectively making it a black box? In our preprint, we demonstrate that we can improve robust...
- [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://arxiv.org/abs/2402.10193): Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given the higher computational demand of pre-trai...
- [In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790): This paper addresses the challenge of processing long documents using generative transformer models. To evaluate different approaches, we introduce BABILong, a new benchmark designed to assess model c...
- [Improving Language Plasticity via Pretraining with Active Forgetting](http://arxiv.org/abs/2307.01163): Pretrained language models (PLMs) are today the primary model for natural language processing. Despite their impressive downstream performance, it can be difficult to apply PLMs to new languages, a ba...
- [CVPR 2016 Open Access Repository](https://openaccess.thecvf.com/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html): no description found
- [UFO: A UI-Focused Agent for Windows OS Interaction](https://arxiv.org/abs/2402.07939): We introduce UFO, an innovative UI-Focused agent to fulfill user requests tailored to applications on Windows OS, harnessing the capabilities of GPT-Vision. UFO employs a dual-agent framework to metic...
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200): In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) promptin...
- [Spike No More: Stabilizing the Pre-training of Large Language Models](https://arxiv.org/abs/2312.16903): Loss spikes often occur during pre-training of large language models. The spikes degrade the performance of large language models and sometimes ruin the pre-training. Since the pre-training needs a va...
- [Headless Language Models: Learning without Predicting with Contrastive Weight Tying](https://arxiv.org/abs/2309.08351): Self-supervised pre-training of language models usually consists in predicting probability distributions over extensive token vocabularies. In this study, we propose an innovative method that shifts a...
- [ZerO Initialization: Initializing Neural Networks with only Zeros and Ones](https://arxiv.org/abs/2110.12661): Deep neural networks are usually initialized with random weights, with adequately selected initial variance to ensure stable signal propagation during training. However, selecting the appropriate vari...
- [Liquid AI: A New Generation of Foundation Models from First Principles](https://liquid.ai): Liquid AI is an MIT spin-off with a presence in Boston, MA, and Palo Alto, CA. Our mission is to build state-of-the-art, general-purpose AI systems from first principles and deploy capable, efficient,...
- [fairseq2/src/fairseq2/models/llama/builder.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/models/llama/builder.py#L262): FAIR Sequence Modeling Toolkit 2. Contribute to facebookresearch/fairseq2 development by creating an account on GitHub.
- [fairseq2/src/fairseq2/models/transformer/frontend.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/models/transformer/frontend.py#L122C1-L123C1): FAIR Sequence Modeling Toolkit 2. Contribute to facebookresearch/fairseq2 development by creating an account on GitHub.
- [fairseq2/src/fairseq2/nn/transformer/decoder_layer.py at f381d9305e2958a8105fce7fae150e3809469076 · facebookresearch/fairseq2](https://github.com/facebookresearch/fairseq2/blob/f381d9305e2958a8105fce7fae150e3809469076/src/fairseq2/nn/transformer/decoder_layer.py#L91): FAIR Sequence Modeling Toolkit 2. Contribute to facebookresearch/fairseq2 development by creating an account on GitHub.
- [Liquid Structural State-Space Models](https://arxiv.org/abs/2209.12951): A proper parametrization of state transition matrices of linear state-space models (SSMs) followed by standard nonlinearities enables them to efficiently learn representations from sequential data, es...
- [Memorizing Transformers](https://arxiv.org/abs/2203.08913): Language models typically need to be trained or finetuned in order to acquire new knowledge, which involves updating their weights. We instead envision language models that can simply read and memoriz...

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1208108282930135112) (3 messages): 

- **Causal Scrubbing Method Highlighted**: `@neelnanda` recommended exploring the concept of causal scrubbing by sharing a [lesswrong post](https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing), although the link provided was malformed.
- **Appreciation for Influential Work**: `@yonghyunpark` expressed gratitude for inspirational work, specifically thanking `@neelnanda`.

**Links mentioned**:

- [no title found](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing&ved=2ahUKEwjHvcGsk7OEAxXoWEEAHRE3DEYQFnoECBUQAQ&usg=AOvVaw33dMhAk1jgQEvSBnTq8uOq): no description found
- [Redirect Notice](https://www.google.com/url?sa=t&source=web&rct=j&opi=89): no description found

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1207984429109088296) (38 messages🔥): 

- **MMLU Task Repository Shared**: `@pminervini` shared the [GitHub link](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu) to tasks for MMLU (Massive Multitask Language Understanding).

- **Fine-Tuning the Logprobs Warning**: `@noldtronics` reported an issue with logprobs data during model evaluation using `llama.cpp server + flask "openai api"-server`, receiving a *WARNING* for invalid token_logprobs list and an *ERROR* for invalid response for loglikelihood when testing a prompt related to roof shingle removal.

- **Integration Issues with Llamacpp and LM Evaluation Harness**: `@hailey_schoelkopf` mentioned looking into potential changes in the Llamacpp interface after `@noldtronics` posted having issues and opening [a related GitHub issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/1437) regarding the Llamacpp and gguf.py interface.

- **lm-evaluation-harness Installation Pitfalls and Guidance**: `@ilanser` sought advice for setting up `lm-evaluation-harness` in a docker environment with Nvidia’s or Coreweave’s images recommended by `@hailey_schoelkopf`. `@vincent163_13311` shared an in-depth installation guide to address the various challenges when setting up lm_eval with vLLM on CUDA environments.

- **Test Running Advice for lm-evaluation-harness**: Several users discussed how to test `lm-evaluation-harness`. `@baber_` and `@stellaathena` suggested using the `--limit 10` parameter to minimize test run times, while `@vincent163_13311` recommended using model `gpt2` and task `arc_easy` for faster evaluation due to fewer samples.

**Links mentioned**:

- [llama / gguf interface broken? · Issue #1437 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1437): Only information i found on this issue was a post on discord: Armando Diaz: Hi! I&#39;m working on a research project and we were wondering if it is possible to use the lm-evaluation-harness to evalua...
- [no title found](http://gguf.py:90)]): no description found
- [lm-evaluation-harness/lm_eval/tasks/mmlu at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/mmlu): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.](https://github.com/EleutherAI/lm-evaluation-harness): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [from vllm._C import cuda_utils raise error · Issue #2797 · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/2797): ImportError: /root/autodl-tmp/conda/envs/wslconda/lib/python3.9/site-packages/vllm/_C.cpython-39-x86_64-linux-gnu.so: undefined symbol: _ZN2at4_ops15to_dtype_layout4callERKNS_6TensorEN3c108optional...
- [no title found](https://download.pytorch.org/whl/cu118): no description found

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1208006545884057630) (13 messages🔥): 

- **FIM-NEO-X Trained with Familiar Architecture**: `@hailey_schoelkopf` clarified that the **FIM-NEO-X** model was trained using the same architecture as **GPT-NeoX** and is compatible with Huggingface using the GPT-NeoX class.

- **Architectural Flexibility in GPT-NeoX**: `@hailey_schoelkopf` also indicated that while **GPT-NeoX** primarily supports transformer-based architectures, it has been adapted by others for training non-transformer models, referencing an article detailing the `Based` architecture.

- **Understanding Multi-Head Attention Distribution**: In response to `@jdranpariya`'s query, `@catboy_slim_` explained that in the `ParallelSelfAttention` class, `mpu.divide` equitably distributes attention heads across partitions for model parallelization without leaving remainder errors.

- **Implementation Details of NeoX Arguments**: `@catboy_slim_` noted that **NeoX** may not have implemented all functionalities from **Megatron** and **Deepspeed**, and some arguments related to pipeline parallelism may remain unaddressed in the codebase.

**Links mentioned**:

[Zoology (Blogpost 2): Simple, Input-Dependent, and Sub-Quadratic Sequence Mixers](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based): no description found

  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1207981486637187072) (328 messages🔥🔥): 

- **Optimizing GPU Utilization for LLMs**: Users are exploring settings to enhance token generation speed. One user with an RTX 4050 and Ryzen 7 setup is achieving 34 tokens/s after offloading 50 layers and is seeking advice for further improvements. `@qiikzx` also queries about fine-tuning `.gguf` models to sound more human and remove censorship.
- **Exploring Local LLM Setup Extensibility**: `@krypt_lynx` is considering an idea to integrate LLMs into a game mod for "RimWorld" to generate "speech bubbles" based on extensive character data from the game and is seeking guides for setting up character "personas."
- **Platform and Hardware Discussions for LLMs**: Discussions revolve around the benefits of running LLMs on Linux vs. Windows, with one comparison showing a 30% improvement in tokens per second. Users are also sharing their experiences with various hardware specs, like using a GTX 1050 with 16GB RAM.
- **Inquiries on Local LLM Capabilities**: Interest spikes around whether LLMs can run on USB-based neural network accelerators such as Coral USB or Intel Compute Stick. Another discussion seeks to clarify if LM Studio can support function calling or only plain text responses.
- **Support Questions for LLM Issues**: Several users, including `@hautc.it`, request guidance on specific LM Studio problems, like document upload, model selection, and connection errors; others request one-to-one assistance to resolve issues like AVX2 instruction set requirements and fine-tuning models.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1204973625518587925): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1197279779603370015): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1110598183144399061/1207719619058733108): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1110598183144399061/1207936682440269844): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1709551591425245338?s=20): But does the model actually _use_ these representations? By looking for neurons with similar weights as the probe, we find many space and time neurons which are sensitive to the spacetime coords of an...
- [LMQL is a programming language for LLM interaction. | LMQL](https://lmql.ai/): no description found
- [Qwen/Qwen1.5-72B-Chat-GGUF · Hugging Face](https://huggingface.co/Qwen/Qwen1.5-72B-Chat-GGUF): no description found
- [Air Canada ordered to pay customer who was misled by airline’s chatbot](https://www.theguardian.com/world/2024/feb/16/air-canada-chatbot-lawsuit): Company claimed its chatbot ‘was responsible for its own actions’ when giving wrong information about bereavement fare
- [NVIDIA Chat With RTX](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/): Your Personalized AI Chatbot.
- [HWiNFO - Free System Information, Monitoring and Diagnostics](https://www.hwinfo.com/): Free Hardware Analysis, Monitoring and Reporting. In-depth Hardware Information, Real-Time System Monitoring, Reporting &amp; more
- [How to mixtral](https://rentry.org/HowtoMixtral): Updated 12/22 Have at least 20GB-ish VRAM / RAM total. The more VRAM the faster / better. Grab latest Kobold: https://github.com/kalomaze/koboldcpp/releases Grab the model Download one of the quants a...
- [KnutJaegersberg/2-bit-LLMs at main](https://huggingface.co/KnutJaegersberg/2-bit-LLMs/tree/main): no description found
- [GitHub - KillianLucas/open-interpreter: A natural language interface for computers](https://github.com/KillianLucas/open-interpreter): A natural language interface for computers. Contribute to KillianLucas/open-interpreter development by creating an account on GitHub.
- [GitHub - Josh-XT/AGiXT: AGiXT is a dynamic AI Agent Automation Platform that seamlessly orchestrates instruction management and complex task execution across diverse AI providers. Combining adaptive memory, smart features, and a versatile plugin system, AGiXT delivers efficient and comprehensive AI solutions.](https://github.com/Josh-XT/AGiXT): AGiXT is a dynamic AI Agent Automation Platform that seamlessly orchestrates instruction management and complex task execution across diverse AI providers. Combining adaptive memory, smart features...
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1747617771796762901?s=46): New version is out (to appear at ICLR)! Main updates: - Additional experiments on Pythia models - Causal interventions on space and time neurons - More related work - Clarify our claims of a literal w...
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1747617820957876317?s=46): However, there was a recent paper from Chen et al. on &#34;Causal Representations of Space&#34; in LLMs that builds on our work and finds &#34;LLMs learn and use an internal model of space in solving ...
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1709551516577902782?s=20,): Do language models have an internal world model? A sense of time? At multiple spatiotemporal scales?  In a new paper with @tegmark we provide evidence that they do by finding a literal map of the worl...
- [Tweet from Biz Stone (@biz)](https://x.com/wesg52/status/1709): making livy take an airborne before bed
- [Implicit Representations of Meaning in Neural Language Models](https://arxiv.org/abs/2106.00737): Does the effectiveness of neural language models derive entirely from accurate modeling of surface word co-occurrence statistics, or do these models represent and reason about the world they describe?...
- [Tweet from Joseph Sarnecki (@JosephSarnecki)](https://x.com/JosephSarnecki/status/1758541761495159011?s=20): @wesg52 @tegmark Can you test whether the large language models do the same with other activations - specifically in an imagined scenario (ie: role-play)? I am curious if llms create an internal world...

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1208037732333658132) (33 messages🔥): 

- **Seeking Long-Context Models**: `@msz_mgs` requested recommendations for the best 7b models supporting 32k context, although no one seemed to have responded with suggestions in the provided messages.
- **LM Studio Model Exploration Tips**: `@heyitsyorkie` recommended checking out TheBloke's repositories and sorting by most downloaded in LMStudio's Model Explorer for finding models suitable for a task, targeting `@mitchalley`'s inquiry about locating the best models for specific tasks.
- **Model Size vs. Performance**: Reacting to `@mitchalley` finding TheBloke’s popular models, `@alastair9776` suggested starting with `Q5_K_M` or `Q4_K_M` models based on their solid performance on various setups and encouraging experimentation to find what works best.
- **Language Translation Model Preferences**: For language translation models, `@mulder1` favored Mistral-7B-openorca-4.0, while `@fabguy` and `@heyitsyorkie` promoted Deepl for its superior performance in translation tasks. However, `.ben.com` observed that fanyi.baidu.com outperforms others for Chinese translation.
- **Choosing the Right Quantized Model**: When asked about the better performing model between `dolphin-2.7-mixtral-8x7b.Q5_0.gguf` and `dolphin-2.7-mixtral-8x7b.Q5_K_M.gguf`, `@alizatjan` indicated that `K_M` is a newer and better quantization method, pointing `@snens9650` to a detailed Reddit post for more insight.

**Links mentioned**:

- [TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ · Hugging Face](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ): no description found
- [TheBloke/CodeLlama-70B-Instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/CodeLlama-70B-Instruct-GGUF): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/159nrh5/the_difference_between_quantization_methods_for/): no description found
- [Qwen/Qwen-VL-Chat · Hugging Face](https://huggingface.co/Qwen/Qwen-VL-Chat): no description found

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1208033125968650300) (14 messages🔥): 

- **Factory Settings Reset Confusion**: `@msz_mgs` inquired about performing a factory reset on LMS settings, initially unsure if their settings were default. After attempting a "Reset to Default," they were directed by `@heyitsyorkie` to manually delete prompt presets from the specified folder.

- **Assistance with Preset Folder Location**: `@msz_mgs` asked for the location of the preset folder to reset LMS settings, and `@heyitsyorkie` guided them to click "Open Presets Folder..." to find it.

- **LMS Default Restoration**: `@msz_mgs` deleted all presets in the folder and confirmed that LMS repopulated the default ones, resolving their problem.

- **Inconsistency in UI Panel Behavior**: `@borisrusev` reported a UI inconsistency where the Settings panel fades when disabled but the Chat panel does not. `@heyitsyorkie` acknowledged the inconsistency, implying that both should uniformly gray out during inference.

- **Continuation Button Bug Frustration**: `@logandark` expressed frustration over a bug preventing the removal of trailing newlines, which affects the continue button's functionality. Citing a previous bug report, `@logandark` mentioned the issue is impairing their use of LM Studio.
  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1208028478968954951) (127 messages🔥🔥): 

- **Intel Cores Harnessed for Virtual Machines**: `@addressofreturnaddress` plans on using Intel’s extra cores to run KVMs with macOS and Windows, seeking balance between performance and portability by considering the 14900K with a mini ITX board, and anticipates upgrading from a 3090 to a 5090 GPU.
- **Hardware Enthusiasts Touting Multi-GPU Setups**: Multiple users, including `@heyitsyorkie`, `@nink1`, and `@goldensun3ds`, discussed complex multi-GPU configurations, like pairing two or more 3090s and issues with PCIe slots and adaptors, highlighting power and space constraints as limiting factors.
- **GPU Adapter and PC Setup Challenges**: `@goldensun3ds` encountered an Ebay issue with a mislabeled 4060 Ti GPU and discussed at length the trials of using a PCIe x1 adapter and a PCIE riser cable for multi-GPU setups. The discussion was dotted with creative but "ghetto" solutions like having the PC on its side, actively debated by users like `@heyitsyorkie`.
- **Multi-GPU Tooling and Software Utilization Queries**: Amidst the hardware talk, `@goldensun3ds` and `@heyitsyorkie` dove into the intricacies of LM Studio's multi-GPU support, exploring config adjustments to balance GPU VRAM utilization between mismatched cards.
- **Tech Support Pointers for AVX2**: `@consuliam` inquired about disabling AVX2 support validation, leading to advice from `@yagilb` about a non-AVX2 beta and `@heyitsyorkie` suggesting using HWiNFO to check CPU capabilities, while `@nink1` humorously cautioned against relying on ChatGPT for accurate hardware information.

**Links mentioned**:

- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): no description found
- [Pro WS WRX90E-SAGE SE｜Motherboards｜ASUS Global](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/): ASUS Workstation motherboards are designed for professionals in AI training, deep learning, animation, or 3D rendering. Featuring expandable graphics, storage, impressive connectivity and reliability,...
- [👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai/beta-r): Find, download, and experiment with local LLMs
- [no title found](https://www.amazon.ca/dp/B0BDCZRBD6): no description found
- [HWiNFO - Free System Information, Monitoring and Diagnostics](https://www.hwinfo.com/): Free Hardware Analysis, Monitoring and Reporting. In-depth Hardware Information, Real-Time System Monitoring, Reporting &amp; more
- [ASUS Global](https://www.asus.com/motherboards-components): no description found
- [Stillesque GIF - Stillesque - Discover &amp; Share GIFs](https://tenor.com/view/stillesque-gif-25544869): Click to view the GIF
- [Amazon.com: StarTech.com PCI Express X1 to X16 Low Profile Slot Extension Adapter - PCIe x1 to x16 Adapter (PEX1TO162) : Electronics](https://www.amazon.com/gp/aw/d/B0039XPS5W/): no description found

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1208127995340914758) (12 messages🔥): 

- **Pin Advocacy for Chat Etiquette**: `@jedd1` suggested the need to pin the website [don't ask to ask](https://dontasktoask.com/), emphasizing better chat question etiquette by leading with the actual problem instead of inquiring if experts are present.
- **Proposed Bot Solution to Discourage Vague Queries**: In response to `@jedd1`'s comment about pinning an etiquette guide, `@heyitsyorkie` humorously proposed the idea of a bot replying to such behavior.

- **Teachable Moments in Chat Rooms**: `@jedd1` humorously commented that people should be taught to explain what they did, what they expected, and what actually happened, referencing a "three-punch reporting template".
- **Common Issues with Unclear Error Reports**: `@heyitsyorkie` agreed with `@jedd1` about the problem with vague questions and mentioned frequent encounters with users who report "exit code" errors without providing sufficient details for troubleshooting.

- **Linux: A Welcome or Sorry Affair?**: A friendly exchange where `@zioalex` introduced themselves as a new Linux user in the channel, followed by `@fabguy` giving a tongue-in-cheek condolence, and `@zioalex` replying with good humor about there being worse operating systems.

**Links mentioned**:

[Don't ask to ask, just ask](https://dontasktoask.com/): no description found

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1208175972994785280) (8 messages🔥): 

- **Seeking Autogen Guidance**: User `@zioalex` inquired about starting with tools like **Autogen or CrewAI** and was looking for a good guide. `@heyitsyorkie` provided a [YouTube tutorial on Autogen](https://www.youtube.com/watch?v=Hds_fJaAu78) for assistance.
- **Broken Link Reported**: `@zioalex` reported a non-working example link and asked for it to be fixed.
- **No Fix for Missing Repository**: `@heyitsyorkie` explained that the repository for the example no longer exists, so the link cannot be fixed.
- **Suggesting Pin Removal for Obsolete Link**: After learning the link cannot be fixed, `@zioalex` suggested removing the pin from the broken link message.
- **Pin Duties Assigned**: `@heyitsyorkie` stated that removing the pin is a task for `<@1108574387889778738>`.
- **Pin Removed**: User `@fabguy` confirmed that they removed the problematic links.
- **Random Gif Shared**: `@carasen12` shared a [Tenor gif](https://tenor.com/view/bobawooyo-dog-confused-dog-huh-dog-meme-shocked-dog-gif-16713203299056947073) of a confused dog, which did not relate to the ongoing discussion.

**Links mentioned**:

- [Bobawooyo Dog Confused GIF - Bobawooyo Dog confused Dog huh - Discover &amp; Share GIFs](https://tenor.com/view/bobawooyo-dog-confused-dog-huh-dog-meme-shocked-dog-gif-16713203299056947073): Click to view the GIF
- [Local AI Agent with ANY Open-Source LLM using LMStudio](https://www.youtube.com/watch?v=Hds_fJaAu78): Hello and welcome to an explanation and tutorial on building your first open-source AI Agent Workforce with LM Studio!  We will also learn how to set it up s...

  

---


### LM Studio ▷ #[rivet](https://discord.com/channels/1110598183144399058/1167546635098804284/) (1 messages): 

mend1440: Dang, this project is fuego!!!
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1208043899252375642) (7 messages): 

- **Inquiries about LM Studio Integration**: `@cyberir` asked if anyone in the group had experience with running **LM Studio** with **Flowise**, which sparked a conversation about technical integration.
- **Exploring LM Studio and LangFlow**: `@mend1440` shared their attempts at integrating LM Studio with **LangFlow**, mentioning the possibility of using an `http_client` and an environment variable to connect with the OpenAI API.
- **Navigating Through Lack of Programming Expertise**: `@cyberir` and `@mend1440` expressed that they are not programmers but are attempting to work through the integration process despite this hurdle.
- **LM Studio Server Connection Issues**: `@cyberir` mentioned having trouble getting LM Studio's server to work properly, implying a need for assistance or guidance in resolving the issue.
- **Success with Manual Settings**: `@mend1440` reported eventually getting the system to function correctly by setting the Base URL variable and manually inputting configuration details into **Langflow**.

**Links mentioned**:

- [no title found](http://my.test.server.example.com:8083",): no description found
- [no title found](http://my.test.proxy.example.com",): no description found

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1207969694464614400) (370 messages🔥🔥): 

- **Mistral Announces Mystery Model**: `@lerela` announced a new prototype model called `next`, inviting users to try it on lmsys and provide feedback. The official details and capabilities of **Mistral-Next** remain under wraps, sparking curiosity and speculation among users.

- **Mistral-Next Performance Debate**: While users like `@sven_72358` find **Mistral-Next** impressive and comparable to GPT-4 in limited tests, others like `@jiha` report that it fails basic logic questions that other models answer correctly. The true performance and parametric size of Mistral-Next against benchmarks such as Mistral Medium and Mixtral have yet to be confirmed.

- **Mistral-Next Coding Responses Limited**: `@Shadow27` and `@mrdragonfox` discussed the seeming limitations of **Mistral-Next** in generating complete coding responses. Some users suggest this could be by design, while others recommend using it as a tool for functions rather than full solutions.

- **Exploring Fine-Tuning Options for Mistral Models**: `@mato8792` sought advice on fine-tuning the **7b (full Mistral)** model for a new language using 2xRTX3090Ti cards. `@mrdragonfox` suggested LoRA tuning as a viable option, and referred to `@dirtytiger` and `TheBloke` on Discord for detailed guidance.

- **Context Length in Large Language Models Discussed**: Users like `@mrdragonfox` and `@thezennou` debated the practical use and performance implications of context length in Large Language Models (LLMs), suggesting that improvements in handling shorter, relevant contexts may be preferable to chasing exceedingly long contexts.

**Links mentioned**:

- [Mistral AI launches Mixtral-Next | Hacker News](https://news.ycombinator.com/item?id=39406168): no description found
- [Chat with Open Large Language Models](https://chat.lmsys.org/): no description found
- [Chat with Open Large Language Models](https://chat.lmsys.org): no description found
- [Infowar Skeptical GIF - Infowar Skeptical Conspiracy - Discover &amp; Share GIFs](https://tenor.com/view/infowar-skeptical-conspiracy-theory-nod-gif-13373739): Click to view the GIF
- [TheBloke/Mixtral-8x7B-v0.1-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-v0.1-GGUF): no description found
- [Issues · vllm-project/vllm](https://github.com/vllm-project/vllm/issues/1002>): A high-throughput and memory-efficient inference and serving engine for LLMs - Issues · vllm-project/vllm
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe): Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. - karpathy/minbpe

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1207982709683523584) (30 messages🔥): 

- **Mistral-Next Unveiled in Bits**: `@mrdragonfox` described **Mistral-Next** as a prototype model, with confirmation from `@lelio`. There’s speculation in the thread about its capabilities and release, but no concrete details have been shared.
- **Local GPT-4 Coding Capabilities Discussed**: `@nemomolok` expressed excitement for the potential of **locally running GPT-4** for coding, indicating a development in AI that could provide more accessible powerful coding assistance.
- **Extracting Tabular Data Challenge**: `@mehdi_guel` is working with **Mistral-small** on a RAG application and is seeking advice on extracting table cells from Word documents, as LLMs typically struggle with structured data.
- **Workarounds for Tabular Data Suggested**: `@mrdragonfox` recommends using in-context learning, scripts to extract data to a database, and potential **text2sql** approaches as workarounds to handle tabular data with Mistral.
- **Looking for Improvements in Contextual Understanding**: `@tensorbender` hopes that finetuned LLMs will better separate user prompts from "Input context," particularly to enhance the performance of smaller models in lengthy context generation tasks.
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1207991233926922250) (43 messages🔥): 

- **6Freedom Studio CTO Looks for Mistral Integration**: `@fangh`, CTO of [6Freedom Studio](https://6freedom.studio), inquired about integrating Mistral on-premise for VR/AI-related products. They were advised to contact Mistral support or talk to Mistral's developer relations, `@803073039716974593`.
- **Size Options for Mistral Discussed**: `@mrdragonfox` mentioned that the sizes "tiny" and "small" with configurations like 7b/8x7b are available, while medium size might require an enterprise deal and sales consultation.
- **Unauthorized Error Addressed**: `@renaudr.` faced an "Unauthorized" error while attempting to use Mistral's API and shared their curl command for assistance.
- **Troubleshooting API Key Issues**: `@mrdragonfox` suggested that the error might be due to an invalid key and asked `@renaudr.` to confirm if their billing was active.
- **Solutions Proposed for API Key Activation Delay**: `@mrdragonfox` clarified that it takes about 5 minutes for a key to activate on a new account and recommended setting an environment variable with the API key before using it in the curl command.

**Links mentioned**:

[6freedom | Experts en technologies immersives](https://6freedom.studio): 6freedom est une agence experte en technologies immersives. Nous vous accompagnons dans l&#039;élaboration de vos projets sur-mesure.

  

---


### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1208059513874153552) (6 messages): 

- **SLEB: A Technique for Accelerating LLMs**: `@alex096170` shared an [arXiv paper](https://arxiv.org/abs/2402.09025v1) introducing **SLEB**, a novel method that prunes LLMs by removing redundant transformer blocks to improve inference speed without significant compromise on performance.
- **In Search of the Best LLM Pretraining Framework**: `@remek1972` inquired about the best framework for **pretraining large language models** (LLMs), specifically not for finetuning but training from scratch on a large corpus.
- **Framework Recommendations for Different Scales**: `@quicksort` suggested that the choice of framework depends on the scale of parallelization; **Accelerate with deepspeed** is good for multi-node environments, while **axolotl** might be easier for a single-node setup.
- **Scaling Up LLM Pretraining Advice**: For pretraining LLMs on **multiple nodes**, `@quicksort` recommended following **Stas Bekman's Twitter** and his book on machine learning engineering, noting the use of Accelerate with deepspeed for such tasks.
- **Gratitude Expressed for Pretraining Guidance**: `@remek1972` expressed thanks for the suggestions, indicating that the information provided was helpful.

**Links mentioned**:

[SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025v1): Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical d...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1207967772236054529) (2 messages): 

- **Elqano is Hiring AI Talent**: `@thomas_saulou` announced a job opportunity at Elqano, a startup seeking a developer in generative AI, located in Biarritz, France. Details about the position, including engagement of generative AI in the knowledge domain and company background, are available at their [Welcometothejungle job listing](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer).

- **AI Launchpad Opportunity for Pre-Seed Startups**: `@deedubs__` shared an opportunity for pre-seed AI startups to gain exposure at Data Council in March presented by **Zero Prime Ventures**. Interested founders can apply through the provided link [Zero Prime Ventures AI Launchpad](https://zeroprime.vc/ai-launchpad) and contact `@deedubs__` for more insights on Data Council.

**Links mentioned**:

[Applied Generative AI Engineer - Elqano - CDI](https://www.welcometothejungle.com/fr/companies/elqano/jobs/applied-generative-ai-engineer): Elqano recrute un(e) Applied Generative AI Engineer !

  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1208523756629135450) (24 messages🔥): 

- **Mistral vs GPT-4 for Coding**: `@mrobino` finds that **Mistral Medium** is closer to **GPT-3.5** rather than **GPT-4** when it comes to coding ability, whereas `@mrdragonfox` reported getting better coding results from **Mistral** compared to **GPT-4**.
  
- **TDD Integration Approach with AI**: `@mrdragonfox` discusses their unique workflow integrating **test-driven development (TDD)** with AI assistance, where they focus on having the AI implement just enough code to pass written tests and formally validate them, steering generation while maintaining control over the architecture.

- **Collaboration Offer for Open Source**: `@nani99` expresses their willingness to share compute resources for open-source contributions, especially for creating high-quality synthetic data, while `@mrdragonfox` shows interest in understanding the nature of the compute resources offered.

- **Augmentoolkit Contribution and Dataset Cleanup**: `@mrdragonfox` shares a link to their contribution on [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch) and discusses the extensive cleanup required for large datasets, mentioning an approximate runtime of 25 days for phase 1 cleanup using their hardware setup.

- **Data Cleaning and Testing Discussions**: `@mrdragonfox` and `@akshay_1` exchange ideas on cleaning up data, with a mention of embedding techniques and regex, and discuss the workload and scheduling for testing various parts of the dataset cleanup process.

**Links mentioned**:

[GitHub - e-p-armstrong/augmentoolkit at api-branch](https://github.com/e-p-armstrong/augmentoolkit/tree/api-branch): Convert Compute And Books Into Instruct-Tuning Datasets - GitHub - e-p-armstrong/augmentoolkit at api-branch

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1208263743306469426) (16 messages🔥): 

- **RWKV Fan Club**: `@hexani` expressed excitement about **RWKV**, and `@vatsadev` enthusiastically agreed.
- **Compress to Impress**: `@elder_plinius` shared [MYLN](https://github.com/elder-plinius/MYLN), a tool designed to *compress and abbreviate text* for improving context length and efficiency, asking for benchmarking ideas.
- **Tokenizing for Truth**: In response to `@elder_plinius`, `@vatsadev` suggested that token comparison before and after compression could reveal if the tool is effective for LLMs.
- **Tool Complexity Unveiled**: `@vatsadev` warned that abbreviation may not be represented in an LLM's tokenizer, potentially harming comprehension, and advised aiming for common tokens.
- **Counterintuitive Outcome**: `@elder_plinius` reported a surprising result where a text sample had fewer characters after using MYLN but resulted in more tokens, indicating a potential issue with their approach.

**Links mentioned**:

[GitHub - elder-plinius/MYLN: A language compressor for enhanced context length and efficiency of LLM-to-LLM communication.](https://github.com/elder-plinius/MYLN): A language compressor for enhanced context length and efficiency of LLM-to-LLM communication. - elder-plinius/MYLN

  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1208029726938234942) (33 messages🔥): 

- **Model Fine-Tuning Showcase**: `@pradeep1148` shared [a YouTube video](https://www.youtube.com/watch?v=EYR_kd3X03M) titled *"Finetune model for Function Calling with Llama Factory"*, offering insights on finetuning models for function calls.
- **YOLO-World Unveiled**: `@pradeep1148` highlighted the release of **YOLO-World** by Tencent’s AI Lab with a [YouTube video](https://www.youtube.com/watch?v=yaqi8xRUsp4) showcasing this real-time, zero-shot object detection model.
- **Battle of the Large Language Models**: `@carsonpoole` mentioned **Senku 70b**, a large language model fine-tuned on Puffin, in comparison to *ChatGPT 4*. He invited others to send prompts for testing Senku's outputs and shared illustrative outputs.
- **A Vision of Web Navigation**: The video "[Web Browsing Agent using LangGraph](https://www.youtube.com/watch?v=gbGYN3YyTS4)" was shared by `@pradeep1148`, featuring WebVoyager, an agent capable of web browsing by controlling mouse and keyboard, `@teknium` praised the share.
- **Exploring the Future of Video Generation**: `@pradeep1148` also posted about OpenAI's SORA Text to Video model in a [YouTube video](https://www.youtube.com/watch?v=7lsOzA3WhSI), discussing its capability of generating videos from text prompts, while in subsequent discussion, users like `@gabriel_syme` and `@teknium` debated the practical applications and future developments of such technology.

**Links mentioned**:

- [Web Browsing Agent using LangGraph](https://www.youtube.com/watch?v=gbGYN3YyTS4): Web VoyagerWebVoyager by He, et. al., is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.It works by viewing annotated brow...
- [Finetune model for Function Calling (Tool Call) with Llama Factory](https://www.youtube.com/watch?v=EYR_kd3X03M): Finetune model for Function Calling (Tool Call) with Llama Factory#llm #ml #ai #largelanguagemodels #deeplearning #python #pythonprogramming https://github.c...
- [OpenAI SORA Text to Video model and Technical Report](https://www.youtube.com/watch?v=7lsOzA3WhSI): Introducing Sora, our text-to-video model. Sora can generate videos up to a minute long while maintaining visual quality and adherence to the user’s prompt.#...
- [YOLO-World: Real-Time, Zero-Shot Object Detection](https://www.youtube.com/watch?v=yaqi8xRUsp4): On January 31st, 2024, Tencent’s AI Lab released YOLO-World (access code on Github), a real-time, open-vocabulary object detection model. YOLO-World is a zer...

  

---


### Nous Research AI ▷ #[benchmarks-log](https://discord.com/channels/1053877538025386074/1131747216244092928/1208738399729754112) (1 messages): 

- **Benchmarks Log Relocated**: `@teknium` announced that the **Benchmarks Log** has moved to a new Github repository. Anyone interested in the benchmark logs for various LLMs can now find them at [LLM-Benchmark-Logs on GitHub](https://github.com/teknium1/LLM-Benchmark-Logs).

**Links mentioned**:

[GitHub - teknium1/LLM-Benchmark-Logs: Just a bunch of benchmark logs for different LLMs](https://github.com/teknium1/LLM-Benchmark-Logs): Just a bunch of benchmark logs for different LLMs. Contribute to teknium1/LLM-Benchmark-Logs development by creating an account on GitHub.

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1207966019788865566) (26 messages🔥): 

- **Synthetic Data Generation with UE**: `@deki04` clarified that using **Unreal Engine (UE)** for synthetic data generation in AI imagery is common practice, and even mentioned the use of **Microsoft’s AirSim plugin** as a tool for these tasks.

- **Insights on Whisper's ASR Capabilities**: `@amgadoz` shared a series of [blog posts](https://amgadhasan.substack.com/) that delve into **Whisper**, OpenAI's state-of-the-art model for **Automatic Speech Recognition (ASR)**, including details on its architecture, multitasking abilities, and the massive data preparation process behind it.

- **Open-Source GritLM Introduced**: `@Muennighoff` launched **GRIT**, a model unifying both text embeddings and generation tasks, claiming improvements in efficiency for operations like **Retrieval-Augmented Generation (RAG)**. The announcement came with links to an [academic paper](https://arxiv.org/abs/2402.09906) and a [GitHub repository](https://github.com/ContextualAI/gritlm).

- **Representation Engineering Exploration**: `@.benxh` shared a resource on **Representation Engineering**, highlighting a [paper](https://arxiv.org/abs/2310.01405) and associated [Github code](https://github.com/andyzoujm/representation-learning) that present methods for analyzing and manipulating AI model behaviors during inference, without the need for prompt engineering or finetuning.

- **NeurIPS 2023 Submission on Gradient Descent**: A **NeurIPS 2023** paper on **DoWG (Distance over Weighted Gradients)**, a new parameter-free gradient-based optimizer, was highlighted in a [link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html) by `@euclaise`, noting its efficiency and universality in adapting to both smooth and nonsmooth problems.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1053877538025386074/1149866623109439599/1207052162203390032): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [DoWG Unleashed: An Efficient Universal Parameter-Free Gradient Descent Method](https://proceedings.neurips.cc/paper_files/paper/2023/hash/15ce36d35622f126f38e90167de1a350-Abstract-Conference.html): no description found
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/): no description found
- [Building Corrective RAG from scratch with open-source, local LLMs](https://youtube.com/watch?v=E2shqsYwxck&si=uF0H5IaMKGiZWPeb): Building LLM apps with more complex logical flows can be challenging with smaller, local LLMs. Graphs offer one way to tackle this, laying out the logic flow...
- [Tweet from Niklas Muennighoff (@Muennighoff)](https://x.com/muennighoff/status/1758307967802224770): Introducing GRIT🦾to unify text embedding 🔢& generation 📝. GritLM is open SoTA on embedding (MTEB) & generative tasks (BBH etc.) – Both in 1 model. See 🧵for how GRIT🦾 makes RAG &gt;60% faster & mo...
- [Decoding Whisper: An In-Depth Look at its Architecture and Transcription Process](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): Part 2 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [Exploring Whisper&#x27;s Multitask Interface: A Closer Look at its Speech Transcription and Translation Capabilities](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): Part 3 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [The Making of Whisper: An In-Depth Exploration of its Training Data and Process](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): A Multi-part series in which we delve deep into whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1207962649216942100) (323 messages🔥🔥): 

- **RAG not dead yet**: Despite rumors to the contrary, `@gabriel_syme` and `@n8programs` engage in a lively debate, suggesting that RAG is very much alive because context isn't its only issue, with the hierarchy being key for large datasets.

- **OpenHermes Finetuning Fun**: `@n8programs` reports successful training of TinyLLaMA on the OpenHermes dataset, achieving a sub-1 train loss, while `@jiha` highlights the effectiveness of selecting long samples for finetuning on the dataset.

- **Groq Makes Waves with Speed**: `@gezegen` stirs excitement with his mention of Groq's LPU Inference Engine achieving impressive token processing speeds, while `@swaystar123` and `@leontello` discuss how it compares to other hardware like the H100.

- **LM's Can Process Vast Documents**: `@gabriel_syme` shares an arXiv paper showing that fine-tuning GPT-2 with recurrent memory augmentations enables document processing of up to $10^7$ elements, beating GPT-4 and RAG's capabilities for long inputs.

- **Green Blob Joins Discord and Discovers This Chat**: `@greenblob6064` expresses surprise upon finding the Nous Research AI Discord, thinking SAIL was the only one, and `@powerful_wolf_14649` questions the general scarcity of information about Nous Research on the internet.

**Links mentioned**:

- [Tweet from Lewis Tunstall (@_lewtun)](https://x.com/_lewtun/status/1758520258132865210?s=20): I&#39;ve tested the &#34;long is more&#34; trick on @Teknium1&#39;s  OpenHermes dataset and it works surprisingly well 🔥!  - Select the 1k longest samples (0.1%) - SFT Mistral-7B for 15 epochs with N...
- [Groq](https://groq.com/): no description found
- [In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790): This paper addresses the challenge of processing long documents using generative transformer models. To evaluate different approaches, we introduce BABILong, a new benchmark designed to assess model c...
- [SOCIAL MEDIA TITLE TAG](https://os-copilot.github.io/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Tweet from Shane Parr (@sparr_ml)](https://x.com/sparr_ml/status/1758246182285914136?s=20): The first thing that a 2024 LLM buyer&#39;s guide should say is that context length is largely irrelevant; what matters is output quality, quality, and quality, and that can only be achieved with a hi...
- [LDJnr/LessWrong-Amplify-Instruct · Datasets at Hugging Face](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct): no description found
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe): Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. - karpathy/minbpe
- [Introducing Sora — OpenAI’s text-to-video model](https://youtube.com/watch?v=HK6y8DAPN_0&si=dm3GMf22C89I2gLB): Introducing Sora, our text-to-video model.Sora can create videos of up to 60 seconds featuring highly detailed scenes, complex camera motion, and multiple ch...
- [Tweet from OpenAI (@OpenAI)](https://fxtwitter.com/OpenAI/status/1758192965703647443?s=20): Prompt: “A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. she wears a black leather jacket, a long red dress, and black boots, and carries a black pur...
- [GitHub - luuyin/OWL: Official Pytorch Implementation of &quot;Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity&quot;](https://github.com/luuyin/OWL): Official Pytorch Implementation of &quot;Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity&quot; - luuyin/OWL
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1709551516577902782?s=20,): Do language models have an internal world model? A sense of time? At multiple spatiotemporal scales?  In a new paper with @tegmark we provide evidence that they do by finding a literal map of the worl...
- [Tweet from Wes Gurnee (@wesg52)](https://x.com/wesg52/status/1709551591425245338?s=20): But does the model actually _use_ these representations? By looking for neurons with similar weights as the probe, we find many space and time neurons which are sensitive to the spacetime coords of an...
- [Implicit Representations of Meaning in Neural Language Models](https://arxiv.org/abs/2106.00737): Does the effectiveness of neural language models derive entirely from accurate modeling of surface word co-occurrence statistics, or do these models represent and reason about the world they describe?...
- [Tweet from Joseph Sarnecki (@JosephSarnecki)](https://x.com/JosephSarnecki/status/1758541761495159011?s=20): @wesg52 @tegmark Can you test whether the large language models do the same with other activations - specifically in an imagined scenario (ie: role-play)? I am curious if llms create an internal world...
- [1.5 bit quantization by ikawrakow · Pull Request #5453 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5453): This draft PR is a WIP that demonstrates 1.5 bits-per-weight (bpw) quantization. Only CUDA works, there is no implementation for the other supported back-ends. CUDA, AVX2 and ARM_NEON are implement...

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1208098652493451355) (54 messages🔥): 

- **In Search of RAM Efficiency for Adam**: `@hexani` raised a question about the amount of RAM needed for fine-tuning a 7B model using Adam due to its high memory requirement (8X weight copies). They also asked for methods to reduce this RAM usage but did not receive a direct response within the given messages.
- **Batch Size Queries Unanswered**: `@hexani` inquired about the typical batch size used for training with GPUs like a 4090 or an H100 when fine-tuning models, but this question went unanswered.
- **PyTorch Over Jax for LLM**: `@carsonpoole` mentioned they are in the process of converting an LLM to PyTorch format from Jax, describing the process as unpleasant.
- **Axolotl Fine-tuning Tutorial Request Goes Unmet**: `@pncdd` sought a complete step-by-step tutorial on fine-tuning using axolotl, from dataset handling to execution, but no responses with such a guide appeared.
- **Longform Text Generation Challenges**: `@benh.1984` asked for advice on generating very long text (~20,000 tokens) using llms without success, and `.ben.com` suggested banning the end-of-sentence token to encourage continuance, though this might lead to degraded quality when the model wants to conclude.

**Links mentioned**:

- [WordNet Search - 3.1](http://wordnetweb.princeton.edu/perl/webwn?c=8&sub=Change&o2=&o0=1&o8=1&o1=1&o7=&): no description found
- [WordNet Search - 3.1](http://wordnetweb.princeton.edu/perl/webwn?c=8&sub=Change&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&i=0&h=1000&s=giraffe): no description found

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1208062922228633610) (5 messages): 

- **Interest in Paper Implementation**: `@qtnx` showed enthusiasm about implementing the methodologies from a recent paper, though they have some *doubts on how to do it*.
- **Recognizing a Peer**: `@vatsadev` expressed surprise and admiration upon discovering `@qtnx` in the channel, referring to them as an **Absolute legend**.
- **Collaboration Opportunity**: `@qtnx` reached out to someone through their Discord ID (`<@282315082749444097>`) suggesting a collaborative effort on the paper's implementation, despite uncertainties on the exact procedure.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1207983642031423498) (368 messages🔥🔥): 

- **Sora's Video Editing Capabilities Discussed**: User `@max_voltage` highlighted the versatility of *Sora*, noting its ability to prompt with images or videos for a range of editing tasks, something they felt was lacking in tools like DALL-E.

- **V-JEPA and AI Model Development**: `@max_voltage` shared a [link to an article](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) about Meta's **V-JEPA** model, discussing its importance in AI and referencing other related research articles. The subsequent discussions with `@twoabove` and others consider differences in handling the pipeline and objectives between various models.

- **Midjourney v6 Critiques**: User `@pseudoterminalx` expressed displeasure at the latest Midjourney update, critiquing the quality and composition of generated images.

- **AMD's Pervasive AI Developer Contest**: `@itali4no` highlighted the **AMD Pervasive AI Developer Contest** with prizes in Generative AI, Robotics AI, and PC AI categories, leading to a conversation on the feasibility and availability of AMD GPUs for AI work with inputs from `@chad_in_the_house`, `@pseudoterminalx`, and `@drhead`.

- **LAION's Community Behavior**: Toward the end of the messages, `@thejonasbrothers` mentioned a perception of toxicity in the LAION community, prompting a call from `@mega_b` for more friendly and constructive engagement, especially with newcomers.

**Links mentioned**:

- [no title found](https://huggingface.co'): no description found
- [FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model](https://arxiv.org/abs/2303.09833): Recently, conditional diffusion models have gained popularity in numerous applications due to their exceptional generation ability. However, many existing methods are training-required. They need to t...
- [GOODY-2 | The world&#x27;s most responsible AI model](https://www.goody2.ai/): Introducing a new AI model with next-gen ethical alignment. Chat now.
- [Create new page · vladmandic/automatic Wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA>): SD.Next: Advanced Implementation of Stable Diffusion and other Diffusion-based generative image models - Create new page · vladmandic/automatic Wiki
- [Sora could ruin peoples lifes](https://community.openai.com/t/sora-could-ruin-peoples-lifes/635220): You guys are going to end so many careers for people. Photographers, artists, animators, filmmakers, and  possibly even actors. Being in these industry’s is hard already, and now with this people migh...
- [Pervasive AI Developer Contest](https://www.hackster.io/contests/amd2023#challengeNav): Fueling Groundbreaking Innovations With AMD.
- [kakaobrain/align-base · Hugging Face](https://huggingface.co/kakaobrain/align-base): no description found
- [no title found](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/): no description found
- [AI x Mental Health Happy Hour · Luma](https://lu.ma/obvb5mzw): Slingshot is building a foundational model for psychology to help scale access to mental healthcare globally. Come join us for a happy hour to talk AI x Mental Health. Location TBD in Central...
- [GitHub - HighCWu/control-lora-v2: ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2](https://github.com/HighCWu/control-lora-v2): ControlLoRA Version 2: A Lightweight Neural Network To Control Stable Diffusion Spatial Information Version 2 - HighCWu/control-lora-v2
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [lllyasviel - Overview](https://github.com/lllyasviel): Lvmin Zhang (Lyumin Zhang). lllyasviel has 40 repositories available. Follow their code on GitHub.
- [Allen T (@Mr_AllenT)](https://nitter.mint.lgbt/Mr_AllenT/status/1758839836021002470?t=REnUyTKEsWpzPqTx-JnMZw&s=19>): Incase your phone has been broken for the past 48 hours  The OpenAI team has been dropping new Sora videos since the official release  Here are 10 incredible Sora videos posted on X:

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1208073505543946251) (27 messages🔥): 

- **HDiT Breakthrough in Image Generation**: `@chad_in_the_house` shared an announcement of a presentation on the **Hourglass Diffusion Transformer (HDiT)**, which boasts linear scaling with pixel count, potentially setting a new state-of-the-art for diffusion models at high resolutions. The [paper is available on arxiv](https://arxiv.org/abs/2401.11605).
  
- **Strange Blue Tint in Offset Noise**: `@chad_in_the_house` inquired if a blue tint in output images was normal when doing offset noise, to which `@thejonasbrothers` simply responded with a no.

- **Questioning LaION Database's Integrity**: `@vrus0188` mentioned [Reddit feedback](https://www.reddit.com/r/StableDiffusion/comments/1ata8gw/feedback_on_base_model_releases/) that criticized the LaION Database used by Stable Cascade for issues like excessive censoring and poor image labeling, sparking discussions on image model training practices and community toxicity.

- **Aesthetic Scoring Critique in Image Modeling**: `@drhead` expressed skepticism about aesthetic scoring methods applied in image models, suggesting it could lead to generically pleasant but unvaried output, reminiscent of midjourney's approach, and proposed a more nuanced understanding of what constitutes quality in images.

- **Prospects for Synthetic Data and Video Modeling**: `@unjay.` and `@spirit_from_germany` discussed the potential of synthetic data for video modeling, citing the results of Sora as a benchmark, and proposed starting a project to generate a large dataset by creating camera images from different perspectives of existing or generated 3D scenes.

**Links mentioned**:

- [Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers](https://arxiv.org/abs/2401.11605): We present the Hourglass Diffusion Transformer (HDiT), an image generative model that exhibits linear scaling with pixel count, supporting training at high-resolution (e.g. $1024 \times 1024$) directl...
- [Reddit - Dive into anything](https://www.reddit.com/r/StableDiffusion/comments/1ata8gw/feedback_on_base_model_releases/): no description found

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1207960486398918741) (131 messages🔥🔥): 

- **GPT-3 Tokens and Video Patches Discussion**: `@lugui` clarified that on GPT, the unit of data pricing is a token, and that patches are an equivalent concept for video format. They suggested the patch concept would become clearer once it's released for use and priced.
- **Concerns Over Sora's Impact on Creativity**: `@thedasenqueen` shared concerns about AI like Sora ending creativity and creative skills, which `@infidelis` countered by suggesting that the issue is with NPC (non-player characters) users of AI, not the technology itself. `@bambooshoots` also mentioned that fears about technology inhibiting creativity historically have always been proven wrong.
- **Users Divided on GPT and Gemini Performance**: `@redstone12345` started a discussion comparing GPT and Gemini, with `@infidelis` noting Gemini's human-like behavior and creativity, while GPT excels in structured tasks; `@exx1` also added that GPT-4-Turbo has advantages over Gemini Ultra in reasoning and reflection.
- **Experimenting with AI on Social Media**: `@fai.hunter` wants to use ChatGPT to manage a social media page and respond to Instagram DMs in their personal style, hinting at a project to mimic personal interaction. `@eskcanta` noted the importance of following OpenAI's terms of service in such endeavors, and `@reynupj` suggested having a substantial archive of personal messages for successful mimicry.
- **Legal and Ethical Discussion on AI Generated Content**: A conversation about the use of AI such as Sora in creating derivative works led to `@johnnyrobert` contending that private, non-commercial use likely would not legally impact the original work's creator. `@eskcanta` discussed the complexities of OpenAI's legal responsibilities and how they may affect user-created, AI-generated content.

**Links mentioned**:

- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [Usage policies](https://openai.com/policies/usage-policies): no description found
- [
   OSWeb
  ](https://jatos.it.ntnu.no/publix/OG92k9q7KYc): no description found

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1208021140832788551) (142 messages🔥🔥): 

- **Flask Server Troubles**: `@will_b_mora` faced an error when trying to use actions with GPT, receiving the message "Server URL is not under the root origin; ignoring it". `@elektronisade` clarified that the server address must be public and reachable by OpenAI's services.
  
- **Custom GPT Saving Glitch**: `@sines303` couldn't save and publish a custom GPT due to a "FatalServerError". This issue occurs in the presence of a sizeable JSON-based knowledge base.

- **GPT Slowness and Response Quality**: Multiple users like `@bigskippa`, `@teamsettle_04535`, and `@silentsushix3` reported slowness and poorer quality in GPT-4's responses compared to version 3.5, observing significant speed degradation and errors.

- **Content Policy Confusion**: `@bazilb` asked about the implications of a content policy warning when asking GPT about adult content, querying whether their account might be at risk. `@eskcanta` suggested reading OpenAI's policies for clarification and shared that if no rules are being broken, typically one's account should be safe.

- **GPT-4 Output Length Advice**: Users like `@iamrobertandrews` and `@darthgustav.` discussed strategies for generating long-form content, such as planning in sections with clear tasks or using templates to guide GPT-4's output.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1209015855849938986): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [no title found](https://lambdalabs[dot]com/service/gpu-cloud#pricing): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1202309673709994065): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [How should AI systems behave, and who should decide?](https://openai.com/blog/how-should-ai-systems-behave): We’re clarifying how ChatGPT’s behavior is shaped and our plans for improving that behavior, allowing more user customization, and getting more public input into our decision-making in these areas.
- [Usage policies](https://openai.com/policies/usage-policies): no description found
- [Terms of use](https://openai.com/policies/terms-of-use): no description found
- [Usage policies](https://web.archive.org/web/20231101074011/https://openai.com/policies/usage-policies): no description found

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1207965884602392597) (54 messages🔥): 

- **Categories Conundrum**: User `@ben.30` expresses challenges with categorization names at their company, considering merging related categories to streamline GPT's understanding when processing service requests related to waste management and skips.

- **AI Training Tips**: `@eskcanta` offers comprehensive advice on training AI, emphasizing the importance of teaching AI about exceptions and edge cases, much like training a new human employee, and suggests avoiding reprimands for AI mistakes, focusing on the desired output instead.

- **Handling PII in Summarization**: `@best.value` and `@madame_architect` engage in a conversation about the challenges of redacting personally identifiable information (PII) before summarization, with suggestions of using Python libraries for PII detection and avoiding direct AI detection of PII.

- **LLM JSON Formatting Struggles**: `@neeagl` faces issues with ensuring the GPT-3.5 model outputs consistently formatted JSON, resolves it by using an example response and avoiding conversational history that was leading to errors.

- **Prompt Crafting Strategies**: `@elegante94` inquires about a GPT tool that refines original prompts into effective instructions for AI tasks, with `@queueh` sharing a detailed, iterative prompt-building strategy that focuses on collaboration between the AI and the user to fine-tune the prompts.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1019652163640762428): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1208676441101963295): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1207965884602392597) (54 messages🔥): 

- **Frustrations with Unstructured Categories**: `@ben.30` discussed the challenges faced with categorizing tickets in categories such as Non Mechanical and Waste Management, and mentioned plans to potentially merge skips and waste categories due to the difficulty in distinguishing them based on customer language.

- **Teaching AI with Human-Like Training**: `@eskcanta` advocated teaching AI using methods akin to training new human employees, emphasizing clarity and reinforcement in training to handle complex or unusual tasks.

- **Preprocessing Text for PII**: `@best.value` explored the dilemma of handling personal identifiable information (PII) within large, unstructured text and the limitations of AI in redaction tasks. Madame_architect and `@exhort_one` suggested using specific PII detection tools before summarization and the importance of clear intent in prompts.

- **The Craft of Prompt Engineering**: `@queueh` shared an intricate prompt built to aid in crafting other effective prompts for use with GPT. Meanwhile, `@elegante94` inquired about AI that could refine original prompt ideas, and `@eskcanta` recommended direct communication with ChatGPT for prompt refinement.

- **Issues with Knowledge Base Retrieval**: `@pawjwp` and `@d1scobo1` reported issues with GPT’s use of knowledge bases, where the model either refuses to consult the provided data or becomes less accurate with subsequent questions within the same thread.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1019652163640762428): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1208676441101963295): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1207962460011761674) (227 messages🔥🔥): 

- **Turbo Confusion Cleared**: `@brknclock1215` seemed puzzled about the difference between mobile screenshots and web versions of perplexity's mysterious 'turbo' feature. `@gooddawg10` clarified that they saw the 'turbo' label on mobile using Kiwi Browser, sparking a conversation about potential variations in user interfaces across platforms.
  
- **Where's the Turbo?**: `@gooddawg10` is confused about the 'turbo' mentioned on their mobile device, while `@brknclock1215` and `@Perplexity AI` join the chat acknowledging the discussion about discrepancies in the turbo feature appearances.

- **API Performance Discussed**: `@brknclock1215` inquired about performance differences with 'turbo' selected models on Perplexity, while `@gooddawg10` has yet to test it due to being at work. `@brknclock1215` appreciated the conversation, though no concrete answers about performance discrepancies were provided.

- **Gemini 1.5's Marketing Strategy Examined**: `@archient` predicts that Google might give access to Gemini 1.5 within two months as a strategic move following a two-month free access period after its release.

- **App Pricing Structure Queried**: `@retonq` sought clarification on the perplexity dedicated pricing structure, and `@icelavaman` explained that there's a usage-based pricing model in place for Perplexity Pro, with details available on the Perplexity pricing documentation page.
  
- **Discord Bot Confusion**: Users `@brickpotato` and `@spectralruler` asked about the Perplexity AI Discord bot status. `@icelavaman` clarified that the bot has been shut down.

**Links mentioned**:

- [Pricing](https://docs.perplexity.ai/docs/pricing): no description found
- [‎What Gemini Apps can do and other frequently asked questions](https://gemini.google.com/faq?hl=en#citation): Learn what Gemini can do, how it works, and different ways to get access to it.
- [What is Perplexity Pro?](https://blog.perplexity.ai/faq/what-is-perplexity-pro): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Gemini 1.5 and The Biggest Night in AI](https://youtu.be/Cs6pe8o7XY8): The biggest day in AI since GPT-4&#39;s release. A new state of the art model, Gemini 1.5, has arrived, on the same night as a bombshell text-to-video model, Sor...

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1207994716403142747) (38 messages🔥): 

- **Repeat Posts not Flattering**: User `@brknclock1215` responded to their own issue about double-posting threads, humorously suggesting that imitation might be the utmost form of flattery.
- **Curiosity About Web Access for LLMs**: `@bishal_saha` inquired about the process by which language models such as Perplexity access the web, leading to a detailed discussion and sharing of resources.
- **Link to the Mysteries of Accessing the Web**: In response to `@bishal_saha`'s curiosity, `@brknclock1215` shared a [NeurIPS paper on Retrieval-Augmented Generation](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) for natural language processing models.
- **Housekeeping and Guidelines for Channel Posts**: `@icelavaman` stepped in to remind everyone that the sharing channel is primarily for sharing Perplexity threads and to direct users to the appropriate channels for general discussions and questions.
- **Discussions of Perplexity API and its Capabilities**: There were links shared related to Perplexity usage, with `@brknclock1215` specifically addressing `@bishal_saha`'s query about the uniqueness of Perplexity's `pplx` models in accessing up-to-date web information.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1208016236877840404): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1047197230748151888/1054944216876331118/1206373264100696094): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [HuggingChat](https://huggingface.co/chat/): Making the community's best AI chat models available to everyone.
- [Introducing PPLX Online LLMs ](https://blog.perplexity.ai/blog/introducing-pplx-online-llms): The first-of-its-kind Online LLM API

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1208142952212205578) (27 messages🔥): 

- **API Rate Limit Increase Request Goes Unanswered**: User `@enelemtal` expressed frustration after being ghosted by api@perplexity.ai regarding a request for an increase in API rate limit. `@ok.alex` responded, asking `@enelemtal` to DM their email for a check, while `@bitshift_` and `@bvfbarten.` indicated they are experiencing the same issue.

- **Transparent Citations Matter**: In response to `@retonq`, `@ai_made_approachable` emphasized the importance of citations for professional use of "pplx-online" models, citing current limitations due to its "blackbox" nature. `@me.lk` assured that work is being done to add citation support to the API for approved use cases in the future.

- **Perplexed by Peculiar Characters in Streaming API**: `@boyn_` reported receiving confusing characters like `00` and `2\n` when using the `pplx-70b-online` model, seeking clarification on why this occurs. `@thedigitalcat` questioned when to expect an update regarding these messy results, and `@icelavaman` confirmed that the team is aware and working on the issue.

- **Model Deprecation and Longevity Queries**: Concerned about the deprecation of `llama-2-70b-chat`, `@rehmatsg` inquired about the typical support duration for a model, such as `mixtral-8x7b-instruct`, before it becomes deprecated.

- **Setting up Perplexity API Endpoints**: After `@nettemple` sought guidance on the correct 'apiUrl' for integration, `@icelavaman` provided the official endpoint `https://api.perplexity.ai/chat/completions` along with a reference to the documentation. Subsequent messages offered code assistance to `@nettemple`, with successful implementation and expression of gratitude for the help offered.

**Links mentioned**:

- [no title found](https://api.perplexity.ai'): no description found
- [no title found](https://api.perplexity.ai';): no description found
- [Moon (Dark Mode)](https://docs.perplexity.ai): no description found
- [Chat Completions](https://docs.perplexity.ai/reference): no description found

  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1207964370882007051) (144 messages🔥🔥): 

- **Seeking Guidance on Fine-Tuning LLMs**: User `@keshav._._.` expressed the need for help with converting PDFs of Indian IPS laws into a format suitable for fine-tuning the llama 2 model. Conversely, another user, `@vishyouluck`, suggested using a RAG approach instead of fine-tuning for the task at hand.

- **Query on Multiplayer Game Development**: `@om7059` sought advice on incorporating model evaluation in a multiplayer doodle game they plan to develop, where doodles are scored by a model after time runs out.

- **Finding the Best Model for Geographical Data**: `@retonq` was curious about identifying which model, between Mistral medium, pplx, and llama, is best at understanding geographic information like coordinates and directions.

- **Looking for Open Source LLMs Supporting Persian**: `@alifthi` in search of a high-performance, Persian-supporting open-source language model like ChatGPT. Another user, `@alchemist_17.`, suggested that any open-source model such as mistral or llama2 could be fine-tuned with a custom dataset.

- **A Dive into Language Model Basics**: Multiple users including `@vipitis`, `@tea3200`, and `@doctorpangloss` contributed to a discussion clarifying that language models primarily predict the probability of the next sequence of words, with some being able to manifest knowledge and reasoning through diverse tasks.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1209022297269080136): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [LLM Visualization](https://bbycroft.net/llm): no description found
- [Join the Hugging Face Discord Server!](https://discord.gg/hugging-face-879548962464493619?event=1203285706949009448): We&#x27;re working to democratize good machine learning 🤗Verify to link your Hub and Discord accounts! | 70607 members
- [Best Image Models V2 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2): no description found
- [bigcode/the-stack · Datasets at Hugging Face](https://huggingface.co/datasets/bigcode/the-stack): no description found
- [codeparrot/github-code · Datasets at Hugging Face](https://huggingface.co/datasets/codeparrot/github-code): no description found
- [Google Colaboratory](https://colab.research.google.com/drive/1i6cDDsZfGB70fNgxiUxbUufeQfaWyCfd?usp=sharing): no description found
- [Models - Hugging Face](https://huggingface.co/models): no description found
- [OpenAI&#39;s Agent 2.0: Excited or Scared?](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx): I want to give you a full run down of browser/mobile/desktop AI agentsGet free HubSpot E-book: Using Generative AI to scale your content operation: https://c...
- [meta-llama/Llama-2-7b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b): no description found
- [meta-llama/Llama-2-7b-chat-hf · Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf): no description found
- [Build software better, together](https://github.com/search?q=language%3Apython&type=repositories): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [GitHub - LegallyCoder/mamba-hf: Implementation of the Mamba SSM with hf_integration.](https://github.com/LegallyCoder/mamba-hf): Implementation of the Mamba SSM with hf_integration. - LegallyCoder/mamba-hf
- [Google Colaboratory](https://colab.research.google.com/drive/1ONevcH1oHOdm4F6DPgju_WyUWLVyIY6b?usp=sharing): no description found
- [Build software better, together](https://github.com/search?q=language%3Ajava&type=repositories): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1208012278209847356) (8 messages🔥): 

- **Plagiarism Tools for Code Quality**: `@brady_kelly` shared an insightful method for detecting overlooked **boilerplate documentation**. They advised using a process similar to plagiarism detection in **software CI/CD pipelines** to ensure all documentation is properly completed.
  
- **Holiday Productivity Report**: `@antiraedus` outlined their holiday achievements, which included focusing on fitness, weight gain, and setting the stage to tackle their university semester with goals like tutoring, socializing, and working on side projects such as a flutter game, regardless of quality.

- **GitHub Copilot's Engine Revealed?**: `@rcdpge` mentioned that they learned about **GitHub Copilot** possibly running on **ChatGPT 3.5 Turbo** for code suggestions, though the quality of the inline completions seemed to be questioned by **@vipitis**.

- **Prompt-Driven RAG System Blog Post**: `@subham5089` invited members to read their LinkedIn blog post, which discusses the challenges and potential solutions related to **prompt-driven RAG systems**. The post explores how this emerging tech can be improved upon. [Read the blog post](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7).

- **Lecture on Reinforcement Learning and Language Models**: `@nagaraj_arvind` shared a lecture video about **RLHF (Reinforcement Learning from Human Feedback)** and a new alternative to PPO called **DPO**. The content is for those interested in enhancing LLM completions with RLHF. [Watch the lecture video](https://youtu.be/Ju-pFJNfOfY) and [read the DPO paper](https://arxiv.org/abs/2305.18290).

**Links mentioned**:

- [RLHF, PPO and DPO for Large language models](https://youtu.be/Ju-pFJNfOfY): Introduction to Reinforcement Learning, RLHF, Proximal policy optimization (PPO) and Direct preference optimization (DPO) algorithms.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised ...

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1208081421025935400) (17 messages🔥): 

- **Understanding Protein Language Models**: `@grimsqueaker` shared insights on the limitations of protein language models (PLMs) for anything but structural predictions. The paper "Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models" concludes from 370 experiments that pretraining helps but doesn't scale with more compute for all tasks, needing new pretraining approaches ([read the abstract](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1), [discuss on Twitter](https://twitter.com/KevinKaichuang/status/1755672999166972319)).

- **Intel Unveils Text to 3D Model Converter**: `@abhinit21` highlighted Intel's new `LDM3D-VR` model, capable of converting text to 3D, focused on virtual reality development ([model on Hugging Face](https://huggingface.co/Intel/ldm3d-pano), [read the paper](https://arxiv.org/pdf/2311.03226.pdf)).

- **Detecting Deepfake Faces with a Web App**: `@lucas_selva` promoted their web app that utilizes XAI to identify deepfake images, stating the current model accuracy at 88% with plans for future training enhancements ([try the app](https://deep-fake-generated-people-facial-recognition.streamlit.app/)).

- **Discussion on Enhancing AI Face Recognition**: In a conversation with `@hrishimax`, `@lucas_selva` discussed the limitations and future improvement plans for a model designed to detect AI-generated faces, including expansion of the training dataset and applying transfer learning.

- **Databricks: Accelerating AI Infrastructure**: `@valeriiakuka` shared an article discussing Databricks’ impact on the generative AI space and its strategy given recent acquisitions. The article showcases Databrick's trajectory and potential growth directions in the AI industry ([read the full story](https://www.turingpost.com/p/databricks)).

**Links mentioned**:

- [MotionCtrl SVD - a Hugging Face Space by TencentARC](https://huggingface.co/spaces/TencentARC/MotionCtrl_SVD): no description found
- [Intel/ldm3d-pano · Hugging Face](https://huggingface.co/Intel/ldm3d-pano): no description found
- [Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157): Reinforcement Learning (RL) has shown remarkable abilities in learning policies for decision-making tasks. However, RL is often hindered by issues such as low sample efficiency, lack of interpretabili...
- [Google Colaboratory](https://colab.research.google.com/drive/1i6cDDsZfGB70fNgxiUxbUufeQfaWyCfd?usp=sharing): no description found
- [LargeWorldModel/LWM-Text-Chat-1M · Hugging Face](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-1M): no description found
- [Large World Models](https://largeworldmodel.github.io/): no description found
- [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268): Current language models fall short in understanding aspects of the world not easily described in words, and struggle with complex, long-form tasks. Video sequences offer valuable temporal information ...
- [Databricks: the Future of Generative AI in the Enterprise Arena](https://www.turingpost.com/p/databricks): Explore Databricks&#x27; unusual history, its contributions to the generative AI field for Enterprise, and the company&#x27;s strategy and vision of the AI industry.
- [Feature Reuse and Scaling: Understanding Transfer Learning with Protein Language Models](https://www.biorxiv.org/content/10.1101/2024.02.05.578959v1): Large pretrained protein language models (PLMs) have improved protein property and structure prediction from sequences via transfer learning, in which weights and representations from PLMs are repurpo...
- [Proof Wallis Product using integration - Art Of Mathematics](https://mathematicsart.com/solved-exercises/proof-wallis-product-using-integration/): Proof Wallis Product Using Integration Home -&gt; Solved problems -&gt; Wallis product Solution Consider (J_{n}=int_{0}^{frac{pi}{2}}
- [no title found](https://deep-fake-generated-people-facial-recognition.streamlit.app/): no description found

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1208013684765491280) (10 messages🔥): 

- **Shout-Out to the Creator**: `@noir_bd` expressed admiration for a creation and showed interest in learning how it was made, giving credit to `<@848983314018336809>`. `@tony_assi` acknowledged the praise with a cool hugging emoji.
  
- **Rollout of New AI Models**: `@myg5702` announced the addition of new models including *OpenDalle 1.1*, *Kandinsky 2.2*, and others to the collection available at [FumesAI Best-Image-Models-V2](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2) with a supportive comment by `@bean217`.

- **Introducing Fluently, a New Diffusion Model**: `@ehristoforu` shared links to the new diffusion model **Fluently** across various platforms including Hugging Face, CivitAI, and a demo on ZeroGPU. Follow-up messages provided additional context for the images produced by [Fluently V1](https://huggingface.co/ehristoforu/Fluently-v1).

- **ProteinBERT's Impressive Launch**: `@grimsqueaker` introduced **ProteinBERT**, detailing its novel architecture and efficiency in a non-traditional domain. Links to both the original Keras-based [GitHub repo](https://github.com/nadavbra/protein_bert) and a PyTorch port by LucidRains, as well as the associated [research paper](https://doi.org/10.1093/bioinformatics/btac020) and recently uploaded untested Hugging Face model weights, were offered.

- **LocalLlm Hosted API Project Unveiled**: `@typoilu` presented their project for hosting an API on a free Google Colaboratory environment, facilitating experiments with large open-source language models. They invited feedback on the [LocalLlm GitHub repository](https://github.com/groloch/LocalLlm).

**Links mentioned**:

- [Best Image Models V2 - a Hugging Face Space by FumesAI](https://huggingface.co/spaces/FumesAI/Best-Image-Models-V2): no description found
- [GitHub - groloch/LocalLlm: Drop-in and advanced solutions to experiment with open source LLM !](https://github.com/groloch/LocalLlm): Drop-in and advanced solutions to experiment with open source LLM ! - groloch/LocalLlm
- [ProteinBERT: a universal deep-learning model of protein sequence and function](https://doi.org/10.1093/bioinformatics/btac020): AbstractSummary. Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biolo
- [GitHub - nadavbra/protein_bert](https://github.com/nadavbra/protein_bert/tree/master): Contribute to nadavbra/protein_bert development by creating an account on GitHub.
- [GitHub - lucidrains/protein-bert-pytorch: Implementation of ProteinBERT in Pytorch](https://github.com/lucidrains/protein-bert-pytorch): Implementation of ProteinBERT in Pytorch. Contribute to lucidrains/protein-bert-pytorch development by creating an account on GitHub.
- [GrimSqueaker/proteinBERT · Hugging Face](https://huggingface.co/GrimSqueaker/proteinBERT): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1207967084370198589) (58 messages🔥🔥): 

- **Countdown to PEFT Integration Presentation**: `@prateeky2806` confirmed a presentation set for Friday, March 1st, discussing the integration of different merging methods in the PEFT library and planning to include a demo. The relevant PR can be found on [GitHub](https://github.com/huggingface/peft/pull/1364).
- **Enthusiastic Participation in Reading Group**: Reading group sessions are sparking positive reactions, with `@tonic_1` and `@chad_in_the_house` expressing excitement for upcoming presentations, while `@samx9128` voiced enthusiasm for attending their first reading group.
- **Navigating Discord Tech Issues**: Members, including `@chad_in_the_house`, `@lunarflu`, and `@tea3200`, assisted others like `@ericauld` with technical difficulties during a mamba presentation, directing them to the correct channels and permissions.
- **YouTube as a Mamba Resource**: `@ericauld` and others share valuable YouTube resources discussing Mamba and SSMs, which can augment understanding for those following along or unable to attend live discussions ([Samuel Albanie's video](https://www.youtube.com/watch?v=ouF-H35atOY&t=305s&ab_channel=SamuelAlbanie), [Umar Jamil's video](https://www.youtube.com/watch?v=8Q_tqwpTpVU&ab_channel=UmarJamil), [a compiled playlist](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe)).
- **Mamba Paper Controversies and Anticipated Talks**: `@chad_in_the_house` highlights issues from a reviewer perspective on the Mamba paper rejection and points to a future presentation by `@1191190979580022875` about "Neural Circuit Diagrams." The GitHub repository for past reading group presentations is mentioned [here](https://github.com/isamu-isozaki/huggingface-reading-group).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/907325990236213288): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/events/879548962464493619/1208115157121896519): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/1203285086624157696): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Mamba and SSM](https://www.youtube.com/playlist?list=PLy8JSKQ3FEvaTTzRDnxnHdquNvrVZDExe): no description found
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a...
- [Paper page - Neural Circuit Diagrams: Robust Diagrams for the Communication,
  Implementation, and Analysis of Deep Learning Architectures](https://huggingface.co/papers/2402.05424): no description found
- [Tweet from FxTwitter / FixupX](https://x.com/vtabbott): Sorry, that user doesn't exist :(
- [Tweet from Vincent Abbott | Deep Learning (@vtabbott_)](https://x.com/vtabbott_/status/1743204563015102594?s=20): Just blogged about my Mixtral by @MistralAI diagrams on @huggingface on the suggestion of @reach_vb. Just got my HF account set up - so throw me a follow. It seems like a great platform to become invo...
- [Hugging Face Reading Group 13: Mamba](https://www.youtube.com/watch?v=CWQuL8dpCRY): Presenter: Eric Auld
- [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv&referrer=%5Bthe%20profile%20of%20Xinlei%20Chen%5D(%2Fprofile%3Fid%3D~Xinlei_Chen1)): This paper shows that the masked-modelling principle driving the success of large foundational language models can be effectively applied to video by making predictions in latent space. We...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Paper Explained)](https://www.youtube.com/watch?v=9dSkvxS2EB0): #mamba #s4 #ssm OUTLINE:0:00 - Introduction0:45 - Transformers vs RNNs vs S46:10 - What are state space models?12:30 - Selective State Space Models17:55 - Th...
- [Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU&ab_channel=UmarJamil): Explanation of the paper Mamba: Linear-Time Sequence Modeling with Selective State SpacesIn this video I will be explaining Mamba, a new sequence modeling ar...
- [Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY&t=305s&ab_channel=SamuelAlbanie): Mamba is a new neural network architecture proposed by Albert Gu and Tri Dao.Timestamps:00:00 - Mamba - a replacement for Transformers?00:19 - The Long Range...
- [GitHub - isamu-isozaki/huggingface-reading-group: This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group](https://github.com/isamu-isozaki/huggingface-reading-group): This repository&#39;s goal is to precompile all past presentations of the Huggingface reading group - isamu-isozaki/huggingface-reading-group

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1207985152605552640) (13 messages🔥): 

- **DPO enthusiasts unite**: `@maxpappa` mentions using full DPO, and `@arturzm` responds, expressing that it's not a problem but instead a "godly speed boost" and casually enquires about what's new without additional context.
- **Looking for one-liner model conversion**: `@blackbox3993` seeks a simple one-liner for converting any model to BitsAndBytes, sharing a [lengthy quantization documentation](https://huggingface.co/docs/bitsandbytes/main/en/quantization).
- **Tips for loading models in BitsAndBytes**: `@gugaime` explains to `@blackbox3993` that to convert a custom model to BitsAndBytes, one should replace specific modules like `Linear8bitLt` with exact shapes and offers code snippets to guide the process.
- **Seeking Diffusion Model Insight**: `@sardarkhan_` looks for resources to understand the mathematics of diffusion models at a deeper level, feeling overwhelmed by the current complexity; `@chad_in_the_house` and `@wandereronarock` suggest starting with the DDPM paper and a blog by Lillian Weng.
- **Diffusion Models Beyond Images**: `@little.stone` questions the applicability of diffusion techniques, such as the DDPM scheduler from the diffusor repo, on time series data rather than images, indicating a potential shift from standard use cases.

**Links mentioned**:

[Quantization primitives](https://huggingface.co/docs/bitsandbytes/main/en/quantization): no description found

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1208155753815220234) (7 messages): 

- **UI Elements for Training Inquiry**: `@conceptron` expressed interest in a project and inquired about **UI elements** such as buttons, checkboxes, and sliders used for training purposes, but provided no further specifics or context.
- **Search for a Sora-building Community**: `@andrew_ulterior` asked if there were any dedicated discord channels for people building their own version of **Sora**, without expanding on what Sora refers to.
- **Patching the Weather Prediction Model**: `@smallcrawler` is seeking advice on **correcting patch edge effects** in autoregressive transformers, specifically for a global weather prediction model that shows artifacts over time.
- **Filtering AI Image Artifacts**: `@korner83` is working on a project that requires identifying and filtering out **AI-generated image artifacts**, and asked for best practices, tools, or models to improve the efficiency of this pre-curation phase.
- **Statricks Founder Offers Data and Experience**: `@ironman5769` provided a back story about his company, Statricks, and his journey towards using **computer vision** to identify products from ads, suggesting he has the data and interest to assist in a related project though currently lacks motivation and capital.
- **Direction for Detecting Image Artifacts**: In response to filtering AI artifacts, `@ironman5769` suggested checking out Google's [Know Your Data](https://knowyourdata-tfds.withgoogle.com), which could potentially be a direction to explore.
- **Guidance on Finetuning BLIP2 Requested**: `@seanb2792` is looking for assistance with **finetuning BLIP2**, but no details were given about the specific issues or context for the request.

**Links mentioned**:

[Know Your Data](https://knowyourdata-tfds.withgoogle.com): no description found

  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1207986070743023686) (10 messages🔥): 

- **Seeking ML Building Blocks for Code Detection**: `@brady_kelly` expressed interest in detecting boilerplate code and autogenerated README files within a CI/CD pipeline. They requested guidance on basic concepts at a high-level overview rather than specific code implementation.

- **Exploring Translation Model Architectures**: `@manavsarkar` found poor results using an encoder-decoder architecture for language translation and questioned if there are alternative methods. In response, `@calmdown.manu` suggested decoder-only architectures and the use of tags to separate original and translated sentences could be a solution.

- **Translation Models Grasping Nuanced Meaning**: In addition to architecture concerns, `@manavsarkar` wondered how translation models understand when to translate nouns with the same pronunciation across languages. `@calmdown.manu` mentioned that pointer networks and attention mechanisms usually learn to handle such nuances.

- **Interest in Fine-tuning WikiSQL on Smaller Models**: `@miguelkjh` inquired about experiences with fine-tuning models like GPT-2 or Pythia on the WikiSQL dataset for smaller projects, seeking insights on challenges and performance improvements.

- **Tools for Dataset Deduplication Needed**: `@abrahamowodunni` requested recommendations for tools capable of performing deduplication on large datasets.

- **Technical Issue with Python Import**: `@vikas8715` experienced an `ImportError` when trying to import `is_torch_sdpa_available` from the `transformers` library, revealing a potential dependency problem or a version incompatibility in their setup.
  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1207985152605552640) (13 messages🔥): 

- **Full DPO Utilized**: `@maxpappa` mentioned they are using **full DPO**, but provided no additional context about their experience or outcomes.
- **BitsAndBytes Made Simple by @gugaime**: To convert a model to BitsAndBytes, `@gugaime` recommends simply loading the model with a parameter `load_in_4bit=True` or using a `BitsAndBytesConfig`. They also shared code snippets for more advanced usage scenarios.
- **Custom Models with BitsAndBytes**: For those not using `AutoModel`, `@gugaime` advised `@blackbox3993` on replacing their custom model’s linear module with `Linear8bitLt` from BitsAndBytes, providing a code example.
- **Seeking Diffusion Models Wisdom**: `@sardarkhan_` is on the hunt for resources to gain a deeper mathematical understanding of diffusion models, feeling overwhelmed by the complexity. Fellow users suggested resources like the DDPM paper and Lillian Weng's blog for clarity.
- **Time Series Diffusion Extraordinaire**: `@little.stone` inquired about applying the diffusor library to time series data with a custom network. They're curious about the compatibility of functions like the DDPM scheduler with non-image modalities.

**Links mentioned**:

[Quantization primitives](https://huggingface.co/docs/bitsandbytes/main/en/quantization): no description found

  

---



### LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/) (1 messages): 

jerryjliu0: webinar happening now!
  

---


### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1208090704941944944) (9 messages🔥): 

- **RAG Retrieval-Evaluation Loop**: A technique to improve **RAG's retrieval quality** involves using an LLM to evaluate and filter the relevance of results before synthesis, as mentioned in a tweet resulting in a [discussion of the method](https://twitter.com/llama_index/status/1758530939276378255).
- **Step-by-Step Video Analysis with RAG**: @lancedb outlined a process for **video analysis using RAG**, including frame splitting, audio transcription, and embedding for database retrieval, detailed in a [blog post](https://t.co/HmkMzF0c1n) with impressive results shared on [Twitter](https://twitter.com/llama_index/status/1758587997178728796).
- **End-to-End ML Deployment Guide**: @DGallitelli95 wrote an article demonstrating how to create a **RAG pipeline with Huggingface and AWS Sagemaker**, providing a guide for deploying ML models from selection to endpoint creation, showcased on [Twitter](https://twitter.com/llama_index/status/1758654210378473731).
- **Nomic Embedding Model's Tradeoff Flexibility**: The **open-source nomic-embed-text-v1.5 embedding model** allows dynamic trade-offs between memory, storage, bandwidth, and performance, spanning 64 to 768 dimensions, inspired by Matryoshka Representation Learning. For details, see tweet threads and [Nomic's blog](https://home.nomic.ai) ([Tweet](https://twitter.com/llama_index/status/1758901855508382149)).
- **Restaurant Menu RAG-powered Chatbot How-To**: A **@weights_biases blog post** demonstrates how to build a full-stack restaurant menu chatbot using LlamaIndex, featuring built-in app usage logging via Weave, as announced in a [Twitter post](https://twitter.com/llama_index/status/1758965798377578688).

**Links mentioned**:

[Unboxing Nomic Embed v1.5: Resizable Production Embeddings with Matryoshka Representation Learning](https://t.co/8gBvrdxlov): Nomic introduces a truly open text embedding model with resizable embeddings.

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1207985513428947014) (222 messages🔥🔥): 

- **Troubleshooting LlamaIndex Installation**: `@lapexer` encountered an import error after installing LlamaIndex, which was resolved by creating a new environment with Python 3.10 and installing `llama-index-core` alongside `llama-index`.
- **Frustration with Parallel Embeddings**: `@ben25635` expressed frustration with the lack of native support for parallelized embeddings in LlamaIndex compared to LangChain. `@cheesyfishes` provided guidance on increasing batch sizes and using `IngestionPipeline` for parallel processing.
- **AzureOpenAI and LlamaIndex 0.10.x Issue**: `@disco.dr` faced an exception using AzureOpenAI with LlamaIndex 0.10.x. The issue was resolved by ensuring both `aclient` and `client` were passed to `QdrantVectorStore` and by installing specific packages.
- **Optimizing Information Extraction from PDFs**: `@gryhkn` sought advice for efficiently extracting information about economic reforms from PDF reports. `@kapa.ai` suggested a process utilizing LlamaIndex's data connectors, indexes, engines, data agents, and integrations.
- **Evaluating Advantages of ReAct Agent and QueryPipeline**: `@andysingal` shared a link explaining the benefits of combining custom AI models in LlamaIndex with external tools like Ollama. They also informed `@vett93` about alternatives mlx and openllm and their relation to LM Studio.
- **Upgrading Llama-Index Packages**: `@vett93` asked about upgrading Llama-Index packages. `@cheesyfishes` confirmed that in version 0.10.x, packages are independently versioned, typically requiring only an update of `llama-index-core`.

**Links mentioned**:

- [no title found](http://192.168.0.105:1234")): no description found
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/]): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [no title found](https://llamahub.ai/l/llama-packs/llama-index-packs-tables?from=all): no description found
- [Documents / Nodes - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/root.html): no description found
- [Ingestion Pipeline - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/root.html#parallel-processing): no description found
- [Local Llama2 + VectorStoreIndex - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local.html): no description found
- [app-llamaindex-v0.10.py](https://gist.github.com/sumvedshami/066e694fe25f51a317135e079b074115): GitHub Gist: instantly share code, notes, and snippets.
- [llama_index/llama-index-core/llama_index/core/download/pack.py at a900d5e67424c2b2c46b0aa9ef62502af556d449 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/a900d5e67424c2b2c46b0aa9ef62502af556d449/llama-index-core/llama_index/core/download/pack.py#L110-L117): LlamaIndex (formerly GPT Index) is a data framework for your LLM applications - run-llama/llama_index
- [Query Pipeline for Advanced Text-to-SQL - LlamaIndex 🦙 v0.10.7](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql.html#advanced-capability-1-text-to-sql-with-query-time-table-retrieval): no description found
- [Not able to use LLama Packs [Bug]:  · Issue #10777 · run-llama/llama_index](https://github.com/run-llama/llama_index/issues/10777): Bug Description I always get this error for any pack i download FileNotFoundError: [Errno 2] No such file or directory: &#39;/content/chain_of_table_pack/llama_index/packs/tables/base.py&#39; Version ...
- [OpenAI&#39;s Agent 2.0: Excited or Scared?](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx): I want to give you a full run down of browser/mobile/desktop AI agentsGet free HubSpot E-book: Using Generative AI to scale your content operation: https://c...
- [Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex.](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5): Discover how to optimize RAG’s chunk size for peak performance using LlamaIndex’s Response Evaluation
- [RAG is Dead! Long Live RAG!](https://vectorize.io/2024/02/16/rag-is-dead-long-live-rag/): Yesterday Google announced Gemini 1.5, which features very long context windows, up to 1 million tokens. This is quite an advancement compared to existing models with longer contexts, such as GPT-4…
- [Training Mixtral 8x7B Locally with LlamaIndex Integration: Customizing AI Models for Your Data](https://medium.com/ai-advances/training-mixtral-8x7b-locally-with-llamaindex-integration-customizing-ai-models-for-your-data-4c704e693e59): Ankush k Singal

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1208031984799055922) (24 messages🔥): 

- **Deciphering Whisper Transcripts for RAG**: `@cablecutter` raises a question about the best practices for utilizing **Whisper Transcripts** with RAG functionalities, especially with regard to keeping metadata like "speaker," "confidence," and "timestamps". `@amgadoz` suggests using an LLM for speaker diarization, chunking the transcript by speakers or turns, and adding metadata and text from previous segments for context. (No links provided)
  
- **LlamaIndex Update Guidance Sought**: `@badrinathsvn_72554` encounters `ImportError` issues after upgrading to **LlamaIndex 0.10.6 from 0.9.13**. `@cheesyfishes` responds with suggestions to check out the [migration guide](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232) for addressing compatibility issues. (Actual link provided leads to a non-public Discord server)

- **New Blog on Prompt-Driven RAG Challenges**: `@subham5089` shares [a blog post](https://www.linkedin.com/posts/subham-kundu-2746b515b_generativeai-knowledgesharing-activity-7164649470624686080-Zno7) discussing the current challenges associated with using **prompt-driven RAG systems** and potential solutions for those challenges.

- **React RAG QA Frontend Boilerplate Released**: `@sl33p1420` introduces an open-source **React RAG QA frontend boilerplate** sponsored by runelab, sharing a comprehensive [Medium guide](https://medium.com/@marco.bertelli/unveiling-the-power-of-rag-building-an-interactive-chatbot-with-react-a-comprehensive-guide-99c409a5f69a) on building an interactive chatbot with React.

- **RAG's Relevance in the Wake of Gemini 1.5**: `@chiajy` expresses optimism for RAG systems post **Gemini 1.5**, citing its non-black box nature and how it provides control over accuracy, costs, and latency. They share a [Medium article](https://medium.com/enterprise-rag/why-gemini-1-5-and-other-large-context-models-are-bullish-for-rag-ce3218930bb4) discussing how emerging large context models like Gemini 1.5 will positively affect RAG systems.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1059199217496772688/1073670729054294197/1207845501660168232): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Why Gemini 1.5 (and other large context models) are bullish for RAG](https://medium.com/enterprise-rag/why-gemini-1-5-and-other-large-context-models-are-bullish-for-rag-ce3218930bb4): Optimization via RAG: How to overcome Accuracy, Cost, Latency and other performance limitations of large context models.
- [Unveiling the Power of RAG: Building an Interactive Chatbot with React — A Comprehensive Guide](https://medium.com/@marco.bertelli/unveiling-the-power-of-rag-building-an-interactive-chatbot-with-react-a-comprehensive-guide-99c409a5f69a): Previous Articles:
- [SOCIAL MEDIA TITLE TAG](https://os-copilot.github.io/): SOCIAL MEDIA DESCRIPTION TAG TAG
- [Unleashing the Power of Agents and QueryPipeline with LlamaIndex](https://medium.com/@andysingal/unleashing-the-power-of-agents-with-llamaindex-3efe72921b10): Ankush k Singal

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1207969569377878016) (32 messages🔥): 

- **Misconceptions about AI Identity**: `@lightningralf` pointed out that asking a model about its identity is not valuable as it can only respond within its system prompt, and any identification like "gemini 500" has no substance.
- **Eugene Yan & Jason Liu Livestream Announcement**: `@swyxio` shared a [YouTube video](https://m.youtube.com/watch?v=PU_MErIaAEU) featuring a livestream discussion between Eugene Yan & Jason Liu, with Hamel joining as a surprise guest, as noted by `@eugeneyan`.
- **Taking AI Guardrails Seriously**: `@sugaroverflow` laughed at an [Ars Technica tweet](https://x.com/arstechnica/status/1758540835132494119?s=20) about Air Canada having to honor a refund policy invented by their chatbot, which `@swyxio` responded might be the only way businesses take AI guardrails seriously.
- **The Legal Status of AI**: `@mdcker` highlighted a situation where Air Canada argued against liability for its chatbot's actions, stating "**the chatbot is a separate legal entity**". The judge, however, did not accept this defense.
- **OpenMoE Paper Review**: `@intheclouddan` shared a link to a paper on [OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf), a family of open-sourced Mixture-of-Experts Large Language Models, though `@swyxio` commented the results were not great with training on less data than tinyllama and a lack of exploration on inference time efficiency.

**Links mentioned**:

- [no title found](https://news.ycombinator.com/item?id=39411748): no description found
- [Tweet from Ars Technica (@arstechnica)](https://x.com/arstechnica/status/1758540835132494119?s=20): Air Canada must honor refund policy invented by airline’s chatbot https://trib.al/s84FkPu
- [OpenMoE/paper/paper.pdf at main · XueFuzhao/OpenMoE](https://github.com/XueFuzhao/OpenMoE/blob/main/paper/paper.pdf): A family of open-sourced Mixture-of-Experts (MoE) Large Language Models - XueFuzhao/OpenMoE
- [Chat w/ Eugene Yan &amp; Jason Liu](https://m.youtube.com/watch?v=PU_MErIaAEU): We&#39;re going to try to live stream our next one on one, see how productive we can be.
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe): Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. - karpathy/minbpe

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1207989359689859092) (8 messages🔥): 

- **BERT Paper in a Flash**: `@ivanleomk` announced a **3-minute discussion on the BERT paper**. Interested participants can sign up [here](https://lu.ma/fcsum9r1).

- **Join the LLM Paper Club (Asia Edition!)**: `@swyxio` invited users to the **LLM Paper Club (Asia Edition!)**; to attend, follow the provided [Discord link](https://discord.com/channels/822583790773862470/1200029657744027658).

- **Podcast Alert - New Episode with Modal**: `@swyxio` shared the latest **Latent Space podcast episode featuring Modal**. The conversation includes the impact of [OpenAI’s Sora](https://news.ycombinator.com/item?id=39386156) and [Gemini 1.5](https://news.ycombinator.com/item?id=39383446), upcoming events, and insights on serverless infrastructure for AI [Listen here](https://www.latent.space/p/modal).

- **Spread the Word about Serverless AI**: `@swyxio` requested help in promoting a **Latent Space blog post** about truly serverless infrastructure for AI engineers [Support here](https://x.com/FanaHOVA/status/1758568180132536471?s=20).

- **Dive into the State of Agents**: `@swyxio` highlighted an ongoing session led by `<@363877777977376768>` focused on the **current state of agents**. Users can join the discussion via this [Discord link](https://discord.com/channels/822583790773862470/1200548371715342479).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200029657744027658): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200548371715342479): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Truly Serverless Infra for AI Engineers - with Erik Bernhardsson of Modal](https://www.latent.space/p/modal): Building the ultimate Self Provisioning Runtime for AI Engineers, why Oracle Cloud is underrated, state of GPU parallelism, and why you should staff up with IOI Gold Medallist engineers
- [Tweet from Alessio Fanelli (@FanaHOVA)](https://x.com/FanaHOVA/status/1758568180132536471?s=20): 🆕 Truly Serverless Infra for AI Engineers https://www.latent.space/p/modal  In 2021 @bernhardsson wrote a &#34;Software Infra 2.0 wishlist&#34;, but quickly decided to take matter into his own hands ...

  

---


### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/1207990281132052520) (32 messages🔥): 

- **Delving into BERT's Impact on Google Search**: `@ivanleomk` shared a [summary](https://llm-paper-club-asia-notes.vercel.app/papers/bert) and `@swyxio` discussed the use of **BERT** in improving Google search results raising curiosity about the *how*, while `@bryanblackbee` suggested that Google might use document embeddings for semantic search.
- **Wonders of BERT's Bidirectionality**: In the discussion, `@swyxio` showed surprise at the invention of the bidirectional model **BERT** before the unidirectional **GPT**, hinting at the nuanced evolution of NLP models.
- **Swyxio's Internet Woes**: Despite connectivity issues, `@swyxio` expressed enjoyment of the session recap before bidding everyone good night.
- **Questions about Model Training and Quality**: Newcomer `@farukga` was curious about the kind of "satisfying" text or quality information against which large models like Google's are trained and fine-tuned.
- **Anticipation for Next LLM Paper**: Towards the end of the chat, members such as `@joellee.` queried about next week's paper and `@mattoshimasu` asked about the recording availability for the session, with participants thanking `@ivanleomk` for the walkthrough.

**Links mentioned**:

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://jalammar.github.io/illustrated-bert/): Discussions: Hacker News (98 points, 19 comments), Reddit r/MachineLearning (164 points, 20 comments)   Translations: Chinese (Simplified), French 1, French 2, Japanese, Korean, Persian, Russian, Span...
- [Nextra: the next docs builder](https://llm-paper-club-asia-notes.vercel.app/papers/bert): Nextra: the next docs builder
- [Understanding searches better than ever before](https://blog.google/products/search/search-language-understanding-bert/): How new advances in the science of language understanding will help you find more useful information in Search.

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1208156189297344552) (182 messages🔥🔥): 

- **Exploration of AI, Agents, and State Machines**: Members, including `@markredito` and `@fanahova`, discussed concepts related to AI agents and state machines, mentioning [an episode on the topic](https://youtu.be/4Ps7ahonRCY) and referencing `@davidkpiano` for insight on state machines used in combination with AI.
- **Resource Compilation in Progress**: Several members including `@yikesawjeez` and `@swyxio` were considering compiling a list of tools and resources related to agents, asking for recommendations on state machines and local models, and sharing links to relevant frameworks like [CrewAI](https://github.com/joaomdmoura/crewAI) and [MagickML](https://www.magickml.com).
- **Community Engagement over AI Development Tools**: During the conversation, `@yikesawjeez` mentioned expanding on a hackathon project, `@swyxio` shared a tool used for the podcast [smol-podcaster](https://github.com/FanaHOVA/smol-podcaster), and `@slono` promised to share prettier versions of their slides.
- **Live Testing and Streaming Plans**: `@swyxio` and `@yikesawjeez` discussed plans for live streaming to experiment with AI agent frameworks, and `@swyxio` shared a [YouTube link](https://www.youtube.com/watch?v=S6MtNDIm3oc&ab_channel=swyx) to his stream comparing CrewAI, LangGraph, and AutoGen.
- **Celebrating Community Milestones**: The conversation marked the one-year anniversary of Latent Space, with `@fanahova` highlighting this milestone and `@slono` sharing their excitement about contributing to the community.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1208167505923932261): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1075282825051385876/1110661051311202406): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/979492707279978586/1208139552946913290): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [AI in Action - Agents - mnml's vault - Obsidian Publish](https://publish.obsidian.md/manuel/Notes/Talks/AI+in+Action+-+Agents): AI in Action - Agents - mnml's vault - Powered by Obsidian Publish.
- [StreamYard | Browser-based live studio for professionals](https://streamyard.com/4mnxw8xnp8): StreamYard is a professional live streaming and recording studio in your browser. Record your content, or stream live to Facebook, YouTube, and other platforms.
- [ai onboarding / normies links](https://arc.net/folder/94AD3E8D-38F8-4A0C-9A91-8F2487AB4B20):  
- [AgentOps](https://agentops.ai/): Build AI agents and LLM apps with observability, evals, and replay analytics. No more black boxes and prompt guessing.
- [A Gentle Introduction to CRDTs – vlcn.io](https://vlcn.io/blog/intro-to-crdts): Conflict Free Replicated Data types (CRDTs) can be tricky. You may spend months reading papers and implementing different algorithms before they finally click and become simple. That or they&#x27;ll s...
- [Making state management intelligent - David Khourshid](https://www.youtube.com/watch?v=Iw8Uf7q4nVc): Making state management intelligentManaging state is complicated. Humans are even more complicated. As developers, it&#39;s our job to deliver seamless and intui...
- [Magick - Cutting-edge tools for AI creators](https://www.magickml.com): Experience the power of advanced AI at your fingertips, no code required. With our comprehensive toolkit, effortlessly build, deploy, maintain, and scale your AI agents, bots, and applications to new ...
- [crewAI/src/crewai/agent.py at main · joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/agent.py): Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. - joaomdmoura/cr...
- [[Livestream] CrewAI vs LangGraph vs AutoGen](https://www.youtube.com/watch?v=S6MtNDIm3oc&ab_channel=swyx): no description found
- [Tweet from Alex Reibman (@AlexReibman)](https://x.com/AlexReibman/status/1757335836482498647?s=20): 4/ Crew AI  Orchestration framework for building reliable autonomous AI agents  This example shows an agent that automatically creates and schedules social media posts, but it can research, code, and ...
- [What is missing from current AI?](https://youtu.be/4Ps7ahonRCY?si=U_V425_OfLORLGHR): Brandon Rohrer who obtained his Ph.D from MIT is driven by understanding algorithms ALL the way down to their nuts and bolts, so he can make them accessible ...
- [Build Agents from Scratch (Building Advanced RAG, Part 3)](https://youtu.be/T0bgevj0vto?si=YeW08q4tVqm3wZej): In this third video of this series we teach you how to build LLM-powered agentic pipelines - specifically we teach you how to build a ReAct agent (Yao et al....
- [GitHub - FanaHOVA/smol-podcaster: smol-podcaster is your autonomous podcast production intern 🐣](https://github.com/FanaHOVA/smol-podcaster): smol-podcaster is your autonomous podcast production intern 🐣 - FanaHOVA/smol-podcaster
- [GitHub - Actioninsight/AutoNL: AutoNL - Natural Language Automation tool](https://github.com/Actioninsight/AutoNL): AutoNL - Natural Language Automation tool. Contribute to Actioninsight/AutoNL development by creating an account on GitHub.
- [crewAI/src/crewai/crew.py at main · joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI/blob/main/src/crewai/crew.py): Framework for orchestrating role-playing, autonomous AI agents. By fostering collaborative intelligence, CrewAI empowers agents to work together seamlessly, tackling complex tasks. - joaomdmoura/cr...
- [Join the The Arena Online Discord Server!](https://discord.gg/eGrzMA2d): ordinary emergency dev day &amp; gpts store hackathon server! definitely not a secret technomancer enclave or anything! | 373 members

  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1207976954343464981) (46 messages🔥): 

- **A New Supercomputing Era in Spain**: `@mdelamor` shared an article about the inauguration of Marenostrum 5, a new extension to a supercomputing facility housed in a former church, signifying a new era for European supercomputing. The article can be read here: [Marenostrum 5 - a new era](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/).

- **Installation Issues with Nvidia Drivers**: `@apaz` expressed frustration with installing proprietary Nvidia drivers on a new PC; they had to manually apply a patch from bookworm-proposed-changes. Other users, like `@joseph_en`, discussed ongoing challenges with Nvidia drivers and the potential for AMD to offer competition in the future.

- **C++ Template Troubles**: Developer `_davidgonmar` experienced issues with a C++ template that compiles with NVCC and g++ on WSL, but causes errors in Visual Studio. `@jeremyhoward` suggested ensuring C++17 is enabled in Visual Studio as the compiler needed to handle fold expressions, which turned out to be the solution.

- **4D Gaussian Splatting Discussions**: `@andreaskoepf` prompted a discussion on 4D Gaussian Splatting by sharing a link to a website [gmix.ai](https://www.gmix.ai/), and `@joseph_en` asked for recommended papers on the topic to gain better insight into these recent advances.

- **Challenging Kernel Debugging Marathon**: `_davidgonmar` detailed the arduous debugging of a CUDA kernel where incorrect memory guard conditions led to out-of-bounds writes. `@gogators.` suggested using Nvidia's compute sanitizer, a tool included with CUDA, to detect illegal memory accesses and race conditions.

**Links mentioned**:

- [Gmix 4D Spatial Video](https://www.gmix.ai/): no description found
- [Error installing Nvidia driver on Debian 12.5 · Issue #361 · NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit/issues/361): Issue Summary: When attempting to install the Nvidia driver (nvidia-driver) on Debian 12.5, I encountered an error preventing successful installation. Error Message: Setting up nvidia-persistenced ...
- [no title found](https://bnnbreaking.com/world/spain/inauguration-of-marenostrum-5-a-new-era-for-european-supercomputing/): no description found

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1208105646579388477) (9 messages🔥): 

- **Nsight Insights**: User `@lancerts` expressed excitement about utilizing **Nsight**, but mentioned the need to decipher the detailed numbers provided by the tool.
- **Seeking GPU Usage Metrics**: `@marvelousmit` inquired about measuring GPU saturation via **Nsight Systems**, and `@lancerts` responded that during a profile with trace, you can see those details.
- **CUDA 12.1 Boosts Kernel Parameter Limit**: `@iron_bound` shared a [blog post](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/) detailing **CUDA 12.1**'s increase of the kernel parameter limit from 4,096 bytes to 32,764 bytes, enhancing functionality for developers like `@morousg` who fuses many kernels in their work.
- **Consider Electrical Performance in Time-Critical Applications**: User `@defaultguyredshirt` advised that aside from time performance, electrical performance is also a factor that may influence the selection of case pickers.
- **Balancing GPU Flops-per-Byte**: `@andreaskoepf` shared a tweet from `@RajaXg` discussing the imbalance of flops-per-byte in GPU development over the years, while `@morousg` emphasized the need for optimization in GPU libraries to improve the usage of memory bandwidth and registers.

**Links mentioned**:

- [Tweet from Raja Koduri (@RajaXg)](https://x.com/RajaXg/status/1758935199508046247): GPU Flops-per-byte went crazy over the years.  It&#39;s even more insane if you plot this for interconnect (PCIE, NVLink, XeLink etc)  bandwidth  We were close to 1:1 when the first floating point sha...
- [CUDA 12.1 Supports Large Kernel Parameters | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/): CUDA 12.1 offers you the option of passing up to 32,764 bytes using kernel parameters, which can be exploited to simplify applications as well as gain performance improvements.

  

---


### CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1208501814153060392) (1 messages): 

- **Tune in for CUDA Mastery**: `@marksaroufim` alerted `@everyone` that **CUDA MODE Lecture 6: Optimizing PyTorch Optimizers** was about to begin, emphasizing that Jane's contributions are pivotal every time a PyTorch model is trained. Audience members looking for practical CUDA insights were highly encouraged to attend.
  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1208069526877503488) (11 messages🔥): 

- **Collaborative Call for Project Integration**: `@shindeirou` is looking to integrate a challenge as part of a course project with the assistance of `@419115250021826560`, inviting them for a call to discuss further, and mentions the inclusion of three university peers.
- **Open to Assistance and Exploration**: `@mickgardner` offers help on the project under discussion, planning to review baseline code and related paper.
- **Large World Models Unveiled**: `@andreaskoepf` shares information about the **Large World Model** (LWM), an open-source model trained from LLaMA-2, providing links to its [Overview](/ifioravanti/lwm), [Tags](/ifioravanti/lwm/tags), [GitHub](https://largeworldmodel.github.io/), and [HuggingFace](https://huggingface.co/LargeWorldModel) pages, and mentioning its capability to process long text documents over 1M tokens.
- **RingAttention Implementation Insight**: `@mickgardner` mentions that LWM is still using JAX and has reimplemented the attention module to include both standard ring attention with BPT and a 'ring+flash_attention on tpu' implementation.
- **Scheduling Meetups for Collaboration**: `@andreaskoepf` and `@__daem0n__` discuss scheduling a meeting to collaborate on the project, with proposed times stated in Discord's localized timestamp format, highlighting that `@andreaskoepf` is available 2 hours before and after a lecture today and `@__daem0n__` confirming that tomorrow works fine.
- **Project-Focused Channel Created**: `@andreaskoepf` announces the creation of a separate channel for the RingAttention project to focus discussions and collaborative efforts there.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1208496482005549086): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [ifioravanti/lwm](https://ollama.com/ifioravanti/lwm): Large World Model is an open-source model trained from LLaMA-2 on a subset of Books3 filtered data
- [Large World Models](https://largeworldmodel.github.io/): no description found

  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1208314222186856451) (23 messages🔥): 

- **Shader Demo Enthusiast**: User `@andreaskoepf` shared their admiration for shader demos, referring to the [twigl.app](https://twigl.app/?ol=true&ss=-NqUk6pcpkcFek1iupHp) as an example of impressive work.

- **GPU Programming Novice Dives In**: `@jollyphoenix.ai`, a newcomer interested in efficient machine learning and GPU programming, sought advice on getting started and shared a [YouTube playlist](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb) about **Heterogenous Parallel Programming**.

- **CUDA Resource Recommendation**: `@marksaroufim` recommended checking out a YouTube channel and a [GitHub resource stream](https://github.com/cuda-mode/resource-stream) for materials on CUDA programming, responding to `@jollyphoenix.ai`.

- **Groq's Inferencing Speed Sparks Discussion**: `@cs_os_05101` brought up [Groq](https://groq.com/), a startup known for fast inferencing, stirring a conversation about how they might achieve this speed and how they differ from traditional CUDA architectures.

- **Real-World CUDA Learning Request**: `@cs_os_05101` expressed a desire for practical code examples in the CUDA-MODE repository for specific topics, citing a recent presentation and the prohibitive cost of the comprehensive book on CUDA.

**Links mentioned**:

- [Groq](https://groq.com/): no description found
- [twigl.app](https://twigl.app/?ol=true&ss=-NqUk6pcpkcFek1iupHp): twigl.app is an online editor for One tweet shader, with gif or webm generator and sound shader.
- [Heterogenous Parallel Programming - CUDA Programming](https://www.youtube.com/playlist?list=PLzn6LN6WhlN06hIOA_ge6SrgdeSiuf9Tb): no description found
- [TensorRT-LLM/docs/source/blogs/H200launch.md at main · NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/H200launch.md#h200-achieves-nearly-12000-tokenssec-on-llama2-13b-with-tensorrt-llm): TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficie...
- [GitHub - cuda-mode/resource-stream: CUDA related news and material links](https://github.com/cuda-mode/resource-stream): CUDA related news and material links. Contribute to cuda-mode/resource-stream development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1207985945199116318) (4 messages): 

- **Surprise at Row/Col Performance**: `@mikkelisk` shared their benchmarking results and expressed surprise at the lack of difference between **row** and **col** operations. They suspect a mistake in implementation and are curious about what GPU others are using.

- **Atomic Add is Necessary for Concurrent Memory Edits**: `@eporat` explained the necessity of `atomicAdd` when multiple threads modify the same memory value, implying it was essential for their code to work properly.

- **Transposing for Performance**: `@andreaskoepf` linked to the `flash-attention` GitHub repo where a transpose operation is used in performance optimization, as demonstrated in the [flash-attention source code](https://github.com/Dao-AILab/flash-attention/blob/5cdabc2809095b98c311283125c05d222500c8ff/csrc/flash_attn/flash_api.cpp#L372-L380).

- **Discrepancy between CUDA Cores and Warp Threads**: `@lucaslingle` queried a statement from Chapter 4 of the referenced book, probing how 32 threads per warp in an A100 could be simultaneously executed by a processing block with only 16 CUDA cores, as per the authors' claim.

**Links mentioned**:

[flash-attention/csrc/flash_attn/flash_api.cpp at 5cdabc2809095b98c311283125c05d222500c8ff · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/blob/5cdabc2809095b98c311283125c05d222500c8ff/csrc/flash_attn/flash_api.cpp#L372-L380): Fast and memory-efficient exact attention. Contribute to Dao-AILab/flash-attention development by creating an account on GitHub.

  

---


### CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1208322419639721984) (1 messages): 

- **New CUDA for Python Video Released**: `@andreaskoepf` announced a new video on the CUDA-mode YouTube channel titled "Lecture 5: Going Further with CUDA for Python Programmers". The materials for the lecture can be found on [GitHub](https://github.com/cuda-mode/lectures).

**Links mentioned**:

[Lecture 5: Going Further with CUDA for Python Programmers](https://www.youtube.com/watch?v=wVsR-YhaHlM): Material here https://github.com/cuda-mode/lectures

  

---


### CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1208082682068734053) (3 messages): 

- **Comparing PyTorch 2.0 and JAX's Graph Capture**: `@vguerra` highlighted differences between PyTorch 2.0 and JAX in terms of graph capturing as outlined in the [PyTorch 2.0 paper](https://pytorch.org/assets/pytorch_2.pdf). JAX's design is heavily influenced by XLA which requires functionally pure programs and doesn't support data-dependent Python control; more details are available in section 2.6 of the paper and an extended comparison can be found in the [torch.fx paper](https://arxiv.org/abs/2112.08429).
- **JAX Fusion Pipeline Analysis**: `@marvelousmit` referenced a paper that explores JAX's fusion pipeline, which can be read in detail at [arXiv](https://arxiv.org/pdf/2301.13062.pdf).
- **Users Express Interest in Deep Learning Papers**: `@marcom79` showed appreciation for the shared resources on deep learning frameworks and confirmed intent to review the provided papers.

**Links mentioned**:

[Torch.fx: Practical Program Capture and Transformation for Deep Learning in Python](https://arxiv.org/abs/2112.08429): Modern deep learning frameworks provide imperative, eager execution programming interfaces embedded in Python to provide a productive development experience. However, deep learning practitioners somet...

  

---


### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1208498435263569940) (79 messages🔥🔥): 

- **CUDA RingAttention Development Commences**: `@andreaskoepf` initiates the development of a CUDA RingAttention implementation, sharing two key papers on Ring Attention techniques and their application in deep learning ([Paper 1: Near-Infinite Context in Transformers](https://arxiv.org/abs/2310.01889), [Paper 2: World Models with RingAttention](https://arxiv.org/abs/2402.08268)) and providing links to the project's [GitHub repo](https://github.com/LargeWorldModel/LWM) and [model on HuggingFace](https://huggingface.co/LargeWorldModel).
  
- **Coordination and Research Begin**: The team, including `@ericauld`, `@jamesmel`, `@nshepperd`, discusses initial steps, such as reviewing papers and exploring existing software implementations for RingAttention, and shares excitement to contribute.

- **Exploration and Brainstorming**: `@andreaskoepf` suggests establishing tasks, reading existing papers on RingAttention, and meeting to discuss proceedings. Existing GitHub repositories for potential RingAttention implementations are [shared](https://github.com/lhao499/RingAttention) by `@lancerts` and `@jku100`.

- **Technical Dialogue and Considerations**: The channel sees technical discussion, with `@andreaskoepf` proposing the use of NCCL for multi-GPU communications and the potential sponsorship of development resources, while `@ericauld` and `@iron_bound` look into existing code and `@lancerts` provides insights on configuring parallelism dimensions for JAX meshes.

- **Setting Up for Collaboration**: `@andreaskoepf` sets up a [new GitHub repository](https://github.com/cuda-mode/ring-attention) specifically for optimized CUDA kernels for RingAttention and invites team members to contribute, with `@jamesmel` expressing interest in taking up an issue related to peer-to-peer memory transfer analysis.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1189498204333543425/1189498205101109301): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [ifioravanti/lwm](https://ollama.com/ifioravanti/lwm): Large World Model is an open-source model trained from LLaMA-2 on a subset of Books3 filtered data
- [Google Colaboratory](https://colab.research.google.com/drive/1PNDTLx2UYYk8XmTb9e_ZBxPx8P6eByvx?usp=sharing): no description found
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867): The Softmax function is ubiquitous in machine learning, multiple previous works suggested faster alternatives for it. In this paper we propose a way to compute classical Softmax with fewer memory acce...
- [Google Colaboratory](https://colab.research.google.com/drive/1X-x6PCRydNY9LZBPLA0DZh3Tj2Dyz60M?usp=sharing): no description found
- [ELI5: Flash Attention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad): Step by step explanation of how one of the most important MLSys breakthroughs work — in gory detail.
- [Analyze overlapped P2P memory transfer and computing · Issue #1 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/1): Create an ipynb to analyze in PyTorch the peer-to-peer (between two GPUs) memory transfer and computing in parallel. Dummy computation could for example be some larger matmuls in a loop. Create not...
- [Comment about use of all gather · Issue #1 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/1): Hi Phil! Hope you&#39;re doing well. As you saw with Gemini Pro 1.5, which works on 1 million tokens, open-source has some work to do to catch up :D porting Ring Attention to PyTorch is definitely the...
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [Papers with Code - Ring Attention with Blockwise Transformers for Near-Infinite Context](https://paperswithcode.com/paper/ring-attention-with-blockwise-transformers): Implemented in 3 code libraries.
- [ring-attention/README.md at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/README.md): Optimized kernels for ring-attention [WIP]. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch?tab=readme-ov-file#usage): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [GitHub - karpathy/minbpe: Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization.](https://github.com/karpathy/minbpe/tree/master): Minimal, clean, code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization. - karpathy/minbpe
- [GitHub - cuda-mode/ring-attention: Optimized kernels for ring-attention [WIP]](https://github.com/cuda-mode/ring-attention): Optimized kernels for ring-attention [WIP]. Contribute to cuda-mode/ring-attention development by creating an account on GitHub.
- [Analyze existing ring-attention implementations · Issue #2 · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/issues/2): Please create a little markdown report about your findings. lhao499/ring-attention/bpt lucidrains/ring-attention-pytorch Get some feeling for the impls: easy to setup and run on our dev machine (e....
- [GitHub - lucidrains/ring-attention-pytorch: Explorations into Ring Attention, from Liu et al. at Berkeley AI](https://github.com/lucidrains/ring-attention-pytorch?tab=readme-ov-file#usa): Explorations into Ring Attention, from Liu et al. at Berkeley AI - lucidrains/ring-attention-pytorch
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889): Transformers have emerged as the architecture of choice for many state-of-the-art AI models, showcasing exceptional performance across a wide range of AI applications. However, the memory demands impo...
- [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268): Current language models fall short in understanding aspects of the world not easily described in words, and struggle with complex, long-form tasks. Video sequences offer valuable temporal information ...
- [Large World Models](https://largeworldmodel.github.io/): no description found
- [GitHub - LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.
- [LargeWorldModel (Large World Model)](https://huggingface.co/LargeWorldModel): no description found
- [RingAttention/bpt/ring_attention.py at main · lhao499/RingAttention](https://github.com/lhao499/ring-attention/blob/main/bpt/ring_attention.py): Transformers with Arbitrarily Large Context. Contribute to lhao499/RingAttention development by creating an account on GitHub.
- [LWM/lwm/ring_attention.py at main · LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM/blob/main/lwm/ring_attention.py#L3?): Contribute to LargeWorldModel/LWM development by creating an account on GitHub.

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1208019765503987742) (54 messages🔥): 

- **8-bit Finetuning Clarification**: User `@nafnlaus00` discussed the possibility of full finetunes on 8bit models and remarked on recent advancements presented by AI Explained. Meanwhile, `@dreamgen` questioned Stability AI's focus given their resources.
- **Hugging Face Down?**: `@le_mess` inquired if Hugging Face was down, but no further information was provided about the service's status.
- **Discussing Stability's New Image Model**: `@nafnlaus00` shared insights on Stability's new diffusion model and pointed out Tesla's use of 3d-informed diffusion for its autonomy tasks.
- **Optimizer Debate**: `@c.gato` questioned whether Adam is still the go-to optimizer for Large Language Models (LLMs), with `@nafnlaus00` suggesting Lion as a more memory-efficient alternative, and `@yamashi` and `@nruaif` discussing how to use and implement Lion optimizers including a link to the GitHub repository ([GitHub - lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch)).
- **Finetune with Axolotl and Model Selection**: Users `@qwerty_qwer`, `@le_mess`, and `@masa_92515` discussed finetuning using Axolotl, with suggestions for 1.6b models like the qwen family as a base. They also discussed VRAM requirements and benchmarks for the 1.3b model.

**Links mentioned**:

- [Tweet from Grant♟️ (@granawkins)](https://x.com/granawkins/status/1758689077472399566?s=20): no description found
- [GitHub - lucidrains/lion-pytorch: 🦁 Lion, new optimizer discovered by Google Brain using genetic algorithms that is purportedly better than Adam(w), in Pytorch](https://github.com/lucidrains/lion-pytorch): 🦁 Lion, new optimizer discovered by Google Brain using genetic algorithms that is purportedly better than Adam(w), in Pytorch - lucidrains/lion-pytorch

  

---


### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1208028060050260048) (19 messages🔥): 

- **Perplexity as a Measure for Difficulty**: `@dreamgen` explored whether perplexity could be used to gauge the difficulty of examples using a baseline model. `@suikamelon` responded by sharing a [paper](https://arxiv.org/abs/2310.13008) that introduces *learnability* as a new dimension for Supervised Fine-Tuning (SFT) of Large Language Models (LLMs), suggesting a method to select SFT data based on the model's ability to learn.

- **Official Implementation of SPIN**: `@nruaif` provided a [GitHub link](https://github.com/uclaml/SPIN?tab=readme) to the official implementation of Self-Play Fine-Tuning (SPIN), indicating an interest or utility for the project in the context of the discussion.

- **Merge Deliberation Over Torch Update**: `@nanobitz` mentioned several `ready to merge` labels in the development pipeline and raised the issue of potentially needing to update PyTorch to version 2.2.x, referencing a link on a Discord channel.

- **Confusion Around Adding New Optimizers**: `@yamashi` expressed confusion about integrating new optimizers into the system and the errors when setting `args.optim` to a value not supported natively by the Transformers library. They later noticed that torchdistx support was added, which includes an optimizer not listed in Transformers' native options, as indicated by a [commit on GitHub](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78).

- **Implementing a New Optimizer Solution**: Despite initial challenges with adding new optimizers, `@yamashi` indicated they found a workaround, which they labeled as "nasty," but could be committed the following day, piquing `@nanobitz` curiosity for the code sharing.



**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1104757954588196865/1111279858136383509/1208306657143169034): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [LoBaSS: Gauging Learnability in Supervised Fine-tuning Data](https://arxiv.org/abs/2310.13008): Supervised Fine-Tuning (SFT) serves as a crucial phase in aligning Large Language Models (LLMs) to specific task prerequisites. The selection of fine-tuning data profoundly influences the model&#39;s ...
- [fdsp config dict fix, todo list, add torchdistx support · OpenAccess-AI-Collective/axolotl@ad2b48c](https://github.com/OpenAccess-AI-Collective/axolotl/commit/ad2b48c0fa61ff55a40279a360d491ebc78c024f#diff-e1c112cb1e8421b1876c8653c1573d4f16d22b9fe28b889890d1e13ef333b36fR78): no description found
- [GitHub - uclaml/SPIN: The official implementation of Self-Play Fine-Tuning (SPIN)](https://github.com/uclaml/SPIN?tab=readme-ov-file): The official implementation of Self-Play Fine-Tuning (SPIN) - uclaml/SPIN

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1208027144286904403) (8 messages🔥): 

- **Mistral's Dream Team**: `dreamgen` mentioned that **Mistral** utilizes what's known as `dreamgen` in its architecture.

- **Mergekit Mayhem**: `philipmay` inquired about the possibility of using **Mergekit** to combine four **Llama 13b models** into a **MoE model** for the base of further finetuning, seeking input from others with experience in such a task.

- **Overfitting Overthrown?**: `noobmaster29` asked if increasing the **rank** of a model can prevent **overfitting**. The question is left open for discussion in the community.

- **Learning Rate Confusion Cleared**: `nafnlaus00` questioned if the **learning rate (LR)** in training applies per training sample or per generated token, which `yamashi` clarified is applied by **batch**.

- **Sample Size Significance**: `nafnlaus00` followed up to speculate that larger training samples could mean fewer samples per batch, thus possibly exerting a greater influence on the training, especially when **sample packing** is enabled.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1208089543291568148) (8 messages🔥): 

- **Dataset Dilemmas and Bean Diets**: `@iatetoomanybeans` expressed frustration with managing and working with datasets, while humorously noting an unsuccessful attempt to reduce bean consumption after a quip from `@c.gato`.
- **Uniform System Message Curiosity**: `@le_mess` inquired about the uniformity of system messages across 100k dataset rows, with `@xzuyn` confirming that the entire dataset indeed shares the same system prompt.
- **Exploring Neural Novel Database**: `@lee0099` shared a link to the [Neural-DPO](https://huggingface.co/datasets/NeuralNovel/Neural-DPO) on Huggingface, showcasing an A.I assistant's varying responses to inquiries about the Aya initiative and a parameter-efficient expert Ai(x) formula.

**Links mentioned**:

[NeuralNovel/Neural-DPO · Datasets at Hugging Face](https://huggingface.co/datasets/NeuralNovel/Neural-DPO): no description found

  

---


### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1208905837276434433) (4 messages): 

- **Confusion Over Mysterious '75 Iterations' in DPO**: User `@filippob82` is puzzled by the appearance of a fixed 75 iterations in the DPO evaluation phase and is unsure of its origin or purpose.
- **DPO Evaluation Might Be Off the Table**: `@noobmaster29` suggests that evaluation may not currently work with DPO, based on their experience from two weeks prior.
- **Persistent Evaluation Troubles with DPO**: Despite attempts, `@noobmaster29` has not been able to successfully run an evaluation in DPO, hinting at potential unresolved issues with the feature.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/) (1 messages): 

c.gato: User Error.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/) (1 messages): 

j_sp_r: https://venki.dev/notes/replicate-vs-fly
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1207985062113579028) (52 messages🔥): 

- **Fine-Tuning Frustration**: `@david1542` seeks guidance on fine-tuning LLMs for domain-specific tasks like sales, highlighting that his agent lacks understanding of the company's detailed processes and context. No specific papers or blog posts were referenced or discussed, leaving `@david1542` in search of insights.
- **Pricing Puzzles**: `@pasko70` presents a pricing issue with LangSmith, where trace costs exceed the cost of LLM invocations, making the service financially impracticle for low to medium token throughput applications. A detailed cost-breakdown shows a discrepancy between Chain Cost and Trace Cost, but no solutions or responses are given.
- **Vector DB Challenges**: `@cablecutter` inquires about processing Whisper transcripts into vector databases for hierarchical summaries and QA, with operations focus on theme extraction and summarization. They face challenges integrating small, short-context segments and their interactions.
- **YouTube & Twitter Highlights**: `@jasonzhou1993` and `@davidzhou8571` share links to a YouTube video discussing OpenAI's Agent 2.0, and a Twitter status about a user creating Multimodal RAG with local LLM discussion documents using Langchain, PrivateGPT, and Ollama tools.
- **Multiple Issues with `langchain_community` Module Updates**: Users `@rajvir3` and `@rajib2189` reported issues with LangChain updates causing errors, particularly with the `TextLoader` module throwing a `ModuleNotFoundError` for 'pwd'. `@dre99899` suggests working fixes include downgrading or manually editing files based on GitHub updates.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1038097195422978059/1208301752605220864): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Multi-LLM AI Gateway to run, secure, and govern AI traffic](https://konghq.com/products/kong-ai-gateway): no description found
- [Tweet from zhouql1978 (@zhouql1978)](https://x.com/zhouql1978/status/1758419213319094592?s=20): Multimodal RAG: Based on Langchain @hwchase17  and PrivateGPT @ivanmartit, I build Multimodal RAG with less than 300 lines of code. You can talk to any documents with local LLM @ollama including Word,...
- [langchain/libs/community/langchain_community/document_loaders/pebblo.py at d7c26c89b2d4f5ff676ba7c3ad4f9075d50a8ab7 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/d7c26c89b2d4f5ff676ba7c3ad4f9075d50a8ab7/libs/community/langchain_community/document_loaders/pebblo.py#L261C8-L262C23): 🦜🔗 Build context-aware reasoning applications. Contribute to langchain-ai/langchain development by creating an account on GitHub.
- [Latest langchain_community is giving an error &quot;No MODULE PWD&quot; while using TEXTLOADER · Issue #17585 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/17585): Checked other resources I added a very descriptive title to this issue. I searched the LangChain documentation with the integrated search. I used the GitHub search to find a similar question and di...
- [LangSmith For Beginners | Must know LLM Evaluation Platform 🔥](https://youtu.be/FgG-trkAMwU): In this video, I will show you how to integrate langsmith into your existing langchain project. LangSmith is a unified DevOps platform for developing, collab...
- [OpenAI&#39;s Agent 2.0: Excited or Scared?](https://youtu.be/JfM1mr2bCuk?si=xOSeTo74JuRZ-TZx): I want to give you a full run down of browser/mobile/desktop AI agentsGet free HubSpot E-book: Using Generative AI to scale your content operation: https://c...
- [GitHub - 13331112522/m-rag: Build your own Multimodal RAG Application using less than 300 lines of code.](https://t.co/4qJrES25Ak): Build your own Multimodal RAG Application using less than 300 lines of code. - 13331112522/m-rag
- [GitHub - Abraxas-365/langchain-rust: LangChain for Rust, the easiest way to write LLM-based programs in Rust](https://github.com/Abraxas-365/langchain-rust): LangChain for Rust, the easiest way to write LLM-based programs in Rust - Abraxas-365/langchain-rust

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1208080534291488822) (1 messages): 

- **Request for RAG API Implementation Guidance**: User `@mamo7410` inquired about implementing a RAG API using **langserv**. They are seeking assistance on how to obtain **streaming**, **runtime ID**, and **context documents** for the frontend, mentioning that `stream_log` might be the solution but the response it yields is complex and no examples are readily found online.
  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/) (1 messages): 

tumultuous_amicable: wow u def don't want to put your api key in a discord channel
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1208031572658495488) (7 messages): 

- **Multimodal RAG Branches Out**: `@davidzhou8571` shared [a Twitter post](https://x.com/zhouql1978/status/1758419213319094592?s=20) by `@zhouql1978` which boasts the creation of a **Multimodal RAG**, combining the features of Langchain and PrivateGPT to chat with documents across various formats, achieved in less than 300 lines of code.
  
- **Seeking Feedback on Scribe**: `@shving90` is requesting feedback on a project called Scribe, showcased at [scribe.oranai.com](https://scribe.oranai.com/), but no further details have been provided in the message.

- **Memory for All via Plastic Labs**: `@courtlandleer` introduced an open-source reimplementation of OpenAI's 'memory' features using Honcho through Plastic Labs, with a [demo and discord bot](https://x.com/vintrotweets/status/1758274129768443946?s=20) available for testing as detailed in their [blog post](https://blog.plasticlabs.ai/blog/Memories-for-All).

- **Deep Dive into Whisper**: `@amgadoz` wrote an in-depth three-part blog series about OpenAI's Whisper for ASR, covering architecture, the model's multitask interface, and the development process, available on Substack: [post 1](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link), [post 2](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link), and [post 3](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link).

- **LangChain Now in Rust**: `@edartru.` shared a link to [GitHub](https://github.com/Abraxas-365/langchain-rust) where they posted a repository for LangChain in Rust, making it simpler to write LLM-based programs in this programming language.

- **AI-Powered Financial Analyst Tutorial**: `@solo78` presented a tutorial that demonstrates how to use the OpenAI Assistant API to build a financial analyst, focused on analyzing the risk profile of insurance companies, detailed in a [Medium article](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e).

**Links mentioned**:

- [Tweet from zhouql1978 (@zhouql1978)](https://x.com/zhouql1978/status/1758419213319094592?s=20): Multimodal RAG: Based on Langchain @hwchase17  and PrivateGPT @ivanmartit, I build Multimodal RAG with less than 300 lines of code. You can talk to any documents with local LLM @ollama including Word,...
- [OranScribe](https://scribe.oranai.com/): The centralized writing platform to ideate, research, write, and edit your cross platform content with AI. Write better, faster, and more efficiently.
- [Using AI to Analyze Risk Profile of an Insurance Company: A Comprehensive Guide](https://medium.com/@bsouleymane78/using-ai-to-analyze-risk-profile-of-an-insurance-company-a-comprehensive-guide-d17d25e2524e): A Use Case of OpenAI Assistant API for financial analysis of an European Insurance Companies.
- [Tweet from vintro (@vintrotweets)](https://x.com/vintrotweets/status/1758274129768443946?s=20): OpenAI shipped personalized memory on Tuesday!  seems like it&#39;s just deriving facts based on the messages you send  so we put together a bot w/ @LangChainAI of how to recreate using Honcho  oh and...
- [Memories for All](https://blog.plasticlabs.ai/blog/Memories-for-All): TL;DR § Personalization is the next frontier. OpenAI gets it: We’re testing ChatGPT&amp;#039;s ability to remember things you discuss to make future chats more helpful.
- [Decoding Whisper: An In-Depth Look at its Architecture and Transcription Process](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr-46b?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): Part 2 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [Exploring Whisper&#x27;s Multitask Interface: A Closer Look at its Speech Transcription and Translation Capabilities](https://amgadhasan.substack.com/p/exploring-whispers-multitask-interface?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): Part 3 of a multi-part series in which we delve deep into Whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [The Making of Whisper: An In-Depth Exploration of its Training Data and Process](https://amgadhasan.substack.com/p/whisper-how-to-create-robust-asr?utm_source=substack&utm_content=feed%3Arecommended%3Acopy_link): A Multi-part series in which we delve deep into whisper, OpenAI&#x27;s state-of-the-art automatic speech recognition model
- [GitHub - Abraxas-365/langchain-rust: LangChain for Rust, the easiest way to write LLM-based programs in Rust](https://github.com/Abraxas-365/langchain-rust): LangChain for Rust, the easiest way to write LLM-based programs in Rust - Abraxas-365/langchain-rust

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1208367940391534622) (6 messages): 

- **ChainLit Tutorial Walkthrough**: A [YouTube video](https://youtu.be/FZrkm0vaYYQ) was shared by `@datasciencebasics` titled "Chat With Websites Using ChainLit / Streamlit, LangChain, Ollama & Mistral 🧠" which demonstrates the creation of a simple Retrieval Augmented Generation UI locally.
- **Vector Database Context Inquiry**: `@tumultuous_amicable` inquired about using a vector database to provide context for models, as opposed to explicitly passing in text context.
- **crewAI Stock Data Tutorial**: `@business24.ai` shared a [YouTube video](https://youtu.be/Q5GUFCpEng4) about adding live stock data to crewAI using LangChain custom tools and storing results in Obsidian, and asked for feedback to improve future videos.
- **Web Browsing Agent with LangGraph**: The [YouTube video](https://www.youtube.com/watch?v=gbGYN3YyTS4) shared by `@pradeep1148` showcases "Web Voyager," a vision-enabled web-browsing agent that controls mouse and keyboard actions.
- **LangSmith LLM Evaluation Platform Introduction**: `@datasciencebasics` posted a [YouTube video](https://youtu.be/FgG-trkAMwU) that serves as a beginner's guide to integrating LangSmith into existing LangChain projects for LLM development and collaboration.

**Links mentioned**:

- [Web Browsing Agent using LangGraph](https://www.youtube.com/watch?v=gbGYN3YyTS4): Web VoyagerWebVoyager by He, et. al., is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.It works by viewing annotated brow...
- [Add Live Stock Data to crewAI using LangChain custom tools and store results in Obsidian](https://youtu.be/Q5GUFCpEng4): In this Tutorial, we add three custom tools to crewAI. With the first custom tool, we connect crewAI to our portfolio and get the current positions. With the...
- [Chat With Websites Using ChainLit / Streamlit, LangChain, Ollama &amp; Mistral 🧠](https://youtu.be/FZrkm0vaYYQ): In this video, I am demonstrating how you can create a simple Retrieval Augmented Generation UI locally in your computer. You can follow along with me by clo...
- [LangSmith For Beginners | Must know LLM Evaluation Platform 🔥](https://youtu.be/FgG-trkAMwU): In this video, I will show you how to integrate langsmith into your existing langchain project. LangSmith is a unified DevOps platform for developing, collab...

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1208033306092900382) (8 messages🔥): 

- **Temporary Spike in Training Loss**: `@bjoernp` pointed out that a high loss during training could just be an **outlier** in the training data, possibly from a single batch.
- **Outliers Might Disturb Training**: `@_jp1_` suggested to `@philipmay` that high training loss could be due to **"bad data"** and recommended checking the data manually, especially when **batch size** isn't too high.
- **Philip Shares Training Config**: `@philipmay` shared his training configuration details, including **micro_batch_size** of 16 and a **learning_rate** of 0.0002, among other specs.
- **Experimentation with Base Models and Expert Configurations**: `@philipmay` described success with using **VAGOsolutions/SauerkrautLM-13b-v1** as a base model, combining it with experts from **LeoLM**, noting comparable training and evaluation loss to **mixtral**, and mentioning issues encountered when attempting to use **LeoLM** as the base due to missing files.
- **Discussion on Pretraining Framework for Language Models**: `@remek1972` sought advice for frameworks to pretrain Language Models from scratch, indicating a goal to create a **national LLM model** without English content, to which `@philipmay` responded understanding the purpose after an explanation.
  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1208490102725283891) (1 messages): 

- **Zero Score Mystery Solved**: `@huunguyen` shared an amusing *oops* moment, revealing their model tallied a **score of 0 on gsm8k** because the prompt output "### Response" was mistaken for the math problem's answer. They noted, with a hint of humor, that pretraining on the gsm8k dataset might have been a good idea.
  

---


### DiscoResearch ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/1208062982605639760) (4 messages): 

- **JinaAI Introduces jina-colbert-v1-en**: `@devnull0` shared a [tweet from JinaAI](https://fxtwitter.com/JinaAI_/status/1758503072999907825?t=1LT1ISg6BCXYdcr0yYEhHw&s=19) announcing **jina-colbert-v1-en**, which improves upon ColBERTv2's zero-shot performance and is available on [Hugging Face under the Apache 2.0 license](https://huggingface.co/jinaai/jina-colbert-v1-en).

- **How to choose an embedding DB**: `@huunguyen` mentioned past experiences with using SQLite and Whoosh for embedding and indicated that Elasticsearch is a more enterprise-focused solution that is non-trivial to set up.

- **Alternatives for advanced search**: `@huunguyen` also brought up Lucene/Solr as alternatives for enterprise search solutions and invited more direct messaging for quicker responses, as regular monitoring of the Discord isn't common practice.

**Links mentioned**:

[Tweet from Jina AI (@JinaAI_)](https://fxtwitter.com/JinaAI_/status/1758503072999907825?t=1LT1ISg6BCXYdcr0yYEhHw&s=19): Introducing jina-colbert-v1-en. It takes late interactions & token-level embeddings of ColBERTv2 and has better zero-shot performance on many tasks (in and out-of-domain). Now on @huggingface under Ap...

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1208070420004208660) (3 messages): 

- **In Search of a German Evaluation Tutorial**: `@thomasrenkert` inquired about resources to build **evaluation datasets** for German, specifically for translation and summarization tasks.
- **Guidance Offered on Evaluation Datasets**: `@bjoernp` mentioned the lack of specific tutorials but offered to provide guidance on the topic, suggesting the use of **lm-evaluation-harness** for evaluation metrics like **chrf**, **bleu**, and **rouge score**.
- **Exploring lm-evaluation-harness for Solutions**: Following `@bjoernp`'s advice, `@thomasrenkert` looked into **lm-evaluation-harness examples** as a potential resource for his needs.
  

---



### Skunkworks AI ▷ #[compute](https://discord.com/channels/1131084849432768614/1131302399370334248/) (1 messages): 

bluetyson: that is interesting - kickstarter for crunching?
  

---


### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1208029743614664754) (5 messages): 

- **Function Calling with Llama Factory**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=EYR_kd3X03M) titled "Finetune model for Function Calling (Tool Call) with Llama Factory," which discusses finetuning a model for function calling in Python programming.
- **Tencent's YOLO-World Introduction**: Another [video link](https://www.youtube.com/watch?v=yaqi8xRUsp4) provided by `@pradeep1148` features "YOLO-World: Real-Time, Zero-Shot Object Detection" by Tencent’s AI Lab, showcasing a new object detection model.
- **OpenAI SORA Text to Video Model**: A [YouTube video](https://www.youtube.com/watch?v=7lsOzA3WhSI) titled "OpenAI SORA Text to Video model and Technical Report" was shared by `@pradeep1148`, introducing Sora, a model that generates videos from text prompts.
- **Web Browsing Agent - LangGraph**: `@pradeep1148` posted a [link to a video](https://www.youtube.com/watch?v=gbGYN3YyTS4) about WebVoyager, a web-browsing agent that uses vision to control a mouse and keyboard.
- **Acknowledgement of Shared Content**: `@sabertoaster` responded with a simple "nice" to the content shared by `@pradeep1148`, acknowledging the videos and AI models discussed.

**Links mentioned**:

- [Finetune model for Function Calling (Tool Call) with Llama Factory](https://www.youtube.com/watch?v=EYR_kd3X03M): Finetune model for Function Calling (Tool Call) with Llama Factory#llm #ml #ai #largelanguagemodels #deeplearning #python #pythonprogramming https://github.c...
- [Web Browsing Agent using LangGraph](https://www.youtube.com/watch?v=gbGYN3YyTS4): Web VoyagerWebVoyager by He, et. al., is a vision-enabled web-browsing agent capable of controlling the mouse and keyboard.It works by viewing annotated brow...
- [YOLO-World: Real-Time, Zero-Shot Object Detection](https://www.youtube.com/watch?v=yaqi8xRUsp4): On January 31st, 2024, Tencent’s AI Lab released YOLO-World (access code on Github), a real-time, open-vocabulary object detection model. YOLO-World is a zer...
- [OpenAI SORA Text to Video model and Technical Report](https://www.youtube.com/watch?v=7lsOzA3WhSI): Introducing Sora, our text-to-video model. Sora can generate videos up to a minute long while maintaining visual quality and adherence to the user’s prompt.#...

  

---


### Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/1209014891307073556) (2 messages): 

- **Exploring Alternatives to PPO in RLHF**: `@nagaraj_arvind` shared a video and paper on **RLHF (Reinforcement Learning from Human Feedback)** and introduced the **DPO (Direct Preference Optimization)** algorithm as a promising alternative to **OpenAI's PPO**. The [lecture video](https://youtu.be/Ju-pFJNfOfY) covers the basics of RLHF, PPO, and DPO for large language models, while the [DPO paper](https://arxiv.org/abs/2305.18290) suggests a new parameterization to improve the alignment of models with human preferences.
- **Curiosity About KTO**: `@salmon_lemon` asked about KTO, but no further details or responses were provided in the given chat history.

**Links mentioned**:

- [RLHF, PPO and DPO for Large language models](https://youtu.be/Ju-pFJNfOfY): Introduction to Reinforcement Learning, RLHF, Proximal policy optimization (PPO) and Direct preference optimization (DPO) algorithms.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290): While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised ...

  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1208800226660450374) (1 messages): 

Since there is only one message provided without additional context or replies, and since there are no explicit links or further points of discussion mentioned in the message, the summary would be:

- **Inquiring About Llama-index for Local RAG**: User `@damiondreggs` asked if **llama-index** is still a viable tool for local Retrieval-Augmented Generation (RAG), or if there's a better tool available. No further context or additional information was provided.
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1208787634676179004) (3 messages): 

- **Curiosity About OpenSora**: User `@cryptossssun` expressed interest in **openSora**, asking if others are also interested.
- **Reverse Engineering Sora's Secrets**: `@rusch` showed interest in **OpenSora**, specifically in reverse engineering some of Sora's capabilities.
  

---



### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1208166697337356289) (1 messages): 

- **AIEF Project Suggestion by swyxio**: User `@swyxio` shared a potential project for the **AI Engineer Foundation** by posting a [link from Hacker News](https://news.ycombinator.com/item?id=39371297), specifically flagging `@296887155819675650` and `@705561973571452938` for attention.

**Links mentioned**:

[no title found](https://news.ycombinator.com/item?id=39371297): no description found

  

---


### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1208159139629367448) (2 messages): 

- **Generative AI Workshop in NYC**: `@tanyarai` announced an upcoming **NYC Developer Generative AI Workshop** on 2/26, inviting participants to learn from industry experts and engage in a hands-on interactive workshop using models from OpenAI, Google, and Anthropic. [RSVP here](https://lu.ma/ai_workshop), and don't forget your laptop!

- **AI Hackathon for OSS Tooling and Model Enthusiasts**: `@hackgoofer` shared an invite to an AI Engineer Foundation-hosted Hackathon on OSS tooling and model set for the next Saturday, featuring sponsors Fireworks.ai & LlamaIndex.ai and cash prizes. Check out the details and [get on the list](https://partiful.com/e/e3arTNNboImbIKdgQjHs) to participate in this coding showdown.

**Links mentioned**:

- [Generative AI Developer Workshop · Luma](https://lu.ma/ai_workshop): 👋 NYC Developers! Join us in Flatiron on 2/26 for an evening focused on building with generative AI! We will kick-off with lightning talks from industry experts and then go into an...
- [RSVP to OSS Hackathon: Functional Calling + RAG Hackathon | Partiful](https://partiful.com/e/e3arTNNboImbIKdgQjHs?): Hi fellow lovely hackers,  The AI Engineer Foundation (Your Friendly Open Source Nonprofit Neighbor - website: aie.foundation) is hosting a Function Calling + RAG hackathon.  We are excited to announc...

  

---



---




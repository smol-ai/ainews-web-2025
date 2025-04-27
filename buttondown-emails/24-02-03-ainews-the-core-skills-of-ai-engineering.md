---
id: c7621e73-b07b-43ec-b2b9-6458dcb3bf6c
title: The Core Skills of AI Engineering
date: '2024-02-04T00:54:29.799988Z'
original_slug: ainews-the-core-skills-of-ai-engineering
description: >-
  **AI Discords** analysis for **2/2/2024** covered **21 guilds**, **312
  channels**, and **4782 messages**, saving an estimated **382 minutes** of
  reading time. A major discussion led by **Eugene Yan** focused on **AI
  Engineering**, highlighting the overlap and challenges between software
  engineering and data science skills in AI development. The **TheBloke
  Discord** featured talks on **large language models** like **MiquMaid** and
  **OLMo**, including **open-source LLM** licensing under **Apache 2.0** for a
  **65B** parameter model, and technical topics such as **Automatic Weight
  Quantization (AWQ)**, **LoRA fine-tuning**, and **quantization** techniques
  like **LoftQ**. Discussions also touched on **AI security** with watermarking,
  **superconductors** and **carbon nanotubes** in hardware, and deployment of
  LLMs via **Hugging Face spaces**. The **LAION Discord** focused on challenges
  in **model distillation** with **SSD-1B**, data quality improvements using
  captioned datasets like **BLIP**, **COCO**, and **LLaVA**, and debates on
  **tokenization** methods for better prompt adherence in image generation.
companies:
  - ai2
  - huggingface
  - laion
  - aliexpress
models:
  - miqumaid
  - olmo
  - awq
  - exl2
  - mistral-medium
  - internlm
  - ssd-1b
  - lora
topics:
  - ai-engineering
  - software-engineering
  - data-science
  - open-source
  - large-language-models
  - automatic-weight-quantization
  - quantization
  - fine-tuning
  - model-distillation
  - dataset-quality
  - tokenization
  - prompt-adherence
  - image-generation
  - ai-security
  - superconductors
  - carbon-nanotubes
  - deployment
  - hf-spaces
people:
  - eugene_yan
---


<!-- buttondown-editor-mode: plaintext -->> AI Discords for 2/2/2024. We checked **21** guilds, **312** channels, and **4782** messages for you. Estimated reading time saved (at 200wpm): **382 minutes**.

---

We really tried to avoid featuring Latent Space twice in a row, but Eugene Yan [kicked off a discussion on AI Engineering](https://twitter.com/eugeneyan/status/1753445305545298314):

 ![image.png](https://assets.buttondown.email/images/910a7ce8-aa5c-4c16-90df-fae68b16489f.png?w=960&fit=max) 

Which [resulted in](https://discord.com/channels/822583790773862470/1075282825051385876/1203011101193801728) the longest ever thread on the topic:

 ![image.png](https://assets.buttondown.email/images/7380228a-1525-44e3-9934-fc3668270325.png?w=960&fit=max) 


The central confusion is the high degree of overlap between what are traditionally software engineer skills and data scientist skills, but also what software engineers struggle with when dealing with probabilistic, data-driven systems. Do they need to be reading papers? Do they need to write CUDA kernels?

Some mental models were created:

 ![image.png](https://assets.buttondown.email/images/cd08308e-70d9-425c-b611-f679113092f1.png?w=960&fit=max) 

as well as a progression path for skill development:

 ![image.png](https://assets.buttondown.email/images/13b6c2fe-a667-4ec3-bb64-2b51dffb03d1.png?w=960&fit=max) 


**Table of Contents**

[TOC] 


# PART 1: High level Discord summaries




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord Summary

- **Late-Night Tech Talk**: Users including `@mrdragonfox`, `@coffeevampir3`, and `@potatooff` engaged in a vivid discussion about the performance of large language models like **MiquMaid** and **OLMo**, and the potential applications of 3D printing for PC hardware alongside the usage of carbon nanotubes.
- **Watermarks and AI Security**: Conversation included techniques for using gradient ascent to make models unlearn information and the challenges around removing deep watermarking from models during training.
- **Open Licensing for OLMo**: AI2's **OLMo GitHub repository** was introduced, noteworthy for its **open-source LLM** availability under the **Apache 2.0 license**, with a 65B model's training mentioned.
- **Superconductors and Nanotubes on AliExpress**: Superconductor materials like **Yttrium barium copper oxide (YBCO)** and carbon nanotubes were topics of interest, highlighting their availability on **AliExpress**.

- **Aphrodite's Capabilities and Limitations**: The **Aphrodite** model was credited for its batching capabilities in AI horde but was pointed out as incompatible with GPUs of differing VRAM sizes.
- **Calibration Dataset Diversity for AWQ**: Discussions around the best calibration datasets for **Automatic Weight Quantization (AWQ)** outlined the importance of diversity in datasets, particularly for AI models like **EXL2**.
- **Local AI and Role-Playing Practices**: Usability of various **AI models for role-playing** was discussed, noting the preference for instruction mode when using instruction-tuned models.
- **Leaderboards and Ethical Model Usage**: The presence of the **Mistral medium model (MoMo)** on leaderboards sparked a debate on the implications of using models with unclear licensing and the lack of corporate transparency in model training.

- **Quantization and Fine-Tuning**: Questions arose about fine-tuning a pre-trained **AWQ model** with **LoRA**, and the benefits of aligning the quantization process during **QLoRA fine-tuning** and serving were debated.
- **LoftQ Introduction and Quantization Discussion**: The linking of a paper on **LoftQ**, a quantization technique that fine-tunes **LoRA and quantizes** a model to improve performance, led to discussions about its effectiveness.

- **internLM Gets a Nod**: In a brief exchange, `kquant` recommended **internLM** as a solid model.

- **Deploying LLMs with HF Tools**: `m.0861` sought advice on the deployment of Large Language Models through **HF spaces**, followed by reflections on the advantage of using HF's inference endpoints for LLM deployment.



---



## [LAION](https://discord.com/channels/823813159592001537) Discord Summary

- **Distillation Downfalls in SSD-1B**: Members `@pseudoterminalx` and `@gothosfolly` discussed the rigidity of **SSD-1B** due to its distillation from a singular fine-tuned model, suggesting that using multiple models could enhance aesthetic aspects of distillation.

- **Optimizing Data Quality with Proper Captioning**: The use of well-captioned images from diverse sources such as BLIP, COCO, and LLaVA was highlighted in a strategy to improve prompt adherence in model training, with mentions of input perturbations and data pipeline refinements for efficacy.

- **Prompt Adherence Through Hybrid Encoding**: A debate surrounded the merits of UTF-8 tokenization versus a hybrid approach amalgamating UTF-8 codes into single tokens, pondering on the potential benefits for image generation by adopting a byte-level encoding similar to **ByT5**.

- **Cropping and Upscaling in Image Generation**: A effective methodology for image-to-image upscaling using cropped model weights was identified, which is credited with preserving scene integrity, especially beneficial for higher resolution enhancements.

- **The Peer Review Bypass**: Discussions underscored a trend towards researchers releasing notable findings on blogs rather than in traditional journals, often due to the cumbersome peer review process, with some considering detailing novel architectures exclusively through blog posts.



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord Summary

- **The Evolution of AI Engineering**: Engaging debates unfolded regarding the essential skills for software engineers to effectively use LLMs and the evolving job role of AI engineers. Discussion highlighted the importance of understanding the probabilistic nature of LLMs, evaluation, debugging, data familiarity, and a mindset shift from deterministic to probabilistic outcomes. The concept of an **AI Engineer Continuum** developed, proposing stages from using APIs to fine-tuning models.

- **Community Growth and Learning Initiatives**: In the LLM Paper Club (East), attendees engaged in technical discussions, such as the methodology of self-rewarding LLMs, improving text embeddings, and the value of retrieving long-tail knowledge for RAG. Suggestions for forming a "code club" to collaboratively walk through code and a "production club" to examine the actual implementation of code/papers reflect the technical-oriented community's learning desires.

- **AI Events and Gatherings Gain Popularity**: Calls for participation in local and online events like the LLM Paper Club (East) and **AI in Action** meetings were made. Enthusiasm was shown for forming local user groups, demonstrated by the proposal of an LA meetup and various social learning events, underlining the proactive approach of community members in sharing knowledge and best practices.

- **Resource Sharing Enriches the Guild**: Members contributed a wealth of resources, ranging from practical guides on using AI, evaluating LLMs, and instructional content for constructing LLMs, to discussions on AI startup strategies and AI in business pitches. This indicates a strong interest in the application of AI technology within the professional and entrepreneurial spaces.

- **Concerns Over Tools Reliance**: Skepticism about OpenAI's Fine-Tuning API/SDK was raised, along with cautions against potential platform lock-in. The discussions leaned towards the benefits of full-scale fine-tuning over simpler API interactions, surfacing concerns relevant to engineers wary of over-reliance on third-party platforms.



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **Elevating Open Models to Large-Scale Science Projects**: `@layl` underlined the growing feasibility of securing EU government support for open model training, resonating with the notion of open models as large-scale science ventures. `@stellaathena` corroborated a shift from negligible to minimal advancements in this area, suggesting prospects for deploying `@layl`'s ML library in High-Performance Computing (HPC) settings such as LUMI.
  
- **Activation Function Efficacy in the Spotlight**: A comprehensive debate on activation functions such as GeLU, ReLU, Mish, and TanhExp was spurred by users including `@xa9ax` and `@fern.bear`, which drew attention to the dearth of extensive empirical tests for these functions in large model training. Despite earlier doubts by `@ad8e` regarding the probity of a paper promoting Mish, `@xa9ax` verified the inclusion of all pre-submission experiments in the final publication.

- **Benchmarking Model Architectures**: Conversations delved into comparisons between `@state-spaces/mamba` models and other architectures like Transformers++ and Pythia, with users like `@ldj` expressing concerns about the basis of comparison and `@stellaathena` highlighting the need for a uniform model suite trained on open data for fair evaluations.

- **Intricacies of Activation Functions Explored**: Users `@catboy_slim_`, `@fern.bear`, and `@nostalgiahurts` pondered over the nuanced influence of activation function choices, discussing how the scale of these functions interacts with other hyperparameters to influence model performance. Empirical findings from EleutherAI's blog and various academic papers were dissected to decode complex interdependencies between activation functions and training dynamics of models.

- **Legal Complexities Shadowing Large Model Training**: `@synquid` highlighted the legal intricacies surrounding transparency in model training data sources, noting how the overt disclosure of training data might lead to intellectual property litigations that could stifle scientific progress.
  
- **Demystifying Knowledge Distillation**: Inquisitive discussions by `@johnryan465` and `@xa9ax` revolved around the efficiency benefits of training a smaller-sized model B to emulate a larger-sized model's A logits over direct training of model A - pondering the *infinity ngrams paper* methodology to generate cost-effective models for potential distillation pipelines.

- **MCTS Sampling Challenges Addressed**: `@blagdad` scrutinized the exploration conundrums in Monte Carlo Tree Search (MCTS), alluding to the utilization of Upper Confidence bounds for Trees (UCT) for guiding the exploration based on uncertainty as opposed to uniform branching.

- **Fine-Tuning Efficiency via Exploration**: The finesse of fine-tuning using efficient exploration was articulated, with a focus on agents that craft queries and a reward model that functions on the feedback received. The discussion encompassed the merits of double Thompson sampling and the employment of epistemic neural networks, detailed in an [arXiv paper](http://arxiv.org/abs/2402.00396).

- **Bayesian Active Learning Awaits Unveiling**: `@fedorovist` indicated the imminent release of a Bayesian active learning implementation by `@322967286606725126`, spurring interest from `@johnryan465` due to past experiences with akin challenges.

- **Probing Adam Optimizer Variations**: A query by `@ai_waifu` examined whether any studies have ventured into modifying the Adam optimizer to utilize variance of parameters rather than the gradient's second moment estimation. However, specifics about such research in response were not highlighted.

- **Collaborating for Vision-Language Model Integration**: Intention to assimilate vision and language support into lm-harness was voiced by `@asuglia`, with `@chrisociepa` and `@1072629185346019358` cited as potential collaborators by `@hailey_schoelkopf`, who also suggested community contributions.

- **Standard Error Conversation in MMLU Results**: `@baber_` inquired about substantial standard errors in the MMLU results for the model *miqu*, and `@hailey_schoelkopf` recognized a possible need for recalibrating standard error computations within the evaluation code.

- **Facilitating Zero-Shot Evaluation**: For forcing a task to run in zero-shot mode in lm-harness, `@asuglia` was directed by `@hailey_schoelkopf` to set `num_fewshot: 0`, referencing the pertinent [source code](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169).

- **Upgrading Grouped Task Evaluation Methodology**: A suggested update to the standard error aggregation method across grouped tasks was put forth by `@hailey_schoelkopf`, with a [PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390) to the repository indicating a move to a Pooled variance-based calculation.

- **Synchronizing Vision-Language Model Contributions**: A cooperative fork for a functioning vision-language pipeline was offered by `@jbdel.`, with an arrangement to transition the work to `@asuglia` post-Feb 15th. Coordination is to be organized with a [scheduling poll](https://www.when2meet.com/?23484385-k3FqO).



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **LLaVA-1.6 Surpasses Gemini Pro**: A YouTube video demonstration suggests that **LLaVA-1.6**, with features like enhanced reasoning, OCR, and world knowledge, outperforms Gemini Pro on several benchmarks. Results and further details can be found on the [LLaVA blog](https://llava-vl.github.io/blog/2024-01).

- **Hugging Face Introduces MiniCPM**: A new model, MiniCPM, showcased on Hugging Face, has sparked interest due to its potential and performance, with discussions comparing it to other models like Mistral and awaiting fine-tuning results.

- **ResNet Growth Techniques Applied to LLMs**: Discussions have surfaced around the application of "growing" techniques, successful with ResNet classifiers and ProGANs, to LLMs, evidenced by Apple's Matroyshka Diffusion model. The new **Miqu** model's entry on the Open LLM Leaderboard with notable scores leads to mixed reactions.

- **Quantization's Impact on AI Model Performance**: Conversations around **miqu-70b** bring up the potential effects of quantization on model performance aspects such as spelling accuracy, provoking thoughts on whether quantized models should be standard on certain platforms.

- **The Ongoing Pursuit for Optimized Tokenization**: The engineering community has discussions around multilingual tokenizers, with a 32000-token vocabulary potentially limiting models like LLaMA/mistral. Efforts to adapt LLaMA models for specific languages, such as VinaLLaMA for Vietnamese and Alpaca for Chinese, indicate progress in model internationalization.



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord Summary

- **Questioning AI Censorship in Geopolitical Contexts**: In a discussion about potential censorship, an OpenAI user, `@bambooshoots`, questioned whether **ChatGPT** censors responses to comply with Chinese regulations. Another user, `@jeremy.o`, made it clear that OpenAI does not engage in such censorship practices.

- **Content Creation Freedoms Celebrated in AI**: `@jeremy.o` highlighted **OpenAI's DALL·E** tool, emphasizing its ability to generate diverse content, including LGBTQI+ representations, showcasing the organization's commitment to freedom of content creation.

- **ChatGPT Conversational Memory and Identity Formation**: Users like `@blckreaper`, `@darthgustav.`, and `@jaicraft` debated the challenges related to GPT models potentially remembering previous sessions or confusing past responses. There's a user desire for **GPT entities** to have separate memories and a clear division of conversation flows to enhance user experience.

- **Invisible Text-to-Speech Modifications Explored**: `@novumclassicum` asked for guidance on making text modifications for text-to-speech applications without the changes being shown to the user. The idea is for GPT to internally replace words before submission, aiming for a seamless and invisible text alteration process for end users.

- **Amplifying AI Dialogues Beyond Concise Summaries**: User `@stealth2077` expressed frustrations with **GPT’s tendency to summarize** dialogues between characters after only a few exchanges. The aspiration here is for the AI to consistently generate extended, realistic, character-driven dialogues, maintaining the play-by-play style without defaulting to summaries.



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord Summary

- **Navigating LLM Creation Complexities**: Users discussed the technical aspects of LLM creation, noting the necessity of expertise in Machine Learning, PyTorch, and other areas. Meanwhile, there's interest in utilizing LM Studio plugins, such as TTS and open interpreters, indicating a push for more integrated and interactive AI solutions.

- **Blazing New Trails with LLMs**: Community members are exploring **Moondream**, for vision-to-text transformations, expressing interest in integrating such models into LM Studio, despite current limitations. In other chat, there's excitement around **CodeLlama 70B** with an experimental preset linked for the community, and the leak of a Mistral Ai fine-tune of **Llama 70B** called **miqu** is also making waves due to its performance in coding tasks.

- **Hardware Hurdles and Optimization Discussions**: Engaging discussions centered on optimizing hardware for LLMs, covering issues like dual GPU setups and VRAM's critical role in model performance. Advice to upgrade to dual RTX 3090 GPUs for improved speed with 70b models was shared, and there's anticipation over new machine setups with P40 GPUs for better LLM functioning. When it comes to benchmarking CPUs for LM Studio, the insights suggested focusing on VRAM usage rather than core counts.

- **Docker Dilemma Drives Conda Consideration**: One user tackled problems with Docker by turning to Conda for setting up environments, highlighting the challenges sometimes faced with containerized environments, and the usefulness of environment managers in resolving them.

- **Embedding Efficiency Vs. Effectiveness**: A brief but insightful exchange on database strategies for storing word embeddings considered the tradeoff between similarity search quality and database performance. It was noted that longer embeddings may give better context for searches but could adversely affect database efficiency.



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord Summary

- **Adventure in Advanced RAG**: `@andysingal` has showcased his work on Advanced RAG, sharing a GitHub [notebook](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb) on the same, hinting at further development similar to OpenAI's interfaces.

- **LLaVA-1.6 Outshines Gemini Pro**: LLaVA-1.6 has been announced, claiming improvements in resolution, OCR, and reasoning, even surpassing Gemini Pro in some benchmarks. For more insights, visit the [LLaVA-1.6 blog post](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/).

- **Diffusers 0.26.0 Release with New Video Models**: The new **Diffusers 0.26.0** release brings two new video models, with full notes accessible [here](https://github.com/huggingface/diffusers/releases/tag/v0.26.0). An implementation error in the release code led to incorrect inference steps, contributing to initial user issues.

- **Tokenizer Pattern Visualization and Conversion**: Tokenization patterns have been visualized by `deeeps.ig` and are demonstrated in a [Kaggle notebook](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook). Additionally, a [script](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee) for converting tiktoken tokenizers to Hugging Face format was shared, although licensing concerns were mentioned.

- **AI & Law and Mamba Dissected**: An ongoing discussion on AI in the legal field is backed by a [Medium article](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34), with a presentation to follow. `@chad_in_the_house` posted about an upcoming presentation on Mamba, a sequence modeling architecture, with relevant details found in the [arXiv paper](https://arxiv.org/abs/2312.00752) and further explanation in Yannic Kilcher's [YouTube video](https://www.youtube.com/watch?v=9dSkvxS2EB0).

- **Livestock Health ML Model Call for Volunteers**: **DalensAI** is arranging a machine learning dataset to detect sickness in livestock and is in need of volunteers to contribute images and labels. This presents an opportunity to contribute to a real-world application of computer vision.

- **Donut's Dicey Performance Across Transformers Versions**: An issue was reported where the modified donut model performs differently during inference across `transformers` library versions **4.36.2** and **4.37.2**. This implies potential backward compatibility challenges to be aware of when updating dependencies.



---



## [Mistral](https://discord.com/channels/1144547040454508606) Discord Summary

- **Groq's Competitive Edge with LPU Chips**: Groq's custom hardware, designated as Local Processing Units (LPUs), was recognized for its local optimization capabilities during runtime, suggesting they may rival Nvidia H100 chips. However, Groq does not provide hosting, and inquiries about its performance highlighted limited video memory, with more details available in the [GroqNode™ Server product brief](https://groq.com/wp-content/uploads/2022/10/GroqNode%E2%84%A2-Server-GN1-B8C-Product-Brief-v1.5.pdf).

- **Curiosity Over MoMo-72B Model**: A Hugging Face model known as MoMo-72B sparked debates about model quality and its 'contaminated' leaderboard scores, with links shared for further investigation - [MoMo-72B Hugging Face Model](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO) and the associated [discussion](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO/discussions/2).

- **Light-hearted Teasing Among Peers**: A brief, playful exchange arose involving a "betweter" comment and expressions of fun jest, alongside an important clarification regarding free model access, which can be explored on Hugging Face rather than through API keys for open-source options.

- **Assistance and Clarification for Mistral Deployment**: Users provided guidance and solutions for running Mistral models on Mac, pointing towards [LMStudio](https://lmstudio.ai) for suitable downloads, with expressions of gratitude for the support.

- **Anticipation for Innovative AI Projects**: The community showcases generated excitement, from [socontextual.com](https://socontextual.com) to a YouTube demo titled "Trying LLaVA-1.6 on Colab" which highlighted LLaVA-1.6's improved reasoning and world knowledge - [YouTube Demo](https://www.youtube.com/watch?v=SEavari8xaU). Additionally, a fan fiction titled "Sapient Contraptions" inspired by Terry Pratchett was shared via Pastebin - [Sapient Contraptions on Pastebin](https://pastebin.com/dNRbi7mY), illustrating creative uses of AI LLM software for story crafting.



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord Summary

- **Base Model Basics**: Newcomer `christolito` inquired about the "base perplexity" model, prompting a response from `mares1317` with assistance and direction to further resources.
- **Perplexity App Developments**:
  - Document attachment functionality is currently unavailable in the Perplexity Android app, a feature existing in the web version.
  - Details were presented concerning Copilot's utilization of GPT-4 and Claude 2 models in offline search-facilitated modes.
- **Membership and UX Concerns**:
  - Limitations of the free version of Perplexity were compared to those found in ChatGPT.
  - Pro user `matthewtaksa` experienced delays and message duplication issues.
- **Learning and Leveraging Perplexity**:
  - `@fkx0647` reported success in uploading and interacting with documents through an API.
  - Perplexity's effectiveness in content creation was highlighted in a shared [YouTube video](https://www.youtube.com/watch?v=aphHCBSTx7Q), with preference over Google and ChatGPT.
- **API Expansion Appeal**: `@bergutman` proposed the integration of **llava-v1.6-34b** for API support, citing the high costs of using 1.6 on replicate and the lack of multimodal API options compared to GPT4-V.



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord Summary

- **SuperServer Unveiled for AI Fine-Tuning**: The community now has access to an **8x3090 SuperServer** specifically for running axolotl fine-tunes, with `@dctanner` inviting DMs for collaboration. Details on the server's capabilities can be found in `dctanner`'s announcement, [The AI SuperServer is live!](https://x.com/dctanner/status/1753013407643562401?s=20).

- **Advantages of axolotl Sample Packing and BYOD Highlighted**: `@nanobitz` emphasized the benefits of **axolotl** over **AutoTrain**, praising its "sample packing and simple yaml sharing + byod" while noting AutoTrain's *automatic model selection* as an appealing feature.

- **FFT Ambitions and Model Fine-Tuning**: `@le_mess` inquired about executing a Fast Fourier Transform (FFT) of **Mistral** on the new SuperServer, and `@dctanner` confirmed that a full finetune of Mistral 7b was in progress, with plans to test Solar 10.7b.

- **In-Depth Exchange on GPU Storage and Training Capabilities**: The technical challenges associated with storing gradients and the necessary communication bandwidth for multiple GPUs during full model finetuning were discussed by `@nafnlaus00` and `@yamashi`.

- **Experience with vLLM Update**: Version 0.3.0 of **vLLM** showed significant speed improvements for specific workloads compared to version 0.2.7, as reported by `@dreamgen`.

- **Premature Termination in Mixtral Instruct Encountered**: `@nafnlaus00` reported that **GGUF Q3_K_M** from **Mixtral Instruct** would sometimes terminate responses early, and also mentioned they were utilizing **llama.cpp** for MoE inference.

- **Launch of Math-Multiturn-100K-ShareGPT Dataset**: A new dataset, **Math-Multiturn-100K-ShareGPT**, has been made available on Hugging Face, featuring conversations designed to solve math problems. It provides up to 64 turn pairs and aims to include more complex equations in the future. [Check out the dataset here](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT).



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord Summary

- **RAGArch Simplifies RAG System Deployment**: The new **RAGArch** tool, introduced by `@HarshadSurya1c`, makes setting up a **Retrieval-Augmented Generation (RAG)** system convenient. It incorporates a [Streamlit UI](https://streamlit.io) allowing for easy component selection and one-click creation of a RAG pipeline, as shared in a [promotive tweet](https://twitter.com/llama_index/status/1753478149743284395).

- **Comprehensive Guide to Hugging Face LLMs with LlamaIndex**: `@kapa.ai` provided a guide for integrating Hugging Face pre-trained language models with LlamaIndex, complete with a [step-by-step example notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/huggingface.ipynb). Additionally, `@whitefang_jr` shared a [Colab notebook](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.ipynb) for users to employ HuggingFace StableLM on Colab with LlamaIndex.

- **Integration Options for Predicative Models with LlamaIndex**: A discussion highlighted the integration potential of LlamaIndex with predictive models’ APIs from various platforms, with guides available for each specific integration. The conversation also included information on running local models and using LlamaIndex in conjunction with or independently from LangChain, along with a mention of [Ollama](https://ollama.ai/library), an optimized local model runner.

- **Perplexity AI's Citation Technique Draws Interest**: `@tyronemichael` inquired about **Perplexity AI's** rapid and advanced citation generation mentioned in [their documentation](https://docs.perplexity.ai/discuss/65af6285e69072005b83eb05) comparing it with their own approach using **SerpAPI** and **LlamaIndex**. However, Perplexity's approach remains unclear, even after inquiries, and a [tweet discussing a Google paper](https://x.com/cto_junior/status/1710638210009706800?s=20) highlights Perplexity AI's capabilities in factual Q&A and debunking.



---



## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord Summary

- **Optimizing With NVIDIA's Finest**: User `@zippika` shared their experience with **Nvidia 4090 GPU**, discussing optimized CUDA code for RGB to grayscale conversion using `uchar3` and integer arithmetic for efficiency. `@jeremyhoward` and `@vim410`, who brings experience from NVIDIA, contributed to discussions around bitwise shifts and welcomed `@vim410` into the community.

- **Compiler Smarts on Bitwise Optimization**: During the discussions, `@apaz` brought up a point about compilers potentially replacing division with bit-shifts automatically in optimization, which was part of a broader conversation on efficiency in CUDA code.

- **Solving CUDA Memory Management Mysteries**: `@_davidgonmar` got assistance from community members like `@lancerts` and `@vim410` with a bug fix and insight into proper C++ memory management techniques in a CUDA context.

- **Numba's Need for Speed using Shared Memory**: `@stefangliga` provided help by sharing [Siboehm's article](https://siboehm.com/articles/22/CUDA-MMM) for `@mishakeyvalue`, which included optimization techniques like shared memory caching and performance enhancements in GPU matrix multiplication.

- **Catch That Missing Brace!**: `@ashpun` was assisted by `@marksaroufim` to fix a `RuntimeError` in a CUDA kernel caused by a syntax error, and they also tackled an ImportError linked to the elusive `GLIBCXX_3.4.32` version, leading to suggestions on updating Conda and setting the `LD_LIBRARY_PATH` appropriately.



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord Summary

- **LangChain Lacks in Docs, Gains in Tools**: Engineers expressed frustrations with **LangChain documentation**, finding it confusing and ironically noting the tool's inability to explain itself. Meanwhile, there's enthusiasm for community contributions like [AutoCrew](https://github.com/yanniedog/autocrew), which automates crew and task creation for CrewAI.

- **Mixing Feelings on LangChain's Viability**: While some developers ceased using **LangChain** due to rapid changes and a lack of modularity, others praise its time-saving features. However, custom modifications like adding `user_id` to `langchain_pg_collection` are queried without clear resolution.

- **Community Driven AI Educational Content**: The sharing of educational materials included a [Stanford DSP tutorial](https://www.youtube.com/watch?v=dTzL8OF_3i0) on *Demonstrate - Search - Predict* models, a [Chat UI adaptation tutorial](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5) by **@esxr_**, and insights into chatting with CSV files using LangChain and OpenAI API despite some bugs, as demonstrated in [this tutorial](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2).

- **Harnessing AI in Productivity Tools**: Innovations highlighted include **[Lutra.ai](https://lutra.ai)**, which merges AI with Google Workspace, and **[Tiny Desk AI](https://tinydesk.ai)**, offering a no-frills, free AI-powered chat app, each touting unique capabilities to enhance productivity and user experience.

- **Routing Multiple AI Agents Discussed**: The challenge of efficiently routing queries across multiple specialized agents was discussed, with inquiries about updating the `router_to_agent` function for optimal performance.



---



## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord Summary

- **MTEB Leaderboard Shines a Light on AI**: Natureplayer highlighted the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard), referencing the latest rankings and performances of language models on various tasks.

- **Feature Request: Browsing with Ease**: A feature request for a **browse channels** option was put forward by `@joshcho_`, noting the difficulty in navigating and selecting channels of interest due to the current lack of such functionality.

- **GPT-3.5 Lauded for Instruction Adherence**: Users discussed the enhanced instruction-following capabilities of **GPT-3.5**, with `@justahvee` observing its improved performance on instruction-heavy tasks, even at the cost of reasoning abilities.

- **Detailed Prompting: A Double-Edged Sword**: The guild covered the trade-off between detailed prompting and latency, with user `@res6969` noting that extended explanations result in smarter AI performance but increased latency, while `@sourya4` discussed experimenting with `gpt-4-turbo` to balance these factors.

- **Chain of Thought Prompts Lead to Brainier AI**: The conversation included insights on using Chain of Thought (CoT) prompts for asynchronous strategies, which yield intelligent responses, and the potential of reusing CoT outputs for a secondary processing step as reported by `@byronhsu` and `@res6969`.



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **Daydream Nation Joins the Chat**: User `@daydream.nation` joined the *[Alignment Lab AI ▷ #general-chat]* and mentioned the team's project going public, expressing regret for not having participated in it yet, and speculated on the intent to test human interaction on a larger scale in the context of alignment, akin to **Google's Bard**.
- **Ready to Tackle the Hard Problems**: In the *[Alignment Lab AI ▷ #looking-for-work]*, `@daydream.nation` offered expertise in **Python, Excel Data Modeling, and SQL**, combined with a background in Philosophy and an interest in addressing consciousness with AI.



---



## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord Summary

- **Infinite Craft Channels Elemental Alchemy**: An interactive game named [Infinite Craft](https://neal.fun/infinite-craft/) built on **llama2** was spotlighted by `@chrisamico`, showcasing gameplay elements such as water, fire, wind, and earth which can be combined through a drag-and-craft mechanism.
- **Game Creator Garners Praise**: `@chrisamico` further recommended games by the creator of Infinite Craft, highlighting them as clever, fun, and occasionally thought-provoking, although no specific titles or links were provided.
- **Endorsing the Endless Fun**: `@dbreunig` affirmed the excitement around Infinite Craft, calling it a great example for its category, while `@bdexter` confided about the game's addictive nature, signaling high engagement potential.



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

- **German Embedding Models Surpass Benchmarks**: [**@damian_89_**'s tweet](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19) discusses the superior performance of **jina-embeddings-v2-base-de by @JinaAI_** and **bge-m3 by @BAAIBeijing** in enterprise data tests, with **BGE** being highlighted as particularly effective.
- **Call for Quantitative Assessment**: **@devnull0** emphasizes the need to test embedding models against a suitable metric, though they do not specify which metrics to use for evaluation.
- **Guide to RAG Evaluation Released**: The [GitHub notebook](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb) provided by **@devnull0** offers a methodological guide to evaluate Retrieval-Augmented Generation (RAG) systems.
- **Blogging Deep Dive into RAG**: A [detailed blog post](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex) complements the notebook which explains how to assess the encoder and reranker components of a RAG system using LlamaIndex and a specifically tailored testing dataset.



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord Summary

- **LLaVA 1.6 Released**: .mrfoo announced the release of [LLaVA 1.6](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/), pointing to the official release notes and documentation.

- **Off-Topic AI Buzz**: Pradeep1148 shared a [YouTube video](https://www.youtube.com/watch?v=SEavari8xaU) in the off-topic channel which seems to be AI-related but lacked any context or discussion around it.


---

# PART 2: Detailed by-Channel summaries and links



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1202888108329275402) (1441 messages🔥🔥🔥): 

- **Discussing Life and Tech in the Late Hours**: Participants like `@mrdragonfox`, `@coffeevampir3`, and `@potatooff` engaged in a late-night conversation about everything from the performance of large language models like MiquMaid and OLMo to the speculative possibility of 3D printing PC hardware and carbon nanotube applications.
- **Model Watermarking Techniques**: `@turboderp_` and `@selea` discussed the idea of using gradient ascent to make models unlearn unwanted information and the concept of watermarking models during training, with claims that watermarks can be so deep within the model that finding and removing them can be nearly impossible.
- **On Track with OLMo**: `@drnicefellow` introduced the OLMo GitHub repository by AI2, highlighting its potential as a complete open-source LLM with checkpoints and the training of a 65B model in progress. It was noted that their model is under the Apache 2.0 license.
- **Academicat and Quantum**: Users discussed the capabilities of academicat on processing very long papers and touched on how quantum materials like superconductors work under particular conditions.
- **Exploring Superconductors and Nanotubes**: In the context of future technologies and materials science, `@selea`, `@rtyax`, and `@spottyluck` talked about superconductor materials like Yttrium barium copper oxide (YBCO) and carbon nanotubes, mentioning the convenience of purchasing them on platforms like AliExpress.

**Links mentioned**:

- [Miau Cat GIF - Miau Cat Meow - Discover &amp; Share GIFs](https://tenor.com/view/miau-cat-meow-gif-9406008167044251375): Click to view the GIF
- [Tkt Smart GIF - Tkt Smart - Discover &amp; Share GIFs](https://tenor.com/view/tkt-smart-gif-20642718): Click to view the GIF
- [Mixture of Experts for Clowns (at a Circus)](https://goddard.blog/posts/clown-moe/): no description found
- [Enterprise Scenarios Leaderboard - a Hugging Face Space by PatronusAI](https://huggingface.co/spaces/PatronusAI/enterprise_scenarios_leaderboard): no description found
- [mlabonne/phixtral-2x2_8 · Hugging Face](https://huggingface.co/mlabonne/phixtral-2x2_8): no description found
- [Yes Lawd GIF - Yes Lawd My Precious - Discover &amp; Share GIFs](https://tenor.com/view/yes-lawd-my-precious-gif-13073178): Click to view the GIF
- [Creepy Talking Cat 🙀](https://www.youtube.com/watch?v=ddroHMg96HA): I made this video into a full song!Watch here: youtu.be/WLryCXyjL_0
- [nVidia Hardware Transcoding Calculator for Plex Estimates](https://www.elpamsoft.com/?p=Plex-Hardware-Transcoding): no description found
- [Thinking Christian Bale GIF - Thinking Christian Bale Patrick Bateman - Discover &amp; Share GIFs](https://tenor.com/view/thinking-christian-bale-patrick-bateman-american-psycho-mad-gif-18161559): Click to view the GIF
- [diable/enable CUDA Sysmem Fallback Policy from command line](https://gist.github.com/itsdotscience/4e29dca91f010a1873d1083fae94a655): diable/enable CUDA Sysmem Fallback Policy from command line - a
- [Crash Course Mix [ RaiZen ]](https://www.youtube.com/watch?v=K3XcrDoc8bQ): Used theme :Crash Course - PhysicsCrash Course - Anatomy &amp; PsychologyCrash Course - AstronomyCrash Course - PhilosophyCrash Course - PsychologyI do not own a...
- [The Molecular Shape of You (Ed Sheeran Parody) | A Capella Science](https://www.youtube.com/watch?v=f8FAJXPBdOg): I&#39;m in love with your bonding orbitals.Support A Capella Science: http://patreon.com/acapellascienceSubscribe! https://www.youtube.com/subscription_center?ad...
- [Lisa Su Amd GIF - Lisa Su Amd Ryzen9 - Discover &amp; Share GIFs](https://tenor.com/view/lisa-su-amd-ryzen9-zen2-ryzen-power-gif-14477035): Click to view the GIF
- [THE TERMINATOR &quot;Final Fight Clip&quot; (1984) Sci Fi Horror Action](https://www.youtube.com/watch?v=72-gVSXt_VU): THE TERMINATOR &quot;Final Fight Clip&quot; (1984) Sci Fi Horror ActionPLOT: In 1984, a human soldier is tasked to stop an indestructible cyborg killing machine, both ...
- [Making YBCO superconductor](https://www.youtube.com/watch?v=sLFaa6RPJIU): How to make and test your own pieces of YBCO superconductor.Best how-to resources for YBCO:http://physlab.org/wp-content/uploads/2016/04/Superconductor_manua...
- [GitHub - bodaay/HuggingFaceModelDownloader: Simple go utility to download HuggingFace Models and Datasets](https://github.com/bodaay/HuggingFaceModelDownloader): Simple go utility to download HuggingFace Models and Datasets - GitHub - bodaay/HuggingFaceModelDownloader: Simple go utility to download HuggingFace Models and Datasets
- [Jack Tanamen GIF - Jack Tanamen Gecky - Discover &amp; Share GIFs](https://tenor.com/view/jack-tanamen-gecky-incorporeal-gif-21102682): Click to view the GIF
- [GitHub - gameltb/ComfyUI_stable_fast: Experimental usage of stable-fast and TensorRT.](https://github.com/gameltb/ComfyUI_stable_fast): Experimental usage of stable-fast and TensorRT. Contribute to gameltb/ComfyUI_stable_fast development by creating an account on GitHub.
- [Nvidia RTX A6000 48GB GDDR6 Ampere Graphics Card PNY  3536403379193 | eBay](https://www.ebay.co.uk/itm/225986462736): no description found
- [GitHub - allenai/OLMo: Modeling, training, eval, and inference code for OLMo](https://github.com/allenai/OLMo): Modeling, training, eval, and inference code for OLMo - GitHub - allenai/OLMo: Modeling, training, eval, and inference code for OLMo
- [I have this paper:Single-cell	multi-omics	defines	the	cell-type	specific	impac - Pastebin.com](https://pastebin.com/EvrWhtJf): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.
- [Vanadium Dioxide as a Natural Disordered Metamaterial: Perfect Thermal Emission and Large Broadband Negative Differential Thermal Emittance](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.3.041004#fulltext): Thermal radiation from conventional emitters, such as the warm glow of a light bulb, increases with temperature: the hotter the bulb, the more it glows. Thermal emitters that buck this trend could lea...

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1202888060258492496) (227 messages🔥🔥): 

- **Aphrodite Batching and GPU Compatibility Issues**: `@sunija` highlighted the advantages of Aphrodite's batching capabilities for services like AI horde but also pointed out it doesn't work well with two GPUs of different VRAM sizes. `@goldkoron` added favorably on batch generation options and expressed disappointment about the GPU issue.
- **Usage of Context in Aphrodite**: According to `@sunija`, Aphrodite can store multiple conversations' contexts for efficient reuse. Meanwhile, `@keyboardking` and `@goldkoron` raised concerns over potential memory usage and discussed the possibility of offloading processed context to the CPU.
- **Calibration Dataset Discussions and AWQ Model Cards**: `@dreamgen` inquired about best calibration datasets for Automatic Weight Quantization (AWQ), and `@turboderp_` highlighted the inclusion of variety in calibration datasets for EXL2, emphasizing the importance of variety in datasets for quality results.
- **Local AI for Roleplay**: `@dxfile` shared experiences with using different models for role-playing and preferred instruction mode to chat mode, receiving feedback from `@sao10k` that instruct mode is optimal when the model is instruction-tuned. `@dreamgen` and `@firepin123` asked for clarification on support for various formats like iq3_xss in koboldcpp.
- **Leaderboards and the MoMo Model**: `@mrdragonfox` and others discussed the controversial presence of the Mistral medium model (MoMo) on a leaderboard, touching on the problems of models without clear licensing and potential legal issues of using or promoting leaked models. `@kaltcit` and `@c.gato` offered critical views on corporate honesty and the secrecy around model training specifics.

**Links mentioned**:

- [TheBloke’s gists](https://gist.github.com/TheBloke): GitHub Gist: star and fork TheBloke&#39;s gists by creating an account on GitHub.
- [Importance matrix calculations work best on near-random data · ggerganov/llama.cpp · Discussion #5006](https://github.com/ggerganov/llama.cpp/discussions/5006): So, I mentioned before that I was concerned that wikitext-style calibration data / data that lacked diversity could potentially be worse for importance matrix calculations in comparison to more &quot;...
- [TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ · Hugging Face](https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ#provided-files-and-awq-parameters): no description found
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/o8n4gcejJS): no description found

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1202905015489134642) (8 messages🔥): 

- **Exploring Quantization and LoRA Fine-Tuning**: `@dreamgen` inquired whether fine-tuning a pre-trained AWQ model with LoRA would perform better than using the base model when planning to quantize later. `@dirtytigerx` clarified that while AWQ is different from standard QLoRA, there's no evidence that it performs better.
- **Clarification on QLoRA Methodology**: In response to `@dreamgen`, `@dirtytigerx` compared AWQ to normal QLoRA, emphasizing that QLoRA uses `load_in_4bit` via `bitsandbytes` while AWQ employs a different quantization method.
- **Introducing LoftQ: Bridging the Quantization Gap**: `@dreamgen` shared a [link to a paper](https://arxiv.org/abs/2310.08659) discussing LoftQ, a technique for quantization that simultaneously fine-tunes LoRa and quantizes a model to improve performance on downstream tasks.
- **Debating the Notions of Quantization and Fine-Tuning**: `@dreamgen` suggested that aligning the quantization process during QLoRA fine-tuning and during serving could offer benefits, but `@dirtytigerx` expressed skepticism regarding the wide replication of the paper's results.

**Links mentioned**:

[LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models](https://arxiv.org/abs/2310.08659): Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning. In this work we focus on the scenario where quantization and L...

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/) (1 messages): 

kquant: internLM is a solid recommendation.
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1203085225274384384) (2 messages): 

- **Seeking Advice on LLM Deployment**: `m.0861` inquired about **best practices for deploying Large Language Models (LLMs)** using the [HF (Hugging Face) spaces](https://huggingface.co/spaces) service, hinting at possible use of the service for such purposes.
- **Exploring HF Inference Endpoints for LLMs**: Shortly after, `m.0861` considered that **HF's inference endpoints** might be a more appropriate tool for deploying LLMs, suggesting a shift in focus to that feature.
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1202989457339588618) (380 messages🔥🔥): 

- **Distillation Insights for SSD-1B**: `@pseudoterminalx` remarked on SSD-1B's inflexibility due to its distillation from a fine-tuned model. `@gothosfolly` concurred, mentioning that distilling from multiple fine-tuned models can enhance aesthetics.

- **Captioning Chat for Data Quality**: In a detailed exchange dominated by `@pseudoterminalx` and `@gothosfolly`, they discussed strategies for training models with properly captioned images to enhance prompt adherence. `@pseudoterminalx` reported using a combination of image sources like BLIP, COCO, and LLaVA, applying input perturbations, and addressing data pipeline issues like resizing and cropping for better training efficiency and data quality.

- **Techniques for Enhanced Prompt Adherence**: `@pseudoterminalx` and `@gothosfolly` debated the value of using UTF-8 tokenization for text encoding and a hybrid approach that combines UTF-8 codes into a single token. They considered whether a model using ByT5's byte-level encoding could offer advantages for image generation, especially in handling text.

- **Cropping Models for Image Upscaling**: The conversation between `@pseudoterminalx` and `@astropulse` highlighted the benefits of using cropped model weights for image-to-image upscaling. They noted that this approach helps maintain scene integrity and seems to work effectively for higher resolution upscaling.

- **Troubleshooting Global Information Issues in VAEs**: A discussion led by `@drhead` considered the problem of visual editing models like the StyleGAN3 and SD VAE sneaking global information through intense regions in generated images. `@thejonasbrothers` also pitched in, suggesting a variety of options to counteract this effect, emphasizing the need for concrete evidence rather than theorizing.

**Links mentioned**:

- [Google&#39;s AI Makes Stunning Progress with Logical Reasoning](https://youtu.be/NrNjvIrCqII?si=Q4hcfBZ__yPnu9ip): 🤓Learn more about Artificial Intelligent on Brilliant! ➜ First 200 to use our link https://brilliant.org/sabine will get 20% off the annual premium subscrip...
- [Sadako The Ring GIF - Sadako The Ring Ringu - Discover &amp; Share GIFs](https://tenor.com/view/sadako-the-ring-ringu-gif-14695335): Click to view the GIF

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1203084179122819152) (24 messages🔥): 

- **Cosine Annealing Takes a Backseat**: `@top_walk_town` shared their surprise about a new report that challenges the effectiveness of cosine annealing, describing it as a "roller coaster." The report is accessible via [Notion](https://shengdinghu.notion.site/).

- **Research Lands on Blogs Over Journals**: `@chad_in_the_house` and others find it noteworthy that significant research findings are often shared in blog posts rather than through traditional academic publishing due to the hassle with peer review processes.

- **Novel Architectures to Skip Traditional Publishing**: `@mkaic` is considering releasing information on a novel architecture they are working on through a blog post, expressing frustration with the current state of academic publishing.

- **Low-Hanging Fruit in Machine Learning Research**: `@mkaic` brought up how machine learning research is often just about applying well-known techniques to new datasets, which has become unexciting and crowds the landscape with incremental papers.

- **Industry Experience Over Academic Publications**: `@twoabove` recounted how their practical achievements in data competitions and industry connections provided opportunities beyond what academic papers could offer, hinting at the diminishing impact of being published in top journals.

**Links mentioned**:

- [How Did Open Source Catch Up To OpenAI? [Mixtral-8x7B]](https://www.youtube.com/watch?v=PYZIOMvkUF8): Sign-up for GTC24 now using this link! https://nvda.ws/48s4tmcFor the giveaway of the RTX4080 Super, the full detailed plans are still being developed. Howev...
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---



### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1203011101193801728) (158 messages🔥🔥): 

- **Defining AI Engineer Skills**: @eugeneyan sought input on the necessary skills for software engineers to effectively use LLMs, leading to discussions about understanding the probabilistic nature of LLMs and the importance of evaluation, debugging, and data familiarity. Recognition of a difference emerged between traditional software engineering and AI engineering roles, with various views on whether calling LLM APIs could shape an SDE into a data scientist role.
  
- **The AI Engineer Continuum**: Community members, including @eugeneyan and @swyxio, debated the stages of AI engineering expertise, from using APIs and rapid prototyping to fine-tuning models. A key focus was on the mindset shift required for engineers to move from deterministic to probabilistic outcomes and handling large data volumes effectively.
  
- **Skill Set Spotlight in AI Sector**: @coffeebean6887 and @eugeneyan discussed the importance of job titles versus actual skill sets in the industry, considering expanding beyond traditional SDEs to other roles like data engineers and analysts. There was consensus that adaptability and rapid learning of evolving best practices in AI take priority over specific titles.

- **Exploration of CUDA Learning**: @420gunna and other community users pondered the value of learning CUDA for future career prospects, contrasting it with the appeal of popular technologies and the rarity of in-depth CUDA knowledge in the LLM field. 

- **Concerns & Curiosities About OpenAI's Fine-Tuning API**: @dtflare raised questions about experiences with OpenAI's Fine-Tuning API/SDK, and @swyxio shared skepticism about the potential for platform lock-in and recommended going "the whole way" with fine-tuning rather than using simplified APIs, unless a substantial gain was evident.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200548371715342479/1203084424221171773): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Getting Started With CUDA for Python Programmers](https://www.youtube.com/watch?v=nOxKexn3iBo&t=7s): I used to find writing CUDA code rather terrifying. But then I discovered a couple of tricks that actually make it quite accessible. In this video I introduc...
- [Tweet from Eugene Yan (@eugeneyan)](https://x.com/eugeneyan/status/1753445305545298314): Trying to write a JD to hire software engineers that build with LLMs.  Beyond making REST calls, what&#39;s essential? Some I can think of:  • Evals: Collect labels & measure task performance • Look a...
- [Buttondown](https://buttondown.email/ainews/archive/ainews-trust-in-gpts-at-all-time-low/)): no description found
- [The Rise of the AI Engineer](https://www.latent.space/p/ai-engineer): Emergent capabilities are creating an emerging job title beyond the Prompt Engineer. Plus: Join 500 AI Engineers at our first summit, Oct 8-10 in SF!
- [GitHub - AbanteAI/rawdog: Generate and auto-execute Python scripts in the cli](https://t.co/Y2DQjpKv6K): Generate and auto-execute Python scripts in the cli - GitHub - AbanteAI/rawdog: Generate and auto-execute Python scripts in the cli

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1202917746677907506) (2 messages): 

- **Join the LLM Paper Club (East) Discussion**: `@swyxio` announces the ongoing LLM Paper Club (East) led by `<@796917146000424970>`. Interested parties are encouraged to [join the discussion](https://discord.com/channels/822583790773862470/1200029657744027658) and check out the upcoming [AI Engineering Singapore meetup](https://lu.ma/aie-sg).

- **Don't Miss AI in Action**: `@kbal11` invites members to the **AI in Action** event currently in session, discussing "Onboarding normies / how to differentiate yourself from the AI grifters". The session is led by `<@315351812821745669>` and accessible [here](https://discord.com/channels/822583790773862470/1200548371715342479).

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200548371715342479): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1200029657744027658): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [AI Engineering Singapore meetup · Luma](https://lu.ma/aie-sg): What is this? Meetup for folks interested in machine learning, LLMs, and all things genAI with craft beers and good vibes downtown 🍻 swyx from https://latent.space/ is home for CNY, do...

  

---


### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/1202916188943032361) (63 messages🔥🔥): 

- **Granting Screen Share Permissions**: User `@ivanleomk` acknowledged that `@796917146000424970` (unidentified user) is sorting out screen share permissions and advised to give it some time.
- **Audio Troubles on Stage**: `@ivanleomk` instructed `@srini5844` to join the stage for audio issues and later mentioned a brief intermission due to their own audio issues.
- **Exploring Self Rewarding LLMs**: `@anthonyivn` raised questions about the methodology for generating preference pairs for self-rewarding LLMs, leading to clarifications by `@ivanleomk` about the paper's process using scores to form preference pairs.
- **Discussion on Improving Text Embeddings and RAG**: `@anthonyivn` shared insights from experiments with different rating scales and discussed a paper on improving text embeddings ([Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368.pdf)) which is being utilized in recent research.
- **Idea for a 'Code Club' and 'Production Club'**: `@j0yk1ll.` and `@jevonm` proposed creating a "code club" for walking through code together and a "production club" to review code/papers with actual implementation results, which can be valuable for engineers and those interested in real-world applications.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/822583790773862470/1202628983343288330): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171): Chain-of-thought prompting combined with pre-trained large language models has achieved encouraging results on complex reasoning tasks. In this paper, we propose a new decoding strategy, self-consiste...
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059): Retrieval-augmented language models can better adapt to changes in world state and incorporate long-tail knowledge. However, most existing methods retrieve only short contiguous chunks from a retrieva...
- [MTEB Leaderboard - a Hugging Face Space by mteb](https://huggingface.co/spaces/mteb/leaderboard): no description found
- [Let&#39;s build GPT with memory: learn to code a custom LLM (Coding a Paper - Ep. 1)](https://www.youtube.com/watch?v=5pjNlL533PA): You&#39;ve used an LLM before, and you might&#39;ve even fine-tuned one, but...have you ever built one yourself? How do you start from scratch and turn a new researc...

  

---


### Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1203082427140931594) (134 messages🔥🔥): 

- **Greetings and Scheduling**: `@alan_95125` initiated the conversation, and `@kbal11` mentioned that they would start after more folks arrived.
- **Anticipation and Time Checks**: A few participants such as `@yikesawjeez` and `@nuvic_` commented on the start time, with `@nuvic_` suggesting that the channel be renamed to "Fridays 1PM" to match the event schedule.
- **Sharing AI-Related Links**: `@yikesawjeez` shared a series of links to various articles and blog posts related to AI, covering topics from founding AI startups to practical AI use cases, with the longest link dump including resources like Hitchhiker’s Guide to AI, The Washington Post, and Towards Data Science, among others.
- **Channel Activity and Enthusiasm**: Users like `@eugeneyan` and `@coffeebean6887` commented on the increasing number of audience members, indicating growing interest and participation in the channel's event.
- **Launching a Local Group**: There was interest in forming a local group for Los Angeles, with `@juliekwak` requesting the creation of a channel and `@coffeebean6887` tagging potential members for an LA meetup, which led to `@swyxio` creating a new channel for it.

**Links mentioned**:

- [Symphony – Interfaces](https://www.symphony.run/blog/interfaces): Another interesting consequence of models like GPT-3.5+ being able to call functions is that this ability can be used to render visual interfaces within a conversation.
- [Gandalf | Lakera – Test your prompting skills to make Gandalf reveal secret information.](https://gandalf.lakera.ai/): Trick Gandalf into revealing information and experience the limitations of large language models firsthand.
- [People + AI Guidebook](https://pair.withgoogle.com/guidebook/patterns/how-do-i-onboard-users-to-new-ai-features): A toolkit for teams building human-centered AI products.
- [GitHub - uptrain-ai/uptrain: Your open-source LLM evaluation toolkit. Get scores for factual accuracy, context retrieval quality, tonality, and many more to understand the quality of your LLM applications](https://github.com/uptrain-ai/uptrain): Your open-source LLM evaluation toolkit. Get scores for factual accuracy, context retrieval quality, tonality, and many more to understand the quality of your LLM applications - GitHub - uptrain-ai...
- [Build software better, together](https://cs50.ai/chat): GitHub is where people build software. More than 100 million people use GitHub to discover, fork, and contribute to over 420 million projects.
- [My Life as a Con Man](https://www.swyx.io/con-man-life): Confidence is a dual edged sword. I trafficked in confidence when I was in finance, and now I see it everywhere I look. 
- [Why we founded Parcha ](https://www.hitchhikersguidetoai.com/p/why-we-founded-parcha): A deeper dive into why we&#x27;re building AI agents to supercharge compliance and operations teams in fintech at Parcha
- [How to use AI to do practical stuff: A new guide](https://www.oneusefulthing.org/p/how-to-use-ai-to-do-practical-stuff): People often ask me how to use AI. Here&#x27;s an overview with lots of links.
- [3 things everyone’s getting wrong about AI](https://www.washingtonpost.com/technology/2023/03/22/ai-red-flags-misinformation/): As AI tools spread, people are struggling to separate fact from fiction.
- [How to talk about AI (even if you don’t know much about AI)](https://www.technologyreview.com/2023/05/30/1073680/how-to-talk-about-ai-even-if-you-dont-know-much-about-ai/): Plus: Catching bad content in the age of AI.
- [What are AI Agents?](https://serokell.io/blog/what-are-ai-agents): In this post, you’ll learn what AI agents are and what they are truly capable of. You’ll also learn how to build an AI agent suitable for your goals.
- [AI Agent Basics: Let’s Think Step By Step](https://www.jonstokes.com/p/ai-agent-basics-lets-think-step-by): An introduction to the concepts behind AgentGPT, BabyAGI, LangChain, and the LLM-powered agent revolution.
- [Pitching Artificial Intelligence to Business People](https://towardsdatascience.com/pitching-artificial-intelligence-to-business-people-f8ddd8fb2da2): From silver bullet syndrome to silver linings
- [How PR people should (not) pitch AI projects](https://thenextweb.com/news/how-pr-people-should-not-pitch-ai-projects-syndication): These are exciting times for the artificial intelligence community. Interest in the field is growing at an accelerating pace, registration at academic and professional machine learning courses is soar...
- [Educating Clients about Machine Learning and AI &#8211; Andy McMahon](https://electricweegie.com/articles/educating-clients/): no description found
- [How to Announce Your Actual AI](https://matt.sh/ai-how-to-announce): no description found
- [no title found](https://dev.to/builderio/dont-build-ai-products-the-way-everyone-else-is-doing-it-9a7): no description found
- [7 Habits of Highly Effective AI Business Projects](https://towardsdatascience.com/7-habits-of-highly-effective-ai-business-projects-6ced590e6db8?gi=e4b47a172d38): What’s the difference between good &amp; great AI business projects? Here are 7 things to consider when doing AI work in your organisation.
- [How to convince Venture Capitalists you’re an expert in Artificial Intelligence](https://medium.com/machine-learning-in-practice/how-to-convince-venture-capitalists-youre-an-expert-in-artificial-intelligence-39d5edaca290): If you like this article, check out another by Robbie:  15 Ways a Venture Capitalist Says “No”
- [Launching your new AI Startup in 2023 &mdash; Building Better Teams](https://buildingbetterteams.de/profiles/brian-graham/navigating-ai-businesses): In the last few months more and more people have been asking me for my thoughts on their AI business ideas, and for help with navigating the space. This post covers the majority of my thoughts on the ...

  

---



### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1202927824953024564) (161 messages🔥🔥): 

- **Open Models as Large-Scale Science Projects**: `@layl` discussed the increasing ease of getting government support in the EU for open model training on national clusters, aligning with treating open models as large-scale science projects. Meanwhile, `@stellaathena` confirmed this trend from zero to small progress over the past years, suggesting possible future applications for `@layl`'s ML library in an HPC environment like LUMI.

- **Activation Function Analysis and OpenAI's Mish Experiment**: Amidst a broad discussion about activation functions like GeLU, ReLU, Mish, and TanhExp, `@xa9ax`, `@fern.bear`, and others exchanged insights and research, highlighting the lack of extensive empirical testing for different activation functions in large model training. `@ad8e` expressed skepticism about the honesty of a paper favoring Mish but was reassured after `@xa9ax` confirmed that all pre-submission experiments were included in the published manuscript.

- **Transformer++ and Mamba Models Examined**: Questions arose around `@state-spaces/mamba` models and how they compare to other architectures like Transformers++ and Pythia. `@ldj` and `@baber_` highlighted concerns about baselines and comparisons, while `@stellaathena` noted the absence of a standard model suite trained on open data for fair comparisons.

- **Diverse Takes on Activation Functions Impact**: Users `@catboy_slim_`, `@fern.bear`, and `@nostalgiahurts` offered thoughts on the subtle influences of activation function choices, like scale interactions with other hyperparameters and their impact on function performance. Shared empirical results from EleutherAI's blog and research papers were discussed as attempts to unravel complex dependencies between activation functions and model training dynamics.

- **Legal Quandaries of Large Model Training**: `@synquid` brought attention to the legal complications related to transparency in model training data sources, suggesting that openly revealing training data can attract intellectual property lawsuits, which could impede scientific progress.

**Links mentioned**:

- [Activation Function Ablation](https://blog.eleuther.ai/activation-fns/): An ablation of activation functions in GPT-like autoregressive language models.
- [Mish: A Self Regularized Non-Monotonic Activation Function](https://arxiv.org/abs/1908.08681): We propose $\textit{Mish}$, a novel self-regularized non-monotonic activation function which can be mathematically defined as: $f(x)=x\tanh(softplus(x))$. As activation functions play a crucial role i...
- [A decoder-only foundation model for time-series forecasting &#8211; Google Research Blog](https://blog.research.google/2024/02/a-decoder-only-foundation-model-for.html): no description found
- [Releasing Transformer++ models · Issue #63 · state-spaces/mamba](https://github.com/state-spaces/mamba/issues/63): Great work! Would it be possible to release your Transformer++ baseline models (specifically the ones trained on the Pile)?
- [TinyGSM: achieving &gt;80% on GSM8k with small language models](https://arxiv.org/abs/2312.09241): Small-scale models offer various computational advantages, and yet to which extent size is critical for problem-solving abilities remains an open question. Specifically for solving grade school math, ...
- [ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models](https://machinelearning.apple.com/research/relu): Large Language Models (LLMs) with billions of parameters have drastically transformed AI applications. However, their demanding computation…
- [Information Theory for Complex Systems Scientists](https://arxiv.org/abs/2304.12482): In the 21st century, many of the crucial scientific and technical issues facing humanity can be understood as problems associated with understanding, modelling, and ultimately controlling complex syst...
- [TanhExp: A Smooth Activation Function with High Convergence Speed for Lightweight Neural Networks](https://arxiv.org/abs/2003.09855): Lightweight or mobile neural networks used for real-time computer vision tasks contain fewer parameters than normal networks, which lead to a constrained performance. In this work, we proposed a novel...
- [Benchmarking PyTorch’s Native Mish](https://benjaminwarner.dev/2021/07/19/benchmarking-pytorch-native-mish#:~:text=Since%20Mish's%20introduction%2C%20Mish%20has,and%20Accuracy%20of%20Object%20Detection.): PyTorch 1.9 added a native implementation of Mish, my go to activation function for computer vision tasks. In this post I benchmark the computational performance of native Mish on a Tesla V100, Tesla ...
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#significance-level): Official Repository for &amp;quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&amp;quot; [BMVC 2020] - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#significance-le): Official Repository for &amp;quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&amp;quot; [BMVC 2020] - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [GitHub - digantamisra98/Mish: Official Repository for &quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&quot; [BMVC 2020]](https://github.com/digantamisra98/Mish?tab=readme-ov-file#summary-of-results-vision-tasks): Official Repository for &amp;quot;Mish: A Self Regularized Non-Monotonic Neural Activation Function&amp;quot; [BMVC 2020] - GitHub - digantamisra98/Mish: Official Repository for &amp;quot;Mish: A Self...
- [Partial entropy decomposition reveals higher-order structures in human brain activity](https://arxiv.org/abs/2301.05307): The standard approach to modeling the human brain as a complex system is with a network, where the basic unit of interaction is a pairwise link between two brain regions. While powerful, this approach...
- [Meet Mish: New Activation function, possible successor to ReLU?](https://forums.fast.ai/t/meet-mish-new-activation-function-possible-successor-to-relu/53299/1): Hi all,  After testing a lot of new activation functions this year, I’m excited to introduce you to one that has delivered in testing - Mish.     Per the paper, Mish outperformed ReLU by 1.67% in thei...

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1202897345386254377) (16 messages🔥): 

- **Seeking Knowledge Distillation Insights**: `@johnryan465` expressed interest in research on the efficiency gains of training a smaller model (size B) to match the logits of a larger, already trained model (size A), compared to training model A directly. `@xa9ax` and `@johnryan465` discussed the potential for using the *infinity ngrams paper* methodology to create inexpensive models that may be used in a distillation pretraining bootstrap pipeline.
  
- **Sampling Challenges in MCTS**: `@blagdad` touched on the exploration problem in Monte Carlo Tree Search (MCTS), highlighting the potential of using Upper Confidence bounds for Trees (UCT) to guide the exploration of the game tree based on uncertainty, as opposed to uniform expansion.

- **Efficient Exploration for Model Improvement Shared**: An interesting paper on efficiently selecting examples for fine-tuning by using human or LLM raters was shared by `@xylthixlm`, focusing on agents that generate queries and a reward model based on received feedback. The paper describes the efficiency of double Thompson sampling and the use of epistemic neural networks, available at [arXiv](http://arxiv.org/abs/2402.00396).

- **Active Learning Implementation Tease**: `@fedorovist` mentioned that `@322967286606725126` is polishing a Bayesian active learning implementation, with `@johnryan465` showing interest in any draft available due to past work on similar problems.

- **Adam Optimizer Variation Inquiry**: The question about whether any papers have explored Adam using the variance of parameters instead of the gradient for the second moment estimation was posed by `@ai_waifu`. No specific papers were mentioned as a response.


**Links mentioned**:

[Efficient Exploration for LLMs](http://arxiv.org/abs/2402.00396): We present evidence of substantial benefit from efficient exploration in gathering human feedback to improve large language models. In our experiments, an agent sequentially generates queries while fi...

  

---


### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1202904159838277642) (22 messages🔥): 

- **Vision-Language Integration**: User `@asuglia` expressed interest in integrating vision and language model support into lm-harness. `@hailey_schoelkopf` mentioned that while it's not a current focus, contributions are welcome with `@chrisociepa` and `@1072629185346019358` identified as possible collaborators.  

- **MMLU Standard Error Clarifications Sought**: `@baber_` raised questions regarding high standard errors in MMLU results for the model *miqu*. `@hailey_schoelkopf` acknowledged that the standard error calculations for groups within the evaluation code might need revisiting.

- **Zero-Shot Configuration Confirmed**: `@asuglia` asked about forcing a task to run in zero-shot mode within lm-harness. `@hailey_schoelkopf` confirmed that setting `num_fewshot: 0` should achieve this and pointed to the relevant [source code](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169) for clarification.

- **Fixes and Improvements to Grouped Task Evaluation**: `@hailey_schoelkopf` proposed an update to the variance calculation method used to aggregate standard errors across groups of tasks with a [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390) on the EleutherAI GitHub repository.

- **Coordination on Vision-Language Model Support**: `@jbdel.` offered a harness fork with a working vision and language pipeline, suggesting a hands-off to `@asuglia` post-Feb 15th. A [When2meet](https://www.when2meet.com/?23484385-k3FqO) was set up by `@hailey_schoelkopf` to find a suitable time to discuss and coordinate efforts.

**Links mentioned**:

- [lm-evaluation-harness/lm_eval/evaluator.py at 7411947112117e0339fe207fb620a70bcec22690 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/7411947112117e0339fe207fb620a70bcec22690/lm_eval/evaluator.py#L166-L169): A framework for few-shot evaluation of language models. - EleutherAI/lm-evaluation-harness
- [[WIP] Use Pooled rather than Combined Variance for calculating stderr of task groupings by haileyschoelkopf · Pull Request #1390 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1390): This PR updates the formula we use for aggregating stderrs / sample std. deviations across groups of tasks. In this PR: formula:  result: hf (pretrained=mistralai/Mistral-7B-v0.1), gen_kwargs: (Non...
- [LM Eval Harness--VLMs - When2meet](https://www.when2meet.com/?23484385-k3FqO): no description found

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1202982562344206437) (11 messages🔥): 

- **LLaVA-1.6 Outshines Gemini Pro**: `@pradeep1148` shared a [YouTube video](https://www.youtube.com/watch?v=SEavari8xaU) titled *"Trying LLaVA-1.6 on Colab"*, demonstrating the improved features of LLaVA-1.6 such as enhanced reasoning, OCR, and world knowledge, noting it even surpasses Gemini Pro on several benchmarks. The results and details are provided on the [LLaVA blog](https://llava-vl.github.io/blog/2024-01).

- **Notorious Hacker Strikes Again**: `@itali4no` posted a [VX Twitter link](https://vxtwitter.com/JackPosobiec/status/1753416551066181672) commenting on the latest feat by "the hacker known as 4chan".

- **Apple Vision Pro Product Launch**: User `@nonameusr` announced the launch of Apple Vision Pro, but did not provide any additional information or link to the product.

- **AI Doomer vs. e/acc Leader Debate**: `@if_a` linked to a [YouTube debate](https://www.youtube.com/watch?v=0zxi0xSBOaQ) featuring a head-to-head between Connor Leahy, dubbed the world's second-most famous AI doomer, and Beff Jezos, founder of the e/acc movement, discussing technology, AI policy, and human agency.

- **In Memoriam of Carl Weathers**: User `@gabriel_syme` expressed condolences over the passing of Carl Weathers, with a statement of remembrance but without linking to any external news source.

**Links mentioned**:

- [Trying LLaVA-1.6 on  Colab](https://www.youtube.com/watch?v=SEavari8xaU): LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.https://llava-vl.github.io/blog/2024-01...
- [Explosive Showdown Between e/acc Leader And Doomer](https://www.youtube.com/watch?v=0zxi0xSBOaQ): The world&#39;s second-most famous AI doomer Connor Leahy sits down with Beff Jezos, the founder of the e/acc movement debating technology, AI policy, and human ...

  

---


### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1202916357327556640) (17 messages🔥): 

- **Hugging Face Introduces MiniCPM**: User `@Fynn` shared a link to a Hugging Face paper on MiniCPM, a new model that could be of interest ([MiniCPM Paper](https://huggingface.co/papers/2402.00838)).
- **Testing MiniCPM on Twitter**: `@burnytech` referenced a [Twitter thread](https://twitter.com/abacaj/status/1753207827458396328) showcasing tests of the new MiniCPM model, sparking discussions about its performance.
- **Healthy Skepticism for MiniCPM Benchmarks**: `@mister_poodle` commented that although MiniCPM's scores are good, it underperforms compared to Mistral on the MMLU benchmark, and the usage of the model for fine-tuning on specific tasks is awaited.
- **Model Comparisons Ignite Discussion**: `@bozoid` pointed out that the MiniCPM not being specifically trained for math but achieving a 53 on the GSM8K benchmark is impressive and underscored that comparisons with newer models like StableLM 2 are missing.
- **Potential for Model Merging**: User `@bozoid` expressed that model merging efforts could potentially enhance the capabilities of ~2B scale models, given the recent advancements in this area.

**Links mentioned**:

[Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1202896238312431626) (114 messages🔥🔥): 

- **Growing LLMs Debate**: `@theluckynick` shared a [tweet by @felix_red_panda](https://x.com/felix_red_panda/status/1752996197940220231?s=20) discussing the success of "growing" models like ResNet and GANs, and questioning the application of this to LLMs. ResNet classifiers and ProGANs benefited from this technique, with examples including Apple's Matroyshka Diffusion model.

- **Miqu's First Impression**: `@weyaxi` announces that **Miqu** has entered the Open LLM Leaderboard with a score of 76.59. Subsequent messages from `@nonameusr` and others compare Miqu's performance metrics, such as ARC and MMLU, to other models like MoMo, with mixed reactions regarding Miqu's potential.

- **Finetuning Trade-offs between autotrain and axolotl**: `@papr_airplane` inquires about compromises when using **autotrain** versus **axolotl** for finetuning, with `@teknium` suggesting sample packing, flash attention, and prompt format selection as potential differences.
  
- **Exploring Multilingual Tokenizers**: `@light4bear` sparked a discussion regarding LLMs and tokenizers, particularly focused on how a 32000-token vocabulary might limit the multilingual capabilities of models like llama/mistral. `@teknium` provided a [link to a paper on VinaLLaMA](https://arxiv.org/abs/2312.11011), an open-weight SOTA Large Language Model for Vietnamese, and `@light4bear` mentioned efforts at adapting LLaMA models for Chinese.

- **Quantized Models on the Leaderboard**: Conversations emerged around quantized models, specifically **miqu-70b**, with various users like `@.ben.com`, `@betadoggo`, and `@nonameusr` discussing the impact of quantization on performance, speeling accuracy, and whether these models are run on specific platforms by default.

**Links mentioned**:

- [Zyphra (Zyphra)](https://huggingface.co/Zyphra): no description found
- [AI News: GPT-4-Level Open Source, New Image Models, Neuralink (And More)](https://youtu.be/PSB_QQTp0GU?si=lslRJ8JexekkpBrx&t=729): Here&#39;s all the AI news from the past week that you might have missed. Check out HubSpot&#39;s Campaign Assistant here: https://clickhubspot.com/xn6Discover More ...
- [VinaLLaMA: LLaMA-based Vietnamese Foundation Model](https://arxiv.org/abs/2312.11011): In this technical report, we present VinaLLaMA, an open-weight, state-of-the-art (SOTA) Large Language Model for the Vietnamese language, built upon LLaMA-2 with an additional 800 billion trained toke...
- [Tweet from Felix (@felix_red_panda)](https://x.com/felix_red_panda/status/1752996197940220231?s=20): We know that &#34;growing&#34; models (=adding more parameters during training) works well for ResNet classifiers (@jeremyphoward did that ages ago), GANs (ProGAN) and image Diffusion models (Apple&#3...
- [Tweet from Weyaxi (@Weyaxi)](https://fxtwitter.com/Weyaxi/status/1753353836373135415): Miqu is now on the 🤗Open LLM Leaderboard, achieving a score of 76.59.  https://hf.co/152334H/miqu-1-70b-sf  Benchmarks  Average: 76.59 ARC: 73.04 HellaSwag: 88.61 MMLU: 75.49 TruthfulQA: 69.38 Winogr...
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20): A new tool that blends your everyday work apps into one. It's the all-in-one workspace for you and your team
- [Chinese-LLaMA-Alpaca/README_EN.md at main · ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/main/README_EN.md): 中文LLaMA&amp;Alpaca大语言模型+本地CPU/GPU训练部署 (Chinese LLaMA &amp; Alpaca LLMs) - ymcui/Chinese-LLaMA-Alpaca
- [Thealexera Soyjak GIF - Thealexera Soyjak Surprised - Discover &amp; Share GIFs](https://tenor.com/view/thealexera-soyjak-surprised-amazed-wojak-gif-19919803): Click to view the GIF

  

---


### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1202892422275801120) (16 messages🔥): 

- **More Data Better?**: `@stefangliga` recommends saving all preference data, not just top choices, mentioning that **Data Preferences Optimization (DPO)** can utilize a ranking of multiple responses, though implementations are rare.
- **Proper Prompt Formatting for Hermes 2**: `@mr.fundamentals` shared a code snippet used for formatting prompts for the [Nous Hermes 2 Mixtral 8x7B DPO model](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO), looking for advice on why initial characters might be missing in responses.
- **Check Your Outputs**: In response to `@mr.fundamentals`, `@teknium` suggested printing out the formatted prompt to help debug issues with skipped initial characters in model responses.
- **Avoiding Lengthy Replies**: `@teknium` advised `@mr.fundamentals` on how to prompt the model to generate shorter responses by providing example turns with desired length, while noting it might increase token usage.

**Links mentioned**:

[NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO?text=My+name+is+Teven+and+I+am): no description found

  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1202911564487725086) (25 messages🔥): 

- **Exploring Dense Languages for Machine Learning**: `@pyhelix` discussed an encoding scheme idea using a "7th bit" to signify cognitive dissonance in machine learning models and pondered the application of **modular forms** for creating a dense language.

- **Does OpenAI Censor ChatGPT for China?**: User `@bambooshoots` inquired whether OpenAI censors **ChatGPT** responses based on Chinese law; `@jeremy.o` responded, clarifying that **OpenAI** does not censor content for reasons related to Chinese regulations.

- **Content Creation Freedom Using DALL·E**: `@jeremy.o` highlighted that **OpenAI** allows users to create diverse representations, including LGBTQI+ content using **DALL·E**, indicating a commitment to content freedom.

- **Contours of Content Restrictions Discussed**: `@bambooshoots` expressed concerns about **ChatGPT** refusing to discuss topics even within the scope of *fair use*, with `@jeremy.o` and `@lugui` providing context on content guidelines and dramatization propensities of ChatGPT.

- **Philosophical Readings on Machine Intelligence**: `@jimmygangster` shared an intriguing read titled *From Deep Learning to Rational Machines*, which delves into the philosophical study comparing animal and human minds.

Note: Other participant messages were casual greetings or undetailed mentions and do not contribute substantive discussion points to summarize.
  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1203007582990770186) (117 messages🔥🔥): 

- **@ Mentions Confusion and Potential for Bot Collaboration**: `@blckreaper` and `@darthgustav.` discussed the concept of using **@ mentions** to collaborate between different instances of GPT, with `@jaicraft` highlighting a desire for separate entities in conversations that don't confuse past responses as their own.
- **GPT Instruction Leakage Concerns**: `@loschess` expressed concerns about GPTs leaking their custom instructions, with `@solbus` explaining that GPT's instructions are akin to client-side code in HTML, and `@bambooshoots` suggesting to secure sensitive content behind an API call action.
- **@ Mentions Integration and Agentic Behavior**: `@jaicraft` and `@darthgustav.` debated the functionality and limitations of @ mentions, discussing the possibility of multiple GPT entities in a chat and the need for better separation of instructions.
- **Bugs and Inconsistencies in GPT Responses**: Users including `@_odaenathus`, `@blckreaper`, and `@loschess` report experiencing bugs and inconsistencies with GPT representations, knowledge file retrieval, and an unwillingness to perform certain tasks, suggesting a recent change in GPT behavior.
- **Request for Enhanced Entity Differentiation**: The discussion led by `@jaicraft` pointed towards a user interest in GPTs acting as separate entities with distinct memories and behaviors, rather than as a continuation of a single conversation flow.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1194685637077499925/1195059846236606564): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1202034296823480391): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/974519864045756446/1001151820170801244/1201968771343061002): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.

  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1203028218546749450) (6 messages): 

- **Request for Implicit Text Modification**: `@novumclassicum` inquired about a method to have GPT **modify text for text-to-speech** without displaying the changes on-screen. They seek to provide an output-ready submission after the GPT performs word replacements in memory.

- **Injecting Personality into AI Responses**: `@_fresnic` looked for advice on making API responses reflect a certain **personality**. They've noted some success with prompting GPT to *“talk like someone who is [personality...]"*.

- **Reducing Repetitive Server Communication Permissions**: `@novumclassicum` asked for a solution to prevent their **custom GPT** from repeatedly asking users for server communication permissions after the initial consent.

- **Challenges in Sustaining Multi-Character Dialogues**: `@stealth2077` sought tips for generating **realistic conversations** between multiple characters. They struggle to get the AI to produce more than three lines of dialogue before it summarizes the conversation.

- **Desire for Detailed Play-by-play Character Interactions**: Further emphasizing the issue, `@stealth2077` expressed a desire for the AI to generate a full dialogue with every line instead of summarizing.
  

---


### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1203028218546749450) (6 messages): 

- **Seeking Stealth Text Processing Tips**: `@novumclassicum` is looking for a way to have GPT perform text modifications invisibly before submission, specifically for a text-to-speech application. The desired outcome is for the GPT to replace words in memory and submit the text without displaying the modifications to the user, but they’re uncertain about the instructions required to achieve this.

- **Personal Touch in API Responses**: `@_fresnic` experimented with giving an AI via the API a personality, starting with a name and interests to be included in the responses. They discovered that phrasing like "talk like someone who is [personality...]" seemed to improve the AI's response.

- **One-Click Connection Conundrum**: `@novumclassicum` inquires about a way to prevent a custom GPT from repeatedly asking for permission to communicate with an outside server after the first approval. They're looking to replicate the feature where the popup will not trigger after the initial click.

- **Generating Those Character Dialogues**: `@stealth2077` seeks advice on generating realistic and extensive discussions between characters in a narrative. They've found that the AI tends to summarize conversations after three lines instead of continuing with the dialogue.

- **Dialogue Expansion Desired**: Continuing the topic, `@stealth2077` expresses difficulty in getting more than a few lines of dialogue before the AI switches to summarization. They wish for it to generate every single line of the conversation.
  

---



### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1202909658831130645) (58 messages🔥🔥): 

- **Learning Curve for LLM Creation**: `@heyitsyorkie` highlighted the complexities of creating a custom LLM, indicating that expertise in Machine Learning, PyTorch, among other skills, is necessary.
- **Exploring the Possibilities of LM Studio Plugins**: `@nntb` inquired about compatible AI agents and plugins for LM Studio like TTS and open interpreter. `@fabguy` directed them to check specific channels for more information.
- **Clarifying LM Studio's Capabilities**: Users queried about running multiple NLP models and agents concurrently and integrating non-conversational elements into chatbots. `@heyitsyorkie`, `@fabguy`, and `@.ben.com` contributed clarifications on the abilities and limitations of LM Studio.
- **Headless Operation for LM Studio**: `@quarky93` asked about the feasibility of running LM Studio backend on a server while using the UI locally, with `@heyitsyorkie` responding that headless operation is not currently supported.
- **Model Recall and Context Window Exploration**: `@kirkouimet` expressed concerns about the fuzzy memory of models within available context windows. `@wildcat_aurora` responded with information about a Mixtral model with a larger 195k token context window and the hardware requirements for running such models.

**Links mentioned**:

[TheBloke/Mixtral_34Bx2_MoE_60B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Mixtral_34Bx2_MoE_60B-GGUF): no description found

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1202941411729866762) (19 messages🔥): 

- **Inquiry on DeepSeek-MoE-16B Support**: `@czkoko` asked if llama.cpp now supports **DeepSeek-MoE-16B**, pointing out the lack of attention towards this expert model. `@heyitsyorkie` responded that it should work if there's a GGUF quant, and later shared the intent to test the model, citing the same creator as for Goliath.

- **Moondream for Vision to Text Transformation**: `@devrifter` introduced **Moondream**, a model proficient in converting pictures to text, available on Hugging Face. `@heyitsyorkie` clarified that Moondream will not run in LMStudio as is, but provided a [link to try it](https://huggingface.co/spaces/vikhyatk/moondream1) for those interested.

- **The CodeLlama 70B Experiment**: `@yagilb` shared an experimental preset for **CodeLlama 70B**, providing a Discord link for those interested in experiencing the cutting-edge in coding models.

- **Interest in Uncensored Models Hinted with "Dolphin"**: `@devrifter` hinted at searching for the phrase "dolphin" when seeking uncensored models, suggesting a keyword associated with such content.

- **Mistral Ai Finetunes Llama**: `.ben.com` mentioned a recent leak of a Mistral Ai fine-tune of **llama 70B** known as **miqu**, and provided a [link to the model](https://huggingface.co/miqudev/miqu-1-70b), stating it performs surprisingly well in coding tasks.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1201953800634507325/1203097188545077248): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [moondream1 - a Hugging Face Space by vikhyatk](https://huggingface.co/spaces/vikhyatk/moondream1): no description found
- [dagbs/deepseek-coder-7b-base-v1.5-GGUF · Hugging Face](https://huggingface.co/dagbs/deepseek-coder-7b-base-v1.5-GGUF): no description found
- [miqudev/miqu-1-70b · Hugging Face](https://huggingface.co/miqudev/miqu-1-70b): no description found
- [List Amazinggrace GIF - List Amazinggrace Court - Discover &amp; Share GIFs](https://tenor.com/view/list-amazinggrace-court-watching-scroll-gif-13488308): Click to view the GIF
- [jartine/llava-v1.5-7B-GGUF at main](https://huggingface.co/jartine/llava-v1.5-7B-GGUF/tree/main): no description found
- [GitHub - deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE): Contribute to deepseek-ai/DeepSeek-MoE development by creating an account on GitHub.

  

---


### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1203027656153628692) (32 messages🔥): 

- **Dual GPU Dilemma**: User `@merpdragon` experiences issues with models failing to load or performing very slowly on a dual GPU setup, consisting of an RTX 3070 and a GTX 1060, despite having 80GB of RAM. `@heyitsyorkie` and `@.ben.com` discuss shared memory problems and VRAM limitations, hinting that the GTX 1060 may not be making a significant contribution to performance.
  
- **Nvidia Control Panel Tip**: `@.ben.com` shares that NVIDIA's control panel has a setting to disable shared memory, potentially helping with `@merpdragon`'s issue.

- **Considering Hardware Upgrade for LLMs**: `@heyitsyorkie` advises that for speed improvements with 70b models, one should consider upgrading to dual RTX 3090 GPUs, as 3070 would still run slow.

- **Big Language Models and VRAM Bound Performance**: `@.ben.com` and `@rugg0064` shed light on how VRAM is a crucial factor in running LLMs, with performance being memory-bound rather than compute-bound in many scenarios.

- **Anticipating a New LLM Machine Setup**: User `@wildcat_aurora` is anticipating setting up a machine with 4 P40 GPUs and considers using Ubuntu for better performance in running 70b models, while `@kujila` enquires about building a similar setup using last-gen AMD motherboards and eBay GPUs.

- **Benchmarking CPUs with LM Studio**: `@goldensun3ds` plans to benchmark different CPUs against each other using LM Studio, questioning the ideal GPU layer settings and considering using Task Manager's VRAM usage as a gauge, with `@rugg0064` confirming that approach. `@.ben.com` points out that Core counts will have less impact due to memory bottlenecks, and parameters like Top P do not affect inference performance.
  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1203058403925172234) (8 messages🔥): 

- **Model Download Double Check**: `@yagilb` inquired if BOTH the primary model and the vision adapter had been downloaded, indicating that both components might be necessary.
- **Help Offered with Screenshot Analysis**: `@yagilb` offered to assist by analyzing a screenshot of the search result screen to solve an issue.
- **Clarification Sought on 30b Model Source**: `@n8programs` asked about the source of the 30b gguf, looking for details on where to obtain the model.
- **Partial Support Acknowledged in Llama Library**: `@n8programs` mentioned that version 1.6 support is only partial in llama.cpp, highlighting limitations in the current implementation.
- **Performance Gains in Llama Library Uncertain**: `@n8programs` pointed out that while performance gains are anticipated in llama.cpp, they have not yet been realized due to incomplete proper image preprocessing.
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1203142295088799815) (6 messages): 

- **Docker Woes Lead to Conda Solutions**: `@nntb` experienced significant issues with Docker, which led them to install Conda as an alternative.
- **Instructions Fell Short for Setup**: Despite following the provided instructions, `@nntb` was unable to resolve the issues without additional installation steps.
- **Environment Created With Conda**: `@nntb` set up a Conda environment to circumvent the problems with Docker.
- **API Key Troubleshooting**: To troubleshoot, `@nntb` mentioned having to add "EMPTY" to the API key, hinting at a possible solution they discovered.
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1203077819815698442) (2 messages): 

- **Debating Embedding Storage Strategies**: `@drale2k` inquired about the tradeoff between storing **longer word embeddings** in fewer database rows against **shorter embeddings** in more rows, considering both **similarity search quality** and **database performance**.
- **Context Matters in Similarity Searches**: `@drale2k` added that longer chunks might yield better search results by providing **more context**, but recognized they would likely impact database efficiency and memory usage.
  

---



### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1202897534192984075) (45 messages🔥): 

- **Choosing the Right Model for Text Embeddings**: `@bwo_28` asked for advice on how to choose among many models with different dimensions for text embeddings. The challenge of selecting a suitable model according to specific needs was outlined, but no specific models or criteria were recommended in the conversation.
- **Innovative Ideas to Integrate Docs**: `@xzuyn` sparked a discussion on whether HuggingFace's documentation could be converted into a dynamic format to assist in training language models, with `@not_lain` referencing similar existing methods and suggesting it would be a valuable addition to the platform.
- **Transformers Upgrade Guidance**: In a troubleshooting exchange, `@7sinstsugluttony` confirmed that following `@not_lain`'s advice to upgrade the `transformers` library resolved their issue, demonstrating peer-to-peer community support in action.
- **Open Call for Project Collaboration**: HuggingFace users such as `@adityaiiitr` are expressing interest in contributing to community projects, with others like `@not_lain` and `@lunarflu` offering guidance on finding repositories and initiatives to join.
- **Enthusiasm About an AI Summit**: `@uncleflowerdj` shared an invitation to the GenAI Summit San Francisco 2024, providing event details and discount codes, clearly generating excitement and community engagement around upcoming AI events.

**Links mentioned**:

- [w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored · Hugging Face](https://huggingface.co/w4r10ck/SOLAR-10.7B-Instruct-v1.0-uncensored): no description found
- [Hugging Face](https://github.com/huggingface/): The AI community building the future. Hugging Face has 190 repositories available. Follow their code on GitHub.
- [GitHub - IDEA-Research/Grounded-Segment-Anything: Grounded-SAM: Marrying Grounding-DINO with Segment Anything &amp; Stable Diffusion &amp; Recognize Anything - Automatically Detect , Segment and Generate Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything): Grounded-SAM: Marrying Grounding-DINO with Segment Anything &amp;amp; Stable Diffusion &amp;amp; Recognize Anything - Automatically Detect , Segment and Generate Anything - GitHub - IDEA-Research/Grou...
- [GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX.](https://github.com/huggingface/transformers): 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, and JAX. - GitHub - huggingface/transformers: 🤗 Transformers: State-of-the-art Machine Learning for Pytorch, TensorFlow, a....
- [GenAI Summit San Francisco 2024](https://www.eventbrite.com/e/genai-summit-san-francisco-2024-tickets-796934722207?aff=eemailordconf&utm_campaign=order_confirm&ref=eemailordconf&utm_medium=email&utm_source=eventbrite&utm_term=viewevent): This summit is an extraordinary convergence of the brightest minds in Generative AI, encapsulating the spirit of the future. #AI_ARE_ALL

  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1202897412780462140) (6 messages): 

- **Advanced RAG experimentation**: `@andysingal` highlighted his work on Advanced RAG using a dataset from HuggingFace and shared his notebook: [llm-course/RAG/Advanced_RAG (1).ipynb](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb). The GitHub preview includes an image, title, and description of the repository.

- **Introducing LLaVA-1.6 with Major Improvements**: `@meatfucker` announced the release of LLaVA-1.6, detailing significant upgrades in resolution, OCR, and reasoning, and even outperforming Gemini Pro on some benchmarks. A comprehensive blog post is available: [LLaVA-1.6 Release Notes](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/).

- **Creating Chatbots with HuggingFace**: `@yaaqob` invited users to try out a new chatbot that knows everything about innovation and challenging the status quo, created on the HuggingFace platform. Access the chatbot here: [Yaaqob's HuggingFace Chatbot](https://hf.co/chat/assistant/65bd51a7a16aaa191b5b50cf).

- **Deep Dive into Direct Preference Optimization**: `@imcoza1915` shared an article they wrote on Direct Preference Optimization, inviting feedback and discussion on the topic. Here's the article on LinkedIn for deeper engagement: [Direct Preference Optimization Article](https://www.linkedin.com/posts/imamashehzad_yesterday-i-was-deciphering-the-exciting-activity-7159323766773157888-rBNC).

**Links mentioned**:

- [LLaVA-1.6: Improved reasoning, OCR, and world knowledge](https://llava-vl.github.io/blog/2024-01-30-llava-1-6/): LLaVA team presents LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.
- [llm-course/RAG/Advanced_RAG (1).ipynb at main · andysingal/llm-course](https://github.com/andysingal/llm-course/blob/main/RAG/Advanced_RAG%20(1).ipynb): Contribute to andysingal/llm-course development by creating an account on GitHub.
- [Innovation Champion - HuggingChat](https://hf.co/chat/assistant/65bd51a7a16aaa191b5b50cf): Use the Innovation Champion assistant inside of HuggingChat

  

---


### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1202975030577987654) (6 messages): 

- **Tokenizer Patterns Unveiled**: User `deeeps.ig` created a Kaggle notebook to compare and visualize tokenization patterns across different language models from the Hugging Face library. They shared the [notebook link](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook) and hinted at a future web application inspired by OpenAI's approach.

- **Volatility Visualized for Traders**: `torres8552` announced an *Options Trading: Long & Short Straddle* app that allows traders to evaluate volatility and payoffs for specific options trading strategies. The app is available for testing and feedback at [Hugging Face's Spaces](https://huggingface.co/spaces/luisotorres/long_and_short_straddle).

- **Music Generation Takes a Leap Forward**: `.bigdookie` successfully demonstrated fine-tuning capabilities with a feature called *the infinite yt remix* and shared a [Twitter link](https://x.com/thepatch_kev/status/1753625904830726456?s=20) to the demonstration.

- **A Shoutout to Hugging Face**:`.bigdookie` expressed gratitude to Hugging Face for making hosting of fine-tuned models free and easy, emphasizing how much it eased their work process.

**Links mentioned**:

- [Tweet from thecollabagepatch (@thepatch_kev)](https://x.com/thepatch_kev/status/1753625904830726456?s=20): finally took the time to use this #musicgen extension&#39;s powers properly  few different fine-tunes used here  lil closer to a good demo of the infinite yt remix  @EzraSandzer and @amli_art good tal...
- [LLM Tokenizer Visualizer](https://www.kaggle.com/code/deeepsig/llm-tokenizer-visualizer/notebook): Explore and run machine learning code with Kaggle Notebooks | Using data from No attached data sources
- [Long &amp; Short Straddle - a Hugging Face Space by luisotorres](https://huggingface.co/spaces/luisotorres/long_and_short_straddle): no description found

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1202952665361944586) (51 messages🔥): 

- **Chad Announces AI and Law Talk**: `@chad_in_the_house` shares a link to a [Medium article](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34) that will form the basis of a presentation on AI in the legal field, discussing why it's hard to replace judges with AI. `@chad_in_the_house` confirms a Discord voice-chat for the presentation and intends to post a recording on Youtube afterward.
- **Engagement for Upcoming Law Presentation**: Users are inquiring about how to participate in the AI and law presentation, with `@chad_in_the_house` providing directions to the Discord voice-chat [link](https://discord.com/channels/879548962464493619/907325990236213288) and mentioning the possibility of future events being adjusted around the presenter's location.
- **Keen Interest in Future Reading Group Sessions**: `@datadev17` shows interest in regularly scheduled Friday discussions, which `@chad_in_the_house` confirms, citing that the next presentation by `@689634697097117750` will focus on *Mamba*, linked to a [When2meet](https://www.when2meet.com/?23471427-n4DUl) page for scheduling.
- **Discussion on Video Presentation Accessibility**: Users are coordinating the best way to access the presentation, with `@chad_in_the_house` promising to upload a trimmed recording link, and `@lunarflu` offering to post larger files due to Discord Nitro benefits.
- **Resource Sharing for the Mamba Paper**: `@chad_in_the_house` posts details about the next week's presentation on Mamba, providing an [arXiv link](https://arxiv.org/abs/2312.00752) and a [YouTube explainer](https://www.youtube.com/watch?v=9dSkvxS2EB0) by Yannic Kilcher. Additional Mamba resources are shared by `@janimo.` and `@swfsql`, linking to further YouTube explainers with detailed insights into the architecture.

**Links mentioned**:

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/879548962464493619/907325990236213288): Discord is the easiest way to communicate over voice, video, and text.  Chat, hang out, and stay close with your friends and communities.
- [Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY): Mamba is a new neural network architecture proposed by Albert Gu and Tri Dao.Timestamps:00:00 - Mamba - a replacement for Transformers?00:19 - The Long Range...
- [Mamba and S4 Explained: Architecture, Parallel Scan, Kernel Fusion, Recurrent, Convolution, Math](https://www.youtube.com/watch?v=8Q_tqwpTpVU): Explanation of the paper Mamba: Linear-Time Sequence Modeling with Selective State SpacesIn this video I will be explaining Mamba, a new sequence modeling ar...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752): Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time a...
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Paper Explained)](https://www.youtube.com/watch?v=9dSkvxS2EB0&ab_channel=YannicKilcher): #mamba #s4 #ssm OUTLINE:0:00 - Introduction0:45 - Transformers vs RNNs vs S46:10 - What are state space models?12:30 - Selective State Space Models17:55 - Th...
- [Literature Review on AI in Law](https://medium.com/@isamu-website/literature-review-on-ai-in-law-7fe80e352c34): This blog was inspired by Owl from the Laion Discord server. Thanks for the discussions! In this blog, my main goal is to go through why…
- [Eric's Presentation - When2meet](https://www.when2meet.com/?23471427-n4DUl): no description found

  

---


### HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1203203120637939764) (10 messages🔥): 

- **Diffusers 0.26.0 Takes Off**: User `@sayakpaul` announced the release of **Diffusers 0.26.0**, featuring two new video models, multi IP-adapter inference, and more, with a cheeky weekend release remark. Full release notes can be found [here](https://github.com/huggingface/diffusers/releases/tag/v0.26.0).

- **Troubleshooting in Action**: User `@meatfucker` reported issues with the example code from the new release, experiencing only noisy gifs as output when attempting to run the video pipeline examples on Windows.

- **A Loud Warning and the Quest for Answers**: During troubleshooting, `@meatfucker` shared a warning related to `flash attention` received in the console but was initially unsure of its impact on the output quality.

- **Detective Work Pays Off**: With some investigation, `@meatfucker` discovered the root cause of the issue, noting that the example code mistakenly set the number of inference steps to 1, which likely led to the poor output.

- **A Matter of Steps and Size**: `@meatfucker` pointed out that both the inference steps and decode size were set to ineffective values (1), which differ from the more appropriate defaults (50 inference steps) indicated in the official diffusers documentation, suggesting it might be a typo in the release notes.

**Links mentioned**:

[Release v0.26.0: New video pipelines, single-file checkpoint revamp, multi IP-Adapter inference with multiple images · huggingface/diffusers](https://github.com/huggingface/diffusers/releases/tag/v0.26.0): This new release comes with two new video pipelines, a more unified and consistent experience for single-file checkpoint loading, support for multiple IP-Adapters’ inference with multiple reference...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1203068363602137218) (2 messages): 

- **Transformer Troubles**: `@vikas.p` is encountering an issue where modified **donut** (with custom mbart decoder, gqa, and moe) performs well during training on **transformers 4.36.2** and **4.37.2**, but inference only works correctly on **4.36.2**. Inference on version **4.37.2** results in repeated output, with no clear explanation found in release notes.
  
- **DalensAI Seeking Volunteers for Livestock ML Model**: `@danielsamuel131`, founder of **DalensAI** and an AI and Computer Vision engineer, is looking for volunteers to help arrange a machine learning dataset for detecting sickness in livestock. The company's project requires images and labels for animals like chickens, sheep, goats, and cows, including those that are sick and healthy.
  

---


### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1202933685486231642) (4 messages): 

- **Seeking Tokenizer Support for GPT Models**: User `@janimo.` inquired about potential support for tiktoken/OpenAI models (GPT3/4) within the tokenizer library, mentioning the existence of a Rust crate named tiktoken-rs.
- **Tokenizer Conversion Script Shared**: `@cakiki` responded to `@janimo.` with a [conversion script](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee) provided by `<@234763281749770243>` that converts tiktoken tokenizers to the Hugging Face tokenizer format, albeit with some concerns about licensing.
- **Acknowledgment of Known Resources**: In response, `@janimo.` acknowledged awareness of the converted tokenizer files without further queries or context.
- **GPTQ Model Issues Raised**: User `.sgp` expressed confusion over being unable to use a tokenizer with GPTQ models, despite not having encountered issues previously.

**Links mentioned**:

[Convert tiktoken tokenizers to the Hugging Face tokenizers format](https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee): Convert tiktoken tokenizers to the Hugging Face tokenizers format - tiktoken-to-hf.ipynb

  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1202896147337715712) (28 messages🔥): 

- **Groq Chips Spark Interest**: User `@ethux` discusses the speed and price of the Groq's chips, suggesting they may be a competitive alternative to Nvidia H100 at a hefty price of 20k for their PCIe variant.
- **Groq Hardware Focus Revealed**: Both `@ethux` and `@mihaj` clarify that Groq is promoting its custom hardware rather than API services, designated as LPU's, with emphasis on its local optimization capabilities during runtime.
- **Seeking Clarification on Groq's Services**: `@lukasgutwinski` inquires about API services and notes budget constraints when considering Groq as a potential solution, with `@mihaj` adding that there is no provided hosting.
- **Groq's Performance Inquiry**: User `@i_am_dom` raises a question about the speed of Groq's chips despite limited video memory, with discussions suggesting that Groq's cards act more as accelerators and details found in the [GroqNode™ Server product brief](https://groq.com/wp-content/uploads/2022/10/GroqNode%E2%84%A2-Server-GN1-B8C-Product-Brief-v1.5.pdf).
- **Debating Model Quality on Hugging Face**: User `@dillfrescott` shares links to a Hugging Face model known as MoMo-72B, pointing out its high leaderboard score and contemplating whether it's "contaminated" while considering running the model on more powerful hardware for testing.

**Links mentioned**:

- [moreh/MoMo-72B-lora-1.8.7-DPO · Hugging Face](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO): no description found
- [moreh/MoMo-72B-lora-1.8.7-DPO · New Leader!](https://huggingface.co/moreh/MoMo-72B-lora-1.8.7-DPO/discussions/2): no description found

  

---


### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1202968654615019550) (7 messages): 

- **Friendly Banter in the Chat**: `@mercercl` jokingly called someone a "betweter," which `@ethux` playfully acknowledged by saying they could indeed read the message.
- **Expressions Can Hurt**: `@ethux` responded to `@mercercl`'s teasing with a "not nice :(" indicating the previous comment might have struck a nerve.
- **All in Good Fun**: `@mercercl` clarified their earlier comment by adding "kidding!" to smooth over the interaction.
- **Confusions about Free Model Access**: `@ashu2024` inquired about using an open-source model for free, expressing confusion about the API key process which seemed to point towards a subscription service after a usage limit.
- **Guidance Provided for Free Model Access**: `@mrdragonfox` clarified that the API is not associated with free model access but directed `@ashu2024` to find free options on Hugging Face.
  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1202923574659125271) (5 messages): 

- **Request for Help**: `@jay.sojitra` asked for assistance with an issue and provided a Discord channel link: [Mistral Discord Issue](https://discord.com/channels/1144547040454508606/1202913948253421578).
- **Mac Support Inquiry**: `@patochex` inquired about availability for Mac, which sparked `@ethux` to respond with a solution.
- **Solution Presented**: `@ethux` guided `@patochex` to use [LMStudio](https://lmstudio.ai) to download a Mistral model suitable for Mac users.
- **Acknowledgment of Solution**: `@patochex` expressed gratitude for the provided help with a brief "ok good thks !"

**Links mentioned**:

[👾 LM Studio - Discover and run local LLMs](https://lmstudio.ai): Find, download, and experiment with local LLMs

  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1202898992137113620) (15 messages🔥): 

- **Project Showcased but Not Open Source**: `@sublimatorniq` shared details about a project on [socontextual.com](https://socontextual.com) which is not open source at the moment.
- **Keen Anticipation for Project Release**: Users `@atomicspies` and `@mrdragonfox` expressed admiration and anticipation for the release of a showcased project, praising its utility in research over traditional methods.
- **The Year-Long Journey to Perfection**: `@sublimatorniq` anticipates that it will likely take a year to advance the project to where they want it to be.
- **"LLaVA-1.6" Performance and YouTube Demo**: `@pradeep1148` linked to a YouTube video titled "Trying LLaVA-1.6 on Colab", showcasing the capabilities of the LLaVA-1.6 version with improved reasoning and world knowledge.
- **Discworld AI Fan Fiction Experiment**: `@caitlyntje` created a fan fiction story titled "Sapient Contraptions" inspired by Sir Terry Pratchett, using open source AI LLM software like Mistral and Huggingface, and shared the story on [Pastebin](https://pastebin.com/dNRbi7mY). `@amagicalbook` expressed interest in learning the process for their own story writing endeavors.

**Links mentioned**:

- [Trying LLaVA-1.6 on  Colab](https://www.youtube.com/watch?v=SEavari8xaU): LLaVA-1.6, with improved reasoning, OCR, and world knowledge. LLaVA-1.6 even exceeds Gemini Pro on several benchmarks.https://llava-vl.github.io/blog/2024-01...
- [My Friend](https://youtu.be/U1Ut-rwxKBA?si=I-6IzIpVcxVSzmw2): Provided to YouTube by Legacy RecordingsMy Friend · Jimi HendrixThe Cry of Love℗ 2009 Experience Hendrix L.L.C., under exclusive license to Sony Music Entert...
- [SAPIENT CONTRAPTIONSA Discworld Fan Fiction storyInspired by Sir Ter - Pastebin.com](https://pastebin.com/dNRbi7mY): Pastebin.com is the number one paste tool since 2002. Pastebin is a website where you can store text online for a set period of time.

  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1202903397410283590) (43 messages🔥): 

- **New User Queries About Perplexity Base Model**: New participant `christolito` asked about the "base perplexity" model. The user `mares1317` welcomed and provided helpful links for further information, suggesting to check out specific Discord channels for detailed explanations.
- **Android App Lacks Document Attachment**: `@nqiwbh07r44p` inquired about attaching documents on the Perplexity Android app, which `@icelavaman` clarified isn't available yet, indicating the web version might be more feature-rich.
- **Inquiry About Copilot's Efficacy and Model**: `@joshuaa71` questioned Copilot's functionality and model identity. `@icelavaman` responded with links to blog posts explaining Copilot's use of GPT-4 and Claude 2 in focused modes and its search capabilities without internet access.
- **Clarifications About Perplexity AI's Features**:
  - `@ruspazyyy` asked if the free version of Perplexity has any limits; `@perplexityai` responded that there are limits similar to what's typically experienced with ChatGPT.
  - `@lukas8a` sought a method to transcribe text from images within Perplexity; `@icelavaman` provided a link to the relevant feature search. 
- **Technical Issues and Feature Requests**: Users are sharing concerns and suggestions:
  - `@guocity` asked if Perplexity can automatically summarize lengthy articles, a feature Edge Copilot has.
  - `@zwgnr` reported potential UX issues on the latest iOS update concerning the copy button and code block background color in responses.
  - `@matthewtaksa`, a Pro user, reported experiencing issues with response generation delay and message duplication.

**Links mentioned**:

- [Introduction | 🦜️🔗 Langchain](https://python.langchain.com/docs/get_started/introduction): LangChain is a framework for developing applications powered by language models. It enables applications that:
- [What is Perplexity Copilot?](https://blog.perplexity.ai/faq/what-is-copilot): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [Working On GIF - Working On It - Discover &amp; Share GIFs](https://tenor.com/view/working-on-it-under-construction-gif-23162421): Click to view the GIF
- [GitHub - BuilderIO/gpt-crawler: Crawl a site to generate knowledge files to create your own custom GPT from a URL](https://github.com/BuilderIO/gpt-crawler?tab=readme-ov-file#example): Crawl a site to generate knowledge files to create your own custom GPT from a URL - GitHub - BuilderIO/gpt-crawler: Crawl a site to generate knowledge files to create your own custom GPT from a URL
- [Perplexity Blog](https://blog.perplexity.ai/faq/what-is-search-focu): Explore Perplexity&#39;s blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.
- [What is Search Focus?](https://blog.perplexity.ai/faq/what-is-search-focus): Explore Perplexity's blog for articles, announcements, product updates, and tips to optimize your experience. Stay informed and make the most of Perplexity.

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1202898413763563591) (6 messages): 

- **Scam Savvy Skills**: User `@byerk_enjoyer_sociology_enjoyer` expressed concerns about identifying legitimate online jobs and the difficulty in discerning scams within that space.
- **Document API Success**: `@fkx0647` shared an experience of successfully uploading and interacting with a document via an API, mentioning an affiliate program.
- **Javascript Journey**: `@stocktown` briefly mentioned that they have been learning *some JS programming*.
- **Perplexity Preference Over Google and ChatGPT**: `@kronokaizen` shared [a YouTube video](https://www.youtube.com/watch?v=aphHCBSTx7Q) titled "I use Perplexity MORE than Google and ChatGPT," praising Perplexity for its usefulness in content creation.
- **Echoing Enthusiasm for Perplexity**: `@andbamjam` echoed `@kronokaizen`'s sentiment, commending Perplexity for being an exceptional learning aid, akin to having the *smartest people* answer every question.

**Links mentioned**:

[I use Perplexity MORE than Google and ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q): Main Takaways From this Video: &quot;I use Perplexity more than ChatGPT, BARD, and Microsoft Copilots for five main reasons, including its use in content creation...

  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203021756772913164) (3 messages): 

- **Queries about API Variants**: `@whodis008` asked if others were using the online variants. `@defektivex` confirmed that they were indeed using the online variants.
- **Request for llava-v1.6-34b API Support**: `@bergutman` suggested that Perplexity should consider adding API support for **llava-v1.6-34b** given the lack of multi-modal API options and the high cost of using 1.6 on replicate compared to GPT4-V.
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1202912046165786654) (46 messages🔥): 

- **AI Fine-Tuning Community Effort with SuperServer**: `@dctanner` announced the completion of an **8x3090 SuperServer** for the community to run novel axolotl fine-tunes, inviting DMs for collaboration. A link to `dctanner`'s announcement [The AI SuperServer is live!](https://x.com/dctanner/status/1753013407643562401?s=20) was shared for more details on the server's capabilities.
- **Sample Packing and BYOD**: In a discussion about finetuning tools, `@nanobitz` mentioned advantages of **axolotl** over **AutoTrain**, highlighting *"sample packing and simple yaml sharing + byod"*. They also referred to AutoTrain's *automatic model selection* as an interesting feature.
- **Exploring FFT on Different Models**: `@le_mess` inquired whether a Fast Fourier Transform (FFT) of **Mistral** would fit on `dctanner`'s SuperServer, to which `dctanner` confirmed a full finetune (FT) of Mistral 7b was the first attempt. They also considered testing with Solar 10.7b upon `le_mess`'s request.
- **Technical Discussion on Model Storage and Training**: In a deeper technical discussion, `@nafnlaus00` shared an interest in building a SuperServer and speculated about the feasibility of using multiple GPUs for full finetuning of models like Mixtral. `@yamashi` and `@nafnlaus00` exchanged thoughts on the complexities of storing gradients and communication bandwidth between GPUs.
- **Performance Gains with vLLM Update**: `@dreamgen` reported that the version 0.3.0 of **vLLM** was significantly faster for their specific workload compared to version 0.2.7, suggesting noticeable performance enhancements in the latest update.

**Links mentioned**:

[Tweet from Damien C. Tanner (@dctanner)](https://x.com/dctanner/status/1753013407643562401?s=20): The AI SuperServer is live!

  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1203011046529310721) (3 messages): 

- **Unexpected Inference Termination in Mixtral Instruct**: `@nafnlaus00` encountered a problem where **Mixtral Instruct**, specifically **GGUF Q3_K_M**, would **terminate responses prematurely** about 5% of the time during a summarization task, cutting off sentences unexpectedly.
- **Inquiry about MoE Inference Methods**: `@nanobitz` asked `@nafnlaus00` what method they were using for MoE (Mixture of Experts) inference, to which `@nafnlaus00` responded that they use **llama.cpp**.
  

---


### OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1203207596379602964) (1 messages): 

- **New Math Dataset Alert**: User `@xzuyn` shared the **Math-Multiturn-100K-ShareGPT** dataset available on Hugging Face, which involves conversations aimed to solve math questions, with responses from a system identified as GPT. The dataset contains up to 64 response pairs per conversation and is designed to be expanded with more complex equations in the future. [Check out the dataset here](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT).

**Links mentioned**:

[PJMixers/Math-Multiturn-100K-ShareGPT · Datasets at Hugging Face](https://huggingface.co/datasets/PJMixers/Math-Multiturn-100K-ShareGPT): no description found

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1203037412570304542) (1 messages): 

- **Streamline RAG System Setup with RAGArch**: User `@HarshadSurya1c` has introduced **RAGArch**, which features a [Streamlit](https://streamlit.io) UI allowing users to easily pick components of a **RAG (Retrieval-Augmented Generation)** system including LLM, embedding model, and vector store. One click creates a fully operational RAG pipeline, combining convenience with customization. [Tweet link with more info](https://twitter.com/llama_index/status/1753478149743284395).
  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1202950550656913468) (35 messages🔥): 

- **Guide to Using Hugging Face LLMs with LlamaIndex**: `@kapa.ai` provided a comprehensive step-by-step on how to use a pre-trained language model (LLM) from Hugging Face with LlamaIndex, mentioning the installation of required packages, setup of tokens, and execution of local or remote model runs. Further guidance can be found in the [detailed example notebook](https://github.com/jerryjliu/llama_index/blob/main/docs/examples/llm/huggingface.ipynb).
- **Colab Notebook for HuggingFace StableLM**: `@whitefang_jr` shared a [Colab notebook link](https://colab.research.google.com/github/jerryjliu/llama_index/blob/main/docs/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.ipynb) that provides hands-on guidance for using HuggingFace StableLM with LlamaIndex, supportive for users looking to install and implement LlamaIndex on Colab.
- **Connecting Predictive Models with LlamaIndex**: Users `@matthews_38512` and `@kapa.ai` discussed the integration of LlamaIndex with various predictive models' APIs such as OpenAI, Hugging Face, and others, with `@kapa.ai` noting specific guides for different integrations and emphasizing LlamaIndex's capability to run local models like Llama 2.
- **Understanding the Role of LlamaIndex vs. LangChain**: `@cheesyfishes` responded to `@affable_honey_badger` clarifying that LlamaIndex can function independently or alongside LangChain, particularly highlighting LlamaIndex's focus on RAG/context augmentation.
- **Ollama - An Optimized Model Runner**: In a conversation with `@affable_honey_badger`, `@cheesyfishes` described Ollama as an optimized local runner for various models without the need for a GPU and acting as a wrapper for llama.cpp, suggesting Ollama for local testing and other solutions for production deployment.

**Links mentioned**:

- [HuggingFace LLM - StableLM - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm.html): no description found
- [library](https://ollama.ai/library): Get up and running with large language models, locally.
- [Module Guides - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules.html#custom-agents): no description found
- [Module Guides - LlamaIndex 🦙 0.9.43](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/modules.html#id1): no description found

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1202982195568975952) (6 messages): 

- **Seeking the Secret of Perplexity AI's Citations**: `@tyronemichael` expressed curiosity about **Perplexity AI's** citation generation mentioned in the [Perplexity AI Documentation](https://docs.perplexity.ai/discuss/65af6285e69072005b83eb05) and shared their own approach which uses **SerpAPI** and **LlamaIndex**. However, they noted that Perplexity's output is more advanced compared to their basic method.
- **Rapid Citation Retrieval Remains a Mystery**: In a follow-up message, `@tyronemichael` remarked on the impressive speed of citation retrieval by **Perplexity AI**, despite the challenges posed by websites blocking bot access.
- **API Limitations Frustrate Curious Developer**: `@tyronemichael` signed up for **SerpAPI** hoping to replicate Perplexity's citation feature but discovered that citations are not yet part of their API's return data.
- **Cryptic Responses Leave Developer Perplexed**: After attempting to inquire directly, `@tyronemichael` received a cryptic answer from **Perplexity AI** about their citation methodology, leaving them without clear insights.
- **Google Paper Highlights Perplexity AI's Strengths**: `@tyronemichael` shared a [link to a tweet](https://x.com/cto_junior/status/1710638210009706800?s=20) by `@cto_junior` discussing a paper from Google which evaluated and praised **Perplexity AI** for its performance in factual Q&A and debunking.

**Links mentioned**:

- [Tweet from TDM (e/λ) (@cto_junior)](https://x.com/cto_junior/status/1710638210009706800?s=20): Interesting, a paper by google that evaluated @perplexity_ai on their new eval.  It performs really well for factual Q&A and debunking, better than vanilla google search.  https://arxiv.org/abs/2310.0...
- [no title found](https://serpapi.com)): no description found

  

---



### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1203033156437934100) (32 messages🔥): 

- **Unveiling the GPU Powerhouse**: User `@zippika` reveals working with a **Nvidia 4090 GPU** and discusses an optimized CUDA code for converting RGB to grayscale using `uchar3`. The code leverages integer arithmetic and bit-shifting to avoid floating-point operations.
- **Divide and Conquer with Bitwise Shifts**: Upon inquiry from `@jeremyhoward`, `@zippika` explains the ` >> 8` operation in the code is a bitwise shift that effectively divides by 256, a more efficient alternative to floating-point division.
- **Spotting Efficiency in CUDA Optimizations**: User `@apaz` hypothesizes that compilers with optimizers are likely to replace square constant divisions with shifts automatically, although they hadn't tested this behavior.
- **Welcoming NVIDIA Expertise**: `@jeremyhoward` extends a welcome to `@vim410`, who has joined the CUDA MODE community; `@vim410` is a researcher at NVIDIA with connections to influential figures in the field.
- **Memory Management Missteps Corrected**: `@_davidgonmar` seeks help with a bug related to memory management in a C++ CUDA array class. Suggestions from other users like `@lancerts` and `@vim410` lead to resolving the issue by using proper C++ memory management techniques.
  

---


### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1202906563015082025) (9 messages🔥): 

- **Shared Memory in Numba Speed Explained**: `@mishakeyvalue` inquired about the speed difference when using **shared memory** compared to global reads/writes in GPU computing with Numba. [Siboehm's article](https://siboehm.com/articles/22/CUDA-MMM) was shared by `@stefangliga`, featuring optimizations in CUDA matrix multiplication and performance characteristics like memory access coalescing and shared memory caching, along with links to relevant GitHub repos for further exploration.

- **Kernel Code Snippet Error Handling**: `@ashpun` faced a `RuntimeError` related to a failed `inline_ext` build in CUDA kernel coding. Following a discussion asking for the full error details, `@marksaroufim` resolved the issue, identifying a missing brace (`}`) in the kernel code.

- **Troubleshooting ImportError in CUDA Coding**: After resolving one issue, `@ashpun` encountered an ImportError indicating a missing `GLIBCXX_3.4.32` version, despite being present on the system. `@marksaroufim` suggested running `conda update --all` and potentially setting the `LD_LIBRARY_PATH` correctly to address the library path issue.

**Links mentioned**:

[How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM): In this post, I’ll iteratively optimize an implementation of matrix multiplication written in CUDA.My goal is not to build a cuBLAS replacement, but to deepl...

  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1202928537074536488) (20 messages🔥): 

- **Struggles with Elasticsearch**: `@shamspias` inquired about **async support options for Elasticsearch vector** but didn't receive a direct answer in the discussed messages.
- **Customizing `langchain` Tables**: `@emile_ibr` researched `langchain + pgvector` and asked if it's possible to add columns, like `user_id`, to tables created by **langchain** such as `lagchain_pg_collection`.
- **LangChain Documentation Frustrations**: Multiple users including `@anthonyj.2048`, `@benjaminbascary`, and others expressed frustration with **LangChain documentation**, mentioning it's driving them insane or ironically noting how LangChain can't explain its own usage.
- **Mixed Opinions on LangChain**: `@engineered.mind` has stopped development with **LangChain due to its rapid changes and lack of modularity**, but `@.jkyle` and `@akshay_1` discussed some of its timesaving features, while still recognizing the limitations that make it not suited for all projects.
- **Multiple Agents Routing Inquiry**: `@crtapps` sought advice on the best approach to route user queries among multiple agents with specific functions, questioning the efficiency of continuously updating `router_to_agent` with each new addition.
  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 

rebelsandrobots_97106: Thanks!
  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1202916924875608106) (5 messages): 

- **AutoCrew by @yannie**: @yannie shared a tool called [AutoCrew](https://github.com/yanniedog/autocrew) which can automatically create a crew and tasks for CrewAI. An image preview of the repository and a brief description of its functionality were included in the message.

- **@esxr_ Presents Chat UI Tutorial**: @esxr_ created a tutorial demonstrating how to adapt an open-source framework to deliver a chat-based user experience akin to ChatGPT in just 15 minutes. The informative [YouTube video](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5) and the accompanying [GitHub repository](https://github.com/esxr/repurposed-ollama-webui) were shared with the community.

- **Tiny Desk AI Chat App**: BrendonJacobs promoted a no frills, free chat app with a website called [Tiny Desk AI](https://tinydesk.ai). They shared links to the tools, documentation, about page, plans, and the signup page for the platform.

- **LangChain CSV Agents Tutorial**: ryannolan shared a [YouTube tutorial](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2) about LangChain CSV Agents, focusing on how to converse with a CSV file using the OpenAI API. The video guide is targeted at beginners and explains the process, despite acknowledging some bugs.

- **Lutra AI Introduces Workspace Integration**: polarbear007. introduced [Lutra.ai](https://lutra.ai), a platform that integrates AI with Google Workspace for data processing and Internet research. It allows actions like extracting information from PDFs in Google Drive and converting it into a Google spreadsheet.

**Links mentioned**:

- [Lutra AI](https://lutra.ai): no description found
- [Creating a ChatGPT like UI for all your AI projects](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5): GitHub repo link:https://github.com/esxr/repurposed-ollama-webui
- [Chat with a CSV - LangChain CSV Agents Tutorial For Beginners (OpenAI API)](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2): In this Langchain video, we take a look at how you can use CSV agents and the OpenAI API to talk directly to a CSV file. While still a bit buggy, this is a p...
- [TinyDesk AI - Powerful tools to help you study smarter](https://tinydesk.ai): tinydesk.ai - Your AI and Stable Diffusion source for cutting-edge insights and advancements
- [GitHub - yanniedog/autocrew: Automatically create a crew and tasks for CrewAI](https://github.com/yanniedog/autocrew): Automatically create a crew and tasks for CrewAI. Contribute to yanniedog/autocrew development by creating an account on GitHub.

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1202887608276099143) (3 messages): 

- **AI DSP Revealed by Stanford**: `@lhc1921` shared a [YouTube video](https://www.youtube.com/watch?v=dTzL8OF_3i0) presenting **Stanford University's Demonstrate - Search - Predict Model (DSP)**, showcasing a method that bootstraps high-level programs with pipeline-aware demonstrations.
- **Generative AI Chat Experience Made Easy**: User `@esxr_` posted a [tutorial](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5) explaining how to adapt an open-source framework to create a **ChatGPT-like user interface** for AI projects in under 15 minutes.
- **Chatting with CSV through LangChain**: `@ryannolan` introduced a [tutorial](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2) on using LangChain CSV Agents to enable conversation directly with a CSV file via **OpenAI API**, acknowledging that while innovative, it is still a bit buggy.

**Links mentioned**:

- [AI DSP: LLM Pipeline to Retriever Model (Stanford)](https://www.youtube.com/watch?v=dTzL8OF_3i0): Demonstrate - Search - Predict Model (DSP) by Stanford Univ. DSP can express high-level programs that bootstrap pipeline-aware demonstrations, search for rel...
- [Chat with a CSV - LangChain CSV Agents Tutorial For Beginners (OpenAI API)](https://youtu.be/VVdzQs-FeHE?si=phIzT4F57gmtSzI2): In this Langchain video, we take a look at how you can use CSV agents and the OpenAI API to talk directly to a CSV file. While still a bit buggy, this is a p...
- [Creating a ChatGPT like UI for all your AI projects](https://youtu.be/sZ1aJ0zfgmY?si=koLhtl_FO6-y3SC5): GitHub repo link:https://github.com/esxr/repurposed-ollama-webui

  

---



### LLM Perf Enthusiasts AI ▷ #[embeddings](https://discord.com/channels/1168579740391710851/1168744166138859580/) (1 messages): 

natureplayer: https://huggingface.co/spaces/mteb/leaderboard
  

---


### LLM Perf Enthusiasts AI ▷ #[feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549/1203225679911718912) (1 messages): 

- **Browse Channels Feature Request**: `@joshcho_` raised a suggestion for the addition of a **browse channels** feature, addressing the challenge of not being able to view selected channels of interest – a feature commonly available with community-enabled setups. They emphasized their tendency to overlook many channels and the desire to streamline their focus.
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1203034070628565037) (8 messages🔥): 

- **GPT-3.5 Passes the Instruction Test**: `@justahvee` found the new **GPT-3.5** to be better for "instruction heavy tasks," indicating an improvement unrelated to the context window size, and solely attributed to the model's enhanced capability to follow given instructions.
- **A Trade-Off for Compliance**: `@justahvee` mentioned that the priority is on instruction-following rather than reasoning abilities, accepting any potential degradation in reasoning if it means the model adheres to instructions more accurately.
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1202999433592176650) (7 messages): 

- **Deep Thoughts Enhance AI Performance**: User `@res6969` observed that a more extensive explanation in prompts tends to allocate more `computation to the tokens` for determining the final output, resulting in better AI performance.
- **Trade-off Alert: Speed vs. Smarts**: `@res6969` acknowledged that while detailed prompts improve intelligence, there is a significant `latency tradeoff`.
- **Smart Responses Draw User Praise**: Asynchronous strategies involving comprehensive Chain of Thought (CoT) prompts hidden from users deliver impressively `smart` AI responses, shared `@res6969`.
- **Practical Applications with GPT-4-Turbo**: User `@sourya4` reported improved function calling accuracy by employing extended thought explanations with `gpt-4-turbo`, while actively exploring ways to balance the `latency tradeoffs`.
- **Iterative Thought Processing**: `@byronhsu` inquired about saving Chain of Thought outputs and reusing them for a secondary processing step, to which `@res6969` replied affirmatively, though no formal evaluations have been done yet.
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1203180973215449088) (6 messages): 

- **Checking In**: User `@daydream.nation` greeted the chat with a simple "hey everyone".
- **Realization of Going Public**: `@daydream.nation` acknowledged that the team went public with their project.
- **Regret on Missing Participation**: `@daydream.nation` expressed regret for not being able to participate in the project so far.
- **Speculating on Large-Scale Interaction Testing**: `@daydream.nation` speculated that the release might be aimed at testing human interaction on a larger scale, much like **Google's Bard**.
- **Alignment Context Considered**: `@daydream.nation` clarified that their comments were in the context of **alignment**.
  

---


### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (1 messages): 

cryptossssun: 🤔
  

---


### Alignment Lab AI ▷ #[looking-for-work](https://discord.com/channels/1087862276448595968/1142242683339944027/1203193305182109696) (1 messages): 

- **Philosophical Pioneer Ready to Engage**: @daydream.nation, skilled in **Python, Excel Data Modeling, and SQL**, and with **experience in Philosophy**, has authored a research paper on **Bard** and expresses a deep commitment to exploring the future of our species and addressing the hard problem of consciousness with AI. Eager to blend their diverse background with AI logic and argumentation, they show readiness for a collaborative discussion on how their unique insights can contribute to the field.
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1202968562487267328) (4 messages): 

- **Infinite Craft Built on Llama2**: `@chrisamico` brought to attention a game called [Infinite Craft](https://neal.fun/infinite-craft/) which is built on **llama2**, showcasing elements like water, fire, wind, and earth that players can drag to craft.
- **Game Endorsement**: `@chrisamico` also recommended trying out more games from the creator of Infinite Craft, praising them as very clever, fun, and sometimes thought-provoking.
- **Confirmation of Infinite Craft's Allure**: `@dbreunig` acknowledged the game's appeal with a concise endorsement, suggesting it's a great example.
- **Addictive Nature of Infinite Craft**: `@bdexter` expressed that the game is indeed addictive, implying personal experience with the game's engaging content.

**Links mentioned**:

[Infinite Craft](https://neal.fun/infinite-craft/): A game about crafting

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1202919998574694430) (4 messages): 

- **German Embedding Models Excel**: `@damian_89_` shared a [tweet](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19) highlighting that two German embedding models, **jina-embeddings-v2-base-de by @JinaAI_** and **bge-m3 by @BAAIBeijing**, outperform others in enterprise data tests, with BGE being the superior.
- **Test Embeddings with Metrics**: `@devnull0` suggests testing these embedding models with a suitable metric to assess performance, without providing a specific metric or method.
- **RAG Evaluation Guide with Notebook**: `@devnull0` shared a [GitHub notebook](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb) to evaluate Retrieval-Augmented Generation (RAG) systems, alongside a visual preview of the repository.
- **Deep Dive into RAG Evaluation on srk.ai**: The accompanying [blog post](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex) provides a comprehensive guide on evaluating a RAG system's encoder and reranker components, using LlamaIndex and a custom testing dataset.

**Links mentioned**:

- [Tweet from Damian Strobel (@damian_89_)](https://fxtwitter.com/damian_89_/status/1753052084511944891?t=GJgqBYsr2brcjyw64xO0pQ&s=19): Whoever does RAG on german data, lately two embedding models were released: jina-embeddings-v2-base-de by @JinaAI_ and bge-m3 by @BAAIBeijing - both outperform in my tests on real enterprise data all ...
- [RAG - Encoder and Reranker evaluation](https://srk.ai/blog/004-ai-llm-retrieval-eval-llamaindex): Evaluate the performance of encoder and reranker in the RAG pipeline using custom datasets
- [srk_ai_blog/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb at master · SudalaiRajkumar/srk_ai_blog](https://github.com/SudalaiRajkumar/srk_ai_blog/blob/master/004-RAG_Retrieval_Eval/004-RAG_Retrieval_Eval_LlamaIndex.ipynb): Codes used for https://srk.ai/ blog. Contribute to SudalaiRajkumar/srk_ai_blog development by creating an account on GitHub.

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=SEavari8xaU
  

---


### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 messages): 

.mrfoo: LLaVA 1.6 dropped : https://llava-vl.github.io/blog/2024-01-30-llava-1-6/
  